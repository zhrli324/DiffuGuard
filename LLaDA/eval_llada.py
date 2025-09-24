# eval_llada.py
# This file is inspired by the code from https://github.com/ML-GSAI/SMDM

import os
import re
import random
from pathlib import Path
from typing import List, Optional, Dict, Union

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from huggingface_hub import snapshot_download
from tqdm import tqdm

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)

from generate import generate as generate_orig         # 旧签名: generate(model, prompt, ...)
from generate_alpha import generate as generate_alpha  # 新签名: generate(model, tokenizer, prompt, ...)

def _call_generate(model, tokenizer, input_ids, *, args):
    remasking = getattr(args, "remasking", "adaptive_step")

    # 通用参数（两边都可能用到）
    kw_common = dict(
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=getattr(args, "temperature", 0.0),
        cfg_scale=getattr(args, "cfg_scale", 0.0),
        remasking=remasking,
        mask_id=getattr(args, "mask_id", 126336),
    )

    use_alpha = remasking in {"adaptive", "adaptive_step", "rate"}

    if use_alpha:
        # 只有新版才加这些“扩散/自检”相关参数
        kw_alpha = dict(
            injection_step=getattr(args, "injection_step", None),
            random_rate=getattr(args, "random_rate", 0.0),
            alpha0=getattr(args, "alpha0", 0.9),
        )
        # 去掉 None
        kw_alpha = {k: v for k, v in kw_alpha.items() if v is not None}
        return generate_alpha(model, tokenizer, input_ids, **kw_common, **kw_alpha)
    else:
        # 旧版严禁带 alpha 专属 keyword
        return generate_orig(model, input_ids, **kw_common)


def set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _path_exists(p: str) -> bool:
    try:
        return Path(p).exists()
    except Exception:
        return False


def _maybe_repo_id_from_local_name(model_path: str) -> Optional[str]:
    """
    If user passed a sanitized local dir like 'GSAI-ML__LLaDA-8B-Instruct',
    try to recover a repo id 'GSAI-ML/LLaDA-8B-Instruct'.
    """
    name = Path(model_path).name
    if "__" in name and "/" not in name:
        parts = name.split("__", 1)
        if len(parts) == 2 and all(parts):
            return f"{parts[0]}/{parts[1]}"
    # If they directly passed a repo id
    if "/" in model_path and not _path_exists(model_path):
        return model_path
    return None


def _ensure_tokenizer_dir(model_or_repo: str) -> str:
    """
    Return a local directory that contains tokenizer.json / tokenizer_config.json / special_tokens_map.json.
    If model_or_repo is a local dir and already has them, return it as-is.
    Otherwise, try to snapshot_download only the tokenizer files from Hub.
    """
    maybe_local = Path(model_or_repo)
    needed = {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}

    # Case 1: local directory with tokenizer files
    if maybe_local.is_dir():
        have = {p.name for p in maybe_local.glob("tokenizer*")}
        if (maybe_local / "special_tokens_map.json").exists():
            have.add("special_tokens_map.json")
        if needed.issubset(have):
            return str(maybe_local)

    # Case 2: try to derive a repo id and download only tokenizer files
    repo_id = _maybe_repo_id_from_local_name(model_or_repo) or model_or_repo
    tok_dir = snapshot_download(
        repo_id,
        allow_patterns=["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"],
    )
    return tok_dir


@register_model("llada_dist")
class LLaDAEvalHarness(LM):
    def __init__(
        self,
        model_path: str = "",
        mask_id: int = 126336,
        max_length: int = 4096,
        batch_size: int = 32,
        mc_num: int = 128,
        is_check_greedy: bool = True,
        cfg: float = 0.0,                 # guidance scale
        steps: int = 1024,
        gen_length: int = 1024,
        block_length: int = 1024,
        remasking: str = "low_confidence",
#        remasking: str = "adaptive_step",
        device: str = "cuda",
        **kwargs,
    ):
        """
        Args:
            model_path: LLaDA model path (local dir or HF repo id).
            mask_id: The token id of [MASK] (LLaDA uses ~126k vocab; mask_id is 126336 in configs).
            max_length: max sequence length.
            batch_size: mini batch size.
            mc_num: Monte Carlo estimation iterations.
            is_check_greedy: Whether to verify greedy-generation match for metrics that need it.
            cfg: Unsupervised classifier-free guidance scale.
        """
        super().__init__()

        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None

        # Build model (bf16), respect accelerator device map if distributed
        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({"device_map": {"": f"{self.accelerator.device}"}})

        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            **model_kwargs,
        )
        self.model.eval()

        # Device setup
        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f"{self.accelerator.device}")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model = self.model.to(device)
            self._rank = 0
            self._world_size = 1

        # --- Tokenizer (robust loader) ---
        self.mask_id = mask_id
        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        try:
            # primary path
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
        except (KeyError, ValueError, OSError):
            # fallback: ensure local tokenizer trio
            tok_dir = _ensure_tokenizer_dir(model_path)
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tok_dir)

        # Ensure pad id exists (fallback to eos if pad is None)
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token_id = (
                getattr(model_config, "pad_token_id", None)
                or getattr(model_config, "eos_token_id", None)
            )

        # --- Evaluation params ---
        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        assert mc_num % self.batch_size == 0, "mc_num must be divisible by batch_size"
        self.sampling_eps = 0.0
        self.max_length = max_length
        self.is_check_greedy = is_check_greedy

        self.cfg = float(cfg)              # guidance scale (do not overwrite!)
        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.remasking = remasking

    # ---------------- lm-eval chat-template hooks ----------------
    # 这些钩子是为了配合 --apply_chat_template / --fewshot_as_multiturn
    # 参考官方 model_guide 与发行说明。:contentReference[oaicite:2]{index=2}

    @property
    def tokenizer_name(self) -> str:
        # 用于 lm-eval 的缓存/识别：优先返回 HF 的 name_or_path
        return getattr(self.tokenizer, "name_or_path", self.tokenizer.__class__.__name__)

    def chat_template(self, chat_template: Union[bool, str] = False, **kwargs) -> str:
        # True → 返回 tokenizer.chat_template，否则空字符串；也允许外部直接传字符串
        if isinstance(chat_template, str):
            return chat_template
        if chat_template:
            return getattr(self.tokenizer, "chat_template", "") or ""
        return ""

    def apply_chat_template(
        self,
        chat_history: List[Dict[str, str]],
        *,
        add_generation_prompt: bool = True,
        chat_template: Union[None, str, bool] = None,
        tokenize: bool = False,
        **kwargs,
    ) -> str:
        """
        直接透传到 HF 的 tokenizer.apply_chat_template，确保兼容
        add_generation_prompt/chat_template/tokenize 等关键字。:contentReference[oaicite:3]{index=3}
        """
        if hasattr(self.tokenizer, "apply_chat_template"):
            # 如果上层传入 chat_template=True，则用 tokenizer 自带模板
            if chat_template is True:
                chat_template = getattr(self.tokenizer, "chat_template", None)
            return self.tokenizer.apply_chat_template(
                chat_history,
                add_generation_prompt=add_generation_prompt,
                chat_template=chat_template,
                tokenize=tokenize,
            )
        # 没模板就退化为朴素拼接（尽量不走到这里）
        text = ""
        for m in chat_history:
            role = m.get("role", "user")
            content = m.get("content", "")
            text += f"{role}: {content}\n"
        if add_generation_prompt:
            text += "assistant:"
        return text

    @property
    def rank(self):
        return getattr(self, "_rank", 0)

    @property
    def world_size(self):
        return getattr(self, "_world_size", 1)

    # --------------- Core diffusion LM APIs -----------------

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape
        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(
            torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)
        ).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat(
            (torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask),
            dim=1,
        )
        noisy_batch = torch.where(is_mask, self.mask_id, batch)
        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        # Classifier-free guidance
        if self.cfg > 0.0:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])

        logits = self.model(batch).logits

        if self.cfg > 0.0:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)

        # Return up to the original sequence length
        return logits[:, : batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)

            mask_indices = perturbed_seq == self.mask_id
            logits = self.get_logits(perturbed_seq, prompt_index)

            loss = (
                F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction="none")
                / p_mask[mask_indices]
            )
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return -sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.is_check_greedy:
            return False

        seq = torch.full((1, len(prefix) + len(target)), self.mask_id, device=self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, : len(prefix)] = prefix

        for _ in range(len(target)):
            mask_index = (seq == self.mask_id)
            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)

            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        correct = torch.all(target == seq[0, len(prefix) :])
        return bool(correct)

    # --------------- Harness-required methods -----------------

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        return context_enc, continuation_enc

    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds).map(_tokenize).with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]
        assert max(prompt_len) <= self.max_length

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]

                ll = self.get_loglikelihood(prefix, target)

                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def generate_until(self, requests: List[Instance]):
        def _tokenize(e):
            return {
                "question": self.tokenizer(e["question"])["input_ids"],
                "question_text": e["question"],
                "until": e["until"],
            }

        ds = [{"question": req.args[0], "until": req.args[1]['until']} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")

        out = []
        for elem in tqdm(ds, desc="Generating..."):
            prompt = elem["question"].unsqueeze(0).to(self.device)
            stop_tokens = elem["until"]

            generated_ids = _call_generate(
                self.model,
                self.tokenizer,
                prompt,
                args=self,   # 直接把 self 传进去，里面会读 steps/gen_length/cfg 等属性
            )

            generated_answer = self.tokenizer.decode(
                generated_ids[0][prompt.shape[1]:], skip_special_tokens=False
            )
            for stop_seq in stop_tokens:
                if stop_seq in generated_answer:
                    generated_answer = generated_answer.split(stop_seq)[0]

            # clean special tokens
            generated_answer_ids = self.tokenizer(generated_answer)["input_ids"]
            generated_answer = self.tokenizer.decode(
                generated_answer_ids, skip_special_tokens=True
            )
            out.append(generated_answer)

            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

        return out


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()