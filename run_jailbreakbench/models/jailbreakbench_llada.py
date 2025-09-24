# models/jailbreakbench_llada.py
# -*- coding: utf-8 -*-

import os
import re
import sys
import json
import torch
import argparse
import logging
import subprocess
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
# from utility.generate_function import generate_llada
from utility.generate_function_llada import generate_llada
from defense_utils import Defender

# ===== Constants =====
DEFAULT_GEN_LENGTH = 128
DEFAULT_STEPS = 64
DEFAULT_BLOCK_LENGTH = 128
DEFAULT_MASK_ID = 126336
DEFAULT_MASK_COUNTS = 36
DEFAULT_TEMPERATURE = 0.5
DEFAULT_REF_TAIL_LEN = DEFAULT_BLOCK_LENGTH  # 检测用参考尾块长度默认等于 block_length

MASK_TOKEN = "<|mdm_mask|>"
START_TOKEN = "<startoftext>"
END_TOKEN = "<endoftext>"
SPECIAL_TOKEN_PATTERN = r"<mask:(\d+)>"

COLOR_BLUE = "\033[94m"
COLOR_RESET = "\033[0m"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate responses using LLaDA(-Instruct) with two-prompt schema")

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--attack_prompt", type=str, required=True,
                        help="Path to JSON with 'vanilla prompt' / 'refined prompt'")
    parser.add_argument("--output_json", type=str, required=True)

    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--gen_length", type=int, default=DEFAULT_GEN_LENGTH)
    parser.add_argument("--block_length", type=int, default=DEFAULT_BLOCK_LENGTH,
                        help="If you prefer tying to steps like older script, set equal to --steps")
    parser.add_argument("--mask_id", type=int, default=DEFAULT_MASK_ID)
    parser.add_argument("--mask_counts", type=int, default=DEFAULT_MASK_COUNTS)

    parser.add_argument("--attack_method", type=str, default="zeroshot",
                        choices=["zeroshot", "DIJA", "PAD", "other"])
    parser.add_argument("--defense_method", type=str, default=None,choices=["self-reminder","None"])

    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--cfg_scale", type=float, default=0.1)
    parser.add_argument("--remasking", type=str, default="low_confidence",
                        choices=["low_confidence", "random", "rate", "adaptive", "adaptive_step"])
    parser.add_argument("--random_rate", type=float, default=0.0)
    parser.add_argument("--injection_step", type=int, default=None)
    parser.add_argument("--alpha0", type=float, default=0.3)

    parser.add_argument("--sp_mode", type=str, default="off", choices=["off", "logit", "hidden"])
    parser.add_argument("--sp_threshold", type=float, default=0.35)
    parser.add_argument("--refinement_steps", type=int, default=8)
    parser.add_argument("--remask_ratio", type=float, default=0.9)
    parser.add_argument("--suppression_value", type=float, default=1e6)
    parser.add_argument("--fill_all_masks", action="store_true")
    parser.add_argument("--debug_print", action="store_true")

    parser.add_argument("--correct_only_first_block", dest="correct_only_first_block", action="store_true")
    parser.add_argument("--no_correct_only_first_block", dest="correct_only_first_block", action="store_false")
    parser.set_defaults(correct_only_first_block=True)

    # ---- 新增：是否自动挑 GPU（默认开启）----
    parser.add_argument("--auto_pick_gpu", dest="auto_pick_gpu", action="store_true")
    parser.add_argument("--no_auto_pick_gpu", dest="auto_pick_gpu", action="store_false")
    parser.set_defaults(auto_pick_gpu=True)

    # ---- 新增：检测用参考尾块长度（仅用于 Self-Detection，不影响生成）----
    parser.add_argument("--ref_tail_len", type=int, default=DEFAULT_REF_TAIL_LEN,
                        help="Length of the reference tail masks used ONLY for detection.")

    return parser.parse_args()


# ---------- GPU 选择工具 ----------
def _pick_gpu_by_torch() -> int | None:
    """
    用 torch.cuda.mem_get_info() 选“剩余显存最多”的 GPU。
    返回 GPU 序号；失败时返回 None。
    """
    if not torch.cuda.is_available():
        return None
    try:
        best_i, best_free = 0, -1
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                free, total = torch.cuda.mem_get_info()  # bytes
            if free > best_free:
                best_free, best_i = free, i
        return best_i
    except Exception:
        return None


def _pick_gpu_by_nvidia_smi() -> int | None:
    """
    回退方案：调用 nvidia-smi 读取每张卡的 free memory（MiB），选最大的。
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT,
            text=True,
        )
        frees = [int(x.strip()) for x in out.strip().splitlines() if x.strip()]
        if not frees:
            return None
        return max(range(len(frees)), key=lambda i: frees[i])
    except Exception:
        return None


def pick_best_gpu_index() -> int | None:
    """
    综合挑选：优先 torch，失败再用 nvidia-smi。
    """
    idx = _pick_gpu_by_torch()
    if idx is not None:
        return idx
    return _pick_gpu_by_nvidia_smi()


# ---------- 仅处理“用户侧文本”的 <mask:x> 展开 / 默认尾巴 ----------
def process_user_text(user_text: str, mask_counts: int) -> str:
    def repl(m):
        n = int(m.group(1))
        return MASK_TOKEN * max(n, 0)

    processed = re.sub(SPECIAL_TOKEN_PATTERN, repl, user_text)

    if (MASK_TOKEN not in processed) and mask_counts:
        processed = processed + START_TOKEN + (MASK_TOKEN * mask_counts) + END_TOKEN

    return processed


# ---------- 构造聊天模板 ----------
def build_chat_prompt(tokenizer, user_text_processed: str, is_instruct: bool, system_prompt: str | None = None) -> str:
    if is_instruct:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_text_processed})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        if system_prompt:
            return f"{system_prompt}\n\n{user_text_processed}"
        return user_text_processed


def get_tokenized_input(prompt: str, tokenizer, device: torch.device):
    inputs = tokenizer(prompt, return_tensors="pt")
    return inputs["input_ids"].to(device), inputs["attention_mask"].to(device)


def compute_baseline_hidden(
    vanilla_text: str,
    tokenizer,
    model,
    is_instruct: bool,
    system_prompt: str | None,
    *,
    debug_print: bool = False,
) -> torch.Tensor | None:
    """
    用“纯 vanilla 文本 + 不追加 mask 尾巴”构建 baseline，并取最后一层 hidden 的序列均值。
    返回 shape [H] 的向量；若模型不支持 hidden_states，则返回 None（上游会降级为 logits 自检）。
    注意：这仅用于你原有的 sp_mode=hidden 分支；本次新增的“尾块检测”独立于此逻辑。
    """
    try:
        # 不追加 mask 尾巴：mask_counts=0
        baseline_user = process_user_text(vanilla_text, mask_counts=0)
        baseline_prompt = build_chat_prompt(tokenizer, baseline_user, is_instruct, system_prompt)

        baseline_ids, _ = get_tokenized_input(baseline_prompt, tokenizer, model.device)
        out = model(baseline_ids, output_hidden_states=True, return_dict=True)

        if hasattr(out, "hidden_states") and out.hidden_states is not None:
            h_last = out.hidden_states[-1]            # [1, L, H]
            baseline_hidden = h_last.mean(dim=1).squeeze(0).detach()  # [H]
            if debug_print:
                logging.info(f"[Baseline] hidden OK, dim={baseline_hidden.numel()}")
            return baseline_hidden
        else:
            if debug_print:
                logging.info("[Baseline] model.hidden_states is None; fallback to logits-only SP.")
            return None
    except Exception as e:
        logging.info(f"[Baseline] failed to compute hidden: {e}; fallback to logits-only SP.")
        return None


# ======== 新增：仅用于检测的“参考尾块”工具函数（不影响生成） ========

def attach_reference_tail(user_text: str, tail_counts: int) -> str:
    """无论文本里是否已有 <mask:n>，都在末尾强制追加一段参考尾块，仅用于检测。"""
    if tail_counts <= 0:
        return user_text
    tail = START_TOKEN + (MASK_TOKEN * tail_counts) + END_TOKEN
    return f"{user_text}{tail}"


def build_detection_prompt(tokenizer, user_text: str, is_instruct: bool,
                           system_prompt: str | None, tail_counts: int) -> str:
    """
    检测用 prompt：
    1) 先展开文本内的 <mask:n>（不追加默认尾巴）；
    2) 再强制追加同长度的参考尾块；
    3) 再套 chat 模板。
    """
    processed = process_user_text(user_text, mask_counts=0)  # 展开内嵌 <mask:n>，不加默认尾巴
    user_with_tail = attach_reference_tail(processed, tail_counts)
    return build_chat_prompt(tokenizer, user_with_tail, is_instruct, system_prompt)


@torch.no_grad()
def _find_tail_span(tok: torch.Tensor, mask_id: int, tail_len: int) -> tuple[int, int]:
    """
    在 token 序列 tok 中找到“最后 tail_len 个 mask_id”的覆盖区间 [start, end)。
    假设我们强制追加的尾块是连续的 <|mdm_mask|> * tail_len。
    """
    mask_pos = (tok == mask_id).nonzero(as_tuple=False).squeeze(1)
    assert mask_pos.numel() >= tail_len, f"Reference tail not found or too short: have {mask_pos.numel()}, need {tail_len}"
    last = mask_pos[-tail_len:]                     # 取最后 tail_len 个 mask 位置
    start = int(last.min().item())
    end   = int(last.max().item()) + 1
    return start, end


@torch.no_grad()
def first_step_tail_mean_hidden(model, tokenizer, prompt: str, mask_id: int, tail_len: int) -> torch.Tensor:
    """
    做一次前向，取“最后 tail_len 个 mask”的片段作为参考尾块，在这段上做首步（近似）隐藏均值。
    对 dLLM，首次前向的最后一层隐藏可近似“首步”。
    返回 [H]
    """
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)  # [1, L]
    out = model(ids, output_hidden_states=True, return_dict=True)
    if not hasattr(out, "hidden_states") or out.hidden_states is None:
        raise RuntimeError("Model did not return hidden_states; cannot compute hidden-based SP.")
    h_last = out.hidden_states[-1]                      # [1, L, H]
    tok = ids[0]
    tail_start, tail_end = _find_tail_span(tok, mask_id, tail_len)
    mu = h_last[:, tail_start:tail_end, :].mean(dim=1).squeeze(0).detach()  # [H]
    return mu


def cosine_distance_vec(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.to(torch.float32); b = b.to(torch.float32)
    a = a / (a.norm(p=2) + 1e-12)
    b = b / (b.norm(p=2) + 1e-12)
    return float(1.0 - torch.dot(a, b).item())


# ===================== 生成响应 =====================

def generate_response(vanilla_prompt: str, prompt: str, tokenizer, model, args, baseline_hidden=None, system_prompt: str | None = None) -> str:
    input_ids, attention_mask = get_tokenized_input(prompt, tokenizer, model.device)
    vanilla_ids, _ = get_tokenized_input(vanilla_prompt, tokenizer, model.device)
    # === 计算 system/self-reminder 的 token 长度，并构造保护掩码 ===
    protected_index = torch.zeros_like(input_ids, dtype=torch.bool, device=model.device)
    if system_prompt:
        # 只包含 system 的同模板 token 化（不加生成提示）
        sys_only_ids = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}],
            tokenize=True, add_generation_prompt=False, return_tensors="pt"
        ).to(model.device)
        sys_len = sys_only_ids.shape[1]
        sys_len = min(sys_len, input_ids.shape[1])  # 防御越界
        protected_index[:, :sys_len] = True

    # Count common prefix length (token-wise)
    matching_count = min(
        sum(a.item() == b.item() for a, b in zip(vanilla_ids[0], input_ids[0])),
        len(vanilla_ids[0]),
    )

    output_ids = generate_llada(
        input_ids=input_ids,
        attention_mask=attention_mask,
        model=model,
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=args.temperature,
        mask_id=args.mask_id,

        # —— 传入 tokenizer + 高级参数 —— 
        tokenizer=tokenizer,
        cfg_scale=args.cfg_scale,
        remasking=args.remasking,
        random_rate=args.random_rate,
        injection_step=args.injection_step,
        alpha0=args.alpha0,
        sp_mode=args.sp_mode,
        sp_threshold=args.sp_threshold,
        refinement_steps=args.refinement_steps,
        remask_ratio=args.remask_ratio,
        suppression_value=args.suppression_value,
        correct_only_first_block=args.correct_only_first_block,
        fill_all_masks=args.fill_all_masks,
        debug_print=args.debug_print,
        baseline_hidden=baseline_hidden,
        attack_method=args.attack_method,
        pad_anchors=["Step 1:", "Step 2:", "Step 3:"],
        pad_in_uncond=True,   #####
        protected_index=protected_index,
    )

    response = tokenizer.batch_decode(output_ids[:, matching_count:], skip_special_tokens=True)[0]
    if args.attack_method == "DIJA":
        # 部分数据会在解码里带上 'assistant\n' 之后的内容，这里做一次硬切
        response = response.split("assistant\n")[0]

    return response


def pick_two_prompt_fields(item: dict) -> tuple[str, str]:
    vanilla = (
        item.get("vanilla prompt")
        or item.get("goal")
        or item.get("Behavior")
        or ""
    )
    refined = (
        item.get("refined prompt")
        or item.get("refined_goal")
        or item.get("Refined_behavior")
        or ""
    )
    return vanilla, refined


def should_use_refined(args, attack_prompt_path, refined_text):
    path_flag = ("refine" in os.path.basename(attack_prompt_path).lower())
    if args.attack_method.lower() == "dija" or path_flag:
        return bool(refined_text.strip())
    return False


def main():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    args = parse_args()

    # ---- 自动挑选显存最空闲的 GPU（可通过 --no_auto_pick_gpu 关闭）----
    if torch.cuda.is_available():
        if args.auto_pick_gpu:
            best_idx = pick_best_gpu_index()
            if best_idx is not None:
                try:
                    torch.cuda.set_device(best_idx)
                except Exception:
                    pass  # set_device 失败就按默认 cuda 走
                device = torch.device(f"cuda:{best_idx}")
                try:
                    with torch.cuda.device(device):
                        free, total = torch.cuda.mem_get_info()
                    logging.info(f"[GPU] Auto-picked cuda:{best_idx} "
                                 f"(free {free/1024/1024/1024:.1f} GB / total {total/1024/1024/1024:.1f} GB)")
                except Exception:
                    logging.info(f"[GPU] Auto-picked cuda:{best_idx}")
            else:
                device = torch.device("cuda")
                logging.info("[GPU] Auto-pick failed, fallback to default CUDA device.")
        else:
            device = torch.device("cuda")
            logging.info("[GPU] Auto-pick disabled, using default CUDA device.")
    else:
        device = torch.device("cpu")
        logging.info("[GPU] CUDA not available, using CPU.")

    # Defender
    defender = Defender(args.defense_method) if args.defense_method and args.defense_method.lower() != "none" else None

    # Heuristic instruct check
    is_instruct = ("instruct" in args.model_path.lower()) or ("1.5" in args.model_path)

    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()

    # Load data
    with open(args.attack_prompt, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []

    with torch.no_grad():
        for item in tqdm(data, desc="Processing data", total=len(data)):
            vanilla_text, refined_text = pick_two_prompt_fields(item)
            use_refined = should_use_refined(args, args.attack_prompt, refined_text)

            chosen_text = refined_text if use_refined and refined_text.strip() else vanilla_text

            # --- 只在用户侧文本做 mask 展开/默认尾巴（生成用）---
            chosen_user = process_user_text(chosen_text, args.mask_counts)
            vanilla_user = process_user_text(vanilla_text, args.mask_counts)

            # --- defense：self-reminder 注入 system；其它防御改写用户文本 ---
            system_for_both = None
            if defender and getattr(defender, "defense", None) == "self-reminder":
                ret = defender.handler(chosen_text)
                if isinstance(ret, tuple) and len(ret) >= 1:
                    system_for_both = ret[0]
                elif isinstance(ret, str):
                    system_for_both = ret
            elif defender:
                chosen_user = defender.handler(chosen_user)
                vanilla_user = defender.handler(vanilla_user)

            # --- 构造最终 prompt（vanilla 与 chosen 使用同一 system） ---
            prompt = build_chat_prompt(tokenizer, chosen_user, is_instruct, system_for_both)
            vanilla_prompt = build_chat_prompt(tokenizer, vanilla_user, is_instruct, system_for_both)

            # ========= 新增：仅用于“模板越狱检测”的双前向（tail-only）=========
            ref_tail_len = getattr(args, "ref_tail_len", DEFAULT_REF_TAIL_LEN)
            det_vanilla_prompt = build_detection_prompt(tokenizer, vanilla_text, is_instruct, system_for_both, ref_tail_len)
            det_refined_prompt = build_detection_prompt(tokenizer, chosen_text,  is_instruct, system_for_both, ref_tail_len)

            sp_hid_tail = None
            template_attack = None
            try:
                van_mu = first_step_tail_mean_hidden(model, tokenizer, det_vanilla_prompt, args.mask_id, ref_tail_len)  # [H]
                ref_mu = first_step_tail_mean_hidden(model, tokenizer, det_refined_prompt, args.mask_id, ref_tail_len)  # [H]
                sp_hid_tail = cosine_distance_vec(van_mu, ref_mu)  # 1 - cos
                template_attack = (sp_hid_tail >= args.sp_threshold)
            except Exception as e:
                logging.info(f"[TailDetection] skipped due to: {e}")

            # --- 计算 baseline_hidden（用于你原有的 sp_mode=hidden；与 tail-only 检测互不影响） ---
            baseline_hidden = None
            if args.sp_mode == "hidden":
                baseline_hidden = compute_baseline_hidden(
                    vanilla_text=vanilla_text,
                    tokenizer=tokenizer,
                    model=model,
                    is_instruct=is_instruct,
                    system_prompt=system_for_both,
                    debug_print=args.debug_print,
                )

            # --- 调试预览 ---
            if os.getenv("DEBUG_SHOW_PROMPT", "0") == "1":
                logging.info(("PROMPT_PREVIEW=" + prompt[:1200]).replace("\n", "\\n"))
            if os.getenv("DEBUG_SHOW_TAIL", "0") == "1":
                logging.info(("TAIL_VAN=" + det_vanilla_prompt[:600]).replace("\n", "\\n"))
                logging.info(("TAIL_REF=" + det_refined_prompt[:600]).replace("\n", "\\n"))

            # --- 生成 ---
            response = generate_response(vanilla_prompt, prompt, tokenizer, model, args, baseline_hidden=baseline_hidden, system_prompt=system_for_both)

            logging.info(f"{COLOR_BLUE}Response: {response}{COLOR_RESET}\n")

            # --- 保存记录：加入 final_prompt & 检测指标 ---
            rec = {
                "vanilla prompt": vanilla_text,
                "refined prompt": refined_text,
                "used_prompt_type": "refined" if (use_refined and refined_text.strip()) else "vanilla",
                "final_prompt": prompt,          # 模型实际输入（生成用）
                "response": response,
                # 新增：检测指标（tail-only hidden distance）
                "sp_hid_tail": sp_hid_tail,
                "template_attack": template_attack,
                "ref_tail_len": ref_tail_len,
            }
            for k in ("source", "category", "goal", "refined_goal", "Behavior", "Refined_behavior", "target"):
                if k in item:
                    rec[k] = item[k]

            results.append(rec)

    out_dir = os.path.dirname(args.output_json) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    logging.info(f"Saved JSON to {args.output_json}")


if __name__ == "__main__":
    main()
