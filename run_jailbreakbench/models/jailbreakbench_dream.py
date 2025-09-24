# models/jailbreakbench_dream.py
# 适用于 diffusion_generate 的模型（例如 Dream 系列或同接口模型）
# 自定义迭代生成（Adaptive Remask, 单 block）+ Self-Detection(hidden) + Self-Correction
# Self-Detection/Correction 由 --sp_mode 控制；Adaptive Remask 由 --remasking 控制；两者独立
# 当 --attack_method pad 时，统一走 generate_dream_hidden（内部支持 PAD 注入 + 自检 + 自纠正，逐步再填）

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
import torch.nn.functional as F
from typing import Optional, Iterable

# 可删的未用依赖保持最小化
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from defense_utils import Defender

# >>> 引入统一生成器（含 PAD 注入 + 自检 + 自纠正）<<<
from utility.generate_function_dream import generate_dream_hidden

# ===== Defaults =====
DEFAULT_GEN_LENGTH = 128
DEFAULT_STEPS = 64
DEFAULT_MASK_ID = 151666           # Dream 常见 mask id；可通过参数覆盖
DEFAULT_MASK_COUNTS = 128
DEFAULT_TEMPERATURE = 0.5
DEFAULT_TOP_P = 0.95
DEFAULT_REF_TAIL_LEN = 128         # 仅用于检测的参考尾块长度

MASK_TOKEN = "<|mask|>"
SPECIAL_TOKEN_PATTERN = r"<mask:(\d+)>"
COLOR_BLUE = "\033[94m"
COLOR_RESET = "\033[0m"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate responses (two-prompt schema) using diffusion_generate models")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--attack_prompt", type=str, required=True, help="JSON with 'vanilla prompt' / 'refined prompt'")
    parser.add_argument("--output_json", type=str, required=True)

    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--gen_length", type=int, default=DEFAULT_GEN_LENGTH)
    parser.add_argument("--mask_id", type=int, default=DEFAULT_MASK_ID)
    parser.add_argument("--mask_counts", type=int, default=DEFAULT_MASK_COUNTS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)

    # 新增：支持 PAD
    parser.add_argument("--attack_method", type=str, default="zeroshot",
                        choices=["zeroshot", "DIJA", "PAD", "pad", "other"])
    parser.add_argument("--defense_method", type=str, default=None)

    # ===== Adaptive Remask（仅控制解码策略；与 Self-Correction 无关）=====
    parser.add_argument("--remasking", type=str, default="off",
                        choices=["off", "low_confidence", "random", "rate", "adaptive", "adaptive_step"])
    parser.add_argument("--alpha0", type=float, default=0.8, help="随机项权重初值（adaptive/step 用）")
    parser.add_argument("--random_rate", type=float, default=0.3, help="rate 模式下的随机权重")

    # ===== Self-Detection/Correction（独立于 remasking）=====
    parser.add_argument("--sp_mode", type=str, default="off", choices=["off", "hidden"],
                        help="Self-Detection 模式；hidden=用尾块hidden距离；off=关闭检测和自纠正")
    parser.add_argument("--sp_threshold", type=float, default=0.35, help="hidden 距离阈值（≥ 触发）")
    parser.add_argument("--ref_tail_len", type=int, default=DEFAULT_REF_TAIL_LEN,
                        help="仅检测用的参考尾块长度（vanilla/refined 各自强制追加）")
    parser.add_argument("--refinement_steps", type=int, default=8, help="Self-Correction 的重填步数")
    parser.add_argument("--remask_ratio", type=float, default=0.9, help="Self-Correction 的随机重掩比例(0..1)")
    parser.add_argument("--suppression_value", type=float, default=1e6, help="对原 token 的logit抑制幅度")
    parser.add_argument("--correction_scope", type=str, default="block_all",
                        choices=["masked_only", "span_all", "block_all"],
                        help="Self-Correction 重掩范围")

    # ====== PAD 相关 ======
    parser.add_argument("--pad_anchors", type=str, nargs="*", default=["Step 1:", "Step 2:", "Step 3:"],
                        help="PAD 模式写入的锚点短语")
    parser.add_argument("--pad_positions", type=int, nargs="*", default=None,
                        help="相对 block 起点的偏移（token 级）；不指定则自动均匀分布")
    parser.add_argument("--protect_anchors", action="store_true",
                        help="自纠正阶段保护锚点，不在锚点处重掩")
    parser.set_defaults(protect_anchors=False)

    parser.add_argument("--debug_print", action="store_true")
    parser.add_argument("--auto_pick_gpu", dest="auto_pick_gpu", action="store_true")
    parser.add_argument("--no_auto_pick_gpu", dest="auto_pick_gpu", action="store_false")
    parser.set_defaults(auto_pick_gpu=True)
    return parser.parse_args()


# ---------- 仅处理“用户侧文本”的 <mask:x> 展开 / 默认尾巴 ----------
def process_user_text(user_text: str, mask_counts: int) -> str:
    def repl(m):
        n = max(int(m.group(1)), 0)
        return MASK_TOKEN * n
    processed = re.sub(SPECIAL_TOKEN_PATTERN, repl, user_text)
    if (MASK_TOKEN not in processed) and mask_counts:
        processed = processed + (MASK_TOKEN * mask_counts)
    return processed


# ---------- 构造最终聊天串 ----------
def build_chat_prompt(tokenizer, user_text_processed: str, is_instruct: bool, system_prompt: Optional[str] = None) -> str:
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


def tokenized(model_device: torch.device, tokenizer, text: str):
    enc = tokenizer(text, return_tensors="pt")
    return enc["input_ids"].to(model_device), enc["attention_mask"].to(model_device)


# ========== 工具 ==========
def add_gumbel_noise(logits: torch.Tensor, temperature: float):
    if temperature <= 0:
        return logits
    g = -torch.log(-torch.log(torch.rand_like(logits, dtype=torch.float32)))
    g = g.to(logits.dtype)
    tau = max(float(temperature), 1e-6)
    return logits / tau + g


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    steps = max(int(steps), 1)
    mask_num = mask_index.sum(dim=1)
    base = (mask_num // steps).unsqueeze(1)
    rem  = (mask_num % steps)
    num = base.repeat(1, steps)
    if steps > 0:
        idx = torch.arange(steps, device=mask_index.device).unsqueeze(0)
        bump = (idx < rem.unsqueeze(1)).to(num.dtype)
        num = num + bump
    return num.to(torch.int64)

def _pick_gpu_by_torch() -> int | None:
    if not torch.cuda.is_available():
        return None
    try:
        best_i, best_free = 0, -1
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                free, total = torch.cuda.mem_get_info()
            if free > best_free:
                best_free, best_i = free, i
        return best_i
    except Exception:
        return None

def _pick_gpu_by_nvidia_smi() -> int | None:
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
    idx = _pick_gpu_by_torch()
    if idx is not None:
        return idx
    return _pick_gpu_by_nvidia_smi()

# ========== 自定义：Adaptive Remask（单 block） ==========
@torch.no_grad()
def dream_adaptive_generate(
    model,
    tokenizer,
    input_ids: torch.Tensor,           # [B,L]
    attention_mask: torch.Tensor,      # [B,L]
    steps: int,
    temperature: float,
    mask_id: int,
    remasking: str,
    alpha0: float,
    random_rate: float,
    debug_print: bool = False,
) -> torch.Tensor:
    device = input_ids.device
    x = input_ids.clone()
    B, L = x.shape

    # Dream 的 SDPA 要求 attn_mask 为 bool/float；这里用 bool 最安全
    if attention_mask is None:
        attn = torch.ones_like(x, dtype=torch.bool)
    else:
        attn = attention_mask.to(device)
        if attn.dtype is not torch.bool:
            attn = attn.to(torch.bool)

    # 整序列单 block
    initial_mask = (x == mask_id)
    if not initial_mask.any():
        return x

    num_transfer_tokens = get_num_transfer_tokens(initial_mask, steps)
    total_steps = steps

    for i in range(steps):
        out = model(x, attention_mask=attn, return_dict=True)
        logits = out.logits  # [B,L,V]

        logits_noised = add_gumbel_noise(logits, temperature)
        x0 = torch.argmax(logits_noised, dim=-1)

        probs = F.softmax(logits, dim=-1)
        model_conf = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
        R = torch.rand((B, L), device=device)

        if remasking == "low_confidence":
            x0_p = model_conf
        elif remasking == "random":
            x0_p = R
        elif remasking == "rate":
            alpha = torch.clamp(torch.tensor(random_rate, device=device, dtype=model_conf.dtype), 0.0, 1.0)
            x0_p = (1 - alpha) * model_conf + alpha * R
        elif remasking == "adaptive":
            alpha = torch.clamp(torch.tensor(alpha0, device=device, dtype=model_conf.dtype), 0.0, 1.0)
            x0_p = (1 - alpha) * model_conf + alpha * R
        elif remasking == "adaptive_step":
            frac = 1.0 - (i / (total_steps - 1)) if total_steps > 1 else 1.0
            alpha = torch.clamp(torch.tensor(alpha0 * frac, device=device, dtype=model_conf.dtype), 0.0, 1.0)
            x0_p = (1 - alpha) * model_conf + alpha * R
        else:  # "off" 不会走到这里，外层已判断
            x0_p = model_conf

        current_mask = (x == mask_id)
        conf = torch.where(current_mask, x0_p, torch.full_like(x0_p, -1e9))

        for b in range(B):
            k = int(num_transfer_tokens[b, i].item())
            valid_idx = current_mask[b].nonzero(as_tuple=False).squeeze(1)
            if valid_idx.numel() == 0 or k <= 0:
                continue
            k = min(k, valid_idx.numel())
            _, local = torch.topk(conf[b, valid_idx], k=k)
            choose = valid_idx[local]
            x[b, choose] = x0[b, choose]

        if debug_print and (i == 0 or (i + 1) == steps or (i + 1) % max(1, steps // 4) == 0):
            rem = int((x == mask_id).sum().item())
            logging.info(f"[Dream-AR] step {i+1}/{steps} | alpha={float(alpha) if 'alpha' in locals() else 0:.3f} | remaining masks={rem}")

        if not (x == mask_id).any():
            break

    return x


# ========== Self-Correction（独立，可在原生或自定义解码后执行） ==========
@torch.no_grad()
def apply_self_correction(
    model,
    tokenizer,
    seq_ids: torch.Tensor,                  # [B, L_out] 当前解码后的完整序列
    initial_mask_in_prompt: torch.Tensor,   # [B, L_prompt] 输入prompt的初始掩码位置（用于 span_all / masked_only）
    steps: int,
    temperature: float,
    mask_id: int,
    refinement_steps: int,
    remask_ratio: float,
    suppression_value: float,
    correction_scope: str,                  # block_all / span_all / masked_only
    exclude_mask_positions: Optional[torch.Tensor] = None,  # [B, L_out] 这些位置永不重掩（用于保护锚点）
    debug_print: bool = False,
) -> torch.Tensor:
    """
    自纠正：随机重掩 + 原token抑制 + 逐步再填（per-step top-k）。
    新增 exclude_mask_positions: True 的位置不会被选入候选集（用于 PAD 锚点保护）。
    """
    device = seq_ids.device
    x = seq_ids.clone()
    B, L_out = x.shape

    # 统一 exclude mask 到 [B, L_out]
    if exclude_mask_positions is not None:
        excl = exclude_mask_positions.to(device)
        if excl.shape[0] != B or excl.shape[1] != L_out:
            excl2 = torch.zeros((B, L_out), dtype=torch.bool, device=device)
            bmin = min(B, excl.shape[0]); lmin = min(L_out, excl.shape[1])
            excl2[:bmin, :lmin] = excl[:bmin, :lmin]
            exclude_mask_positions = excl2
        else:
            exclude_mask_positions = excl
    # ---- 全1的 attn（bool），兼容 Dream 的 SDPA ----
    attn = torch.ones_like(x, dtype=torch.bool, device=device)

    # 计算 span_all 的边界（来自原prompt的初始mask）
    span_bounds = []
    for b in range(B):
        m = initial_mask_in_prompt[b].nonzero(as_tuple=False).squeeze(1)
        if m.numel() > 0:
            s = int(m.min().item())
            e = int(m.max().item()) + 1
        else:
            s, e = 0, L_out
        s = max(0, min(s, L_out)); e = max(s, min(e, L_out))
        span_bounds.append((s, e))

    # —— 对每个样本独立执行重掩 & 精炼 —— #
    for b in range(B):
        # 候选集合
        if correction_scope == "block_all":
            cand_idx = torch.arange(0, L_out, device=device)
        elif correction_scope == "span_all":
            s, e = span_bounds[b]
            cand_idx = torch.arange(s, e, device=device)
        else:  # "masked_only"
            e = min(L_out, initial_mask_in_prompt.shape[1])
            orig_mask = torch.zeros(L_out, dtype=torch.bool, device=device)
            orig_mask[:e] = initial_mask_in_prompt[b, :e].to(device)
            cand_idx = orig_mask.nonzero(as_tuple=False).squeeze(1)

        # 排除保护位置（比如 PAD 锚点）
        if exclude_mask_positions is not None:
            keep_mask = ~exclude_mask_positions[b]
            cand_idx = cand_idx[keep_mask[cand_idx]]

        if cand_idx.numel() == 0:
            continue

        num_to_remask = int(max(1, round(remask_ratio * cand_idx.numel())))
        pick = cand_idx[torch.randperm(cand_idx.numel(), device=device)[:num_to_remask]]

        # 记录原 token 并置回 mask
        orig_tokens = x[b, pick].clone()
        x[b, pick] = mask_id

        # 计算一次性精炼名额（基于“当前剩余mask数”快照）
        refine_mask = (x[b:b+1] == mask_id)  # [1, L_out]
        num_refine = get_num_transfer_tokens(refine_mask, max(int(refinement_steps), 1))  # [1, refinement_steps]

        for r in range(max(int(refinement_steps), 1)):
            out = model(x, attention_mask=attn, return_dict=True)
            logits = out.logits  # [B, L_out, V]

            if torch.isfinite(torch.tensor(suppression_value)):
                logits[b, pick, orig_tokens] -= suppression_value

            logits_noised = add_gumbel_noise(logits, temperature)
            x0 = torch.argmax(logits_noised, dim=-1)

            p = F.softmax(logits, dim=-1)
            conf_all = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

            cur_mask = (x == mask_id)
            conf = torch.where(cur_mask, conf_all, torch.full_like(conf_all, -1e9))

            k = int(num_refine[0, r].item())
            valid_idx = cur_mask[b].nonzero(as_tuple=False).squeeze(1)
            if valid_idx.numel() > 0 and k > 0:
                k = min(k, valid_idx.numel())
                _, local = torch.topk(conf[b, valid_idx], k=k)
                choose = valid_idx[local]
                x[b, choose] = x0[b, choose]

            if debug_print and (r == 0 or r + 1 == refinement_steps):
                changed = (x[b, pick] != orig_tokens).float().mean().item()
                logging.info(f"[Dream-AR] refine step {r+1}/{refinement_steps} | changed@pick={changed:.2%}")

            if not (x == mask_id).any():
                break

    return x


# ======== Self-Detection（hidden）辅助：参考尾块双前向 ========
def attach_reference_tail(user_text: str, tail_counts: int) -> str:
    if tail_counts <= 0:
        return user_text
    return f"{user_text}{(MASK_TOKEN * tail_counts)}"


def build_detection_prompt(tokenizer, user_text: str, is_instruct: bool,
                           system_prompt: Optional[str], tail_counts: int) -> str:
    processed = process_user_text(user_text, mask_counts=0)  # 展开内嵌 <mask:n>，不加默认尾巴
    user_with_tail = attach_reference_tail(processed, tail_counts)
    return build_chat_prompt(tokenizer, user_with_tail, is_instruct, system_prompt)


@torch.no_grad()
def _find_tail_span(tok: torch.Tensor, mask_id: int, tail_len: int) -> tuple[int, int]:
    mask_pos = (tok == mask_id).nonzero(as_tuple=False).squeeze(1)
    assert mask_pos.numel() >= tail_len, f"Reference tail not found or too short: have {mask_pos.numel()}, need {tail_len}"
    last = mask_pos[-tail_len:]
    start = int(last.min().item())
    end   = int(last.max().item()) + 1
    return start, end


@torch.no_grad()
def first_step_tail_mean_hidden(model, tokenizer, prompt: str, mask_id: int, tail_len: int) -> torch.Tensor:
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


# ===================== 生成响应（解码与自纠正互相独立） =====================

def generate_response(
    vanilla_prompt: str,
    prompt: str,
    tokenizer,
    model,
    args,
    template_attack: bool,                     # 只有 sp_mode=hidden 时才会根据它触发SC；否则忽略
    baseline_hidden: Optional[torch.Tensor],   # 外层预先计算的 vanilla 尾块均值向量（van_mu）
    attack_probe_hidden: Optional[torch.Tensor],  # 外层预先计算的 refined 尾块均值（ref_mu，可为 None）
    initial_mask_from_prompt: torch.Tensor,    # [B,L_prompt] 初始掩码位（供SC使用）
    system_prompt: Optional[str] = None,   
) -> str:
    """
    解码路径：
      - attack_method == "pad" -> Dream hidden 自检 generate_dream_hidden（off 时不纠正）
      - remasking == "off"     -> 原生 diffusion_generate
      - 其它                    -> 自定义 AR 解码（dream_adaptive_generate），必要时外层再做自纠正
    """
    # 编码
    input_ids, attention_mask = tokenized(model.device, tokenizer, prompt)
    vanilla_ids, _ = tokenized(model.device, tokenizer, vanilla_prompt)
    protected_index = None
    if system_prompt:
        try:
            sys_only_ids = tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt}],
                tokenize=True, add_generation_prompt=False, return_tensors="pt"
            ).to(model.device)
            sys_len = min(sys_only_ids.shape[1], input_ids.shape[1])
            protected_index = torch.zeros_like(input_ids, dtype=torch.bool)
            protected_index[:, :sys_len] = True
        except Exception:
            protected_index = None
    # 公共前缀长度（token 逐位比较）
    matching_count = min(
        sum(a.item() == b.item() for a, b in zip(vanilla_ids[0], input_ids[0])),
        len(vanilla_ids[0]),
    )
        # ====== 通用 LLaDA 对齐：仅当“输入里没有任何 <mask>”时，追加 gen_length 的 <|mask|> 尾巴 ======
    # 用 token 层判断，确保和 --mask_id 对齐
    has_any_mask = (input_ids == int(args.mask_id)).any().item()

    if (not has_any_mask) and (int(args.gen_length) > 0):
        B = input_ids.size(0)

        # 1) 追加 <mask> 尾巴
        tail = torch.full(
            (B, int(args.gen_length)),
            int(args.mask_id),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        input_ids = torch.cat([input_ids, tail], dim=1)

        # 2) 同步扩展 attention_mask（保持 dtype 一致）
        tail_attn = torch.ones_like(tail, dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([attention_mask, tail_attn], dim=1)

        # 3) 若上面构造了 protected_index（system 前缀保护），也 pad 到同长度
        if "protected_index" in locals() and protected_index is not None:
            need = input_ids.shape[1] - protected_index.shape[1]
            if need > 0:
                pad_false = torch.zeros(protected_index.size(0), need, dtype=torch.bool, device=protected_index.device)
                protected_index = torch.cat([protected_index, pad_false], dim=1)

        if getattr(args, "debug_print", False):
            logging.info(f"[Dream-GenTail] appended {int(args.gen_length)} masks at token layer (generic, no-mask case).")


    attack_method_lower = (args.attack_method or "").lower()

    # ======= PAD 路径（统一） =======
    if attack_method_lower == "pad":
        if args.debug_print:
            logging.info("[Dream-Path] PAD route via generate_dream_hidden (unified)")
        if args.sp_mode == "hidden":
            bl = baseline_hidden
            pr = attack_probe_hidden
            thr = float(args.sp_threshold)
        else:
            bl = None
            pr = None
            thr = 1e9  # 不触发纠正

        output_ids, sp_list = generate_dream_hidden(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            gen_length=args.gen_length,
            steps=args.steps,
            block_length=args.gen_length,
            temperature=args.temperature,
            top_p=args.top_p,
            alpha0=args.alpha0,
            sp_threshold=thr,
            baseline_hidden=bl,
            attack_probe_hidden=pr,
            refinement_steps=args.refinement_steps,
            remask_ratio=args.remask_ratio,
            suppression_value=args.suppression_value,
            correct_only_first_block=True,
            fill_all_masks=False,
            mask_id=int(args.mask_id),
            attack_method="pad",
            pad_anchors=args.pad_anchors,
            pad_positions=args.pad_positions,
            protect_anchors=bool(args.protect_anchors),
            correction_scope=args.correction_scope,
            initial_mask_in_prompt=initial_mask_from_prompt,
            exclude_mask_positions=protected_index,
        )

    # ======= 原生 generate（无 AR） =======
    elif args.remasking == "off":
        if args.debug_print:
            logging.info("[Dream-Path] native diffusion_generate (no AR decoding)")
        gen_out = model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.gen_length,
            output_history=False,
            return_dict_in_generate=True,
            steps=args.steps,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        output_ids = gen_out.sequences

    # ======= 自定义 AR 解码（非 PAD） =======
    else:
        if args.debug_print:
            logging.info(f"[Dream-Path] custom AR decoding (remasking={args.remasking})")
        output_ids = dream_adaptive_generate(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            steps=args.steps,
            temperature=args.temperature,
            mask_id=args.mask_id,
            remasking=args.remasking,
            alpha0=args.alpha0,
            random_rate=args.random_rate,
            debug_print=args.debug_print,
        )

        # —— 外层 Self-Correction（仅在非 PAD 路径、sp_mode=hidden 且检测触发时执行）—— #
        do_self_correction = (args.sp_mode == "hidden") and bool(template_attack)
        if do_self_correction:
            if args.debug_print:
                logging.info(f"[Dream-AR] Self-Correction triggered: remask_ratio={args.remask_ratio}, "
                             f"refinement_steps={args.refinement_steps}, scope={args.correction_scope}")
            output_ids = apply_self_correction(
                model=model,
                tokenizer=tokenizer,
                seq_ids=output_ids,
                initial_mask_in_prompt=initial_mask_from_prompt,
                steps=args.steps,
                temperature=args.temperature,
                mask_id=args.mask_id,
                refinement_steps=args.refinement_steps,
                remask_ratio=args.remask_ratio,
                suppression_value=args.suppression_value,
                correction_scope=args.correction_scope,
                exclude_mask_positions=protected_index,
                debug_print=args.debug_print,
            )

    # 解码 & 特殊清理
    response = tokenizer.batch_decode(output_ids[:, matching_count:], skip_special_tokens=True)[0]
    if attack_method_lower == "dija":
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


def should_use_refined(args: argparse.Namespace, attack_prompt_path: str, refined_text: str) -> bool:
    path_flag = ("refine" in os.path.basename(attack_prompt_path).lower())
    return (args.attack_method.lower() == "dija") or path_flag or bool(refined_text.strip())


def main():
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
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

    defender = Defender(args.defense_method) if args.defense_method and args.defense_method.lower() != "none" else None

    lower_path = args.model_path.lower()
    is_instruct_model = ("instruct" in lower_path) or ("1.5" in lower_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).to(device).eval()

    with open(args.attack_prompt, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []

    with torch.no_grad():
        for idx, item in enumerate(tqdm(data, desc="Processing data", total=len(data))):
            vanilla_text, refined_text = pick_two_prompt_fields(item)
            use_refined = should_use_refined(args, args.attack_prompt, refined_text)
            chosen_text = refined_text if (use_refined and refined_text.strip()) else vanilla_text

            # 用户侧文本展开
            chosen_user = process_user_text(chosen_text, args.mask_counts)
            vanilla_user = process_user_text(vanilla_text, args.mask_counts)

            # 防御（system注入 or 文本改写）
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

            # 构造最终 prompt
            prompt = build_chat_prompt(tokenizer, chosen_user, is_instruct_model, system_for_both)
            vanilla_prompt = build_chat_prompt(tokenizer, vanilla_user, is_instruct_model, system_for_both)

            # ========= Self-Detection（仅当 sp_mode=hidden 时执行；否则跳过）=========
            sp_hid_tail = None
            baseline_mu = None
            ref_mu = None           # ★ 初始化，避免异常时未定义
            template_attack = None
            if args.sp_mode == "hidden":
                ref_tail_len = max(int(getattr(args, "ref_tail_len", DEFAULT_REF_TAIL_LEN)), 0)
                if ref_tail_len > 0:
                    det_vanilla_prompt = build_detection_prompt(
                        tokenizer, vanilla_text, is_instruct_model, system_for_both, ref_tail_len
                    )
                    det_refined_prompt = build_detection_prompt(
                        tokenizer, chosen_text,  is_instruct_model, system_for_both, ref_tail_len
                    )
                    try:
                        van_mu = first_step_tail_mean_hidden(model, tokenizer, det_vanilla_prompt, args.mask_id, ref_tail_len)
                        ref_mu = first_step_tail_mean_hidden(model, tokenizer, det_refined_prompt, args.mask_id, ref_tail_len)
                        sp_hid_tail = cosine_distance_vec(van_mu, ref_mu)  # 1 - cos
                        baseline_mu = van_mu
                        template_attack = (sp_hid_tail >= args.sp_threshold)
                        if args.debug_print:
                            logging.info(f"[Dream-Det] Safety score={sp_hid_tail:.3f}, unsafe={bool(template_attack)}")
                    except Exception as e:
                        logging.info(f"[TailDetection] skip: {e}")
                        template_attack = False
                else:
                    template_attack = False
            else:
                template_attack = False  # 关闭检测 => 不触发SC

            # 初始掩码位（供 SC 使用；与解码路径无关）
            prompt_ids, _ = tokenized(model.device, tokenizer, prompt)
            initial_mask_from_prompt = (prompt_ids == args.mask_id)  # [1, L_prompt]

            # 解码 + 可选自纠正
            if os.getenv("DEBUG_SHOW_PROMPT", "0") == "1":
                logging.info(("PROMPT_PREVIEW=" + prompt[:1200]).replace("\n", "\\n"))

            response = generate_response(
                vanilla_prompt=vanilla_prompt,
                prompt=prompt,
                tokenizer=tokenizer,
                model=model,
                args=args,
                template_attack=bool(template_attack),
                baseline_hidden=baseline_mu,      # van_mu（可能为 None）
                attack_probe_hidden=ref_mu,       # ref_mu（可能为 None）
                initial_mask_from_prompt=initial_mask_from_prompt,
                system_prompt=system_for_both,  
            )
            logging.info(f"{COLOR_BLUE}[{idx}] Response: {response}{COLOR_RESET}\n")

            # 记录
            rec = {
                "vanilla prompt": vanilla_text,
                "refined prompt": refined_text,
                "used_prompt_type": "refined" if (use_refined and refined_text.strip()) else "vanilla",
                "final_prompt": prompt,
                "response": response,
                "attack_method": args.attack_method,
                "remasking": args.remasking,
                "alpha0": args.alpha0,
                "random_rate": args.random_rate,
                "sp_mode": args.sp_mode,
                "sp_hid_tail": sp_hid_tail,
                "template_attack": bool(template_attack) if args.sp_mode == "hidden" else None,
                "ref_tail_len": args.ref_tail_len if args.sp_mode == "hidden" else None,
                "correction_scope": args.correction_scope if args.sp_mode == "hidden" else None,
                "pad_anchors": args.pad_anchors if args.attack_method.lower() == "pad" else None,
                "pad_positions": args.pad_positions if args.attack_method.lower() == "pad" else None,
                "protect_anchors": bool(args.protect_anchors) if args.attack_method.lower() == "pad" else None,
            }
            for k in ("source", "category", "goal", "refined_goal", "Behavior", "Refined_behavior", "behavior", "target"):
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
