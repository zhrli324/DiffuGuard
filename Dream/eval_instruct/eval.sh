# # 强制把本地 eval_instruct 放到 sys.path 的最前面
# export PYTHONPATH=/opt/tiger/dllm-safety/Dream/eval_instruct:$PYTHONPATH

# # 最好先切到这个目录，避免 PYTHONPATH=. 指向错地方
# cd /opt/tiger/dllm-safety/Dream/eval_instruct
# python - <<'PY'
# import lm_eval, sys
# print("lm_eval loaded from:", lm_eval.__file__)
# PY
PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
    --model diffllm \
    --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,max_new_tokens=128,diffusion_steps=128,dtype="bfloat16",temperature=0.1,top_p=0.9,alg="entropy" \
    --tasks mmlu_generative \
    --device cuda \
    --batch_size 1 \
    --num_fewshot 4 \
    --output_path output_reproduce/mmlu \
    --log_samples --confirm_run_unsafe_code \
    --apply_chat_template

PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
    --model diffllm \
    --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,max_new_tokens=128,diffusion_steps=128,dtype="bfloat16",temperature=0.1,top_p=0.9,alg="entropy" \
    --tasks mmlu_pro \
    --device cuda \
    --batch_size 1 \
    --num_fewshot 4 \
    --output_path output_reproduce/mmlu_pro \
    --log_samples --confirm_run_unsafe_code \
    --apply_chat_template

PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
    --model diffllm \
    --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,max_new_tokens=128,diffusion_steps=128,dtype="bfloat16",temperature=0.1,top_p=0.9,alg="entropy" \
    --tasks gsm8k \
    --device cuda \
    --batch_size 1 \
    --num_fewshot 0 \
    --output_path output_reproduce/gsm8k \
    --log_samples --confirm_run_unsafe_code \
    --apply_chat_template

PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
    --model diffllm \
    --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,max_new_tokens=512,diffusion_steps=512,dtype="bfloat16",temperature=0.1,top_p=0.9,alg="entropy" \
    --tasks minerva_math \
    --device cuda \
    --batch_size 1 \
    --num_fewshot 0 \
    --output_path output_reproduce/math \
    --log_samples --confirm_run_unsafe_code \
    --apply_chat_template

PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
    --model diffllm \
    --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,dtype="bfloat16" \
    --tasks gpqa_main_n_shot \
    --device cuda \
    --batch_size 1 \
    --num_fewshot 5 \
    --output_path output_reproduce/gpqa \
    --log_samples --confirm_run_unsafe_code \
    --apply_chat_template

####
HF_ALLOW_CODE_EVAL=1 PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
    --model diffllm \
    --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,max_new_tokens=512,diffusion_steps=512,dtype="bfloat16",temperature=0.1,top_p=0.9,alg="entropy" \
    --tasks humaneval \
    --device cuda \
    --batch_size 1 \
    --num_fewshot 0 \
    --output_path output_reproduce/humaneval \
    --log_samples --confirm_run_unsafe_code \
    --apply_chat_template

HF_ALLOW_CODE_EVAL=1 PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
    --model diffllm \
    --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,max_new_tokens=1024,diffusion_steps=1024,dtype="bfloat16",temperature=0.1,top_p=0.9,alg="entropy" \
    --tasks mbpp_instruct \
    --device cuda \
    --batch_size 1 \
    --num_fewshot 0 \
    --output_path output_reproduce/mbpp \
    --log_samples --confirm_run_unsafe_code \
    --apply_chat_template

PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
    --model diffllm \
    --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,max_new_tokens=1280,diffusion_steps=1280,dtype="bfloat16",temperature=0.1,top_p=0.9,alg="entropy" \
    --tasks ifeval \
    --device cuda \
    --batch_size 1 \
    --num_fewshot 0 \
    --output_path output_reproduce/ifeval \
    --log_samples --confirm_run_unsafe_code \
    --apply_chat_template


HF_ALLOW_CODE_EVAL=1 PYTHONPATH=/opt/tiger/dllm-safety/Dream/eval_instruct:$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 --main_process_port 12334 -m lm_eval \
  --model diffllm \
  --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,max_new_tokens=128,diffusion_steps=128,dtype="bfloat16",temperature=0.1,top_p=0.9,alg="entropy",enable_remask=True,remask_mode=adaptive_step,alpha0=0.9,random_rate=0.3,block_length=128,use_chat_template=True \
  --tasks gsm8k \
  --device cuda \
  --batch_size 32 \
  --num_fewshot 0 \
  --output_path output_reproduce/gsm8k \
  --log_samples --confirm_run_unsafe_code \
  --apply_chat_template


diffllm (pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,max_new_tokens=128,diffusion_steps=128,dtype=bfloat16,temperature=0.1,top_p=0.9,alg=entropy,enable_remask=True,remask_mode=adaptive_step,alpha0=0.3,random_rate=0.3,block_length=128,use_chat_template=True), gen_kwargs: (None), limit: None, num_fewshot: 0, batch_size: 32
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     0|exact_match|↑  |0.7635|±  |0.0117|
|     |       |strict-match    |     0|exact_match|↑  |0.2206|±  |0.0114|