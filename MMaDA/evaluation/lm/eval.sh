# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

task=gsm8k # gsm8k
length=256
block_length=64
num_fewshot=5
model_path=Gen-Verse/MMaDA-8B-MixCoT

# baseline
accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True \
--output_path evals_results/baseline/gsm8k-ns0-${length} \
--log_samples \
