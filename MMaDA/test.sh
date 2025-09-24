python generate_new.py \
    --model_name "MMaDA-8B" \
    --custom_cache_dir "/opt/tiger/sft_entity/models/Gen-Verse__MMaDA-8B-MixCoT" \
    --device "cuda:3" \
    --input_path "/opt/tiger/sft_entity/dllm-safety/gcg.csv" \
    --safety

python /opt/tiger/sft_entity/MMaDA/generation.py \
    --model_path "Gen-Verse/MMaDA-8B-MixCoT" \
    --steps 64 --gen_length 128 --block_length 128 \
    --device "cuda:7" \
    --input_path "/opt/tiger/sft_entity/dllm-safety/gcg.csv"

python /opt/tiger/dllm-safety/MMaDA/generation.py \
  --model_path "Gen-Verse/MMaDA-8B-MixCoT" \
  --steps 64 --gen_length 128 --block_length 128 \
  --input_path "/opt/tiger/sft_entity/datasets/allenai__wildjailbreak/eval/eval_harmful.tsv" \
  --temperature 0.5 \
  --safety

  # /opt/tiger/sft_entity/datasets/JailbreakBench__JBB-Behaviors/data/judge-comparison.csv
# /opt/tiger/sft_entity/datasets/allenai__wildjailbreak/eval/eval_harmful.tsv