python generate_new.py \
    --model_name "LLaDA-8B-Instruct" \
    --custom_cache_dir "/opt/tiger/sft_entity/models/GSAI-ML__LLaDA-8B-Instruct" \
    --device "cuda:1" \
    --input_path "/opt/tiger/sft_entity/datasets/allenai__wildjailbreak/eval/eval_harmful.tsv" \
    --safety

python generate_new.py \
    --model_name "LLaDA-8B-Instruct" \
    --custom_cache_dir "/opt/tiger/sft_entity/models/GSAI-ML__LLaDA-8B-Instruct" \
    --device "cuda:1" \
    --input_path "/opt/tiger/sft_entity/dllm-safety/gcg.csv" \
    --safety
# /opt/tiger/sft_entity/dllm-safety/deep_inception.csv
# /opt/tiger/sft_entity/dllm-safety/autodan.csv
# /opt/tiger/sft_entity/dllm-safety/gcg.csv

# /opt/tiger/sft_entity/datasets/JailbreakBench__JBB-Behaviors/data/judge-comparison.csv
# /opt/tiger/sft_entity/datasets/allenai__wildjailbreak/eval/eval_harmful.tsv
python generate_new.py \
    --model_name "LLaDA-1.5B" \
    --custom_cache_dir "/opt/tiger/sft_entity/models/GSAI-ML__LLaDA-1.5" \
    --device "cuda:7" \
    --input_path "/opt/tiger/sft_entity/dllm-safety/deep_inception.csv" \
    --safety
/opt/tiger/sft_entity/LLaDA/generation_alpha.py

python generate_new.py \
    --model_name "LLaDA-8B-Instruct" \
    --custom_cache_dir "/opt/tiger/sft_entity/models/GSAI-ML__LLaDA-8B-Instruct" \
    --device "cuda:4" \
    --input_path "/opt/tiger/sft_entity/dllm-safety/gcg.csv"

# /opt/tiger/sft_entity/datasets/JailbreakBench__JBB-Behaviors/data/judge-comparison.csv
# /opt/tiger/sft_entity/datasets/allenai__wildjailbreak/eval/eval_harmful.tsv
python generate_new.py \
    --model_name "LLaDA-1.5B" \
    --custom_cache_dir "/opt/tiger/sft_entity/models/GSAI-ML__LLaDA-1.5" \
    --device "cuda:5" \
    --input_path "/opt/tiger/sft_entity/datasets/allenai__wildjailbreak/eval/eval_harmful.tsv"

python generate_new.py \
    --model_name "LLaDA-1.5B" \
    --custom_cache_dir "/opt/tiger/sft_entity/models/GSAI-ML__LLaDA-1.5" \
    --device "cuda:5" \
    --input_path "/opt/tiger/sft_entity/dllm-safety/gcg.csv"

# --remasking adaptive_step --alpha0 0.3
python /opt/tiger/sft_entity/dllm-safety/LLaDA/generate_alpha.py \
    --model_name "LLaDA-8B-Instruct" \
    --custom_cache_dir "/opt/tiger/sft_entity/models/GSAI-ML__LLaDA-8B-Instruct" \
    --device "cuda:4" \
    --remasking adaptive_step \
    --safety \
    --input_path "/opt/tiger/sft_entity/dllm-safety/autodan.csv"
#   --remasking adaptive_step --alpha0 0.4

python /opt/tiger/sft_entity/dllm-safety/LLaDA/generate_alpha.py \
    --model_name "LLaDA-1.5B" \
    --custom_cache_dir "/opt/tiger/sft_entity/models/GSAI-ML__LLaDA-1.5" \
    --device "cuda:3" \
    --remasking adaptive_step \
    --input_path "/opt/tiger/sft_entity/dllm-safety/autodan.csv"