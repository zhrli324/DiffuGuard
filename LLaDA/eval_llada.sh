pip install transformers==4.49.0 lm_eval==0.4.8 accelerate==0.34.2
pip install antlr4-python3-runtime==4.11 math_verify sympy hf_xet


# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true


# conditional likelihood estimation benchmarks
accelerate launch eval_llada.py --tasks gpqa_main_n_shot --num_fewshot 5 --model llada_dist --batch_size 8 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.5,is_check_greedy=False,mc_num=128

accelerate launch eval_llada.py --tasks truthfulqa_mc2 --num_fewshot 0 --model llada_dist --batch_size 8 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=2.0,is_check_greedy=False,mc_num=128

accelerate launch eval_llada.py --tasks arc_challenge --num_fewshot 0 --model llada_dist --batch_size 8 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.5,is_check_greedy=False,mc_num=128

accelerate launch eval_llada.py --tasks hellaswag --num_fewshot 0 --model llada_dist --batch_size 8 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.5,is_check_greedy=False,mc_num=128

accelerate launch eval_llada.py --tasks winogrande --num_fewshot 5 --model llada_dist --batch_size 8 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False,mc_num=128

accelerate launch eval_llada.py --tasks piqa --num_fewshot 0 --model llada_dist --batch_size 8 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.5,is_check_greedy=False,mc_num=128

accelerate launch eval_llada.py --tasks mmlu --num_fewshot 5 --model llada_dist --batch_size 1 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False,mc_num=1

accelerate launch eval_llada.py --tasks cmmlu --num_fewshot 5 --model llada_dist --batch_size 1 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False,mc_num=1

accelerate launch eval_llada.py --tasks ceval-valid --num_fewshot 5 --model llada_dist --batch_size 1 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False,mc_num=1


# conditional generation benchmarks
accelerate launch eval_llada.py --tasks bbh --model llada_dist --model_args model_path='GSAI-ML/LLaDA-8B-Base',gen_length=1024,steps=1024,block_length=1024

accelerate launch eval_llada.py --tasks gsm8k --model llada_dist --model_args model_path='GSAI-ML/LLaDA-8B-Base',gen_length=1024,steps=1024,block_length=1024

accelerate launch eval_llada.py --tasks minerva_math --model llada_dist --model_args model_path='GSAI-ML/LLaDA-8B-Base',gen_length=1024,steps=1024,block_length=1024

accelerate launch eval_llada.py --tasks humaneval --model llada_dist --confirm_run_unsafe_code --model_args model_path='GSAI-ML/LLaDA-8B-Base',gen_length=1024,steps=1024,block_length=1024

accelerate launch eval_llada.py --tasks mbpp --model llada_dist --confirm_run_unsafe_code --model_args model_path='GSAI-ML/LLaDA-8B-Base',gen_length=1024,steps=1024,block_length=1024

unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=0
HF_ALLOW_CODE_EVAL=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
accelerate launch \
  --num_processes=4 \
  --mixed_precision=bf16 \
  --main_process_port=24599 \
  eval_llada.py \
  --tasks gsm8k --num_fewshot 0 --batch_size 32 \
  --model llada_dist \
  --model_args model_path='/opt/tiger/sft_entity/models/GSAI-ML__LLaDA-8B-Instruct',gen_length=128,steps=128,block_length=32,remasking=adaptive_step,cfg_scale=0,trust_remote_code=True

unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=0
HF_ALLOW_CODE_EVAL=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
accelerate launch \
  --num_processes=4 \
  --mixed_precision=bf16 \
  --main_process_port=24599 \
  eval_llada.py \
  --tasks mmlu --num_fewshot 0 --batch_size 32 \
  --confirm_run_unsafe_code \
  --model llada_dist \
  --model_args model_path='/opt/tiger/sft_entity/models/GSAI-ML__LLaDA-8B-Instruct',gen_length=512,steps=512,block_length=64,remasking=low_confidence,cfg_scale=0,trust_remote_code=True

llada_dist (model_path=/opt/tiger/sft_entity/models/GSAI-ML__LLaDA-8B-Instruct,gen_length=128,steps=128,block_length=32,remasking=adaptive_step,cfg_scale=0,trust_remote_code=True), gen_kwargs: (None), limit: None, num_fewshot: 0, batch_size: 32
n214-176-214:114596:118174 [3] NCCL INFO comm 0xf5a0d10 rank 3 nranks 4 cudaDev 3 busId 4e000 - Abort COMPLETE
n214-176-214:114595:118172 [2] NCCL INFO comm 0xe48c0d0 rank 2 nranks 4 cudaDev 2 busId 4a000 - Abort COMPLETE
|  Tasks  |Version|  Filter   |n-shot|Metric|   |Value |   |Stderr|
|---------|------:|-----------|-----:|------|---|-----:|---|-----:|
|humaneval|      1|create_test|     0|pass@1|   |0.3171|±  |0.0364|

# 1) 确保不离线
unset HF_HUB_OFFLINE
export HF_DATASETS_OFFLINE=0

# 2) 固定缓存路径（可写）
export HF_HOME=/opt/tiger/hf
export HF_DATASETS_CACHE=/opt/tiger/hf/datasets

# 3) 打开 datasets 的调试日志，方便看到失败的 URL
export HF_DATASETS_VERBOSITY=debug

# 4) 用 Python 脚本逐个配置下载，并显式信任远程代码
python - <<'PY'
from datasets import get_dataset_config_names, load_dataset
names = get_dataset_config_names("hails/mmlu_no_train")
print("found", len(names), "configs")
ok, fail = [], []
for name in names:
    try:
        print(f"==> downloading config: {name}")
        ds = load_dataset("hails/mmlu_no_train", name, trust_remote_code=True)
        # 访问一下长度，触发实际构建
        for split in ds:
            print(f"[cached] {name}/{split}: {len(ds[split])}")
        ok.append(name)
    except Exception as e:
        print(f"[FAILED] {name}: {e}")
        fail.append((name, repr(e)))
print("OK:", ok)
print("FAIL:", fail)
PY

unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=0
HF_ALLOW_CODE_EVAL=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
accelerate launch \
  --num_processes=4 \
  --mixed_precision=bf16 \
  --main_process_port=24599 \
  eval_llada.py \
  --tasks mmlu --num_fewshot 0 --batch_size 32 \
  --confirm_run_unsafe_code \
  --model llada_dist \
  --model_args model_path='/opt/tiger/sft_entity/models/GSAI-ML__LLaDA-8B-Instruct',gen_length=128,steps=128,block_length=32,remasking=low_confidence,cfg_scale=0,trust_remote_code=True

llada_dist (model_path=/opt/tiger/sft_entity/models/GSAI-ML__LLaDA-8B-Instruct,gen_length=128,steps=128,block_length=32,remasking=low_confidence,cfg_scale=0,trust_remote_code=True), gen_kwargs: (None), limit: None, num_fewshot: 0, batch_size: 32
n214-176-214:123544:216479 [3] NCCL INFO comm 0xf616920 rank 3 nranks 4 cudaDev 3 busId 4e000 - Abort COMPLETE
n214-176-214:123543:216481 [2] NCCL INFO comm 0xf9bc1c0 rank 2 nranks 4 cudaDev 2 busId 4a000 - Abort COMPLETE
n214-176-214:123542:216477 [1] NCCL INFO comm 0xf6cb900 rank 1 nranks 4 cudaDev 1 busId 16000 - Abort COMPLETE
|                 Tasks                 |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|---------------------------------------|------:|------|-----:|------|---|-----:|---|-----:|
|mmlu                                   |      2|none  |      |acc   |↑  |0.6367|±  |0.0039|
| - humanities                          |      2|none  |      |acc   |↑  |0.5826|±  |0.0069|
|  - formal_logic                       |      1|none  |     0|acc   |↑  |0.5397|±  |0.0446|
|  - high_school_european_history       |      1|none  |     0|acc   |↑  |0.7818|±  |0.0323|
|  - high_school_us_history             |      1|none  |     0|acc   |↑  |0.8137|±  |0.0273|
|  - high_school_world_history          |      1|none  |     0|acc   |↑  |0.8101|±  |0.0255|
|  - international_law                  |      1|none  |     0|acc   |↑  |0.7934|±  |0.0370|
|  - jurisprudence                      |      1|none  |     0|acc   |↑  |0.7500|±  |0.0419|
|  - logical_fallacies                  |      1|none  |     0|acc   |↑  |0.7485|±  |0.0341|
|  - moral_disputes                     |      1|none  |     0|acc   |↑  |0.6908|±  |0.0249|
|  - moral_scenarios                    |      1|none  |     0|acc   |↑  |0.4458|±  |0.0166|
|  - philosophy                         |      1|none  |     0|acc   |↑  |0.6334|±  |0.0274|
|  - prehistory                         |      1|none  |     0|acc   |↑  |0.6914|±  |0.0257|
|  - professional_law                   |      1|none  |     0|acc   |↑  |0.4524|±  |0.0127|
|  - world_religions                    |      1|none  |     0|acc   |↑  |0.7836|±  |0.0316|
| - other                               |      2|none  |      |acc   |↑  |0.6724|±  |0.0082|
|  - business_ethics                    |      1|none  |     0|acc   |↑  |0.7500|±  |0.0435|
|  - clinical_knowledge                 |      1|none  |     0|acc   |↑  |0.6943|±  |0.0284|
|  - college_medicine                   |      1|none  |     0|acc   |↑  |0.6358|±  |0.0367|
|  - global_facts                       |      1|none  |     0|acc   |↑  |0.3900|±  |0.0490|
|  - human_aging                        |      1|none  |     0|acc   |↑  |0.6637|±  |0.0317|
|  - management                         |      1|none  |     0|acc   |↑  |0.7767|±  |0.0412|
|  - marketing                          |      1|none  |     0|acc   |↑  |0.9017|±  |0.0195|
|  - medical_genetics                   |      1|none  |     0|acc   |↑  |0.6400|±  |0.0482|
|  - miscellaneous                      |      1|none  |     0|acc   |↑  |0.7395|±  |0.0157|
|  - nutrition                          |      1|none  |     0|acc   |↑  |0.7092|±  |0.0260|
|  - professional_accounting            |      1|none  |     0|acc   |↑  |0.4894|±  |0.0298|
|  - professional_medicine              |      1|none  |     0|acc   |↑  |0.5919|±  |0.0299|
|  - virology                           |      1|none  |     0|acc   |↑  |0.5000|±  |0.0389|
| - social sciences                     |      2|none  |      |acc   |↑  |0.7189|±  |0.0079|
|  - econometrics                       |      1|none  |     0|acc   |↑  |0.4386|±  |0.0467|
|  - high_school_geography              |      1|none  |     0|acc   |↑  |0.7980|±  |0.0286|
|  - high_school_government_and_politics|      1|none  |     0|acc   |↑  |0.8653|±  |0.0246|
|  - high_school_macroeconomics         |      1|none  |     0|acc   |↑  |0.6692|±  |0.0239|
|  - high_school_microeconomics         |      1|none  |     0|acc   |↑  |0.7185|±  |0.0292|
|  - high_school_psychology             |      1|none  |     0|acc   |↑  |0.8404|±  |0.0157|
|  - human_sexuality                    |      1|none  |     0|acc   |↑  |0.6641|±  |0.0414|
|  - professional_psychology            |      1|none  |     0|acc   |↑  |0.6127|±  |0.0197|
|  - public_relations                   |      1|none  |     0|acc   |↑  |0.6909|±  |0.0443|
|  - security_studies                   |      1|none  |     0|acc   |↑  |0.6939|±  |0.0295|
|  - sociology                          |      1|none  |     0|acc   |↑  |0.7861|±  |0.0290|
|  - us_foreign_policy                  |      1|none  |     0|acc   |↑  |0.8100|±  |0.0394|
| - stem                                |      2|none  |      |acc   |↑  |0.6023|±  |0.0084|
|  - abstract_algebra                   |      1|none  |     0|acc   |↑  |0.4100|±  |0.0494|
|  - anatomy                            |      1|none  |     0|acc   |↑  |0.5630|±  |0.0428|
|  - astronomy                          |      1|none  |     0|acc   |↑  |0.7171|±  |0.0367|
|  - college_biology                    |      1|none  |     0|acc   |↑  |0.7361|±  |0.0369|
|  - college_chemistry                  |      1|none  |     0|acc   |↑  |0.4600|±  |0.0501|
|  - college_computer_science           |      1|none  |     0|acc   |↑  |0.5200|±  |0.0502|
|  - college_mathematics                |      1|none  |     0|acc   |↑  |0.4100|±  |0.0494|
|  - college_physics                    |      1|none  |     0|acc   |↑  |0.4118|±  |0.0490|
|  - computer_security                  |      1|none  |     0|acc   |↑  |0.7500|±  |0.0435|
|  - conceptual_physics                 |      1|none  |     0|acc   |↑  |0.6170|±  |0.0318|
|  - electrical_engineering             |      1|none  |     0|acc   |↑  |0.6276|±  |0.0403|
|  - elementary_mathematics             |      1|none  |     0|acc   |↑  |0.7302|±  |0.0229|
|  - high_school_biology                |      1|none  |     0|acc   |↑  |0.8129|±  |0.0222|
|  - high_school_chemistry              |      1|none  |     0|acc   |↑  |0.5369|±  |0.0351|
|  - high_school_computer_science       |      1|none  |     0|acc   |↑  |0.7700|±  |0.0423|
|  - high_school_mathematics            |      1|none  |     0|acc   |↑  |0.4370|±  |0.0302|
|  - high_school_physics                |      1|none  |     0|acc   |↑  |0.4305|±  |0.0404|
|  - high_school_statistics             |      1|none  |     0|acc   |↑  |0.5787|±  |0.0337|
|  - machine_learning                   |      1|none  |     0|acc   |↑  |0.4732|±  |0.0474|

|      Groups      |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|------------------|------:|------|------|------|---|-----:|---|-----:|
|mmlu              |      2|none  |      |acc   |↑  |0.6367|±  |0.0039|
| - humanities     |      2|none  |      |acc   |↑  |0.5826|±  |0.0069|
| - other          |      2|none  |      |acc   |↑  |0.6724|±  |0.0082|
| - social sciences|      2|none  |      |acc   |↑  |0.7189|±  |0.0079|
| - stem           |      2|none  |      |acc   |↑  |0.6023|±  |0.0084|
