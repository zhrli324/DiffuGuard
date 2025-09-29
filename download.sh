#!/usr/bin/env bash
set -euo pipefail

git clone https://github.com/ML-GSAI/LLaDA
git clone https://github.com/EleutherAI/lm-evaluation-harness
pip install lm_eval
pip install -U 'transformers>=4.52.3'
git clone https://github.com/DreamLM/Dream
git clone https://github.com/ZichenWen1/DIJA

ROOT="/opt/tiger/sft_entity"
HF_TOKEN=""
USE_SYMLINKS="True"   # Change to "False" to copy real files instead of using symlinks (uses more disk)

MODELS=(
  "GSAI-ML/LLaDA-8B-Instruct"
  "Dream-org/Dream-v0-Instruct-7B"
  "GSAI-ML/LLaDA-1.5"
  "Gen-Verse/MMaDA-8B-MixCoT"
)

DATASETS=(
  "allenai/wildjailbreak"
  "JailbreakBench/JBB-Behaviors"
  "walledai/AdvBench"
)

# === Prepare directories ===
mkdir -p "${ROOT}/models" "${ROOT}/datasets"

echo "===> Root directory: ${ROOT}"
echo "===> Use symlinks: ${USE_SYMLINKS}  (False=copy real files)"

echo "==== Downloading models ===="
for rid in "${MODELS[@]}"; do
  safe_name="${rid//\//__}"
  out_dir="${ROOT}/models/${safe_name}"
  mkdir -p "${out_dir}"

  echo ">>> Model: ${rid}"
  huggingface-cli download "${rid}" \
    --repo-type model \
    --token "${HF_TOKEN}" \
    --local-dir "${out_dir}" \
    --local-dir-use-symlinks "${USE_SYMLINKS}" \
    --resume
  echo "✔ Saved to: ${out_dir}"
  echo
done

echo "==== Downloading datasets ===="
for rid in "${DATASETS[@]}"; do
  safe_name="${rid//\//__}"
  out_dir="${ROOT}/datasets/${safe_name}"
  mkdir -p "${out_dir}"

  echo ">>> Dataset: ${rid}"
  huggingface-cli download "${rid}" \
    --repo-type dataset \
    --token "${HF_TOKEN}" \
    --local-dir "${out_dir}" \
    --local-dir-use-symlinks "${USE_SYMLINKS}" \
    --resume
  echo "✔ Saved to: ${out_dir}"
  echo
done

echo "==== Convert AdvBench parquet → csv ===="
PARQUET_FILE="${ROOT}/datasets/walledai__AdvBench/data/train-00000-of-00001.parquet"
CSV_FILE="${ROOT}/datasets/walledai__AdvBench/data/train.csv"

if [ -f "$PARQUET_FILE" ]; then
  # Pass paths via env vars and use a SINGLE-QUOTED heredoc to avoid shell expansion
  PARQUET_FILE="$PARQUET_FILE" CSV_FILE="$CSV_FILE" python3 - <<'PYCODE'
import os
import pandas as pd

parquet_path = os.environ["PARQUET_FILE"]
csv_path = os.environ["CSV_FILE"]

print(f"Converting {parquet_path} → {csv_path}")
df = pd.read_parquet(parquet_path)  # Requires 'pyarrow' or 'fastparquet'
df.to_csv(csv_path, index=False)
print("✔ Conversion finished")
PYCODE
else
  echo "⚠️ ${PARQUET_FILE} not found, skipping conversion"
fi

echo "==== All done ===="
echo "Models directory:   ${ROOT}/models"
echo "Datasets directory: ${ROOT}/datasets"
