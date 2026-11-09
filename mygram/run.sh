#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

gpu_id='0'
dataset='FBDB15K'
ratio='0.2'
seed='2023'
il_start='50'
bsize='3500'
dropout='0.0'
tau='0.1'
if [[ "$dataset" != *"FB"* ]]; then
    ratio='0.3'
fi

echo "Running with dataset=${dataset}, ratio=${ratio}"
current_datetime=$(date +"%Y-%m-%d-%H-%M")
file_name="${current_datetime}_${dataset}_${ratio}"
echo "$file_name"
mkdir -p logs outputs
CUDA_VISIBLE_DEVICES="$gpu_id" python3 -u -m src.run \
    --config configs/repaired_pmf.json \
    --data-root data/mmkg \
    --dataset "$dataset" \
    --device cuda:0 \
    --output-root outputs \
    --rate "$ratio" \
    --lr .0005 \
    --epochs 1000 \
    --dropout "$dropout" \
    --eval-every 50 \
    --loss-batch-size "$bsize" \
    --il-start "$il_start" \
    --il-step 5 \
    --csls-k 3 \
    --seed "$seed" \
    --nce-temperature "$tau" > "logs/${file_name}.log"
printf '%s\n' "$0" >> "logs/${file_name}.log"
