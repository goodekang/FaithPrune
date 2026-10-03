#!/usr/bin/env bash
# Table V: efficiency on LLaVA-1.5-7B (A100, FP16, batch size 1).
set -euo pipefail
cd "$(dirname "$0")/../.."
BASE="configs/default.yaml"
L="configs/models/llava15_7b.yaml"

python scripts/run_efficiency.py --config $BASE $L configs/method/full.yaml configs/eval/efficiency.yaml --set run_name=full
for K in 192 128 64; do
  python scripts/run_efficiency.py --config $BASE $L configs/eval/efficiency.yaml \
    --set run_name=faithprune_K$K method.budget=$K
  python scripts/run_efficiency.py --config $BASE $L configs/ablation/a1_relevance_only.yaml configs/eval/efficiency.yaml \
    --set run_name=fastv_K$K method.budget=$K
  # FlashAttention-2 with the folded VAMR bias
  python scripts/run_efficiency.py --config $BASE $L configs/method/faithprune_folded.yaml configs/eval/efficiency.yaml \
    --set run_name=faithprune_folded_K$K method.budget=$K
done
