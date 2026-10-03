#!/usr/bin/env bash
# Protocol B POPE, CHAIR and AMBER (Table II), HallusionBench (Table VI) and
# CHAIR on LLaVA-NeXT (Table VII).
set -euo pipefail
cd "$(dirname "$0")/../.."
BASE="configs/default.yaml"
L="configs/models/llava15_7b.yaml"

hallu () {  # $1 run_name, $2 overrides, remaining: extra config files
  local name=$1 ov=$2
  shift 2
  python scripts/run_pope.py  --config $BASE $L "$@" configs/eval/pope_b.yaml           --set run_name=$name $ov
  python scripts/run_chair.py --config $BASE $L "$@" configs/eval/chair.yaml            --set run_name=$name $ov
  python scripts/run_amber.py --config $BASE $L "$@" configs/eval/amber_generative.yaml --set run_name=$name $ov
}

hallu full "" configs/method/full.yaml
for K in 128 64; do
  hallu faithprune_llava15_K${K} "method.budget=$K"
done
hallu abl_sr2s_no_vamr "method.budget=64" configs/ablation/a4_sr2s_no_vamr.yaml
python scripts/calibrate.py --config $BASE $L configs/ablation/a10_without_sr2s.yaml
hallu abl_without_sr2s "method.budget=64" configs/ablation/a10_without_sr2s.yaml

# HallusionBench and CHAIR on Qwen2.5-VL
Q="configs/models/qwen25vl_7b.yaml"
python scripts/run_hallusionbench.py --config $BASE $Q configs/method/full.yaml configs/eval/hallusionbench.yaml --set run_name=full_qwen25vl
python scripts/run_chair.py --config $BASE $Q configs/method/full.yaml configs/eval/chair.yaml --set run_name=full_qwen25vl
for K in 192 64; do
  python scripts/run_hallusionbench.py --config $BASE $Q configs/eval/hallusionbench.yaml \
    --set run_name=faithprune_qwen25vl_K${K} method.budget=$K
done

# CHAIR on LLaVA-NeXT
N="configs/models/llava_next_7b.yaml"
python scripts/run_chair.py --config $BASE $N configs/method/full.yaml configs/eval/chair.yaml --set run_name=full_llava_next
python scripts/run_chair.py --config $BASE $N configs/eval/chair.yaml --set run_name=faithprune_llava_next
