#!/usr/bin/env bash
# Layer-sequential calibration of gamma_l, one table per model and budget.
set -euo pipefail
cd "$(dirname "$0")/../.."
BASE="configs/default.yaml"

# LLaVA-1.5-7B: main budgets and the supplementary budget sweep
for K in 288 192 128 96 64 32; do
  python scripts/calibrate.py --config $BASE configs/models/llava15_7b.yaml --set method.budget=$K
done
# Qwen2.5-VL-7B (576 full tokens): rho = 33.3% and 11.1%
for K in 192 64; do
  python scripts/calibrate.py --config $BASE configs/models/qwen25vl_7b.yaml --set method.budget=$K
done
# LLaVA-NeXT-7B: calibrated per retention ratio
python scripts/calibrate.py --config $BASE configs/models/llava_next_7b.yaml
