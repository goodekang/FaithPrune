#!/usr/bin/env bash
# Splits (calibration / held-out) and the C = 256 k-means centroids of the
# vocabulary embeddings for every model.
set -euo pipefail
cd "$(dirname "$0")/../.."

python scripts/make_splits.py --seed 0
for M in llava15_7b llava_next_7b qwen25vl_7b; do
  python scripts/build_centroids.py --config configs/default.yaml configs/models/${M}.yaml
done
