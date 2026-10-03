#!/usr/bin/env bash
# Attention statistics (Sec. III, Sec. V-H) and paired bootstrap CIs.
set -euo pipefail
cd "$(dirname "$0")/../.."
python scripts/run_analysis.py --config configs/default.yaml configs/models/llava15_7b.yaml configs/eval/analysis.yaml \
  --set run_name=analysis_llava15_K64 --num-images 200
python scripts/run_bootstrap.py --spec configs/eval/bootstrap.yaml --output outputs/bootstrap_ci.json
