#!/usr/bin/env bash
# Table IV: decoding-time mitigators on top of the token sets (K = 128).
set -euo pipefail
cd "$(dirname "$0")/../.."
BASE="configs/default.yaml"
L="configs/models/llava15_7b.yaml"
DIV="method.external_selection_file=outputs/selections/divprune_K128.json"

for E in pope_b chair; do
  if [ "$E" = pope_b ]; then S=run_pope.py; else S=run_chair.py; fi
  # FaithPrune + VCD / + PAI (contrastive term only)
  python scripts/$S --config $BASE $L configs/mitigators/vcd.yaml configs/eval/${E}.yaml \
    --set run_name=faithprune_K128_vcd method.budget=128
  python scripts/$S --config $BASE $L configs/mitigators/pai_faithprune.yaml configs/eval/${E}.yaml \
    --set run_name=faithprune_K128_pai method.budget=128
  # Full model + PAI (amplification + contrast)
  python scripts/$S --config $BASE $L configs/method/full.yaml configs/mitigators/pai_full.yaml configs/eval/${E}.yaml \
    --set run_name=full_pai
  # DivPrune token sets come from the official DivPrune code (exported indices)
  python scripts/$S --config $BASE $L configs/plugin/external_selector_no_vamr.yaml configs/mitigators/vcd.yaml \
    configs/eval/${E}.yaml --set run_name=divprune_K128_vcd method.budget=128 $DIV
  python scripts/$S --config $BASE $L configs/plugin/external_selector_no_vamr.yaml configs/mitigators/pai_full.yaml \
    configs/eval/${E}.yaml --set run_name=divprune_K128_pai method.budget=128 $DIV
done

# CMAC: export the FaithPrune token sets; CMAC itself runs with its official code
python scripts/export_selection.py --config $BASE $L --set method.budget=128 run_name=faithprune \
  --image-list data/chair/chair_500_images.txt --image-dir data/coco/val2014
