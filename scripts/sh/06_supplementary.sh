#!/usr/bin/env bash
# Supplementary experiments: calibration variants, sampling decoding, budget
# sweep (CHAIR_i), hallucination beyond COCO, controls.
set -euo pipefail
cd "$(dirname "$0")/../.."
BASE="configs/default.yaml"
L="configs/models/llava15_7b.yaml"

# calibration variants
for V in mean_matching gqa_prompts non_sequential visual_genome text_rows_only headwise; do
  python scripts/calibrate.py --config $BASE $L configs/calibration/${V}.yaml
  python scripts/run_pope.py  --config $BASE $L configs/calibration/${V}.yaml configs/eval/pope_b.yaml
  python scripts/run_chair.py --config $BASE $L configs/calibration/${V}.yaml configs/eval/chair.yaml
done

# multinomial sampling, three seeds
for K in 128 64; do
  python scripts/run_chair.py --config $BASE $L configs/eval/chair.yaml configs/eval/sampling.yaml \
    --set run_name=sample_faithprune_K$K method.budget=$K --seeds 0 1 2
  for S in 0 1 2; do
    python scripts/run_pope.py --config $BASE $L configs/eval/pope_b.yaml configs/eval/sampling.yaml \
      --set run_name=sample_faithprune_K${K}_s$S method.budget=$K decoding.seed=$S
  done
done
python scripts/run_chair.py --config $BASE $L configs/method/full.yaml configs/eval/chair.yaml configs/eval/sampling.yaml \
  --set run_name=sample_full --seeds 0 1 2
for S in 0 1 2; do
  python scripts/run_pope.py --config $BASE $L configs/method/full.yaml configs/eval/pope_b.yaml configs/eval/sampling.yaml \
    --set run_name=sample_full_s$S decoding.seed=$S
done

# CHAIR_i under the budget sweep
for K in 288 192 128 96 64 32; do
  python scripts/run_chair.py --config $BASE $L configs/eval/chair.yaml --set run_name=faithprune_llava15_K$K method.budget=$K
done

# beyond COCO (gamma_l unchanged); MMHal / Object HalBench responses are
# scored with their official scripts
R="run_name=faithprune_llava15_K64"
python scripts/run_amber.py --config $BASE $L configs/eval/amber_discriminative.yaml --set $R
python scripts/run_pope.py  --config $BASE $L configs/eval/pope_aokvqa.yaml --set ${R}_aokvqa
python scripts/run_pope.py  --config $BASE $L configs/eval/pope_gqa.yaml    --set ${R}_gqa
python scripts/run_hallusionbench.py --config $BASE $L configs/eval/hallusionbench.yaml --set $R
python scripts/run_generation.py --config $BASE $L --questions data/mmhal/mmhal_questions.jsonl \
  --image-dir data/mmhal/images --set $R
python scripts/run_generation.py --config $BASE $L --questions data/objhal/objhal_questions.jsonl \
  --image-dir data/coco/val2014 --set $R

# controls
for C in uniform_grid_vamr avgpool_vamr; do
  python scripts/calibrate.py --config $BASE $L configs/controls/${C}.yaml
done
for C in uniform_grid uniform_grid_vamr avgpool avgpool_vamr sr2s_amplify_only; do
  python scripts/run_pope.py  --config $BASE $L configs/controls/${C}.yaml configs/eval/pope_b.yaml
  python scripts/run_chair.py --config $BASE $L configs/controls/${C}.yaml configs/eval/chair.yaml
done
DIV="method.external_selection_file=outputs/selections/divprune_K64.json"
python scripts/run_pope.py  --config $BASE $L configs/controls/divprune_amplify_only.yaml configs/eval/pope_b.yaml --set $DIV
python scripts/run_chair.py --config $BASE $L configs/controls/divprune_amplify_only.yaml configs/eval/chair.yaml  --set $DIV
