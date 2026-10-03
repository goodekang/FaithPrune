#!/usr/bin/env bash
# Protocol A (lmms-eval, following RESTORE): Tables I, VI and VII.
set -euo pipefail
cd "$(dirname "$0")/../.."
BASE="configs/default.yaml"
T_LLAVA="gqa,mmbench_en_dev,mme,pope,scienceqa_img,vqav2_val,textvqa_val,seedbench"
T_QWEN="mmbench_en_dev,mme,pope,scienceqa_img,textvqa_val"
T_NEXT="gqa,mmbench_en_dev,pope,textvqa_val"

run () {  # $1 tag, $2 ';'-separated configs, $3 '|'-separated key:value overrides, $4 tasks
  python scripts/run_lmms_eval.py --model_args "config=$2,set=$3" --tasks "$4" \
    --batch_size 1 --log_samples --output_path "outputs/lmms_eval/$1"
}

run full_llava15 "$BASE;configs/models/llava15_7b.yaml;configs/method/full.yaml" "" "$T_LLAVA"
for K in 192 128 64; do
  run faithprune_K${K} "$BASE;configs/models/llava15_7b.yaml" "method.budget:$K" "$T_LLAVA"
done
# budget sweep (supplementary, average accuracy)
for K in 288 96 32; do
  run faithprune_K${K} "$BASE;configs/models/llava15_7b.yaml" "method.budget:$K" "$T_LLAVA"
done

run full_qwen25vl "$BASE;configs/models/qwen25vl_7b.yaml;configs/method/full.yaml" "" "$T_QWEN"
for K in 192 64; do
  run faithprune_qwen25vl_K${K} "$BASE;configs/models/qwen25vl_7b.yaml" "method.budget:$K" "$T_QWEN"
done

run full_llava_next "$BASE;configs/models/llava_next_7b.yaml;configs/method/full.yaml" "" "$T_NEXT"
run faithprune_llava_next "$BASE;configs/models/llava_next_7b.yaml" "" "$T_NEXT"
