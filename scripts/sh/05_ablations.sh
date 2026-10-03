#!/usr/bin/env bash
# Table VIII (component analysis and design choices), Table IX (VAMR as a
# plug-in) and the hyper-parameter study, LLaVA-1.5-7B, K = 64.
set -euo pipefail
cd "$(dirname "$0")/../.."
BASE="configs/default.yaml"
L="configs/models/llava15_7b.yaml"
T_LLAVA="gqa,mmbench_en_dev,mme,pope,scienceqa_img,vqav2_val,textvqa_val,seedbench"

eval_all () {  # $1 run_name, $2 config file, $3 extra overrides
  local ov="run_name=$1 ${3:-}"
  local lm="${ov// /|}"
  lm="${lm//=/:}"
  python scripts/run_lmms_eval.py --model_args "config=$BASE;$L;$2,set=$lm" --tasks "$T_LLAVA" \
    --batch_size 1 --log_samples --output_path "outputs/lmms_eval/$1"
  python scripts/run_pope.py  --config $BASE $L $2 configs/eval/pope_b.yaml --set $ov
  python scripts/run_chair.py --config $BASE $L $2 configs/eval/chair.yaml  --set $ov
}

# variants whose token set or target changes need their own gamma_l
for A in a5_position_only_kernel a7_sinks_kept a8_target_with_sink a9_random_vamr; do
  python scripts/calibrate.py --config $BASE $L configs/ablation/${A}.yaml
done
for A in a1_relevance_only a2_rel_sink a3_rel_sink_align a4_sr2s_no_vamr a5_position_only_kernel \
         a6_global_gamma a7_sinks_kept a8_target_with_sink a9_random_vamr; do
  eval_all abl_${A} configs/ablation/${A}.yaml
done

# Table IX: plug-in. Indices of VisionZip / DivPrune / HoloV come from their official code.
for SEL in visionzip divprune holov; do
  OV="method.external_selection_file=outputs/selections/${SEL}_K64.json method.vamr.gamma_file=outputs/calibration/llava-1.5-7b_K64_${SEL}.json"
  python scripts/calibrate.py --config $BASE $L configs/plugin/external_selector.yaml --set $OV
  eval_all ${SEL}_no_vamr configs/plugin/external_selector_no_vamr.yaml "$OV"
  eval_all ${SEL}_vamr configs/plugin/external_selector.yaml "$OV"
done
python scripts/calibrate.py --config $BASE $L configs/plugin/full_sinks_removed.yaml
eval_all full_sinks_removed configs/plugin/full_sinks_removed.yaml
eval_all full_with_gamma_k64 configs/plugin/full_with_gamma_k64.yaml

# Hyper-parameters on the held-out split (Sec. V-G)
for B in 0.2 0.3 0.4 0.5; do
  python scripts/run_chair.py --config $BASE $L configs/eval/chair_heldout.yaml --set run_name=hp_beta$B method.beta=$B
done
for LAM in 0.25 0.5 0.75 1.0; do
  python scripts/run_chair.py --config $BASE $L configs/eval/chair_heldout.yaml --set run_name=hp_lam$LAM method.lam=$LAM
done
for TAU in 2.5 3.0 3.5 4.0; do
  python scripts/run_chair.py --config $BASE $L configs/eval/chair_heldout.yaml --set run_name=hp_tau$TAU method.tau=$TAU
done
for LS in 1 2 3 4; do
  python scripts/calibrate.py --config $BASE $L --set method.scoring_layer=$LS
  python scripts/run_chair.py --config $BASE $L configs/eval/chair_heldout.yaml --set run_name=hp_ls$LS method.scoring_layer=$LS
done
python scripts/calibrate.py --config $BASE $L configs/calibration/n_cal_32.yaml
