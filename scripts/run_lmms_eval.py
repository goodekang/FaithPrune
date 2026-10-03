"""Launch lmms-eval with the FaithPrune model wrapper (Protocol A).

Example:
    python scripts/run_lmms_eval.py \
        --model_args "config=configs/default.yaml;configs/models/llava15_7b.yaml,set=method.budget:64" \
        --tasks gqa,mmbench_en_dev,mme,pope,scienceqa_img,vqav2_val,textvqa_val,seedbench \
        --batch_size 1 --log_samples --output_path outputs/lmms_eval/llava15_K64
All arguments are forwarded to lmms-eval; ``--model faithprune`` is added.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import faithprune.lmms_eval_plugin  # noqa: E402,F401  (registers the model)


def main():
    from lmms_eval.__main__ import cli_evaluate

    if "--model" not in sys.argv:
        sys.argv[1:1] = ["--model", "faithprune"]
    cli_evaluate()


if __name__ == "__main__":
    main()
