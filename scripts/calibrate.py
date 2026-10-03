"""Layer-sequential calibration of gamma_l (Algorithm S2).

One full and one pruned teacher-forced pass over N_cal = 64 images with
generic captioning prompts; the result is written as a JSON table that
``method.vamr.gamma_file`` points to.
"""
from __future__ import annotations

import os
from pathlib import Path

from _common import base_parser, parse, save_run_config

from faithprune.calibration import Calibrator
from faithprune.core.alignment import load_centroids
from faithprune.modeling.adapters import build_adapter
from faithprune.pipeline import default_centroids_path, default_gamma_path, set_seed
from faithprune.utils import load_image, read_jsonl, read_lines


def main():
    args, cfg = parse(base_parser(__doc__))
    set_seed(cfg.decoding.seed)
    cal = cfg.calibration
    prompts = None
    if cal.prompt_file:
        rows = read_jsonl(cal.prompt_file)[: cal.num_images]
        names = [r["image"] for r in rows]
        prompts = [r["question"] for r in rows]
    else:
        names = read_lines(cal.image_list)[: cal.num_images]
    images = [load_image(os.path.join(cal.image_dir, n)) for n in names]

    adapter = build_adapter(cfg.model)
    centroids = None
    if cfg.method.selector == "sr2s" and cfg.method.use_alignment:
        centroids = load_centroids(cfg.method.centroids_file or default_centroids_path(cfg), adapter.device)

    table = Calibrator(adapter, cfg, centroids).calibrate(images, names, prompts)
    out = Path(args.output or cal.output or cfg.method.vamr.gamma_file or default_gamma_path(cfg))
    table.meta["images"] = names
    table.meta["prompts"] = prompts or cal.prompts
    table.save(out)
    save_run_config(cfg, out.parent / (out.stem + "_run"))
    print(f"saved gamma_l to {out}")


if __name__ == "__main__":
    main()
