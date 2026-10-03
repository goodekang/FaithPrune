"""CHAIR on the 500 COCO val2014 images released with the CMAC / OPERA code,
prompt "Please describe this image in detail.", 512-token limit."""
from __future__ import annotations

import os
from pathlib import Path

from tqdm import tqdm

from _common import base_parser, parse, save_run_config

from faithprune.eval.chair import CHAIR, coco_image_id
from faithprune.pipeline import FaithPrune
from faithprune.utils import JsonlAppender, load_image, read_jsonl, read_lines, save_json


def main():
    p = base_parser(__doc__)
    p.add_argument("--seeds", type=int, nargs="*", default=None,
                   help="sampling seeds (supplementary Table S-II); greedy if omitted")
    args, cfg = parse(p)
    ex = cfg.eval.extra
    names = read_lines(ex["image_list"])
    if cfg.eval.limit:
        names = names[: cfg.eval.limit]
    out_dir = Path(args.output or os.path.join(cfg.eval.output_dir, cfg.run_name, "chair"))
    save_run_config(cfg, out_dir)
    fp = FaithPrune(cfg)
    scorer = CHAIR(ex["synonyms_file"], ex["instances_file"], ex["captions_file"])

    seeds = args.seeds if args.seeds else [None]
    all_metrics = {}
    for seed in seeds:
        tag = "greedy" if seed is None else f"seed{seed}"
        writer = JsonlAppender(out_dir / f"captions_{tag}.jsonl", key="image_id")
        for name in tqdm(names, desc=f"CHAIR-{tag}"):
            iid = coco_image_id(name)
            if iid in writer:
                continue
            res = fp.generate(load_image(os.path.join(ex["image_dir"], name)), ex["prompt"],
                              sample_id=name, max_new_tokens=cfg.decoding.max_new_tokens, seed=seed)
            writer.write(dict(image_id=iid, image=name, caption=res["text"],
                              keep=res["selection"].get("keep"), sinks=res["selection"].get("sinks")))
        writer.close()
        m = scorer.compute(read_jsonl(out_dir / f"captions_{tag}.jsonl"))
        save_json(out_dir / f"per_sample_{tag}.json", m.pop("per_sample"))
        all_metrics[tag] = m
        print(tag, m)
    save_json(out_dir / "metrics.json", all_metrics)


if __name__ == "__main__":
    main()
