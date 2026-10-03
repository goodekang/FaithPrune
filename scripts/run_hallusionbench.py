"""HallusionBench question-level accuracy (aAcc)."""
from __future__ import annotations

import os
from pathlib import Path

from tqdm import tqdm

from _common import base_parser, parse, save_run_config

from faithprune.eval.hallusionbench import a_acc, image_path
from faithprune.pipeline import FaithPrune
from faithprune.utils import JsonlAppender, load_image, load_json, read_jsonl, save_json


def main():
    args, cfg = parse(base_parser(__doc__))
    ex = cfg.eval.extra
    items = load_json(ex["question_file"])
    if cfg.eval.limit:
        items = items[: cfg.eval.limit]
    out_dir = Path(args.output or os.path.join(cfg.eval.output_dir, cfg.run_name, "hallusionbench"))
    save_run_config(cfg, out_dir)
    fp = FaithPrune(cfg)

    writer = JsonlAppender(out_dir / "responses.jsonl", key="uid")
    for n, it in enumerate(tqdm(items, desc="HallusionBench")):
        uid = f"{it['category']}/{it['subcategory']}/{it['set_id']}/{it['figure_id']}/{it['question_id']}"
        if uid in writer:
            continue
        path = image_path(ex["image_root"], it)
        image = load_image(path) if path else None
        res = fp.generate(image, it["question"], sample_id=uid, max_new_tokens=cfg.decoding.max_new_tokens)
        writer.write(dict(uid=uid, response=res["text"], gt_answer=it["gt_answer"],
                          category=it["category"], subcategory=it["subcategory"]))
    writer.close()
    m = a_acc(read_jsonl(out_dir / "responses.jsonl"))
    save_json(out_dir / "per_sample.json", m.pop("per_sample"))
    save_json(out_dir / "metrics.json", m)
    print(m)


if __name__ == "__main__":
    main()
