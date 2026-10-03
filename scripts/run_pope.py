"""POPE evaluation (Protocol B by default; Protocol A on local files).

Protocol B question files follow the official POPE format, one JSON per line:
    {"question_id": 1, "image": "COCO_val2014_000000310196.jpg",
     "text": "Is there a snowboard in the image?", "label": "yes"}
The same script evaluates POPE built on A-OKVQA and GQA (supplementary
Table S-V) by pointing ``eval.extra.question_files`` / ``image_dir`` to them.
"""
from __future__ import annotations

import os
from pathlib import Path

from tqdm import tqdm

from _common import base_parser, parse, save_run_config

from faithprune.eval.pope import SPLITS, build_question, parse_answer, summarize
from faithprune.pipeline import FaithPrune
from faithprune.utils import JsonlAppender, load_image, read_jsonl, save_json


def main():
    args, cfg = parse(base_parser(__doc__))
    ex = cfg.eval.extra
    protocol = cfg.eval.protocol
    out_dir = Path(args.output or os.path.join(cfg.eval.output_dir, cfg.run_name, f"pope_{protocol}"))
    save_run_config(cfg, out_dir)
    fp = FaithPrune(cfg)

    records = []
    for split in ex.get("splits", list(SPLITS)):
        qfile = ex["question_files"][split]
        items = read_jsonl(qfile)
        if cfg.eval.limit:
            items = items[: cfg.eval.limit]
        writer = JsonlAppender(out_dir / f"answers_{split}.jsonl", key="question_id")
        for it in tqdm(items, desc=f"POPE-{split}"):
            qid = it["question_id"]
            if qid not in writer:
                image = load_image(os.path.join(ex["image_dir"], it["image"]))
                res = fp.generate(image, build_question(it, protocol), sample_id=f"{split}/{qid}",
                                  max_new_tokens=cfg.decoding.max_new_tokens)
                writer.write(dict(question_id=qid, image=it["image"], answer=res["text"],
                                  label=it["label"], num_kept=res["num_kept_tokens"]))
        writer.close()
        for r in read_jsonl(out_dir / f"answers_{split}.jsonl"):
            records.append(dict(split=split, pred=parse_answer(r["answer"], protocol), label=r["label"]))

    metrics = summarize(records, protocol)
    save_json(out_dir / "metrics.json", metrics)
    for k, v in metrics.items():
        print(k, v)


if __name__ == "__main__":
    main()
