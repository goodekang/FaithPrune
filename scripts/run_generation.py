"""Generic open-ended generation over a JSONL question file.

Used for benchmarks whose scoring is done by their official scripts
(MMHal-Bench with the GPT-4 judge, Object HalBench), and for the held-out
GQA split used for hyper-parameter selection. Each input line needs
``id``, ``image`` and ``question``; outputs keep all input fields and add
``response``.
"""
from __future__ import annotations

import os
from pathlib import Path

from tqdm import tqdm

from _common import base_parser, parse, save_run_config

from faithprune.pipeline import FaithPrune
from faithprune.utils import JsonlAppender, load_image, read_jsonl


def main():
    p = base_parser(__doc__)
    p.add_argument("--questions", required=True)
    p.add_argument("--image-dir", required=True)
    p.add_argument("--suffix", default="", help="appended to every question")
    args, cfg = parse(p)
    items = read_jsonl(args.questions)
    if cfg.eval.limit:
        items = items[: cfg.eval.limit]
    out = Path(args.output or os.path.join(cfg.eval.output_dir, cfg.run_name,
                                           Path(args.questions).stem + "_responses.jsonl"))
    save_run_config(cfg, out.parent)
    fp = FaithPrune(cfg)
    writer = JsonlAppender(out, key="id")
    for it in tqdm(items):
        if it["id"] in writer:
            continue
        res = fp.generate(load_image(os.path.join(args.image_dir, it["image"])), it["question"] + args.suffix,
                          sample_id=it["id"], max_new_tokens=cfg.decoding.max_new_tokens)
        writer.write(dict(it, response=res["text"]))
    writer.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
