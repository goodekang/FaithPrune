"""AMBER generative track (CHAIR / Cover / Hal / Cog) and, with
``eval.extra.track=discriminative``, the existence / attribute / relation F1."""
from __future__ import annotations

import os
from pathlib import Path

from tqdm import tqdm

from _common import base_parser, parse, save_run_config

from faithprune.eval.amber import AMBER
from faithprune.pipeline import FaithPrune
from faithprune.utils import JsonlAppender, load_image, load_json, read_jsonl, save_json


def main():
    args, cfg = parse(base_parser(__doc__))
    ex = cfg.eval.extra
    track = ex.get("track", "generative")
    queries = load_json(ex["query_file"][track])
    if cfg.eval.limit:
        queries = queries[: cfg.eval.limit]
    out_dir = Path(args.output or os.path.join(cfg.eval.output_dir, cfg.run_name, f"amber_{track}"))
    save_run_config(cfg, out_dir)
    fp = FaithPrune(cfg)

    writer = JsonlAppender(out_dir / "responses.jsonl", key="id")
    for q in tqdm(queries, desc=f"AMBER-{track}"):
        if q["id"] in writer:
            continue
        res = fp.generate(load_image(os.path.join(ex["image_dir"], q["image"])), q["query"],
                          sample_id=q["id"], max_new_tokens=cfg.decoding.max_new_tokens)
        writer.write(dict(id=q["id"], response=res["text"]))
    writer.close()

    scorer = AMBER(ex["annotation_file"], ex["association_file"], ex["safe_words_file"],
                   similarity_threshold=ex.get("similarity_threshold", 0.8))
    records = read_jsonl(out_dir / "responses.jsonl")
    if track == "generative":
        m = scorer.compute_generative(records)
        save_json(out_dir / "per_sample.json", m.pop("per_sample"))
    else:
        m = scorer.compute_discriminative(records)
    save_json(out_dir / "metrics.json", m)
    print(m)


if __name__ == "__main__":
    main()
