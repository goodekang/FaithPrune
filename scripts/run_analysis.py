"""Attention statistics of Sec. III and Sec. V-H on 200 COCO images
(numbers only; written to JSON)."""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from _common import base_parser, parse, save_run_config

from faithprune.analysis import AttentionAnalyzer
from faithprune.core.alignment import load_centroids
from faithprune.core.vamr import GammaTable
from faithprune.eval.chair import coco_image_id
from faithprune.modeling.adapters import build_adapter
from faithprune.pipeline import default_centroids_path, default_gamma_path
from faithprune.utils import JsonlAppender, load_image, read_jsonl, read_lines, save_json


def main():
    p = base_parser(__doc__)
    p.add_argument("--num-images", type=int, default=200)
    args, cfg = parse(p)
    ex = cfg.eval.extra
    names = read_lines(ex["image_list"])[: args.num_images]
    with open(ex["instances_file"], "r", encoding="utf-8") as f:
        inst = json.load(f)
    boxes = defaultdict(list)
    for a in inst["annotations"]:
        boxes[a["image_id"]].append(a["bbox"])

    out_dir = Path(args.output or os.path.join(cfg.eval.output_dir, cfg.run_name, "analysis"))
    save_run_config(cfg, out_dir)
    adapter = build_adapter(cfg.model)
    centroids = load_centroids(cfg.method.centroids_file or default_centroids_path(cfg), adapter.device)
    gpath = cfg.method.vamr.gamma_file or default_gamma_path(cfg)
    gammas = GammaTable.load(gpath) if Path(gpath).exists() else None
    an = AttentionAnalyzer(adapter, cfg, centroids, gammas, mass_layers=tuple(ex.get("mass_layers", [4, 16, 28])))

    writer = JsonlAppender(out_dir / "per_image.jsonl", key="id")
    for name in tqdm(names, desc="analysis"):
        if name in writer:
            continue
        iid = coco_image_id(name)
        res = an.analyze_image(load_image(os.path.join(ex["image_dir"], name)), ex["prompt"], boxes[iid], name)
        writer.write(res)
    writer.close()

    rows = read_jsonl(out_dir / "per_image.jsonl")
    eta = np.concatenate([np.array(r["eta"]) for r in rows])
    summary = dict(
        num_images=len(rows),
        sinks_per_image=dict(mean=float(np.mean([r["num_sinks"] for r in rows])),
                             min=int(np.min([r["num_sinks"] for r in rows])),
                             max=int(np.max([r["num_sinks"] for r in rows]))),
        sink_fraction=float((eta > cfg.method.tau).mean()),
        sink_share_per_layer=dict(
            median=np.median([r["sink_share"] for r in rows], axis=0).tolist(),
            q25=np.percentile([r["sink_share"] for r in rows], 25, axis=0).tolist(),
            q75=np.percentile([r["sink_share"] for r in rows], 75, axis=0).tolist()),
        spearman={k: float(np.mean([r["spearman"][k] for r in rows])) for k in rows[0]["spearman"]},
        mass={k: np.mean([r[k] for r in rows if k in r], axis=0).tolist()
              for k in ("mass_full", "mass_full_nonsink", "mass_pruned", "mass_pruned_vamr") if k in rows[0]},
    )
    for key in ("decomp_full", "decomp_pruned", "decomp_pruned_vamr"):
        if key in rows[0]:
            agg = {}
            for layer in rows[0][key]:
                parts = {p: float(np.mean([r[key][layer][p] for r in rows])) for p in ("on", "off", "sink")}
                parts["on_fraction_nonsink"] = parts["on"] / max(parts["on"] + parts["off"], 1e-12)
                agg[layer] = parts
            summary[key] = agg
    save_json(out_dir / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("sinks_per_image", "spearman")}, indent=2))


if __name__ == "__main__":
    main()
