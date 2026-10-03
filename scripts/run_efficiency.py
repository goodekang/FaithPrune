"""Efficiency on LLaVA-1.5-7B (Table V): analytic decoder FLOPs and KV memory,
measured TTFT (encoding + prefilling) and per-token decoding latency,
averaged over 200 samples at batch size 1 (A100, FP16, SDPA)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from _common import base_parser, parse, save_run_config

from faithprune.eval.efficiency import kv_bytes, measure_sample, prefill_flops, shape_from_config
from faithprune.pipeline import FaithPrune
from faithprune.utils import load_image, read_lines, save_json


def main():
    p = base_parser(__doc__)
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--decode-tokens", type=int, default=64)
    args, cfg = parse(p)
    ex = cfg.eval.extra
    names = read_lines(ex["image_list"])[: args.num_samples + args.warmup]
    out_dir = Path(args.output or os.path.join(cfg.eval.output_dir, cfg.run_name, "efficiency"))
    save_run_config(cfg, out_dir)
    fp = FaithPrune(cfg)
    prompt = ex.get("prompt", "Please describe this image in detail.")

    rows = []
    for n, name in enumerate(tqdm(names, desc="efficiency")):
        r = measure_sample(fp, load_image(os.path.join(ex["image_dir"], name)), prompt, args.decode_tokens)
        if n >= args.warmup:
            rows.append(r)
        torch.cuda.empty_cache()

    shape = shape_from_config(fp.adapter.model.config)
    inp = fp.adapter.build_inputs(load_image(os.path.join(ex["image_dir"], names[0])), prompt)
    n_full = inp.num_tokens
    n_vis = inp.vis_idx.numel()
    m = cfg.method
    k = min(m.budget, n_vis) if m.enabled else n_vis
    n_kept = n_full - n_vis + k
    ls = m.scoring_layer if m.enabled else shape.num_layers
    flops = prefill_flops(shape, n_full, n_kept, ls, n_vis=n_vis, n_ins=int(inp.ins_idx.numel()),
                          budget=k if m.enabled else 0, num_centroids=m.num_centroids)
    res = dict(
        budget=k, n_full=n_full, n_kept=n_kept,
        flops_T=flops / 1e12,
        kv_MB=kv_bytes(shape, n_kept) / 2 ** 20,
        ttft_ms=float(np.mean([r["ttft_ms"] for r in rows])),
        decode_ms=float(np.mean([r["decode_ms"] for r in rows])),
        kv_MB_measured=float(np.mean([r["kv_mb_measured"] for r in rows])),
        num_samples=len(rows), gpu=torch.cuda.get_device_name(0), impl=cfg.model.attn_impl,
    )
    save_json(out_dir / "efficiency.json", res)
    print(res)


if __name__ == "__main__":
    main()
