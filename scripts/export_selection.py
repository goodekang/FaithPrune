"""Export the token sets S selected by a FaithPrune configuration as
{sample_id: [indices]}.

Used (i) to run CMAC with its official code on the FaithPrune token set and
(ii) as the input format of ``method.selector=external`` when VAMR is
plugged into other selectors whose official code exports their indices.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from tqdm import tqdm

from _common import base_parser, parse

from faithprune.pipeline import FaithPrune
from faithprune.utils import load_image, read_lines, save_json


def main():
    p = base_parser(__doc__)
    p.add_argument("--image-list", required=True)
    p.add_argument("--image-dir", required=True)
    p.add_argument("--prompt", default="Please describe this image in detail.")
    args, cfg = parse(p)
    cfg.method.vamr.enabled = False
    fp = FaithPrune(cfg, load_gammas=False)
    table = {}
    for name in tqdm(read_lines(args.image_list)):
        inp = fp.adapter.build_inputs(load_image(os.path.join(args.image_dir, name)), args.prompt)
        with torch.no_grad():
            st = fp.runner.forward_prefill(inp, sample_id=name, use_cache=False)
        table[name] = st.info["keep"]
    out = Path(args.output or f"outputs/selections/{cfg.run_name}_K{cfg.method.budget}.json")
    save_json(out, table)
    print(f"wrote {len(table)} selections to {out}")


if __name__ == "__main__":
    main()
