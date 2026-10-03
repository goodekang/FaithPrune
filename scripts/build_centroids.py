"""Pre-compute the C = 256 k-means centroids of the vocabulary embeddings E
used by the language-alignment score a_i (Eq. (align))."""
from __future__ import annotations

from pathlib import Path

import torch

from _common import base_parser, parse

from faithprune.core.alignment import kmeans
from faithprune.pipeline import default_centroids_path


def main():
    p = base_parser(__doc__)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args, cfg = parse(p)

    from faithprune.modeling.adapters import build_adapter

    adapter = build_adapter(cfg.model)
    emb = adapter.embedding_matrix.detach().float()
    centroids = kmeans(emb, num_clusters=cfg.method.num_centroids, num_iters=args.iters, seed=args.seed)
    out = Path(args.output or cfg.method.centroids_file or default_centroids_path(cfg))
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"centroids": centroids.cpu(), "model": cfg.model.name,
                "num_centroids": cfg.method.num_centroids, "seed": args.seed}, out)
    print(f"saved {tuple(centroids.shape)} centroids to {out}")


if __name__ == "__main__":
    main()
