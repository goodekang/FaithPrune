"""Language-alignment score (Eq. (align)).

The vocabulary embedding matrix E is summarized offline by C = 256 k-means
centroids; a_i is the maximum cosine similarity between the projected visual
token v_i and the centroids.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


@torch.no_grad()
def kmeans(
    x: torch.Tensor,
    num_clusters: int = 256,
    num_iters: int = 100,
    seed: int = 0,
    tol: float = 1e-5,
    chunk: int = 16384,
) -> torch.Tensor:
    """Euclidean k-means with k-means++ initialization.

    Args:
        x: [n, d] float tensor (rows of the embedding matrix).
    Returns:
        [num_clusters, d] centroids (float32).
    """
    x = x.float()
    n, _ = x.shape
    g = torch.Generator(device="cpu").manual_seed(seed)

    # k-means++ initialization
    first = torch.randint(n, (1,), generator=g).item()
    centers = [x[first]]
    d2 = ((x - x[first]) ** 2).sum(-1)
    for _ in range(1, num_clusters):
        probs = (d2 / d2.sum()).cpu()
        idx = torch.multinomial(probs, 1, generator=g).item()
        centers.append(x[idx])
        d2 = torch.minimum(d2, ((x - x[idx]) ** 2).sum(-1))
    c = torch.stack(centers)

    x_sq = (x ** 2).sum(-1, keepdim=True)
    prev_inertia = None
    for _ in range(num_iters):
        assign = torch.empty(n, dtype=torch.long, device=x.device)
        inertia = x.new_zeros(())
        c_sq = (c ** 2).sum(-1)
        for s in range(0, n, chunk):
            xs = x[s:s + chunk]
            dist = x_sq[s:s + chunk] - 2 * xs @ c.T + c_sq[None]
            val, idx = dist.min(-1)
            assign[s:s + chunk] = idx
            inertia = inertia + val.clamp_min(0).sum()
        sums = torch.zeros_like(c).index_add_(0, assign, x)
        counts = torch.bincount(assign, minlength=num_clusters).float()
        empty = counts == 0
        new_c = sums / counts.clamp_min(1)[:, None]
        if empty.any():  # re-seed empty clusters with random points
            ridx = torch.randint(n, (int(empty.sum()),), generator=g).to(x.device)
            new_c[empty] = x[ridx]
        c = new_c
        inertia = inertia.item()
        if prev_inertia is not None and abs(prev_inertia - inertia) <= tol * max(prev_inertia, 1.0):
            break
        prev_inertia = inertia
    return c


def alignment_score(vis_tokens: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """a_i = max_c cos(v_i, mu_c). vis_tokens: [N_v, d]; centroids: [C, d]."""
    v = F.normalize(vis_tokens.float(), dim=-1)
    mu = F.normalize(centroids.to(v.device).float(), dim=-1)
    return (v @ mu.T).max(dim=-1).values


def load_centroids(path: Optional[str], device: torch.device | str = "cpu") -> Optional[torch.Tensor]:
    if path is None:
        return None
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        obj = obj["centroids"]
    return obj.to(device)
