"""Coverage kernel (Eq. (kernel)).

kappa(j, i) = exp(-||p_j - p_i||^2 / (2 sigma^2)) * (1 + cos(v_j, v_i)) / 2
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def coverage_kernel(
    positions: torch.Tensor,
    features: torch.Tensor | None,
    sigma: float = 0.15,
    kind: str = "feature_position",
) -> torch.Tensor:
    """Returns the [N_v, N_v] kernel matrix K[j, i] = kappa(j, i).

    positions: [N_v, 2] normalized grid positions in [0, 1]^2.
    features:  [N_v, d] projected visual tokens v_i.
    kind: "feature_position" (default) or "position_only" (ablation).
    """
    p = positions.float()
    d2 = torch.cdist(p, p).pow(2)
    k = torch.exp(-d2 / (2.0 * sigma ** 2))
    if kind == "feature_position":
        v = F.normalize(features.float(), dim=-1)
        k = k * (1.0 + v @ v.T) / 2.0
    elif kind != "position_only":
        raise ValueError(f"Unknown kernel '{kind}'")
    return k.clamp_(0.0, 1.0)
