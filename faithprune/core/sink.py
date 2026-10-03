"""Attention-sink identification (Eq. (sink)).

A visual token is a sink if the l2 norm of its hidden state at the scoring
layer exceeds ``tau`` times the median norm over the visual tokens.
"""
from __future__ import annotations

import torch


def norm_ratio(hidden_vis: torch.Tensor) -> torch.Tensor:
    """eta_i = ||x_i|| / med_j ||x_j|| for hidden states of shape [N_v, d]."""
    norms = hidden_vis.float().norm(dim=-1)
    med = norms.median()
    return norms / med.clamp_min(1e-6)


def find_sinks(hidden_vis: torch.Tensor, tau: float = 3.0) -> torch.Tensor:
    """Boolean mask [N_v], True for sink tokens (Sigma)."""
    return norm_ratio(hidden_vis) > tau
