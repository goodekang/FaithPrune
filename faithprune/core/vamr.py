"""Visual Attention Mass Restoration (VAMR).

For a query q with logits z_q, the visual attention mass is
    m_q(S) = P^S_q / (P^S_q + P^T_q),
with P the sum of exponentiated logits over a key set. Adding gamma to the
logits of the retained visual keys multiplies P^S_q by exp(gamma). Requiring
the restored mass to equal the target m*_q gives (Eq. (gamma_q))
    gamma(q) = log( m*_q P^T_q / ((1 - m*_q) P^S_q) ).
Everything below is computed in log space.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

NEG_INF = float("-inf")


# ---------------------------------------------------------------------------
# Log-partition sums over key sets
# ---------------------------------------------------------------------------
def masked_logsumexp(logits: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
    """logsumexp over keys selected by key_mask.

    logits: [..., N] (causally masked entries must already be -inf).
    key_mask: bool [N] or broadcastable to logits.
    """
    return torch.logsumexp(logits.float().masked_fill(~key_mask, NEG_INF), dim=-1)


def full_model_target(logits: torch.Tensor, vis_mask: torch.Tensor, sink_mask: torch.Tensor,
                      target: str = "non_sink") -> torch.Tensor:
    """Restoration target m*_q from the full-token logits.

    non_sink : m* = P^{V \\ Sigma} / (P^V + P^T)   (default)
    with_sink: m* = P^V / (P^V + P^T)               (ablation "target incl. sink mass")

    logits: [H, Q, N] with causal entries at -inf. vis_mask / sink_mask: bool [N].
    Returns m* of shape [H, Q] (float32).
    """
    log_all = torch.logsumexp(logits.float(), dim=-1)                 # P^V + P^T
    if target == "non_sink":
        log_num = masked_logsumexp(logits, vis_mask & ~sink_mask)
    elif target == "with_sink":
        log_num = masked_logsumexp(logits, vis_mask)
    else:
        raise ValueError(f"Unknown target '{target}'")
    return torch.exp(log_num - log_all)


def per_query_gamma(logits: torch.Tensor, kept_vis_mask: torch.Tensor,
                    m_star: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Closed form gamma(q) on the pruned logits.

    logits: [H, Q, N] pruned-pass logits of layer l (without gamma_l).
    kept_vis_mask: bool [N], True for the retained visual keys S.
    m_star: [H, Q] targets.
    """
    m = m_star.float().clamp(eps, 1.0 - eps)
    log_ps = masked_logsumexp(logits, kept_vis_mask)
    log_pt = masked_logsumexp(logits, ~kept_vis_mask)
    return torch.log(m) - torch.log1p(-m) + log_pt - log_ps


def log_ps_minus_log_pt(logits: torch.Tensor, kept_vis_mask: torch.Tensor) -> torch.Tensor:
    """u_q = log P^S_q - log P^T_q, so that m~_q(gamma) = sigmoid(gamma + u_q)."""
    return masked_logsumexp(logits, kept_vis_mask) - masked_logsumexp(logits, ~kept_vis_mask)


# ---------------------------------------------------------------------------
# Aggregation over (I, q, h)
# ---------------------------------------------------------------------------
def aggregate_median(samples: torch.Tensor) -> float:
    """gamma_l = max(0, median over images, text queries and heads) (Eq. (gamma))."""
    return max(0.0, float(torch.median(samples.float()).item()))


def aggregate_mean_matching(u: torch.Tensor, m_star: torch.Tensor,
                            lo: float = -10.0, hi: float = 10.0, iters: int = 60) -> float:
    """Supplementary variant: solve E[m~_q(gamma)] = E[m*_q] by bisection."""
    target = m_star.float().mean()
    u = u.float()
    f = lambda g: torch.sigmoid(g + u).mean() - target
    if f(torch.tensor(lo)) > 0:
        return 0.0
    if f(torch.tensor(hi)) < 0:
        return hi
    a, b = lo, hi
    for _ in range(iters):
        mid = 0.5 * (a + b)
        if f(torch.tensor(mid)) > 0:
            b = mid
        else:
            a = mid
    return max(0.0, 0.5 * (a + b))


# ---------------------------------------------------------------------------
# Bias construction
# ---------------------------------------------------------------------------
class GammaTable:
    """Layer-wise (optionally head-wise) biases gamma_l, 1-indexed layers."""

    def __init__(self, gammas: Dict[int, float | List[float]], meta: Optional[dict] = None):
        self.gammas = {int(k): v for k, v in gammas.items()}
        self.meta = meta or {}

    def get(self, layer: int):
        """layer is 1-indexed; returns 0.0 for layers without a bias."""
        return self.gammas.get(int(layer), 0.0)

    def scaled(self, s: float) -> "GammaTable":
        out = {}
        for k, v in self.gammas.items():
            out[k] = [x * s for x in v] if isinstance(v, list) else v * s
        return GammaTable(out, dict(self.meta, scale=s))

    def as_global(self) -> "GammaTable":
        """Ablation: one scalar for all biased layers (mean of the gamma_l)."""
        vals = [float(sum(v) / len(v)) if isinstance(v, list) else float(v) for v in self.gammas.values()]
        g = sum(vals) / max(len(vals), 1)
        return GammaTable({k: g for k in self.gammas}, dict(self.meta, global_gamma=g))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"gammas": {str(k): v for k, v in sorted(self.gammas.items())},
                       "meta": self.meta}, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "GammaTable":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return cls(obj["gammas"], obj.get("meta"))

    @classmethod
    def zeros(cls, layers: Sequence[int]) -> "GammaTable":
        return cls({l: 0.0 for l in layers})


def key_bias_vector(num_keys: int, kept_vis_cols: torch.Tensor, gamma, device,
                    dtype=torch.float32) -> torch.Tensor:
    """gamma * 1_S over the key axis.

    gamma may be a float (shared by all heads) or a list of per-head values,
    in which case the result is [H, num_keys].
    """
    if isinstance(gamma, (list, tuple)):
        g = torch.tensor(gamma, device=device, dtype=dtype)
        bias = torch.zeros(len(gamma), num_keys, device=device, dtype=dtype)
        bias[:, kept_vis_cols] = g[:, None]
        return bias
    bias = torch.zeros(num_keys, device=device, dtype=dtype)
    if gamma != 0.0:
        bias[kept_vis_cols] = float(gamma)
    return bias


def fold_bias_into_qk(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                      key_bias: torch.Tensor):
    """Kernel-compatible form for attention kernels without bias support.

    Appending sqrt(d_h) to every query and gamma * 1[i in S] to every key adds
    exactly gamma to the visual logits when the softmax scale stays 1/sqrt(d_h).
    The value tensor is padded with a zero column so that all head dims match;
    the extra output column is dropped by the caller.

    q: [B, H, Tq, D]; k, v: [B, H, Tk, D]; key_bias: [Tk] or [H, Tk].
    """
    d = q.shape[-1]
    b, h, tq, _ = q.shape
    tk = k.shape[2]
    q_extra = torch.full((b, h, tq, 1), math.sqrt(d), dtype=q.dtype, device=q.device)
    if key_bias.dim() == 1:
        k_extra = key_bias.to(k.dtype)[None, None, :, None].expand(b, h, tk, 1)
    else:
        k_extra = key_bias.to(k.dtype)[None, :, :, None].expand(b, h, tk, 1)
    v_extra = torch.zeros((b, h, tk, 1), dtype=v.dtype, device=v.device)
    return (torch.cat([q, q_extra], -1), torch.cat([k, k_extra], -1),
            torch.cat([v, v_extra], -1))
