"""Sink-Robust Relevance and Alignment Scoring (SR^2S) and the selectors
used in the ablations / controls.

All selectors return the indices (into the N_v visual tokens) of the K
retained tokens, sorted in ascending order so that the original token order
and RoPE indices are preserved.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .greedy import greedy_facility_location
from .kernel import coverage_kernel


def minmax(x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Min-max normalization computed over ``mask`` (all entries if None)."""
    x = x.float()
    ref = x if mask is None else x[mask]
    if ref.numel() == 0:
        return torch.zeros_like(x)
    lo, hi = ref.min(), ref.max()
    out = (x - lo) / (hi - lo).clamp_min(1e-12)
    return out.clamp(0.0, 1.0)


@dataclass
class SelectionInputs:
    """Per-image quantities available at the scoring layer."""
    relevance: Optional[torch.Tensor]   # r_i, [N_v]
    alignment: Optional[torch.Tensor]   # a_i, [N_v]
    sinks: torch.Tensor                 # bool [N_v]
    features: torch.Tensor              # projected visual tokens v_i, [N_v, d]
    positions: torch.Tensor             # normalized grid positions p_i, [N_v, 2]
    grid_hw: Optional[tuple] = None     # (H, W) when the tokens form a single grid


@dataclass
class SelectionResult:
    keep: torch.Tensor                  # sorted indices, [K]
    scores: Optional[torch.Tensor] = None
    candidates: Optional[torch.Tensor] = None


def unified_score(inp: SelectionInputs, candidates: torch.Tensor, beta: float,
                  use_relevance: bool = True, use_alignment: bool = True,
                  zero_sink_relevance: bool = False) -> torch.Tensor:
    """s_i = (1 - beta) r_hat_i + beta a_hat_i, normalized over C (Eq. (score))."""
    n_v = inp.features.shape[0]
    device = inp.features.device
    r_hat = torch.zeros(n_v, device=device)
    a_hat = torch.zeros(n_v, device=device)
    if use_relevance and inp.relevance is not None:
        r = inp.relevance.float().clone()
        if zero_sink_relevance:
            r[inp.sinks] = 0.0
        r_hat = minmax(r, candidates)
    if use_alignment and inp.alignment is not None:
        a_hat = minmax(inp.alignment, candidates)
    if use_relevance and use_alignment:
        s = (1.0 - beta) * r_hat + beta * a_hat
    elif use_relevance:
        s = r_hat
    elif use_alignment:
        s = a_hat
    else:
        s = torch.zeros(n_v, device=device)
    return s.masked_fill(~candidates, 0.0)


def select_sr2s(inp: SelectionInputs, budget: int, beta: float = 0.3, lam: float = 0.5,
                sigma: float = 0.15, sink_mode: str = "exclude", use_relevance: bool = True,
                use_alignment: bool = True, use_coverage: bool = True,
                kernel: str = "feature_position") -> SelectionResult:
    """SR^2S selection.

    sink_mode:
        exclude         -- default; C = [N_v] minus Sigma.
        zero_relevance  -- ablation "sinks kept as candidates (r_i = 0 only)".
        keep            -- no sink handling (FastV-style rows of the ablation).
    """
    n_v = inp.features.shape[0]
    if sink_mode == "exclude":
        candidates = ~inp.sinks
    elif sink_mode in ("zero_relevance", "keep"):
        candidates = torch.ones(n_v, dtype=torch.bool, device=inp.features.device)
    else:
        raise ValueError(f"Unknown sink_mode '{sink_mode}'")

    scores = unified_score(inp, candidates, beta, use_relevance, use_alignment,
                           zero_sink_relevance=(sink_mode == "zero_relevance"))
    k = min(int(budget), int(candidates.sum().item()))
    if use_coverage:
        kern = coverage_kernel(inp.positions, inp.features, sigma=sigma, kind=kernel)
        keep = greedy_facility_location(scores, kern, candidates, budget, lam=lam)
    else:
        keep = torch.topk(scores.masked_fill(~candidates, float("-inf")), k).indices
    return SelectionResult(keep=torch.sort(keep).values, scores=scores, candidates=candidates)


def select_topk_relevance(inp: SelectionInputs, budget: int) -> SelectionResult:
    """FastV-style top-K by the raw relevance (first row of the ablation)."""
    keep = torch.topk(inp.relevance.float(), budget).indices
    return SelectionResult(keep=torch.sort(keep).values)


def select_random(inp: SelectionInputs, budget: int, seed: int = 0,
                  from_candidates: bool = False) -> SelectionResult:
    n_v = inp.features.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    pool = torch.arange(n_v)
    if from_candidates:
        pool = pool[~inp.sinks.cpu()]
    perm = pool[torch.randperm(pool.numel(), generator=g)[:budget]]
    return SelectionResult(keep=torch.sort(perm).values.to(inp.features.device))


def select_uniform_grid(inp: SelectionInputs, budget: int) -> SelectionResult:
    """Uniform g x g grid (g = sqrt(K)); the token closest to each cell center."""
    h, w = inp.grid_hw
    g = int(round(budget ** 0.5))
    assert g * g == budget, "uniform grid control requires a square budget"
    rows = ((torch.arange(g) + 0.5) * h / g).floor().long().clamp_max(h - 1)
    cols = ((torch.arange(g) + 0.5) * w / g).floor().long().clamp_max(w - 1)
    idx = (rows[:, None] * w + cols[None, :]).flatten()
    return SelectionResult(keep=torch.sort(idx).values.to(inp.features.device))


def avgpool_tokens(features: torch.Tensor, grid_hw: tuple, budget: int):
    """2D average pooling of the visual tokens to g x g (g = sqrt(K)).

    Returns pooled features [K, d] and, for each pooled token, the index of
    the original token at the center of its pooling window, whose RoPE index
    the pooled token inherits.
    """
    h, w = grid_hw
    g = int(round(budget ** 0.5))
    assert g * g == budget and h % g == 0 and w % g == 0
    sh, sw = h // g, w // g
    x = features.view(h, w, -1).permute(2, 0, 1)[None].float()
    pooled = torch.nn.functional.avg_pool2d(x, kernel_size=(sh, sw), stride=(sh, sw))
    pooled = pooled[0].permute(1, 2, 0).reshape(g * g, -1).to(features.dtype)
    rows = torch.arange(g) * sh + sh // 2
    cols = torch.arange(g) * sw + sw // 2
    anchor = (rows[:, None] * w + cols[None, :]).flatten().to(features.device)
    return pooled, anchor


def load_external_selection(path: str, sample_id: str) -> torch.Tensor:
    """Indices produced by an external selector (e.g. the official DivPrune /
    VisionZip / HoloV code) and stored as {sample_id: [indices]}."""
    import json
    with open(path, "r", encoding="utf-8") as f:
        table = json.load(f)
    return torch.tensor(sorted(table[str(sample_id)]), dtype=torch.long)
