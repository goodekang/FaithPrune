"""Coverage-constrained greedy selection (Eqs. (obj) and (greedy)).

F(S) = 1/K sum_{i in S} s_i + lambda/N_v sum_{j=1}^{N_v} max_{i in S} kappa(j, i)

F is normalized, monotone and submodular, so the greedy rule returns a set
within (1 - 1/e) of the optimum. The coverage vector c_j = max_{i in S}
kappa(j, i) is maintained so that each step is one N_v x N_v masked reduction.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def greedy_facility_location(
    scores: torch.Tensor,
    kernel: torch.Tensor,
    candidates: torch.Tensor,
    budget: int,
    lam: float = 0.5,
) -> torch.Tensor:
    """Greedy maximization of F(S) over S subset of the candidates.

    Args:
        scores: [N_v] unified scores s_i (>= 0; ignored for non-candidates).
        kernel: [N_v, N_v] kernel, kernel[j, i] = kappa(j, i). The coverage
            sum runs over all N_v rows j, including sinks.
        candidates: [N_v] boolean mask of the candidate set C.
        budget: K.
        lam: lambda.
    Returns:
        Selected indices in the order they were added (LongTensor [min(K, |C|)]).
    """
    n_v = kernel.shape[0]
    k_eff = min(int(budget), int(candidates.sum().item()))
    s = scores.float()
    kern = kernel.float()
    cover = torch.zeros(n_v, device=kern.device)
    available = candidates.clone()
    chosen = []
    for _ in range(k_eff):
        # marginal coverage gain of every candidate i: sum_j max(0, kappa(j,i) - c_j)
        cov_gain = (kern - cover[:, None]).clamp_min_(0).sum(dim=0)
        gain = s / budget + (lam / n_v) * cov_gain
        gain = gain.masked_fill(~available, float("-inf"))
        i_star = int(torch.argmax(gain).item())
        chosen.append(i_star)
        available[i_star] = False
        cover = torch.maximum(cover, kern[:, i_star])
    return torch.tensor(chosen, dtype=torch.long, device=kern.device)


def objective(scores: torch.Tensor, kernel: torch.Tensor, selected: torch.Tensor,
              budget: int, lam: float = 0.5) -> float:
    """Value of F(S) (used for diagnostics)."""
    if selected.numel() == 0:
        return 0.0
    n_v = kernel.shape[0]
    first = scores[selected].float().sum() / budget
    second = kernel[:, selected].float().max(dim=1).values.sum() * lam / n_v
    return float(first + second)
