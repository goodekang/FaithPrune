"""Paired bootstrap confidence intervals over test instances (Sec. S-IV-D).

Two methods evaluated on the same instances are resampled with the same
indices, the metric difference is recomputed on each of the 1,000 resamples,
and the 2.5 / 97.5 percentiles give the 95% interval.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Sequence

import numpy as np

MetricFn = Callable[[Dict[str, np.ndarray], np.ndarray], float]


# ---------------------------------------------------------------------------
# Metrics on resampled per-instance arrays
# ---------------------------------------------------------------------------
def mean_metric(key: str = "score", scale: float = 100.0) -> MetricFn:
    return lambda d, idx: scale * float(d[key][idx].mean())


def ratio_metric(num: str, den: str, scale: float = 100.0) -> MetricFn:
    """Ratio of sums, e.g. CHAIR_i = sum(hallucinated) / sum(mentions)."""
    return lambda d, idx: scale * float(d[num][idx].sum() / max(d[den][idx].sum(), 1))


def f1_metric(pred: str = "pred", label: str = "label") -> MetricFn:
    """F1 with 'yes' (1) as the positive class; pred/label are 0/1 arrays."""
    def fn(d, idx):
        p, l = d[pred][idx], d[label][idx]
        tp = float(((p == 1) & (l == 1)).sum())
        fp = float(((p == 1) & (l == 0)).sum())
        fn_ = float(((p == 0) & (l == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn_, 1)
        return 100 * 2 * prec * rec / max(prec + rec, 1e-12)
    return fn


# ---------------------------------------------------------------------------
def paired_bootstrap(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray], metric: MetricFn,
                     n_resamples: int = 1000, seed: int = 0, alpha: float = 0.05) -> Dict:
    """Difference metric(a) - metric(b) with a paired bootstrap CI."""
    n = len(next(iter(a.values())))
    assert all(len(v) == n for v in list(a.values()) + list(b.values())), "unpaired inputs"
    rng = np.random.default_rng(seed)
    full = np.arange(n)
    point = metric(a, full) - metric(b, full)
    diffs = np.empty(n_resamples)
    for r in range(n_resamples):
        idx = rng.integers(0, n, n)
        diffs[r] = metric(a, idx) - metric(b, idx)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return dict(diff=float(point), ci_low=float(lo), ci_high=float(hi), se=float(diffs.std(ddof=1)))


def bootstrap_se(a: Dict[str, np.ndarray], metric: MetricFn, n_resamples: int = 1000, seed: int = 0) -> float:
    """Unpaired bootstrap standard error of a single metric."""
    n = len(next(iter(a.values())))
    rng = np.random.default_rng(seed)
    vals = [metric(a, rng.integers(0, n, n)) for _ in range(n_resamples)]
    return float(np.std(vals, ddof=1))


def paired_bootstrap_average(benchmarks: Sequence[Dict], n_resamples: int = 1000, seed: int = 0,
                             alpha: float = 0.05) -> Dict:
    """Difference of the relative average over several benchmarks.

    benchmarks: list of dicts with keys
        a, b      -- per-instance arrays of the two methods (dicts of np.ndarray)
        metric    -- MetricFn of the benchmark
        reference -- full-model score used for the relative score (Avg. = mean_k score_k / ref_k)
    Each benchmark is resampled independently on every bootstrap replicate.
    """
    rng = np.random.default_rng(seed)

    def avg(which: str, idxs: List[np.ndarray]) -> float:
        rel = [bm["metric"](bm[which], idx) / bm["reference"] for bm, idx in zip(benchmarks, idxs)]
        return 100 * float(np.mean(rel))

    full = [np.arange(len(next(iter(bm["a"].values())))) for bm in benchmarks]
    point = avg("a", full) - avg("b", full)
    diffs = np.empty(n_resamples)
    for r in range(n_resamples):
        idxs = [rng.integers(0, len(f), len(f)) for f in full]
        diffs[r] = avg("a", idxs) - avg("b", idxs)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return dict(diff=float(point), ci_low=float(lo), ci_high=float(hi))
