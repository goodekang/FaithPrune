"""Paired bootstrap 95% confidence intervals (supplementary Table S-IV).

The comparisons are listed in a YAML file (configs/eval/bootstrap.yaml);
each entry names two result directories produced by the evaluation scripts,
a metric and, for the eight-benchmark average, the lmms-eval sample logs.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from faithprune.eval.bootstrap import (f1_metric, mean_metric, paired_bootstrap,  # noqa: E402
                                       paired_bootstrap_average, ratio_metric)
from faithprune.eval.pope import parse_answer  # noqa: E402
from faithprune.utils import load_json, read_jsonl, save_json  # noqa: E402


# ---------------------------------------------------------------------------
# loaders: each returns (arrays keyed by instance order, metric fn)
# ---------------------------------------------------------------------------
def load_chair(run_dir: str, metric: str, tag: str = "greedy"):
    per = load_json(Path(run_dir) / "chair" / f"per_sample_{tag}.json")
    per = sorted(per, key=lambda p: p["image_id"])
    d = dict(chair_s=np.array([p["chair_s"] for p in per], float),
             hal=np.array([p["num_hallucinated"] for p in per], float),
             men=np.array([p["num_mentions"] for p in per], float))
    fn = mean_metric("chair_s") if metric == "CHAIR_s" else ratio_metric("hal", "men")
    return d, fn


def load_amber(run_dir: str, metric: str):
    per = sorted(load_json(Path(run_dir) / "amber_generative" / "per_sample.json"), key=lambda p: p["id"])
    if metric == "Hal":
        return dict(x=np.array([p["hal"] for p in per], float)), mean_metric("x")
    if metric == "Cover":
        return (dict(n=np.array([p["safe_hit"] for p in per], float),
                     d=np.array([p["safe_total"] for p in per], float)), ratio_metric("n", "d"))
    raise ValueError(metric)


def load_pope_b(run_dir: str, split: str = "adversarial"):
    rows = sorted(read_jsonl(Path(run_dir) / "pope_B" / f"answers_{split}.jsonl"), key=lambda r: r["question_id"])
    return (dict(pred=np.array([parse_answer(r["answer"], "B") == "yes" for r in rows], int),
                 label=np.array([r["label"] == "yes" for r in rows], int)), f1_metric())


def load_hallusionbench(run_dir: str):
    per = load_json(Path(run_dir) / "hallusionbench" / "per_sample.json")
    return dict(x=np.array(per, float)), mean_metric("x")


MME_PERCEPTION = {"existence", "count", "position", "color", "posters", "celebrity", "scene",
                  "landmark", "artwork", "OCR"}


def mme_metric():
    """MME perception score: sum over categories of acc + acc+ (in %), where
    acc+ counts images whose two questions are both answered correctly."""
    def fn(d, idx):
        total = 0.0
        cats, imgs, sc = d["cat"][idx], d["img"][idx], d["x"][idx]
        for c in np.unique(cats):
            sel = cats == c
            acc = sc[sel].mean()
            per_img = {}
            for i, s in zip(imgs[sel], sc[sel]):
                per_img.setdefault(i, []).append(s)
            acc_plus = np.mean([float(all(v)) for v in per_img.values()])
            total += 100 * (acc + acc_plus)
        return total
    return fn


def load_lmms_samples(log_dir: str, task: str, field: str, scale: float = 100.0):
    """Per-instance arrays and metric from lmms-eval --log_samples output.

    POPE: F1 recomputed from the per-sample prediction / ground truth.
    MME: perception score recomputed from the per-sample correctness.
    Others: mean of the per-sample numeric field.
    """
    files = sorted(glob.glob(str(Path(log_dir) / "**" / f"*samples_{task}*.jsonl"), recursive=True))
    assert files, f"no lmms-eval sample log for {task} in {log_dir}"
    rows = [json.loads(l) for l in open(files[-1], encoding="utf-8")]
    rows.sort(key=lambda r: r["doc_id"])
    if task == "pope":
        pred = [int(str(r[field]["prediction"]).strip().lower() == "yes") for r in rows]
        label = [int(str(r[field]["ground_truth"]).strip().lower() == "yes") for r in rows]
        return dict(pred=np.array(pred), label=np.array(label)), f1_metric()
    if task == "mme":
        rows = [r for r in rows if r[field]["category"] in MME_PERCEPTION]
        return (dict(x=np.array([float(r[field]["score"]) for r in rows]),
                     cat=np.array([r[field]["category"] for r in rows]),
                     img=np.array([str(r[field]["question_id"]) for r in rows])), mme_metric())
    vals = []
    for r in rows:
        v = r[field]
        if isinstance(v, dict):
            v = v.get("score", 0.0)
        vals.append(float(v))
    return dict(x=np.array(vals, float)), mean_metric("x", scale)


def load(entry: dict, which: str):
    src = entry[which]
    kind = entry["kind"]
    if kind == "chair":
        return load_chair(src, entry["metric"], entry.get("tag", "greedy"))
    if kind == "amber":
        return load_amber(src, entry["metric"])
    if kind == "pope_b":
        return load_pope_b(src, entry.get("split", "adversarial"))
    if kind == "hallusionbench":
        return load_hallusionbench(src)
    if kind == "lmms":
        return load_lmms_samples(src, entry["task"], entry["field"], entry.get("scale", 100.0))
    raise ValueError(kind)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", default="configs/eval/bootstrap.yaml")
    p.add_argument("--output", default="outputs/bootstrap_ci.json")
    p.add_argument("--n-resamples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    spec = yaml.safe_load(open(a.spec, encoding="utf-8"))

    results = []
    for entry in spec["comparisons"]:
        if entry["kind"] == "average":
            bms = []
            for bm in entry["benchmarks"]:
                da, fn = load_lmms_samples(entry["a"], bm["task"], bm["field"], bm.get("scale", 100.0))
                db, _ = load_lmms_samples(entry["b"], bm["task"], bm["field"], bm.get("scale", 100.0))
                bms.append(dict(a=da, b=db, metric=fn, reference=bm["reference"]))
            r = paired_bootstrap_average(bms, a.n_resamples, a.seed)
        else:
            da, fn = load(entry, "a")
            db, _ = load(entry, "b")
            r = paired_bootstrap(da, db, fn, a.n_resamples, a.seed)
        r.update(name=entry["name"], metric=entry.get("metric", entry["kind"]))
        results.append(r)
        print(f"{r['name']:<55s} {r['diff']:+.1f} [{r['ci_low']:+.1f}, {r['ci_high']:+.1f}]")
    save_json(a.output, results)


if __name__ == "__main__":
    main()
