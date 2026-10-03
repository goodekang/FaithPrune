"""POPE under the two protocols of the paper (supplementary Sec. S-I).

Protocol A (lmms-eval / RESTORE): all 8,910 COCO questions of the three
    splits, suffix "Answer the question using a single word or phrase.",
    greedy decoding, first word matched against yes/no, F1 averaged over the
    three splits.
Protocol B (CMAC / PAI / OPERA): 3,000 questions per split with the template
    "Is there a {object} in the image?", no suffix, greedy decoding with a
    16-token limit, keyword parsing ("yes"/"no" anywhere in the lower-cased
    answer, "no" taking precedence), per-split accuracy and F1.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List

PROTOCOL_A_SUFFIX = " Answer the question using a single word or phrase."
SPLITS = ("random", "popular", "adversarial")


def build_question(item: Dict, protocol: str) -> str:
    if protocol == "A":
        return item["question"] + PROTOCOL_A_SUFFIX
    if protocol == "B":
        q = item.get("text") or item.get("question")
        if q is None:
            q = f"Is there a {item['object']} in the image?"
        return q
    raise ValueError(protocol)


def parse_answer(text: str, protocol: str) -> str:
    t = text.strip().lower()
    if protocol == "A":
        words = re.findall(r"[a-z]+", t)
        first = words[0] if words else ""
        return "yes" if first == "yes" else "no"
    if protocol == "B":
        words = set(re.findall(r"[a-z]+", t))
        if "no" in words:
            return "no"
        if "yes" in words:
            return "yes"
        return "no"
    raise ValueError(protocol)


def binary_metrics(preds: Iterable[str], labels: Iterable[str]) -> Dict[str, float]:
    """'yes' is the positive class."""
    tp = fp = tn = fn = 0
    for p, l in zip(preds, labels):
        p, l = p == "yes", l.strip().lower() == "yes"
        if p and l:
            tp += 1
        elif p and not l:
            fp += 1
        elif not p and l:
            fn += 1
        else:
            tn += 1
    n = tp + fp + tn + fn
    acc = (tp + tn) / max(n, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return dict(accuracy=100 * acc, precision=100 * prec, recall=100 * rec, f1=100 * f1,
                yes_ratio=100 * (tp + fp) / max(n, 1),
                false_positive_rate=100 * fp / max(fp + tn, 1), n=n)


def summarize(records: List[Dict], protocol: str) -> Dict:
    """records: dicts with keys split, pred, label."""
    out = {}
    for split in SPLITS:
        rs = [r for r in records if r["split"] == split]
        if rs:
            out[split] = binary_metrics([r["pred"] for r in rs], [r["label"] for r in rs])
    if protocol == "A" and out:
        out["avg_f1"] = sum(v["f1"] for v in out.values()) / len(out)
    return out
