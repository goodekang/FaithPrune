"""HallusionBench question-level accuracy (aAcc).

Each record of ``HallusionBench.json`` has a yes/no question, a gt_answer
("1" = yes, "0" = no) and, if visual_input != "0", an image under
``hallusion_bench/<category>/<subcategory>/<set_id>_<figure_id>.png``.
Answers are parsed with yes/no keywords; unparsable answers count as wrong.
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional


def image_path(root: str, item: Dict) -> Optional[str]:
    if str(item.get("visual_input", "1")) == "0":
        return None
    if item.get("filename"):
        return os.path.join(root, item["filename"].lstrip("./"))
    return os.path.join(root, "hallusion_bench", item["category"], item["subcategory"],
                        f"{item['set_id']}_{item['figure_id']}.png")


def parse(text: str) -> Optional[str]:
    words = re.findall(r"[a-z]+", text.lower())
    if not words:
        return None
    if words[0] in ("yes", "no"):
        return "1" if words[0] == "yes" else "0"
    s = set(words)
    if "no" in s or "not" in s:
        return "0"
    if "yes" in s:
        return "1"
    return None


def a_acc(records: List[Dict]) -> Dict:
    """records: dicts with gt_answer and response."""
    correct = sum(int(parse(r["response"]) == str(r["gt_answer"])) for r in records)
    return dict(aAcc=100 * correct / max(len(records), 1), n=len(records),
                per_sample=[int(parse(r["response"]) == str(r["gt_answer"])) for r in records])
