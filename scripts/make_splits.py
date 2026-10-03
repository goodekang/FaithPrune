"""Image lists for calibration and hyper-parameter selection.

* calibration: 64 COCO train2014 images (Sec. IV-B), disjoint from all test sets;
* held-out split for hyper-parameter selection: 500 COCO train2014 images
  for CHAIR and 1,000 GQA balanced-val questions.
COCO test images (val2014: POPE, CHAIR, AMBER does not use COCO) live in a
different split, so disjointness only has to be enforced between the two
train2014 lists.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coco-train-dir", default="data/coco/train2014")
    p.add_argument("--gqa-questions", default="data/gqa/val_balanced_questions.json")
    p.add_argument("--vg-dir", default="data/visual_genome/images")
    p.add_argument("--out-dir", default="data/splits")
    p.add_argument("--num-calib", type=int, default=64)
    p.add_argument("--num-heldout-chair", type=int, default=500)
    p.add_argument("--num-heldout-gqa", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    rng = random.Random(a.seed)
    images = sorted(f for f in os.listdir(a.coco_train_dir) if f.endswith(".jpg"))
    rng.shuffle(images)
    calib = sorted(images[: a.num_calib])
    heldout = sorted(images[a.num_calib: a.num_calib + a.num_heldout_chair])
    assert not set(calib) & set(heldout)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"calib_coco_train2014_{a.num_calib}.txt").write_text("\n".join(calib) + "\n")
    (out / f"heldout_chair_coco_train2014_{a.num_heldout_chair}.txt").write_text("\n".join(heldout) + "\n")

    if a.vg_dir and Path(a.vg_dir).exists():  # non-COCO calibration variant (Table S-I)
        vg = sorted(f for f in os.listdir(a.vg_dir) if f.endswith(".jpg"))
        rng.shuffle(vg)
        (out / f"calib_vg_{a.num_calib}.txt").write_text("\n".join(sorted(vg[: a.num_calib])) + "\n")

    if Path(a.gqa_questions).exists():
        with open(a.gqa_questions, "r", encoding="utf-8") as f:
            qs = json.load(f)
        ids = sorted(qs.keys())
        rng.shuffle(ids)
        sel = sorted(ids[: a.num_heldout_gqa])
        with open(out / f"heldout_gqa_balanced_val_{a.num_heldout_gqa}.jsonl", "w", encoding="utf-8") as f:
            for qid in sel:
                q = qs[qid]
                f.write(json.dumps({"id": qid, "image": q["imageId"] + ".jpg",
                                    "question": q["question"], "answer": q["answer"]}) + "\n")
    print(f"wrote splits to {out}")


if __name__ == "__main__":
    main()
