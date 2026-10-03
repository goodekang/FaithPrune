"""CHAIR (Rohrbach et al., 2018) on COCO val2014.

CHAIR_s: fraction of captions with at least one hallucinated object.
CHAIR_i: fraction of mentioned object instances that are hallucinated.

The ground-truth object set of an image is the union of its instance
annotations and of the objects mentioned in its reference captions, both
mapped to the 80 COCO categories with the original synonym list
(``synonyms.txt``: one line per category, comma-separated, category first).
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, List, Optional

COCO_DOUBLE_WORDS = [
    "motor bike", "motor cycle", "air plane", "traffic light", "street light", "traffic signal",
    "stop light", "fire hydrant", "stop sign", "parking meter", "suit case", "sports ball",
    "baseball bat", "baseball glove", "tennis racket", "wine glass", "hot dog", "cell phone",
    "mobile phone", "teddy bear", "hair drier", "potted plant", "bow tie", "laptop computer",
    "stove top oven", "home plate", "train track",
]
ANIMAL_WORDS = ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
                "giraffe", "animal", "cub"]
VEHICLE_WORDS = ["jet", "train"]


class CHAIR:
    def __init__(self, synonyms_file: str, instances_file: str, captions_file: str):
        import nltk
        from nltk.stem import WordNetLemmatizer

        self._tokenize = nltk.word_tokenize
        self._lemma = WordNetLemmatizer()

        self.inverse_synonym: Dict[str, str] = {}
        self.objects: List[str] = []
        with open(synonyms_file, "r", encoding="utf-8") as f:
            for line in f:
                words = [w.strip() for w in line.strip().split(",") if w.strip()]
                if not words:
                    continue
                for w in words:
                    self.inverse_synonym[w] = words[0]
                    self.objects.append(w)
        self.object_set = set(self.objects)

        self.double_word = {w: w for w in COCO_DOUBLE_WORDS}
        for a in ANIMAL_WORDS:
            self.double_word[f"baby {a}"] = a
            self.double_word[f"adult {a}"] = a
        for v in VEHICLE_WORDS:
            self.double_word[f"passenger {v}"] = v
        self.double_word["bow tie"] = "tie"
        self.double_word["toilet seat"] = "toilet"
        self.double_word["wine glas"] = "wine glass"

        self.imid_to_objects = defaultdict(set)
        self._load_instances(instances_file)
        self._load_captions(captions_file)

    # ------------------------------------------------------------------
    def _singular(self, w: str) -> str:
        return self._lemma.lemmatize(w)

    def caption_to_words(self, caption: str):
        words = [self._singular(w) for w in self._tokenize(caption.lower())]
        merged, idxs = [], []
        i = 0
        while i < len(words):
            pair = " ".join(words[i:i + 2])
            idxs.append(i)
            if pair in self.double_word:
                merged.append(self.double_word[pair])
                i += 2
            else:
                merged.append(words[i])
                i += 1
        if "toilet" in merged and "seat" in merged:
            keep = [k for k, w in enumerate(merged) if w != "seat"]
            merged = [merged[k] for k in keep]
            idxs = [idxs[k] for k in keep]
        sel = [k for k, w in enumerate(merged) if w in self.object_set]
        words = [merged[k] for k in sel]
        node_words = [self.inverse_synonym[w] for w in words]
        return words, node_words, [idxs[k] for k in sel]

    def _load_instances(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        id_to_name = {c["id"]: c["name"] for c in data["categories"]}
        for ann in data["annotations"]:
            name = id_to_name[ann["category_id"]]
            self.imid_to_objects[ann["image_id"]].add(self.inverse_synonym.get(name, name))

    def _load_captions(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for ann in data["annotations"]:
            _, nodes, _ = self.caption_to_words(ann["caption"])
            self.imid_to_objects[ann["image_id"]].update(nodes)

    # ------------------------------------------------------------------
    def score_caption(self, image_id: int, caption: str) -> Dict:
        words, nodes, _ = self.caption_to_words(caption)
        gt = self.imid_to_objects[image_id]
        hallucinated = [(w, n) for w, n in zip(words, nodes) if n not in gt]
        mentioned_unique = set(nodes)
        return dict(
            image_id=image_id,
            mentioned=list(zip(words, nodes)),
            hallucinated=hallucinated,
            num_mentions=len(nodes),
            num_hallucinated=len(hallucinated),
            chair_s=int(len(hallucinated) > 0),
            num_objects=len(mentioned_unique),
            num_gt=len(gt),
            num_covered=len(mentioned_unique & gt),
            length=len(caption.split()),
        )

    def compute(self, records: List[Dict], caption_key: str = "caption") -> Dict:
        """records: dicts with image_id and the generated caption."""
        per = [self.score_caption(int(r["image_id"]), r[caption_key]) for r in records]
        n = max(len(per), 1)
        mentions = sum(p["num_mentions"] for p in per)
        return dict(
            CHAIR_s=100 * sum(p["chair_s"] for p in per) / n,
            CHAIR_i=100 * sum(p["num_hallucinated"] for p in per) / max(mentions, 1),
            recall=100 * sum(p["num_covered"] for p in per) / max(sum(p["num_gt"] for p in per), 1),
            objects_per_caption=sum(p["num_objects"] for p in per) / n,
            length=sum(p["length"] for p in per) / n,
            n=len(per),
            per_sample=per,
        )


def coco_image_id(file_name: str) -> int:
    """COCO_val2014_000000310196.jpg -> 310196."""
    return int(file_name.rsplit("_", 1)[-1].split(".")[0])
