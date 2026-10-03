"""AMBER (Wang et al., 2023): generative track (CHAIR, Cover, Hal, Cog) and the
discriminative existence / attribute / relation F1 used in the supplementary
material. The matching logic follows the official ``inference.py``: nouns are
extracted with NLTK, filtered by the AMBER vocabulary, matched against the
truth / hallucination annotations and their word associations, and finally
against them with spaCy similarity (threshold 0.8)."""
from __future__ import annotations

import json
import re
from typing import Dict, List

from .pope import binary_metrics


class AMBER:
    def __init__(self, annotation_file: str, association_file: str, safe_words_file: str,
                 similarity_threshold: float = 0.8, spacy_model: str = "en_core_web_lg"):
        import nltk
        import spacy
        from nltk.stem import WordNetLemmatizer

        self._nltk = nltk
        self._lemma = WordNetLemmatizer()
        self._nlp = spacy.load(spacy_model)
        self.threshold = similarity_threshold
        with open(annotation_file, "r", encoding="utf-8") as f:
            self.annotations = {int(a["id"]): a for a in json.load(f)}
        with open(association_file, "r", encoding="utf-8") as f:
            self.association: Dict[str, List[str]] = json.load(f)
        self.vocab = set()
        for w1, ws in self.association.items():
            self.vocab.add(w1)
            self.vocab.update(ws)
        with open(safe_words_file, "r", encoding="utf-8") as f:
            self.global_safe = {l.strip() for l in f if l.strip()}

    # ------------------------------------------------------------------
    def extract_nouns(self, text: str) -> List[str]:
        tokens = self._nltk.word_tokenize(text)
        tagged = self._nltk.pos_tag(tokens)
        return [self._lemma.lemmatize(w.lower()) for w, pos in tagged if pos.startswith("NN")]

    def _similar(self, a: str, b: str) -> bool:
        da, db = self._nlp(a), self._nlp(b)
        if not da.has_vector or not db.has_vector:
            return False
        return da.similarity(db) > self.threshold

    def _expand(self, words: List[str]):
        """Words + their associations; owner[k] is the index of the source word."""
        exp, owner = [], []
        for i, w in enumerate(words):
            for a in self.association.get(w, []):
                exp.append(a)
                owner.append(i)
        return exp, owner

    def score_generative(self, sample_id: int, response: str) -> Dict:
        ann = self.annotations[sample_id]
        truth, hallu = ann["truth"], ann["hallu"]
        nouns = [n for n in self.extract_nouns(response) if n in self.vocab]

        safe_exp, safe_owner = self._expand(truth)
        hal_exp, hal_owner = self._expand(hallu)
        safe_hit = [0] * len(truth)
        hal_hit = [0] * len(hallu)
        hallucinated_flags = []

        def mark(word, base, exp, owner, hits) -> bool:
            if word in base:
                hits[base.index(word)] = 1
                return True
            if word in exp:
                hits[owner[exp.index(word)]] = 1
                return True
            return False

        for noun in nouns:
            if noun in self.global_safe:
                continue
            if mark(noun, truth, safe_exp, safe_owner, safe_hit):
                continue
            mark(noun, hallu, hal_exp, hal_owner, hal_hit)
            for j, w in enumerate(hallu + hal_exp):
                if self._similar(noun, w):
                    hal_hit[j if j < len(hallu) else hal_owner[j - len(hallu)]] = 1
                    break
            safe_by_sim = False
            for j, w in enumerate(truth + safe_exp):
                if self._similar(noun, w):
                    safe_hit[j if j < len(truth) else safe_owner[j - len(truth)]] = 1
                    safe_by_sim = True
                    break
            hallucinated_flags.append(0 if safe_by_sim else 1)
        # nouns matched directly to the truth count as non-hallucinated generated objects
        num_generated = len([n for n in nouns if n not in self.global_safe])
        num_hallucinated = sum(hallucinated_flags)
        return dict(id=sample_id, num_generated=num_generated, num_hallucinated=num_hallucinated,
                    safe_hit=sum(safe_hit), safe_total=len(truth),
                    hal_hit=sum(hal_hit), hal_total=len(hallu),
                    hal=int(num_hallucinated > 0))

    def compute_generative(self, records: List[Dict]) -> Dict:
        per = [self.score_generative(int(r["id"]), r["response"]) for r in records]
        n = max(len(per), 1)
        return dict(
            CHAIR=100 * sum(p["num_hallucinated"] for p in per) / max(sum(p["num_generated"] for p in per), 1),
            Cover=100 * sum(p["safe_hit"] for p in per) / max(sum(p["safe_total"] for p in per), 1),
            Hal=100 * sum(p["hal"] for p in per) / n,
            Cog=100 * sum(p["hal_hit"] for p in per) / max(sum(p["hal_total"] for p in per), 1),
            n=len(per), per_sample=per)

    # ------------------------------------------------------------------
    @staticmethod
    def parse_yes_no(text: str) -> str:
        words = set(re.findall(r"[a-z]+", text.lower()))
        if "no" in words:
            return "no"
        return "yes" if "yes" in words else "no"

    def compute_discriminative(self, records: List[Dict]) -> Dict:
        """records: id, response. Grouped by the AMBER question type
        (discriminative-attribute-* are reported together as Attr.)."""
        groups: Dict[str, List] = {"existence": [], "attribute": [], "relation": []}
        for r in records:
            ann = self.annotations[int(r["id"])]
            t = ann["type"]
            key = ("existence" if "existence" in t else "relation" if "relation" in t
                   else "attribute" if "attribute" in t else None)
            if key is None:
                continue
            groups[key].append((self.parse_yes_no(r["response"]), ann["truth"]))
        out = {}
        for k, v in groups.items():
            if v:
                out[k] = binary_metrics([p for p, _ in v], [l for _, l in v])
        return out
