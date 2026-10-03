"""lmms-eval model wrapper used for the eight general benchmarks and POPE
Protocol A (Tables I, VI, VII), following the lmms-eval protocol of RESTORE.

Registered as ``faithprune``; launched through ``scripts/run_lmms_eval.py``.
model_args: config=<yaml>[;<yaml>...],set=<key:value>|<key:value>...
(lmms-eval splits model_args on "=", so overrides use ":" between key and value)
"""
from __future__ import annotations

from typing import List, Tuple

from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

from .config import load_config
from .pipeline import FaithPrune


@register_model("faithprune")
class FaithPruneLMM(lmms):
    def __init__(self, config: str = "configs/default.yaml", set: str = "", batch_size: int = 1, **kwargs):
        super().__init__()
        assert int(batch_size) == 1, "FaithPrune is evaluated at batch size 1"
        paths = [p for p in config.split(";") if p]
        overrides = [s.replace(":", "=", 1) for s in str(set).split("|") if s]
        self.cfg = load_config(paths, overrides)
        self.fp = FaithPrune(self.cfg)
        self.batch_size_per_gpu = 1

    @property
    def rank(self):
        return 0

    @property
    def world_size(self):
        return 1

    def generate_until(self, requests: List[Instance]) -> List[str]:
        outputs = []
        for req in tqdm(requests, desc="faithprune"):
            context, gen_kwargs, doc_to_visual, doc_id, task, split = req.args
            visuals = doc_to_visual(self.task_dict[task][split][doc_id])
            image = visuals[0].convert("RGB") if visuals else None
            prompt = context.replace("<image>", "").strip()
            max_new = int(gen_kwargs.get("max_new_tokens", 128))
            res = self.fp.generate(image, prompt, sample_id=f"{task}/{doc_id}", max_new_tokens=max_new)
            text = res["text"]
            for stop in gen_kwargs.get("until", []) or []:
                if stop and stop in text:
                    text = text.split(stop)[0]
            outputs.append(text)
        return outputs

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("all benchmarks are evaluated in generation mode")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError
