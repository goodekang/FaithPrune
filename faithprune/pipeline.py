"""Build a ready-to-use FaithPrune model from a Config."""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .config import Config
from .core.alignment import load_centroids
from .core.vamr import GammaTable
from .generation import Generator
from .modeling.adapters import build_adapter
from .modeling.runner import FaithPruneRunner

log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def default_centroids_path(cfg: Config) -> str:
    return f"outputs/centroids/{cfg.model.name}_C{cfg.method.num_centroids}.pt"


def default_gamma_path(cfg: Config) -> str:
    m = cfg.method
    size = f"rho{m.budget_ratio:g}" if m.budget_ratio is not None else f"K{m.budget}"
    tag = f"{size}_ls{m.scoring_layer}_{m.selector}"
    return f"outputs/calibration/{cfg.model.name}_{tag}.json"


class FaithPrune:
    """adapter + runner + generator."""

    def __init__(self, cfg: Config, adapter=None, load_gammas: bool = True):
        self.cfg = cfg
        set_seed(cfg.decoding.seed)
        self.adapter = adapter if adapter is not None else build_adapter(cfg.model)
        m = cfg.method

        centroids = None
        if m.enabled and m.selector == "sr2s" and m.use_alignment:
            path = m.centroids_file or default_centroids_path(cfg)
            if not Path(path).exists():
                raise FileNotFoundError(f"{path} not found; run scripts/build_centroids.py first")
            centroids = load_centroids(path, self.adapter.device)
        self.centroids = centroids

        gammas = None
        if m.vamr.enabled and load_gammas:
            path = m.vamr.gamma_file or default_gamma_path(cfg)
            if not Path(path).exists():
                raise FileNotFoundError(f"{path} not found; run scripts/calibrate.py first")
            gammas = GammaTable.load(path)
            log.info("loaded gamma_l from %s", path)
        self.gammas = gammas

        self.runner = FaithPruneRunner(self.adapter, cfg, centroids=centroids, gammas=gammas)
        self.generator = Generator(self.runner, cfg)

    def generate(self, image, prompt: str, sample_id=None, max_new_tokens: Optional[int] = None,
                 seed: Optional[int] = None):
        return self.generator.generate(image, prompt, sample_id=sample_id,
                                       max_new_tokens=max_new_tokens, seed=seed)
