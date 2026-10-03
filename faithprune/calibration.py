"""Layer-sequential calibration of {gamma_l} (Algorithm S2).

For every calibration image I:
  * y^(I) <- greedy caption of the full model;
  * full-token pass with teacher forcing on y^(I); for every l > l_s, head h
    and text query q in T store m*_q = P^{V\\Sigma}_q / (P^V_q + P^T_q).
Then the pruned pass is advanced layer by layer for all images at once:
  * layers 1..l_s, SR^2S selection, drop tokens not in S;
  * for l = l_s+1..L: compute the logits of layer l (which already reflect
    gamma_{l_s+1..l-1}), set
        gamma_l = max(0, median_{(I,q,h)} log(m* P^T / ((1 - m*) P^S))),
    apply gamma_l * 1_S and finish layer l.
"""
from __future__ import annotations

import copy
import logging
from typing import Dict, List, Optional, Sequence

import torch
from PIL import Image

from .config import Config
from .core.sink import find_sinks
from .core.vamr import (GammaTable, aggregate_mean_matching, aggregate_median, full_model_target,
                        log_ps_minus_log_pt, per_query_gamma)
from .generation import Generator
from .modeling.adapters.base import VIS, ModelAdapter
from .modeling.attention import query_key_logits
from .modeling.runner import FaithPruneRunner, PassState, append_answer

log = logging.getLogger(__name__)


def _full_cfg(cfg: Config) -> Config:
    c = copy.deepcopy(cfg)
    c.method.enabled = False
    c.method.vamr.enabled = False
    c.method.amplify.enabled = False
    c.decoding.strategy = "greedy"
    c.decoding.mitigator = "none"
    c.model.attn_impl = "sdpa" if cfg.model.attn_impl == "folded" else cfg.model.attn_impl
    return c


def _pruned_cfg(cfg: Config) -> Config:
    c = copy.deepcopy(cfg)
    c.method.enabled = True
    c.method.vamr.enabled = False      # biases are applied explicitly during calibration
    c.method.amplify.enabled = False
    c.model.attn_impl = "sdpa" if cfg.model.attn_impl == "folded" else cfg.model.attn_impl
    return c


def _layer_logits(runner: FaithPruneRunner, st: PassState, layer_idx0: int) -> torch.Tensor:
    """Pre-softmax logits [H, |T|, T_keys] of the text query rows at one layer."""
    rows = st.text_query_rows
    pos = torch.arange(st.types.numel(), device=st.h.device)
    return query_key_logits(runner.adapter.layers[layer_idx0], st.h, st.cos, st.sin,
                            runner.adapter.rope_fn, runner.dims, rows=rows, key_pos=pos, row_pos=rows)


class Calibrator:
    def __init__(self, adapter: ModelAdapter, cfg: Config, centroids: Optional[torch.Tensor]):
        self.adapter = adapter
        self.cfg = cfg
        self.cal = cfg.calibration
        self.full_runner = FaithPruneRunner(adapter, _full_cfg(cfg))
        self.full_gen = Generator(self.full_runner, _full_cfg(cfg))
        self.pruned_runner = FaithPruneRunner(adapter, _pruned_cfg(cfg), centroids=centroids)
        self.ls = cfg.method.scoring_layer
        self.sink_layer = cfg.effective_sink_layer()
        self.L = adapter.num_layers

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _teacher_forced_inputs(self, image: Image.Image, prompt: str):
        inp = self.adapter.build_inputs(image, prompt)
        y = self.full_gen.generate_ids(image, prompt, self.cal.max_new_tokens)
        return append_answer(self.adapter, inp, y)

    @torch.no_grad()
    def _sinks_full(self, inp) -> torch.Tensor:
        """Sigma of the image, identified at the sink layer of the full pass."""
        st = self.full_runner.begin(inp, use_cache=False)
        for l in range(self.sink_layer):
            self.full_runner.run_layer(st, l)
        return find_sinks(st.h[0, inp.vis_idx], self.cfg.method.tau)

    @torch.no_grad()
    def _full_targets(self, inp, sinks: torch.Tensor) -> Dict[int, torch.Tensor]:
        """m* of every layer l > l_s, as {layer_number: [H, |T|]} on the CPU."""
        st = self.full_runner.begin(inp, use_cache=False)
        n = inp.num_tokens
        vis_mask = torch.zeros(n, dtype=torch.bool, device=st.h.device)
        vis_mask[inp.vis_idx] = True
        sink_mask = torch.zeros_like(vis_mask)
        sink_mask[inp.vis_idx[sinks]] = True
        targets = {}
        for l in range(self.L):
            if l + 1 > self.ls:
                logits = _layer_logits(self.full_runner, st, l)
                targets[l + 1] = full_model_target(logits, vis_mask, sink_mask, self.cal.target).cpu()
            self.full_runner.run_layer(st, l)
        return targets

    @torch.no_grad()
    def _pruned_prefix(self, inp, sample_id) -> PassState:
        """Layers 1..l_s of the pruned pass followed by SR^2S selection."""
        r = self.pruned_runner
        st = r.begin(inp, use_cache=False, sample_id=sample_id)  # l_s = 0: selection happens in begin()
        h_in = None
        for l in range(self.ls):
            if l == self.ls - 1:
                h_in = st.h
            r.run_layer(st, l, gamma=0.0)
        if self.ls > 0:
            r.select_and_prune(st, h_in, sample_id=sample_id)
        return st

    # ------------------------------------------------------------------
    @torch.no_grad()
    def calibrate(self, images: Sequence[Image.Image], sample_ids: Sequence[str],
                  prompts: Optional[Sequence[str]] = None) -> GammaTable:
        """``prompts``: one prompt per image; by default the generic captioning
        prompts of the config are cycled over the images."""
        if prompts is None:
            prompts = [self.cal.prompts[n % len(self.cal.prompts)] for n in range(len(images))]
        states, targets = [], []
        for n, (img, sid) in enumerate(zip(images, sample_ids)):
            prompt = prompts[n]
            inp = self._teacher_forced_inputs(img, prompt)
            sinks = self._sinks_full(inp)
            targets.append(self._full_targets(inp, sinks))
            states.append(self._pruned_prefix(inp, sid))
            log.info("calibration image %d/%d prepared (|Sigma|=%d)", n + 1, len(images), int(sinks.sum()))

        gammas: Dict[int, float | List[float]] = {}
        pending: Dict[int, tuple] = {}            # non-sequential variant
        for l in range(self.ls, self.L):
            layer_no = l + 1
            samples, us, ms = [], [], []
            for st, tg in zip(states, targets):
                logits = _layer_logits(self.pruned_runner, st, l)
                kept = torch.zeros(st.types.numel(), dtype=torch.bool, device=st.h.device)
                kept[st.kept_vis_cols] = True
                m_star = tg[layer_no].to(logits.device)
                samples.append(per_query_gamma(logits, kept, m_star).cpu())       # [H, |T|]
                if self.cal.statistic == "mean_matching":
                    us.append(log_ps_minus_log_pt(logits, kept).cpu())
                    ms.append(m_star.cpu())
            g = self._aggregate(samples, us, ms)
            if self.cal.sequential:
                gammas[layer_no] = g
                for st in states:
                    self.pruned_runner.run_layer(st, l, gamma=g)
            else:
                pending[layer_no] = g
                for st in states:
                    self.pruned_runner.run_layer(st, l, gamma=0.0)
            log.info("layer %d: gamma = %s", layer_no, g)
        if not self.cal.sequential:
            gammas = pending

        meta = dict(model=self.cfg.model.name, budget=self.cfg.method.budget,
                    scoring_layer=self.ls, sink_layer=self.sink_layer,
                    num_images=len(images), statistic=self.cal.statistic,
                    sequential=self.cal.sequential, target=self.cal.target,
                    headwise=self.cal.headwise, selector=self.cfg.method.selector)
        return GammaTable(gammas, meta)

    def _aggregate(self, samples, us, ms):
        if self.cal.headwise:
            per_head = torch.cat([s for s in samples], dim=1)                  # [H, sum |T|]
            if self.cal.statistic == "mean_matching":
                u = torch.cat(us, dim=1)
                m = torch.cat(ms, dim=1)
                return [aggregate_mean_matching(u[h], m[h]) for h in range(per_head.shape[0])]
            return [aggregate_median(per_head[h]) for h in range(per_head.shape[0])]
        flat = torch.cat([s.flatten() for s in samples])
        if self.cal.statistic == "mean_matching":
            return aggregate_mean_matching(torch.cat([u.flatten() for u in us]),
                                           torch.cat([m.flatten() for m in ms]))
        if self.cal.statistic != "median":
            raise ValueError(f"Unknown statistic '{self.cal.statistic}'")
        return aggregate_median(flat)
