"""FaithPrune inference (Algorithm S1).

The decoder is driven layer by layer:

  1. layers 1..l_s run on the full sequence X^0 = [H_sys; V; H_ins];
  2. at layer l_s, SR^2S computes the relevance r_i, the sink set Sigma, the
     alignment a_i and selects S with the coverage-constrained greedy rule;
  3. visual tokens outside S are dropped from the hidden states and from the
     KV cache of every layer; retained tokens keep their RoPE indices;
  4. layers l_s+1..L run on the reduced sequence with the VAMR bias
     gamma_l * 1_S added to the logits of the retained visual keys.

The same machinery (``PassState`` + ``run_layer``) is reused by the
layer-sequential calibration of gamma_l (Algorithm S2).
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import torch

from ..config import Config
from ..core.alignment import alignment_score
from ..core.relevance import repeat_kv
from ..core.selector import (SelectionInputs, avgpool_tokens, load_external_selection,
                             select_random, select_sr2s, select_topk_relevance,
                             select_uniform_grid)
from ..core.sink import find_sinks, norm_ratio
from ..core.vamr import GammaTable, key_bias_vector
from .adapters.base import GEN, INS, VIS, ModelAdapter, PreparedInputs
from .attention import decoder_layer_forward, query_key_logits
from .cache import KVCache

ENCODER_SIDE = {"uniform_grid", "avgpool", "random", "external"}


@dataclass
class PassState:
    """Hidden state of one sequence while it is pushed through the decoder."""
    inp: PreparedInputs
    h: torch.Tensor                       # [1, T, d] input of the next layer
    position_ids: torch.Tensor            # [1, T] or [3, 1, T]
    types: torch.Tensor                   # [T] token types of the current positions
    orig_index: torch.Tensor              # [T] index of each position in the unpruned sequence
    cache: Optional[KVCache] = None
    layers_done: int = 0
    pruned: bool = False
    cos: Optional[torch.Tensor] = None
    sin: Optional[torch.Tensor] = None
    info: Dict = field(default_factory=dict)   # selection diagnostics

    @property
    def kept_vis_cols(self) -> torch.Tensor:
        return torch.nonzero(self.types == VIS).flatten()

    @property
    def text_query_rows(self) -> torch.Tensor:
        return torch.nonzero((self.types == INS) | (self.types == GEN)).flatten()


@dataclass
class DecodeState:
    pass_state: PassState
    next_base: int                        # position id of the first generated token
    steps: int = 0


def append_answer(adapter: ModelAdapter, inp: PreparedInputs, answer_ids: torch.Tensor) -> PreparedInputs:
    """Teacher forcing: X = [H_sys; V; H_ins; y]. Generated tokens are GEN queries."""
    ans = answer_ids.view(1, -1).to(inp.inputs_embeds.device)
    n_ans = ans.shape[1]
    emb = torch.cat([inp.inputs_embeds, adapter.embed(ans)], dim=1)
    base = int(inp.position_ids.max().item()) + 1
    new_pos = torch.arange(base, base + n_ans, device=emb.device)
    if inp.position_ids.dim() == 3:
        pos = torch.cat([inp.position_ids, new_pos[None, None].expand(3, 1, n_ans)], dim=-1)
    else:
        pos = torch.cat([inp.position_ids, new_pos[None]], dim=-1)
    types = torch.cat([inp.token_types, torch.full((n_ans,), GEN, device=emb.device,
                                                   dtype=inp.token_types.dtype)])
    ids = None if inp.input_ids is None else torch.cat([inp.input_ids, ans], dim=1)
    return PreparedInputs(emb, pos, types, inp.vis_idx, inp.vis_features, inp.grid_positions,
                          inp.grid_hw, ids)


class FaithPruneRunner:
    def __init__(self, adapter: ModelAdapter, cfg: Config, centroids: Optional[torch.Tensor] = None,
                 gammas: Optional[GammaTable] = None):
        self.adapter = adapter
        self.cfg = cfg
        self.m = cfg.method
        self.impl = cfg.model.attn_impl
        self.centroids = centroids
        self.gammas = gammas
        if gammas is not None and self.m.vamr.global_gamma:
            self.gammas = gammas.as_global()
        if self.gammas is not None and self.m.vamr.scale != 1.0:
            self.gammas = self.gammas.scaled(self.m.vamr.scale)
        if self.m.amplify.enabled and self.impl != "eager":
            raise ValueError("amplification-only bias requires model.attn_impl=eager")
        if self.m.vamr.rows == "text" and self.impl == "folded":
            raise ValueError("text-row-only bias cannot be folded into q/k; use sdpa or eager")
        self.L = adapter.num_layers
        self.dims = adapter.dims

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @property
    def ls(self) -> int:
        return int(self.m.scoring_layer)

    def _refresh_rope(self, st: PassState) -> None:
        st.cos, st.sin = self.adapter.position_embeddings(st.h, st.position_ids)

    def gamma_for(self, layer_number: int):
        """VAMR bias of a 1-indexed layer (0 outside l_s+1..L or when disabled)."""
        if not self.m.vamr.enabled or self.gammas is None or layer_number <= self.ls:
            return 0.0
        return self.gammas.get(layer_number)

    def _bias_rows(self, q_types: torch.Tensor) -> Optional[torch.Tensor]:
        if self.m.vamr.rows == "all":
            return None
        return (q_types == INS) | (q_types == GEN)

    def build_mask(self, q_types: torch.Tensor, k_types: torch.Tensor, q_offset: int,
                   gamma, causal: bool = True) -> Optional[torch.Tensor]:
        """Additive mask M + gamma * 1_S of shape [1, 1|H, Tq, Tk]."""
        tq, tk = q_types.numel(), k_types.numel()
        device = q_types.device
        dtype = torch.float32 if self.impl == "eager" else self.adapter.dtype
        neg = torch.finfo(dtype).min
        mask = torch.zeros(1, 1, tq, tk, device=device, dtype=torch.float32)
        if causal and tq > 1:
            qpos = torch.arange(tq, device=device) + q_offset
            kpos = torch.arange(tk, device=device)
            mask = mask.masked_fill((kpos[None, :] > qpos[:, None])[None, None], float("-inf"))
        has_bias = (isinstance(gamma, (list, tuple)) and any(g != 0 for g in gamma)) or \
                   (not isinstance(gamma, (list, tuple)) and gamma != 0.0)
        if has_bias:
            cols = torch.nonzero(k_types == VIS).flatten()
            bias = key_bias_vector(tk, cols, gamma, device)          # [Tk] or [H, Tk]
            bias = bias[None, None] if bias.dim() == 1 else bias[None, :, None, :]
            rows = self._bias_rows(q_types)
            if rows is not None:
                bias = bias * rows.float()[None, None, :, None]
            mask = mask + bias
        if not causal and not has_bias:
            return None
        mask = torch.where(torch.isinf(mask), torch.full_like(mask, neg), mask)
        return mask.to(dtype)

    def _amplify_hook(self, layer_idx0: int, q_types: torch.Tensor, k_types: torch.Tensor):
        a = self.m.amplify
        if not a.enabled or not (a.start_layer <= layer_idx0 < a.end_layer):
            return None
        cols = torch.nonzero(k_types == VIS).flatten()
        tq = q_types.numel()
        rows = torch.tensor([tq - 1], device=cols.device) if a.last_row_only else torch.arange(tq, device=cols.device)

        def hook(logits):
            sub = logits[:, :, rows][:, :, :, cols]
            logits[:, :, rows[:, None], cols[None, :]] = sub + a.alpha * sub.abs()
            return logits
        return hook

    # ------------------------------------------------------------------
    # pass construction
    # ------------------------------------------------------------------
    def begin(self, inp: PreparedInputs, use_cache: bool = True, sample_id=None) -> PassState:
        n = inp.num_tokens
        st = PassState(inp=inp, h=inp.inputs_embeds, position_ids=inp.position_ids,
                       types=inp.token_types.clone(),
                       orig_index=torch.arange(n, device=inp.inputs_embeds.device),
                       cache=KVCache(self.L) if use_cache else None)
        if self.prune_active(inp) and self.ls == 0:
            self.select_and_prune(st, sample_id=sample_id)
        self._refresh_rope(st)
        return st

    def budget_for(self, n_v: int) -> int:
        if self.m.budget_ratio is not None:
            return max(1, min(n_v, int(round(self.m.budget_ratio * n_v))))
        return min(int(self.m.budget), n_v)

    def prune_active(self, inp: PreparedInputs) -> bool:
        """Pruning applies only to inputs that contain visual tokens (HallusionBench
        has image-free questions)."""
        return self.m.enabled and inp.vis_idx.numel() > 0

    @torch.no_grad()
    def run_layer(self, st: PassState, layer_idx0: int, gamma=None) -> None:
        """Run decoder layer ``layer_idx0`` (0-indexed) on the whole current sequence."""
        layer = self.adapter.layers[layer_idx0]
        if gamma is None:
            gamma = self.gamma_for(layer_idx0 + 1)
        # prefill: queries and keys are the same positions (decoding is handled in step())
        k_types = st.types
        mask = self.build_mask(st.types, k_types, 0, gamma, causal=True)
        key_bias = None
        if self.impl == "folded":
            key_bias = key_bias_vector(k_types.numel(), torch.nonzero(k_types == VIS).flatten(),
                                       gamma, st.h.device)
            mask = None
        hook = self._amplify_hook(layer_idx0, st.types, k_types) if self.impl == "eager" else None
        st.h = decoder_layer_forward(layer, st.h, st.cos, st.sin, self.adapter.rope_fn, self.dims,
                                     cache=st.cache, layer_idx=layer_idx0, mask=mask, impl=self.impl,
                                     key_bias=key_bias, logit_hook=hook)
        st.layers_done = layer_idx0 + 1

    # ------------------------------------------------------------------
    # SR^2S at the scoring layer
    # ------------------------------------------------------------------
    @torch.no_grad()
    def relevance(self, st: PassState, h_in: torch.Tensor) -> torch.Tensor:
        """r_i at layer l_s from the input of that layer (Eq. (rel))."""
        layer = self.adapter.layers[self.ls - 1]
        ins_rows = torch.nonzero(st.types == INS).flatten()
        t = st.types.numel()
        pos = torch.arange(t, device=h_in.device)
        logits = query_key_logits(layer, h_in, st.cos, st.sin, self.adapter.rope_fn, self.dims,
                                  rows=ins_rows, key_pos=pos, row_pos=ins_rows)
        attn = torch.softmax(logits, dim=-1)                        # [H, |T_ins|, T]
        vis_cols = torch.nonzero(st.types == VIS).flatten()
        return attn[:, :, vis_cols].mean(dim=(0, 1))

    @torch.no_grad()
    def select(self, st: PassState, relevance: Optional[torch.Tensor],
               hidden_vis: Optional[torch.Tensor], sample_id=None):
        m = self.m
        inp = st.inp
        n_v = inp.vis_idx.numel()
        k = self.budget_for(n_v)
        sinks = (find_sinks(hidden_vis, m.tau) if hidden_vis is not None
                 else torch.zeros(n_v, dtype=torch.bool, device=inp.vis_features.device))
        align = None
        if m.selector == "sr2s" and m.use_alignment:
            assert self.centroids is not None, "alignment score requires the k-means centroids"
            align = alignment_score(inp.vis_features, self.centroids)
        sel_in = SelectionInputs(relevance=relevance, alignment=align, sinks=sinks,
                                 features=inp.vis_features, positions=inp.grid_positions,
                                 grid_hw=inp.grid_hw)
        if m.selector == "sr2s":
            res = select_sr2s(sel_in, k, beta=m.beta, lam=m.lam, sigma=m.sigma, sink_mode=m.sink_mode,
                              use_relevance=m.use_relevance and relevance is not None,
                              use_alignment=m.use_alignment, use_coverage=m.use_coverage,
                              kernel=m.kernel)
            keep = res.keep
        elif m.selector == "topk_relevance":
            keep = select_topk_relevance(sel_in, k).keep
        elif m.selector == "drop_sinks":
            keep = torch.nonzero(~sinks).flatten()
        elif m.selector == "random":
            offset = zlib.crc32(str(sample_id).encode()) % 100003 if sample_id is not None else 0
            keep = select_random(sel_in, k, seed=m.seed + offset,
                                 from_candidates=m.random_from_candidates).keep
        elif m.selector == "uniform_grid":
            keep = select_uniform_grid(sel_in, k).keep
        elif m.selector == "external":
            keep = load_external_selection(m.external_selection_file, sample_id).to(sinks.device)
        else:
            raise ValueError(f"Unknown selector '{m.selector}'")
        st.info.update(dict(keep=keep.tolist(), sinks=torch.nonzero(sinks).flatten().tolist(),
                            num_sinks=int(sinks.sum())))
        if hidden_vis is not None:
            st.info["eta"] = norm_ratio(hidden_vis).tolist()
        return keep

    @torch.no_grad()
    def prune(self, st: PassState, keep_vis: torch.Tensor) -> None:
        """Drop visual tokens not in S from the sequence and from the KV cache."""
        vis_rows = torch.nonzero(st.types == VIS).flatten()      # current rows of the N_v visual tokens
        keep_rows = vis_rows[keep_vis.to(vis_rows.device)]
        other = torch.nonzero(st.types != VIS).flatten()
        rows = torch.sort(torch.cat([other, keep_rows])).values
        st.h = st.h.index_select(1, rows)
        st.position_ids = st.position_ids.index_select(-1, rows)
        st.types = st.types[rows]
        st.orig_index = st.orig_index[rows]
        if st.cache is not None:
            st.cache.gather(rows)
        st.pruned = True
        self._refresh_rope(st)

    @torch.no_grad()
    def select_and_prune(self, st: PassState, h_in: Optional[torch.Tensor] = None, sample_id=None):
        m = self.m
        if m.selector == "avgpool":
            self._avgpool(st)
            return
        if self.ls == 0 or m.selector in ENCODER_SIDE:
            relevance, hidden_vis = None, None
        else:
            relevance = self.relevance(st, h_in) if (m.use_relevance or m.selector == "topk_relevance") else None
            hidden_vis = st.h[0, torch.nonzero(st.types == VIS).flatten()]
        keep = self.select(st, relevance, hidden_vis, sample_id=sample_id)
        self.prune(st, keep)

    def _avgpool(self, st: PassState) -> None:
        """Control: 2D average pooling of V to sqrt(K) x sqrt(K) before the decoder."""
        assert self.ls == 0 and st.layers_done == 0
        inp = st.inp
        pooled, anchor = avgpool_tokens(inp.vis_features, inp.grid_hw, self.m.budget)
        vis_rows = torch.nonzero(st.types == VIS).flatten()
        h = st.h.clone()
        h[0, vis_rows[anchor]] = pooled.to(h.dtype)
        st.h = h
        st.info.update(dict(keep=anchor.tolist(), sinks=[], num_sinks=0))
        self.prune(st, anchor)

    # ------------------------------------------------------------------
    # prefill / decode
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward_prefill(self, inp: PreparedInputs, sample_id=None, use_cache: bool = True,
                        layer_callback: Optional[Callable] = None) -> PassState:
        st = self.begin(inp, use_cache=use_cache, sample_id=sample_id)
        h_in = None
        for l in range(self.L):
            if self.prune_active(inp) and self.ls > 0 and l == self.ls and not st.pruned:
                self.select_and_prune(st, h_in, sample_id=sample_id)
            if l == self.ls - 1:
                h_in = st.h
            if layer_callback is not None:
                layer_callback(self, st, l)
            self.run_layer(st, l)
        return st

    @torch.no_grad()
    def prefill(self, inp: PreparedInputs, sample_id=None):
        st = self.forward_prefill(inp, sample_id=sample_id, use_cache=True)
        logits = self.adapter.final_logits(st.h[:, -1])
        base = int(inp.position_ids.max().item()) + 1
        return DecodeState(pass_state=st, next_base=base), logits[0]

    @torch.no_grad()
    def step(self, ds: DecodeState, token_id: int) -> torch.Tensor:
        st = ds.pass_state
        dev = st.h.device
        ids = torch.tensor([[token_id]], device=dev)
        h = self.adapter.embed(ids)
        pos = self.adapter.next_position_ids(ds.next_base, ds.steps)
        cos, sin = self.adapter.position_embeddings(h, pos)
        q_types = torch.tensor([GEN], device=dev, dtype=st.types.dtype)
        k_types = torch.cat([st.types, q_types])
        for l in range(self.L):
            gamma = self.gamma_for(l + 1)
            mask = self.build_mask(q_types, k_types, q_offset=k_types.numel() - 1, gamma=gamma, causal=False)
            key_bias = None
            if self.impl == "folded":
                key_bias = key_bias_vector(k_types.numel(), torch.nonzero(k_types == VIS).flatten(), gamma, dev)
                mask = None
            hook = self._amplify_hook(l, q_types, k_types) if self.impl == "eager" else None
            h = decoder_layer_forward(self.adapter.layers[l], h, cos, sin, self.adapter.rope_fn, self.dims,
                                      cache=st.cache, layer_idx=l, mask=mask, impl=self.impl,
                                      key_bias=key_bias, logit_hook=hook)
        st.types = k_types
        st.orig_index = torch.cat([st.orig_index, torch.tensor([-1], device=dev)])
        ds.steps += 1
        return self.adapter.final_logits(h[:, -1])[0]
