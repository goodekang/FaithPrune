"""Statistics behind the empirical analysis (Sec. III) and Sec. V-H.

For a set of COCO images (LLaVA-1.5, single 24 x 24 grid) this computes
  * the norm ratio eta_i at the sink layer and the number of sinks;
  * the share of instruction-to-vision attention absorbed by sinks per layer;
  * Spearman correlations of token scores with an object-overlap oracle
    (raw attention, sink-masked attention, SR^2S without coverage, random,
    alignment a_i, feature norm, local patch variance);
  * the visual attention mass M_l (Eq. (mass)) per layer for the full model,
    its non-sink part, and the pruned model with / without VAMR;
  * the decomposition of the visual mass into on-object, off-object and sink
    parts at selected layers.
Only numbers are produced (JSON); no figures.
"""
from __future__ import annotations

import copy
import zlib
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from .config import Config
from .core.alignment import alignment_score
from .core.selector import SelectionInputs, minmax, select_sr2s
from .core.sink import find_sinks, norm_ratio
from .core.vamr import GammaTable
from .generation import Generator
from .modeling.adapters.base import INS, VIS, ModelAdapter
from .modeling.attention import query_key_logits
from .modeling.runner import FaithPruneRunner, PassState, append_answer


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    r = spearmanr(a, b).correlation
    return float(0.0 if np.isnan(r) else r)


# ---------------------------------------------------------------------------
# Object-overlap oracle
# ---------------------------------------------------------------------------
def object_overlap_oracle(boxes: List[List[float]], image_size, grid_hw=(24, 24), crop: str = "center",
                          input_res: int = 336) -> np.ndarray:
    """Fraction of each token's patch covered by the union of GT boxes.

    boxes: COCO [x, y, w, h] in original pixels. crop="center" follows the
    shortest-edge resize + center crop of the HF CLIP processor; crop="pad"
    follows pad-to-square preprocessing.
    """
    w0, h0 = image_size
    if crop == "center":
        s = input_res / min(w0, h0)
        ox, oy = (w0 * s - input_res) / 2, (h0 * s - input_res) / 2
    else:
        s = input_res / max(w0, h0)
        ox, oy = -(input_res - w0 * s) / 2, -(input_res - h0 * s) / 2
    mask = np.zeros((input_res, input_res), dtype=np.float32)
    for x, y, bw, bh in boxes:
        x1, y1 = int(np.floor(x * s - ox)), int(np.floor(y * s - oy))
        x2, y2 = int(np.ceil((x + bw) * s - ox)), int(np.ceil((y + bh) * s - oy))
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, input_res), min(y2, input_res)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1.0
    gh, gw = grid_hw
    ph, pw = input_res // gh, input_res // gw
    return mask[: gh * ph, : gw * pw].reshape(gh, ph, gw, pw).mean(axis=(1, 3)).flatten()


def local_patch_variance(image: Image.Image, grid_hw=(24, 24), input_res: int = 336) -> np.ndarray:
    w0, h0 = image.size
    s = input_res / min(w0, h0)
    img = image.resize((round(w0 * s), round(h0 * s)), Image.BICUBIC)
    l, t = (img.width - input_res) // 2, (img.height - input_res) // 2
    arr = np.asarray(img.crop((l, t, l + input_res, t + input_res)).convert("L"), dtype=np.float32)
    gh, gw = grid_hw
    ph, pw = input_res // gh, input_res // gw
    return arr.reshape(gh, ph, gw, pw).var(axis=(1, 3)).flatten()


# ---------------------------------------------------------------------------
class AttentionAnalyzer:
    def __init__(self, adapter: ModelAdapter, cfg: Config, centroids: torch.Tensor,
                 gammas: Optional[GammaTable], mass_layers=(4, 16, 28), crop: str = "center"):
        self.adapter = adapter
        self.cfg = cfg
        full = copy.deepcopy(cfg)
        full.method.enabled = False
        full.method.vamr.enabled = False
        full.model.attn_impl = "sdpa"
        self.full = FaithPruneRunner(adapter, full)
        self.gen = Generator(self.full, full)
        pr = copy.deepcopy(cfg)
        pr.method.enabled = True
        pr.method.vamr.enabled = False
        pr.model.attn_impl = "sdpa"
        self.pruned = FaithPruneRunner(adapter, pr, centroids=centroids)
        self.centroids = centroids
        self.gammas = gammas
        self.mass_layers = set(mass_layers)
        self.crop = crop
        self.L = adapter.num_layers

    def _probs(self, runner, st: PassState, l: int, gamma: float = 0.0) -> torch.Tensor:
        rows = st.text_query_rows
        pos = torch.arange(st.types.numel(), device=st.h.device)
        logits = query_key_logits(runner.adapter.layers[l], st.h, st.cos, st.sin, runner.adapter.rope_fn,
                                  runner.dims, rows=rows, key_pos=pos, row_pos=rows)
        if gamma:
            logits[:, :, st.types == VIS] += gamma
        return torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def analyze_image(self, image: Image.Image, prompt: str, boxes: List[List[float]],
                      sample_id: str) -> Dict:
        m = self.cfg.method
        ad = self.adapter
        inp0 = ad.build_inputs(image, prompt)
        y = self.gen.generate_ids(image, prompt, self.cfg.calibration.max_new_tokens)
        inp = append_answer(ad, inp0, y)
        vis_idx = inp.vis_idx
        oracle = object_overlap_oracle(boxes, image.size, inp.grid_hw, self.crop)
        on_obj = torch.tensor(oracle > 0, device=vis_idx.device)

        # ---------------- full pass ----------------
        st = self.full.begin(inp, use_cache=False)
        out: Dict = dict(id=sample_id, sink_share=[], mass_full=[], mass_full_nonsink=[])
        sinks = None
        relevance = None
        for l in range(self.L):
            if l == m.scoring_layer - 1:
                relevance = self.full.relevance(st, st.h)
            if l == self.cfg.effective_sink_layer():
                hv = st.h[0, vis_idx]
                sinks = find_sinks(hv, m.tau)
                out["eta"] = norm_ratio(hv).tolist()
                out["num_sinks"] = int(sinks.sum())
            probs = self._probs(self.full, st, l)                     # [H, |T|, N]
            pv = probs[:, :, vis_idx]
            mass = pv.sum(-1).mean().item()
            out["mass_full"].append(mass)
            ins_rows = (st.types[st.text_query_rows] == INS)
            if sinks is not None:
                out["mass_full_nonsink"].append(pv[:, :, ~sinks].sum(-1).mean().item())
                ins_vis = pv[:, ins_rows]
                out["sink_share"].append((ins_vis[:, :, sinks].sum(-1) / ins_vis.sum(-1).clamp_min(1e-12)).mean().item())
                if l + 1 in self.mass_layers:
                    out.setdefault("decomp_full", {})[l + 1] = dict(
                        on=pv[:, :, on_obj & ~sinks].sum(-1).mean().item(),
                        off=pv[:, :, ~on_obj & ~sinks].sum(-1).mean().item(),
                        sink=pv[:, :, sinks].sum(-1).mean().item())
            self.full.run_layer(st, l)

        # ---------------- score / oracle correlations ----------------
        align = alignment_score(inp.vis_features, self.centroids)
        r_masked = relevance.clone()
        r_masked[sinks] = 0.0
        sel_in = SelectionInputs(relevance, align, sinks, inp.vis_features, inp.grid_positions, inp.grid_hw)
        s_nocov = select_sr2s(sel_in, m.budget, beta=m.beta, use_coverage=False).scores
        rng = np.random.default_rng(zlib.crc32(str(sample_id).encode()))
        o = oracle
        out["spearman"] = dict(
            raw_attention=spearman(relevance.cpu().numpy(), o),
            sink_masked_attention=spearman(r_masked.cpu().numpy(), o),
            sr2s_no_coverage=spearman(s_nocov.cpu().numpy(), o),
            random=spearman(rng.random(o.shape[0]), o),
            alignment=spearman(align.cpu().numpy(), o),
            feature_norm=spearman(inp.vis_features.float().norm(dim=-1).cpu().numpy(), o),
            patch_variance=spearman(local_patch_variance(image, inp.grid_hw), o),
            relevance_vs_alignment=spearman(minmax(relevance).cpu().numpy(), align.cpu().numpy()),
        )

        # ---------------- pruned passes (without / with VAMR) ----------------
        for tag, use_gamma in (("pruned", False), ("pruned_vamr", True)):
            if use_gamma and self.gammas is None:
                continue
            st = self.pruned.begin(inp, use_cache=False, sample_id=sample_id)
            h_in = None
            masses, decomp = [], {}
            for l in range(self.L):
                if l == m.scoring_layer and not st.pruned:
                    self.pruned.select_and_prune(st, h_in, sample_id=sample_id)
                if l == m.scoring_layer - 1:
                    h_in = st.h
                g = self.gammas.get(l + 1) if (use_gamma and l + 1 > m.scoring_layer) else 0.0
                probs = self._probs(self.pruned, st, l, gamma=g)
                cols = st.types == VIS
                pv = probs[:, :, cols]
                masses.append(pv.sum(-1).mean().item())
                if l + 1 in self.mass_layers and st.pruned:
                    kept = torch.tensor(st.info["keep"], device=pv.device)
                    k_on, k_sink = on_obj[kept], sinks[kept]
                    decomp[l + 1] = dict(on=pv[:, :, k_on & ~k_sink].sum(-1).mean().item(),
                                         off=pv[:, :, ~k_on & ~k_sink].sum(-1).mean().item(),
                                         sink=pv[:, :, k_sink].sum(-1).mean().item())
                self.pruned.run_layer(st, l, gamma=g)
            out[f"mass_{tag}"] = masses
            out[f"decomp_{tag}"] = decomp
            out[f"keep_{tag}"] = st.info.get("keep")
        out["sinks"] = torch.nonzero(sinks).flatten().tolist()
        return out
