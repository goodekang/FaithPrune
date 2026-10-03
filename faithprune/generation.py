"""Autoregressive decoding with the reduced KV cache, plus the decoding-time
hallucination mitigators combined with FaithPrune (VCD and the contrastive
term of PAI). CMAC is run with its official code on top of the token set
exported by FaithPrune (see README)."""
from __future__ import annotations

import copy
import math
from typing import Dict, Optional

import torch
from PIL import Image

from .config import Config, DecodingConfig
from .modeling.runner import FaithPruneRunner


# ---------------------------------------------------------------------------
# VCD image distortion (forward diffusion noise, 1000-step schedule)
# ---------------------------------------------------------------------------
def add_diffusion_noise(pixel_values: torch.Tensor, noise_step: int = 500,
                        generator: Optional[torch.Generator] = None) -> torch.Tensor:
    num_steps = 1000
    betas = torch.linspace(-6, 6, num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
    alphas_prod = torch.cumprod(1 - betas, dim=0)
    a = alphas_prod[noise_step].sqrt().item()
    b = (1 - alphas_prod[noise_step]).sqrt().item()
    noise = torch.randn(pixel_values.shape, generator=generator, device="cpu").to(pixel_values)
    return a * pixel_values + b * noise


def plausibility_mask(logits: torch.Tensor, beta: float) -> torch.Tensor:
    """Adaptive plausibility constraint: keep tokens with p >= beta * max p."""
    logp = torch.log_softmax(logits.float(), dim=-1)
    return logp < (logp.max() + math.log(beta))


def pick_token(logits: torch.Tensor, dec: DecodingConfig, gen: torch.Generator) -> int:
    if dec.strategy == "greedy":
        return int(torch.argmax(logits).item())
    logits = logits.float() / max(dec.temperature, 1e-6)
    if dec.top_k:
        kth = torch.topk(logits, dec.top_k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if dec.top_p:
        sorted_l, idx = torch.sort(logits, descending=True)
        cum = torch.softmax(sorted_l, -1).cumsum(-1)
        drop = cum - torch.softmax(sorted_l, -1) > dec.top_p
        logits[idx[drop]] = float("-inf")
    probs = torch.softmax(logits, dim=-1).cpu()
    return int(torch.multinomial(probs, 1, generator=gen).item())


class Generator:
    """Wraps a FaithPruneRunner and handles decoding / mitigators."""

    def __init__(self, runner: FaithPruneRunner, cfg: Config):
        self.runner = runner
        self.adapter = runner.adapter
        self.cfg = cfg
        self._uncond_runner = None

    def _unconditional_runner(self) -> FaithPruneRunner:
        """Image-free branch for the contrastive term of PAI (no pruning, no bias)."""
        if self._uncond_runner is None:
            cfg = copy.deepcopy(self.cfg)
            cfg.method.enabled = False
            cfg.method.vamr.enabled = False
            cfg.method.amplify.enabled = False
            self._uncond_runner = FaithPruneRunner(self.adapter, cfg)
        return self._uncond_runner

    @torch.no_grad()
    def generate(self, image: Optional[Image.Image], prompt: str, sample_id=None,
                 max_new_tokens: Optional[int] = None, seed: Optional[int] = None) -> Dict:
        dec = self.cfg.decoding
        max_new = max_new_tokens or dec.max_new_tokens
        gen = torch.Generator(device="cpu").manual_seed(dec.seed if seed is None else seed)
        eos = set(self.adapter.eos_token_ids)

        inp = self.adapter.build_inputs(image, prompt)
        state, logits = self.runner.prefill(inp, sample_id=sample_id)

        aux_state, aux_logits, aux_runner = None, None, None
        if dec.mitigator == "vcd":
            noisy = self.adapter.build_inputs(
                image, prompt, pixel_transform=lambda px: add_diffusion_noise(px, dec.vcd_noise_step, gen))
            aux_runner = self.runner
            aux_state, aux_logits = aux_runner.prefill(noisy, sample_id=sample_id)
        elif dec.mitigator == "pai":
            aux_runner = self._unconditional_runner()
            aux_state, aux_logits = aux_runner.prefill(self.adapter.build_inputs(None, prompt))
        elif dec.mitigator == "cmac":
            raise NotImplementedError("CMAC is evaluated with its official implementation; "
                                      "export the FaithPrune token sets with scripts/export_selection.py")

        out_ids = []
        for _ in range(max_new):
            final = self._combine(logits, aux_logits, dec)
            tok = pick_token(final, dec, gen)
            if tok in eos:
                break
            out_ids.append(tok)
            logits = self.runner.step(state, tok)
            if aux_runner is not None:
                aux_logits = aux_runner.step(aux_state, tok)
        return dict(text=self.adapter.decode(out_ids), ids=out_ids,
                    num_kept_tokens=len(state.pass_state.info.get("keep", [])) or None,
                    selection=state.pass_state.info)

    @staticmethod
    def _combine(logits: torch.Tensor, aux: Optional[torch.Tensor], dec: DecodingConfig) -> torch.Tensor:
        if aux is None or dec.mitigator == "none":
            return logits
        if dec.mitigator == "vcd":
            cd = (1 + dec.vcd_alpha) * logits - dec.vcd_alpha * aux
            return cd.masked_fill(plausibility_mask(logits, dec.vcd_beta), float("-inf"))
        if dec.mitigator == "pai":
            out = aux + dec.pai_gamma * (logits - aux)
            if dec.pai_beta is not None:
                out = out.masked_fill(plausibility_mask(logits, dec.pai_beta), float("-inf"))
            return out
        return logits

    @torch.no_grad()
    def generate_ids(self, image, prompt, max_new_tokens: int) -> torch.Tensor:
        """Greedy caption ids (used for teacher forcing in calibration)."""
        res = self.generate(image, prompt, max_new_tokens=max_new_tokens)
        return torch.tensor(res["ids"], dtype=torch.long, device=self.adapter.device)
