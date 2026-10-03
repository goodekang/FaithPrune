"""Efficiency measurement (Sec. V-E).

* FLOPs and KV memory are computed analytically for the decoder, including
  the overhead of SR^2S and VAMR.
* TTFT (encoding + prefilling) and per-token decoding latency are measured
  with CUDA events on one GPU at batch size 1 and averaged over samples.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class DecoderShape:
    num_layers: int
    hidden: int
    intermediate: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    vocab: int
    bytes_per_elem: int = 2


def shape_from_config(cfg) -> DecoderShape:
    tc = getattr(cfg, "text_config", None) or cfg
    nh = tc.num_attention_heads
    nkv = getattr(tc, "num_key_value_heads", None) or nh
    hd = getattr(tc, "head_dim", None) or tc.hidden_size // nh
    return DecoderShape(tc.num_hidden_layers, tc.hidden_size, tc.intermediate_size, nh, nkv, hd,
                        tc.vocab_size)


def layer_flops(s: DecoderShape, n: int) -> float:
    """FLOPs (2 x MACs) of one decoder layer on n tokens (prefill)."""
    d, kv = s.hidden, s.num_kv_heads * s.head_dim
    proj = 2 * n * (d * d + 2 * d * kv + d * d)          # q, k, v, o
    attn = 2 * 2 * n * n * s.num_heads * s.head_dim / 2   # QK^T and AV with causal masking
    mlp = 2 * n * 3 * d * s.intermediate                  # gate, up, down
    return proj + attn + mlp


def prefill_flops(s: DecoderShape, n_full: int, n_kept: int, scoring_layer: int,
                  n_vis: int = 576, n_ins: int = 0, budget: int = 0,
                  num_centroids: int = 256, with_overhead: bool = True) -> float:
    """Decoder prefilling FLOPs: l_s layers on n_full tokens, the rest on n_kept.

    The overhead terms are the alignment score O(N_v C d), the instruction
    attention block O(|T_ins| N d), the kernel matrix O(N_v^2 d) and the
    greedy selection O(K N_v^2).
    """
    total = scoring_layer * layer_flops(s, n_full) + (s.num_layers - scoring_layer) * layer_flops(s, n_kept)
    total += 2 * s.hidden * s.vocab                        # lm_head on the last position
    if with_overhead and budget > 0:
        total += 2 * n_vis * num_centroids * s.hidden
        total += 2 * n_ins * n_full * s.num_heads * s.head_dim + 2 * n_ins * s.hidden * s.hidden
        total += 2 * n_vis * n_vis * s.hidden
        total += 3 * budget * n_vis * n_vis
    return total


def kv_bytes(s: DecoderShape, n_tokens: int) -> int:
    return 2 * s.num_layers * n_tokens * s.num_kv_heads * s.head_dim * s.bytes_per_elem


class CudaTimer:
    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        torch.cuda.synchronize()
        self.start.record()
        return self

    def __exit__(self, *exc):
        self.end.record()
        torch.cuda.synchronize()
        self.ms = self.start.elapsed_time(self.end)


@torch.no_grad()
def measure_sample(fp, image, prompt: str, decode_tokens: int = 64) -> Dict[str, float]:
    """TTFT (vision encoding + prefilling) and mean per-token decoding latency."""
    runner, adapter = fp.runner, fp.adapter
    with CudaTimer() as t_first:
        inp = adapter.build_inputs(image, prompt)
        state, logits = runner.prefill(inp)
    tok = int(torch.argmax(logits).item())
    times: List[float] = []
    for _ in range(decode_tokens):
        with CudaTimer() as t:
            logits = runner.step(state, tok)
        times.append(t.ms)
        tok = int(torch.argmax(logits).item())
    kv = state.pass_state.cache.nbytes()
    return dict(ttft_ms=t_first.ms, decode_ms=sum(times) / max(len(times), 1),
                kv_mb_measured=kv / 2 ** 20, seq_len=state.pass_state.cache.length())
