"""Decoder-layer forward with an explicit additive attention bias.

The HF decoder layers of LLaMA (LLaVA-1.5 / LLaVA-NeXT) and Qwen2 (Qwen2.5-VL)
share the structure input_layernorm -> self_attn -> residual ->
post_attention_layernorm -> mlp -> residual. We re-implement the attention
part so that

* the VAMR bias gamma_l * 1_S can be folded into the additive mask used by
  PyTorch SDPA (``impl="sdpa"``),
* or folded into an extra query/key coordinate for FlashAttention-2
  (``impl="folded"``),
* or applied to explicit logits (``impl="eager"``), where a ``logit_hook``
  can read or modify the logits (calibration, analysis, PAI amplification).
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F

from ..core.relevance import repeat_kv
from ..core.vamr import fold_bias_into_qk

LogitHook = Callable[[torch.Tensor], torch.Tensor]


def project_qkv(attn, hidden: torch.Tensor, num_heads: int, num_kv_heads: int, head_dim: int,
                cos: torch.Tensor, sin: torch.Tensor, rope_fn) -> Tuple[torch.Tensor, ...]:
    """Returns rotated q [B,H,T,D], rotated k [B,Hkv,T,D] and v [B,Hkv,T,D]."""
    b, t, _ = hidden.shape
    q = attn.q_proj(hidden).view(b, t, num_heads, head_dim).transpose(1, 2)
    k = attn.k_proj(hidden).view(b, t, num_kv_heads, head_dim).transpose(1, 2)
    v = attn.v_proj(hidden).view(b, t, num_kv_heads, head_dim).transpose(1, 2)
    q, k = rope_fn(q, k, cos, sin)
    return q, k, v


def _flash_folded(q, k, v, key_bias, causal: bool):
    from flash_attn import flash_attn_func  # optional dependency

    d = q.shape[-1]
    if key_bias is not None:
        q, k, v = fold_bias_into_qk(q, k, v, key_bias)
    # flash_attn expects [B, T, H, D]
    out = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                          softmax_scale=1.0 / math.sqrt(d), causal=causal)
    out = out.transpose(1, 2)
    return out[..., :d]


def attention_core(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   mask: Optional[torch.Tensor], impl: str,
                   key_bias: Optional[torch.Tensor] = None,
                   logit_hook: Optional[LogitHook] = None,
                   causal_for_flash: bool = True) -> torch.Tensor:
    """q: [B,H,Tq,D]; k, v: [B,H,Tk,D] (already repeated for GQA).

    mask: additive [1, 1 or H, Tq, Tk] containing the causal mask and, for
          impl in {sdpa, eager}, the VAMR bias.
    key_bias: [Tk] or [H, Tk], used only by impl="folded".
    """
    d = q.shape[-1]
    if impl == "sdpa":
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0 / math.sqrt(d))
    if impl == "folded":
        return _flash_folded(q, k, v, key_bias, causal=causal_for_flash)
    if impl == "eager":
        logits = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(d)
        if logit_hook is not None:
            logits = logit_hook(logits)
        if mask is not None:
            logits = logits + mask.float()
        probs = torch.softmax(logits, dim=-1)
        return torch.matmul(probs.to(v.dtype), v)
    raise ValueError(f"Unknown attention impl '{impl}'")


def decoder_layer_forward(layer, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                          rope_fn, dims: Tuple[int, int, int],
                          cache=None, layer_idx: int = 0,
                          mask: Optional[torch.Tensor] = None, impl: str = "sdpa",
                          key_bias: Optional[torch.Tensor] = None,
                          logit_hook: Optional[LogitHook] = None,
                          causal_for_flash: bool = True) -> torch.Tensor:
    """One decoder layer. ``dims`` = (num_heads, num_kv_heads, head_dim)."""
    num_heads, num_kv_heads, head_dim = dims
    residual = hidden
    x = layer.input_layernorm(hidden)
    attn = layer.self_attn
    q, k, v = project_qkv(attn, x, num_heads, num_kv_heads, head_dim, cos, sin, rope_fn)
    if cache is not None:
        k, v = cache.append(layer_idx, k, v)
    n_rep = num_heads // num_kv_heads
    k_r, v_r = repeat_kv(k, n_rep), repeat_kv(v, n_rep)
    out = attention_core(q, k_r, v_r, mask, impl, key_bias=key_bias, logit_hook=logit_hook,
                         causal_for_flash=causal_for_flash)
    b, _, t, _ = out.shape
    out = out.transpose(1, 2).reshape(b, t, num_heads * head_dim)
    out = attn.o_proj(out)
    hidden = residual + out
    residual = hidden
    x = layer.post_attention_layernorm(hidden)
    hidden = residual + layer.mlp(x)
    return hidden


def query_key_logits(layer, hidden: torch.Tensor, cos, sin, rope_fn, dims,
                     rows: torch.Tensor, key_pos: torch.Tensor, row_pos: torch.Tensor,
                     past_k: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Explicit pre-softmax logits of selected query rows at one layer.

    Used where only a small block of the attention matrix is needed:
    the relevance r_i at layer l_s and the VAMR calibration statistics.

    hidden: [1, T, d] input of the layer; rows: query indices into the T positions.
    key_pos / row_pos: sequence order of keys / query rows for the causal mask.
    Returns [H, |rows|, T_keys] float32 logits with causal entries at -inf.
    """
    num_heads, num_kv_heads, head_dim = dims
    x = layer.input_layernorm(hidden)
    q, k, _ = project_qkv(layer.self_attn, x, num_heads, num_kv_heads, head_dim, cos, sin, rope_fn)
    if past_k is not None:
        k = torch.cat([past_k, k], dim=2)
    k = repeat_kv(k, num_heads // num_kv_heads)
    q = q[:, :, rows]
    logits = torch.matmul(q.float(), k.float().transpose(-1, -2))[0] / math.sqrt(head_dim)
    causal = key_pos[None, :] > row_pos[:, None]
    return logits.masked_fill(causal[None], float("-inf"))
