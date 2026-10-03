"""Instruction-to-vision relevance at the scoring layer (Eq. (rel)).

Only the |T_ins| x N block of the attention matrix is needed. The query
vectors of the instruction positions are recomputed from the input of layer
l_s and multiplied with the (RoPE-rotated) keys of that layer, so that the
main attention can still run with a fused kernel.
"""
from __future__ import annotations

import math

import torch


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[B, H_kv, T, D] -> [B, H_kv * n_rep, T, D]."""
    if n_rep == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)


def instruction_attention(
    q_ins: torch.Tensor,
    keys: torch.Tensor,
    ins_pos: torch.Tensor,
    key_pos: torch.Tensor,
    key_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Causal softmax attention of the instruction rows.

    Args:
        q_ins: [1, H, |T_ins|, D] rotated queries of the instruction positions.
        keys:  [1, H, N, D] rotated keys of all positions (already repeated for GQA).
        ins_pos: [|T_ins|] sequence indices of the instruction positions.
        key_pos: [N] sequence indices of the keys.
        key_bias: optional additive bias on the keys, [N].
    Returns:
        [H, |T_ins|, N] attention probabilities (float32).
    """
    d = q_ins.shape[-1]
    logits = torch.matmul(q_ins.float(), keys.float().transpose(-1, -2)) / math.sqrt(d)
    logits = logits[0]
    causal = key_pos[None, :] > ins_pos[:, None]
    logits = logits.masked_fill(causal[None], float("-inf"))
    if key_bias is not None:
        logits = logits + key_bias.float()[None, None, :]
    return torch.softmax(logits, dim=-1)


def relevance_from_attention(attn: torch.Tensor, vis_idx: torch.Tensor) -> torch.Tensor:
    """r_i: head- and query-averaged attention received by visual token i.

    attn: [H, |T_ins|, N]; vis_idx: [N_v] key columns of the visual tokens.
    """
    return attn[:, :, vis_idx].mean(dim=(0, 1))
