"""Model adapters: expose the decoder pieces and build the multimodal input
sequence X^0 = [H_sys; V; H_ins] together with the token-type bookkeeping
FaithPrune needs (visual span, instruction span, grid positions)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch
from PIL import Image

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

# token types of the sequence positions
SYS, VIS, INS, GEN = 0, 1, 2, 3


@dataclass
class PreparedInputs:
    inputs_embeds: torch.Tensor            # [1, N, d]
    position_ids: torch.Tensor             # [1, N] or [3, 1, N] (M-RoPE)
    token_types: torch.Tensor              # [N] in {SYS, VIS, INS}
    vis_idx: torch.Tensor                  # [N_v] sequence indices of the visual tokens
    vis_features: torch.Tensor             # [N_v, d] projected visual tokens v_i
    grid_positions: torch.Tensor           # [N_v, 2] normalized positions p_i in [0, 1]^2
    grid_hw: Optional[Tuple[int, int]] = None
    input_ids: Optional[torch.Tensor] = None

    @property
    def num_tokens(self) -> int:
        return self.inputs_embeds.shape[1]

    @property
    def ins_idx(self) -> torch.Tensor:
        return torch.nonzero(self.token_types == INS).flatten()


def grid_centers(h: int, w: int, device=None) -> torch.Tensor:
    """Row-major normalized (x, y) centers of an h x w token grid."""
    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx.flatten(), yy.flatten()], dim=-1)


class ModelAdapter:
    """Common interface. Subclasses fill in the model-specific parts."""

    name: str = "base"

    def __init__(self, path: str, dtype: str = "float16", device: str = "cuda", **kwargs):
        self.path = path
        self.dtype = DTYPES[dtype]
        self.device = torch.device(device)
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.decoder = None          # module holding .layers / .norm / .embed_tokens / .rotary_emb
        self.lm_head = None

    # --- decoder pieces ------------------------------------------------
    @property
    def layers(self):
        return self.decoder.layers

    @property
    def num_layers(self) -> int:
        return len(self.decoder.layers)

    @property
    def dims(self) -> Tuple[int, int, int]:
        cfg = self.text_config
        num_heads = cfg.num_attention_heads
        num_kv = getattr(cfg, "num_key_value_heads", None) or num_heads
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // num_heads
        return num_heads, num_kv, head_dim

    @property
    def text_config(self):
        cfg = self.model.config
        return getattr(cfg, "text_config", None) or cfg

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.decoder.embed_tokens(ids)

    @property
    def embedding_matrix(self) -> torch.Tensor:
        return self.decoder.embed_tokens.weight

    def position_embeddings(self, hidden: torch.Tensor, position_ids: torch.Tensor):
        return self.decoder.rotary_emb(hidden, position_ids)

    def rope_fn(self, q, k, cos, sin):
        raise NotImplementedError

    def final_logits(self, hidden_last: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.decoder.norm(hidden_last)).float()

    def next_position_ids(self, base: int, step: int) -> torch.Tensor:
        """Position id of the step-th generated token; retained tokens keep their
        original indices, so the generated positions equal those of the full model."""
        return torch.tensor([[base + step]], device=self.device)

    # --- inputs --------------------------------------------------------
    def build_inputs(self, image: Optional[Image.Image], prompt: str,
                     pixel_transform: Optional[Callable] = None) -> PreparedInputs:
        raise NotImplementedError

    def encode_answer(self, text: str) -> torch.Tensor:
        """Token ids of a (teacher-forced) answer, without special tokens."""
        ids = self.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
        return ids.to(self.device)

    @property
    def eos_token_ids(self) -> List[int]:
        out = set()
        for ids in (self.tokenizer.eos_token_id,
                    getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)):
            if ids is None:
                continue
            out.update([ids] if isinstance(ids, int) else list(ids))
        return sorted(out)

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True).strip()


def split_language_model(model):
    """Return (decoder, lm_head) across transformers versions."""
    lm = getattr(model, "language_model", None)
    if lm is not None and hasattr(lm, "model") and hasattr(lm, "lm_head"):
        return lm.model, lm.lm_head                      # <= 4.51 (LLaVA family)
    if lm is not None and hasattr(lm, "layers"):
        return lm, model.lm_head                         # >= 4.52 layout
    inner = getattr(model, "model", None)
    if inner is not None:
        if hasattr(inner, "language_model"):
            return inner.language_model, model.lm_head   # >= 4.52 Qwen2.5-VL / LLaVA
        if hasattr(inner, "layers"):
            return inner, model.lm_head                  # <= 4.51 Qwen2.5-VL
    raise RuntimeError("Unsupported model layout")
