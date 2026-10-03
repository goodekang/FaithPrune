"""LLaVA-1.5 (576 visual tokens on a 24 x 24 grid)."""
from __future__ import annotations

from typing import Callable, Optional

import torch
from PIL import Image

from .base import INS, SYS, VIS, ModelAdapter, PreparedInputs, grid_centers, split_language_model

LLAVA_V1_SYSTEM = ("A chat between a curious human and an artificial intelligence assistant. "
                   "The assistant gives helpful, detailed, and polite answers to the human's questions.")


def llava_v1_prompt(prompt: str, with_image: bool = True) -> str:
    image = "<image>\n" if with_image else ""
    return f"{LLAVA_V1_SYSTEM} USER: {image}{prompt} ASSISTANT:"


class LlavaAdapter(ModelAdapter):
    name = "llava"

    def __init__(self, path: str = "llava-hf/llava-1.5-7b-hf", dtype: str = "float16",
                 device: str = "cuda", **kwargs):
        super().__init__(path, dtype, device)
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

        self._apply_rope = apply_rotary_pos_emb
        self.model = LlavaForConditionalGeneration.from_pretrained(
            path, torch_dtype=self.dtype, attn_implementation="sdpa").to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(path)
        self.tokenizer = self.processor.tokenizer
        self.decoder, self.lm_head = split_language_model(self.model)
        self.image_token_id = self.model.config.image_token_index
        n = self.model.config.vision_config.image_size // self.model.config.vision_config.patch_size
        self.grid_hw = (n, n)

    def rope_fn(self, q, k, cos, sin):
        return self._apply_rope(q, k, cos, sin)

    @torch.no_grad()
    def image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        cfg = self.model.config
        feats = self.model.get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=cfg.vision_feature_layer,
            vision_feature_select_strategy=cfg.vision_feature_select_strategy)
        if isinstance(feats, (list, tuple)):
            feats = torch.cat(list(feats), dim=0)
        return feats.reshape(-1, feats.shape[-1])

    @torch.no_grad()
    def build_inputs(self, image: Optional[Image.Image], prompt: str,
                     pixel_transform: Optional[Callable] = None) -> PreparedInputs:
        text = llava_v1_prompt(prompt, with_image=image is not None)
        if image is None:
            enc = self.tokenizer(text, return_tensors="pt")
            input_ids = enc.input_ids.to(self.device)
            embeds = self.embed(input_ids)
            n = input_ids.shape[1]
            types = torch.full((n,), INS, dtype=torch.long, device=self.device)
            empty = torch.zeros(0, dtype=torch.long, device=self.device)
            return PreparedInputs(embeds, torch.arange(n, device=self.device)[None], types, empty,
                                  embeds[0, :0], torch.zeros(0, 2, device=self.device), None, input_ids)

        enc = self.processor(images=image, text=text, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        pixel_values = enc["pixel_values"].to(self.device, self.dtype)
        if pixel_transform is not None:
            pixel_values = pixel_transform(pixel_values)
        feats = self.image_features(pixel_values).to(self.dtype)

        is_vis = input_ids[0] == self.image_token_id
        vis_idx = torch.nonzero(is_vis).flatten()
        assert vis_idx.numel() == feats.shape[0], "image token count mismatch"
        embeds = self.embed(input_ids).clone()
        embeds[0, vis_idx] = feats

        n = input_ids.shape[1]
        types = torch.full((n,), INS, dtype=torch.long, device=self.device)
        types[: int(vis_idx[0])] = SYS
        types[vis_idx] = VIS
        pos = torch.arange(n, device=self.device)[None]
        return PreparedInputs(
            inputs_embeds=embeds, position_ids=pos, token_types=types, vis_idx=vis_idx,
            vis_features=feats, grid_positions=grid_centers(*self.grid_hw, device=self.device),
            grid_hw=self.grid_hw, input_ids=input_ids)
