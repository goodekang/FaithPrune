"""LLaVA-NeXT (AnyRes, up to 2880 visual tokens).

The visual sequence is the 24 x 24 base image followed by the unpadded
high-resolution grid with one ``image_newline`` token per row. Grid positions
are computed by pushing a coordinate map through the same reshape/unpad
operations as ``pack_image_features``; base and high-resolution tokens that
look at the same image location therefore share the same normalized position.
Newline tokens are placed at the right border of their row.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
from PIL import Image

from .base import INS, SYS, VIS, ModelAdapter, PreparedInputs, grid_centers, split_language_model
from .llava import llava_v1_prompt


class LlavaNextAdapter(ModelAdapter):
    name = "llava_next"

    def __init__(self, path: str = "llava-hf/llava-v1.6-vicuna-7b-hf", dtype: str = "float16",
                 device: str = "cuda", **kwargs):
        super().__init__(path, dtype, device)
        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        from transformers.models.llava_next.modeling_llava_next import (get_anyres_image_grid_shape,
                                                                        unpad_image)

        self._apply_rope = apply_rotary_pos_emb
        self._grid_shape = get_anyres_image_grid_shape
        self._unpad = unpad_image
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            path, torch_dtype=self.dtype, attn_implementation="sdpa").to(self.device).eval()
        self.processor = LlavaNextProcessor.from_pretrained(path)
        self.tokenizer = self.processor.tokenizer
        self.decoder, self.lm_head = split_language_model(self.model)
        self.image_token_id = self.model.config.image_token_index
        vc = self.model.config.vision_config
        self.base_hw = (vc.image_size // vc.patch_size, vc.image_size // vc.patch_size)

    def rope_fn(self, q, k, cos, sin):
        return self._apply_rope(q, k, cos, sin)

    def _owner(self):
        # image_newline / pack_image_features live on the top model (<=4.51) or on .model (>=4.52)
        return self.model if hasattr(self.model, "pack_image_features") else self.model.model

    def _token_positions(self, image_size) -> torch.Tensor:
        cfg = self.model.config
        h, w = self.base_hw
        base = grid_centers(h, w, device=self.device)
        nph, npw = self._grid_shape(image_size, cfg.image_grid_pinpoints, cfg.vision_config.image_size)
        dummy = torch.zeros(1, nph * h, npw * w)
        hh, ww = self._unpad(dummy, image_size).shape[-2:]
        ys = (torch.arange(hh, dtype=torch.float32) + 0.5) / hh
        xs = (torch.arange(ww, dtype=torch.float32) + 0.5) / ww
        rows = []
        for y in ys:
            row = torch.stack([xs, y.expand(ww)], dim=-1)
            newline = torch.tensor([[1.0, float(y)]])
            rows.append(torch.cat([row, newline], dim=0))
        hi = torch.cat(rows, dim=0).to(self.device)
        return torch.cat([base, hi], dim=0)

    @torch.no_grad()
    def build_inputs(self, image: Optional[Image.Image], prompt: str,
                     pixel_transform: Optional[Callable] = None) -> PreparedInputs:
        text = llava_v1_prompt(prompt, with_image=image is not None)
        if image is None:
            input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
            embeds = self.embed(input_ids)
            n = input_ids.shape[1]
            types = torch.full((n,), INS, dtype=torch.long, device=self.device)
            empty = torch.zeros(0, dtype=torch.long, device=self.device)
            return PreparedInputs(embeds, torch.arange(n, device=self.device)[None], types, empty,
                                  embeds[0, :0], torch.zeros(0, 2, device=self.device), None, input_ids)

        cfg = self.model.config
        enc = self.processor(images=image, text=text, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        pixel_values = enc["pixel_values"].to(self.device, self.dtype)
        image_sizes = enc["image_sizes"].to(self.device)
        if pixel_transform is not None:
            pixel_values = pixel_transform(pixel_values)
        owner = self._owner()
        feats = owner.get_image_features(
            pixel_values, image_sizes, vision_feature_layer=cfg.vision_feature_layer,
            vision_feature_select_strategy=cfg.vision_feature_select_strategy)
        feats, _ = owner.pack_image_features(
            feats, image_sizes, vision_feature_select_strategy=cfg.vision_feature_select_strategy,
            image_newline=owner.image_newline)
        feats = feats.to(self.dtype)

        vis_idx = torch.nonzero(input_ids[0] == self.image_token_id).flatten()
        assert vis_idx.numel() == feats.shape[0], "image token count mismatch"
        embeds = self.embed(input_ids).clone()
        embeds[0, vis_idx] = feats
        n = input_ids.shape[1]
        types = torch.full((n,), INS, dtype=torch.long, device=self.device)
        types[: int(vis_idx[0])] = SYS
        types[vis_idx] = VIS
        positions = self._token_positions(image_sizes[0].tolist())
        assert positions.shape[0] == feats.shape[0]
        return PreparedInputs(
            inputs_embeds=embeds, position_ids=torch.arange(n, device=self.device)[None],
            token_types=types, vis_idx=vis_idx, vis_features=feats, grid_positions=positions,
            grid_hw=None, input_ids=input_ids)
