"""Qwen2.5-VL-7B-Instruct (dynamic resolution, 2D M-RoPE, Qwen2 LLM).

Following RESTORE, images are resized so that the full model uses 576 visual
tokens (672 x 672 pixels -> 48 x 48 patches -> 24 x 24 merged tokens).
"""
from __future__ import annotations

from typing import Callable, List, Optional

import torch
from PIL import Image

from .base import INS, SYS, VIS, ModelAdapter, PreparedInputs, grid_centers, split_language_model


class Qwen25VLAdapter(ModelAdapter):
    name = "qwen2_5_vl"

    def __init__(self, path: str = "Qwen/Qwen2.5-VL-7B-Instruct", dtype: str = "bfloat16",
                 device: str = "cuda", image_size: Optional[List[int]] = None, **kwargs):
        super().__init__(path, dtype, device)
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb

        self._apply_mrope = apply_multimodal_rotary_pos_emb
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            path, torch_dtype=self.dtype, attn_implementation="sdpa").to(self.device).eval()
        self.image_size = tuple(image_size) if image_size else (672, 672)
        pixels = self.image_size[0] * self.image_size[1]
        self.processor = AutoProcessor.from_pretrained(path, min_pixels=pixels, max_pixels=pixels)
        self.tokenizer = self.processor.tokenizer
        self.decoder, self.lm_head = split_language_model(self.model)
        self.image_token_id = self.model.config.image_token_id
        self.mrope_section = self.text_config.rope_scaling["mrope_section"]
        self.merge = self.model.config.vision_config.spatial_merge_size

    def rope_fn(self, q, k, cos, sin):
        return self._apply_mrope(q, k, cos, sin, self.mrope_section)

    def _rope_index_owner(self):
        return self.model if hasattr(self.model, "get_rope_index") else self.model.model

    def next_position_ids(self, base: int, step: int) -> torch.Tensor:
        return torch.full((3, 1, 1), base + step, device=self.device, dtype=torch.long)

    def _messages(self, prompt: str, with_image: bool):
        content = ([{"type": "image"}] if with_image else []) + [{"type": "text", "text": prompt}]
        return [{"role": "user", "content": content}]

    @torch.no_grad()
    def build_inputs(self, image: Optional[Image.Image], prompt: str,
                     pixel_transform: Optional[Callable] = None) -> PreparedInputs:
        text = self.processor.apply_chat_template(self._messages(prompt, image is not None),
                                                  tokenize=False, add_generation_prompt=True)
        if image is None:
            input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
            embeds = self.embed(input_ids)
            n = input_ids.shape[1]
            types = torch.full((n,), INS, dtype=torch.long, device=self.device)
            pos = torch.arange(n, device=self.device)[None, None].expand(3, 1, n)
            empty = torch.zeros(0, dtype=torch.long, device=self.device)
            return PreparedInputs(embeds, pos, types, empty, embeds[0, :0],
                                  torch.zeros(0, 2, device=self.device), None, input_ids)

        image = image.convert("RGB").resize(self.image_size, Image.BICUBIC)
        enc = self.processor(text=[text], images=[image], return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attn_mask = enc["attention_mask"].to(self.device)
        pixel_values = enc["pixel_values"].to(self.device, self.model.visual.dtype)
        grid_thw = enc["image_grid_thw"].to(self.device)
        if pixel_transform is not None:
            pixel_values = pixel_transform(pixel_values)
        feats = self.model.visual(pixel_values, grid_thw=grid_thw).to(self.dtype)

        vis_idx = torch.nonzero(input_ids[0] == self.image_token_id).flatten()
        assert vis_idx.numel() == feats.shape[0], "image token count mismatch"
        embeds = self.embed(input_ids).clone()
        embeds[0, vis_idx] = feats
        position_ids, _ = self._rope_index_owner().get_rope_index(
            input_ids, grid_thw, None, attention_mask=attn_mask)

        n = input_ids.shape[1]
        types = torch.full((n,), INS, dtype=torch.long, device=self.device)
        types[: int(vis_idx[0])] = SYS
        types[vis_idx] = VIS
        _, gh, gw = grid_thw[0].tolist()
        hw = (gh // self.merge, gw // self.merge)
        return PreparedInputs(
            inputs_embeds=embeds, position_ids=position_ids.to(self.device), token_types=types,
            vis_idx=vis_idx, vis_features=feats, grid_positions=grid_centers(*hw, device=self.device),
            grid_hw=hw, input_ids=input_ids)
