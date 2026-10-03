"""A minimal per-layer KV cache that supports dropping positions."""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch


class KVCache:
    def __init__(self, num_layers: int):
        self.keys: List[Optional[torch.Tensor]] = [None] * num_layers
        self.values: List[Optional[torch.Tensor]] = [None] * num_layers

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """k, v: [B, H_kv, T, D] (keys already rotated). Returns the full cache of the layer."""
        if self.keys[layer] is None:
            self.keys[layer], self.values[layer] = k, v
        else:
            self.keys[layer] = torch.cat([self.keys[layer], k], dim=2)
            self.values[layer] = torch.cat([self.values[layer], v], dim=2)
        return self.keys[layer], self.values[layer]

    def gather(self, index: torch.Tensor) -> None:
        """Keep only the given sequence positions in every filled layer."""
        for l in range(len(self.keys)):
            if self.keys[l] is not None:
                self.keys[l] = self.keys[l].index_select(2, index)
                self.values[l] = self.values[l].index_select(2, index)

    def length(self) -> int:
        for k in self.keys:
            if k is not None:
                return k.shape[2]
        return 0

    def nbytes(self) -> int:
        total = 0
        for k, v in zip(self.keys, self.values):
            if k is not None:
                total += k.numel() * k.element_size() + v.numel() * v.element_size()
        return total
