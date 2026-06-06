"""Embedding and positional encoding layers.

Currently implements:
  - RotaryEmbedding (RoPE): Rotary Position Embedding

Add your own embedding variants here (sinusoidal, learned positional, ALiBi, etc.).
"""

import torch

from ..config import CoreConfig
from ..core import CoreBlock


class RotaryEmbedding(CoreBlock):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, config: CoreConfig):
        super().__init__(config)
        self.head_dim = config.head_dim
        self.base = config.rope_theta
        inv_freq = 1.0 / (self.base ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim
        ))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x: torch.Tensor, seq_len: int = None) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len is None:
            seq_len = x.shape[-2]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(0)
        sin = emb.sin().unsqueeze(0).unsqueeze(0)
        return cos, sin

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    @staticmethod
    def apply_rotary_pos_emb(
        q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_embed = (q * cos) + (RotaryEmbedding.rotate_half(q) * sin)
        k_embed = (k * cos) + (RotaryEmbedding.rotate_half(k) * sin)
        return q_embed, k_embed
