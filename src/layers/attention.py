"""Attention mechanisms.

Currently implements:
  - GQAttention: Grouped Query Attention with optional weight storage

Add your own attention variants here (MHA, MQA, FlashAttention wrappers, etc.).
"""

import torch
import torch.nn as nn

from ..config import CoreConfig
from ..core import CoreBlock
from .normalization import RMSNorm
from .embeddings import RotaryEmbedding


class GQAttention(CoreBlock):
    """Grouped Query Attention with optional attention weight storage for visualization."""

    def __init__(self, config: CoreConfig):
        super().__init__(config)
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = config.num_heads // config.num_kv_heads

        self.q_proj = nn.Linear(config.d_model, config.num_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.num_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.num_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_heads * config.head_dim, config.d_model, bias=False)

        self.q_norm = RMSNorm(config, dim=config.head_dim)  # normalize Q per-head (128-dim)
        self.k_norm = RMSNorm(config, dim=config.head_dim)  # normalize K per-head (128-dim)

        self.rope = RotaryEmbedding(config)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self._attn_weights: torch.Tensor | None = None  # Populated when store_attn=True

    def forward(self, x: torch.Tensor, store_attn: bool = False, **kwargs) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # project → reshape to heads → QK norm → transpose → RoPE
        q = self.q_norm(self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(x, seq_len=seq_len)
        q, k = RotaryEmbedding.apply_rotary_pos_emb(q, k, cos, sin)

        # GQA: expand KV heads to match Q head count (each KV group serves multiple Q heads)
        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        scale = self.head_dim ** -0.5
        attn_weights = (q @ k.transpose(-2, -1)) * scale

        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=x.device, dtype=q.dtype),
            diagonal=1,
        )
        attn_weights = attn_weights + causal_mask
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        if store_attn:
            self._attn_weights = attn_weights.detach().cpu()

        attn_output = attn_weights @ v
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.resid_dropout(self.o_proj(attn_output))
