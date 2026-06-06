"""Transformer block — the composable unit that models stack N of.

This is the key to extensibility. By accepting attention_cls, ffn_cls, and
norm_cls as parameters, you can build completely different architectures
from the same block:

    # Qwen3-style (GQA + SwiGLU + RMSNorm)
    TransformerBlock(config)

    # Swap FFN for MoE on specific layers
    TransformerBlock(config, ffn_cls=MoEFeedForward)

    # Use standard multi-head attention instead of GQA
    TransformerBlock(config, attn_cls=MultiHeadAttention)

    # Use LayerNorm instead of RMSNorm
    TransformerBlock(config, norm_cls=LayerNorm)
"""

import torch

from ..config import CoreConfig
from ..core import CoreBlock
from .normalization import RMSNorm
from .attention import GQAttention
from .feedforward import FeedForward


class TransformerBlock(CoreBlock):
    """Single transformer layer: Pre-Norm Attention + Pre-Norm FFN with residuals.

    Args:
        config: architecture configuration
        attn_cls: attention class to use (default: GQAttention)
        ffn_cls: feed-forward class to use (default: FeedForward)
        norm_cls: normalization class to use (default: RMSNorm)
    """

    def __init__(
        self,
        config: CoreConfig,
        attn_cls: type = None,
        ffn_cls: type = None,
        norm_cls: type = None,
    ):
        super().__init__(config)
        attn_cls = attn_cls or GQAttention
        ffn_cls = ffn_cls or FeedForward
        norm_cls = norm_cls or RMSNorm

        self.attn_norm = norm_cls(config)
        self.attn = attn_cls(config)
        self.ffn_norm = norm_cls(config)
        self.ffn = ffn_cls(config)

    def forward(self, x: torch.Tensor, store_attn: bool = False, **kwargs) -> torch.Tensor:
        # Pre-norm attention with residual
        x = self.attn(self.attn_norm(x), store_attn=store_attn) + x
        # Pre-norm FFN with residual
        x = self.ffn(self.ffn_norm(x)) + x
        return x
