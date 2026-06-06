"""RQwen3 — Qwen3-style transformer built from scratch.

Architecture: Embedding -> N x TransformerBlock(RMSNorm -> GQA -> RMSNorm -> SwiGLU FFN) -> RMSNorm -> LM Head

This model uses the default TransformerBlock (GQA + SwiGLU + RMSNorm).
To build a variant, pass different classes to TransformerBlock:

    # MoE variant — swap FFN for MoE on every other layer
    layers = []
    for i in range(config.n_layer):
        ffn = MoEFeedForward if i % 2 == 0 else FeedForward
        layers.append(TransformerBlock(config, ffn_cls=ffn))
"""

import torch
import torch.nn as nn

from ..config import CoreConfig
from ..core import CoreBlock
from ..layers import RMSNorm, TransformerBlock


class RQwen3Transformer(CoreBlock):
    """N-layer transformer: stacks TransformerBlocks."""

    def __init__(self, config: CoreConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])

    def forward(self, x: torch.Tensor, store_attn: bool = False) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, store_attn=store_attn)
        return x


class LMHeadRQwen3(CoreBlock):
    """Final norm + linear projection to vocabulary."""

    def __init__(self, config: CoreConfig):
        super().__init__(config)
        self.norm_layer = RMSNorm(config)
        self.out_layer = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(self.norm_layer(x))


class RQwen3(CoreBlock):
    """Full model: Embedding -> Transformer -> LM Head."""

    def __init__(self, config: CoreConfig):
        super().__init__(config)
        self.embedding_layer = nn.Embedding(config.vocab_size, config.d_model)
        self.model_transformer = RQwen3Transformer(config)
        self.lm_head = LMHeadRQwen3(config)
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor, store_attn: bool = False) -> torch.Tensor:
        x = self.embedding_layer(x)
        x = self.model_transformer(x, store_attn=store_attn)
        x = self.lm_head(x)
        return x
