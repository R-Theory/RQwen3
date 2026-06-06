"""Feed-forward blocks.

Currently implements:
  - FeedForward: Wrapper around SwiGLU MLP

Add your own FFN variants here (standard MLP, MoE feed-forward, etc.).
"""

import torch
import torch.nn as nn

from ..config import CoreConfig
from ..core import CoreBlock
from .activations import SwiGLU


class FeedForward(CoreBlock):
    """Feed-forward block wrapping SwiGLU MLP with residual dropout."""

    def __init__(self, config: CoreConfig):
        super().__init__(config)
        self.mlp = SwiGLU(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.mlp(x))
