"""Normalization layers.

Currently implements:
  - RMSNorm: Root Mean Square Layer Normalization

Add your own normalization variants here (LayerNorm, BatchNorm, etc.).
"""

import torch
import torch.nn as nn

from ..config import CoreConfig
from ..core import CoreBlock


class RMSNorm(CoreBlock):
    """Root Mean Square Layer Normalization."""

    def __init__(self, config: CoreConfig, dim: int = None):
        super().__init__(config)
        self.weight = nn.Parameter(torch.ones(dim or config.d_model))
        self.eps = config.rms_norm_eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return norm * self.weight
