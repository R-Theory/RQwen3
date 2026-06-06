"""Activation / gated MLP layers.

Currently implements:
  - SwiGLU: Gated Linear Unit with Swish activation

Add your own activation variants here (GELU, ReLU, Mish, GeGLU, etc.).
"""

import torch
import torch.nn as nn

from ..config import CoreConfig
from ..core import CoreBlock


class SwiGLU(CoreBlock):
    """SwiGLU activation: gate * swish(up), then down projection."""

    def __init__(self, config: CoreConfig):
        super().__init__(config)
        self.gate_proj = nn.Linear(config.d_model, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))
