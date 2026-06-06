"""Reusable neural network layers.

Each module groups related layer types so you can easily add variants:
  - normalization: RMSNorm, (add LayerNorm, etc.)
  - embeddings: RotaryEmbedding, (add sinusoidal, ALiBi, etc.)
  - activations: SwiGLU, (add GELU, Mish, etc.)
  - attention: GQAttention, (add MHA, MQA, etc.)
  - feedforward: FeedForward, (add MoE, standard MLP, etc.)
  - block: TransformerBlock (composable — swap attn/ffn/norm via constructor)
"""

from .normalization import RMSNorm
from .embeddings import RotaryEmbedding
from .activations import SwiGLU
from .attention import GQAttention
from .feedforward import FeedForward
from .block import TransformerBlock

__all__ = [
    "RMSNorm",
    "RotaryEmbedding",
    "SwiGLU",
    "GQAttention",
    "FeedForward",
    "TransformerBlock",
]
