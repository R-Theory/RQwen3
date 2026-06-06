"""src — Deep Learning library for building and experimenting with models.

Package structure:
  src.config      - CoreConfig (architecture settings)
  src.core        - CoreBlock (base class for all components)
  src.layers      - Reusable layers (normalization, embeddings, attention, activations, feedforward)
  src.models      - Complete model architectures (RQwen3, ...)
  src.data        - Datasets and data loading (StreamingTokenDataset, ...)
  src.training    - Training utilities (TrainConfig, optimizer, scheduler, loss, eval, checkpointing)
  src.tokenizers  - Tokenizer wrappers
  src.pipeline    - End-to-end inference pipelines
  src.utils       - Device detection, model summary helpers
"""

from .config import CoreConfig
from .core import CoreBlock
from .layers import RMSNorm, RotaryEmbedding, SwiGLU, GQAttention, FeedForward, TransformerBlock
from .models import RQwen3Transformer, LMHeadRQwen3, RQwen3
from .data import StreamingTokenDataset, PreTokenizedDataset
from .training import (
    TrainConfig, create_optimizer, create_scheduler, compute_loss, evaluate,
    save_checkpoint, load_checkpoint, save_snapshot, generate_sample, train,
)
from .tokenizers import ModelTokenizer
from .pipeline import PipeLine
from .utils import get_device, print_model_summary, format_param_count

__all__ = [
    # Config & base
    "CoreConfig",
    "CoreBlock",
    # Layers
    "RMSNorm",
    "RotaryEmbedding",
    "SwiGLU",
    "GQAttention",
    "FeedForward",
    "TransformerBlock",
    # Models
    "RQwen3Transformer",
    "LMHeadRQwen3",
    "RQwen3",
    # Data
    "StreamingTokenDataset",
    "PreTokenizedDataset",
    # Training
    "TrainConfig",
    "create_optimizer",
    "create_scheduler",
    "compute_loss",
    "evaluate",
    "save_checkpoint",
    "load_checkpoint",
    "save_snapshot",
    "generate_sample",
    "train",
    # Tokenizer & pipeline
    "ModelTokenizer",
    "PipeLine",
    # Utilities
    "get_device",
    "print_model_summary",
    "format_param_count",
]
