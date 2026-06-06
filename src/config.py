"""CoreConfig — single source of truth for architecture-wide settings."""

import torch
from dataclasses import dataclass


@dataclass
class CoreConfig:
    """Single source of truth for architecture-wide settings.

    Usage:
        config = CoreConfig(d_model=256, n_layer=6)     # keyword args
        config = CoreConfig.from_dict({"d_model": 256})  # from a dict
    """
    d_model: int = 512
    dtype: torch.dtype = torch.float32
    rms_norm_eps: float = 1e-6
    dropout: float = 0.1
    max_seq_len: int = 2048
    vocab_size: int = 151_936
    n_layer: int = 28
    num_heads: int = 16
    num_kv_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 6144
    rope_theta: float = 1_000_000.0

    @classmethod
    def from_dict(cls, config: dict) -> "CoreConfig":
        """Create a CoreConfig from a dictionary, ignoring unknown keys."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config.items() if k in valid_fields}
        return cls(**filtered)
