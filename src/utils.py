"""Shared utilities for Qwen3 analysis notebooks."""

import torch


def get_device() -> torch.device:
    """Return the best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def print_model_summary(model) -> None:
    """Print a summary of model architecture and parameter counts."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model type: {type(model).__name__}")


def format_param_count(n: int) -> str:
    """Format a parameter count as human-readable (e.g., 1.7B, 600M)."""
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    elif n >= 1e6:
        return f"{n / 1e6:.0f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.0f}K"
    return str(n)
