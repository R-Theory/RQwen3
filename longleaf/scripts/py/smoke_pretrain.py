"""Smoke test: verify the full pretrain pipeline works on Longleaf.

Runs 20 steps with tiny batches on TinyStories. Should complete in
under 2 minutes on an A100. If this finishes cleanly, the real
pretrain.py will work too.

Usage:
    python longleaf/scripts/py/smoke_pretrain.py
"""

import os
import sys

# Project root — go up 3 levels: py/ → scripts/ → longleaf/ → repo root
PROJECT_ROOT: str = os.path.join(os.path.dirname(__file__), "..", "..", "..")
sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader

from src import (
    CoreConfig,
    RQwen3,
    TrainConfig,
    ModelTokenizer,
    StreamingTokenDataset,
    get_device,
    print_model_summary,
    format_param_count,
)
from scripts.py.TrainSession import TrainSession


def main() -> None:
    device: torch.device = get_device()
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Same architecture as pretrain.py (Qwen3-0.6B) but shorter context
    model_config: CoreConfig = CoreConfig(
        d_model=1024,
        n_layer=28,
        num_heads=16,
        num_kv_heads=8,
        head_dim=128,
        intermediate_size=3072,
        vocab_size=151_936,
        max_seq_len=256,       # short context for speed
        dropout=0.1,
    )

    model: RQwen3 = RQwen3(model_config)
    print_model_summary(model)

    tokenizer: ModelTokenizer = ModelTokenizer("Qwen/Qwen3-0.6B")

    train_config: TrainConfig = TrainConfig(
        dataset_name="roneneldan/TinyStories",
        seq_len=256,
        batch_size=2,
        grad_accum_steps=2,
        max_steps=20,
        learning_rate=3e-4,
        min_lr=3e-5,
        warmup_steps=5,
        weight_decay=0.1,
        max_grad_norm=1.0,
        log_every=5,
        save_every=10,
        sample_every=10,
        checkpoint_dir=os.path.join(PROJECT_ROOT, "checkpoints", "smoke"),
        snapshot_dir=os.path.join(PROJECT_ROOT, "snapshots", "smoke"),
    )

    dataset: StreamingTokenDataset = StreamingTokenDataset(
        dataset_name=train_config.dataset_name,
        tokenizer=tokenizer.tokenizer,
        seq_len=train_config.seq_len,
    )
    train_dataloader: DataLoader = DataLoader(
        dataset, batch_size=train_config.batch_size,
    )

    session: TrainSession = TrainSession(
        model=model,
        config=train_config,
        tokenizer=tokenizer,
    )

    print(f"\n{'='*60}")
    print(f"  SMOKE TEST — 20 steps on TinyStories")
    print(f"  {format_param_count(sum(p.numel() for p in model.parameters()))}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    results = session.train(
        train_dataloader=train_dataloader,
        sample_prompt="Once upon a time",
    )

    print(f"\nSmoke test PASSED.")
    print(f"Final loss: {results['loss_history'][-1]:.4f}")
    print(f"Loss trend: {results['loss_history'][0]:.4f} -> {results['loss_history'][-1]:.4f}")


if __name__ == "__main__":
    main()
