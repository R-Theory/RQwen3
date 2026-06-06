"""Pretrain RQwen3 (Qwen3-0.6B scale) on Longleaf with a single A100.

Run via SLURM:
    sbatch longleaf/pretrain.slurm

Or directly on a GPU node:
    python longleaf/pretrain.py

Supports automatic resumption: if checkpoints exist in the checkpoint
directory, training picks up from the latest one. Just re-submit the
same SLURM job after a wall-time cutoff.

Handles SIGTERM gracefully: SLURM sends SIGTERM before killing the job.
The signal handler saves an emergency checkpoint so no work is lost.
"""

import os
import sys
import signal
import glob as globmod

# Project root — go up 3 levels: py/ → scripts/ → longleaf/ → repo root
PROJECT_ROOT: str = os.path.join(os.path.dirname(__file__), "..", "..", "..")
sys.path.insert(0, PROJECT_ROOT)

from typing import Any

import torch
from torch.utils.data import DataLoader

from src import (
    CoreConfig,
    RQwen3,
    TrainConfig,
    ModelTokenizer,
    StreamingTokenDataset,
    PreTokenizedDataset,
    get_device,
    print_model_summary,
    format_param_count,
)
from scripts.py.TrainSession import TrainSession

# Path to pre-tokenized dataset (built by scripts/build_dataset.py)
LOCAL_DATA_DIR: str = "/work/users/t/r/treese20/data/rqwen3-pretrain/v1"


def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    """Find the most recent checkpoint file in the checkpoint directory.

    Looks for step_*.pt files and returns the one with the highest step number.
    Returns None if no checkpoints exist.
    """
    pattern: str = os.path.join(checkpoint_dir, "step_*.pt")
    checkpoints: list[str] = globmod.glob(pattern)
    if not checkpoints:
        return None

    def step_number(path: str) -> int:
        """Extract step number from filename like 'step_1000.pt'."""
        basename: str = os.path.basename(path)
        name: str = basename.replace(".pt", "")  # "step_1000"
        parts: list[str] = name.split("_")
        try:
            return int(parts[-1])
        except ValueError:
            return -1

    checkpoints.sort(key=step_number)
    return checkpoints[-1]


def main() -> None:

    device: torch.device = get_device()
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Model: Qwen3-0.6B blank architecture ────────────────────────
    model_config: CoreConfig = CoreConfig(
        d_model=1024,            # Qwen3-0.6B hidden_size
        n_layer=28,              # 28 transformer blocks
        num_heads=16,            # 16 query heads
        num_kv_heads=8,          # 8 KV heads (GQA, 2:1 ratio)
        head_dim=128,            # per-head dimension
        intermediate_size=3072,  # SwiGLU FFN intermediate dim
        vocab_size=151_936,      # Qwen3 tokenizer vocabulary
        max_seq_len=2048,        # full context window for pretraining
        dropout=0.1,
    )

    model: RQwen3 = RQwen3(model_config)
    print_model_summary(model)

    # ── Tokenizer ───────────────────────────────────────────────────
    tokenizer: ModelTokenizer = ModelTokenizer("Qwen/Qwen3-0.6B")

    # ── Training config for A100 ────────────────────────────────────
    #
    # Memory budget (A100 40GB PCIe — Longleaf's a100-gpu partition):
    #   Model:     ~3 GB  (751M params * 4 bytes)
    #   Gradients: ~3 GB  (same size as model)
    #   Optimizer: ~6 GB  (AdamW stores momentum + variance = 2x model)
    #   Activations: depends on batch_size * seq_len
    #   Total base: ~12 GB + activations
    #
    # With batch_size=4, seq_len=2048 on A100 40GB: fits with headroom.
    # Effective batch = 4 * 32 = 128 examples = 262,144 tokens per step.
    #
    train_config: TrainConfig = TrainConfig(
        # ── Data ──
        dataset_name="HuggingFaceFW/fineweb-edu",  # curated web text, popular for pretraining
        seq_len=2048,              # full context window

        # ── Batch sizing ──
        batch_size=4,              # micro-batch size (fits A100 40GB)
        grad_accum_steps=32,       # effective batch = 4 * 32 = 128

        # ── Training duration ──
        max_steps=10_000,          # ~1.3B tokens seen (128 * 2048 * 10000 / 2)

        # ── Learning rate ──
        learning_rate=3e-4,        # standard peak for this model scale
        min_lr=3e-5,               # 10x below peak at end of cosine decay
        warmup_steps=500,          # ramp up over first 500 steps

        # ── Regularization ──
        weight_decay=0.1,          # L2 penalty on weight matrices
        max_grad_norm=1.0,         # gradient clipping for stability

        # ── Logging & saving ──
        log_every=10,              # print loss every 10 steps
        save_every=1000,           # checkpoint every 1000 steps (~1 hour)
        sample_every=500,          # generate a sample every 500 steps

        # ── Paths (on /work, not home — home has small quota) ──
        checkpoint_dir=os.path.join(PROJECT_ROOT, "checkpoints"),
        snapshot_dir=os.path.join(PROJECT_ROOT, "snapshots"),
    )

    # ── Dataset ─────────────────────────────────────────────────────
    # Auto-detect: use pre-tokenized local data if available (built by
    # scripts/build_dataset.py), otherwise fall back to streaming.
    manifest_path: str = os.path.join(LOCAL_DATA_DIR, "manifest.json")
    use_local: bool = os.path.exists(manifest_path)

    if use_local:
        print(f"Using pre-tokenized local data: {LOCAL_DATA_DIR}")
        dataset = PreTokenizedDataset(
            data_dir=LOCAL_DATA_DIR,
            seq_len=train_config.seq_len,
            split='train',
        )
        train_dataloader: DataLoader = DataLoader(
            dataset,
            batch_size=train_config.batch_size,
            shuffle=True,        # random access — proper shuffling
            num_workers=4,
            pin_memory=True,
        )
        dataset_label = f"pre-tokenized ({len(dataset):,} chunks)"
    else:
        print("No local data found — streaming from HuggingFace")
        # Score filtering: FineWeb-Edu 'score' column (0-5) rates educational
        # quality. Filtering to score >= 3 drops low-quality pages.
        dataset = StreamingTokenDataset(
            dataset_name=train_config.dataset_name,
            tokenizer=tokenizer.tokenizer,  # raw HF tokenizer
            seq_len=train_config.seq_len,
            min_score=3,          # educational quality threshold
            min_length=100,       # drop noise/stubs
            max_length=100_000,   # drop data dumps
        )
        train_dataloader: DataLoader = DataLoader(
            dataset,
            batch_size=train_config.batch_size,
            num_workers=4,
            pin_memory=True,
        )
        dataset_label = f"{train_config.dataset_name} (score >= 3, streaming)"

    # ── Resume or start fresh ─────────────────────────────────────
    latest_ckpt: str | None = find_latest_checkpoint(train_config.checkpoint_dir)

    if latest_ckpt is not None:
        print(f"\nFound checkpoint: {latest_ckpt}")
        print("Resuming training from checkpoint...")
        session: TrainSession = TrainSession.from_checkpoint(
            path=latest_ckpt,
            model=model,
            config=train_config,
            tokenizer=tokenizer,
        )
        print(f"Resumed at step {session.step}/{train_config.max_steps}")
    else:
        print("\nNo checkpoint found — starting fresh.")
        session = TrainSession(
            model=model,
            config=train_config,
            tokenizer=tokenizer,
        )

    # ── SIGTERM handler ────────────────────────────────────────────
    # SLURM sends SIGTERM before killing the job at wall time.
    # This catches it and saves a checkpoint so you can just
    # re-submit the same job to continue where you left off.
    def handle_sigterm(signum: int, frame) -> None:
        print(f"\n{'!'*60}")
        print(f"  SIGTERM received (SLURM wall time approaching)")
        print(f"  Saving emergency checkpoint at step {session.step}...")
        print(f"{'!'*60}")
        session.save_checkpoint()
        session.save_snapshot(f"sigterm_step_{session.step}")
        print("  Emergency checkpoint saved. Exiting gracefully.")
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_sigterm)

    # ── Train ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Pretraining RQwen3 ({format_param_count(sum(p.numel() for p in model.parameters()))})")
    print(f"  Dataset: {dataset_label}")
    print(f"  Tokens per step: {train_config.batch_size * train_config.grad_accum_steps * train_config.seq_len:,}")
    print(f"  Device: {device}")
    print(f"  Starting from step: {session.step}")
    print(f"{'='*60}\n")

    results: dict[str, Any] = session.train(
        train_dataloader=train_dataloader,
        sample_prompt="The theory of general relativity",
    )

    # Save final checkpoint and snapshot
    session.save_checkpoint("final")
    session.save_snapshot("final")

    print(f"\nDone. Final loss: {results['loss_history'][-1]:.4f}")
    print(f"Checkpoints in: {train_config.checkpoint_dir}")


if __name__ == "__main__":
    main()
