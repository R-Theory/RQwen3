"""Training utilities: config, optimizer, scheduler, loss, evaluation, checkpointing, and generation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from .utils import get_device


@dataclass
class TrainConfig:
    """Training hyperparameters and paths."""
    # Data
    dataset_name: str = 'HuggingFaceFW/fineweb-edu'
    seq_len: int = 2048

    # Training
    batch_size: int = 4
    grad_accum_steps: int = 8
    max_steps: int = 10_000
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    warmup_steps: int = 500
    max_grad_norm: float = 1.0

    # Logging & checkpoints
    log_every: int = 10
    save_every: int = 1000
    sample_every: int = 500

    # Paths
    checkpoint_dir: str = '../checkpoints'
    snapshot_dir: str = '../snapshots'


def create_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.AdamW:
    """Create AdamW optimizer with proper weight-decay groups.

    Weight decay is applied to large weight matrices (attention, FFN) but NOT to:
      - 1-D parameters (biases, norm scaling weights) — these are tiny and
        decaying them hurts training stability.
      - Any parameter with 'norm' in its name.

    Returns the optimizer ready for use.
    """
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2 or 'norm' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': config.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=config.learning_rate)
    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer, config: TrainConfig) -> LRScheduler:
    """Create a cosine learning rate schedule with linear warmup.

    Phase 1 — Warmup (steps 0 → warmup_steps):
        LR ramps linearly from ~0 to peak. This prevents the randomly-initialized
        weights from getting blown up by large early gradients.

    Phase 2 — Cosine decay (warmup_steps → max_steps):
        LR follows a cosine curve from peak down to ~0. Early training makes big
        updates, later training fine-tunes with smaller steps.
    """
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps,
    )


def compute_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss for next-token prediction.

    Args:
        logits: model output, shape (batch, seq_len, vocab_size)
        labels: target token IDs, shape (batch, seq_len)

    Returns:
        Scalar loss tensor (how wrong the model's predictions were).

    Cross-entropy measures the distance between two probability distributions:
    the model's predicted distribution (softmax of logits) and the true distribution
    (a one-hot vector at the correct token). Lower = better predictions.
    """
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),   # flatten to (batch*seq_len, vocab_size)
        labels.view(-1),                     # flatten to (batch*seq_len,)
    )


@torch.no_grad()
def evaluate(model: nn.Module, dataloader, device, max_batches: int = 50) -> float:
    """Compute average validation loss over a set of batches.

    This is how you know if the model is actually learning vs. just memorizing:
    - Training loss going down + val loss going down = real learning
    - Training loss going down + val loss going UP   = overfitting (memorizing)

    Args:
        model: the model to evaluate
        dataloader: validation data (separate from training data!)
        device: torch device
        max_batches: how many batches to average over (more = more accurate but slower)

    Returns:
        Average loss as a float.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for input_ids, labels in dataloader:
        if n_batches >= max_batches:
            break
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        logits = model(input_ids)
        total_loss += compute_loss(logits, labels).item()
        n_batches += 1

    model.train()
    return total_loss / max(n_batches, 1)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LRScheduler,
    step: int,
    loss_history: list[float],
    path: str,
    data_offset: int = 0,
) -> None:
    """Save everything needed to resume training.

    Args:
        data_offset: Number of training examples consumed so far. Used to
            resume from the correct position in a PreTokenizedDataset.
    """
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss_history': loss_history,
        'data_offset': data_offset,
    }, path)
    print(f'  Checkpoint saved: {path}')


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LRScheduler,
) -> tuple[int, list[float], int]:
    """Resume training from a checkpoint.

    Returns:
        (step, loss_history, data_offset) tuple. data_offset defaults to 0
        for backward compatibility with older checkpoints.
    """
    device = get_device()
    ckpt = torch.load(path, weights_only=False, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    data_offset = ckpt.get('data_offset', 0)
    print(f'Resumed from step {ckpt["step"]} (data_offset={data_offset})')
    return ckpt['step'], ckpt['loss_history'], data_offset


def save_snapshot(model: nn.Module, name: str, snapshot_dir: str) -> str:
    """Save weight stats snapshot (compatible with notebook 03/04 comparison tools)."""
    snapshot = {}
    for pname, param in model.named_parameters():
        data = param.detach().cpu().float()
        snapshot[pname] = {
            'mean': data.mean().item(),
            'std': data.std().item(),
            'min': data.min().item(),
            'max': data.max().item(),
            'shape': list(data.shape),
            'histogram': np.histogram(data.numpy().flatten(), bins=50),
        }
    path = os.path.join(snapshot_dir, f'{name}.pt')
    torch.save(snapshot, path)
    return path


@torch.no_grad()
def generate_sample(model: nn.Module, tokenizer: Any, prompt: str, max_new_tokens: int = 60) -> str:
    """Simple greedy generation to monitor training progress."""
    device = get_device()
    model.eval()
    ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    for _ in range(max_new_tokens):
        logits = model(ids)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
        if next_id.item() == tokenizer.eos_token_id:
            break
    model.train()
    return tokenizer.decode(ids[0], skip_special_tokens=True)


def train(
    model: nn.Module,
    train_dataloader,
    train_config: TrainConfig,
    tokenizer: Any | None = None,
    val_dataloader: DataLoader | None = None,
    resume_from: str | None = None,
    sample_prompt: str = "Once upon a time",
) -> dict[str, Any]:
    """Complete training loop with gradient accumulation, logging, and checkpointing.

    This ties together all the training utilities into a single callable function.
    Call it from a notebook or script like:

        train(model, train_dl, train_config, tokenizer=tok, val_dataloader=val_dl)

    Args:
        model: the model to train (e.g., RQwen3)
        train_dataloader: yields (input_ids, labels) batches
        train_config: TrainConfig with hyperparameters
        tokenizer: HF tokenizer (for sample generation during training). Optional.
        val_dataloader: validation data. If provided, val loss is logged. Optional.
        resume_from: path to a checkpoint to resume from. Optional.
        sample_prompt: text prompt used to generate samples during training.

    Returns:
        dict with 'model', 'optimizer', 'loss_history', 'step' for further use.
    """
    device = get_device()
    model = model.to(device)
    model.train()

    optimizer = create_optimizer(model, train_config)
    scheduler = create_scheduler(optimizer, train_config)

    # Resume from checkpoint if provided
    start_step = 0
    loss_history = []
    if resume_from and os.path.exists(resume_from):
        start_step, loss_history, _data_offset = load_checkpoint(resume_from, model, optimizer, scheduler)

    # Ensure output directories exist
    os.makedirs(train_config.checkpoint_dir, exist_ok=True)
    os.makedirs(train_config.snapshot_dir, exist_ok=True)

    # Training loop
    data_iter = iter(train_dataloader)
    running_loss = 0.0
    optimizer.zero_grad()

    print(f"Training for {train_config.max_steps} steps "
          f"(batch={train_config.batch_size}, accum={train_config.grad_accum_steps}, "
          f"effective batch={train_config.batch_size * train_config.grad_accum_steps})")
    print(f"LR: {train_config.learning_rate} -> cosine decay, "
          f"warmup: {train_config.warmup_steps} steps")

    for step in range(start_step, train_config.max_steps):
        # --- Gradient accumulation ---
        for micro_step in range(train_config.grad_accum_steps):
            try:
                input_ids, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(train_dataloader)
                input_ids, labels = next(data_iter)

            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits = model(input_ids)
            loss = compute_loss(logits, labels) / train_config.grad_accum_steps
            loss.backward()
            running_loss += loss.item()

        # --- Optimizer step ---
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        loss_history.append(running_loss)

        # --- Logging ---
        if (step + 1) % train_config.log_every == 0:
            lr = scheduler.get_last_lr()[0]
            msg = f"  step {step + 1:>6d}/{train_config.max_steps} | loss: {running_loss:.4f} | lr: {lr:.2e}"
            if val_dataloader is not None:
                val_loss = evaluate(model, val_dataloader, device)
                msg += f" | val_loss: {val_loss:.4f}"
            print(msg)

        running_loss = 0.0

        # --- Sample generation ---
        if tokenizer and (step + 1) % train_config.sample_every == 0:
            sample = generate_sample(model, tokenizer, sample_prompt)
            print(f"  [sample @ step {step + 1}]: {sample[:200]}")

        # --- Checkpointing ---
        if (step + 1) % train_config.save_every == 0:
            ckpt_path = os.path.join(
                train_config.checkpoint_dir, f"step_{step + 1}.pt"
            )
            save_checkpoint(model, optimizer, scheduler, step + 1, loss_history, ckpt_path)

    print("Training complete.")
    return {
        'model': model,
        'optimizer': optimizer,
        'loss_history': loss_history,
        'step': train_config.max_steps,
    }
