"""TrainSession — stateful training session that wraps src/ utilities.

Usage:
    session = TrainSession(model, config, tokenizer)
    results = session.train(train_dataloader)

Or build a custom loop:
    for step in range(100):
        loss = session.train_step(data_iter)
        print(f"step {step}: loss={loss:.4f}")

Run directly for a test:
    python3 scripts/py/TrainSession.py
"""

import contextlib
import os
import sys
from typing import Any, Iterator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root so `from src import ...` works when run as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src import (
    CoreConfig,
    RQwen3,
    TrainConfig,
    ModelTokenizer,
    StreamingTokenDataset,
    create_optimizer,
    create_scheduler,
    compute_loss,
    evaluate as _evaluate,
    save_checkpoint as _save_checkpoint,
    load_checkpoint as _load_checkpoint,
    save_snapshot as _save_snapshot,
    generate_sample as _generate_sample,
    get_device,
    print_model_summary,
)


class TrainSession:
    """Stateful training session manager.

    Wraps a model with its optimizer, scheduler, and training state into one
    object. Delegates to the proven utilities in src/training.py rather than
    reimplementing them.

    Key features over the old version:
      - Proper weight-decay groups (no decay on norms/biases)
      - Cosine LR schedule with linear warmup
      - Gradient accumulation (simulate larger batch sizes)
      - Checkpointing and snapshot saving
      - Evaluation and sample generation
      - Composable: use train() for the full loop, or train_step() for custom loops
    """

    # -- Type annotations for instance attributes --
    model: nn.Module
    config: TrainConfig
    tokenizer: ModelTokenizer | None
    device: torch.device
    optimizer: torch.optim.AdamW
    scheduler: torch.optim.lr_scheduler.LRScheduler
    step: int
    loss_history: list[float]

    def __init__(
        self,
        model: nn.Module,
        config: TrainConfig,
        tokenizer: ModelTokenizer | None = None,
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer

        # Device setup — use MPS on Apple Silicon, CUDA if available, else CPU
        self.device = get_device()
        self.model = model.to(self.device)

        # Optimizer with smart weight-decay groups:
        # large weight matrices get decay, biases/norms don't
        self.optimizer = create_optimizer(model, config)

        # Cosine LR schedule with linear warmup to avoid blowing up early gradients
        self.scheduler = create_scheduler(self.optimizer, config)

        # Training state
        self.step: int = 0
        self.loss_history: list[float] = []
        self.data_offset: int = 0

        # Ensure output directories exist
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.snapshot_dir, exist_ok=True)

        self.model.train()

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def info(self) -> dict[str, Any]:
        """Summary of the session: model, device, step, config."""
        total: int = sum(p.numel() for p in self.model.parameters())
        trainable: int = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            "model_type": type(self.model).__name__,
            "total_params": total,
            "trainable_params": trainable,
            "device": str(self.device),
            "step": self.step,
            "learning_rate": self.config.learning_rate,
            "effective_batch_size": self.effective_batch_size,
        }

    @property
    def current_lr(self) -> float:
        """Current learning rate from the scheduler."""
        return self.scheduler.get_last_lr()[0]

    @property
    def effective_batch_size(self) -> int:
        """batch_size * grad_accum_steps — the true number of examples per optimizer step."""
        return self.config.batch_size * self.config.grad_accum_steps

    # ── Core Training ───────────────────────────────────────────────────

    def train_step(self, data_iter: Iterator[tuple[torch.Tensor, torch.Tensor]]) -> float:
        """Execute one optimizer step with gradient accumulation.

        Gradient accumulation lets you simulate a larger batch without needing
        more memory. Instead of one big batch, we process N micro-batches,
        accumulate their gradients, then do a single optimizer step.

        Args:
            data_iter: iterator yielding (input_ids, labels) tensors

        Returns:
            The accumulated loss for this step.
        """
        self.model.train()
        accumulated_loss: float = 0.0

        # bf16 autocast on CUDA: cuts activation memory ~2x and uses BF16
        # tensor cores. No-op on MPS/CPU (preserves local smoke-test behavior).
        # No GradScaler needed because bf16 has fp32-equivalent dynamic range.
        use_amp = self.device.type == 'cuda'
        amp_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if use_amp \
                  else contextlib.nullcontext()

        # Process grad_accum_steps micro-batches before updating weights
        for _ in range(self.config.grad_accum_steps):
            input_ids, labels = next(data_iter)
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)

            with amp_ctx:
                logits: torch.Tensor = self.model(input_ids)
                # Divide loss by accum steps so the total gradient magnitude
                # is the same regardless of how many micro-batches we use
                loss: torch.Tensor = compute_loss(logits, labels) / self.config.grad_accum_steps
            loss.backward()
            accumulated_loss += loss.item()

        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

        # Update weights, advance LR schedule, clear gradients
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        self.step += 1
        self.data_offset += self.effective_batch_size
        self.loss_history.append(accumulated_loss)
        return accumulated_loss

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
        sample_prompt: str = "Once upon a time",
    ) -> dict[str, Any]:
        """Run the full training loop for config.max_steps.

        Includes logging, evaluation, sample generation, and checkpointing —
        everything you need for a complete training run.

        Args:
            train_dataloader: yields (input_ids, labels) batches
            val_dataloader: optional validation data for overfitting detection
            sample_prompt: text prompt for monitoring what the model generates

        Returns:
            dict with model, optimizer, loss_history, step for further use.
        """
        print(f"Training for {self.config.max_steps} steps "
              f"(batch={self.config.batch_size}, accum={self.config.grad_accum_steps}, "
              f"effective batch={self.effective_batch_size})")
        print(f"LR: {self.config.learning_rate} -> cosine decay, "
              f"warmup: {self.config.warmup_steps} steps")

        data_iter: Iterator = iter(train_dataloader)
        self.optimizer.zero_grad()

        while self.step < self.config.max_steps:
            # Re-create iterator if the dataset is exhausted
            try:
                loss: float = self.train_step(data_iter)
            except StopIteration:
                data_iter = iter(train_dataloader)
                loss = self.train_step(data_iter)

            # Logging
            if self.step % self.config.log_every == 0:
                lr: float = self.current_lr
                msg: str = (f"  step {self.step:>6d}/{self.config.max_steps} "
                            f"| loss: {loss:.4f} | lr: {lr:.2e}")
                if val_dataloader is not None:
                    val_loss: float = self.evaluate(val_dataloader)
                    msg += f" | val_loss: {val_loss:.4f}"
                print(msg)

            # Sample generation — see what the model outputs as it learns
            if self.tokenizer and self.step % self.config.sample_every == 0:
                sample: str = self.generate_sample(sample_prompt)
                print(f"  [sample @ step {self.step}]: {sample[:200]}")

            # Checkpointing — save so you can resume if training crashes
            if self.step % self.config.save_every == 0:
                self.save_checkpoint()

        print("Training complete.")
        return {
            "model": self.model,
            "optimizer": self.optimizer,
            "loss_history": self.loss_history,
            "step": self.step,
        }

    # ── Evaluation & Generation ─────────────────────────────────────────

    def evaluate(self, dataloader: DataLoader, max_batches: int = 50) -> float:
        """Compute average validation loss.

        Training loss going down + val loss down = real learning.
        Training loss going down + val loss UP  = overfitting.
        """
        return _evaluate(self.model, dataloader, self.device, max_batches)

    def generate_sample(self, prompt: str, max_new_tokens: int = 60) -> str:
        """Generate text from a prompt to monitor training progress."""
        if self.tokenizer is None:
            return "(no tokenizer — skipping sample generation)"
        # generate_sample expects the raw HF tokenizer, not our ModelTokenizer wrapper
        return _generate_sample(self.model, self.tokenizer.tokenizer, prompt, max_new_tokens)

    # ── Persistence ─────────────────────────────────────────────────────

    def save_checkpoint(self, tag: str | None = None) -> str:
        """Save full training state (model, optimizer, scheduler, step, loss history).

        Args:
            tag: optional name for the checkpoint file. Defaults to "step_{N}".

        Returns:
            Path to the saved checkpoint.
        """
        name: str = tag or f"step_{self.step}"
        path: str = os.path.join(self.config.checkpoint_dir, f"{name}.pt")
        _save_checkpoint(self.model, self.optimizer, self.scheduler,
                         self.step, self.loss_history, path,
                         data_offset=self.data_offset)
        return path

    def load_checkpoint(self, path: str) -> None:
        """Resume training from a previously saved checkpoint."""
        self.step, self.loss_history, self.data_offset = _load_checkpoint(
            path, self.model, self.optimizer, self.scheduler
        )

    def save_snapshot(self, name: str | None = None) -> str:
        """Save weight statistics snapshot for analysis in notebooks 03/04."""
        snapshot_name: str = name or f"step_{self.step}"
        return _save_snapshot(self.model, snapshot_name, self.config.snapshot_dir)

    # ── Class Methods ───────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        model: nn.Module,
        config: TrainConfig,
        tokenizer: ModelTokenizer | None = None,
    ) -> "TrainSession":
        """Create a session and immediately load a checkpoint to resume training."""
        session: TrainSession = cls(model, config, tokenizer)
        session.load_checkpoint(path)
        return session


# ════════════════════════════════════════════════════════════════════════
# Test run: blank Qwen3-0.6B architecture + FineWeb-Edu
# ════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Load a blank Qwen3-0.6B and run a short training test on FineWeb-Edu."""

    print("=" * 60)
    print("  TrainSession Test Run")
    print("  Model:   RQwen3 (Qwen3-0.6B dimensions, random init)")
    print("  Dataset: FineWeb-Edu (streaming)")
    print("=" * 60)

    # 1. Model config matching Qwen3-0.6B architecture
    model_config: CoreConfig = CoreConfig(
        d_model=1024,            # Qwen3-0.6B hidden_size
        n_layer=28,              # 28 transformer blocks
        num_heads=16,            # 16 query heads
        num_kv_heads=8,          # 8 KV heads (GQA, 2:1 ratio)
        head_dim=128,            # per-head dimension
        intermediate_size=3072,  # SwiGLU intermediate dim
        vocab_size=151_936,      # Qwen3 tokenizer vocabulary
        max_seq_len=256,         # short for testing (real: 40960)
        dropout=0.1,
    )

    # 2. Instantiate the blank (randomly initialized) model
    MODEL: RQwen3 = RQwen3(model_config)
    print_model_summary(MODEL)

    # 3. Load tokenizer (downloads once, then cached in ~/.cache/huggingface)
    tokenizer: ModelTokenizer = ModelTokenizer("Qwen/Qwen3-0.6B")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # 4. Training config — small for testing
    train_config: TrainConfig = TrainConfig(
        # ── Data ──
        dataset_name="HuggingFaceFW/fineweb-edu",  # HuggingFace dataset to stream from
        seq_len=256,              # number of tokens per training example (context window)

        # ── Batch sizing ──
        batch_size=2,             # examples per micro-batch (limited by GPU memory)
        grad_accum_steps=2,       # micro-batches before one optimizer step (effective batch = 2*2 = 4)

        # ── Training duration ──
        max_steps=50,             # total optimizer steps (not epochs — streaming data has no fixed size)

        # ── Learning rate ──
        learning_rate=3e-4,       # peak LR after warmup (AdamW default for transformers)
        min_lr=3e-5,              # floor LR at end of cosine decay (10x below peak)
        warmup_steps=10,          # steps to linearly ramp LR from ~0 to peak (prevents early instability)

        # ── Regularization ──
        weight_decay=0.1,         # L2 penalty on large weight matrices (prevents overfitting)
        max_grad_norm=1.0,        # clip gradients above this norm (prevents exploding gradients)

        # ── Logging & saving ──
        log_every=5,              # print loss + LR every N steps
        save_every=25,            # save a full checkpoint every N steps (for resuming)
        sample_every=25,          # generate sample text every N steps (to see learning progress)

        # ── Paths (outside iCloud — large checkpoint files corrupt during sync) ──
        checkpoint_dir=os.path.expanduser("~/.cache/qwen3-analysis/checkpoints"),
        snapshot_dir=os.path.expanduser("~/.cache/qwen3-analysis/snapshots"),
    )

    # 5. Streaming dataset — tokenizes FineWeb-Edu on the fly
    # .tokenizer gives the raw HF tokenizer that StreamingTokenDataset expects
    dataset: StreamingTokenDataset = StreamingTokenDataset(
        dataset_name=train_config.dataset_name,
        tokenizer=tokenizer.tokenizer,
        seq_len=train_config.seq_len,
    )
    train_dataloader: DataLoader = DataLoader(dataset, batch_size=train_config.batch_size)

    # 6. Create session and run
    session: TrainSession = TrainSession(
        model=MODEL,
        config=train_config,
        tokenizer=tokenizer,
    )
    print(f"\nSession info: {session.info}\n")

    results: dict[str, Any] = session.train(
        train_dataloader=train_dataloader,
        sample_prompt="Once upon a time there was",
    )

    # 7. Print final results
    print(f"\nFinal step: {results['step']}")
    print(f"Final loss: {results['loss_history'][-1]:.4f}")
    print(f"Loss trend: {results['loss_history'][0]:.4f} -> {results['loss_history'][-1]:.4f}")


if __name__ == "__main__":
    main()
