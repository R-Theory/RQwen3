import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- Qwen3-0.6B Configuration ------------------------------------------------

CONFIG = {
    "d_model": 1024,
    "dtype": torch.bfloat16,
    "rms_norm_eps": 1e-6,
    "max_seq_len": 40_960,
    "vocab_size": 151_936,
    "n_layers": 28,
    "num_heads": 16,
    "num_kv_heads": 8,
    "head_dim": 128,
    "intermediate_size": 3072,
    "rope_theta": 1_000_000.0,
    "tie_word_embeddings": True,
}


# ---- Reusable Helper Functions -----------------------------------------------

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Root Mean Square Layer Normalization."""
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap and negate halves of the last dimension for RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def build_rope_cache(
    seq_len: int, head_dim: int, theta: float,
    device: torch.device, dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables for Rotary Position Embeddings."""
    inv_freq = 1.0 / (theta ** (
        torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim
    ))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)            # (seq_len, head_dim)
    cos = emb.cos().unsqueeze(0).unsqueeze(0).to(dtype) # (1, 1, seq_len, head_dim)
    sin = emb.sin().unsqueeze(0).unsqueeze(0).to(dtype)
    return cos, sin


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Rotary Position Embeddings to Q and K."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def swiglu(
    x: torch.Tensor,
    gate_proj: nn.Linear, up_proj: nn.Linear, down_proj: nn.Linear,
) -> torch.Tensor:
    """SwiGLU FFN: down(silu(gate(x)) * up(x))."""
    return down_proj(F.silu(gate_proj(x)) * up_proj(x))


# ---- Core Classes ------------------------------------------------------------

class GQA(nn.Module):
    """Grouped Query Attention with QK-Norm and RoPE."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.num_kv_groups = config["num_heads"] // config["num_kv_heads"]

        self.q_proj = nn.Linear(config["d_model"], config["num_heads"] * config["head_dim"], bias=False)
        self.k_proj = nn.Linear(config["d_model"], config["num_kv_heads"] * config["head_dim"], bias=False)
        self.v_proj = nn.Linear(config["d_model"], config["num_kv_heads"] * config["head_dim"], bias=False)
        self.o_proj = nn.Linear(config["num_heads"] * config["head_dim"], config["d_model"], bias=False)

        # QK-Norm (per-head, head_dim-dimensional)
        self.q_norm_weight = nn.Parameter(torch.ones(config["head_dim"]))
        self.k_norm_weight = nn.Parameter(torch.ones(config["head_dim"]))

        self._attn_weights = None

    def forward(self, x: torch.Tensor, store_attn: bool = False) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        cfg = self.config
        eps = cfg["rms_norm_eps"]

        # Project → reshape to (B, S, heads, head_dim) → QK-norm → transpose to (B, heads, S, head_dim)
        q = rms_norm(
            self.q_proj(x).view(batch, seq_len, cfg["num_heads"], cfg["head_dim"]),
            self.q_norm_weight, eps,
        ).transpose(1, 2)

        k = rms_norm(
            self.k_proj(x).view(batch, seq_len, cfg["num_kv_heads"], cfg["head_dim"]),
            self.k_norm_weight, eps,
        ).transpose(1, 2)

        v = self.v_proj(x).view(batch, seq_len, cfg["num_kv_heads"], cfg["head_dim"]).transpose(1, 2)

        # RoPE (Q and K only)
        cos, sin = build_rope_cache(seq_len, cfg["head_dim"], cfg["rope_theta"], x.device, q.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Expand KV heads to match Q heads
        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Scaled dot-product attention
        scale = cfg["head_dim"] ** -0.5
        attn_weights = (q @ k.transpose(-2, -1)) * scale

        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device, dtype=q.dtype),
            diagonal=1,
        )
        attn_weights = torch.softmax(attn_weights + causal_mask, dim=-1)

        if store_attn:
            self._attn_weights = attn_weights.detach().cpu()

        out = (attn_weights @ v).transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(out)


class RLModel0Transformer(nn.Module):
    """Pre-Norm Transformer Block: RMSNorm → GQA → residual, RMSNorm → SwiGLU → residual."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # Norm weights (used via rms_norm function)
        self.attn_norm_weight = nn.Parameter(torch.ones(config["d_model"]))
        self.mlp_norm_weight = nn.Parameter(torch.ones(config["d_model"]))

        # Attention
        self.attn = GQA(config)

        # SwiGLU FFN projections (used via swiglu function)
        self.gate_proj = nn.Linear(config["d_model"], config["intermediate_size"], bias=False)
        self.up_proj   = nn.Linear(config["d_model"], config["intermediate_size"], bias=False)
        self.down_proj = nn.Linear(config["intermediate_size"], config["d_model"], bias=False)

    def forward(self, x: torch.Tensor, store_attn: bool = False) -> torch.Tensor:
        eps = self.config["rms_norm_eps"]
        x = x + self.attn(rms_norm(x, self.attn_norm_weight, eps), store_attn=store_attn)
        x = x + swiglu(rms_norm(x, self.mlp_norm_weight, eps), self.gate_proj, self.up_proj, self.down_proj)
        return x


class LMHead(nn.Module):
    """Final RMSNorm + linear projection to vocab logits."""

    def __init__(self, config: dict):
        super().__init__()
        self.eps = config["rms_norm_eps"]
        self.norm_weight = nn.Parameter(torch.ones(config["d_model"]))
        self.out_proj = nn.Linear(config["d_model"], config["vocab_size"], bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(rms_norm(x, self.norm_weight, self.eps))


# ---- Top-Level Model ---------------------------------------------------------

class RLModel0(nn.Module):
    """Full language model matching the Qwen3-0.6B architecture."""

    def __init__(self, config: dict = None):
        super().__init__()
        self.config = config or CONFIG

        self.embedding = nn.Embedding(self.config["vocab_size"], self.config["d_model"])

        self.layers = nn.ModuleList([
            RLModel0Transformer(self.config) for _ in range(self.config["n_layers"])
        ])

        self.lm_head = LMHead(self.config)

        # Qwen3-0.6B ties embedding ↔ output weights
        if self.config.get("tie_word_embeddings", True):
            self.lm_head.out_proj.weight = self.embedding.weight

        self.apply(self._init_weights)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None, **kwargs):
        x = self.embedding(input_ids)

        for layer in self.layers:
            x = layer(x, **kwargs)

        logits = self.lm_head(x)

        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            return logits, loss

        return logits

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True


# ---- Smoke Test --------------------------------------------------------------

def main():
    """Instantiate RLModel0 + TrainSession, run 10 steps on random data.

    This is a quick sanity check that the model trains without crashing.
    Real training happens on Longleaf with A100 GPUs and the full 28-layer config.

    Usage:
        python3 scripts/py/RLModel0.py
    """
    import os, sys
    from torch.utils.data import DataLoader, IterableDataset

    # Add project root + scripts/py to path so imports resolve
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_here, "..", ".."))
    sys.path.insert(0, _here)

    from TrainSession import TrainSession
    from src.training import TrainConfig

    print("=" * 60)
    print("  RLModel0 Smoke Test")
    print("  2 layers, random data, 10 steps")
    print("  (Full 28-layer training → Longleaf A100)")
    print("=" * 60)

    # Shrink to 2 layers for smoke test — keeps it fast on CPU/MPS
    test_config = {**CONFIG, "n_layers": 2}
    model = RLModel0(test_config)
    print(f"\nModel: {model.num_params:,} params ({model.num_trainable_params:,} trainable)")

    # Random token data — no HuggingFace download needed for smoke test
    class RandomTokenData(IterableDataset):
        def __init__(self, vocab_size, seq_len):
            self.vocab_size = vocab_size
            self.seq_len = seq_len

        def __iter__(self):
            while True:
                tokens = torch.randint(0, self.vocab_size, (self.seq_len + 1,))
                yield tokens[:-1], tokens[1:]

    seq_len = 128
    dataloader = DataLoader(
        RandomTokenData(test_config["vocab_size"], seq_len),
        batch_size=1,
    )

    train_config = TrainConfig(
        seq_len=seq_len,
        batch_size=1,
        grad_accum_steps=1,
        max_steps=10,
        learning_rate=3e-4,
        warmup_steps=2,
        log_every=1,
        save_every=9999,    # don't checkpoint during smoke test
        sample_every=9999,  # don't generate samples (no tokenizer)
        checkpoint_dir=os.path.expanduser("~/.cache/rlmodel0/checkpoints"),
        snapshot_dir=os.path.expanduser("~/.cache/rlmodel0/snapshots"),
    )

    session = TrainSession(model=model, config=train_config)
    print(f"Device: {session.device}\n")

    results = session.train(train_dataloader=dataloader)

    print(f"\nSmoke test complete!")
    print(f"Loss: {results['loss_history'][0]:.4f} -> {results['loss_history'][-1]:.4f}")


if __name__ == "__main__":
    main()
