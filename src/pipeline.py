"""End-to-end inference pipeline."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_device
from .tokenizers import ModelTokenizer


class PipeLine:
    """Chains tokenizer -> model -> decode into a single callable.

    Usage:
        # Fresh model (random weights)
        pipe = PipeLine(model=RQwen3(config), tokenizer=ModelTokenizer())

        # Load trained checkpoint
        pipe = PipeLine.from_checkpoint("checkpoints/step_5000.pt", model=RQwen3(config))

        # Generate text
        pipe("Once upon a time")                           # greedy
        pipe("Once upon a time", temperature=0.8)          # sampling with temperature
        pipe("Once upon a time", top_k=50, temperature=0.9)  # top-k sampling
    """

    def __init__(self, model: nn.Module, tokenizer: ModelTokenizer = None):
        self.device = get_device()
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer or ModelTokenizer()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, model: nn.Module,
                        tokenizer: ModelTokenizer = None) -> "PipeLine":
        """Load a trained model from a training checkpoint."""
        device = get_device()
        ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        pipe = cls(model=model, tokenizer=tokenizer)
        print(f"Loaded checkpoint from step {ckpt.get('step', '?')}")
        return pipe

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 100,
                 temperature: float = 1.0, top_k: int = 0,
                 store_attn: bool = False) -> str:
        """Generate text from a prompt.

        Args:
            prompt: input text
            max_new_tokens: how many tokens to generate
            temperature: >1.0 = more random, <1.0 = more focused, 1.0 = raw probabilities
            top_k: if >0, only sample from the top-k most likely tokens
            store_attn: whether to save attention weights for visualization

        Returns:
            Generated text string.
        """
        ids = self.tokenizer.encode(prompt, device=self.device)

        for _ in range(max_new_tokens):
            logits = self.model(ids, store_attn=store_attn)
            next_logits = logits[:, -1, :]  # only care about last position

            if temperature == 0 or (temperature == 1.0 and top_k == 0):
                # Greedy decoding (fastest, deterministic)
                next_id = next_logits.argmax(dim=-1, keepdim=True)
            else:
                # Temperature scaling
                next_logits = next_logits / temperature
                # Top-k filtering
                if top_k > 0:
                    top_values, _ = torch.topk(next_logits, top_k)
                    next_logits[next_logits < top_values[:, -1:]] = float('-inf')
                # Sample from distribution
                probs = F.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            ids = torch.cat([ids, next_id], dim=1)
            if next_id.item() == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(ids[0])

    def __call__(self, prompt: str, **kwargs: object) -> str:
        """Shorthand: pipe("text") calls pipe.generate("text")."""
        return self.generate(prompt, **kwargs)
