"""Tokenizer wrappers."""

import torch
from transformers import AutoTokenizer

from .utils import get_device


class ModelTokenizer:
    """Wraps a HuggingFace tokenizer.

    This is NOT a neural network component — it's a plain Python class.
    Tokenizers don't have trainable parameters, so they should not inherit nn.Module.

    Usage:
        tok = ModelTokenizer()                             # default Qwen3-1.7B tokenizer
        tok = ModelTokenizer("Qwen/Qwen3-0.6B")           # different model's tokenizer
        ids = tok.encode("Hello world")                    # str -> tensor on device
        text = tok.decode(ids[0])                          # tensor -> str
        tok.tokenizer                                      # access raw HF tokenizer
    """

    def __init__(self, tokenizer_name: str = 'Qwen/Qwen3-1.7B'):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def encode(self, text: str, device=None) -> torch.Tensor:
        """Tokenize text and return input_ids tensor on the given device."""
        if device is None:
            device = get_device()
        return self.tokenizer(text, return_tensors='pt')['input_ids'].to(device)

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def __call__(self, text: str, device=None) -> torch.Tensor:
        """Shorthand: tok("Hello") is the same as tok.encode("Hello")."""
        return self.encode(text, device=device)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id
