"""CoreBlock — base class for all model components."""

import torch
import torch.nn as nn
from abc import abstractmethod

from .config import CoreConfig
from .utils import get_device


class CoreBlock(nn.Module):
    """Base class for all model components."""

    def __init__(self, config: CoreConfig):
        super().__init__()
        self.config = config

    @property
    def d_model(self) -> int:
        return self.config.d_model

    @property
    def device(self) -> torch.device:
        return get_device()

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        pass

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.1)

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = True
