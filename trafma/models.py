"""Common two-head GRU predictor for queue metres and congestion class."""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .features import FEATURE_DIM, SEQUENCE_LENGTH


class GRUPredictor(nn.Module):
    """Encode a ten-step common feature history into queue ratio and class logits."""

    def __init__(
        self,
        input_size: int = FEATURE_DIM,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        shared_size: int = 16,
    ):
        super().__init__()
        if input_size != FEATURE_DIM:
            raise ValueError(f"input_size must match the frozen feature schema ({FEATURE_DIM})")
        if hidden_size < 1 or num_layers < 1 or shared_size < 1:
            raise ValueError("hidden_size, num_layers and shared_size must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.shared_size = shared_size
        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.shared = nn.Sequential(nn.Linear(hidden_size, shared_size), nn.ReLU())
        self.queue_head = nn.Linear(shared_size, 1)
        self.class_head = nn.Linear(shared_size, 3)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.ndim != 3 or features.shape[1:] != (SEQUENCE_LENGTH, self.input_size):
            raise ValueError(
                f"expected input [batch, {SEQUENCE_LENGTH}, {self.input_size}], got {tuple(features.shape)}"
            )
        encoded, _ = self.encoder(features)
        shared = self.shared(encoded[:, -1, :])
        queue_ratio = torch.sigmoid(self.queue_head(shared).squeeze(-1))
        return {"queue_ratio": queue_ratio, "class_logits": self.class_head(shared)}


def multitask_loss(
    outputs: dict[str, torch.Tensor],
    queue_ratio: torch.Tensor,
    congestion_class: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    regression_weight: float = 1.0,
    classification_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if regression_weight < 0 or classification_weight < 0:
        raise ValueError("loss weights must be nonnegative")
    if regression_weight == 0 and classification_weight == 0:
        raise ValueError("at least one task loss weight must be positive")
    regression = F.smooth_l1_loss(outputs["queue_ratio"], queue_ratio)
    classification = F.cross_entropy(outputs["class_logits"], congestion_class, weight=class_weights)
    total = regression_weight * regression + classification_weight * classification
    return total, {"regression": regression, "classification": classification}
