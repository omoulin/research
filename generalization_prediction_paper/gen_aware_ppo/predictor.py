"""Generalization predictor: weight features → predicted mean reward."""
from __future__ import annotations

import torch
import torch.nn as nn


class GeneralizationPredictor(nn.Module):
    """Small MLP that maps weight-statistic features to a scalar reward estimate."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.0) -> None:
        super().__init__()
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim // 2, 1))
        self.net = nn.Sequential(*layers)
        self._input_dim = input_dim

    @property
    def input_dim(self) -> int:
        return self._input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (D,) or (B, D) feature tensor.
        Returns:
            Scalar or (B, 1) predicted reward.
        """
        was_1d = x.dim() == 1
        if was_1d:
            x = x.unsqueeze(0)
        out = self.net(x)           # (B, 1)
        return out.squeeze(1) if not was_1d else out.squeeze()
