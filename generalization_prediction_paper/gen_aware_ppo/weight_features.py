"""Extract per-layer weight statistics from a policy network.

Two variants are provided:
  - `extract_numpy`: detached, for data collection in phase 1.
  - `extract_diff`:  keeps the computation graph, for use in the PPO loss.

Features per layer (7 total), computed over all parameters of the layer
concatenated into a single flat vector:
  0: mean
  1: variance
  2: percentile   0  (min)
  3: percentile  25
  4: percentile  50  (median)
  5: percentile  75
  6: percentile 100  (max)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

_QUANTILES = torch.tensor([0.0, 0.25, 0.50, 0.75, 1.0])


def _layer_features(flat: torch.Tensor) -> list[torch.Tensor]:
    """7 statistics for a flat 1-D parameter vector of a single layer."""
    n = flat.numel()
    if n == 0:
        return []
    mean = flat.mean()
    var = flat.var(unbiased=False) if n > 1 else torch.zeros(1, device=flat.device).squeeze()
    quantiles = torch.quantile(flat, _QUANTILES.to(flat.device))
    return [mean, var] + [quantiles[i] for i in range(5)]


def extract_diff(policy: nn.Module) -> torch.Tensor:
    """Return a 1-D feature tensor with gradients w.r.t. policy parameters.

    One set of 7 statistics per leaf module (layer) that has trainable params.
    """
    all_feats: list[torch.Tensor] = []
    for module in policy.modules():
        # only leaf modules that directly own trainable parameters
        own_params = [p for p in module.parameters(recurse=False) if p.requires_grad]
        if not own_params:
            continue
        flat = torch.cat([p.reshape(-1).float() for p in own_params])
        all_feats.extend(_layer_features(flat))
    return torch.stack(all_feats)          # shape: (D,)


def extract_numpy(policy: nn.Module) -> np.ndarray:
    """Return a 1-D numpy feature vector (no gradient tracking)."""
    with torch.no_grad():
        feats = extract_diff(policy)
    return feats.cpu().numpy()


def feature_dim(policy: nn.Module) -> int:
    """Number of features produced for a given policy architecture."""
    return extract_numpy(policy).shape[0]
