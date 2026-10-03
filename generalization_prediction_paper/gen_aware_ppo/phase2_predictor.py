"""Phase 2 – train a predictor: weight features → generalization score.

The predictor is a small MLP trained with MSE loss on the dataset collected
in phase 1.  Targets are z-scored so the network output lives in ~[-3, 3].
The raw mean/std are saved alongside the predictor so that phase 3 can
interpret the raw scale (though we only need the gradient direction there).
"""
from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from config import Config
from predictor import GeneralizationPredictor


def train_predictor(
    X: np.ndarray,
    y: np.ndarray,
    cfg: Config,
    device: str = "cpu",
) -> GeneralizationPredictor:
    """Train and return a GeneralizationPredictor.

    Args:
        X: (N, D) weight features.
        y: (N,)   generalization scores.
        cfg:      experiment config.
        device:   torch device string.
    """
    os.makedirs(os.path.dirname(cfg.predictor_path), exist_ok=True)

    # ----- normalise targets ------------------------------------------------
    y_mean = float(y.mean())
    y_std = float(y.std()) if y.std() > 1e-8 else 1.0
    y_norm = (y - y_mean) / y_std

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y_norm, dtype=torch.float32)

    dataset = TensorDataset(X_t, y_t)
    n_val = max(1, int(len(dataset) * cfg.predictor_val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=max(4, n_train), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=n_val)

    # ----- model ------------------------------------------------------------
    dropout = getattr(cfg, "predictor_dropout", 0.0)
    weight_decay = getattr(cfg, "predictor_weight_decay", 0.0)
    model = GeneralizationPredictor(
        input_dim=X.shape[1],
        hidden_dim=cfg.predictor_hidden_dim,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.predictor_lr,
                                 weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.predictor_epochs
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(cfg.predictor_epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= n_val

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:4d}/{cfg.predictor_epochs}"
                  f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    # restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

    # ----- save with normalisation stats ------------------------------------
    torch.save(
        {"state_dict": model.state_dict(),
         "input_dim": X.shape[1],
         "hidden_dim": cfg.predictor_hidden_dim,
         "dropout": dropout,
         "y_mean": y_mean,
         "y_std": y_std},
        cfg.predictor_path,
    )
    print(f"\nPredictor saved to {cfg.predictor_path}  best_val_loss={best_val_loss:.4f}")
    return model


def load_predictor(path: str, device: str = "cpu") -> tuple[GeneralizationPredictor, float, float]:
    """Load a saved predictor.  Returns (model, y_mean, y_std)."""
    ckpt = torch.load(path, map_location=device)
    model = GeneralizationPredictor(
        input_dim=ckpt["input_dim"],
        hidden_dim=ckpt["hidden_dim"],
        dropout=ckpt.get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    # freeze – predictor weights must not change during phase 3
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt["y_mean"], ckpt["y_std"]
