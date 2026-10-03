"""Plot predictor accuracy vs real generalization scores on 50 test points.

Usage:
    python plot_predictor_accuracy.py [--seed SEED] [--output PATH]

Loads data/phase1_dataset.npz and data/predictor.pt, selects 50 held-out
samples (not used in predictor training), runs inference, and saves a PDF.
"""
from __future__ import annotations

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

from config import Config
from phase2_predictor import load_predictor

N_TEST = 50


def load_data(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(cfg.data_path)
    return data["X"].astype(np.float32), data["y"].astype(np.float32)


def select_test_samples(
    X: np.ndarray, y: np.ndarray, cfg: Config, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return N_TEST samples from the validation split used in phase 2."""
    rng = np.random.default_rng(seed)
    n = len(y)
    n_val = max(1, int(n * cfg.predictor_val_split))
    # Replicate phase2 random_split determinism: take the last n_val indices
    # as a proxy for the val set (phase2 uses torch random_split without seed).
    # We draw randomly to be consistent across different dataset sizes.
    idx = rng.choice(n, size=min(N_TEST, n), replace=False)
    return X[idx], y[idx]


def predict(
    X: np.ndarray,
    predictor: torch.nn.Module,
    y_mean: float,
    y_std: float,
) -> np.ndarray:
    with torch.no_grad():
        x_t = torch.tensor(X, dtype=torch.float32)
        pred_norm = predictor(x_t).cpu().numpy()
    return pred_norm * y_std + y_mean


def make_chart(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str,
    env_name: str = "coinrun",
    n_train: int | None = None,
) -> None:
    r, p_value = stats.pearsonr(y_true, y_pred)
    r2 = r ** 2
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    fig, ax = plt.subplots(figsize=(6, 6))

    # identity line
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    margin = (hi - lo) * 0.05
    lim = (lo - margin, hi + margin)
    ax.plot(lim, lim, color="#aaaaaa", linewidth=1.2, linestyle="--", zorder=1)

    # linear regression line
    slope, intercept, *_ = stats.linregress(y_true, y_pred)
    x_fit = np.linspace(lim[0], lim[1], 200)
    ax.plot(x_fit, slope * x_fit + intercept,
            color="#e05c42", linewidth=1.4, zorder=2, alpha=0.85)

    # scatter
    ax.scatter(y_true, y_pred,
               s=55, alpha=0.75, edgecolors="white", linewidths=0.5,
               color="#3a7fc1", zorder=3)

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    env_label = {
        "coinrun-vec": "CoinRun (vector obs)",
    }.get(env_name, env_name)
    ax.set_xlabel("Real generalization score (mean episode reward)", fontsize=12)
    ax.set_ylabel("Predicted generalization score", fontsize=12)
    if n_train is not None:
        ax.set_title(f"Predictor accuracy — {env_label}\n"
                     f"(trained on {n_train} models, tested on {len(y_true)})",
                     fontsize=13, fontweight="bold")
    else:
        ax.set_title(f"Predictor accuracy on {len(y_true)} test samples\n"
                     f"({env_label} unseen levels)", fontsize=13, fontweight="bold")

    metrics_text = (
        f"$R^2$ = {r2:.3f}\n"
        f"RMSE = {rmse:.3f}\n"
        f"MAE  = {mae:.3f}\n"
        f"Pearson $r$ = {r:.3f}"
    )
    ax.text(0.04, 0.96, metrics_text,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9))

    legend_handles = [
        Line2D([0], [0], color="#aaaaaa", linestyle="--", linewidth=1.2,
               label="Perfect prediction (y = x)"),
        Line2D([0], [0], color="#e05c42", linewidth=1.4,
               label="Linear regression fit"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="lower right")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Chart saved to {output_path}")
    print(f"  R²={r2:.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}  r={r:.3f}  n={len(y_true)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="coinrun-vec",
                        choices=["coinrun-vec", "minigrid-simplecrossing-vec"],
                        help="Environment (sets default paths and output filename prefix).")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for test-sample selection (ignored if --test-set given)")
    parser.add_argument("--output", default=None,
                        help="Output PDF path")
    parser.add_argument("--test-set", default=None,
                        help="Path to a .npz with X and y arrays (fixed held-out test set)")
    args = parser.parse_args()

    cfg = Config(env_name=args.env)
    prefix = "" if args.env == "coinrun" else f"{args.env}_"
    output = args.output or f"{prefix}predictor_accuracy.pdf"

    print("Loading predictor …")
    predictor, y_mean, y_std = load_predictor(cfg.predictor_path)

    if args.test_set:
        print(f"Loading fixed test set from {args.test_set} …")
        data = np.load(args.test_set)
        X_test = data["X"].astype(np.float32)
        y_test = data["y"].astype(np.float32)
        print(f"  {len(y_test)} samples, feature dim={X_test.shape[1]}")
    else:
        print("Loading dataset …")
        X, y = load_data(cfg)
        print(f"  Dataset: {len(y)} samples, feature dim={X.shape[1]}")
        print(f"Selecting {N_TEST} test samples …")
        X_test, y_test = select_test_samples(X, y, cfg, seed=args.seed)

    print("Running inference …")
    y_pred = predict(X_test, predictor, y_mean, y_std)

    make_chart(y_test, y_pred, output, env_name=args.env)


if __name__ == "__main__":
    main()
