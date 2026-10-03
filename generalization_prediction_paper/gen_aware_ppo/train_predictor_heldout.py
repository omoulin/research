"""Train the generalization predictor on a train/test split of the full
coinrun-vec dataset, evaluate on the untouched held-out models, and save an
accuracy chart.

Does NOT overwrite the official data/coinrun_vec/predictor.pt (trained on
all models, used by phase 3) — saves to a separate path instead.

Usage:
    python train_predictor_heldout.py [--n-test 30] [--seed 0]
"""
from __future__ import annotations

import argparse
import copy
import os

import numpy as np

from config import Config
from phase2_predictor import train_predictor, load_predictor
from plot_predictor_accuracy import predict, make_chart
from main import resolve_device

ENV_NAME = "coinrun-vec"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-test", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = resolve_device("auto")
    cfg = Config(env_name=ENV_NAME)

    data = np.load(cfg.data_path)
    X, y = data["X"].astype("float32"), data["y"].astype("float32")
    n_total = len(y)
    n_train = n_total - args.n_test
    print(f"Dataset: {n_total} models  ->  train={n_train}  test={args.n_test}")

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(n_total)
    test_idx, train_idx = idx[: args.n_test], idx[args.n_test :]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    cfg_ho = copy.deepcopy(cfg)
    data_dir = os.path.dirname(cfg.predictor_path)
    cfg_ho.predictor_path = os.path.join(data_dir, f"predictor_heldout{n_train}.pt")

    print(f"\nTraining predictor on {n_train} models -> {cfg_ho.predictor_path}")
    train_predictor(X_train, y_train, cfg_ho, device=device)

    predictor_cpu, y_mean, y_std = load_predictor(cfg_ho.predictor_path, device="cpu")
    y_pred = predict(X_test, predictor_cpu, y_mean, y_std)

    chart_path = os.path.join(
        data_dir, "charts", f"predictor_accuracy_heldout{args.n_test}_train{n_train}.pdf"
    )
    make_chart(y_test, y_pred, chart_path, env_name=ENV_NAME, n_train=n_train)


if __name__ == "__main__":
    main()
