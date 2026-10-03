"""Plot generalization score vs training timesteps for both Phase 3 approaches.

Reads all data/gen_curve_baseline_NN.npy and data/gen_curve_genppo_NN.npy files,
interpolates them onto a common timestep grid, and plots mean ± standard error.

Usage:
    python plot_gen_curves.py [--output PATH]
"""
from __future__ import annotations

import argparse
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


BASELINE_COLOR = "#3a7fc1"
GENPPO_COLOR   = "#e05c42"


def load_curve(path: str) -> tuple[np.ndarray, np.ndarray]:
    arr = np.load(path)          # shape (N, 2)
    return arr[:, 0], arr[:, 1]


def load_all_curves(pattern: str) -> list[tuple[np.ndarray, np.ndarray]]:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {pattern}")
    print(f"  Found {len(files)} curve files: {[f.split('/')[-1] for f in files]}")
    return [load_curve(f) for f in files]


def interpolate_to_grid(
    curves: list[tuple[np.ndarray, np.ndarray]],
    n_points: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate all curves onto a shared grid; return (t_grid, scores[n_agents, n_points])."""
    t_min = max(c[0][0]  for c in curves)
    t_max = min(c[0][-1] for c in curves)
    t_grid = np.linspace(t_min, t_max, n_points)
    scores = np.stack([np.interp(t_grid, t, s) for t, s in curves])
    return t_grid, scores


def make_chart(
    baseline_pattern: str,
    genppo_pattern: str,
    output_path: str,
    env_label: str = "CoinRun",
) -> None:
    print("Loading baseline curves …")
    baseline_curves = load_all_curves(baseline_pattern)
    print("Loading gen-aware curves …")
    genppo_curves   = load_all_curves(genppo_pattern)

    t_base, s_base = interpolate_to_grid(baseline_curves)
    t_gen,  s_gen  = interpolate_to_grid(genppo_curves)

    n_base = s_base.shape[0]
    n_gen  = s_gen.shape[0]

    mean_base = s_base.mean(axis=0)
    mean_gen  = s_gen.mean(axis=0)
    se_base   = s_base.std(axis=0, ddof=1) / np.sqrt(n_base)
    se_gen    = s_gen.std(axis=0,  ddof=1) / np.sqrt(n_gen)

    fig, ax = plt.subplots(figsize=(9, 5))

    for color, t, mean, se, label, n in [
        (BASELINE_COLOR, t_base, mean_base, se_base, "Baseline PPO",               n_base),
        (GENPPO_COLOR,   t_gen,  mean_gen,  se_gen,  "Generalization-aware PPO",   n_gen),
    ]:
        ax.fill_between(t, mean - se, mean + se, color=color, alpha=0.20, zorder=1)
        ax.plot(t, mean, color=color, linewidth=2.2,
                label=f"{label} (n={n}, mean ± SE)", zorder=2)
        ax.annotate(
            f"{mean[-1]:.2f}",
            xy=(t[-1], mean[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=9, color=color, va="center",
        )

    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax.set_xlabel("Training timesteps", fontsize=12)
    ax.set_ylabel("Generalization score\n(mean episode reward, unseen levels)", fontsize=11)
    ax.set_title(
        f"Generalization during training: baseline vs gen-aware PPO\n"
        f"({env_label} – unseen levels, shaded = ±1 SE)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)

    delta = mean_gen[-1] - mean_base[-1]
    print(f"Chart saved to {output_path}")
    print(f"  Baseline final : {mean_base[-1]:.3f} +/- {se_base[-1]:.3f}")
    print(f"  GenPPO  final  : {mean_gen[-1]:.3f} +/- {se_gen[-1]:.3f}   (delta = {delta:+.3f})")


ENV_LABELS = {
    "minigrid-simplecrossing-vec":   "MiniGrid-SimpleCrossingS9N2 (vector obs)",
}

ENV_DATA_DIRS = {
    "minigrid-simplecrossing-vec":   "data/simplecrossing_vec",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="minigrid-simplecrossing-vec",
                        choices=list(ENV_LABELS.keys()),
                        help="Environment (sets default paths and title).")
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--genppo",   default=None)
    parser.add_argument("--output",   default=None)
    args = parser.parse_args()

    data_dir = ENV_DATA_DIRS[args.env]
    prefix   = "" if args.env == "coinrun" else f"{args.env}_"
    baseline = args.baseline or f"{data_dir}/gen_curve_baseline_??.npy"
    genppo   = args.genppo   or f"{data_dir}/gen_curve_genppo_??.npy"
    output   = args.output   or f"{data_dir}/{prefix}gen_curves.pdf"

    make_chart(baseline, genppo, output, env_label=ENV_LABELS[args.env])


if __name__ == "__main__":
    main()
