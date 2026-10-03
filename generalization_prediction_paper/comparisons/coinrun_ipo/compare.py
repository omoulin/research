"""Compare IPO (Sonar et al. 2021) against the main gen-aware PPO experiment
(and, when present, the IBAC-SNI and surprise-minimization comparison runs).

Reads (never writes) the reference curves:
    ../../gen_aware_ppo/data/coinrun_vec/2000K_n800/gen_curve_baseline_NN.npy        baseline PPO
    ../../gen_aware_ppo/data/coinrun_vec/2000K_n800/phase3_coef*/gen_curve_genppo_NN.npy
    ../coinrun_ibac_sni/results/ibac-sni/                          (if present)
    ../coinrun_surprise_min/results/sm-normal-centered/            (if present)
and this experiment's curves under results/<variant>/.

Outputs to results/charts/: comparison_3arms (baseline, GenPPO, IPO),
comparison_all_methods (+ IBAC-SNI, surprise min.), comparison_all_arms
(every GenPPO coefficient too), and summary.csv.

Metric: last GenEval point (100 deterministic episodes on unseen levels at
~2M steps), mean ± SE over agents, as in gen_aware_ppo's _compare(); plus peak,
mean over the curve and Welch's t-test vs the baseline.

Usage:
    python compare.py
    python compare.py --genppo-coef 0.02    # GenPPO arm shown in the 3-arm chart
    python compare.py --include-quick
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from config import Config

HERE = os.path.dirname(os.path.abspath(__file__))


def welch_p(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided Welch's t-test p-value (numpy only; scipy isn't in the env)."""
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    t = abs(a.mean() - b.mean()) / math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    logc = math.lgamma((df + 1) / 2) - math.lgamma(df / 2) - 0.5 * math.log(df * math.pi)
    x = np.linspace(t, t + 200.0, 400_001)
    pdf = np.exp(logc - (df + 1) / 2 * np.log1p(x ** 2 / df))
    return float(min(1.0, 2 * np.trapz(pdf, x)))


def load_curves(pattern: str):
    files = sorted(glob.glob(pattern))
    return [np.load(f) for f in files]


def on_grid(curves, n=200):
    t_min = max(c[0, 0] for c in curves)
    t_max = min(c[-1, 0] for c in curves)
    t = np.linspace(t_min, t_max, n)
    return t, np.stack([np.interp(t, c[:, 0], c[:, 1]) for c in curves])


def plot(arms, out_base: str, title: str, colors=None) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    greys = iter(plt.cm.Greys(np.linspace(0.35, 0.75, 8)))
    warm = iter(["#e05c42", "#8e44ad", "#2a9d8f", "#e9a13b", "#6c757d"])
    for k, (name, curves) in enumerate(arms):
        t, s = on_grid(curves)
        mean, se = s.mean(0), s.std(0, ddof=1) / np.sqrt(len(curves))
        if colors is not None:
            color, lw, alpha = colors[k], 2.4, 0.20
        elif name.startswith("Baseline"):
            color, lw, alpha = "#3a7fc1", 2.4, 0.20
        elif name.startswith("GenPPO"):
            color, lw, alpha = next(greys), 1.2, 0.0
        else:
            color, lw, alpha = next(warm), 2.4, 0.20
        if alpha:
            ax.fill_between(t, mean - se, mean + se, color=color, alpha=alpha)
        ax.plot(t, mean, color=color, linewidth=lw, label=f"{name} (n={len(curves)}, mean ± SE)")
        ax.annotate(f"{mean[-1]:.2f}", xy=(t[-1], mean[-1]), xytext=(6, 0),
                    textcoords="offset points", fontsize=9, color=color, va="center")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1e6:g}M"))
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Generalization score (unseen levels)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=9, loc="lower left", ncol=1 if colors is not None else 2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--genppo-coefs", nargs="*", default=None)
    p.add_argument("--include-quick", action="store_true")
    p.add_argument("--genppo-coef", default="0.1",
                   help="GenPPO coefficient shown in the 3-arm chart (0.1 = best of the sweep)")
    args = p.parse_args()
    os.chdir(HERE)
    cfg = Config()
    ref = cfg.reference_dir

    arms: list[tuple[str, list]] = []
    arms.append(("Baseline PPO (ref)", load_curves(f"{ref}/gen_curve_baseline_??.npy")))
    for d in sorted(glob.glob(f"{ref}/phase3_coef*"), key=lambda s: float(s.split("coef")[-1])):
        coef = d.split("coef")[-1]
        if args.genppo_coefs is not None and coef not in args.genppo_coefs:
            continue
        arms.append((f"GenPPO coef={coef} (ref)", load_curves(f"{d}/gen_curve_genppo_??.npy")))
    for d in sorted(glob.glob(f"{cfg.results_dir}/*/")):
        name = os.path.basename(d.rstrip("/"))
        if name == "charts" or (name.startswith("quick_") and not args.include_quick):
            continue
        curves = load_curves(f"{d}/gen_curve_*_??.npy")
        if curves:
            arms.append((name, curves))
    for label, pattern in (("ibac-sni (ref)", f"{cfg.ibac_dir}/gen_curve_ibac_??.npy"),
                           ("sm-normal-centered (ref)", f"{cfg.sm_dir}/gen_curve_sm_??.npy")):
        curves = load_curves(pattern)
        if curves:
            arms.append((label, curves))
    arms = [(n, c) for n, c in arms if c]

    base = np.array([c[-1, 1] for c in arms[0][1]])
    rows = []
    print(f"{'arm':<28}{'n':>3}  {'final (mean±SE)':>16}  {'Δ vs base':>9}  {'p (Welch)':>9}  "
          f"{'peak':>6}  {'curve mean':>10}")
    for name, curves in arms:
        final = np.array([c[-1, 1] for c in curves])
        _, s = on_grid(curves)
        se = final.std(ddof=1) / np.sqrt(len(final)) if len(final) > 1 else float("nan")
        pval = welch_p(final, base) if len(final) > 1 and name != arms[0][0] else float("nan")
        row = dict(arm=name, n=len(final), final_mean=final.mean(), final_se=se,
                   delta=final.mean() - base.mean(), p_welch=pval,
                   peak=s.mean(0).max(), curve_mean=s.mean())
        rows.append(row)
        print(f"{name:<28}{len(final):>3}  {final.mean():>8.3f} ± {se:<5.3f}  {row['delta']:>+9.3f}  "
              f"{pval:>9.3g}  {row['peak']:>6.2f}  {row['curve_mean']:>10.3f}")

    out_dir = os.path.join(cfg.results_dir, "charts")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    plot(arms, os.path.join(out_dir, "comparison_all_arms"),
         "CoinRun (200 train levels): all arms")

    by_name = dict(arms)
    base_arm = ("Baseline PPO", arms[0][1])
    gen_arm = (f"Gen-aware PPO (coef={args.genppo_coef})", by_name.get(f"GenPPO coef={args.genppo_coef} (ref)"))
    ipo_arm = ("IPO (Sonar et al. 2021, 2 level-domains)", by_name.get("ipo"))
    ibac_arm = ("IBAC-SNI (Igl et al. 2019)", by_name.get("ibac-sni (ref)"))
    sm_arm = ("Surprise min. (Chen 2020, α=1e-6, centered)", by_name.get("sm-normal-centered (ref)"))
    title = "Generalization on unseen CoinRun levels — 2M steps, 200 training levels"
    if gen_arm[1] and ipo_arm[1]:
        plot([base_arm, gen_arm, ipo_arm], os.path.join(out_dir, f"comparison_3arms_coef{args.genppo_coef}"), title,
             colors=["#3a7fc1", "#e05c42", "#e9a13b"])
        extra = [(a, col) for a, col in ((ibac_arm, "#2a9d8f"), (sm_arm, "#8e44ad")) if a[1]]
        if extra:
            plot([base_arm, gen_arm, ipo_arm] + [a for a, _ in extra],
                 os.path.join(out_dir, f"comparison_all_methods_coef{args.genppo_coef}"), title,
                 colors=["#3a7fc1", "#e05c42", "#e9a13b"] + [col for _, col in extra])
    else:
        print("[3-arm chart skipped: ipo or GenPPO curves missing]")
    print(f"\nChart + summary → {out_dir}/")


if __name__ == "__main__":
    main()
