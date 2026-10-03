"""Main experiment runner.

Usage:
    python main.py                 # full experiment
    python main.py --quick         # fast smoke-test (small N, few timesteps)
    python main.py --skip-phase1   # reuse saved dataset (phases 2+3 only)
    python main.py --skip-phase2   # reuse saved predictor (phase 3 only)

The three phases are:
  Phase 1 – collect generalization data by training N PPO models.
  Phase 2 – train a predictor: weight features → generalization score.
  Phase 3 – train a GeneralizationAwarePPO and compare it to a baseline PPO.
"""
from __future__ import annotations

import argparse
import glob
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from config import Config
from phase1_collect import collect, reextract_features
from phase2_predictor import load_predictor, train_predictor
from phase3_gen_ppo import GeneralizationAwarePPO
from gen_eval_callback import GenEvalCallback
from env_factory import make_train_env
from evaluate import evaluate_generalization
from stable_baselines3 import PPO


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generalization-aware PPO (CoinRun / MiniGrid)")
    p.add_argument("--quick", action="store_true",
                   help="Drastically reduce all expensive constants for testing.")
    p.add_argument("--skip-phase1", action="store_true",
                   help="Skip data collection; load existing dataset.")
    p.add_argument("--skip-phase2", action="store_true",
                   help="Skip predictor training; load existing predictor.")
    p.add_argument("--skip-phase3", action="store_true",
                   help="Skip phase 3 comparison training (run phases 1+2 only).")
    p.add_argument("--num-base-models", type=int, default=None,
                   help="Override cfg.num_base_models for phase 1.")
    p.add_argument("--env", default="coinrun-vec",
                   choices=["coinrun-vec", "minigrid-simplecrossing-vec"],
                   help="Environment to run the experiment on.")
    p.add_argument("--device", default="auto",
                   help="Torch device: 'cpu', 'cuda', or 'auto'.")
    p.add_argument("--gen-coef", type=float, default=None,
                   help="Override cfg.gen_coef for phase 3.")
    p.add_argument("--gen-timesteps", type=int, default=None,
                   help="Override cfg.gen_timesteps for phase 3 (e.g. 2000000).")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def resolve_device(device_str: str) -> str:
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


def ppo_kwargs(cfg: Config) -> dict:
    return dict(
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        learning_rate=cfg.learning_rate,
        ent_coef=cfg.ent_coef,
        clip_range=cfg.clip_range,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        max_grad_norm=cfg.max_grad_norm,
        verbose=1,
    )


# --------------------------------------------------------------------------- #
# Phase 3 training helpers                                                     #
# --------------------------------------------------------------------------- #

def train_baseline(cfg: Config, seed: int) -> tuple[PPO, GenEvalCallback]:
    env = make_train_env(cfg, seed=seed)
    model = PPO(cfg.policy_type, env, seed=seed, **ppo_kwargs(cfg))
    cb = GenEvalCallback(cfg, verbose=1)
    model.learn(total_timesteps=cfg.gen_timesteps, callback=cb)
    model.env.close()
    return model, cb


def train_gen_ppo(
    cfg: Config, predictor, seed: int
) -> tuple[GeneralizationAwarePPO, GenEvalCallback]:
    env = make_train_env(cfg, seed=seed)
    model = GeneralizationAwarePPO(
        cfg.policy_type,
        env,
        predictor=predictor,
        gen_coef=cfg.gen_coef,
        seed=seed,
        **ppo_kwargs(cfg),
    )
    cb = GenEvalCallback(cfg, verbose=1)
    model.learn(total_timesteps=cfg.gen_timesteps, callback=cb)
    model.env.close()
    return model, cb


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def _compare(phase3_base: str, genppo_dir: str, run_index: int) -> None:
    """Print text stats and save a PDF chart after gen-aware run `run_index` finishes."""
    base_files = sorted(glob.glob(os.path.join(phase3_base, "gen_curve_baseline_??.npy")))
    gen_files  = sorted(glob.glob(os.path.join(genppo_dir,  "gen_curve_genppo_??.npy")))
    if not base_files or not gen_files:
        return

    def _load(files):
        return [(arr[:, 0], arr[:, 1]) for arr in (np.load(f) for f in files)]

    base_curves = _load(base_files)
    gen_curves  = _load(gen_files)

    def _interp_grid(curves, n=200):
        t_min = max(c[0][0]  for c in curves)
        t_max = min(c[0][-1] for c in curves)
        t = np.linspace(t_min, t_max, n)
        s = np.stack([np.interp(t, tc, sc) for tc, sc in curves])
        return t, s

    t_b, s_b = _interp_grid(base_curves)
    t_g, s_g = _interp_grid(gen_curves)

    b_mean, b_se = s_b.mean(0), s_b.std(0, ddof=1) / np.sqrt(len(base_curves))
    g_mean, g_se = s_g.mean(0), s_g.std(0, ddof=1) / np.sqrt(len(gen_curves))
    delta = g_mean[-1] - b_mean[-1]

    print(
        f"\n  --- comparison after gen-aware run {run_index:02d} "
        f"({len(gen_curves)}/{len(gen_files)} done) ---"
    )
    print(f"  Baseline  (n={len(base_curves):2d}): {b_mean[-1]:.3f} ± {b_se[-1]:.3f}")
    print(f"  GenPPO    (n={len(gen_curves):2d}): {g_mean[-1]:.3f} ± {g_se[-1]:.3f}   Δ={delta:+.3f}")

    # ── chart ──
    fig, ax = plt.subplots(figsize=(9, 5))
    for t, curves, s, mean, se, color, label in [
        (t_b, base_curves, s_b, b_mean, b_se, "#3a7fc1", "Baseline PPO"),
        (t_g, gen_curves,  s_g, g_mean, g_se, "#e05c42", "Gen-aware PPO"),
    ]:
        ax.fill_between(t, mean - se, mean + se, color=color, alpha=0.20, zorder=2)
        ax.plot(t, mean, color=color, linewidth=2.2,
                label=f"{label} (n={len(curves)}, mean ± SE)", zorder=3)
        ax.annotate(f"{mean[-1]:.3f}", xy=(t[-1], mean[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=9, color=color, va="center")

    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax.set_xlabel("Training timesteps", fontsize=12)
    ax.set_ylabel("Generalization score (unseen levels)", fontsize=11)
    ax.set_title(
        f"Baseline vs gen-aware PPO — after {len(gen_curves)} gen-aware run(s)\n"
        f"(Δ final = {delta:+.3f})",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()

    out = os.path.join(genppo_dir, f"comparison_after_{run_index:02d}.pdf")
    fig.savefig(out, format="pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Chart → {out}")


def main() -> None:
    args = parse_args()
    cfg = Config(env_name=args.env, quick=args.quick)
    if args.num_base_models is not None:
        cfg.num_base_models = args.num_base_models
    if args.gen_coef is not None:
        cfg.gen_coef = args.gen_coef
    if args.gen_timesteps is not None:
        cfg.gen_timesteps = args.gen_timesteps
    device = resolve_device(args.device)

    print(f"Device: {device}  |  quick={cfg.quick}  |  env={cfg.env_name}")
    print(f"Base models: {cfg.num_base_models}  |  "
          f"train steps: {cfg.train_timesteps:,}  |  "
          f"eval episodes: {cfg.n_eval_episodes}")

    # ------------------------------------------------------------------ #
    # Phase 1 – collect (weight_features, gen_score) pairs               #
    # ------------------------------------------------------------------ #
    if args.skip_phase1 or args.skip_phase2:
        if not os.path.exists(cfg.data_path):
            print("\n" + "=" * 60)
            print("Phase 1 – re-extracting features from saved models")
            print("=" * 60)
            X, y = reextract_features(cfg)
        else:
            loaded = np.load(cfg.data_path)
            X, y = loaded["X"], loaded["y"]
            print(f"\nLoaded existing dataset: X={X.shape}  y={y.shape}")
    else:
        print("\n" + "=" * 60)
        print("Phase 1 – collecting generalization data")
        print("=" * 60)
        X, y = collect(cfg)

    # ------------------------------------------------------------------ #
    # Phase 2 – train generalization predictor                           #
    # ------------------------------------------------------------------ #
    if args.skip_phase2:
        if not os.path.exists(cfg.predictor_path):
            raise FileNotFoundError(
                f"Predictor not found at {cfg.predictor_path}. "
                "Run without --skip-phase2 first."
            )
        predictor, y_mean, y_std = load_predictor(cfg.predictor_path, device=device)
        print(f"\nLoaded existing predictor from {cfg.predictor_path}")
    else:
        print("\n" + "=" * 60)
        print("Phase 2 – training generalization predictor")
        print("=" * 60)
        predictor = train_predictor(X, y, cfg, device=device)
        _, y_mean, y_std = load_predictor(cfg.predictor_path, device=device)

    if args.skip_phase3:
        print("\nPhase 3 skipped (--skip-phase3).")
        return

    # ------------------------------------------------------------------ #
    # Phase 3 – n_phase3_agents per approach                             #
    # ------------------------------------------------------------------ #
    n = cfg.n_phase3_agents
    data_dir = os.path.dirname(cfg.predictor_path)
    # Use a timestep-tagged subdir when gen_timesteps differs from default 500K
    default_ts = 500_000
    if cfg.gen_timesteps != default_ts:
        phase3_base = os.path.join(data_dir, f"{cfg.gen_timesteps // 1_000}K")
    else:
        phase3_base = data_dir
    genppo_dir = os.path.join(phase3_base, f"phase3_coef{cfg.gen_coef:g}")
    os.makedirs(phase3_base, exist_ok=True)
    os.makedirs(genppo_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"Phase 3a – baseline PPO ({cfg.gen_timesteps:,} steps, dir={phase3_base})")
    print("=" * 60)
    baseline_scores = []
    for i in range(n):
        path = f"{phase3_base}/gen_curve_baseline_{i:02d}.npy"
        if os.path.exists(path):
            curve = np.load(path)
            final_score = float(curve[-1, 1])
            baseline_scores.append(final_score)
            print(f"  [baseline {i+1}/{n}]  loaded  final gen_score={final_score:.3f}")
        else:
            print(f"\n  [baseline {i+1}/{n}]  training (seed={9000 + i * 137})")
            model, cb = train_baseline(cfg, seed=9000 + i * 137)
            cb.save(path)
            final_score = evaluate_generalization(model, cfg, seed=i + 100)
            baseline_scores.append(final_score)
            print(f"  [baseline {i+1}/{n}]  done  final gen_score={final_score:.3f}")

    print("\n" + "=" * 60)
    print(f"Phase 3b – training {n} GeneralizationAwarePPO agents")
    print("=" * 60)
    gen_scores = []
    for i in range(n):
        path = f"{genppo_dir}/gen_curve_genppo_{i:02d}.npy"
        if os.path.exists(path):
            curve = np.load(path)
            final_score = float(curve[-1, 1])
            gen_scores.append(final_score)
            print(f"  [gen-aware {i+1}/{n}]  loaded  final gen_score={final_score:.3f}")
        else:
            print(f"\n  [gen-aware {i+1}/{n}]  training (seed={8000 + i * 137})")
            model, cb = train_gen_ppo(cfg, predictor, seed=8000 + i * 137)
            cb.save(path)
            gen_scores.append(evaluate_generalization(model, cfg, seed=i + 1))
            print(f"  [gen-aware {i+1}/{n}]  done  final gen_score={gen_scores[-1]:.3f}")
        _compare(phase3_base, genppo_dir, run_index=i)

    # ------------------------------------------------------------------ #
    # Final evaluation summary                                            #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("Final evaluation summary")
    print("=" * 60)
    b_mean, b_std = float(np.mean(baseline_scores)), float(np.std(baseline_scores))
    g_mean, g_std = float(np.mean(gen_scores)), float(np.std(gen_scores))
    print(f"\n  Baseline PPO  (n={n}) :  {b_mean:.3f} +/- {b_std:.3f}")
    print(f"  GenAware PPO  (n={n}) :  {g_mean:.3f} +/- {g_std:.3f}")
    print(f"  delta mean (gen - base)  :  {g_mean - b_mean:+.3f}")

    print(f"\nCurves saved to {genppo_dir}/gen_curve_genppo_NN.npy")
    print(f"Baseline curves at {phase3_base}/gen_curve_baseline_NN.npy")


if __name__ == "__main__":
    main()
