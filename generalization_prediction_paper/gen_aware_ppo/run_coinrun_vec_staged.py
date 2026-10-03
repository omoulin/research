"""Staged phase 1/2/3 runner for the coinrun-vec experiment.

Trains 800 base models (phase 1). Every 50 models, retrains the
generalization predictor (phase 2) on all models collected so far and saves
a predictor-accuracy chart tagged with the checkpoint size. At the 400- and
800-model checkpoints, additionally runs phase 3 (10 baseline + 10
GeneralizationAwarePPO agents, 2M steps each) using the predictor trained on
that checkpoint, writing results to a checkpoint-scoped subdirectory so the
two phase-3 runs don't collide.

Safe to interrupt and resume:
  - phase1_collect.collect() skips models whose .zip already exists and
    reuses cached gen scores.
  - phase 3 skips any agent whose gen_curve_*.npy already exists.

Usage:
    python run_coinrun_vec_staged.py
"""
from __future__ import annotations

import copy
import os

from config import Config
from phase1_collect import collect
from phase2_predictor import train_predictor, load_predictor
from plot_predictor_accuracy import select_test_samples, predict, make_chart
from evaluate import evaluate_generalization
from main import resolve_device, train_baseline, train_gen_ppo, _compare

ENV_NAME = "coinrun-vec"
TOTAL_AGENTS = 800
CHECKPOINT_EVERY = 50
PHASE3_AT = (400, 800)
PHASE3_AGENTS = 10
PHASE3_TIMESTEPS = 2_000_000

CHART_DIR = "data/coinrun_vec/charts"


def run_phase3(cfg: Config, predictor, n_checkpoint: int) -> None:
    cfg3 = copy.deepcopy(cfg)
    cfg3.gen_timesteps = PHASE3_TIMESTEPS
    cfg3.n_phase3_agents = PHASE3_AGENTS
    n = cfg3.n_phase3_agents

    data_dir = os.path.dirname(cfg3.predictor_path)
    default_ts = 500_000
    if cfg3.gen_timesteps != default_ts:
        phase3_base = os.path.join(data_dir, f"{cfg3.gen_timesteps // 1_000}K_n{n_checkpoint}")
    else:
        phase3_base = os.path.join(data_dir, f"n{n_checkpoint}")
    genppo_dir = os.path.join(phase3_base, f"phase3_coef{cfg3.gen_coef:g}")
    os.makedirs(phase3_base, exist_ok=True)
    os.makedirs(genppo_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Phase 3a - baseline PPO ({cfg3.gen_timesteps:,} steps) "
          f"@ checkpoint n={n_checkpoint}  dir={phase3_base}")
    print("=" * 60)
    for i in range(n):
        path = f"{phase3_base}/gen_curve_baseline_{i:02d}.npy"
        if os.path.exists(path):
            print(f"  [baseline {i+1}/{n}]  already exists, skipping")
            continue
        print(f"\n  [baseline {i+1}/{n}]  training (seed={9000 + i * 137})")
        model, cb = train_baseline(cfg3, seed=9000 + i * 137)
        cb.save(path)
        final_score = evaluate_generalization(model, cfg3, seed=i + 100)
        print(f"  [baseline {i+1}/{n}]  done  final gen_score={final_score:.3f}")

    print(f"\n{'='*60}")
    print(f"Phase 3b - {n} GeneralizationAwarePPO agents @ checkpoint n={n_checkpoint}")
    print("=" * 60)
    for i in range(n):
        path = f"{genppo_dir}/gen_curve_genppo_{i:02d}.npy"
        if os.path.exists(path):
            print(f"  [gen-aware {i+1}/{n}]  already exists, skipping")
        else:
            print(f"\n  [gen-aware {i+1}/{n}]  training (seed={8000 + i * 137})")
            model, cb = train_gen_ppo(cfg3, predictor, seed=8000 + i * 137)
            cb.save(path)
            gen_score = evaluate_generalization(model, cfg3, seed=i + 1)
            print(f"  [gen-aware {i+1}/{n}]  done  final gen_score={gen_score:.3f}")
        _compare(phase3_base, genppo_dir, run_index=i)


def main() -> None:
    device = resolve_device("auto")
    print(f"Device: {device}  |  env={ENV_NAME}")
    os.makedirs(CHART_DIR, exist_ok=True)

    cfg = Config(env_name=ENV_NAME)

    for n in range(CHECKPOINT_EVERY, TOTAL_AGENTS + 1, CHECKPOINT_EVERY):
        print(f"\n{'#'*60}")
        print(f"# Phase 1 checkpoint: training up to {n} base models")
        print("#" * 60)
        cfg.num_base_models = n
        X, y = collect(cfg)

        print(f"\n{'='*60}")
        print(f"Phase 2 checkpoint @ n={n}")
        print("=" * 60)
        train_predictor(X, y, cfg, device=device)

        # Performance chart: predicted vs actual gen_score on a random
        # subsample of the models trained so far (same convention as
        # plot_predictor_accuracy.py's default, non-held-out mode).
        predictor_cpu, y_mean, y_std = load_predictor(cfg.predictor_path, device="cpu")
        X_test, y_test = select_test_samples(X, y, cfg, seed=0)
        y_pred = predict(X_test.astype("float32"), predictor_cpu, y_mean, y_std)
        chart_path = os.path.join(CHART_DIR, f"predictor_accuracy_n{n:03d}.pdf")
        make_chart(y_test.astype("float32"), y_pred, chart_path, env_name=ENV_NAME)

        if n in PHASE3_AT:
            predictor, _, _ = load_predictor(cfg.predictor_path, device=device)
            run_phase3(cfg, predictor, n_checkpoint=n)

    print("\nStaged run complete.")


if __name__ == "__main__":
    main()
