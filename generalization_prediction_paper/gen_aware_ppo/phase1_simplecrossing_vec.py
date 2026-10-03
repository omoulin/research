"""Phase 1 – train 800 PPO agents on MiniGrid-SimpleCrossingS9N2-v0 (vector obs).

Uses the symbolic/vector representation of the environment (147-dim flat
observation) with MlpPolicy — no CNN, no pixel rendering.

Features:
  - Auto-selects CUDA when available, falls back to CPU.
  - Fully resumable: skips models whose .zip + gen score already exist.
  - Checkpoint every CHECKPOINT models: retrains predictor on all data so far
    and saves an accuracy PDF (fixed 30-model held-out test set).
  - Progress report every REPORT_EVERY seconds.

Outputs under data/simplecrossing_vec/:
  base_models/model_NNN.zip            trained policy weights
  gen_scores.npy                       generalization scores (NaN = not done)
  phase1_dataset.npz                   (X, y) arrays for all completed models
  test_set.npz                         fixed 30-model held-out test set
  predictor_after_NNN.pt               predictor checkpoint
  predictor_accuracy_after_NNN.pdf     accuracy scatter chart
"""
from __future__ import annotations

import sys
import os
import time
import datetime

# Force UTF-8 stdout so Unicode log characters work on Windows cp1252 consoles.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from config import Config
import gym_minigrid  # noqa: F401 – registers MiniGrid envs
from simplecrossing_vec_env import make_train_env, make_eval_env
from weight_features import extract_numpy
from predictor import GeneralizationPredictor

# ─── constants ───────────────────────────────────────────────────────────────
N_MODELS     = 800
CHECKPOINT   = 100          # retrain predictor every N models
N_TEST       = 30           # held-out models for accuracy charts
REPORT_EVERY = 15 * 60     # seconds between progress lines
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────────────────────────


def _ts() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_ts()}]  {msg}", flush=True)


# ─── environment / PPO ───────────────────────────────────────────────────────

def _make_ppo(cfg: Config, seed: int) -> PPO:
    env = make_train_env(cfg, seed=seed)
    return PPO(
        cfg.policy_type,          # "MlpPolicy"
        env,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        learning_rate=cfg.learning_rate,
        ent_coef=cfg.ent_coef,
        clip_range=cfg.clip_range,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        max_grad_norm=cfg.max_grad_norm,
        device=DEVICE,
        verbose=0,
        seed=seed,
    )


def _evaluate(model: PPO, cfg: Config, seed: int) -> float:
    eval_env = make_eval_env(cfg, seed=seed)
    mean_reward, _ = evaluate_policy(
        model, eval_env,
        n_eval_episodes=cfg.n_eval_episodes,
        deterministic=True,
        warn=False,
    )
    eval_env.close()
    return float(mean_reward)


# ─── predictor ───────────────────────────────────────────────────────────────

def _train_predictor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: Config,
) -> tuple[GeneralizationPredictor, float, float]:
    y_mean = float(y_train.mean())
    y_std  = float(y_train.std()) if y_train.std() > 1e-8 else 1.0
    y_norm = (y_train - y_mean) / y_std

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_norm,  dtype=torch.float32)

    dataset = TensorDataset(X_t, y_t)
    n_val   = max(1, int(len(dataset) * cfg.predictor_val_split))
    n_tr    = len(dataset) - n_val
    tr_ds, val_ds = random_split(dataset, [n_tr, n_val])

    tr_loader  = DataLoader(tr_ds,  batch_size=max(4, n_tr), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=n_val)

    model = GeneralizationPredictor(
        input_dim=X_train.shape[1],
        hidden_dim=cfg.predictor_hidden_dim,
        dropout=cfg.predictor_dropout,
    ).to(DEVICE)

    opt  = torch.optim.Adam(model.parameters(), lr=cfg.predictor_lr,
                            weight_decay=cfg.predictor_weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.predictor_epochs)
    crit  = nn.MSELoss()
    best_val = float("inf")
    best_st  = None

    for epoch in range(cfg.predictor_epochs):
        model.train()
        tr_loss = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = crit(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= n_tr

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_loss += crit(model(xb), yb).item() * len(xb)
        val_loss /= n_val
        sched.step()

        if val_loss < best_val:
            best_val = val_loss
            best_st  = {k: v.clone() for k, v in model.state_dict().items()}

    if best_st is not None:
        model.load_state_dict(best_st)
    model.eval()
    return model, y_mean, y_std


def _save_predictor(
    model: GeneralizationPredictor,
    y_mean: float,
    y_std: float,
    path: str,
    cfg: Config,
) -> None:
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim":  model.input_dim,
        "hidden_dim": cfg.predictor_hidden_dim,
        "dropout":    cfg.predictor_dropout,
        "y_mean":     y_mean,
        "y_std":      y_std,
    }, path)


# ─── accuracy chart ──────────────────────────────────────────────────────────

def _accuracy_chart(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str,
    n_train: int,
) -> None:
    r, _ = stats.pearsonr(y_true, y_pred)
    r2   = r ** 2
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae  = float(np.mean(np.abs(y_true - y_pred)))

    fig, ax = plt.subplots(figsize=(6, 6))
    lo  = min(y_true.min(), y_pred.min())
    hi  = max(y_true.max(), y_pred.max())
    mg  = (hi - lo) * 0.05
    lim = (lo - mg, hi + mg)
    ax.plot(lim, lim, color="#aaaaaa", linewidth=1.2, linestyle="--", zorder=1)
    sl, ic, *_ = stats.linregress(y_true, y_pred)
    xf = np.linspace(lim[0], lim[1], 200)
    ax.plot(xf, sl * xf + ic, color="#e05c42", linewidth=1.4, zorder=2, alpha=0.85)
    ax.scatter(y_true, y_pred, s=55, alpha=0.75,
               edgecolors="white", linewidths=0.5, color="#3a7fc1", zorder=3)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Real generalization score", fontsize=12)
    ax.set_ylabel("Predicted generalization score", fontsize=12)
    ax.set_title(
        f"Predictor accuracy — SimpleCrossingS9N2 (vector obs)\n"
        f"(trained on {n_train} models, tested on {len(y_true)})",
        fontsize=12, fontweight="bold",
    )
    ax.text(0.04, 0.96,
            f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE  = {mae:.3f}\n"
            f"Pearson $r$ = {r:.3f}",
            transform=ax.transAxes, verticalalignment="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9))
    ax.legend(handles=[
        Line2D([0], [0], color="#aaaaaa", linestyle="--", linewidth=1.2,
               label="Perfect prediction (y = x)"),
        Line2D([0], [0], color="#e05c42", linewidth=1.4,
               label="Linear regression fit"),
    ], fontsize=9, loc="lower right")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    _log(f"  Chart saved -> {output_path}  R²={r2:.3f}  RMSE={rmse:.3f}  r={r:.3f}")


# ─── predictor checkpoint ────────────────────────────────────────────────────

def _snapshot(
    model_indices: list[int],
    X_all: dict[int, np.ndarray],
    y_all: dict[int, float],
    test_indices: set[int],
    n_done: int,
    cfg: Config,
    data_dir: str,
) -> None:
    train_idx = [i for i in model_indices if i not in test_indices]
    test_idx  = [i for i in model_indices if i in test_indices]

    if len(train_idx) < 10:
        _log(f"  Snapshot skipped — only {len(train_idx)} training samples.")
        return

    X_train = np.stack([X_all[i] for i in train_idx]).astype(np.float32)
    y_train = np.array([y_all[i] for i in train_idx], dtype=np.float32)
    X_test  = np.stack([X_all[i] for i in test_idx]).astype(np.float32)
    y_test  = np.array([y_all[i] for i in test_idx], dtype=np.float32)

    _log(f"  Training predictor on {len(train_idx)} models …")
    pred_model, y_mean, y_std = _train_predictor(X_train, y_train, cfg)

    pred_path  = os.path.join(data_dir, f"predictor_after_{n_done:03d}.pt")
    chart_path = os.path.join(data_dir, f"predictor_accuracy_after_{n_done:03d}.pdf")

    _save_predictor(pred_model, y_mean, y_std, pred_path, cfg)
    _log(f"  Predictor saved -> {pred_path}")

    with torch.no_grad():
        x_t    = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        y_pred = pred_model(x_t).cpu().numpy().squeeze() * y_std + y_mean

    _accuracy_chart(y_test, y_pred, chart_path, n_train=len(train_idx))

    # overwrite the canonical predictor.pt used by phases 2/3 via main.py
    _save_predictor(pred_model, y_mean, y_std, cfg.predictor_path, cfg)


# ─── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    _log(f"Device: {DEVICE}")

    cfg = Config(env_name="minigrid-simplecrossing-vec")
    cfg.num_base_models = N_MODELS

    data_dir  = os.path.dirname(cfg.data_path)   # data/simplecrossing_vec
    model_dir = cfg.base_model_dir
    os.makedirs(model_dir, exist_ok=True)

    _log(f"Experiment: {N_MODELS} models on MiniGrid-SimpleCrossingS9N2-v0 (vector obs)")
    _log(f"Train steps/model: {cfg.train_timesteps:,}  |  "
         f"n_envs: {cfg.n_envs}  |  eval episodes: {cfg.n_eval_episodes}")
    _log(f"Checkpoint every {CHECKPOINT} models  |  "
         f"Progress report every {REPORT_EVERY//60} min")

    # ── load / init gen scores ──────────────────────────────────────────────
    if os.path.exists(cfg.gen_scores_path):
        saved  = np.load(cfg.gen_scores_path)
        scores = np.full(N_MODELS, np.nan)
        scores[:min(len(saved), N_MODELS)] = saved[:N_MODELS]
        n_already = int(np.sum(~np.isnan(scores)))
        _log(f"Resuming — {n_already}/{N_MODELS} models already done.")
    else:
        scores = np.full(N_MODELS, np.nan)
        _log("Starting fresh.")

    # ── re-extract features for already-done models ─────────────────────────
    X_all: dict[int, np.ndarray] = {}
    y_all: dict[int, float]      = {}

    done_before = [
        i for i in range(N_MODELS)
        if not np.isnan(scores[i]) and
        os.path.exists(os.path.join(model_dir, f"model_{i:03d}.zip"))
    ]

    if done_before:
        _log(f"Re-extracting features for {len(done_before)} existing models …")
        dummy_cfg        = Config(env_name="minigrid-simplecrossing-vec")
        dummy_cfg.n_envs = 1
        dummy_env        = make_train_env(dummy_cfg, seed=0)
        for i in done_before:
            mp       = os.path.join(model_dir, f"model_{i:03d}.zip")
            m        = PPO.load(mp, env=dummy_env, device=DEVICE)
            X_all[i] = extract_numpy(m.policy)
            y_all[i] = float(scores[i])
        dummy_env.close()
        _log("  Feature re-extraction done.")

    # ── load fixed test-set if it exists ────────────────────────────────────
    test_path    = os.path.join(data_dir, "test_set.npz")
    test_indices: set[int] = set()
    if os.path.exists(test_path):
        test_indices = set(int(x) for x in np.load(test_path)["indices"])
        _log(f"Loaded fixed test set: {len(test_indices)} models.")

    # ── main training loop ──────────────────────────────────────────────────
    t_start               = time.time()
    t_last_report         = t_start
    n_done_at_last_snap   = (len(done_before) // CHECKPOINT) * CHECKPOINT

    for i in range(N_MODELS):
        # ── timed progress line ──────────────────────────────────────────────
        now = time.time()
        if now - t_last_report >= REPORT_EVERY:
            n_done  = int(np.sum(~np.isnan(scores)))
            elapsed = now - t_start
            rate    = n_done / elapsed if elapsed > 0 else 0
            eta_s   = (N_MODELS - n_done) / rate if rate > 0 else float("inf")
            eta_str = (str(datetime.timedelta(seconds=int(eta_s)))
                       if eta_s < 1e6 else "unknown")
            done_y  = [y_all[j] for j in sorted(y_all)]
            gen_str = (f"gen_mean={np.mean(done_y):.3f}  gen_std={np.std(done_y):.3f}"
                       if done_y else "no gen scores yet")
            _log(f"PROGRESS  model {i}/{N_MODELS}  done={n_done}  "
                 f"elapsed={str(datetime.timedelta(seconds=int(elapsed)))}  "
                 f"ETA={eta_str}  {gen_str}")
            t_last_report = now

        if i in X_all:
            continue

        seed       = i * 137
        model_path = os.path.join(model_dir, f"model_{i:03d}.zip")

        # ── train ────────────────────────────────────────────────────────────
        if os.path.exists(model_path):
            dummy_cfg        = Config(env_name="minigrid-simplecrossing-vec")
            dummy_cfg.n_envs = 1
            env   = make_train_env(dummy_cfg, seed=seed)
            model = PPO.load(model_path, env=env, device=DEVICE)
        else:
            model = _make_ppo(cfg, seed=seed)
            model.learn(total_timesteps=cfg.train_timesteps)
            model.save(model_path)

        # ── evaluate ─────────────────────────────────────────────────────────
        if np.isnan(scores[i]):
            scores[i] = _evaluate(model, cfg, seed=seed + 1)
            np.save(cfg.gen_scores_path, scores)

        X_all[i] = extract_numpy(model.policy)
        y_all[i] = float(scores[i])
        model.env.close()

        n_done = int(np.sum(~np.isnan(scores)))
        _log(f"  model {i:03d}  gen_score={scores[i]:.4f}  ({n_done}/{N_MODELS} done)")

        # ── fix test set once we have enough models ──────────────────────────
        if not test_indices and n_done >= N_TEST + 10:
            rng          = np.random.default_rng(42)
            done_ids     = sorted(y_all.keys())
            test_indices = set(int(x) for x in rng.choice(done_ids, size=N_TEST, replace=False))
            np.savez(test_path, indices=np.array(sorted(test_indices)))
            _log(f"Fixed test set saved ({len(test_indices)} models) -> {test_path}")

        # ── checkpoint ───────────────────────────────────────────────────────
        if n_done % CHECKPOINT == 0 and n_done > n_done_at_last_snap:
            n_done_at_last_snap = n_done
            _log(f"=== Checkpoint at {n_done} models ===")
            all_ids = sorted(y_all.keys())
            X_arr   = np.stack([X_all[j] for j in all_ids]).astype(np.float32)
            y_arr   = np.array([y_all[j] for j in all_ids], dtype=np.float32)
            np.savez(cfg.data_path, X=X_arr, y=y_arr)
            _log(f"  Dataset saved ({len(all_ids)} models) -> {cfg.data_path}")
            if test_indices:
                _snapshot(
                    model_indices=all_ids,
                    X_all=X_all,
                    y_all=y_all,
                    test_indices=test_indices,
                    n_done=n_done,
                    cfg=cfg,
                    data_dir=data_dir,
                )
            else:
                _log("  Test set not yet fixed — skipping predictor snapshot.")

    # ── final dataset save ───────────────────────────────────────────────────
    all_ids = sorted(y_all.keys())
    X_arr   = np.stack([X_all[j] for j in all_ids]).astype(np.float32)
    y_arr   = np.array([y_all[j] for j in all_ids], dtype=np.float32)
    np.savez(cfg.data_path, X=X_arr, y=y_arr)

    done_scores = [y_all[j] for j in all_ids]
    _log(
        f"\nPhase 1 complete: {len(all_ids)} models  "
        f"gen_mean={np.mean(done_scores):.3f}  "
        f"gen_std={np.std(done_scores):.3f}  "
        f"gen_range=[{np.min(done_scores):.3f}, {np.max(done_scores):.3f}]"
    )
    _log(f"Dataset -> {cfg.data_path}")
    _log("Run phases 2+3 with:  python main.py --env minigrid-simplecrossing-vec --skip-phase1")


if __name__ == "__main__":
    main()
