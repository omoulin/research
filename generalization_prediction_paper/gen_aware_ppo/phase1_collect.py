"""Phase 1 – train base PPO models and record their generalization scores.

For each model we:
  1. Train a PPO agent on cfg.train_num_levels CoinRun levels.
  2. Evaluate on cfg.n_eval_episodes *unseen* levels → scalar gen_score.
  3. Extract weight features from the trained policy.

Gen scores are saved incrementally to cfg.gen_scores_path so they survive
crashes and never need to be recomputed, even if the feature definition changes.
The resulting (features, gen_score) dataset is saved to cfg.data_path.
"""
from __future__ import annotations

import os
import numpy as np
from tqdm import tqdm
from stable_baselines3 import PPO

from config import Config
from env_factory import make_train_env
from evaluate import evaluate_generalization
from weight_features import extract_numpy


def _make_ppo(cfg: Config, seed: int) -> PPO:
    env = make_train_env(cfg, seed=seed)
    model = PPO(
        cfg.policy_type,
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
        verbose=0,
        seed=seed,
    )
    return model


def _load_saved_scores(path: str, n: int) -> np.ndarray:
    """Load gen scores array of length n, filling missing entries with NaN."""
    scores = np.full(n, np.nan)
    if os.path.exists(path):
        saved = np.load(path)
        scores[:len(saved)] = saved[:n]
    return scores


def _save_scores(path: str, scores: np.ndarray) -> None:
    np.save(path, scores)


def reextract_features(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild the dataset from saved models and cached gen scores.

    Feature extraction is recomputed from model weights (fast).
    Gen scores are loaded from cfg.gen_scores_path (no env rollouts needed).
    """
    if not os.path.exists(cfg.gen_scores_path):
        raise FileNotFoundError(
            f"Gen scores not found at {cfg.gen_scores_path}. "
            "Run phase 1 training first."
        )
    y = np.load(cfg.gen_scores_path)
    if len(y) < cfg.num_base_models or np.any(np.isnan(y[:cfg.num_base_models])):
        raise ValueError(
            f"Gen scores incomplete ({len(y)}/{cfg.num_base_models}). "
            "Run phase 1 training to completion first."
        )
    y = y[:cfg.num_base_models]

    X_list: list[np.ndarray] = []
    for i in tqdm(range(cfg.num_base_models), desc="Re-extracting features"):
        model_path = os.path.join(cfg.base_model_dir, f"model_{i:03d}.zip")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}. Run phase 1 training first."
            )
        seed = i * 137
        env = make_train_env(cfg, seed=seed)
        model = PPO.load(model_path, env=env)
        X_list.append(extract_numpy(model.policy))
        model.env.close()

    X = np.stack(X_list)
    np.savez(cfg.data_path, X=X, y=y)
    print(f"\nDataset saved to {cfg.data_path}  shape X={X.shape}  y={y.shape}")
    return X, y


def collect(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Train base models and return (X, y) arrays.

    X: (num_base_models, D)  weight features
    y: (num_base_models,)    generalization scores (mean episode reward)

    Resumes automatically: if a saved model file exists for index i, it is
    loaded instead of re-trained; if its gen score is cached it is reused too.
    """
    os.makedirs(os.path.dirname(cfg.data_path), exist_ok=True)
    os.makedirs(cfg.base_model_dir, exist_ok=True)

    scores = _load_saved_scores(cfg.gen_scores_path, cfg.num_base_models)

    X_list: list[np.ndarray] = []
    y_list: list[float] = []

    for i in tqdm(range(cfg.num_base_models), desc="Phase 1 – training base models"):
        seed = i * 137
        model_path = os.path.join(cfg.base_model_dir, f"model_{i:03d}.zip")

        if os.path.exists(model_path):
            env = make_train_env(cfg, seed=seed)
            model = PPO.load(model_path, env=env)
        else:
            model = _make_ppo(cfg, seed=seed)
            model.learn(total_timesteps=cfg.train_timesteps)
            model.save(os.path.join(cfg.base_model_dir, f"model_{i:03d}"))

        if np.isnan(scores[i]):
            scores[i] = evaluate_generalization(model, cfg, seed=seed + 1)
            _save_scores(cfg.gen_scores_path, scores)

        X_list.append(extract_numpy(model.policy))
        y_list.append(float(scores[i]))
        model.env.close()

        print(f"  model {i:02d}  gen_score={scores[i]:.3f}")

    X = np.stack(X_list)
    y = np.array(y_list)

    np.savez(cfg.data_path, X=X, y=y)
    print(f"\nDataset saved to {cfg.data_path}  shape X={X.shape}  y={y.shape}")
    print(f"  gen_score  mean={y.mean():.3f}  std={y.std():.3f}  "
          f"min={y.min():.3f}  max={y.max():.3f}")
    return X, y
