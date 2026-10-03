"""Generalization evaluation utilities."""
from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from env_factory import make_eval_env
from config import Config


def evaluate_generalization(model: PPO, cfg: Config, seed: int = 42) -> float:
    """Return the mean episode reward over cfg.n_eval_episodes unseen levels.

    Each episode is one level from [eval_start_level, ∞).  We run with a
    single environment to be reproducible and avoid reusing levels.
    """
    eval_env = make_eval_env(cfg, seed=seed)
    mean_reward, _ = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=cfg.n_eval_episodes,
        deterministic=True,
        warn=False,
    )
    eval_env.close()
    return float(mean_reward)
