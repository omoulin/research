"""Dispatch make_train_env / make_eval_env based on cfg.env_name."""
from __future__ import annotations


def make_train_env(cfg, seed: int = 0):
    if cfg.env_name == "coinrun-vec":
        from coinrun_vec_env import make_train_env as _f
    elif cfg.env_name == "minigrid-simplecrossing-vec":
        from simplecrossing_vec_env import make_train_env as _f
    else:
        raise ValueError(f"Unknown env_name: {cfg.env_name!r}")
    return _f(cfg, seed=seed)


def make_eval_env(cfg, seed: int = 0):
    if cfg.env_name == "coinrun-vec":
        from coinrun_vec_env import make_eval_env as _f
    elif cfg.env_name == "minigrid-simplecrossing-vec":
        from simplecrossing_vec_env import make_eval_env as _f
    else:
        raise ValueError(f"Unknown env_name: {cfg.env_name!r}")
    return _f(cfg, seed=seed)
