"""MiniGrid-SimpleCrossingS9N2-v0 with symbolic vector observations.

Instead of RGB pixels (48×48×3 via RGBImgObsWrapper), this module exposes
the native MiniGrid symbolic grid as a flat float vector:
  - Raw obs: dict{'image': (7, 7, 3) uint8} where each cell encodes
    (object_type [0-10], color [0-5], state [0-2])
  - Wrapper: extracts 'image', normalises each channel by its max value,
    flattens to shape (147,) float32
  - Policy: use MlpPolicy (not CnnPolicy)

Generalization framing:
  - Train env: cycles through minigrid_train_num_seeds distinct seeds
  - Eval  env: seeds >= minigrid_eval_seed_start (never seen during training)
"""
from __future__ import annotations

import gym
import gym_minigrid  # noqa: F401 – side-effect: registers MiniGrid envs
import numpy as np
from gym import spaces
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

ENV_ID = "MiniGrid-SimpleCrossingS9N2-v0"

# Per-channel max values for normalisation: (object_type, color, state)
_CHANNEL_MAX = np.array([10.0, 5.0, 2.0], dtype=np.float32)


class _VecObsWrapper(gym.ObservationWrapper):
    """Normalise and flatten the symbolic image observation to a 1-D vector."""

    def __init__(self, env) -> None:
        super().__init__(env)
        h, w, c = env.observation_space["image"].shape  # typically (7, 7, 3)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(h * w * c,), dtype=np.float32
        )

    def observation(self, obs):
        img = obs["image"].astype(np.float32)   # (7, 7, 3)
        img = img / _CHANNEL_MAX                # normalise per channel
        return img.flatten()                    # (147,)


class _SeededWrapper(gym.Wrapper):
    """Re-seeds the environment before every reset to control diversity."""

    def __init__(
        self,
        env,
        seed_start: int,
        pool_size: int,
        worker_id: int = 0,
        n_workers: int = 1,
    ) -> None:
        super().__init__(env)
        self._seed_start = seed_start
        self._pool_size = pool_size
        self._worker_id = worker_id
        self._n_workers = n_workers
        self._episode = 0

    def reset(self):
        if self._pool_size > 0:
            raw = self._worker_id + self._episode * self._n_workers
            seed = self._seed_start + (raw % self._pool_size)
        else:
            seed = self._seed_start + self._worker_id + self._episode
        self._episode += 1
        self.env.seed(seed)
        return self.env.reset()


def _make_base_env():
    env = gym.make(ENV_ID)
    env = _VecObsWrapper(env)
    return env


def _make_train_fn(worker_id: int, n_workers: int, pool_size: int):
    def _init():
        env = _make_base_env()
        env = _SeededWrapper(
            env, seed_start=0, pool_size=pool_size,
            worker_id=worker_id, n_workers=n_workers,
        )
        return env
    return _init


def _make_eval_fn(seed_start: int):
    def _init():
        env = _make_base_env()
        env = _SeededWrapper(env, seed_start=seed_start, pool_size=0)
        return env
    return _init


def make_train_env(cfg, seed: int = 0) -> VecMonitor:
    n = cfg.n_envs
    fns = [_make_train_fn(worker_id=i, n_workers=n,
                          pool_size=cfg.minigrid_train_num_seeds)
           for i in range(n)]
    return VecMonitor(DummyVecEnv(fns))


def make_eval_env(cfg, seed: int = 0) -> VecMonitor:
    fn = _make_eval_fn(seed_start=cfg.minigrid_eval_seed_start + seed)
    return VecMonitor(DummyVecEnv([fn]))
