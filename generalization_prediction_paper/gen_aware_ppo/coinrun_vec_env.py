"""CoinRun environment helpers using procgen's native vectorized interface.

A gym3-based setup wraps a single procgen instance with `gym3.interop.ToGymEnv`
and then stacks `n_envs` of those through SB3's `make_vec_env` — one Python
gym env per worker. This module instead uses `procgen.ProcgenEnv` directly,
which is procgen's own vectorized environment: a single object that steps
`n_envs` game instances together in the native (C++) simulator and returns
batched observations, matching how procgen's own baselines integrate with
stable-baselines3.

Same game, same RGB pixel observations, same CnnPolicy — only the
env/vectorization backend differs from that gym3-based setup.
"""
from __future__ import annotations

from procgen import ProcgenEnv
from procgen.env import ToBaselinesVecEnv
from stable_baselines3.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecTransposeImage,
)

# SB3's PPO calls `self.env.seed(seed)` during setup; ToBaselinesVecEnv has no
# such method (procgen randomisation is controlled by start_level/num_levels,
# not a gym seed). Patch it to accept and ignore a seed, mirroring the same
# fix needed for gym3.interop.ToGymEnv.
if not hasattr(ToBaselinesVecEnv, "seed"):
    ToBaselinesVecEnv.seed = lambda self, seed=None: None


def make_coinrun_vec(
    num_levels: int,
    start_level: int,
    n_envs: int = 1,
    seed: int = 0,
    num_threads: int = 16,
) -> VecTransposeImage:
    """Return a vectorised, monitor-wrapped, CHW-transposed CoinRun env.

    Args:
        num_levels:  0 → unlimited; >0 → restrict to that many levels.
        start_level: first level index (use large values for unseen levels).
        n_envs:      number of parallel game instances.
        seed:        base random seed (passed to procgen's own RNG).
        num_threads: CPU threads procgen's native simulator uses to step the
            n_envs game instances. procgen's own default is 4, which under-uses
            a many-core CPU and leaves the GPU starved of
            batched observations to run inference on.
    """
    venv = ProcgenEnv(
        num_envs=n_envs,
        env_name="coinrun",
        num_levels=num_levels,
        start_level=start_level,
        distribution_mode="easy",
        rand_seed=seed,
        num_threads=num_threads,
    )
    venv = VecExtractDictObs(venv, "rgb")   # dict{'rgb': ...} → Box
    venv = VecMonitor(venv=venv)
    venv = VecTransposeImage(venv)          # HWC → CHW for NatureCNN
    return venv


def make_train_env(cfg, seed: int = 0) -> VecTransposeImage:
    return make_coinrun_vec(
        num_levels=cfg.train_num_levels,
        start_level=cfg.train_start_level,
        n_envs=cfg.n_envs,
        seed=seed,
    )


def make_eval_env(cfg, seed: int = 0) -> VecTransposeImage:
    """Evaluation env uses unseen levels (start past the training range).

    Uses cfg.n_envs parallel game instances rather than a single instance:
    SB3's evaluate_policy() natively splits n_eval_episodes across a
    multi-env VecEnv, so this doesn't change what's measured (still
    cfg.n_eval_episodes episodes on unseen levels) — it just runs them
    concurrently instead of one at a time, which is where nearly all of the
    per-model wall-clock time was going.
    """
    return make_coinrun_vec(
        num_levels=0,                   # unlimited
        start_level=cfg.eval_start_level,
        n_envs=cfg.n_envs,
        seed=seed,
    )
