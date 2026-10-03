"""SB3 callback that periodically evaluates generalization on unseen levels."""
from __future__ import annotations

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from env_factory import make_eval_env
from config import Config


class GenEvalCallback(BaseCallback):
    """Evaluates generalization every `eval_freq` timesteps and stores results.

    Results are a list of (timestep, mean_reward) tuples accessible via
    `self.results` and saveable with `self.save(path)`.
    """

    def __init__(
        self,
        cfg: Config,
        eval_freq: int | None = None,
        n_eval_episodes: int | None = None,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.cfg = cfg
        self.eval_freq = eval_freq if eval_freq is not None else cfg.gen_eval_freq
        self.n_eval_episodes = (
            n_eval_episodes if n_eval_episodes is not None else cfg.gen_eval_episodes
        )
        self.results: list[tuple[int, float]] = []
        self._last_eval_t: int = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_t >= self.eval_freq:
            self._last_eval_t = self.num_timesteps
            score = self._evaluate()
            self.results.append((self.num_timesteps, score))
            if self.verbose:
                print(
                    f"  [GenEval] t={self.num_timesteps:>10,}  "
                    f"gen_score={score:.3f}  (n={self.n_eval_episodes})"
                )
        return True

    def _evaluate(self) -> float:
        from stable_baselines3.common.evaluation import evaluate_policy

        eval_env = make_eval_env(self.cfg)
        mean_reward, _ = evaluate_policy(
            self.model,
            eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=True,
            warn=False,
        )
        eval_env.close()
        return float(mean_reward)

    def save(self, path: str) -> None:
        """Save results as a (N, 2) numpy array: column 0 = timestep, 1 = score."""
        np.save(path, np.array(self.results, dtype=np.float64))
        if self.verbose:
            print(f"  [GenEval] curve saved to {path}  ({len(self.results)} points)")
