"""PPO + Normal surprise-minimizing reward (Chen 2020, Algorithm 1).

    J. Z. Chen, "Reinforcement Learning Generalization with Surprise
    Minimization", ICML 2020 BIG workshop, arXiv:2004.12399.

After every rollout (Algorithm 1, lines 7-14):
  1. greyscale(s_t) for the rollout's observations is appended to a FIFO
     buffer D holding the most recent `buffer_size` frames;
  2. per-pixel mean mu_i and std sigma_i are computed over D;
  3. r_SM(s) = -sum_i ( log sigma_i + (s_i - mu_i)^2 / (2 sigma_i^2) )
     for every state s in the rollout;
  4. PPO is trained on r_task + alpha * r_SM.

With sm_center=True (default here, NOT in the paper) step 4 uses
r_SM - mean_rollout(r_SM): the constant -sum_i log sigma_i otherwise acts as a
large per-step survival bonus that makes the agent stall (see config.py).

In SB3 terms: after the parent's collect_rollouts() filled the rollout buffer
(with task rewards and GAE computed on them), the rewards are augmented in
place and returns/advantages are recomputed with the same bootstrap values.
The Monitor/eval statistics still see the raw task reward only.
"""
from __future__ import annotations

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor

# ITU-R BT.601 luma, as in PIL / cv2 greyscale conversion
_LUMA = (0.299, 0.587, 0.114)


class SurpriseMinPPO(PPO):
    def __init__(self, *args, sm_alpha: float = 1e-4, sm_buffer_size: int = 40_960,
                 sm_var_eps: float = 1e-4, sm_center: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sm_alpha = sm_alpha
        self.sm_buffer_size = sm_buffer_size
        self.sm_var_eps = sm_var_eps
        self.sm_center = sm_center
        self._sm_buf: th.Tensor | None = None   # (buffer_size, H*W) uint8 greyscale ring buffer
        self._sm_ptr = 0
        self._sm_count = 0

    # ------------------------------------------------------------------ #
    def _greyscale(self, obs: np.ndarray) -> th.Tensor:
        """(N, 3, H, W) uint8 → (N, H*W) uint8 greyscale on the policy device."""
        x = th.as_tensor(obs, device=self.device).float()
        w = th.tensor(_LUMA, device=self.device).view(1, 3, 1, 1)
        g = (x * w).sum(1).round().clamp_(0, 255).to(th.uint8)
        return g.flatten(1)

    def _push(self, frames: th.Tensor) -> None:
        n = frames.shape[0]
        if self._sm_buf is None:
            self._sm_buf = th.empty((self.sm_buffer_size, frames.shape[1]), dtype=th.uint8, device=self.device)
        if n >= self.sm_buffer_size:
            self._sm_buf.copy_(frames[-self.sm_buffer_size:])
            self._sm_ptr, self._sm_count = 0, self.sm_buffer_size
            return
        end = self._sm_ptr + n
        if end <= self.sm_buffer_size:
            self._sm_buf[self._sm_ptr:end] = frames
        else:
            k = self.sm_buffer_size - self._sm_ptr
            self._sm_buf[self._sm_ptr:] = frames[:k]
            self._sm_buf[: n - k] = frames[k:]
        self._sm_ptr = end % self.sm_buffer_size
        self._sm_count = min(self._sm_count + n, self.sm_buffer_size)

    def _buffer_stats(self) -> tuple[th.Tensor, th.Tensor]:
        """Per-pixel mean and std over the buffer, pixels scaled to [0, 1]."""
        d = self._sm_buf[: self._sm_count]
        s1 = th.zeros(d.shape[1], dtype=th.float64, device=self.device)
        s2 = th.zeros_like(s1)
        for chunk in d.split(8192):
            c = chunk.double() / 255.0
            s1 += c.sum(0)
            s2 += (c * c).sum(0)
        mu = s1 / self._sm_count
        var = (s2 / self._sm_count - mu * mu).clamp_min(0.0)
        # small floor keeps log sigma / 1/sigma^2 finite for pixels that are
        # constant in the buffer (the paper does not specify one)
        sigma = th.sqrt(var + self.sm_var_eps)
        return mu.float(), sigma.float()

    # ------------------------------------------------------------------ #
    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps) -> bool:
        ok = super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)
        if not ok:
            return ok

        obs = rollout_buffer.observations                       # (T, E, 3, H, W) uint8
        T, E = obs.shape[:2]
        frames = self._greyscale(obs.reshape((T * E,) + obs.shape[2:]))
        self._push(frames)                                       # D_t = D_{t-1} ∪ greyscale(s_t)
        mu, sigma = self._buffer_stats()                         # θ_t from D_t

        r_sm = th.empty(T * E, device=self.device)
        log_sigma_sum = th.log(sigma).sum()
        inv_2var = 0.5 / (sigma * sigma)
        for idx in th.arange(T * E, device=self.device).split(4096):
            s = frames[idx].float() / 255.0
            r_sm[idx] = -(log_sigma_sum + ((s - mu) ** 2 * inv_2var).sum(1))
        r_sm = r_sm.view(T, E).cpu().numpy()
        r_sm_used = r_sm - r_sm.mean() if self.sm_center else r_sm

        rollout_buffer.rewards += (self.sm_alpha * r_sm_used).astype(np.float32)

        # recompute GAE / returns on the combined reward, same bootstrap as SB3
        with th.no_grad():
            last_values = self.policy.predict_values(obs_as_tensor(self._last_obs, self.device))
        rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=self._last_episode_starts)

        self.logger.record("sm/r_sm_mean", float(r_sm.mean()))
        self.logger.record("sm/r_sm_std", float(r_sm.std()))
        self.logger.record("sm/alpha_r_sm_used_absmean", float(self.sm_alpha * np.abs(r_sm_used).mean()))
        self.logger.record("sm/buffer_frames", self._sm_count)
        return ok
