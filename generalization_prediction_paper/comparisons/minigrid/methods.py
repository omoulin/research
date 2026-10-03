"""The three comparison methods, adapted to the MLP policy of the MiniGrid
SimpleCrossing-vec reference experiment (SB3 MlpPolicy: separate pi / vf
MLPs 147 → 64 → 64, tanh).

  * IBAC-SNI  (Igl et al., NeurIPS 2019) - IBACMlpPolicy + IBACSNIPPO
  * Surprise minimization, Normal density (Chen 2020) - SurpriseMinPPO
  * IPO, Fixed-Phi (Sonar et al., L4DC 2021) - IPOMlpPolicy + IPODomainPPO

Same algorithms as ../coinrun_ibac_sni, ../coinrun_surprise_min and
../coinrun_ipo (copied here so this directory is self-contained); only the
network architecture / observation handling differ.
"""
from __future__ import annotations

import math
from functools import partial

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import explained_variance, obs_as_tensor


# =========================================================================== #
# IBAC-SNI                                                                    #
# =========================================================================== #
class IBACMlpPolicy(ActorCriticPolicy):
    """Baseline MlpPolicy with the policy's last hidden layer made stochastic.

        baseline pi : obs → Linear(147,64) → tanh → Linear(64,64) → tanh → Linear(64,7)
        IBAC pi     : obs → Linear(147,64) → tanh → Linear(64, 2·64) → (mu, rho)
                      z ~ N(mu, softplus(rho)) → tanh → Linear(64,7)
        vf          : unchanged baseline value MLP (deterministic; with SNI the
                      critic is noise-free anyway)

    Bottleneck parameterisation follows the authors' MiniGrid code
    (torch_rl/bottleneck.py): std = softplus(raw), N(0, I) prior, KL in nats
    summed over latent dims, one reparameterised sample, tanh after z.
    """

    def __init__(self, observation_space, action_space, lr_schedule: Schedule,
                 bottleneck_dim: int = 64, n_samples: int = 1, sni: bool = True, **kwargs) -> None:
        assert isinstance(action_space, spaces.Discrete)
        self.bottleneck_dim = bottleneck_dim
        self.n_samples = n_samples
        self.sni = sni
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        d, h = self.bottleneck_dim, 64
        n_in = self.features_dim
        self.mlp_extractor = MlpExtractor(n_in, net_arch=dict(pi=[h], vf=[h, h]),
                                          activation_fn=self.activation_fn, device=self.device)
        self.encoder = nn.Linear(h, 2 * d)
        self.action_net = nn.Linear(d, self.action_space.n)
        self.value_net = nn.Linear(h, 1)
        if self.ortho_init:
            for module, gain in ((self.mlp_extractor, np.sqrt(2)), (self.encoder, np.sqrt(2)),
                                 (self.action_net, 0.01), (self.value_net, 1.0)):
                module.apply(partial(self.init_weights, gain=gain))
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _trunk(self, obs):
        x = self.extract_features(obs)
        return self.mlp_extractor.policy_net(x), self.mlp_extractor.value_net(x)

    def _encode(self, h_pi):
        mu, rho = self.encoder(h_pi).chunk(2, dim=-1)
        return mu, F.softplus(rho)

    def _logp(self, z):
        return F.log_softmax(self.action_net(th.tanh(z)), dim=-1)

    def ibac_forward(self, obs, n_samples: int | None = None):
        k = n_samples or self.n_samples
        h_pi, h_vf = self._trunk(obs)
        mu, sigma = self._encode(h_pi)
        v = self.value_net(h_vf).squeeze(-1)
        eps = th.randn((k,) + mu.shape, device=mu.device, dtype=mu.dtype)
        logp_comp = self._logp(mu.unsqueeze(0) + sigma.unsqueeze(0) * eps)
        kl = (0.5 * (sigma.pow(2) + mu.pow(2) - 1.0) - th.log(sigma)).sum(-1).mean()
        return dict(logp_det=self._logp(mu), v_det=v, logp_comp=logp_comp,
                    logp_mix=th.logsumexp(logp_comp, 0) - math.log(k), v_mix=v, kl=kl)

    def _run(self, obs):
        if self.sni:
            h_pi, h_vf = self._trunk(obs)
            mu, _ = self._encode(h_pi)
            return self._logp(mu), self.value_net(h_vf).squeeze(-1)
        out = self.ibac_forward(obs)
        return out["logp_mix"], out["v_mix"]

    def forward(self, obs, deterministic: bool = False):
        logp, v = self._run(obs)
        dist = self.action_dist.proba_distribution(action_logits=logp)
        a = dist.get_actions(deterministic=deterministic)
        return a, v.unsqueeze(-1), dist.log_prob(a)

    def get_distribution(self, obs):
        return self.action_dist.proba_distribution(action_logits=self._run(obs)[0])

    def predict_values(self, obs):
        return self._run(obs)[1].unsqueeze(-1)

    def _predict(self, obs, deterministic: bool = False):
        return self.get_distribution(obs).get_actions(deterministic=deterministic)

    def evaluate_actions(self, obs, actions):
        logp, v = self._run(obs)
        dist = self.action_dist.proba_distribution(action_logits=logp)
        return v.unsqueeze(-1), dist.log_prob(actions), dist.entropy()


def _entropy(logp):
    return -(logp.exp() * logp).sum(-1)


class IBACSNIPPO(PPO):
    """PPO with the IBAC-SNI loss (eqs. 7 + 12); identical to
    ../coinrun_ibac_sni/ibac_sni_ppo.py."""

    def __init__(self, *args, beta: float = 1e-6, sni_lambda: float = 0.5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.beta, self.sni_lambda = beta, sni_lambda

    @staticmethod
    def _pg(logp_a, old, adv, clip):
        ratio = th.exp(logp_a - old)
        loss = -th.min(adv * ratio, adv * th.clamp(ratio, 1 - clip, 1 + clip)).mean()
        with th.no_grad():
            lr = logp_a - old
            kl = th.mean((th.exp(lr) - 1) - lr).item()
        return loss, th.mean((th.abs(ratio - 1) > clip).float()).item(), kl

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip = self.clip_range(self._current_progress_remaining)
        sni = self.policy.sni
        lam = self.sni_lambda if sni else 0.0
        logs = {k: [] for k in ("pg", "vf", "ent", "kl_ib", "cf", "akl_det", "akl_stoch")}
        for _ in range(self.n_epochs):
            for rd in self.rollout_buffer.get(self.batch_size):
                a_idx = rd.actions.long().flatten().unsqueeze(-1)
                out = self.policy.ibac_forward(rd.observations)
                adv = rd.advantages
                if self.normalize_advantage and len(adv) > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                pg_s, cf_s, akl_s = self._pg(out["logp_mix"].gather(1, a_idx).squeeze(1), rd.old_log_prob, adv, clip)
                ent_s = _entropy(out["logp_comp"]).mean()
                if sni:
                    pg_d, cf_d, akl_d = self._pg(out["logp_det"].gather(1, a_idx).squeeze(1), rd.old_log_prob, adv, clip)
                    pg = lam * pg_d + (1 - lam) * pg_s
                    ent = lam * _entropy(out["logp_det"]).mean() + (1 - lam) * ent_s
                    values, cf = out["v_det"], lam * cf_d + (1 - lam) * cf_s
                    logs["akl_det"].append(akl_d)
                else:
                    pg, ent, values, cf = pg_s, ent_s, out["v_mix"], cf_s
                logs["akl_stoch"].append(akl_s)
                vf = F.mse_loss(rd.returns, values)
                loss = pg - self.ent_coef * ent + self.vf_coef * vf + self.beta * out["kl"]
                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                for k, v in (("pg", pg.item()), ("vf", vf.item()), ("ent", ent.item()),
                             ("kl_ib", out["kl"].item()), ("cf", cf)):
                    logs[k].append(v)
        self._n_updates += self.n_epochs
        self.logger.record("train/policy_gradient_loss", np.mean(logs["pg"]))
        self.logger.record("train/value_loss", np.mean(logs["vf"]))
        self.logger.record("train/entropy_loss", -np.mean(logs["ent"]))
        self.logger.record("train/ib_kl_nats", np.mean(logs["kl_ib"]))
        self.logger.record("train/clip_fraction", np.mean(logs["cf"]))
        if logs["akl_det"]:
            self.logger.record("train/approx_kl_det", np.mean(logs["akl_det"]))
        self.logger.record("train/approx_kl_stoch", np.mean(logs["akl_stoch"]))
        self.logger.record("train/explained_variance",
                           explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()))


# =========================================================================== #
# Surprise minimization (PPO + Normal)                                        #
# =========================================================================== #
class SurpriseMinPPO(PPO):
    """Chen (2020) Algorithm 1 on the observation vector itself.

    The paper fits an independent Gaussian "for each dimension in our
    observations" (greyscale pixels in procgen). Here the observation is the
    147-d normalised symbolic grid, so each of its dimensions is modelled
    directly. Buffer of the most recent frames, r_SM = log p(s) (up to the
    constant 0.5·log 2π), centered per rollout as in ../coinrun_surprise_min.
    """

    def __init__(self, *args, sm_alpha: float = 1e-3, sm_buffer_size: int = 40_960,
                 sm_var_eps: float = 1e-4, sm_center: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sm_alpha, self.sm_buffer_size = sm_alpha, sm_buffer_size
        self.sm_var_eps, self.sm_center = sm_var_eps, sm_center
        self._buf = None
        self._ptr = self._count = 0

    def _push(self, x: th.Tensor) -> None:
        if self._buf is None:
            self._buf = th.empty((self.sm_buffer_size, x.shape[1]), device=x.device)
        for row in x.split(self.sm_buffer_size):
            n = row.shape[0]
            idx = (th.arange(n, device=x.device) + self._ptr) % self.sm_buffer_size
            self._buf[idx] = row
            self._ptr = (self._ptr + n) % self.sm_buffer_size
            self._count = min(self._count + n, self.sm_buffer_size)

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps) -> bool:
        ok = super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)
        if not ok:
            return ok
        obs = rollout_buffer.observations                      # (T, E, 147) float32
        T, E = obs.shape[:2]
        x = th.as_tensor(obs.reshape(T * E, -1), device=self.device).float()
        self._push(x)
        d = self._buf[: self._count].double()
        mu = d.mean(0)
        sigma = th.sqrt(d.var(0, unbiased=False) + self.sm_var_eps)
        r_sm = -(th.log(sigma).sum() + (((x.double() - mu) / sigma) ** 2 * 0.5).sum(1))
        r_sm = r_sm.view(T, E).cpu().numpy()
        used = r_sm - r_sm.mean() if self.sm_center else r_sm
        rollout_buffer.rewards += (self.sm_alpha * used).astype(np.float32)
        with th.no_grad():
            last_values = self.policy.predict_values(obs_as_tensor(self._last_obs, self.device))
        rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=self._last_episode_starts)
        self.logger.record("sm/r_sm_mean", float(r_sm.mean()))
        self.logger.record("sm/r_sm_centered_std", float(used.std()))
        self.logger.record("sm/alpha_r_used_absmean", float(self.sm_alpha * np.abs(used).mean()))
        return ok


# =========================================================================== #
# IPO (Fixed-Phi)                                                             #
# =========================================================================== #
class _MlpBranch(nn.Module):
    """One domain's actor-critic: SB3 MlpPolicy's layers."""

    def __init__(self, n_in: int, n_actions: int, device) -> None:
        super().__init__()
        self.mlp = MlpExtractor(n_in, net_arch=dict(pi=[64, 64], vf=[64, 64]), activation_fn=nn.Tanh, device=device)
        self.action_net = nn.Linear(64, n_actions)
        self.value_net = nn.Linear(64, 1)

    def forward(self, x):
        h_pi, h_vf = self.mlp(x)
        return self.action_net(h_pi), self.value_net(h_vf).squeeze(-1)


class IPOMlpPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule: Schedule, n_domains: int = 2, **kwargs) -> None:
        self.n_domains = n_domains
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        self.branches = nn.ModuleList(
            [_MlpBranch(self.features_dim, self.action_space.n, self.device) for _ in range(self.n_domains)])
        if self.ortho_init:
            for b in self.branches:
                for module, gain in ((b.mlp, np.sqrt(2)), (b.action_net, 0.01), (b.value_net, 1.0)):
                    module.apply(partial(self.init_weights, gain=gain))
        self.domain_optimizers = [self.optimizer_class(b.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
                                  for b in self.branches]
        self.optimizer = self.domain_optimizers[0]

    def set_active_domain(self, d: int) -> None:
        self.optimizer = self.domain_optimizers[d]
        for i, b in enumerate(self.branches):
            for p in b.parameters():
                p.requires_grad_(i == d)
                if i != d:
                    p.grad = None

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad_(True)

    def _avg(self, obs):
        x = self.extract_features(obs)
        outs = [b(x) for b in self.branches]
        return th.stack([o[0] for o in outs]).mean(0), th.stack([o[1] for o in outs]).mean(0)

    def forward(self, obs, deterministic: bool = False):
        logits, v = self._avg(obs)
        dist = self.action_dist.proba_distribution(action_logits=logits)
        a = dist.get_actions(deterministic=deterministic)
        return a, v.unsqueeze(-1), dist.log_prob(a)

    def evaluate_actions(self, obs, actions):
        logits, v = self._avg(obs)
        dist = self.action_dist.proba_distribution(action_logits=logits)
        return v.unsqueeze(-1), dist.log_prob(actions), dist.entropy()

    def get_distribution(self, obs):
        return self.action_dist.proba_distribution(action_logits=self._avg(obs)[0])

    def predict_values(self, obs):
        return self._avg(obs)[1].unsqueeze(-1)

    def _predict(self, obs, deterministic: bool = False):
        return self.get_distribution(obs).get_actions(deterministic=deterministic)


class IPODomainPPO(PPO):
    def __init__(self, *args, domain: int = 0, **kwargs) -> None:
        self.domain = domain
        super().__init__(*args, **kwargs)

    def train(self) -> None:
        self.policy.set_active_domain(self.domain)
        try:
            super().train()
        finally:
            self.policy.unfreeze_all()
