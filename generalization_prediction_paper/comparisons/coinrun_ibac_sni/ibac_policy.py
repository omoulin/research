"""IBAC actor-critic policy for SB3 1.8 (Igl et al., NeurIPS 2019, Sec. 4.2).

Architecture, compared with the baseline SB3 `CnnPolicy` used in the main
gen-aware PPO experiment:

    baseline : NatureCNN conv → flatten → Linear(n, 512) → ReLU → {pi, v}
    IBAC     : NatureCNN conv → flatten → Linear(n, 2·512) → (mu, rho)
               z ~ N(mu, softplus(rho - 5))          # variational bottleneck
               ReLU(z) → {pi, v}

i.e. the last hidden layer (the one the paper puts the bottleneck on) becomes
a stochastic encoding p(z|s) with a N(0, I) prior. Everything else - conv
stack, heads, initialisation - is the SB3 default.

The noise-suspended versions (pi-bar, V-bar in the paper) use z = mu.
The noisy policy is the mixture pi(a|s) = 1/K sum_k q(a|z_k), z_k ~ p(z|s),
exactly as in the reference implementation (MixtureSameFamily over
NR_SAMPLES = 12 samples).
"""
from __future__ import annotations

import math
from functools import partial
from typing import Dict, Tuple

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule


class NatureConvTrunk(BaseFeaturesExtractor):
    """NatureCNN's convolutional stack without its final Linear+ReLU.

    Identical layers to SB3's NatureCNN.cnn; the final Linear is replaced by
    the bottleneck encoder in IBACPolicy.
    """

    def __init__(self, observation_space: spaces.Box) -> None:
        n_input_channels = observation_space.shape[0]
        cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            n_flatten = cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]
        super().__init__(observation_space, n_flatten)
        self.cnn = cnn

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.cnn(observations)


class IBACPolicy(ActorCriticCnnPolicy):
    """Actor-critic with a variational information bottleneck (IBAC).

    Only Discrete action spaces are supported (CoinRun has 15 actions).
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        bottleneck_dim: int = 512,
        n_samples: int = 12,
        sni: bool = True,
        **kwargs,
    ) -> None:
        assert isinstance(action_space, spaces.Discrete), "IBACPolicy: Discrete actions only"
        self.bottleneck_dim = bottleneck_dim
        self.n_samples = n_samples
        self.sni = sni
        kwargs.setdefault("features_extractor_class", NatureConvTrunk)
        kwargs.setdefault("net_arch", [])
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    # ------------------------------------------------------------------ #
    # construction                                                        #
    # ------------------------------------------------------------------ #
    def _build(self, lr_schedule: Schedule) -> None:
        d = self.bottleneck_dim
        # identity MLP extractor (kept so inherited helpers still work)
        from stable_baselines3.common.torch_layers import MlpExtractor
        self.mlp_extractor = MlpExtractor(d, net_arch=[], activation_fn=self.activation_fn, device=self.device)

        self.encoder = nn.Linear(self.features_dim, 2 * d)       # → (mu, rho)
        self.action_net = self.action_dist.proba_distribution_net(latent_dim=d)
        self.value_net = nn.Linear(d, 1)

        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.encoder: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self) -> Dict:
        data = super()._get_constructor_parameters()
        data.update(bottleneck_dim=self.bottleneck_dim, n_samples=self.n_samples, sni=self.sni)
        return data

    # ------------------------------------------------------------------ #
    # core computation                                                    #
    # ------------------------------------------------------------------ #
    def encode(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """Return (mu, sigma) of p(z|s)."""
        h = self.extract_features(obs)
        params = self.encoder(h)
        mu, rho = params.chunk(2, dim=-1)
        sigma = F.softplus(rho - 5.0)
        return mu, sigma

    def _heads(self, z: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        latent = F.relu(z)
        return F.log_softmax(self.action_net(latent), dim=-1), self.value_net(latent).squeeze(-1)

    def ibac_forward(self, obs: th.Tensor, n_samples: int | None = None) -> Dict[str, th.Tensor]:
        """Everything the IBAC / IBAC-SNI loss needs, from one conv pass.

        Returns:
            logp_det   (B, A)    log q(a | z = mu)             - pi-bar
            v_det      (B,)      V(z = mu)                      - V-bar
            logp_comp  (K, B, A) log q(a | z_k), z_k ~ p(z|s)
            logp_mix   (B, A)    log 1/K sum_k q(a | z_k)       - pi
            v_mix      (B,)      1/K sum_k V(z_k)
            kl         ()        sum_dim mean_batch KL(p(z|s) || N(0,1)) in bits
        """
        k = n_samples or self.n_samples
        mu, sigma = self.encode(obs)
        logp_det, v_det = self._heads(mu)

        eps = th.randn((k,) + mu.shape, device=mu.device, dtype=mu.dtype)
        z = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps              # (K, B, d)
        logp_comp, v_comp = self._heads(z)
        logp_mix = th.logsumexp(logp_comp, dim=0) - math.log(k)

        kl = 0.5 * (sigma.pow(2) + mu.pow(2) - 1.0) - th.log(sigma)
        kl = kl.mean(0).sum() / math.log(2.0)

        return dict(logp_det=logp_det, v_det=v_det, logp_comp=logp_comp,
                    logp_mix=logp_mix, v_mix=v_comp.mean(0), kl=kl)

    def _run_logp_value(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """Rollout ("run") policy and critic.

        SNI: noise suspended (z = mu) for both.  No SNI: noisy mixture policy
        and sample-averaged critic, as in the reference implementation.
        """
        if self.sni:
            mu, _ = self.encode(obs)
            return self._heads(mu)
        out = self.ibac_forward(obs)
        return out["logp_mix"], out["v_mix"]

    # ------------------------------------------------------------------ #
    # SB3 interface (used for rollouts, bootstrapping and evaluation)     #
    # ------------------------------------------------------------------ #
    def _dist(self, logp: th.Tensor) -> CategoricalDistribution:
        return self.action_dist.proba_distribution(action_logits=logp)

    def forward(self, obs: th.Tensor, deterministic: bool = False):
        logp, values = self._run_logp_value(obs)
        dist = self._dist(logp)
        actions = dist.get_actions(deterministic=deterministic)
        return actions, values.unsqueeze(-1), dist.log_prob(actions)

    def get_distribution(self, obs: th.Tensor):
        return self._dist(self._run_logp_value(obs)[0])

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        return self._run_logp_value(obs)[1].unsqueeze(-1)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.get_distribution(observation).get_actions(deterministic=deterministic)

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor):
        # Not used by IBACSNIPPO.train(); provided for SB3 API completeness.
        logp, values = self._run_logp_value(obs)
        dist = self._dist(logp)
        return values.unsqueeze(-1), dist.log_prob(actions), dist.entropy()
