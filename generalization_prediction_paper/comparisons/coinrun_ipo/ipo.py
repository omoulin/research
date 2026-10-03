"""Invariant Policy Optimization (Sonar et al., L4DC 2021) on SB3 1.8 PPO.

Mirrors the reference implementation (Colored-Keys: model.ACModel_average,
torch_ac/algos/ipo.py, scripts/multi_domain_train.py), Fixed-Phi variant:

  * IPOPolicy holds one complete actor-critic per domain (here the baseline's
    CnnPolicy architecture: NatureCNN → 512 → {pi, v}). The acting policy is
    Categorical(softmax(mean_d logits_d)); the value is mean_d v_d.
  * IPODomainPPO is a PPO learner bound to one domain's env; its train()
    only updates that domain's network (the other networks are frozen for the
    duration of the update), i.e. one best-response step
        pi_d <- PPO( R_d(pi_av) ),   pi_i (i != d) fixed.
  * run_ipo() alternates: collect on domain 1 with pi_av, update pi_1;
    collect on domain 2 with the new pi_av, update pi_2; ...
"""
from __future__ import annotations

from functools import partial

import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import NatureCNN
from stable_baselines3.common.type_aliases import Schedule


class _Branch(nn.Module):
    """One domain's actor-critic: SB3 CnnPolicy's layers."""

    def __init__(self, observation_space, n_actions: int) -> None:
        super().__init__()
        self.features = NatureCNN(observation_space, features_dim=512)
        self.action_net = nn.Linear(512, n_actions)
        self.value_net = nn.Linear(512, 1)

    def forward(self, obs: th.Tensor):
        h = self.features(obs)
        return self.action_net(h), self.value_net(h).squeeze(-1)


class IPOPolicy(ActorCriticCnnPolicy):
    def __init__(self, observation_space, action_space, lr_schedule: Schedule, n_domains: int = 2, **kwargs) -> None:
        self.n_domains = n_domains
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        # the default single features extractor is unused (keep only the
        # /255 preprocessing done by extract_features)
        self.features_extractor = nn.Identity()
        self.pi_features_extractor = self.features_extractor
        self.vf_features_extractor = self.features_extractor
        n_actions = self.action_space.n
        self.branches = nn.ModuleList([_Branch(self.observation_space, n_actions) for _ in range(self.n_domains)])
        if self.ortho_init:
            for b in self.branches:
                for module, gain in ((b.features, np.sqrt(2)), (b.action_net, 0.01), (b.value_net, 1.0)):
                    module.apply(partial(self.init_weights, gain=gain))
        # one optimizer per domain, over that domain's network only
        self.domain_optimizers = [
            self.optimizer_class(b.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs) for b in self.branches
        ]
        self.optimizer = self.domain_optimizers[0]
        self.active_domain = 0

    def set_active_domain(self, d: int) -> None:
        """Select the optimizer for d and freeze every other network."""
        self.active_domain = d
        self.optimizer = self.domain_optimizers[d]
        for i, b in enumerate(self.branches):
            for p in b.parameters():
                p.requires_grad_(i == d)
                if i != d:
                    p.grad = None

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad_(True)

    # ------------------------------------------------------------------ #
    def _avg(self, obs: th.Tensor):
        x = self.extract_features(obs)            # preprocessing only (/255); extractor is Identity
        outs = [b(x) for b in self.branches]
        logits = th.stack([o[0] for o in outs]).mean(0)
        values = th.stack([o[1] for o in outs]).mean(0)
        return logits, values

    def forward(self, obs: th.Tensor, deterministic: bool = False):
        logits, values = self._avg(obs)
        dist = self.action_dist.proba_distribution(action_logits=logits)
        actions = dist.get_actions(deterministic=deterministic)
        return actions, values.unsqueeze(-1), dist.log_prob(actions)

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor):
        logits, values = self._avg(obs)
        dist = self.action_dist.proba_distribution(action_logits=logits)
        return values.unsqueeze(-1), dist.log_prob(actions), dist.entropy()

    def get_distribution(self, obs: th.Tensor):
        return self.action_dist.proba_distribution(action_logits=self._avg(obs)[0])

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        return self._avg(obs)[1].unsqueeze(-1)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.get_distribution(observation).get_actions(deterministic=deterministic)


class IPODomainPPO(PPO):
    """PPO learner for one domain; train() is one best-response update."""

    def __init__(self, *args, domain: int = 0, **kwargs) -> None:
        self.domain = domain
        super().__init__(*args, **kwargs)

    def train(self) -> None:
        self.policy.set_active_domain(self.domain)
        try:
            super().train()
        finally:
            self.policy.unfreeze_all()
