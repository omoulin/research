"""Phase 3 – GeneralizationAwarePPO.

Subclasses SB3 1.8 PPO and adds one extra term to the per-mini-batch loss:

    loss_total = loss_ppo  -  gen_coef * predictor(weight_features(policy))

where weight_features() is computed *differentiably* so that gradients flow
back through the predictor into the policy parameters.

The predictor is frozen (no gradient w.r.t. predictor weights).
The effect is that the optimiser nudges the policy weights toward
configurations that the predictor associates with high generalization.
This is equivalent to a learned regulariser derived from empirical data.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.utils import explained_variance

from predictor import GeneralizationPredictor
from weight_features import extract_diff


class GeneralizationAwarePPO(PPO):
    """PPO whose loss includes a predictor-based generalization term."""

    def __init__(
        self,
        policy: Union[str, type[ActorCriticPolicy]],
        env: GymEnv,
        predictor: Optional[GeneralizationPredictor] = None,
        gen_coef: float = 0.005,
        **kwargs,
    ) -> None:
        super().__init__(policy, env, **kwargs)
        self.predictor = predictor
        self.gen_coef = gen_coef

        if predictor is not None:
            # Move predictor to the same device as the policy and keep frozen.
            self.predictor = predictor.to(self.device)
            for p in self.predictor.parameters():
                p.requires_grad_(False)

    # ---------------------------------------------------------------------- #
    # Override train() – copied from SB3 1.8.0 with the gen term added.      #
    # Lines marked "# [GEN]" are the only modifications.                     #
    # ---------------------------------------------------------------------- #
    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses, pg_losses, value_losses, clip_fractions = [], [], [], []
        gen_score_log: list[float] = []  # [GEN]

        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []

            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()

                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.item())

                clip_fraction = torch.mean(
                    (torch.abs(ratio - 1) > clip_range).float()
                ).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                )

                # ---------------------------------------------------------- #
                # [GEN] Add generalization term when a predictor is available #
                # ---------------------------------------------------------- #
                if self.predictor is not None:
                    # extract_diff keeps gradient connections to policy params
                    weight_feats = extract_diff(self.policy)
                    gen_score = self.predictor(weight_feats)   # scalar tensor
                    # subtract to *maximise* the predicted generalization score
                    gen_term = -self.gen_coef * gen_score
                    loss = loss + gen_term
                    gen_score_log.append(gen_score.item())
                # ---------------------------------------------------------- #

                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = (
                        torch.mean((torch.exp(log_ratio) - 1) - log_ratio)
                        .cpu()
                        .numpy()
                    )
                    approx_kl_divs.append(approx_kl_div)

                if (
                    self.target_kl is not None
                    and approx_kl_div > 1.5 * self.target_kl
                ):
                    continue_training = False
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if gen_score_log:
            self.logger.record("train/gen_score", np.mean(gen_score_log))  # [GEN]
        if hasattr(self, "clip_range_vf") and self.clip_range_vf is not None:
            self.logger.record(
                "train/clip_range_vf", clip_range_vf
            )
        self.logger.record("train/clip_range", clip_range)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
