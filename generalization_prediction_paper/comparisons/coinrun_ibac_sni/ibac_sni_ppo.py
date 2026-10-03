"""IBAC-SNI PPO (Igl et al., NeurIPS 2019) on top of SB3 1.8 PPO.

train() is SB3 1.8's PPO.train() with the loss replaced by eq. (7) + (12):

    SNI gradient (eq. 7):
        G = lambda * G_AC(pi-bar_r, pi-bar, V-bar) + (1 - lambda) * G_AC(pi-bar_r, pi, V-bar)
    IBAC loss (eq. 12):
        L = L_AC^IB + lambda_V L_V - lambda_H H_IB[pi] + beta * L_KL

Implementation details follow the reference code
(microsoft/IBAC-SNI coinrun/coinrun/ppo2.py):
  * both PG terms use the same stored rollout log-probs (from pi-bar_r);
  * with SNI, the critic uses z = mu only (V-bar);
  * entropy of the noisy policy is the mean entropy of the components
    q(a|z_k) (H_IB, eq. 11), mixed with the entropy of pi-bar by lambda;
  * KL is summed over latent dims, averaged over the batch, in bits.

Lines that differ from SB3's train() are marked "# [IBAC]".
"""
from __future__ import annotations

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance


def _entropy(logp: th.Tensor) -> th.Tensor:
    return -(logp.exp() * logp).sum(-1)


class IBACSNIPPO(PPO):
    def __init__(self, *args, beta: float = 1e-4, sni_lambda: float = 0.5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.sni_lambda = sni_lambda

    def _clipped_pg(self, logp_a, old_logp, adv, clip_range):
        ratio = th.exp(logp_a - old_logp)
        loss = -th.min(adv * ratio, adv * th.clamp(ratio, 1 - clip_range, 1 + clip_range)).mean()
        clip_frac = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
        with th.no_grad():
            log_ratio = logp_a - old_logp
            approx_kl = th.mean((th.exp(log_ratio) - 1) - log_ratio).item()
        return loss, clip_frac, approx_kl

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        sni = self.policy.sni                                                  # [IBAC]
        lam = self.sni_lambda if sni else 0.0                                  # [IBAC]

        entropy_losses, pg_losses, value_losses, clip_fractions = [], [], [], []
        kl_losses, kl_det_log, kl_stoch_log = [], [], []                       # [IBAC]
        approx_kl_divs = []

        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions.long().flatten()
                a_idx = actions.unsqueeze(-1)

                out = self.policy.ibac_forward(rollout_data.observations)     # [IBAC]

                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                old_logp = rollout_data.old_log_prob

                # ---- policy loss: SNI mixture of det / stochastic terms [IBAC]
                logp_mix_a = out["logp_mix"].gather(1, a_idx).squeeze(1)
                pg_stoch, cf_stoch, kl_stoch = self._clipped_pg(logp_mix_a, old_logp, advantages, clip_range)
                ent_stoch = _entropy(out["logp_comp"]).mean()                 # H_IB (eq. 11)
                if sni:
                    logp_det_a = out["logp_det"].gather(1, a_idx).squeeze(1)
                    pg_det, cf_det, kl_det = self._clipped_pg(logp_det_a, old_logp, advantages, clip_range)
                    ent_det = _entropy(out["logp_det"]).mean()
                    policy_loss = lam * pg_det + (1 - lam) * pg_stoch
                    entropy = lam * ent_det + (1 - lam) * ent_stoch
                    values = out["v_det"]                                     # V-bar
                    clip_fraction = lam * cf_det + (1 - lam) * cf_stoch
                    approx_kl_div = kl_det                                    # vs rollout policy pi-bar
                    kl_det_log.append(kl_det)
                else:
                    policy_loss, entropy = pg_stoch, ent_stoch
                    values = out["v_mix"]
                    clip_fraction, approx_kl_div = cf_stoch, kl_stoch
                kl_stoch_log.append(kl_stoch)

                pg_losses.append(policy_loss.item())
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                entropy_loss = -entropy
                entropy_losses.append(entropy_loss.item())

                kl_losses.append(out["kl"].item())                             # [IBAC]
                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + self.beta * out["kl"]                                    # [IBAC]
                )

                approx_kl_divs.append(approx_kl_div)
                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/ib_kl_bits", np.mean(kl_losses))              # [IBAC]
        if kl_det_log:
            self.logger.record("train/approx_kl_det", np.mean(kl_det_log))      # [IBAC] paper Fig.3 right
        self.logger.record("train/approx_kl_stoch", np.mean(kl_stoch_log))      # [IBAC]
        self.logger.record("train/clip_range", clip_range)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
