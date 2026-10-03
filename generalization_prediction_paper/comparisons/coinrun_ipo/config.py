"""Configuration for the Invariant Policy Optimization (IPO) comparison.

Environment / PPO / evaluation values are identical to the main gen-aware PPO
experiment (`run_coinrun_vec_staged.py` → `data/coinrun_vec/2000K_n800/`).

IPO (Sonar, Pacelli & Majumdar, L4DC 2021; github.com/irom-lab/
Invariant-Policy-Optimization, Colored-Keys) adapted to CoinRun:
  * domains: the same 200 training levels, split into 2 domains
    (levels 0-99 and 100-199), 32 envs each → 64 envs total as the baseline
  * Fixed-Phi (Phi = identity): one full actor-critic per domain
    (NatureCNN + heads, the baseline's CnnPolicy architecture); the acting
    policy is softmax(mean of per-domain logits), value = mean of critics
  * best-response dynamics: collect a rollout on domain d with pi_av, PPO
    update of network d only (others frozen), for d = 1..n_d, repeat
  * learning rate: half of PPO's (paper, Tables 6-7: 1e-3 → 5e-4)
"""
from dataclasses import dataclass


@dataclass
class Config:
    # ---------------- identical to the main gen-aware PPO experiment ---------- #
    env_name: str = "coinrun-vec"
    train_num_levels: int = 200
    train_start_level: int = 0
    eval_start_level: int = 10_000
    n_eval_episodes: int = 1000
    gen_eval_freq: int = 100_000
    gen_eval_episodes: int = 100

    n_envs: int = 64                 # total over domains (also used by the eval env)
    n_steps: int = 256
    batch_size: int = 256
    n_epochs: int = 3
    learning_rate: float = 5e-4      # baseline PPO value; IPO uses ipo_lr_factor * this
    ent_coef: float = 0.01
    clip_range: float = 0.2
    gamma: float = 0.999
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5

    gen_timesteps: int = 2_000_000   # total over both domains
    n_agents: int = 10
    seed_base: int = 8000            # same seeds as the GenPPO / IBAC-SNI / SM arms
    seed_stride: int = 137

    # ---------------- IPO ------------------------------------------------ #
    n_domains: int = 2
    ipo_lr_factor: float = 0.5

    # ---------------- paths ---------------------------------------------- #
    reference_dir: str = "../../gen_aware_ppo/data/coinrun_vec/2000K_n800"             # read-only
    ibac_dir: str = "../coinrun_ibac_sni/results/ibac-sni"         # read-only
    sm_dir: str = "../coinrun_surprise_min/results/sm-normal-centered"  # read-only
    results_dir: str = "results"

    def seed(self, i: int) -> int:
        return self.seed_base + i * self.seed_stride

    def domain_levels(self, d: int) -> tuple[int, int]:
        """(start_level, num_levels) of domain d."""
        n = self.train_num_levels // self.n_domains
        return self.train_start_level + d * n, n
