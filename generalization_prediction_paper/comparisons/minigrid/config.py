"""Configuration for the MiniGrid comparison experiment.

Reference experiment (read-only): SimpleCrossingS9N2 with symbolic vector
observations, `python main.py --env minigrid-simplecrossing-vec
--skip-phase1 --skip-phase2 --gen-timesteps 2000000 --gen-coef <c>`:
    baseline PPO            ../../gen_aware_ppo/data/simplecrossing_vec/2000K/gen_curve_baseline_NN.npy
    GeneralizationAwarePPO  ../../gen_aware_ppo/data/simplecrossing_vec/2000K/phase3_coef0.2/gen_curve_genppo_NN.npy

All environment / PPO / evaluation values below are copied from gen_aware_ppo's
Config for env_name="minigrid-simplecrossing-vec"; only the method changes.
"""
from dataclasses import dataclass


@dataclass
class Config:
    # ---------------- identical to the reference experiment -------------- #
    env_name: str = "minigrid-simplecrossing-vec"
    policy_type: str = "MlpPolicy"      # SB3 default: pi=[64,64], vf=[64,64], tanh
    minigrid_train_num_seeds: int = 150  # training seeds 0..149
    minigrid_eval_seed_start: int = 10_000
    n_eval_episodes: int = 1000
    gen_eval_freq: int = 100_000
    gen_eval_episodes: int = 100

    n_envs: int = 32
    n_steps: int = 256
    batch_size: int = 256
    n_epochs: int = 3
    learning_rate: float = 5e-4
    ent_coef: float = 0.01
    clip_range: float = 0.2
    gamma: float = 0.999
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5

    gen_timesteps: int = 2_000_000
    n_agents: int = 10
    seed_base: int = 8000               # same seeds as the GenPPO arm
    seed_stride: int = 137

    # ---------------- IBAC-SNI (Igl et al. 2019, MiniGrid/Multiroom code) - #
    ibac_bottleneck_dim: int = 64       # = baseline's last hidden width
    ibac_beta: float = 1e-6             # paper App. C: best for IBAC on Multiroom
    ibac_sni_lambda: float = 0.5
    ibac_n_samples: int = 1             # torch_rl code: one rsample()

    # ---------------- Surprise minimization (Chen 2020) ------------------ #
    sm_buffer_size: int = 40_960        # same frame count as the CoinRun run
    sm_var_eps: float = 1e-4
    sm_center: bool = True
    # alpha is passed on the command line (run.py --alpha). 1e-6 gives the
    # centered r_SM (std ~270/step here) the same size relative to the task
    # reward (max 1) as the alpha=1e-6 setting that trained on CoinRun
    # (std ~2500/step, max reward 10).
    sm_alpha: float = 1e-6

    # ---------------- IPO (Sonar et al. 2021) ---------------------------- #
    ipo_n_domains: int = 2              # training seeds 0-74 / 75-149
    ipo_lr_factor: float = 0.5

    # ---------------- paths ---------------------------------------------- #
    reference_dir: str = "../../gen_aware_ppo/data/simplecrossing_vec/2000K"   # read-only
    genppo_coef: str = "0.2"
    results_dir: str = "results"

    def seed(self, i: int) -> int:
        return self.seed_base + i * self.seed_stride
