"""Configuration for the surprise-minimization comparison experiment.

Environment / PPO / evaluation values are identical to the main gen-aware PPO
experiment (`run_coinrun_vec_staged.py` → `data/coinrun_vec/2000K_n800/`);
the only change is the reward: r_task + alpha * r_SM (Chen 2020, Algorithm 1).

Surprise-minimization settings follow Chen (2020), Sec. 3.1 / 4.1 (CoinRun):
  * alpha = 1e-4
  * buffer = 20 x mini-batch size of the paper's setup. Their rollout is
    16,384 frames (64 envs x 256 steps, identical to ours) split into 8
    procgen-baselines mini-batches of 2,048 → 40,960 most recent frames.
  * observations → 64x64 greyscale, normalised to [0, 1]
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

    n_envs: int = 64
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
    seed_base: int = 8000            # same seeds as the GenPPO / IBAC-SNI arms
    seed_stride: int = 137

    # ---------------- surprise minimization (Chen 2020) ------------------ #
    # Paper: alpha = 1e-4 for CoinRun. Here it collapses the agent (stands
    # still until the 1000-step timeout, score 0), also with the centered
    # reward and at 1e-5 (200K-step smoke tests). 1e-6 is
    # the largest tested value that trains, i.e. the paper's own selection
    # rule ("downscale the SM reward to a similar level as the task reward");
    # the paper used 1e-6 for BossFight.
    sm_alpha: float = 1e-6
    sm_buffer_size: int = 40_960
    sm_var_eps: float = 1e-4         # variance floor (std >= 0.01), not in the paper
    # Deviation from the paper (chosen after a smoke test): subtract the
    # rollout mean of r_SM. As written, -sum_i log sigma_i is a large constant
    # per-step bonus (+0.5..1.7 per step at alpha=1e-4 with [0,1] pixels) and
    # the agent learns to stand still until the 1000-step timeout (score 0).
    # Centering keeps "prefer less surprising states" without that bias.
    sm_center: bool = True

    # ---------------- paths ---------------------------------------------- #
    reference_dir: str = "../../gen_aware_ppo/data/coinrun_vec/2000K_n800"         # read-only
    ibac_dir: str = "../coinrun_ibac_sni/results/ibac-sni"     # read-only
    results_dir: str = "results"

    def seed(self, i: int) -> int:
        return self.seed_base + i * self.seed_stride
