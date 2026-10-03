"""Configuration for the IBAC-SNI comparison experiment.

Every environment / PPO / evaluation value below is copied from the
configuration that produced the main gen-aware PPO experiment
(`run_coinrun_vec_staged.py` → `data/coinrun_vec/2000K_n800/`), so the only
difference between the arms is the regulariser:

    baseline PPO          data/coinrun_vec/2000K_n800/gen_curve_baseline_NN.npy
    GeneralizationAwarePPO data/coinrun_vec/2000K_n800/phase3_coef*/gen_curve_genppo_NN.npy
    IBAC-SNI (this dir)    results/<variant>/gen_curve_ibac_NN.npy

IBAC-SNI hyper-parameters follow Igl et al. (NeurIPS 2019), Appendix D and the
reference implementation (github.com/microsoft/IBAC-SNI, coinrun/):
beta = 1e-4, lambda = 0.5, 12 posterior samples, softplus(rho - 5) scale.
"""
from dataclasses import dataclass


@dataclass
class Config:
    # ---------------- identical to the main gen-aware PPO experiment ---------- #
    env_name: str = "coinrun-vec"
    train_num_levels: int = 200
    train_start_level: int = 0
    eval_start_level: int = 10_000
    n_eval_episodes: int = 1000          # final evaluation
    gen_eval_freq: int = 100_000         # periodic evaluation (the curves)
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

    gen_timesteps: int = 2_000_000       # PHASE3_TIMESTEPS
    n_agents: int = 10                   # PHASE3_AGENTS
    seed_base: int = 8000                # same seeds as the GenPPO arm
    seed_stride: int = 137
    baseline_seed_base: int = 9000       # same seeds as the baseline arm

    # ---------------- IBAC-SNI (Igl et al. 2019) ------------------------- #
    bottleneck_dim: int = 512   # = width of the baseline's last hidden layer
    beta: float = 1e-4          # KL weight (paper: best of 1e-3..1e-6)
    n_samples: int = 12         # z-samples for the stochastic policy
    sni: bool = True            # selective noise injection
    sni_lambda: float = 0.5     # mix of deterministic / stochastic PG terms
    l2_weight: float = 0.0      # weight decay (paper's CoinRun runs: 1e-4)

    # ---------------- paths ---------------------------------------------- #
    # Read-only reference results from the main gen-aware PPO experiment.
    reference_dir: str = "../../gen_aware_ppo/data/coinrun_vec/2000K_n800"
    results_dir: str = "results"

    def seed(self, i: int) -> int:
        return self.seed_base + i * self.seed_stride

    def baseline_seed(self, i: int) -> int:
        return self.baseline_seed_base + i * self.seed_stride


# Named variants, mirroring the arms in Igl et al., Fig. 3 / Fig. 6.
VARIANTS = {
    # main arm: the paper's best method
    "ibac-sni":      dict(sni=True,  sni_lambda=0.5),
    # lambda = 1: only the deterministic term (= L2 on activations)
    "ibac-sni-l1":   dict(sni=True,  sni_lambda=1.0),
    # IBAC without SNI: stochastic rollouts, noisy critic
    "ibac":          dict(sni=False),
    # the paper's CoinRun setting also adds weight decay to every arm
    "ibac-sni-l2w":  dict(sni=True,  sni_lambda=0.5, l2_weight=1e-4),
    # weight decay only (no bottleneck) - the paper's "Baseline"
    "l2w":           dict(beta=None, l2_weight=1e-4),
}
