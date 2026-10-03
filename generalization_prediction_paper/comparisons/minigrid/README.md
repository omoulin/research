# MiniGrid comparison: IBAC-SNI, surprise minimization, IPO vs Generalization-Aware PPO

This directory holds a self-contained comparison on MiniGrid `SimpleCrossingS9N2-v0`, using the
symbolic 147-d vector observation. The code in `../../gen_aware_ppo` is not imported;
`simplecrossing_vec_env.py` is a verbatim copy. Reference results are only read, from
`../../gen_aware_ppo/data/simplecrossing_vec/2000K/`:
- baseline PPO: `gen_curve_baseline_NN.npy`
- gen-aware PPO: `phase3_coef0.2/gen_curve_genppo_NN.npy`; charts use coef 0.2 by default

## Protocol (same as the reference run: `main.py --env minigrid-simplecrossing-vec --gen-timesteps 2000000`)
| | value |
|---|---|
| env | SimpleCrossingS9N2, 32 envs, training seeds 0–149 (cycled per worker) |
| eval | unseen seeds from 10000, single env, deterministic policy |
| policy | the MlpPolicy size: separate pi / vf MLPs, 147 → 64 → 64, tanh |
| PPO | n_steps 256, batch 256, 3 epochs, lr 5e-4, ent 0.01, clip 0.2, γ 0.999, λ 0.95, grad-clip 0.5 |
| budget | 10 agents × 2M steps, seeds 8000 + 137·i |
| curve | GenEval every 100K steps, 100 episodes |
| final | 1000 episodes → `final_scores.json` |

## Methods (`methods.py`)
- **IBAC-SNI** (Igl et al. 2019). This follows the authors' *MiniGrid* code (`torch_rl/`),
  not their CoinRun code:
  - the policy's last hidden layer (64) becomes z ~ N(μ, softplus(ρ)), followed by tanh
  - one sample, KL in nats, β = 1e-6 (paper App. C: best IBAC β on Multiroom), λ = 0.5
  - the critic keeps the baseline's separate value MLP. With SNI the critic is noise-free
    anyway; in the authors' code the critic also reads z.
- **Surprise minimization** (Chen 2020, PPO + Normal).
  - A per-dimension Gaussian is fitted over a 40,960-frame FIFO buffer of the observation vectors.
  - r_SM = log p(s), **centered per rollout**, with α = 1e-6.
  - As on CoinRun, the paper-exact form (uncentered, with α set by the paper's CoinRun value)
    collapses. α = 1e-6 gives the centered reward (std ≈ 270 per step) the same size relative
    to the task reward (max 1) as the CoinRun setting that trained.
- **IPO** (Sonar et al. 2021), with fixed Φ.
  - There are two domains: training seeds 0–74 and 75–149, 16 envs each.
  - Each domain has its own complete MlpPolicy-sized actor-critic. The policy uses averaged
    logits, and the value is the average of the critics.
  - Training uses best-response PPO updates with lr × 0.5. The paper's Colored-Keys
    experiment is itself MiniGrid; there, the domains were key colours.

## Running (CUDA required)
```bash
python run.py --method ibac-sni
python run.py --method sm --alpha 1e-6
python run.py --method ipo
python compare.py                  # charts in results/charts/, gen-aware PPO coef 0.2
python compare.py --genppo-coef 0.5
```
`--quick` runs 1 agent for 300K steps into `results/quick_*`. `compare.py` ignores those
unless `--include-quick` is given.

## Note on the reference
In this environment gen-aware PPO at coef 0.2 is *not* significantly better than the baseline:
0.565 ± 0.031 vs 0.540 ± 0.055, Welch p = 0.71.
