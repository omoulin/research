# Surprise minimization (Chen 2020) vs Generalization-Aware PPO on CoinRun

Paper: J. Z. Chen, "Reinforcement Learning Generalization with Surprise Minimization",
ICML 2020 BIG workshop, arXiv:2004.12399. No code was published.

This directory is self-contained. The code in `../../gen_aware_ppo` is not imported, and
`coinrun_vec_env.py` is a verbatim copy. Reference results are only read, from:
- `../../gen_aware_ppo/data/coinrun_vec/2000K_n800/`: baseline PPO and gen-aware PPO
- `../coinrun_ibac_sni/results/ibac-sni/`

## Method: PPO + Normal (Algorithm 1)
After every rollout:
1. The rollout's frames are converted to 64×64 greyscale in [0, 1] and appended to a
   FIFO buffer of 40,960 frames. That is 20 × the paper's 2,048-frame mini-batch; the
   paper's rollout is 16,384 frames, identical to ours.
2. A per-pixel mean μ and standard deviation σ are computed over the buffer.
3. Each state gets r_SM(s) = −Σᵢ (log σᵢ + (sᵢ−μᵢ)² / 2σᵢ²).
4. PPO trains on r_task + α·r_SM.

Everything else is identical to the main gen-aware PPO experiment:
- 200 levels, easy mode, 64 envs, the same PPO hyperparameters and the NatureCNN CnnPolicy
- 10 agents × 2M steps, seeds 8000 + 137·i
- the same GenEval curve protocol, plus a 1000-episode final evaluation

## Deviations from the paper (both forced by 200K-step smoke tests)
1. **Centered r_SM** (`sm_center=True`). The −Σ log σᵢ term is a large constant added to
   every step's reward. At the paper's α = 1e-4 it is worth +0.5 to +1.7 per step, against
   a coin reward of 10 per episode. By 120K steps the agent stands still until the
   1000-step timeout and its test score is 0.
   The paper-exact version is still available: `--no-center --alpha 1e-4`.
2. **α = 1e-6.** Centering alone still collapses at α = 1e-4 and at 1e-5: standing still
   produces the least surprising frames. α = 1e-6 is the largest value tested that trains
   (6.4 at 200K steps). This follows the paper's own rule of scaling the surprise reward
   to the size of the task reward; the paper used 1e-6 for BossFight.
- A variance floor of 1e-4 (σ ≥ 0.01) keeps log σ finite for constant pixels. The paper
  does not specify one.
- The paper's VAE density model (Algorithm 2) is not implemented. On CoinRun it did not
  generalize better than the Normal model.

## Running
```bash
python run_sm.py     # 10 agents, results/sm-normal-centered/
python compare.py    # results/charts/comparison_3arms.{pdf,png} (+ comparison_4arms with IBAC-SNI)
```
