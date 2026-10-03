# IBAC-SNI vs Generalization-Aware PPO on CoinRun

This directory holds a self-contained comparison between the main gen-aware PPO experiment and

> M. Igl, K. Ciosek, Y. Li, S. Tschiatschek, C. Zhang, S. Devlin, K. Hofmann,
> "Generalization in Reinforcement Learning with Selective Noise Injection and
> Information Bottleneck", NeurIPS 2019. Reference code: github.com/microsoft/IBAC-SNI

Nothing outside this directory is written. The code in `../../gen_aware_ppo` is not imported;
`coinrun_vec_env.py` is a verbatim copy. The reference results are read from
`../../gen_aware_ppo/data/coinrun_vec/2000K_n800/`, and only read.

## Protocol (same as the main gen-aware PPO experiment, `run_coinrun_vec_staged.py`, n=800 checkpoint)

| | value |
|---|---|
| env | CoinRun, procgen native vec env, `easy`, 200 train levels (0–199) |
| eval | unseen levels from 10000, deterministic policy |
| PPO | 64 envs, n_steps 256, batch 256, 3 epochs, lr 5e-4, ent 0.01, clip 0.2, γ 0.999, λ_GAE 0.95, grad-clip 0.5 |
| budget | 10 agents × 2,000,000 steps, seeds 8000 + 137·i (the GenPPO arm's seeds) |
| curve | GenEval every 100K steps, 100 episodes → `gen_curve_*_NN.npy` (same format) |
| metric | last curve point (≈2M steps), mean ± SE over 10 agents, as in `main._compare` |
| extra | final 1000-episode evaluation saved to `final_scores.json`. gen_aware_ppo prints this but does not save it |

## IBAC-SNI implementation

- `ibac_policy.py`: the NatureCNN's last hidden layer (512) becomes a Gaussian bottleneck,
  z ~ N(μ, softplus(ρ−5)), prior N(0, I), followed by ReLU(z) → policy / value heads.
  The noisy policy is the mixture over 12 z-samples.
- `ibac_sni_ppo.py`: the PPO loss is eqs. (7) + (12) of the paper.
  - loss = λ·PPO(π̄) + (1−λ)·PPO(π), where π̄ uses z = μ (the noise-suspended policy)
  - the entropy is mixed with the same λ, using H_IB for the stochastic part
  - the critic is V̄ (z = μ)
  - β · KL is measured in bits
  - rollouts use π̄
- Hyper-parameters are from the paper's Appendix D: β = 1e-4, λ = 0.5, 12 samples.

## Variants (`--variant`)

| variant | meaning |
|---|---|
| `ibac-sni` (default) | the paper's best method, λ = 0.5 |
| `ibac-sni-l1` | λ = 1, deterministic term only (≡ L2 on activations) |
| `ibac` | IBAC without SNI |
| `ibac-sni-l2w` | IBAC-SNI + weight decay 1e-4 (the paper's CoinRun setting) |
| `l2w` | weight decay 1e-4 only (the paper's "Baseline") |
| `baseline` | plain PPO re-run in this software stack, as a reproducibility check against the main experiment's baseline |

## Running (CUDA required; the runner refuses to fall back to CPU)

```bash
cd comparisons/coinrun_ibac_sni
bash run.sh                       # main arm (~75 min on a single GPU)
bash run.sh ibac-sni baseline     # + reproducibility check of the baseline
python compare.py                 # table + results/charts/comparison.{pdf,png}, summary.csv
```

Runs can be resumed: agents whose curve file already exists are skipped.
`python run_ibac_sni.py --quick` runs a 2-agent, 200K-step smoke test into `results/quick_*`.
`compare.py` ignores those runs unless you pass `--include-quick`.

## Differences from the paper (deliberate, to stay comparable with the main gen-aware PPO experiment)

- The paper uses an IMPALA-CNN, 500 levels, hard mode, 600M frames, and weight decay plus
  data augmentation on every arm. Here the architecture, levels, budget and PPO settings are
  those of the main gen-aware PPO experiment. That way the only thing that changes between arms
  is the regulariser.
- Data augmentation is not implemented. Weight decay is available through the `*-l2w` variants.
- The bottleneck width is 512, the baseline's hidden width. The paper used 256, which was its
  baseline's hidden width.
