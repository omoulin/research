# Invariant Policy Optimization (Sonar et al. 2021) vs Generalization-Aware PPO on CoinRun

Paper: A. Sonar, V. Pacelli, A. Majumdar, "Invariant Policy Optimization: Towards Stronger
Generalization in Reinforcement Learning", L4DC 2021.
Reference code: github.com/irom-lab/Invariant-Policy-Optimization (Colored-Keys).

This directory is self-contained. The code in `../../gen_aware_ppo` is not imported, and
`coinrun_vec_env.py` is a verbatim copy. Reference results are only read, from:
- `../../gen_aware_ppo/data/coinrun_vec/2000K_n800/`: baseline PPO and gen-aware PPO
- `../coinrun_ibac_sni/` and `../coinrun_surprise_min/`

## Method (Fixed-Φ IPO, as in the paper's discrete-action experiment)
- **Networks:** one complete actor-critic per training domain, each with the baseline's
  CnnPolicy architecture. The acting policy is softmax(mean of the per-domain logits),
  and the value is the mean of the critics (`ACModel_average`).
- **Best-response dynamics (Algorithm 1):** for each domain d in turn, collect a rollout on
  domain d with the averaged policy, then run a PPO update of network d only. The other
  networks are frozen.
- **Learning rate:** half of PPO's (paper Tables 6–7), so 2.5e-4.

## Adaptation to CoinRun
- **Domains:** CoinRun has no natural domains such as the paper's key colours, so the same
  200 training levels are split into two domains: levels 0–99 and 100–199.
- **Envs:** 32 envs per domain, 64 in total as in the baseline.
- **Budget:** 2M environment steps summed over both domains, with the same GenEval curve
  protocol and 1000-episode final evaluation.
- **Seeds:** 8000 + 137·i.
- **Size:** IPO has 2× the baseline's parameters (two networks). This is inherent to the method.

## Running
```bash
python run_ipo.py    # 10 agents, results/ipo/
python compare.py    # results/charts/comparison_3arms.{pdf,png}, comparison_all_methods, summary.csv
```
