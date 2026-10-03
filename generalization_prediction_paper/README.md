# Generalization-Aware PPO

Code accompanying the paper. The idea is in three phases:

1. **Collect.** Train many PPO agents and measure how well each one generalizes to unseen
   levels.
2. **Predict.** Train a small MLP that predicts that generalization score from simple
   statistics of the policy's weights.
3. **Regularize.** Add the frozen predictor to the PPO loss so that training is pushed
   towards weights the predictor associates with good generalization:

   ```
   loss = loss_PPO - gen_coef * predictor(weight_features(policy))
   ```

The experiments use two environments:

| Environment | Observation | Policy | Train levels / seeds | Eval levels / seeds |
|---|---|---|---|---|
| **CoinRun** (procgen, `easy`, native vectorized env) | 64×64 RGB | SB3 `CnnPolicy` | levels 0–199 | levels ≥ 10000 |
| **MiniGrid SimpleCrossingS9N2** | symbolic 7×7×3 grid, flattened to 147-d | SB3 `MlpPolicy` | seeds 0–149 | seeds ≥ 10000 |

Gen-aware PPO is compared with IBAC-SNI (Igl et al. 2019), surprise minimization
(Chen 2020) and Invariant Policy Optimization (Sonar et al. 2021) on both environments.

## Repository layout

```
.
├── requirements.txt
├── setup.sh                      # creates a Python 3.10 venv and installs requirements
├── gen_aware_ppo/                # the proposed method; run its scripts from inside this folder
│   ├── config.py                 # all hyperparameters (Config dataclass, per-env overrides)
│   ├── env_factory.py            # dispatches to the environment modules below
│   ├── coinrun_vec_env.py        # CoinRun via procgen's native vectorized env
│   ├── simplecrossing_vec_env.py # MiniGrid SimpleCrossingS9N2, symbolic vector observations
│   ├── weight_features.py        # per-layer weight statistics (numpy + differentiable)
│   ├── predictor.py              # GeneralizationPredictor (MLP)
│   ├── evaluate.py               # generalization score on unseen levels
│   ├── gen_eval_callback.py      # periodic generalization evaluation during training
│   ├── phase1_collect.py         # phase 1 library: train base models and score them
│   ├── phase2_predictor.py       # phase 2 library: train / load the predictor
│   ├── phase3_gen_ppo.py         # phase 3: GeneralizationAwarePPO (PPO subclass)
│   ├── main.py                   # phases 1-3 CLI (baseline vs gen-aware PPO)
│   │
│   ├── run_coinrun_vec_staged.py   # CoinRun: full phase 1/2/3 pipeline (800 models)
│   ├── train_predictor_heldout.py  # CoinRun: predictor accuracy on held-out models
│   ├── phase1_simplecrossing_vec.py# MiniGrid: phase 1 (800 models) + predictor checkpoints
│   │
│   ├── plot_predictor_accuracy.py  # predicted vs real generalization scatter plot
│   ├── plot_gen_curves.py          # generalization curves, baseline vs gen-aware PPO
│   └── regenerate_phase3_charts.py # re-draw all phase 3 comparison charts under data/
│
└── comparisons/                  # baselines from the literature (each folder is self-contained)
    ├── coinrun_ibac_sni/         # IBAC-SNI on CoinRun
    ├── coinrun_surprise_min/     # surprise minimization on CoinRun
    ├── coinrun_ipo/              # IPO on CoinRun (its compare.py also charts all CoinRun methods)
    └── minigrid/                 # IBAC-SNI, surprise minimization and IPO on MiniGrid
```

All outputs (trained models, datasets, predictors, curves and charts) go to
`gen_aware_ppo/data/` and `comparisons/*/results/`. They are created when the scripts run and
are not part of the repository. The comparison scripts read the gen-aware PPO and baseline
curves from `gen_aware_ppo/data/`, so the main experiments must be run first.

## Installation

procgen has no wheels for Python 3.11 or later, so **Python 3.10 is required**.

```bash
bash setup.sh              # or: create any Python 3.10 env and `pip install -r requirements.txt`
source .venv/bin/activate
```

Main dependencies: `stable-baselines3==1.8.0`, `gym==0.21.0`, `procgen==0.10.7`,
`gym-minigrid==1.0.3` and PyTorch. You need a CUDA GPU in practice, and the comparison runners
refuse to fall back to the CPU.

A quick smoke test (a few small models, a few hundred thousand steps):

```bash
cd gen_aware_ppo
python main.py --env coinrun-vec --quick
```

## Reproducing the experiments

All commands in this section run from inside `gen_aware_ppo/`.

### CoinRun

**Phases 1-3 (staged).** This trains 800 base models (1M steps each). Every 50 models it
retrains the predictor and saves an accuracy chart. At 400 and 800 models it runs phase 3:
10 baseline PPO agents and 10 gen-aware PPO agents, 2M steps each.

```bash
python run_coinrun_vec_staged.py
```

Outputs:
- `data/coinrun_vec/`: base models, `gen_scores.npy`, `phase1_dataset.npz`, `predictor.pt`
  and `charts/`
- `data/coinrun_vec/2000K_n{400,800}/`: baseline curves
- `data/coinrun_vec/2000K_n{400,800}/phase3_coef<c>/`: gen-aware curves and charts

The runner is resumable: models and curves that already exist are skipped.

**Predictor accuracy on held-out models:**

```bash
python train_predictor_heldout.py --n-test 30 --seed 0
```

**Other values of `gen_coef`.** The staged runner uses `gen_coef` from `config.py`. The
paper's CoinRun comparisons use `gen_coef = 0.1`. Phase 3 uses the predictor trained on all
800 models (`data/coinrun_vec/predictor.pt`), and the comparison scripts read the gen-aware
curves from `data/coinrun_vec/2000K_n800/phase3_coef<c>/`.

### MiniGrid SimpleCrossingS9N2 (vector observations)

**Phase 1.** This trains 800 base models (500K steps each). Every 100 models it saves a
predictor checkpoint and an accuracy chart on a fixed set of 30 held-out models.

```bash
python phase1_simplecrossing_vec.py
```

**Phase 2.** Train the predictor on all 800 models:

```bash
python main.py --env minigrid-simplecrossing-vec --skip-phase1 --skip-phase3
```

**Phase 3.** 10 baseline agents and 10 gen-aware agents, 2M steps each. The paper uses
`gen_coef = 0.2`.

```bash
python main.py --env minigrid-simplecrossing-vec --skip-phase1 --skip-phase2 \
               --gen-timesteps 2000000 --gen-coef 0.2
```

Results go to `data/simplecrossing_vec/2000K/`. The baselines are trained once and reused
when you run other values of `--gen-coef`.

### Comparison methods

Each folder in `comparisons/` has a README that describes the method, its adaptation to these
environments and any deviations from the original paper. In short:

```bash
# CoinRun
cd comparisons/coinrun_ibac_sni     && bash run.sh             && python compare.py
cd comparisons/coinrun_surprise_min && python run_sm.py        && python compare.py
cd comparisons/coinrun_ipo          && python run_ipo.py       && python compare.py

# MiniGrid
cd comparisons/minigrid
python run.py --method ibac-sni
python run.py --method sm --alpha 1e-6
python run.py --method ipo
python compare.py
```

Every method uses the same protocol as gen-aware PPO:
- 10 agents, with seeds `8000 + 137·i`
- 2M steps per agent
- a generalization evaluation every 100K steps (100 episodes)
- a final 1000-episode evaluation on unseen levels

## Citation

If you use this code, please cite the paper:

```bibtex
@article{TODO,
  title  = {TODO},
  author = {TODO},
  year   = {TODO}
}
```
