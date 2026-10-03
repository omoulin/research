"""Train IBAC-SNI agents with the exact protocol of the main gen-aware PPO experiment.

Protocol (copied from run_coinrun_vec_staged.run_phase3):
  * CoinRun (procgen native vec env, easy), 200 training levels, 64 envs
  * 10 agents x 2,000,000 steps, seeds 8000 + i*137
  * GenEval every 100K steps: 100 deterministic episodes on unseen levels
    (start_level=10000) -> curve saved as (N, 2) array [timestep, score]
  * final evaluation: 1000 deterministic episodes on unseen levels

The main experiment's baseline curves are only read (by compare.py), never
written. Use --variant baseline to retrain baseline PPO in this software stack
as a reproducibility check; it goes to results/baseline_rerun/.

Resumable: agents whose curve file already exists are skipped.

Usage (from this directory):
    python run_ibac_sni.py                       # IBAC-SNI, lambda=0.5, 10 agents
    python run_ibac_sni.py --variant ibac        # IBAC without SNI
    python run_ibac_sni.py --variant baseline    # re-run baseline PPO
    python run_ibac_sni.py --quick               # smoke test (2 agents, 200K steps)
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # never pick up gen_aware_ppo's modules

from coinrun_vec_env import make_train_env, make_eval_env  # noqa: E402
from config import Config, VARIANTS  # noqa: E402
from ibac_policy import IBACPolicy  # noqa: E402
from ibac_sni_ppo import IBACSNIPPO  # noqa: E402


# --------------------------------------------------------------------------- #
# Generalization evaluation - same logic as gen_aware_ppo's GenEvalCallback      #
# and evaluate_generalization()                                               #
# --------------------------------------------------------------------------- #
class GenEvalCallback(BaseCallback):
    def __init__(self, cfg: Config, verbose: int = 1) -> None:
        super().__init__(verbose)
        self.cfg = cfg
        self.results: list[tuple[int, float]] = []
        self._last_eval_t = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_t >= self.cfg.gen_eval_freq:
            self._last_eval_t = self.num_timesteps
            score = evaluate(self.model, self.cfg, self.cfg.gen_eval_episodes, seed=0)
            self.results.append((self.num_timesteps, score))
            if self.verbose:
                print(f"  [GenEval] t={self.num_timesteps:>10,}  gen_score={score:.3f}  "
                      f"(n={self.cfg.gen_eval_episodes})", flush=True)
        return True

    def save(self, path: str) -> None:
        np.save(path, np.array(self.results, dtype=np.float64))
        print(f"  [GenEval] curve saved to {path}  ({len(self.results)} points)")


def evaluate(model, cfg: Config, n_episodes: int, seed: int) -> float:
    env = make_eval_env(cfg, seed=seed)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=n_episodes, deterministic=True, warn=False)
    env.close()
    return float(mean_reward)


# --------------------------------------------------------------------------- #
# Model construction                                                          #
# --------------------------------------------------------------------------- #
def ppo_kwargs(cfg: Config) -> dict:
    return dict(
        n_steps=cfg.n_steps, batch_size=cfg.batch_size, n_epochs=cfg.n_epochs,
        learning_rate=cfg.learning_rate, ent_coef=cfg.ent_coef, clip_range=cfg.clip_range,
        gamma=cfg.gamma, gae_lambda=cfg.gae_lambda, max_grad_norm=cfg.max_grad_norm,
        verbose=1,
    )


def apply_weight_decay(model, l2_weight: float) -> None:
    """Weight decay on weights only (not biases), as in the paper's coinrun code
    (L2_WEIGHT * sum tf.nn.l2_loss(w) → gradient l2_weight * w)."""
    if l2_weight <= 0:
        return
    decay, no_decay = [], []
    for name, p in model.policy.named_parameters():
        (decay if p.dim() > 1 else no_decay).append(p)
    opt = model.policy.optimizer
    model.policy.optimizer = type(opt)(
        [dict(params=decay, weight_decay=l2_weight), dict(params=no_decay, weight_decay=0.0)],
        **{k: v for k, v in opt.defaults.items() if k != "weight_decay"},
    )


def build_model(cfg: Config, variant: str, seed: int, device: str):
    env = make_train_env(cfg, seed=seed)
    if variant == "baseline" or cfg.beta is None:
        model = PPO("CnnPolicy", env, seed=seed, device=device, **ppo_kwargs(cfg))
    else:
        model = IBACSNIPPO(
            IBACPolicy, env, seed=seed, device=device,
            beta=cfg.beta, sni_lambda=cfg.sni_lambda,
            policy_kwargs=dict(bottleneck_dim=cfg.bottleneck_dim, n_samples=cfg.n_samples, sni=cfg.sni),
            **ppo_kwargs(cfg),
        )
    apply_weight_decay(model, cfg.l2_weight)
    return model


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", default="ibac-sni", choices=list(VARIANTS) + ["baseline"])
    p.add_argument("--agents", type=int, default=None, help="override number of agents")
    p.add_argument("--start", type=int, default=0, help="first agent index")
    p.add_argument("--timesteps", type=int, default=None, help="override training steps")
    p.add_argument("--beta", type=float, default=None, help="override IB weight")
    p.add_argument("--quick", action="store_true", help="smoke test: 2 agents, 200K steps, 20-episode evals")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(HERE)

    if args.device.startswith("cuda") and not th.cuda.is_available():
        sys.exit("CUDA requested but torch.cuda.is_available() is False - refusing to fall back to CPU.")

    cfg = Config()
    if args.variant != "baseline":
        cfg = dataclasses.replace(cfg, **VARIANTS[args.variant])
    if args.beta is not None:
        cfg.beta = args.beta
    if args.quick:
        cfg.n_agents, cfg.gen_timesteps = 2, 200_000
        cfg.gen_eval_freq, cfg.gen_eval_episodes, cfg.n_eval_episodes = 20_000, 20, 100
    if args.agents is not None:
        cfg.n_agents = args.agents
    if args.timesteps is not None:
        cfg.gen_timesteps = args.timesteps

    name = "baseline_rerun" if args.variant == "baseline" else args.variant
    if args.beta is not None:
        name += f"_beta{args.beta:g}"
    if args.quick:
        name = f"quick_{name}"
    out_dir = os.path.join(cfg.results_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(dict(dataclasses.asdict(cfg), variant=args.variant), f, indent=2)

    tag = "baseline" if args.variant == "baseline" else "ibac"
    print(f"torch {th.__version__} | device={args.device} "
          f"({th.cuda.get_device_name(0) if th.cuda.is_available() else 'cpu'})")
    print(f"variant={args.variant} | {cfg.n_agents} agents x {cfg.gen_timesteps:,} steps | out={out_dir}")
    print(json.dumps(dataclasses.asdict(cfg)))

    finals_path = os.path.join(out_dir, "final_scores.json")
    finals = json.load(open(finals_path)) if os.path.exists(finals_path) else {}

    for i in range(args.start, cfg.n_agents):
        curve_path = os.path.join(out_dir, f"gen_curve_{tag}_{i:02d}.npy")
        if os.path.exists(curve_path):
            print(f"  [{name} {i+1}/{cfg.n_agents}]  already exists, skipping")
            continue
        seed = cfg.baseline_seed(i) if args.variant == "baseline" else cfg.seed(i)
        print(f"\n  [{name} {i+1}/{cfg.n_agents}]  training (seed={seed})", flush=True)
        t0 = time.time()
        model = build_model(cfg, args.variant, seed, args.device)
        assert next(model.policy.parameters()).device.type == th.device(args.device).type
        cb = GenEvalCallback(cfg)
        model.learn(total_timesteps=cfg.gen_timesteps, callback=cb)
        model.env.close()
        cb.save(curve_path)
        th.save(model.policy.state_dict(), os.path.join(out_dir, f"policy_{i:02d}.pt"))

        # same seed convention as gen_aware_ppo: baseline i+100, treatment i+1
        final = evaluate(model, cfg, cfg.n_eval_episodes, seed=(i + 100 if tag == "baseline" else i + 1))
        finals[f"{i:02d}"] = dict(seed=seed, final_gen_score=final, curve_last=cb.results[-1][1],
                                  minutes=(time.time() - t0) / 60)
        json.dump(finals, open(finals_path, "w"), indent=2)
        print(f"  [{name} {i+1}/{cfg.n_agents}]  done  final gen_score={final:.3f} "
              f"({cfg.n_eval_episodes} ep)  in {(time.time()-t0)/60:.1f} min", flush=True)

    print(f"\n{name}: all {cfg.n_agents} agents done. Run `python compare.py` for the comparison.")


if __name__ == "__main__":
    main()
