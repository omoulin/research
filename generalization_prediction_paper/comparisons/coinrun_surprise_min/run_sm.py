"""Train PPO + Normal surprise-minimizing reward agents (Chen 2020) with the
exact protocol of the main gen-aware PPO experiment.

Protocol (copied from run_coinrun_vec_staged.run_phase3):
  * CoinRun (procgen native vec env, easy), 200 training levels, 64 envs
  * 10 agents x 2,000,000 steps, seeds 8000 + i*137
  * GenEval every 100K steps: 100 deterministic episodes on unseen levels
    (start_level=10000) → curve saved as (N, 2) array [timestep, score]
  * final evaluation: 1000 deterministic episodes on unseen levels

Previous experiments are never re-run or written to. Resumable: agents whose
curve file already exists are skipped.

Usage (from this directory):
    python run_sm.py                  # alpha = 1e-6, centered r_SM, 10 agents
    python run_sm.py --no-center --alpha 1e-4   # paper-exact (collapses)
    python run_sm.py --quick          # smoke test (2 agents, 200K steps)
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
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # never pick up gen_aware_ppo's modules

from coinrun_vec_env import make_train_env, make_eval_env  # noqa: E402
from config import Config  # noqa: E402
from sm_ppo import SurpriseMinPPO  # noqa: E402


class GenEvalCallback(BaseCallback):
    """Same logic as gen_aware_ppo's GenEvalCallback."""

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


def build_model(cfg: Config, seed: int, device: str) -> SurpriseMinPPO:
    env = make_train_env(cfg, seed=seed)
    return SurpriseMinPPO(
        "CnnPolicy", env, seed=seed, device=device,
        sm_alpha=cfg.sm_alpha, sm_buffer_size=cfg.sm_buffer_size, sm_var_eps=cfg.sm_var_eps,
        sm_center=cfg.sm_center,
        n_steps=cfg.n_steps, batch_size=cfg.batch_size, n_epochs=cfg.n_epochs,
        learning_rate=cfg.learning_rate, ent_coef=cfg.ent_coef, clip_range=cfg.clip_range,
        gamma=cfg.gamma, gae_lambda=cfg.gae_lambda, max_grad_norm=cfg.max_grad_norm,
        verbose=1,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--alpha", type=float, default=None, help="override SM reward weight")
    p.add_argument("--agents", type=int, default=None)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--no-center", action="store_true", help="paper-exact r_SM (no mean subtraction)")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    os.chdir(HERE)

    if args.device.startswith("cuda") and not th.cuda.is_available():
        sys.exit("CUDA requested but torch.cuda.is_available() is False - refusing to fall back to CPU.")

    cfg = Config()
    if args.alpha is not None:
        cfg.sm_alpha = args.alpha
    if args.no_center:
        cfg.sm_center = False
    if args.quick:
        cfg.n_agents, cfg.gen_timesteps = 2, 200_000
        cfg.gen_eval_freq, cfg.gen_eval_episodes, cfg.n_eval_episodes = 20_000, 20, 100
    if args.agents is not None:
        cfg.n_agents = args.agents
    if args.timesteps is not None:
        cfg.gen_timesteps = args.timesteps

    name = ("sm-normal-centered" if cfg.sm_center else "sm-normal")
    name += f"_alpha{args.alpha:g}" if args.alpha is not None else ""
    if args.quick:
        name = f"quick_{name}"
    out_dir = os.path.join(cfg.results_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(dataclasses.asdict(cfg), open(os.path.join(out_dir, "config.json"), "w"), indent=2)

    print(f"torch {th.__version__} | device={args.device} "
          f"({th.cuda.get_device_name(0) if th.cuda.is_available() else 'cpu'})")
    print(f"{name} | {cfg.n_agents} agents x {cfg.gen_timesteps:,} steps | out={out_dir}")
    print(json.dumps(dataclasses.asdict(cfg)))

    finals_path = os.path.join(out_dir, "final_scores.json")
    finals = json.load(open(finals_path)) if os.path.exists(finals_path) else {}

    for i in range(args.start, cfg.n_agents):
        curve_path = os.path.join(out_dir, f"gen_curve_sm_{i:02d}.npy")
        if os.path.exists(curve_path):
            print(f"  [{name} {i+1}/{cfg.n_agents}]  already exists, skipping")
            continue
        seed = cfg.seed(i)
        print(f"\n  [{name} {i+1}/{cfg.n_agents}]  training (seed={seed})", flush=True)
        t0 = time.time()
        model = build_model(cfg, seed, args.device)
        assert next(model.policy.parameters()).device.type == th.device(args.device).type
        cb = GenEvalCallback(cfg)
        model.learn(total_timesteps=cfg.gen_timesteps, callback=cb)
        model.env.close()
        cb.save(curve_path)
        th.save(model.policy.state_dict(), os.path.join(out_dir, f"policy_{i:02d}.pt"))

        final = evaluate(model, cfg, cfg.n_eval_episodes, seed=i + 1)   # gen_aware_ppo's treatment-arm convention
        finals[f"{i:02d}"] = dict(seed=seed, final_gen_score=final, curve_last=cb.results[-1][1],
                                  minutes=(time.time() - t0) / 60)
        json.dump(finals, open(finals_path, "w"), indent=2)
        print(f"  [{name} {i+1}/{cfg.n_agents}]  done  final gen_score={final:.3f} "
              f"({cfg.n_eval_episodes} ep)  in {(time.time()-t0)/60:.1f} min", flush=True)

    print(f"\n{name}: all {cfg.n_agents} agents done. Run `python compare.py` for the comparison.")


if __name__ == "__main__":
    main()
