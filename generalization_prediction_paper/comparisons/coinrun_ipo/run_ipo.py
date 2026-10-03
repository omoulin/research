"""Train IPO agents (Sonar et al. 2021) with the protocol of the main gen-aware PPO
experiment.

Protocol (copied from run_coinrun_vec_staged.run_phase3):
  * CoinRun (procgen native vec env, easy), the same 200 training levels,
    64 envs in total (here 2 domains x 32 envs: levels 0-99 / 100-199)
  * 10 agents x 2,000,000 environment steps (summed over domains),
    seeds 8000 + i*137
  * GenEval every 100K steps: 100 deterministic episodes on unseen levels
    (start_level=10000, 64 envs) → curve saved as (N, 2) [timestep, score]
  * final evaluation: 1000 deterministic episodes on unseen levels

Previous experiments are never re-run or written to. Resumable: agents whose
curve file already exists are skipped.

Usage (from this directory):
    python run_ipo.py            # 10 agents
    python run_ipo.py --quick    # smoke test (2 agents, 200K steps)
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

from coinrun_vec_env import make_coinrun_vec, make_eval_env  # noqa: E402
from config import Config  # noqa: E402
from ipo import IPODomainPPO, IPOPolicy  # noqa: E402


def evaluate(model, cfg: Config, n_episodes: int, seed: int) -> float:
    env = make_eval_env(cfg, seed=seed)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=n_episodes, deterministic=True, warn=False)
    env.close()
    return float(mean_reward)


class GenEvalCallback(BaseCallback):
    """Parent's GenEvalCallback, but counting steps over all domain learners."""

    def __init__(self, cfg: Config, total_steps, verbose: int = 1) -> None:
        super().__init__(verbose)
        self.cfg = cfg
        self.total_steps = total_steps
        self.results: list[tuple[int, float]] = []
        self._last_eval_t = 0

    def _on_step(self) -> bool:
        t = self.total_steps()
        if t - self._last_eval_t >= self.cfg.gen_eval_freq:
            self.evaluate_now(t)
        return True

    def evaluate_now(self, t: int) -> None:
        self._last_eval_t = t
        score = evaluate(self.model, self.cfg, self.cfg.gen_eval_episodes, seed=0)
        self.results.append((t, score))
        if self.verbose:
            print(f"  [GenEval] t={t:>10,}  gen_score={score:.3f}  (n={self.cfg.gen_eval_episodes})", flush=True)

    def save(self, path: str) -> None:
        np.save(path, np.array(self.results, dtype=np.float64))
        print(f"  [GenEval] curve saved to {path}  ({len(self.results)} points)")


def train_ipo(cfg: Config, seed: int, device: str):
    envs_per_domain = cfg.n_envs // cfg.n_domains
    learners = []
    for d in range(cfg.n_domains):
        start, num = cfg.domain_levels(d)
        env = make_coinrun_vec(num_levels=num, start_level=start, n_envs=envs_per_domain, seed=seed + d)
        algo = IPODomainPPO(
            IPOPolicy, env, domain=d, seed=seed + d, device=device,
            policy_kwargs=dict(n_domains=cfg.n_domains),
            n_steps=cfg.n_steps, batch_size=cfg.batch_size, n_epochs=cfg.n_epochs,
            learning_rate=cfg.learning_rate * cfg.ipo_lr_factor, ent_coef=cfg.ent_coef,
            clip_range=cfg.clip_range, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda,
            max_grad_norm=cfg.max_grad_norm, verbose=0,
        )
        learners.append(algo)
    policy = learners[0].policy
    for algo in learners[1:]:
        algo.policy = policy                       # one shared (averaged) policy

    total = lambda: sum(a.num_timesteps for a in learners)  # noqa: E731
    cb = GenEvalCallback(cfg, total)
    wrapped = []
    for algo in learners:
        _, c = algo._setup_learn(cfg.gen_timesteps, cb, True, "ipo", False)
        wrapped.append(c)
    cb.on_training_start(locals(), globals())

    it = 0
    while total() < cfg.gen_timesteps:
        for algo, c in zip(learners, wrapped):     # best-response dynamics
            algo.collect_rollouts(algo.env, c, algo.rollout_buffer, n_rollout_steps=algo.n_steps)
            algo._update_current_progress_remaining(algo.num_timesteps, cfg.gen_timesteps // cfg.n_domains)
            algo.train()
        it += 1
        if it % 10 == 0:
            rews = [np.mean([e["r"] for e in a.ep_info_buffer]) if a.ep_info_buffer else float("nan") for a in learners]
            print(f"  it={it:4d}  steps={total():>10,}  train ep_rew_mean per domain="
                  f"{', '.join(f'{r:.2f}' for r in rews)}", flush=True)
    if not cb.results or cb.results[-1][0] < cfg.gen_timesteps:
        cb.evaluate_now(total())
    for algo in learners:
        algo.env.close()
    return learners[0], cb


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agents", type=int, default=None)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    os.chdir(HERE)

    if args.device.startswith("cuda") and not th.cuda.is_available():
        sys.exit("CUDA requested but torch.cuda.is_available() is False - refusing to fall back to CPU.")

    cfg = Config()
    if args.quick:
        cfg.n_agents, cfg.gen_timesteps = 2, 200_000
        cfg.gen_eval_freq, cfg.gen_eval_episodes, cfg.n_eval_episodes = 20_000, 20, 100
    if args.agents is not None:
        cfg.n_agents = args.agents
    if args.timesteps is not None:
        cfg.gen_timesteps = args.timesteps

    name = "quick_ipo" if args.quick else "ipo"
    out_dir = os.path.join(cfg.results_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(dataclasses.asdict(cfg), open(os.path.join(out_dir, "config.json"), "w"), indent=2)

    print(f"torch {th.__version__} | device={args.device} "
          f"({th.cuda.get_device_name(0) if th.cuda.is_available() else 'cpu'})")
    print(f"{name} | {cfg.n_agents} agents x {cfg.gen_timesteps:,} steps | domains="
          f"{[cfg.domain_levels(d) for d in range(cfg.n_domains)]} | out={out_dir}")
    print(json.dumps(dataclasses.asdict(cfg)))

    finals_path = os.path.join(out_dir, "final_scores.json")
    finals = json.load(open(finals_path)) if os.path.exists(finals_path) else {}

    for i in range(args.start, cfg.n_agents):
        curve_path = os.path.join(out_dir, f"gen_curve_ipo_{i:02d}.npy")
        if os.path.exists(curve_path):
            print(f"  [{name} {i+1}/{cfg.n_agents}]  already exists, skipping")
            continue
        seed = cfg.seed(i)
        print(f"\n  [{name} {i+1}/{cfg.n_agents}]  training (seed={seed})", flush=True)
        t0 = time.time()
        model, cb = train_ipo(cfg, seed, args.device)
        assert next(model.policy.parameters()).device.type == th.device(args.device).type
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
