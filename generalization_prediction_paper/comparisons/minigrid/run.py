"""Train IBAC-SNI / surprise-minimization / IPO agents on MiniGrid
SimpleCrossingS9N2 (symbolic vector obs) with the protocol of the reference
experiment (`main.py --env minigrid-simplecrossing-vec --gen-timesteps 2000000`):

  * 32 envs, training seeds 0..149 (cycled per worker), MlpPolicy-sized nets
  * 10 agents x 2,000,000 steps, seeds 8000 + i*137 (GenPPO arm's seeds)
  * GenEval every 100K steps: 100 deterministic episodes on unseen seeds
    10000.. (single env, as gen_aware_ppo's make_eval_env) → (N, 2) curve
  * final evaluation: 1000 deterministic episodes, eval seed offset i+1

Previous experiments are never re-run or written to. Resumable: agents whose
curve file already exists are skipped.

Usage (from this directory):
    python run.py --method ibac-sni
    python run.py --method sm --alpha 3e-7
    python run.py --method ipo
    python run.py --method <m> --quick        # 1 agent, 300K steps
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
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # never pick up gen_aware_ppo's modules

import simplecrossing_vec_env as scv  # noqa: E402  (verbatim copy of gen_aware_ppo's env module)
from config import Config  # noqa: E402
from methods import (IBACMlpPolicy, IBACSNIPPO, IPODomainPPO, IPOMlpPolicy,  # noqa: E402
                     SurpriseMinPPO)


def evaluate(model, cfg: Config, n_episodes: int, seed: int) -> float:
    env = scv.make_eval_env(cfg, seed=seed)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=n_episodes, deterministic=True, warn=False)
    env.close()
    return float(mean_reward)


class GenEvalCallback(BaseCallback):
    """Parent's GenEvalCallback; step counter can span several learners (IPO)."""

    def __init__(self, cfg: Config, total_steps=None, verbose: int = 1) -> None:
        super().__init__(verbose)
        self.cfg = cfg
        self.total_steps = total_steps
        self.results: list[tuple[int, float]] = []
        self._last_eval_t = 0

    def _on_step(self) -> bool:
        t = self.total_steps() if self.total_steps else self.num_timesteps
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


def ppo_kwargs(cfg: Config, lr_factor: float = 1.0, verbose: int = 1) -> dict:
    return dict(n_steps=cfg.n_steps, batch_size=cfg.batch_size, n_epochs=cfg.n_epochs,
                learning_rate=cfg.learning_rate * lr_factor, ent_coef=cfg.ent_coef, clip_range=cfg.clip_range,
                gamma=cfg.gamma, gae_lambda=cfg.gae_lambda, max_grad_norm=cfg.max_grad_norm, verbose=verbose)


def domain_env(cfg: Config, d: int, n_envs: int) -> VecMonitor:
    """IPO domain d: training seeds [d*P, (d+1)*P), P = 150 / n_domains, cycled
    per worker exactly like gen_aware_ppo's make_train_env."""
    pool = cfg.minigrid_train_num_seeds // cfg.ipo_n_domains

    def fn(worker_id):
        def _init():
            return scv._SeededWrapper(scv._make_base_env(), seed_start=d * pool, pool_size=pool,
                                      worker_id=worker_id, n_workers=n_envs)
        return _init
    return VecMonitor(DummyVecEnv([fn(i) for i in range(n_envs)]))


def train_single(cfg: Config, method: str, seed: int, device: str):
    env = scv.make_train_env(cfg, seed=seed)
    if method == "ibac-sni":
        model = IBACSNIPPO(IBACMlpPolicy, env, seed=seed, device=device,
                           beta=cfg.ibac_beta, sni_lambda=cfg.ibac_sni_lambda,
                           policy_kwargs=dict(bottleneck_dim=cfg.ibac_bottleneck_dim,
                                              n_samples=cfg.ibac_n_samples, sni=True),
                           **ppo_kwargs(cfg))
    else:
        model = SurpriseMinPPO("MlpPolicy", env, seed=seed, device=device,
                               sm_alpha=cfg.sm_alpha, sm_buffer_size=cfg.sm_buffer_size,
                               sm_var_eps=cfg.sm_var_eps, sm_center=cfg.sm_center, **ppo_kwargs(cfg))
    cb = GenEvalCallback(cfg)
    model.learn(total_timesteps=cfg.gen_timesteps, callback=cb)
    model.env.close()
    return model, cb


def train_ipo(cfg: Config, seed: int, device: str):
    n_env = cfg.n_envs // cfg.ipo_n_domains
    learners = [IPODomainPPO(IPOMlpPolicy, domain_env(cfg, d, n_env), domain=d, seed=seed + d, device=device,
                             policy_kwargs=dict(n_domains=cfg.ipo_n_domains),
                             **ppo_kwargs(cfg, cfg.ipo_lr_factor, verbose=0))
                for d in range(cfg.ipo_n_domains)]
    for a in learners[1:]:
        a.policy = learners[0].policy
    total = lambda: sum(a.num_timesteps for a in learners)  # noqa: E731
    cb = GenEvalCallback(cfg, total)
    wrapped = [a._setup_learn(cfg.gen_timesteps, cb, True, "ipo", False)[1] for a in learners]
    cb.on_training_start(locals(), globals())
    it = 0
    while total() < cfg.gen_timesteps:
        for a, c in zip(learners, wrapped):          # best-response dynamics
            a.collect_rollouts(a.env, c, a.rollout_buffer, n_rollout_steps=a.n_steps)
            a._update_current_progress_remaining(a.num_timesteps, cfg.gen_timesteps // cfg.ipo_n_domains)
            a.train()
        it += 1
        if it % 20 == 0:
            r = [np.mean([e["r"] for e in a.ep_info_buffer]) if a.ep_info_buffer else float("nan") for a in learners]
            print(f"  it={it:4d}  steps={total():>10,}  train ep_rew_mean per domain="
                  f"{', '.join(f'{x:.3f}' for x in r)}", flush=True)
    if not cb.results or cb.results[-1][0] < cfg.gen_timesteps:
        cb.evaluate_now(total())
    for a in learners:
        a.env.close()
    return learners[0], cb


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True, choices=["ibac-sni", "sm", "ipo"])
    p.add_argument("--alpha", type=float, default=None, help="SM reward weight (required for --method sm)")
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
    if args.method == "sm":
        if args.alpha is None:
            sys.exit("--alpha is required for --method sm")
        cfg.sm_alpha = args.alpha
    if args.quick:
        cfg.n_agents, cfg.gen_timesteps = 1, 300_000
    if args.agents is not None:
        cfg.n_agents = args.agents
    if args.timesteps is not None:
        cfg.gen_timesteps = args.timesteps

    name = args.method + (f"_alpha{args.alpha:g}" if args.method == "sm" else "")
    if args.quick:
        name = f"quick_{name}"
    out_dir = os.path.join(cfg.results_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(dict(dataclasses.asdict(cfg), method=args.method), open(os.path.join(out_dir, "config.json"), "w"), indent=2)
    print(f"torch {th.__version__} | device={args.device} "
          f"({th.cuda.get_device_name(0) if th.cuda.is_available() else 'cpu'})")
    print(f"{name} | {cfg.n_agents} agents x {cfg.gen_timesteps:,} steps | out={out_dir}")

    finals_path = os.path.join(out_dir, "final_scores.json")
    finals = json.load(open(finals_path)) if os.path.exists(finals_path) else {}
    tag = {"ibac-sni": "ibac", "sm": "sm", "ipo": "ipo"}[args.method]
    for i in range(args.start, cfg.n_agents):
        curve_path = os.path.join(out_dir, f"gen_curve_{tag}_{i:02d}.npy")
        if os.path.exists(curve_path):
            print(f"  [{name} {i+1}/{cfg.n_agents}]  already exists, skipping")
            continue
        seed = cfg.seed(i)
        print(f"\n  [{name} {i+1}/{cfg.n_agents}]  training (seed={seed})", flush=True)
        t0 = time.time()
        model, cb = train_ipo(cfg, seed, args.device) if args.method == "ipo" else \
            train_single(cfg, args.method, seed, args.device)
        assert next(model.policy.parameters()).device.type == th.device(args.device).type
        cb.save(curve_path)
        th.save(model.policy.state_dict(), os.path.join(out_dir, f"policy_{i:02d}.pt"))
        final = evaluate(model, cfg, cfg.n_eval_episodes, seed=i + 1)
        finals[f"{i:02d}"] = dict(seed=seed, final_gen_score=final, curve_last=cb.results[-1][1],
                                  minutes=(time.time() - t0) / 60)
        json.dump(finals, open(finals_path, "w"), indent=2)
        print(f"  [{name} {i+1}/{cfg.n_agents}]  done  final gen_score={final:.3f} "
              f"({cfg.n_eval_episodes} ep)  in {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"\n{name}: all {cfg.n_agents} agents done.")


if __name__ == "__main__":
    main()
