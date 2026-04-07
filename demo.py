#!/usr/bin/env python3
"""
demo.py — Demonstrate ERTriageEnv with three agents across all tasks.

No server needed — runs the environment directly in Python.

  cd er_triage_env
  python demo.py              # full comparison table
  python demo.py --verbose    # step-by-step trace for easy/greedy
  python demo.py --task hard  # single task only

Agents compared:
  RandomAgent    — uniform random patient selection
  GreedyAgent    — always admits highest severity_noisy
  WaitTimeAgent  — admits whoever has waited longest (FIFO)
  OracleAgent    — admits highest severity_TRUE (upper bound; cheats)
"""

import argparse
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from envs.er_triage.server.environment import ERTriageEnvironment
from envs.er_triage.models import ERTriageAction, StepResult


# ── Agents ────────────────────────────────────────────────────────────────────

class RandomAgent:
    name = "Random"
    def act(self, obs) -> ERTriageAction:
        if not obs.patients or obs.beds_available == 0:
            return ERTriageAction(patient_index=-1)
        return ERTriageAction(patient_index=random.randint(0, len(obs.patients) - 1))


class GreedyAgent:
    """Admits the patient with highest severity_noisy — the practical baseline."""
    name = "Greedy"
    def act(self, obs) -> ERTriageAction:
        if not obs.patients or obs.beds_available == 0:
            return ERTriageAction(patient_index=-1)
        best = max(range(len(obs.patients)), key=lambda i: obs.patients[i].severity_noisy)
        return ERTriageAction(patient_index=best)


class WaitTimeAgent:
    """Admits whoever waited longest — ignores clinical severity entirely."""
    name = "FIFO"
    def act(self, obs) -> ERTriageAction:
        if not obs.patients or obs.beds_available == 0:
            return ERTriageAction(patient_index=-1)
        longest = max(range(len(obs.patients)), key=lambda i: obs.patients[i].wait_minutes)
        return ERTriageAction(patient_index=longest)


class OracleAgent:
    """Admits the patient with highest severity_TRUE (ground truth — upper bound).
    In hard mode severity_true is hidden from the agent; oracle bypasses this."""
    name = "Oracle*"
    def act(self, obs) -> ERTriageAction:
        if not obs.patients or obs.beds_available == 0:
            return ERTriageAction(patient_index=-1)
        best = max(
            range(len(obs.patients)),
            key=lambda i: (obs.patients[i].severity_true or obs.patients[i].severity_noisy)
        )
        return ERTriageAction(patient_index=best)


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(env: ERTriageEnvironment, agent, verbose: bool = False):
    obs = env.reset()
    total_reward = 0.0

    while not obs.done:
        action = agent.act(obs)
        result: StepResult = env.step(action)

        total_reward += result.reward.total
        if verbose:
            r = result.reward
            print(
                f"  step {obs.step:3d} | queue={len(obs.patients):3d} "
                f"beds={obs.beds_available} | action={action.patient_index:3d} | "
                f"reward={r.total:7.2f}  "
                f"[sev={r.severity_saved:+.2f} pri={r.priority_bonus:+.2f} "
                f"wait={r.wait_penalty:+.2f} death={r.death_penalty:+.2f}]"
            )
        obs = result.observation

    s = env.state
    return total_reward, s.admitted_count, s.deaths


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(task: str, n_episodes: int = 10, seed: int = 42):
    agents = [RandomAgent(), GreedyAgent(), WaitTimeAgent(), OracleAgent()]
    env    = ERTriageEnvironment(task=task, seed=seed)

    print(f"\n  {'Agent':<18} {'Avg reward':>12} {'Avg admitted':>14} {'Avg deaths':>11}")
    print(f"  {'─'*18} {'─'*12} {'─'*14} {'─'*11}")

    for agent in agents:
        rewards, admitted, deaths = [], [], []
        for ep in range(n_episodes):
            env.seed = seed + ep
            r, a, d = run_episode(env, agent)
            rewards.append(r)
            admitted.append(a)
            deaths.append(d)

        avg_r = sum(rewards)  / n_episodes
        avg_a = sum(admitted) / n_episodes
        avg_d = sum(deaths)   / n_episodes
        star  = "  ← practical best" if agent.name == "Greedy" else (
                "  ← upper bound"     if agent.name == "Oracle*" else "")
        print(f"  {agent.name:<18} {avg_r:>12.2f} {avg_a:>14.1f} {avg_d:>11.1f}{star}")


# ── Verbose trace ─────────────────────────────────────────────────────────────

def verbose_trace(task: str = "easy", seed: int = 0):
    env   = ERTriageEnvironment(task=task, seed=seed)
    agent = GreedyAgent()
    print(f"\n  Greedy agent on {task.upper()} (seed={seed}) — step-by-step:")
    print(f"  {'─'*80}")
    run_episode(env, agent, verbose=True)
    s = env.state
    print(f"  {'─'*80}")
    print(f"  Episode ended: admitted={s.admitted_count}  deaths={s.deaths}  "
          f"total_reward={s.total_reward:.2f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ERTriageEnv demo")
    parser.add_argument("--task",     choices=["easy","medium","hard","all"], default="all")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--verbose",  action="store_true")
    args = parser.parse_args()

    print("=" * 64)
    print("  ERTriageEnv — Agent Comparison Demo")
    print("  (* Oracle sees ground-truth severity — upper bound only)")
    print("=" * 64)

    tasks = ["easy", "medium", "hard"] if args.task == "all" else [args.task]

    for task in tasks:
        print(f"\n{'─'*64}")
        print(f"  TASK: {task.upper()}")
        evaluate(task, n_episodes=args.episodes, seed=args.seed)

    if args.verbose:
        verbose_trace(task=tasks[0], seed=args.seed)


if __name__ == "__main__":
    main()
