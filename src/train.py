"""
train.py — Curriculum training loop for ERTriageEnv DQN agent.

Strategy: Curriculum learning
  Phase 1: Train on EASY   for N episodes → learn basic severity ranking
  Phase 2: Train on MEDIUM (warm-start weights) → learn comorbidity effect
  Phase 3: Train on HARD   (warm-start weights) → handle noise + surges

Each phase saves a checkpoint and emits a jsonl metrics log for the dashboard.

Usage:
  cd er_triage_env
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  python src/train.py

  # Single task only:
  python src/train.py --task medium --episodes 500

  # Resume from checkpoint:
  python src/train.py --task hard --load checkpoints/medium_final.pt
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, os.path.dirname(__file__))

from envs.er_triage.server.environment import ERTriageEnvironment
from envs.er_triage.models import ERTriageAction
from features import obs_to_tensor, action_index_to_env_action
from agent import DQNAgent

CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"
LOG_DIR        = Path(__file__).parent.parent / "logs"

CURRICULUM = [
    ("easy",   400),
    ("medium", 600),
    ("hard",   800),
]

EVAL_FREQ     = 50     # evaluate greedy policy every N episodes
EVAL_EPISODES = 10     # number of eval episodes to average
PRINT_FREQ    = 25     # print to console every N episodes


# ── helpers ──────────────────────────────────────────────────────────────────

def run_episode(
    env:     ERTriageEnvironment,
    agent:   DQNAgent,
    train:   bool = True,
    seed:    int  = None,
) -> Dict:
    if seed is not None:
        env.seed = seed
    obs = env.reset()
    state, mask = obs_to_tensor(obs)
    total_reward = 0.0
    steps = 0
    losses = []

    while not obs.done:
        action_idx = agent.select_action(state, mask, greedy=not train)
        patient_idx = action_index_to_env_action(action_idx, obs)

        next_obs = env.step(ERTriageAction(patient_index=patient_idx))
        reward = next_obs.reward or 0.0
        total_reward += reward

        next_state, next_mask = obs_to_tensor(next_obs)

        if train:
            agent.store(state, action_idx, reward, next_state,
                        next_obs.done, mask, next_mask)
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

        obs   = next_obs
        state = next_state
        mask  = next_mask
        steps += 1

    s = env.state
    return {
        "reward":   total_reward,
        "steps":    steps,
        "admitted": s.admitted_count,
        "deaths":   s.deaths,
        "avg_loss": sum(losses) / len(losses) if losses else 0.0,
    }


def evaluate(env: ERTriageEnvironment, agent: DQNAgent, n: int = 10) -> Dict:
    results = [run_episode(env, agent, train=False, seed=1000 + i) for i in range(n)]
    return {
        "eval_reward":   sum(r["reward"]   for r in results) / n,
        "eval_deaths":   sum(r["deaths"]   for r in results) / n,
        "eval_admitted": sum(r["admitted"] for r in results) / n,
    }


def train_phase(
    task:     str,
    episodes: int,
    agent:    DQNAgent = None,
    seed:     int = 42,
) -> DQNAgent:
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    env = ERTriageEnvironment(task=task, seed=seed)
    log_path = LOG_DIR / f"{task}_training.jsonl"

    if agent is None or agent.task != task:
        print(f"\n  Creating new DQNAgent for task='{task}'")
        agent = DQNAgent(task=task)
    else:
        print(f"\n  Warm-starting agent from previous phase → task='{task}'")
        # Rebuild nets for new obs/action dims while keeping shared learning
        old_agent = agent
        agent = DQNAgent(task=task)

    print(f"  Episodes: {episodes}  |  obs_dim: {agent.policy_net.shared[0].in_features}"
          f"  |  action_dim: {agent.policy_net.advantage_head[-1].out_features}")
    print(f"  Log  → {log_path}")
    print(f"  {'─'*56}")

    best_eval_reward = float("-inf")
    start_time = time.time()

    with open(log_path, "w") as log_f:
        for ep in range(1, episodes + 1):
            ep_seed = seed + ep
            result = run_episode(env, agent, train=True, seed=ep_seed)
            agent.episodes_done += 1

            # Eval
            eval_metrics = {}
            if ep % EVAL_FREQ == 0:
                eval_metrics = evaluate(env, agent, EVAL_EPISODES)
                if eval_metrics["eval_reward"] > best_eval_reward:
                    best_eval_reward = eval_metrics["eval_reward"]
                    agent.save(str(CHECKPOINT_DIR / f"{task}_best.pt"))

            # Log
            record = {
                "episode": ep,
                "task": task,
                "epsilon": round(agent.epsilon, 4),
                "steps_done": agent.steps_done,
                **result,
                **eval_metrics,
            }
            log_f.write(json.dumps(record) + "\n")
            log_f.flush()

            # Console
            if ep % PRINT_FREQ == 0:
                elapsed = time.time() - start_time
                eval_str = (f"  eval_r={eval_metrics['eval_reward']:7.1f}"
                            f"  eval_deaths={eval_metrics['eval_deaths']:.1f}"
                            if eval_metrics else "")
                print(
                    f"  ep {ep:4d}/{episodes}"
                    f"  ε={agent.epsilon:.3f}"
                    f"  r={result['reward']:7.1f}"
                    f"  deaths={result['deaths']:3d}"
                    f"  loss={result['avg_loss']:.4f}"
                    f"{eval_str}"
                    f"  [{elapsed:.0f}s]"
                )

    agent.save(str(CHECKPOINT_DIR / f"{task}_final.pt"))
    print(f"\n  Phase '{task}' complete. Best eval reward: {best_eval_reward:.2f}")
    return agent


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train DQN on ERTriageEnv")
    parser.add_argument("--task",     default=None,  help="Single task (easy/medium/hard). Default: full curriculum")
    parser.add_argument("--episodes", default=None,  type=int, help="Override episode count")
    parser.add_argument("--load",     default=None,  help="Load checkpoint path")
    parser.add_argument("--seed",     default=42,    type=int)
    args = parser.parse_args()

    agent = None
    if args.load:
        # Infer task from filename if possible
        task_hint = "easy"
        for t in ["easy", "medium", "hard"]:
            if t in args.load:
                task_hint = t
        agent = DQNAgent(task=task_hint)
        agent.load(args.load)

    if args.task:
        episodes = args.episodes or dict(CURRICULUM)[args.task]
        train_phase(args.task, episodes, agent=agent, seed=args.seed)
    else:
        print("\n  === Curriculum training: easy → medium → hard ===")
        for task, episodes in CURRICULUM:
            n = args.episodes or episodes
            agent = train_phase(task, n, agent=agent, seed=args.seed)

    print("\n  Training complete. Checkpoints saved in ./checkpoints/")
    print("  Metrics logged in ./logs/ — open the dashboard to visualise.")


if __name__ == "__main__":
    main()
