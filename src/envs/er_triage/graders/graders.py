"""
graders.py — Deterministic 0.0-1.0 scoring for each ERTriage task.

Each grader runs N_EVAL_EPISODES with fixed seeds, collects clinical metrics,
and returns a composite score in [0.0, 1.0].

Score weights:
  deaths_score     0.50  — deaths avoided vs worst-case baseline
  priority_score   0.30  — fraction of admit steps where near-optimal patient chosen
  efficiency_score 0.20  — low hold-when-free rate

All scores are deterministic: same agent + same seeds = same score every run.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from typing import Callable, Dict
from envs.er_triage.server.environment import ERTriageEnvironment, TASK_CONFIG
from envs.er_triage.models import ERTriageAction

EVAL_SEEDS       = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
N_EVAL_EPISODES  = 10

# Empirical baselines
ORACLE_DEATHS        = {"easy": 0.0,  "medium": 2.0,  "hard":  6.0}
WORST_DEATHS         = {"easy": 7.0,  "medium": 18.0, "hard":  28.0}
ORACLE_PRIORITY_RATE = {"easy": 1.0,  "medium": 0.90, "hard":  0.75}
WORST_PRIORITY_RATE  = {"easy": 0.20, "medium": 0.20, "hard":  0.20}
ORACLE_HOLD_RATE     = {"easy": 0.0,  "medium": 0.0,  "hard":  0.05}
WORST_HOLD_RATE      = {"easy": 0.5,  "medium": 0.5,  "hard":  0.50}

# AgentFn receives a plain dict (JSON-serialisable) and returns patient_index int
AgentFn = Callable[[dict], int]


def _obs_to_dict(obs) -> dict:
    """Convert ERTriageObservation (or the .observation of a StepResult) to plain dict."""
    # Handle both ERTriageObservation directly and StepResult wrappers
    if hasattr(obs, "observation"):
        obs = obs.observation
    return {
        "patients": [
            {
                "patient_id":         p.patient_id,
                "heart_rate":         p.heart_rate,
                "bp_systolic":        p.bp_systolic,
                "spo2":               p.spo2,
                "respiratory_rate":   p.respiratory_rate,
                "temperature":        p.temperature,
                "pain_score":         p.pain_score,
                "age":                p.age,
                "wait_minutes":       p.wait_minutes,
                "deterioration_rate": p.deterioration_rate,
                "comorbidities":      p.comorbidities,
                "severity_noisy":     p.severity_noisy,
                "severity_true":      p.severity_true,
            }
            for p in obs.patients
        ],
        "beds_available": obs.beds_available,
        "step":           obs.step,
        "task":           obs.task,
    }


def _run_graded_episode(env: ERTriageEnvironment, agent_fn: AgentFn, seed: int) -> Dict:
    env.seed = seed
    obs = env.reset()

    deaths = 0
    priority_hits = 0
    total_admit_steps = 0
    hold_when_free = 0
    total_steps = 0

    while not obs.done:
        obs_dict = _obs_to_dict(obs)
        action_idx = agent_fn(obs_dict)

        # Priority quality: did agent choose near-optimal patient?
        if obs.patients and obs.beds_available > 0 and action_idx >= 0:
            true_sevs = [p.severity_true or p.severity_noisy for p in obs.patients]
            best_true = max(true_sevs)
            total_admit_steps += 1
            if action_idx < len(obs.patients):
                chosen_sev = obs.patients[action_idx].severity_true or obs.patients[action_idx].severity_noisy
                if chosen_sev >= best_true * 0.90:
                    priority_hits += 1

        if action_idx == -1 and obs.beds_available > 0 and obs.patients:
            hold_when_free += 1

        result = env.step(ERTriageAction(patient_index=action_idx))
        obs = result.observation
        deaths = env.state.deaths  # track running total
        total_steps += 1

    priority_rate = priority_hits / total_admit_steps if total_admit_steps > 0 else 0.0
    hold_rate     = hold_when_free / total_steps if total_steps > 0 else 0.0

    return {
        "deaths":        env.state.deaths,
        "priority_rate": priority_rate,
        "hold_rate":     hold_rate,
        "admitted":      env.state.admitted_count,
        "total_reward":  env.state.total_reward,
    }


def grade(task: str, agent_fn: AgentFn) -> Dict:
    """
    Run N_EVAL_EPISODES and return:
      score, deaths_score, priority_score, efficiency_score,
      avg_deaths, avg_priority_rate, avg_hold_rate, episodes, task
    """
    assert task in TASK_CONFIG
    env     = ERTriageEnvironment(task=task)
    results = [_run_graded_episode(env, agent_fn, seed) for seed in EVAL_SEEDS]

    avg_deaths        = sum(r["deaths"]        for r in results) / N_EVAL_EPISODES
    avg_priority_rate = sum(r["priority_rate"] for r in results) / N_EVAL_EPISODES
    avg_hold_rate     = sum(r["hold_rate"]     for r in results) / N_EVAL_EPISODES

    def norm(val, worst, oracle):
        if worst == oracle: return 1.0
        return max(0.0, min(1.0, (worst - val) / (worst - oracle)))

    deaths_score     = norm(avg_deaths,        WORST_DEATHS[task],        ORACLE_DEATHS[task])
    priority_score   = norm(avg_priority_rate, WORST_PRIORITY_RATE[task], ORACLE_PRIORITY_RATE[task])
    efficiency_score = norm(avg_hold_rate,     WORST_HOLD_RATE[task],     ORACLE_HOLD_RATE[task])

    score = round(max(0.0, min(1.0,
        0.50 * deaths_score + 0.30 * priority_score + 0.20 * efficiency_score
    )), 4)

    return {
        "score":             score,
        "deaths_score":      round(deaths_score,     4),
        "priority_score":    round(priority_score,   4),
        "efficiency_score":  round(efficiency_score, 4),
        "avg_deaths":        round(avg_deaths,        2),
        "avg_priority_rate": round(avg_priority_rate, 4),
        "avg_hold_rate":     round(avg_hold_rate,     4),
        "episodes":          N_EVAL_EPISODES,
        "task":              task,
    }


# ── reference agents ──────────────────────────────────────────────────────────

def greedy_agent(obs_dict: dict) -> int:
    patients = obs_dict["patients"]
    if not patients or obs_dict["beds_available"] == 0:
        return -1
    return max(range(len(patients)), key=lambda i: patients[i]["severity_noisy"])

def random_agent(obs_dict: dict) -> int:
    import random
    patients = obs_dict["patients"]
    if not patients or obs_dict["beds_available"] == 0:
        return -1
    return random.randint(0, len(patients) - 1)

def fifo_agent(obs_dict: dict) -> int:
    patients = obs_dict["patients"]
    if not patients or obs_dict["beds_available"] == 0:
        return -1
    return max(range(len(patients)), key=lambda i: patients[i]["wait_minutes"])


if __name__ == "__main__":
    print("Reference grader scores\n")
    for task in ["easy", "medium", "hard"]:
        for name, agent in [("greedy", greedy_agent), ("random", random_agent), ("fifo", fifo_agent)]:
            r = grade(task, agent)
            print(f"  [{task:6s}] {name:8s}  score={r['score']:.4f}  deaths={r['avg_deaths']:.1f}  priority={r['avg_priority_rate']:.3f}")
        print()
