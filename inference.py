#!/usr/bin/env python3
"""
inference.py — LLM-based triage agent for ERTriageEnv.

Uses OpenAI API client to make clinical triage decisions.
Reads credentials from environment variables per OpenEnv spec.

Required environment variables:
  OPENAI_API_KEY   Your OpenAI API key
  MODEL_NAME       Model to use (default: gpt-4o-mini)
  API_BASE_URL     API base URL (default: https://api.openai.com/v1)

Usage:
  export OPENAI_API_KEY=sk-...
  export MODEL_NAME=gpt-4o-mini
  python inference.py

  # Against running server:
  export ER_ENV_URL=http://localhost:7860
  python inference.py --mode http

  # Direct Python (no server, default):
  python inference.py --mode direct
"""

import argparse, json, os, sys, time
from typing import Dict, List

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai not installed. Run: pip install openai")
    sys.exit(1)

# ── credentials (OpenEnv spec: OPENAI_API_KEY) ───────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL_NAME     = os.environ.get("MODEL_NAME",     "gpt-4o-mini")
API_BASE_URL   = os.environ.get("API_BASE_URL",   "https://api.openai.com/v1")
ER_ENV_URL     = os.environ.get("ER_ENV_URL",     "http://localhost:7860")

if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY not set.")

client = OpenAI(api_key=OPENAI_API_KEY or "placeholder", base_url=API_BASE_URL)

# ── direct Python imports ────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from envs.er_triage.server.environment import ERTriageEnvironment
from envs.er_triage.models import ERTriageAction
from envs.er_triage.graders.graders import grade, greedy_agent

# ── system prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert emergency room triage nurse AI.
Given a waiting room of patients, decide which patient to admit to treatment next.

Clinical priority rules (NEWS2 system):
- SpO2 < 90%: life-threatening hypoxia → HIGHEST priority
- BP systolic < 80 mmHg: shock → HIGHEST priority
- Heart rate > 130 or < 40 bpm: critical arrhythmia
- Respiratory rate > 30 or < 8: respiratory failure
- deterioration_rate > 0.10: patient is crashing rapidly
- High comorbidities amplify all risks

Respond with ONLY valid JSON:
{"patient_index": <integer>}

Where patient_index is the 0-based index into the patients array.
Return -1 only when beds_available == 0. Never return -1 when beds are free.
"""


def _patient_summary(i: int, p: dict) -> str:
    return (
        f"[{i}] ID={p['patient_id']}  "
        f"HR={p['heart_rate']:.0f}bpm  BP={p['bp_systolic']:.0f}  "
        f"SpO2={p['spo2']:.1f}%  RR={p['respiratory_rate']:.0f}  "
        f"Temp={p['temperature']:.1f}°C  Pain={p['pain_score']:.0f}/10  "
        f"Age={p['age']:.0f}  Wait={p['wait_minutes']:.0f}min  "
        f"DeterioRate={p['deterioration_rate']:.3f}  "
        f"Comorbidities={p['comorbidities']}  "
        f"RiskScore={p['severity_noisy']:.3f}"
    )


def llm_decide(obs_dict: dict, history: List[Dict]) -> int:
    """Call LLM; fall back to greedy on any error."""
    if obs_dict["beds_available"] == 0:
        return -1

    lines = [
        f"Step {obs_dict['step']} | Beds free: {obs_dict['beds_available']} | "
        f"Patients waiting: {len(obs_dict['patients'])}",
        "",
        "Waiting room:",
    ]
    for i, p in enumerate(obs_dict["patients"]):
        lines.append(_patient_summary(i, p))
    lines += ["", "Respond with JSON only: {\"patient_index\": <int>}"]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-4:])  # keep last 2 turns for context
    messages.append({"role": "user", "content": "\n".join(lines)})

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME, messages=messages,
            temperature=0.0, max_tokens=32,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content.strip()
        idx = int(json.loads(content).get("patient_index", -1))
        n = len(obs_dict["patients"])
        if not (-1 <= idx < n):
            raise ValueError(f"index {idx} out of range [0,{n-1}]")
        history.append({"role": "assistant", "content": content})
        return idx
    except Exception as e:
        print(f"    [llm warn] {e} — greedy fallback")
        return greedy_agent(obs_dict)


def _build_obs_dict(obs) -> dict:
    return {
        "patients": [
            {"patient_id": p.patient_id, "heart_rate": p.heart_rate,
             "bp_systolic": p.bp_systolic, "spo2": p.spo2,
             "respiratory_rate": p.respiratory_rate, "temperature": p.temperature,
             "pain_score": p.pain_score, "age": p.age,
             "wait_minutes": p.wait_minutes, "deterioration_rate": p.deterioration_rate,
             "comorbidities": p.comorbidities, "severity_noisy": p.severity_noisy}
            for p in obs.patients
        ],
        "beds_available": obs.beds_available,
        "step": obs.step, "task": obs.task,
    }


def run_demo_episode(task: str, seed: int, agent_fn=None, verbose: bool = False) -> int:
    """Run one episode, print [STEP] blocks to stdout, return step count.

    If agent_fn is None, uses the LLM. Pass greedy_agent for a no-API-key demo.
    """
    env = ERTriageEnvironment(task=task, seed=seed)
    obs = env.reset()
    history: List[Dict] = []
    step_count = 0

    while not obs.done:
        obs_dict = _build_obs_dict(obs)
        if agent_fn is not None:
            action = agent_fn(obs_dict)
        else:
            action = llm_decide(obs_dict, history)
        result = env.step(ERTriageAction(patient_index=action))
        step_count += 1
        print(f"[STEP] step={step_count} reward={result.reward.total:.4f}", flush=True)
        if verbose:
            print(f"    step={obs.step:3d} queue={len(obs.patients):3d} "
                  f"beds={obs.beds_available} action={action:3d} "
                  f"reward={result.reward.total:.2f}")
        obs = result.observation

    if verbose:
        s = env.state
        print(f"    → admitted={s.admitted_count} deaths={s.deaths} reward={s.total_reward:.1f}")
    return step_count


def run_llm_episode(task: str, seed: int, verbose: bool = False) -> List[int]:
    env = ERTriageEnvironment(task=task, seed=seed)
    obs = env.reset()
    history: List[Dict] = []
    actions: List[int] = []

    while not obs.done:
        obs_dict = _build_obs_dict(obs)
        action = llm_decide(obs_dict, history)
        actions.append(action)
        result = env.step(ERTriageAction(patient_index=action))
        print(f"[STEP] step={result.observation.step} reward={result.reward.total:.4f}", flush=True)
        if verbose:
            print(f"    step={obs.step:3d} queue={len(obs.patients):3d} "
                  f"beds={obs.beds_available} action={action:3d} "
                  f"reward={result.reward.total:.2f}")
        obs = result.observation

    if verbose:
        s = env.state
        print(f"    → admitted={s.admitted_count} deaths={s.deaths} reward={s.total_reward:.1f}")
    return actions


def make_llm_agent():
    _history: List[Dict] = []
    def agent(obs_dict: dict) -> int:
        return llm_decide(obs_dict, _history)
    return agent


def main():
    parser = argparse.ArgumentParser(description="ERTriageEnv LLM inference + grading")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--greedy", action="store_true",
        help="Use greedy baseline agent instead of LLM (no API key required). "
             "Produces the reproducible baseline scores in scores.json.",
    )
    args = parser.parse_args()

    # Auto-fall-back to greedy when no API key is set
    use_greedy = args.greedy or not OPENAI_API_KEY
    if use_greedy and not args.greedy:
        print("INFO: OPENAI_API_KEY not set — using greedy baseline agent.")
        print("      Set OPENAI_API_KEY and re-run for LLM inference.")
        print()

    print("=" * 64)
    print("  ERTriageEnv — Inference & Grading")
    print(f"  Agent : {'Greedy baseline' if use_greedy else MODEL_NAME}")
    if not use_greedy:
        print(f"  API   : {API_BASE_URL}")
    print("=" * 64)

    results: Dict = {}
    t0 = time.time()

    for task in ["easy", "medium", "hard"]:
        print(f"\n[{task.upper()}]")
        print(f"[START] task={task}", flush=True)
        t1 = time.time()

        if use_greedy:
            # Reproducible baseline — no API calls
            print("  Demo episode (greedy)...")
            demo_steps = run_demo_episode(task, seed=args.seed, agent_fn=greedy_agent,
                                          verbose=args.verbose)
            print("  Grading with greedy agent (10 fixed-seed episodes)...")
            r = grade(task, greedy_agent)
        else:
            print("  Demo episode...")
            demo_actions = run_llm_episode(task, seed=args.seed, verbose=args.verbose)
            demo_steps = len(demo_actions)
            print("  Grading (10 episodes)...")
            agent_fn = make_llm_agent()
            r = grade(task, agent_fn)

        results[task] = r
        print(f"[END] task={task} score={r['score']:.4f} steps={demo_steps}", flush=True)
        print(f"  score={r['score']:.4f}  deaths={r['avg_deaths']:.1f}  "
              f"priority={r['avg_priority_rate']:.3f}  [{time.time()-t1:.1f}s]")

    total = time.time() - t0
    overall = sum(r["score"] for r in results.values()) / 3

    print("\n" + "=" * 64)
    print("  SCORES")
    print("=" * 64)
    for task, r in results.items():
        bar = "█" * int(r["score"] * 30)
        print(f"  {task:6s}  {r['score']:.4f}  [{bar:<30}]")
    print(f"\n  Overall: {overall:.4f}  |  Runtime: {total:.1f}s")

    payload = {
        "scores":  {t: r["score"] for t, r in results.items()},
        "details": results,
        "overall": round(overall, 4),
        "model":   MODEL_NAME,
        "runtime_seconds": round(total, 2),
    }
    with open("scores.json", "w") as f:
        json.dump(payload, f, indent=2)
    print("  scores.json written")


if __name__ == "__main__":
    main()
