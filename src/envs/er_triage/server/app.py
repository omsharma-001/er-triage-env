"""
app.py — FastAPI server for ERTriageEnv.

OpenEnv-spec endpoints:
  GET  /health      liveness check
  GET  /tasks       list tasks with metadata
  POST /reset       {"task": "easy"|"medium"|"hard", "seed": int?}
  POST /step        {"patient_index": int}   returns StepResult
  GET  /state       episode metadata
  POST /grade       {"task": str, "actions": [int]}  returns score 0.0-1.0

Port 7860 for HF Spaces. Override with PORT env var.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from envs.er_triage.server.environment import ERTriageEnvironment, TASK_CONFIG
from envs.er_triage.models import (
    ERTriageAction, ERTriageObservation, ERTriageReward,
    ERTriageState, StepResult, PatientVitals,
)
from envs.er_triage.graders.graders import grade as run_grade, greedy_agent

PORT = int(os.getenv("PORT", "7860"))

app = FastAPI(
    title="ERTriageEnv",
    description="Hospital ER triage reinforcement learning environment (OpenEnv spec).",
    version="2.1.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_env: Optional[ERTriageEnvironment] = None


# ── request bodies ─────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task: str = Field("easy", description="easy | medium | hard")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")

class StepRequest(BaseModel):
    patient_index: int = Field(..., description="Patient index to admit, or -1 to hold")

class GradeRequest(BaseModel):
    task:    str       = Field(..., description="Task name")
    actions: List[int] = Field(..., description="Sequence of patient_index actions to replay")


# ── response models ─────────────────────────────────────────────────────────────

class TaskInfo(BaseModel):
    name:              str
    description:       str
    queue_size:        int
    beds:              int
    max_steps:         int
    comorbidities:     bool
    noisy_vitals:      bool
    surge_events:      bool
    action_space:      str
    observation_space: str

class GradeOut(BaseModel):
    task:              str
    score:             float
    deaths_score:      float
    priority_score:    float
    efficiency_score:  float
    avg_deaths:        float
    avg_priority_rate: float
    avg_hold_rate:     float
    episodes:          int
    explanation:       str


# ── helpers ─────────────────────────────────────────────────────────────────────

def _hide_true_severity(obs: ERTriageObservation, task: str) -> ERTriageObservation:
    """Remove severity_true from hard-mode observations (it's hidden from the agent)."""
    if task != "hard":
        return obs
    patients = [p.model_copy(update={"severity_true": None}) for p in obs.patients]
    return obs.model_copy(update={"patients": patients})


# ── endpoints ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "2.1.0"}


@app.get("/tasks", response_model=List[TaskInfo])
def list_tasks():
    descs = {
        "easy":   "10 patients, 3 beds. Clean vitals, no comorbidities. Learn basic severity ranking.",
        "medium": "20 patients, 5 beds. Comorbidities amplify risk. Fast deterioration rates.",
        "hard":   "30 patients, 8 beds. Noisy vitals, surge arrivals every 10 steps, bed timeouts.",
    }
    return [
        TaskInfo(
            name=task, description=descs[task],
            queue_size=cfg["queue_size"], beds=cfg["beds"], max_steps=cfg["max_steps"],
            comorbidities=cfg["comorbidities"], noisy_vitals=cfg["noisy_vitals"],
            surge_events=cfg["surge_events"],
            action_space="Discrete(queue_size+1) — patient_index: int, -1=hold",
            observation_space=(
                "patients: List[PatientVitals(11 fields)] | "
                "beds_available: int | step: int | done: bool | "
                "reward: ERTriageReward | info: dict"
            ),
        )
        for task, cfg in TASK_CONFIG.items()
    ]


@app.post("/reset", response_model=ERTriageObservation)
def reset(body: ResetRequest):
    """Start a new episode. Returns initial ERTriageObservation (reward=None)."""
    global _env
    if body.task not in TASK_CONFIG:
        raise HTTPException(400, f"Unknown task '{body.task}'. Choose: {list(TASK_CONFIG)}")
    _env = ERTriageEnvironment(task=body.task, seed=body.seed)
    obs  = _env.reset()
    return _hide_true_severity(obs, body.task)


@app.post("/step", response_model=StepResult)
def step(body: StepRequest):
    """
    Take one action. Returns StepResult with:
      observation, reward (ERTriageReward), done, info
    """
    if _env is None or _env.state is None:
        raise HTTPException(400, "Call POST /reset first")
    result = _env.step(ERTriageAction(patient_index=body.patient_index))
    obs    = _hide_true_severity(result.observation, _env.task)
    return result.model_copy(update={"observation": obs})


@app.get("/state", response_model=ERTriageState)
def state():
    """Return episode metadata without advancing the environment."""
    if _env is None or _env.state is None:
        raise HTTPException(400, "Call POST /reset first")
    return _env.state


@app.post("/grade", response_model=GradeOut)
def grade_endpoint(body: GradeRequest):
    """
    Grade an action sequence. Replays actions across 10 fixed-seed episodes.
    Returns composite score in [0.0, 1.0].
    """
    if body.task not in TASK_CONFIG:
        raise HTTPException(400, f"Unknown task '{body.task}'")
    actions_iter = list(body.actions)

    def replay_agent(obs_dict):
        return actions_iter.pop(0) if actions_iter else greedy_agent(obs_dict)

    r = run_grade(body.task, replay_agent)
    explanation = (
        f"Score={r['score']:.4f} | "
        f"Deaths {r['deaths_score']:.3f}×0.50 | "
        f"Priority {r['priority_score']:.3f}×0.30 | "
        f"Efficiency {r['efficiency_score']:.3f}×0.20"
    )
    return GradeOut(explanation=explanation, **r)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=False)
