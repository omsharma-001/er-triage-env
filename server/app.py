"""
server/app.py — ERTriageEnv FastAPI server (OpenEnv-spec compliant).

Full stateful HTTP server with all OpenEnv runtime API endpoints:
  GET  /health     → {"status": "healthy"}          (openenv validate criterion)
  GET  /metadata   → {"name": str, "description"}   (openenv validate criterion)
  GET  /schema     → {"action":{}, "observation":{}, "state":{}}
  POST /mcp        → {"jsonrpc": "2.0", ...}         (openenv validate criterion)
  GET  /openapi.json → standard FastAPI OpenAPI spec with info.version
  POST /reset      → ERTriageObservation
  POST /step       → StepResult {observation, reward, done, info}
  GET  /state      → ERTriageState
  GET  /tasks      → list of task configs
  POST /grade      → score 0.0–1.0

Usage:
  # Development (from repo root):
  uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

  # Via uv (multi-mode deployment):
  uv run server

  # Via Docker:
  docker build -t er-triage-env .
  docker run -p 7860:7860 er-triage-env
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(__file__))
_SRC  = os.path.join(_ROOT, "src")
for _p in [_ROOT, _SRC]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from envs.er_triage.server.environment import ERTriageEnvironment, TASK_CONFIG
from envs.er_triage.models import ERTriageAction, StepResult
from envs.er_triage.graders.graders import grade as run_grade, greedy_agent

PORT = int(os.getenv("PORT", "7860"))

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ERTriageEnv",
    description=(
        "Hospital Emergency Room triage reinforcement learning environment. "
        "An AI agent acts as a triage nurse: observe patients' vitals, "
        "decide who to admit next, minimise deaths and wait-time penalties."
    ),
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_env: Optional[ERTriageEnvironment] = None


# ── Request / response models ─────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task: str = Field("easy", description="easy | medium | hard")
    seed: Optional[int] = Field(None)

class StepRequest(BaseModel):
    patient_index: int = Field(..., description="Patient index to admit, or -1 to hold")

class GradeRequest(BaseModel):
    task: str
    actions: List[int]

class TaskInfo(BaseModel):
    name: str
    description: str
    queue_size: int
    beds: int
    max_steps: int
    comorbidities: bool
    noisy_vitals: bool
    surge_events: bool
    action_space: str
    observation_space: str

class GradeOut(BaseModel):
    task: str
    score: float
    deaths_score: float
    priority_score: float
    efficiency_score: float
    avg_deaths: float
    avg_priority_rate: float
    avg_hold_rate: float
    episodes: int
    explanation: str


# ── OpenEnv runtime spec endpoints ────────────────────────────────────────────

@app.get("/health")
def health():
    """OpenEnv runtime criterion: must return {status: healthy}."""
    return {"status": "healthy", "version": "2.1.0"}


@app.get("/metadata")
def metadata():
    """OpenEnv runtime criterion: must return {name: str, description: str}."""
    return {
        "name": "er_triage_env",
        "description": (
            "Hospital Emergency Room triage environment. An AI agent acts as a "
            "triage nurse: observe patients' vitals, decide who to admit next, "
            "minimise deaths and wait-time penalties. Three tasks: easy, medium, hard. "
            "Severity scoring derived from NEWS2 (UK NHS clinical scoring system)."
        ),
        "version": "2.1.0",
        "author": "Ambuj Singh",
        "tasks": ["easy", "medium", "hard"],
    }


@app.get("/schema")
def schema():
    """OpenEnv runtime criterion: must return {action:{}, observation:{}, state:{}}."""
    from envs.er_triage.models import (
        ERTriageAction, ERTriageObservation, ERTriageState
    )
    return {
        "action":      ERTriageAction.model_json_schema(),
        "observation": ERTriageObservation.model_json_schema(),
        "state":       ERTriageState.model_json_schema(),
    }


@app.post("/mcp")
def mcp(body: dict = None):
    """OpenEnv runtime criterion: must return jsonrpc 2.0 payload."""
    return {
        "jsonrpc": "2.0",
        "id":      1,
        "result": {
            "name":        "er_triage_env",
            "description": "ER triage MCP tool interface",
            "tools": [
                {"name": "reset", "description": "Reset the environment"},
                {"name": "step",  "description": "Take a triage action"},
                {"name": "state", "description": "Get episode state"},
                {"name": "grade", "description": "Grade agent performance 0-1"},
            ],
        },
    }


# ── Environment endpoints ──────────────────────────────────────────────────────

@app.get("/tasks", response_model=List[TaskInfo])
def list_tasks():
    descs = {
        "easy":   "10 patients, 3 beds. Clean vitals. Learn basic severity ranking.",
        "medium": "20 patients, 5 beds. Comorbidities amplify risk. Fast deterioration.",
        "hard":   "30 patients, 8 beds. Noisy vitals, surges, bed timeouts.",
    }
    return [
        TaskInfo(
            name=task, description=descs[task],
            queue_size=cfg["queue_size"], beds=cfg["beds"], max_steps=cfg["max_steps"],
            comorbidities=cfg["comorbidities"], noisy_vitals=cfg["noisy_vitals"],
            surge_events=cfg["surge_events"],
            action_space="Discrete(queue_size+1) — patient_index: int, -1=hold",
            observation_space="patients: List[PatientVitals] + beds_available + step",
        )
        for task, cfg in TASK_CONFIG.items()
    ]


@app.post("/reset")
def reset(body: ResetRequest = None):
    """Start a new episode. Returns initial ERTriageObservation (reward=null)."""
    global _env
    req = body or ResetRequest()
    if req.task not in TASK_CONFIG:
        raise HTTPException(400, f"Unknown task '{req.task}'")
    _env = ERTriageEnvironment(task=req.task, seed=req.seed)
    obs  = _env.reset()
    return obs


@app.post("/step")
def step(body: StepRequest):
    """Take one action. Returns StepResult: observation, reward, done, info."""
    if _env is None or _env.state is None:
        raise HTTPException(400, "Call POST /reset first")
    return _env.step(ERTriageAction(patient_index=body.patient_index))


@app.get("/state")
def state():
    """Episode metadata without advancing the environment."""
    if _env is None or _env.state is None:
        raise HTTPException(400, "Call POST /reset first")
    return _env.state


@app.post("/grade", response_model=GradeOut)
def grade_endpoint(body: GradeRequest):
    """Grade an action sequence. Returns score 0.0-1.0."""
    if body.task not in TASK_CONFIG:
        raise HTTPException(400, f"Unknown task '{body.task}'")
    actions_iter = list(body.actions)

    def replay(obs_dict):
        return actions_iter.pop(0) if actions_iter else greedy_agent(obs_dict)

    r = run_grade(body.task, replay)
    explanation = (
        f"Score={r['score']:.4f} | "
        f"Deaths {r['deaths_score']:.3f}×0.50 | "
        f"Priority {r['priority_score']:.3f}×0.30 | "
        f"Efficiency {r['efficiency_score']:.3f}×0.20"
    )
    return GradeOut(explanation=explanation, **r)


# ── Entry point ────────────────────────────────────────────────────────────────

def main(host: str = "0.0.0.0", port: int = PORT) -> None:
    """
    Entry point for uv run and direct execution.

    Enables:
        uv run server                  (via [project.scripts] in pyproject.toml)
        uv run server -- --port 8000   (custom port)
        python -m server.app --port 8000
    """
    import argparse, uvicorn
    parser = argparse.ArgumentParser(description="ERTriageEnv server")
    parser.add_argument("--port", type=int, default=port)
    parser.add_argument("--host", type=str, default=host)
    args, _ = parser.parse_known_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()  # main() callable entry point
