"""
client.py — Python HTTP client for ERTriageEnv.

Drop-in replacement for talking to a running ERTriageEnv server.
Identical API to the direct Python environment — swap the import
and everything else stays the same.

Usage:
    from envs.er_triage.client import ERTriageClient
    from envs.er_triage.models import ERTriageAction

    env = ERTriageClient("http://localhost:7860")

    obs = env.reset(task="easy", seed=42)
    print(f"{len(obs.patients)} patients, {obs.beds_available} beds free")

    # Admit the most critical patient
    best = max(range(len(obs.patients)),
               key=lambda i: obs.patients[i].severity_noisy)
    result = env.step(ERTriageAction(patient_index=best))

    print(f"reward={result.reward.total:.2f}  done={result.done}")
    print(f"  severity_saved={result.reward.severity_saved:.2f}")
    print(f"  wait_penalty={result.reward.wait_penalty:.2f}")

    state = env.state()
    print(f"admitted={state['admitted_count']}  deaths={state['deaths']}")
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from typing import Any, Dict, Optional

from core.http_env_client import HTTPEnvClient
from envs.er_triage.models import (
    ERTriageAction,
    ERTriageObservation,
    ERTriageReward,
    PatientVitals,
    StepResult,
)


class ERTriageClient(HTTPEnvClient[ERTriageAction, ERTriageObservation]):
    """
    HTTP client for ERTriageEnv.

    Connects to a running server (Docker or local uvicorn) and exposes
    the same reset() / step() / state() interface as the direct Python env.
    """

    def _action_to_payload(self, action: ERTriageAction) -> Dict[str, Any]:
        return {"patient_index": action.patient_index}

    def _parse_observation(self, payload: Dict[str, Any]) -> ERTriageObservation:
        return ERTriageObservation.model_validate(payload)

    def _parse_step_result(self, payload: Dict[str, Any]) -> StepResult:
        return StepResult.model_validate(payload)
