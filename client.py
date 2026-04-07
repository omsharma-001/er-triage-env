"""
client.py — ERTriageEnv Python client (OpenEnv-spec, TRL-compatible).

Provides ERTriageEnv — a WebSocket-based persistent session client that
integrates directly with TRL's GRPOTrainer rollout_func.

Usage (synchronous, recommended for TRL):
    from client import ERTriageEnv, ERTriageAction

    # Connect to a running server
    with ERTriageEnv(base_url="http://localhost:7860").sync() as env:
        result = env.reset(task="easy", seed=42)
        while not result.done:
            action = ERTriageAction(patient_index=0)
            result = env.step(action)

    # Connect from HF Space (pulls Docker image automatically)
    env = ERTriageEnv.from_env("your-username/er-triage-env").sync()
    with env:
        result = env.reset()

Usage in TRL rollout_func:
    from client import ERTriageEnv, ERTriageAction
    from trl.experimental.openenv import generate_rollout_completions

    env = ERTriageEnv.from_env("your-username/er-triage-env").sync()

    def rollout_func(prompts, trainer):
        outputs = generate_rollout_completions(trainer, prompts)
        env.reset(task="hard")
        rewards = []
        for out in outputs:
            text = tokenizer.decode(out["completion_ids"], skip_special_tokens=True)
            patient_idx = parse_action(text)
            result = env.step(ERTriageAction(patient_index=patient_idx))
            rewards.append(float(result.reward or 0.0))
        return {
            "prompt_ids":      [o["prompt_ids"] for o in outputs],
            "completion_ids":  [o["completion_ids"] for o in outputs],
            "logprobs":        [o["logprobs"] for o in outputs],
            "env_reward":      rewards,
        }
"""

import os
import sys
from typing import Dict, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in [_HERE, os.path.join(_HERE, "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from server.models import ERTriageAction, ERTriageObservation, ERTriageState


class ERTriageEnv(
    EnvClient[ERTriageAction, ERTriageObservation, ERTriageState]
):
    """
    WebSocket client for ERTriageEnv.

    Inherits from openenv.core.EnvClient — gives you:
      .sync()                         → SyncEnvClient wrapper for non-async code
      .from_env("user/space-name")    → pull Docker image from HF and start locally
      .from_docker_image("image:tag") → start from a local Docker image

    The client uses persistent WebSocket sessions (lower latency, stateful).
    For TRL training, use the .sync() wrapper inside rollout_func.
    """

    def _step_payload(self, action: ERTriageAction) -> Dict:
        return {"patient_index": action.patient_index}

    def _parse_result(self, payload: Dict) -> StepResult[ERTriageObservation]:
        obs_data = payload.get("observation", payload)
        patients_raw = obs_data.get("patients", [])
        from server.models import PatientOut
        patients = [
            PatientOut(
                patient_id=p["patient_id"],
                heart_rate=p["heart_rate"],
                bp_systolic=p["bp_systolic"],
                spo2=p["spo2"],
                respiratory_rate=p["respiratory_rate"],
                temperature=p["temperature"],
                pain_score=p["pain_score"],
                age=p["age"],
                wait_minutes=p["wait_minutes"],
                deterioration_rate=p["deterioration_rate"],
                comorbidities=p.get("comorbidities", 0),
                severity_noisy=p.get("severity_noisy", 0.0),
                severity_true=p.get("severity_true"),
                done=False,
                reward=None,
            )
            for p in patients_raw
        ]
        reward_raw = payload.get("reward")
        if isinstance(reward_raw, dict):
            reward_val = reward_raw.get("total")
        else:
            reward_val = reward_raw

        observation = ERTriageObservation(
            patients=patients,
            beds_available=obs_data.get("beds_available", 0),
            step=obs_data.get("step", 0),
            task=obs_data.get("task", "easy"),
            done=payload.get("done", False),
            reward=reward_val,
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=reward_val,
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> ERTriageState:
        return ERTriageState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task=payload.get("task", "easy"),
            admitted_count=payload.get("admitted_count", 0),
            deaths=payload.get("deaths", 0),
            total_reward=payload.get("total_reward", 0.0),
            queue_size=payload.get("queue_size", 0),
            beds_total=payload.get("beds_total", 0),
            beds_available=payload.get("beds_available", 0),
        )
