"""
server/environment.py — OpenEnv-native ERTriageEnvironment.

Wraps the simulation engine from src/ and adapts it to the openenv
Environment interface: reset(seed, episode_id) -> Observation,
step(action) -> Observation, state -> State, get_metadata() -> EnvironmentMetadata.

The reward field on each observation is a plain float (openenv spec).
The full reward breakdown (severity_saved, wait_penalty, etc.) is
stored in observation.metadata["reward_breakdown"] for agents that
want it.
"""

import os, sys
_ROOT = os.path.dirname(os.path.dirname(__file__))
_SRC  = os.path.join(_ROOT, "src")
for _p in [_ROOT, _SRC]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from typing import Any, Dict, List, Optional
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

try:
    from models import ERTriageAction, ERTriageObservation, ERTriageState, PatientOut
except ModuleNotFoundError:
    from server.models import ERTriageAction, ERTriageObservation, ERTriageState, PatientOut

# Import the simulation engine
from envs.er_triage.server.environment import ERTriageEnvironment as _SimEnv, TASK_CONFIG
from envs.er_triage.models import ERTriageAction as _SimAction


def _task_from_seed_or_env(seed: Optional[int] = None) -> str:
    """Read task from ER_TASK env var (default: easy)."""
    return os.getenv("ER_TASK", "easy")


class ERTriageEnvironment(Environment):
    """
    OpenEnv-compatible ER Triage environment.

    Wraps the full simulation (src/envs/er_triage/) and exposes it
    through the openenv Environment interface.

    Set ER_TASK=easy|medium|hard environment variable to choose difficulty.
    The task can also be passed via reset(task="hard") in metadata kwargs.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task: Optional[str] = None, seed: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        chosen_task = task or _task_from_seed_or_env()
        if chosen_task not in TASK_CONFIG:
            chosen_task = "easy"
        self._sim = _SimEnv(task=chosen_task, seed=seed)
        self._task = chosen_task

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task: Optional[str] = None,
        **kwargs,
    ) -> ERTriageObservation:
        # Allow task override via kwargs (e.g. from POST /reset body extras)
        chosen_task = task or kwargs.get("task") or _task_from_seed_or_env()
        if chosen_task != self._task or chosen_task not in TASK_CONFIG:
            chosen_task = chosen_task if chosen_task in TASK_CONFIG else "easy"
            self._sim = _SimEnv(task=chosen_task, seed=seed)
            self._task = chosen_task
        else:
            self._sim.seed = seed

        obs = self._sim.reset()
        return self._to_obs(obs, reward=None)

    def step(
        self,
        action: ERTriageAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> ERTriageObservation:
        sim_action = _SimAction(patient_index=action.patient_index)
        result = self._sim.step(sim_action)
        return self._to_obs(result.observation, reward=result.reward.total,
                            reward_breakdown=result.reward.model_dump())

    @property
    def state(self) -> ERTriageState:
        s = self._sim.state
        if s is None:
            return ERTriageState()
        return ERTriageState(
            episode_id=s.episode_id,
            step_count=s.step_count,
            task=s.task,
            admitted_count=s.admitted_count,
            deaths=s.deaths,
            total_reward=s.total_reward,
            queue_size=s.queue_size,
            beds_total=s.beds_total,
            beds_available=s.beds_available,
        )

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="er_triage_env",
            description=(
                "Hospital Emergency Room triage environment. An AI agent acts as a "
                "triage nurse: observe patients' vitals, decide who to admit next, "
                "minimise deaths and wait-time penalties. Three tasks: easy, medium, hard. "
                "Severity scoring derived from NEWS2 (UK NHS clinical scoring system)."
            ),
            version="2.1.0",
            author="Ambuj Singh",
        )

    # ── helper ──────────────────────────────────────────────────────────────

    def _to_obs(self, sim_obs, reward=None, reward_breakdown=None) -> ERTriageObservation:
        hide = self._task == "hard"
        patients = [
            PatientOut(
                patient_id=p.patient_id,
                heart_rate=round(p.heart_rate, 1),
                bp_systolic=round(p.bp_systolic, 1),
                spo2=round(p.spo2, 1),
                respiratory_rate=round(p.respiratory_rate, 1),
                temperature=round(p.temperature, 2),
                pain_score=round(p.pain_score, 1),
                age=round(p.age, 0),
                wait_minutes=round(p.wait_minutes, 1),
                deterioration_rate=round(p.deterioration_rate, 4),
                comorbidities=p.comorbidities,
                severity_noisy=round(p.severity_noisy, 3),
                severity_true=None if hide else p.severity_true,
                done=False,
                reward=None,
            )
            for p in sim_obs.patients
        ]
        meta: Dict = {
            "admitted": self._sim.state.admitted_count if self._sim.state else 0,
            "deaths":   self._sim.state.deaths         if self._sim.state else 0,
            "task":     self._task,
        }
        if reward_breakdown:
            meta["reward_breakdown"] = reward_breakdown

        # Build prompt string for TRL compatibility
        cfg = TASK_CONFIG[self._task]
        prompt = (
            f"You are an ER triage nurse. {len(sim_obs.patients)} patients are waiting. "
            f"{sim_obs.beds_available} beds available. "
            f"Task: {self._task}. Step {sim_obs.step}/{cfg['max_steps']}. "
            f"Choose the patient index (0-{len(sim_obs.patients)-1}) to admit, or -1 to hold. "
            f"Prioritise by: SpO2<90 (critical), BP<80 (shock), deterioration_rate>0.1 (crashing)."
        )

        # Build messages history for TRL compatibility
        from server.models import Message
        messages = []
        if reward is not None and reward_breakdown:
            rb = reward_breakdown
            messages.append(Message(
                category="REWARD",
                content=(
                    f"Last action reward: {rb.get('total', reward):.3f} "
                    f"(severity_saved={rb.get('severity_saved',0):.2f}, "
                    f"wait_penalty={rb.get('wait_penalty',0):.2f}, "
                    f"deaths={meta.get('deaths',0)})"
                ),
                done=False,
                reward=None,
            ))
        messages.append(Message(
            category="OBSERVATION",
            content=prompt,
            done=sim_obs.done,
            reward=None,
        ))

        return ERTriageObservation(
            patients=patients,
            beds_available=sim_obs.beds_available,
            step=sim_obs.step,
            task=self._task,
            done=sim_obs.done,
            reward=round(reward, 4) if reward is not None else None,
            metadata=meta,
            prompt=prompt,
            messages=messages,
        )
