"""
server/models.py — OpenEnv-native Action/Observation models for ERTriageEnv.

These models inherit from openenv.core base types so create_app and
the openenv validate runtime checks work correctly:
  - Action inherits openenv Action  (extra=forbid, has metadata field)
  - Observation inherits openenv Observation (done, reward: float|None, metadata)
  - State inherits openenv State (episode_id, step_count, extra=allow)

The rich ERTriageReward breakdown is embedded in observation.metadata
so agents that want it can access it, while the base reward field stays
a plain float per the spec.
"""

from typing import Any, Dict, List, Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State, EnvironmentMetadata


class ERTriageAction(Action):
    """
    Triage decision: which patient to admit to a treatment bed.
    patient_index is the 0-based index into the patients list.
    Use -1 to hold the queue (penalised when beds are free).
    """
    patient_index: int = Field(
        ...,
        ge=-1,
        description="0-based index of patient to admit. -1 = hold queue.",
    )


class PatientOut(Observation):
    """Single patient's visible vitals (embedded inside ERTriageObservation)."""
    patient_id:         int   = Field(..., description="Unique patient ID")
    heart_rate:         float = Field(..., description="bpm, normal 60-100")
    bp_systolic:        float = Field(..., description="mmHg, normal 90-140")
    spo2:               float = Field(..., description="% saturation, normal 95-100")
    respiratory_rate:   float = Field(..., description="breaths/min, normal 12-20")
    temperature:        float = Field(..., description="Celsius")
    pain_score:         float = Field(..., description="0-10 self-reported")
    age:                float = Field(..., description="years")
    wait_minutes:       float = Field(..., description="minutes already waiting")
    deterioration_rate: float = Field(..., description="severity increase per step")
    comorbidities:      int   = Field(0, description="chronic condition count")
    severity_noisy:     float = Field(0.0, description="estimated risk 0.0-1.0")
    severity_true:      Optional[float] = Field(None, description="ground truth (hidden in hard)")



class Message(Observation):
    """A single message in the observation history (for TRL compatibility)."""
    category: str = Field("OBSERVATION", description="Message type/category")
    content:  str = Field("", description="Message text content")


class ERTriageObservation(Observation):
    """
    Full state visible to the agent at each step.

    reward: scalar reward for this step (None after reset).
            Full breakdown is in metadata["reward_breakdown"].
    prompt:   natural-language task description (TRL GRPOTrainer compatibility).
    messages: observation history list (TRL GRPOTrainer compatibility).
    """
    patients:       List[PatientOut]   = Field(default_factory=list)
    beds_available: int                = Field(0)
    step:           int                = Field(0)
    task:           str                = Field("easy")
    prompt:         str                = Field("", description="Task prompt for the agent")
    messages:       List[Message]      = Field(default_factory=list, description="Observation history")
    # reward inherited from Observation (float|None) — satisfies openenv spec
    # done  inherited from Observation


class ERTriageState(State):
    """Episode-level metadata. Inherits episode_id + step_count from openenv State."""
    task:           str   = Field("easy")
    admitted_count: int   = Field(0)
    deaths:         int   = Field(0)
    total_reward:   float = Field(0.0)
    queue_size:     int   = Field(0)
    beds_total:     int   = Field(0)
    beds_available: int   = Field(0)
