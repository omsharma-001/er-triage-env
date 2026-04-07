"""
models.py — Pydantic typed contracts for ERTriageEnv.

OpenEnv spec requires typed Pydantic models for Observation, Action, Reward.
All HTTP serialisation goes through these models — no separate Pydantic/dataclass split.

Patient vitals schema (NEWS2-inspired):
  heart_rate        bpm          normal 60-100
  bp_systolic       mmHg         normal 90-140
  spo2              %            normal 95-100
  respiratory_rate  breaths/min  normal 12-20
  temperature       Celsius      normal 36.1-37.2
  pain_score        0-10         self-reported
  age               years
  wait_minutes      minutes in queue
  deterioration_rate  severity increase per step (0.0-0.25)
  comorbidities     count of chronic conditions (medium/hard)
  severity_noisy    estimated risk 0.0-1.0 (noisy in hard mode)
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


# ── Patient ───────────────────────────────────────────────────────────────────

class PatientVitals(BaseModel):
    patient_id:         int
    heart_rate:         float  = Field(..., description="bpm, normal 60-100")
    bp_systolic:        float  = Field(..., description="mmHg, normal 90-140")
    spo2:               float  = Field(..., description="% oxygen saturation, normal 95-100")
    respiratory_rate:   float  = Field(..., description="breaths/min, normal 12-20")
    temperature:        float  = Field(..., description="Celsius, normal 36.1-37.2")
    pain_score:         float  = Field(..., description="0-10 self-reported")
    age:                float  = Field(..., description="years")
    wait_minutes:       float  = Field(..., description="minutes already waiting")
    deterioration_rate: float  = Field(..., description="severity increase per step")
    comorbidities:      int    = Field(0,   description="count of chronic conditions")
    severity_noisy:     float  = Field(0.0, description="estimated risk score 0.0-1.0")
    # severity_true exposed only in easy/medium; hidden in hard
    severity_true:      Optional[float] = Field(None, description="ground-truth severity (None in hard mode)")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "patient_id": 3, "heart_rate": 118.5, "bp_systolic": 82.3,
            "spo2": 91.2, "respiratory_rate": 26.0, "temperature": 38.9,
            "pain_score": 8.0, "age": 71.0, "wait_minutes": 12.5,
            "deterioration_rate": 0.0842, "comorbidities": 2,
            "severity_noisy": 0.743, "severity_true": None,
        }
    })


# ── Action ────────────────────────────────────────────────────────────────────

class ERTriageAction(BaseModel):
    """
    The agent's decision: which patient to admit to treatment.
    patient_index: 0-based index into the current patient list.
    Use -1 to hold the queue this step (penalised when beds are free).
    """
    patient_index: int = Field(
        ...,
        ge=-1,
        description="0-based index of patient to admit, or -1 to hold queue",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(json_schema_extra={"example": {"patient_index": 2}})


# ── Reward ────────────────────────────────────────────────────────────────────

class ERTriageReward(BaseModel):
    """
    Typed reward breakdown for the last action.
    total is the scalar reward passed to the RL algorithm.
    The breakdown fields explain *why* that reward was assigned.
    """
    total:          float = Field(..., description="Scalar reward for this step")
    severity_saved: float = Field(0.0, description="+severity_true × 10 for admitted patient")
    priority_bonus: float = Field(0.0, description="+2 if agent chose highest-severity patient")
    wait_penalty:   float = Field(0.0, description="-0.5 × total_wait_minutes / 60")
    death_penalty:  float = Field(0.0, description="-20 per patient who died this step")
    hold_penalty:   float = Field(0.0, description="-5 if held queue when beds were free")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "total": 6.32, "severity_saved": 8.50, "priority_bonus": 2.0,
            "wait_penalty": -3.68, "death_penalty": 0.0, "hold_penalty": 0.0,
        }
    })


# ── Observation ───────────────────────────────────────────────────────────────

class ERTriageObservation(BaseModel):
    """
    Full state visible to the agent at each step.
    patients: current waiting room queue (variable length).
    """
    patients:       List[PatientVitals] = Field(default_factory=list)
    beds_available: int   = Field(0,      description="Free treatment beds right now")
    step:           int   = Field(0,      description="Current step within this episode")
    task:           str   = Field("easy", description="Task difficulty: easy | medium | hard")
    done:           bool  = Field(False,  description="True when episode has ended")
    reward:         Optional[ERTriageReward] = Field(None, description="Reward for last action; None after reset()")
    info:           Dict[str, Any] = Field(default_factory=dict, description="Extra metadata")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "patients": [], "beds_available": 2, "step": 4,
            "task": "medium", "done": False,
            "reward": {"total": 6.32, "severity_saved": 8.5, "priority_bonus": 2.0,
                       "wait_penalty": -3.68, "death_penalty": 0.0, "hold_penalty": 0.0},
            "info": {"admitted": 3, "deaths": 1},
        }
    })


# ── StepResult ────────────────────────────────────────────────────────────────

class StepResult(BaseModel):
    """
    Canonical return type of step(). OpenEnv spec:
      step(action) → observation, reward, done, info
    """
    observation: ERTriageObservation
    reward:      ERTriageReward
    done:        bool
    info:        Dict[str, Any] = Field(default_factory=dict)


# ── State ─────────────────────────────────────────────────────────────────────

class ERTriageState(BaseModel):
    """Episode-level metadata returned by state()."""
    episode_id:     Optional[str] = None
    step_count:     int   = 0
    task:           str   = "easy"
    admitted_count: int   = 0
    deaths:         int   = 0
    total_reward:   float = 0.0
    queue_size:     int   = 0
    beds_total:     int   = 0
    beds_available: int   = 0
