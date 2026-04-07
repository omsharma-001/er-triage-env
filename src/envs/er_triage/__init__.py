"""
envs.er_triage — Hospital ER Triage OpenEnv environment.

Public API:
    from envs.er_triage import ERTriageClient          # HTTP client
    from envs.er_triage import ERTriageEnvironment     # Direct Python env
    from envs.er_triage.models import (
        ERTriageAction,       # What the agent sends
        ERTriageObservation,  # What the agent sees
        ERTriageReward,       # Typed reward breakdown
        StepResult,           # Return type of step()
        ERTriageState,        # Episode metadata
        PatientVitals,        # Per-patient clinical data
    )
    from envs.er_triage.graders.graders import grade   # Score an agent 0-1
"""

from envs.er_triage.client import ERTriageClient
from envs.er_triage.server.environment import ERTriageEnvironment
from envs.er_triage.models import (
    ERTriageAction,
    ERTriageObservation,
    ERTriageReward,
    ERTriageState,
    PatientVitals,
    StepResult,
)

__all__ = [
    "ERTriageClient",
    "ERTriageEnvironment",
    "ERTriageAction",
    "ERTriageObservation",
    "ERTriageReward",
    "ERTriageState",
    "PatientVitals",
    "StepResult",
]
