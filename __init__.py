"""
er_triage_env — Hospital ER Triage OpenEnv environment.

Install from HF Space:
    pip install git+https://huggingface.co/spaces/<your-username>/er-triage-env

Usage:
    from client import ERTriageEnv, ERTriageAction

    env = ERTriageEnv.from_env("your-username/er-triage-env").sync()
    with env:
        result = env.reset(task="medium")
        while not result.done:
            result = env.step(ERTriageAction(patient_index=0))
"""

from client import ERTriageEnv
from server.models import ERTriageAction, ERTriageObservation, ERTriageState

__all__ = ["ERTriageEnv", "ERTriageAction", "ERTriageObservation", "ERTriageState"]
