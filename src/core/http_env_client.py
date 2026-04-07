"""
http_env_client.py — Base HTTP client for OpenEnv environments.

Subclass ERTriageClient (or your own client) from this base.
The base handles all HTTP plumbing; subclasses implement the
two abstract methods to translate between JSON payloads and
typed Pydantic models.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional, TypeVar

import requests

ActionT = TypeVar("ActionT")
ObsT    = TypeVar("ObsT")


class HTTPEnvClient(ABC, Generic[ActionT, ObsT]):
    """
    Base class for HTTP-based OpenEnv clients.

    Usage pattern:
        env = MyEnvClient("http://localhost:7860")
        obs  = env.reset(task="easy", seed=42)
        result = env.step(MyAction(field=value))
        print(result.reward.total, result.done)
        state = env.state()
    """

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")

    # ── public API ────────────────────────────────────────────────────────────

    def reset(self, task: str = "easy", seed: Optional[int] = None) -> ObsT:
        """Start a new episode. Returns initial observation (reward=None)."""
        body: Dict[str, Any] = {"task": task}
        if seed is not None:
            body["seed"] = seed
        resp = requests.post(f"{self.base_url}/reset", json=body, timeout=30)
        resp.raise_for_status()
        return self._parse_observation(resp.json())

    def step(self, action: ActionT):
        """Take one action. Returns StepResult with observation, reward, done, info."""
        payload = self._action_to_payload(action)
        resp = requests.post(f"{self.base_url}/step", json=payload, timeout=30)
        resp.raise_for_status()
        return self._parse_step_result(resp.json())

    def state(self) -> Dict[str, Any]:
        """Return episode metadata without advancing the environment."""
        resp = requests.get(f"{self.base_url}/state", timeout=30)
        resp.raise_for_status()
        return resp.json()

    def tasks(self):
        """List all available tasks."""
        resp = requests.get(f"{self.base_url}/tasks", timeout=30)
        resp.raise_for_status()
        return resp.json()

    def health(self) -> Dict[str, Any]:
        resp = requests.get(f"{self.base_url}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()

    # ── abstract ──────────────────────────────────────────────────────────────

    @abstractmethod
    def _action_to_payload(self, action: ActionT) -> Dict[str, Any]:
        """Convert typed action to JSON-serialisable dict for POST /step."""
        ...

    @abstractmethod
    def _parse_observation(self, payload: Dict[str, Any]) -> ObsT:
        """Parse POST /reset JSON response into typed observation."""
        ...

    @abstractmethod
    def _parse_step_result(self, payload: Dict[str, Any]):
        """Parse POST /step JSON response into typed StepResult."""
        ...
