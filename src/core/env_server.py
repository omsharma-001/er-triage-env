from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, Optional, TypeVar

ActionT = TypeVar("ActionT", bound="Action")
ObsT = TypeVar("ObsT", bound="Observation")
StateT = TypeVar("StateT", bound="State")


@dataclass
class Action:
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Observation:
    done: bool = False
    reward: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class State:
    episode_id: Optional[str] = None
    step_count: int = 0


class Environment(ABC, Generic[ActionT, ObsT, StateT]):
    @abstractmethod
    def reset(self) -> ObsT: ...

    @abstractmethod
    def step(self, action: ActionT) -> ObsT: ...

    @property
    @abstractmethod
    def state(self) -> StateT: ...
