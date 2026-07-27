"""Small public contract between the course runner and a student agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


SUBMISSION_API_VERSION = 1


class SubmissionContractError(RuntimeError):
    """The submitted agent does not follow the required interface."""


@dataclass(frozen=True, slots=True)
class AgentConfig:
    role: str
    observation_dim: int
    num_actions: int
    total_steps: int
    seed: int
    device: str

    def __post_init__(self) -> None:
        if self.role not in {"red", "blue"}:
            raise ValueError("role must be 'red' or 'blue'")
        if self.observation_dim <= 0 or self.num_actions <= 0:
            raise ValueError("observation_dim and num_actions must be positive")
        if self.total_steps < 0:
            raise ValueError("total_steps cannot be negative")


@dataclass(frozen=True, slots=True)
class Observation:
    features: np.ndarray
    action_mask: np.ndarray


@dataclass(frozen=True, slots=True)
class Transition:
    observation: Observation
    action: int
    reward: float
    next_observation: Observation
    done: bool


@runtime_checkable
class StudentAgent(Protocol):
    def reset(self, seed: int) -> None: ...

    def act(self, observation: Observation, training: bool) -> int: ...

    def learn(self, transition: Transition) -> None: ...


class AgentFactory(Protocol):
    def __call__(self, config: AgentConfig) -> StudentAgent: ...


def copy_observation(
    observation,
    *,
    observation_dim: int,
    num_actions: int,
) -> Observation:
    try:
        features = np.asarray(observation["features"], dtype=np.float32)
        action_mask = np.asarray(observation["action_mask"], dtype=bool)
    except KeyError as error:
        raise SubmissionContractError(
            f"Observation is missing {error.args[0]!r}"
        ) from error
    if features.shape != (observation_dim,):
        raise SubmissionContractError(
            f"Expected features shape {(observation_dim,)}, got {features.shape}"
        )
    if action_mask.shape != (num_actions,):
        raise SubmissionContractError(
            f"Expected action mask shape {(num_actions,)}, got {action_mask.shape}"
        )
    if not np.all(np.isfinite(features)):
        raise SubmissionContractError("Observation contains non-finite values")
    if not np.any(action_mask):
        raise SubmissionContractError("Observation has no valid action")
    return Observation(features.copy(), action_mask.copy())


def validate_action(action: object, observation: Observation, num_actions: int) -> int:
    if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)):
        raise SubmissionContractError("act() must return an integer")
    action = int(action)
    if not 0 <= action < num_actions:
        raise SubmissionContractError(
            f"Action {action} is outside [0, {num_actions})"
        )
    if not bool(observation.action_mask[action]):
        raise SubmissionContractError(f"Action {action} is masked out")
    return action
