"""Single-role interface for conventional reinforcement-learning agents."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np

from .agents import Agent
from .environment import CyberRLEnv
from .model import Role
from .scenario import ScenarioSpec


class EnvironmentOpponent(Protocol):
    """Opponent that can consume the simulator's structured observation."""

    name: str

    def reset(self, seed: int | None) -> None: ...

    def act(
        self,
        observation: Mapping[str, Any],
        environment: CyberRLEnv,
    ) -> int: ...


class _ScriptedOpponent:
    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self.name = agent.name
        self.rng = np.random.default_rng()

    def reset(self, seed: int | None) -> None:
        self.agent.reset()
        self.rng = np.random.default_rng(seed)

    def act(
        self,
        observation: Mapping[str, Any],
        environment: CyberRLEnv,
    ) -> int:
        return int(
            self.agent.act(
                observation,
                environment.action_spaces[self.role.value],
                self.rng,
            )
        )

    role: Role


class SingleRoleEnv:
    """Expose one side of :class:`CyberRLEnv` and run a fixed opponent.

    The wrapped simulator still resolves Red and Blue actions simultaneously.
    The opponent therefore acts on the same pre-transition observation as the
    learning agent; it never gets to react to the learner's current action.

    This small API deliberately does not depend on Gymnasium. Its reset/step
    return values use a familiar five-value convention while the trusted
    submission runner retains control of the rollout loop.
    """

    def __init__(
        self,
        scenario: ScenarioSpec | str | Path,
        *,
        role: Role | str,
        opponent: Agent | None = None,
        opponent_controller: EnvironmentOpponent | None = None,
    ) -> None:
        if (opponent is None) == (opponent_controller is None):
            raise ValueError("Provide exactly one fixed opponent")
        self.env = CyberRLEnv(scenario)
        self.role = Role(role)
        self.opponent_role = Role.BLUE if self.role is Role.RED else Role.RED
        if opponent_controller is None:
            scripted = _ScriptedOpponent(opponent)
            scripted.role = self.opponent_role
            opponent_controller = scripted
        self.opponent = opponent_controller
        self._sampling_rng = np.random.default_rng()
        self._observations: dict[str, dict[str, Any]] | None = None

    @property
    def action_count(self) -> int:
        """Number of discrete actions for the controlled role."""

        return len(self.env.action_spaces[self.role.value])

    @property
    def observation_size(self) -> int:
        """Length of the flattened feature vector."""

        node_features = self.env.num_hosts * sum(self.env.node_cardinalities(self.role))
        adjacency = self.env.num_hosts * self.env.num_hosts
        return node_features + adjacency + 1

    @property
    def action_labels(self) -> tuple[str, ...]:
        return tuple(self.env.action_spaces[self.role.value].labels())

    @property
    def mission_loss_history(self) -> tuple[float, ...]:
        return self.env.mission_loss_history

    def reset(
        self,
        seed: int | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        opponent_seed = None if seed is None else int(seed) ^ 0x6A09E667
        sampling_seed = None if seed is None else int(seed) ^ 0xBB67AE85
        self.opponent.reset(opponent_seed)
        self._sampling_rng = np.random.default_rng(sampling_seed)
        self._observations, infos = self.env.reset(seed=seed)
        info = dict(infos[self.role.value])
        info.update(
            {
                "role": self.role.value,
                "opponent": str(getattr(self.opponent, "name", self.opponent.__class__.__name__)),
            }
        )
        return self._model_observation(self._observations[self.role.value]), info

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self._observations is None:
            raise RuntimeError("Environment has not been reset")

        opponent_action = self.opponent.act(
            self._observations[self.opponent_role.value],
            self.env,
        )
        joint_actions = {
            self.role.value: int(action),
            self.opponent_role.value: int(opponent_action),
        }
        observations, rewards, terminations, truncations, infos = self.env.step(joint_actions)
        self._observations = observations
        info = dict(infos[self.role.value])
        info["opponent_action"] = infos[self.opponent_role.value]["action"]
        return (
            self._model_observation(observations[self.role.value]),
            float(rewards[self.role.value]),
            bool(terminations[self.role.value]),
            bool(truncations[self.role.value]),
            info,
        )

    def sample_valid_action(
        self,
        observation: Mapping[str, np.ndarray],
        rng: np.random.Generator | None = None,
    ) -> int:
        """Sample uniformly from actions allowed by the supplied mask."""

        mask = np.asarray(observation["action_mask"], dtype=bool)
        if mask.shape != (self.action_count,):
            raise ValueError(
                f"Expected an action mask of shape {(self.action_count,)}, got {mask.shape}"
            )
        valid = np.flatnonzero(mask)
        if valid.size == 0:
            raise ValueError("Action mask contains no valid action")
        generator = rng if rng is not None else self._sampling_rng
        return int(generator.choice(valid))

    def _model_observation(
        self,
        structured: Mapping[str, Any],
    ) -> dict[str, np.ndarray]:
        features = self.env.flatten(self.role, structured)
        mask = np.asarray(structured["action_mask"], dtype=bool).copy()
        return {"features": features, "action_mask": mask}
