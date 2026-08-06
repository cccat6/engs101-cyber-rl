"""Small reference agents used to smoke-test and calibrate scenarios."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

import numpy as np

from .actions import ActionSpace
from .model import ActionKind, BlueEvidence


class Agent(Protocol):
    name: str

    def reset(self) -> None: ...

    def act(
        self,
        observation: Mapping[str, Any],
        action_space: ActionSpace,
        rng: np.random.Generator,
    ) -> int: ...


class SleepAgent:
    name = "sleep"

    def reset(self) -> None:
        return None

    def act(
        self,
        observation: Mapping[str, Any],
        action_space: ActionSpace,
        rng: np.random.Generator,
    ) -> int:
        return 0


class RandomValidAgent:
    name = "random-valid"

    def reset(self) -> None:
        return None

    def act(
        self,
        observation: Mapping[str, Any],
        action_space: ActionSpace,
        rng: np.random.Generator,
    ) -> int:
        valid = action_space.valid_indices(np.asarray(observation["action_mask"]))
        return int(rng.choice(valid))


class GreedyRedAgent:
    """Progresses through discover, scan, exploit, escalate, and impact."""

    name = "greedy-red"
    priority = (
        ActionKind.IMPACT,
        ActionKind.ESCALATE,
        ActionKind.EXPLOIT,
        ActionKind.SCAN,
        ActionKind.DISCOVER_SUBNET,
        ActionKind.SLEEP,
    )

    def reset(self) -> None:
        return None

    def act(
        self,
        observation: Mapping[str, Any],
        action_space: ActionSpace,
        rng: np.random.Generator,
    ) -> int:
        mask = np.asarray(observation["action_mask"], dtype=bool)
        for kind in self.priority:
            candidates = [
                index
                for index, action in enumerate(action_space.actions)
                if action.kind is kind and mask[index]
            ]
            if candidates:
                return int(rng.choice(candidates))
        return 0


class FocusedRedAgent:
    """Expands through frontier subnets before cleaning up known hosts.

    Unlike :class:`GreedyRedAgent`, this policy uses only the structured Red
    observation and action mask; it contains no scenario or host names. It is
    useful for calibrating larger branching topologies where scanning every
    host in the entry subnet gives Blue an unrealistic amount of time.
    """

    name = "focused-red"

    def reset(self) -> None:
        return None

    def act(
        self,
        observation: Mapping[str, Any],
        action_space: ActionSpace,
        rng: np.random.Generator,
    ) -> int:
        mask = np.asarray(observation["action_mask"], dtype=bool)
        nodes = np.asarray(observation["nodes"])

        impact = self._valid(action_space, mask, ActionKind.IMPACT)
        if impact:
            return int(rng.choice(impact))

        discoveries = self._valid(action_space, mask, ActionKind.DISCOVER_SUBNET)
        if discoveries:
            return int(rng.choice(discoveries))

        controlled_subnets = set(nodes[nodes[:, 3] > 0, 0].tolist())
        host_targets = [
            action.target
            for action in action_space.actions
            if action.kind is ActionKind.SCAN
        ]
        host_index = {target: index for index, target in enumerate(host_targets)}

        for kind in (ActionKind.EXPLOIT, ActionKind.SCAN):
            frontier = [
                index
                for index in self._valid(action_space, mask, kind)
                if nodes[host_index[action_space.actions[index].target], 0]
                not in controlled_subnets
            ]
            if frontier:
                return int(rng.choice(frontier))

        for kind in (ActionKind.ESCALATE, ActionKind.EXPLOIT, ActionKind.SCAN):
            candidates = self._valid(action_space, mask, kind)
            if candidates:
                return int(rng.choice(candidates))
        return 0

    @staticmethod
    def _valid(
        action_space: ActionSpace,
        mask: np.ndarray,
        kind: ActionKind,
    ) -> list[int]:
        return [
            index
            for index, action in enumerate(action_space.actions)
            if action.kind is kind and mask[index]
        ]


class ReactiveBlueAgent:
    """Restores confirmed hosts, analyses suspicions, then deploys decoys."""

    name = "reactive-blue"

    def reset(self) -> None:
        return None

    def act(
        self,
        observation: Mapping[str, Any],
        action_space: ActionSpace,
        rng: np.random.Generator,
    ) -> int:
        nodes = np.asarray(observation["nodes"])
        mask = np.asarray(observation["action_mask"], dtype=bool)

        for evidence, kind in (
            (BlueEvidence.CONFIRMED, ActionKind.RESTORE),
            (BlueEvidence.SUSPECTED, ActionKind.ANALYSE),
        ):
            host_indices = set(np.flatnonzero(nodes[:, 3] == evidence).tolist())
            candidates = [
                index
                for index, action in enumerate(action_space.actions)
                if action.kind is kind
                and mask[index]
                and self._target_index(action_space, action.target) in host_indices
            ]
            if candidates:
                return int(rng.choice(candidates))

        decoys = [
            index
            for index, action in enumerate(action_space.actions)
            if action.kind is ActionKind.DECOY and mask[index]
        ]
        if decoys:
            return int(rng.choice(decoys))
        return 0

    @staticmethod
    def _target_index(action_space: ActionSpace, target: str | None) -> int:
        targets = list(dict.fromkeys(
            action.target
            for action in action_space.actions
            if action.target is not None
        ))
        return targets.index(target)


AGENT_FACTORIES = {
    "sleep": SleepAgent,
    "random": RandomValidAgent,
    "greedy-red": GreedyRedAgent,
    "focused-red": FocusedRedAgent,
    "reactive-blue": ReactiveBlueAgent,
}


def make_agent(name: str) -> Agent:
    try:
        return AGENT_FACTORIES[name]()
    except KeyError as error:
        raise ValueError(f"Unknown trusted agent: {name}") from error
