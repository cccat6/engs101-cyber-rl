"""Fixed discrete action mappings compiled from a topology."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import Action, ActionKind, Role
from .topology import CompiledTopology


RED_HOST_ACTIONS = (
    ActionKind.SCAN,
    ActionKind.EXPLOIT,
    ActionKind.ESCALATE,
    ActionKind.IMPACT,
)
BLUE_HOST_ACTIONS = (
    ActionKind.ANALYSE,
    ActionKind.DECOY,
    ActionKind.REMOVE,
    ActionKind.RESTORE,
)


@dataclass(frozen=True, slots=True)
class ActionSpace:
    role: Role
    actions: tuple[Action, ...]

    @classmethod
    def for_topology(cls, role: Role, topology: CompiledTopology) -> "ActionSpace":
        actions: list[Action] = [Action(role, ActionKind.SLEEP)]
        if role is Role.RED:
            actions.extend(
                Action(role, ActionKind.DISCOVER_SUBNET, subnet)
                for subnet in topology.subnet_names
            )
            actions.extend(
                Action(role, kind, host)
                for kind in RED_HOST_ACTIONS
                for host in topology.host_names
            )
        else:
            actions.extend(
                Action(role, kind, host)
                for kind in BLUE_HOST_ACTIONS
                for host in topology.host_names
            )
        return cls(role=role, actions=tuple(actions))

    def __len__(self) -> int:
        return len(self.actions)

    def decode(self, index: int) -> Action:
        if not 0 <= int(index) < len(self.actions):
            raise IndexError(f"Action index {index} outside [0, {len(self.actions)})")
        return self.actions[int(index)]

    def encode(self, kind: ActionKind, target: str | None = None) -> int:
        for index, action in enumerate(self.actions):
            if action.kind is kind and action.target == target:
                return index
        raise KeyError(f"No {self.role.value} action {kind.value}:{target}")

    def labels(self) -> tuple[str, ...]:
        return tuple(action.label for action in self.actions)

    def valid_indices(self, mask: np.ndarray) -> np.ndarray:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (len(self.actions),):
            raise ValueError(f"Expected action mask shape {(len(self.actions),)}, got {mask.shape}")
        return np.flatnonzero(mask)

