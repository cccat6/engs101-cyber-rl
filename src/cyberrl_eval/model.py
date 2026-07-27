"""Shared domain types for the simulator.

The model deliberately keeps only state that affects reachability, observation,
reward, or scoring. It does not emulate packets, processes, users, or virtual
machines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any


class Role(StrEnum):
    RED = "red"
    BLUE = "blue"


class AccessLevel(IntEnum):
    NONE = 0
    USER = 1
    ROOT = 2


class Activity(IntEnum):
    NONE = 0
    SCAN = 1
    EXPLOIT = 2


class BlueEvidence(IntEnum):
    UNKNOWN = 0
    NO_EVIDENCE = 1
    SUSPECTED = 2
    CONFIRMED = 3


class DiscoveryStage(IntEnum):
    UNKNOWN = 0
    DISCOVERED = 1
    SCANNED = 2


class ServiceEvidence(IntEnum):
    UNKNOWN = 0
    NOT_EXPLOITABLE = 1
    EXPLOITABLE = 2
    DECOY_TRIGGERED = 3


class ActionKind(StrEnum):
    SLEEP = "sleep"
    DISCOVER_SUBNET = "discover_subnet"
    SCAN = "scan"
    EXPLOIT = "exploit"
    ESCALATE = "escalate"
    IMPACT = "impact"
    ANALYSE = "analyse"
    DECOY = "decoy"
    REMOVE = "remove"
    RESTORE = "restore"


@dataclass(frozen=True, slots=True)
class Action:
    role: Role
    kind: ActionKind
    target: str | None = None

    @property
    def label(self) -> str:
        return self.kind.value if self.target is None else f"{self.kind.value}:{self.target}"


@dataclass(slots=True)
class Event:
    step: int
    actor: Role | None
    kind: str
    target: str | None
    success: bool
    detected: bool = False
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "actor": self.actor.value if self.actor is not None else None,
            "kind": self.kind,
            "target": self.target,
            "success": self.success,
            "detected": self.detected,
            "message": self.message,
            "details": dict(self.details),
        }

