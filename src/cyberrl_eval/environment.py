"""A compact two-sided cyber environment with simultaneous actions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .actions import ActionSpace
from .model import (
    AccessLevel,
    Action,
    ActionKind,
    Activity,
    BlueEvidence,
    DiscoveryStage,
    Event,
    Role,
    ServiceEvidence,
)
from .observation import flatten_observation
from .scenario import ScenarioSpec, ServiceSpec, load_scenario
from .topology import CompiledTopology, compile_topology


@dataclass(slots=True)
class WorldState:
    step: int
    access: np.ndarray
    known_subnets: np.ndarray
    discovery: np.ndarray
    red_service_evidence: np.ndarray
    impacted: np.ndarray
    decoys: np.ndarray
    blue_activity: np.ndarray
    blue_evidence: np.ndarray


class CyberRLEnv:
    """Config-driven Red/Blue cyber simulation.

    The interface mirrors the useful portion of PettingZoo's parallel API but
    intentionally has no dependency on Gymnasium or PettingZoo. Both agents
    choose from the same pre-transition state and submit one discrete action.
    """

    possible_agents = (Role.RED.value, Role.BLUE.value)

    def __init__(self, scenario: ScenarioSpec | str | Path):
        self.scenario = load_scenario(scenario) if isinstance(scenario, (str, Path)) else scenario
        self.topology: CompiledTopology = compile_topology(self.scenario)
        self.action_spaces = {
            Role.RED.value: ActionSpace.for_topology(Role.RED, self.topology),
            Role.BLUE.value: ActionSpace.for_topology(Role.BLUE, self.topology),
        }
        self.agents = list(self.possible_agents)
        self._rng = np.random.default_rng()
        self._seed: int | None = None
        self._state: WorldState | None = None
        self._mission_loss_history: list[float] = []
        self._last_events: list[Event] = []

    @property
    def num_hosts(self) -> int:
        return self.topology.num_hosts

    @property
    def mission_loss_history(self) -> tuple[float, ...]:
        return tuple(self._mission_loss_history)

    @property
    def mission_loss(self) -> float:
        state = self._require_state()
        return float(np.dot(self.topology.mission_weights_by_host, state.impacted.astype(float)))

    def reset(
        self,
        seed: int | None = None,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self.agents = list(self.possible_agents)

        host_count = self.topology.num_hosts
        subnet_count = self.topology.num_subnets
        access = np.zeros(host_count, dtype=np.int8)
        known_subnets = np.zeros(subnet_count, dtype=bool)
        discovery = np.zeros(host_count, dtype=np.int8)
        red_service_evidence = np.zeros(host_count, dtype=np.int8)
        impacted = np.zeros(host_count, dtype=bool)
        decoys = np.zeros(host_count, dtype=np.int8)
        blue_activity = np.zeros(host_count, dtype=np.int8)
        blue_evidence = np.full(host_count, BlueEvidence.NO_EVIDENCE, dtype=np.int8)

        start_host = self.topology.host_index[self.scenario.red_start.host]
        start_subnet = int(self.topology.host_to_subnet[start_host])
        access[start_host] = int(self.scenario.red_start.access)
        known_subnets[start_subnet] = True
        discovery[self.topology.host_to_subnet == start_subnet] = DiscoveryStage.DISCOVERED
        discovery[start_host] = DiscoveryStage.SCANNED
        red_service_evidence[start_host] = (
            ServiceEvidence.EXPLOITABLE
            if self.scenario.hosts[start_host].services
            else ServiceEvidence.NOT_EXPLOITABLE
        )

        self._state = WorldState(
            step=0,
            access=access,
            known_subnets=known_subnets,
            discovery=discovery,
            red_service_evidence=red_service_evidence,
            impacted=impacted,
            decoys=decoys,
            blue_activity=blue_activity,
            blue_evidence=blue_evidence,
        )
        self._mission_loss_history = []
        self._last_events = []
        observations = self._observations()
        infos = {
            role: {
                "scenario": self.scenario.name,
                "seed": seed,
                "action_labels": self.action_spaces[role].labels(),
                "node_cardinalities": self.node_cardinalities(role),
            }
            for role in self.possible_agents
        }
        return observations, infos

    def step(
        self,
        actions: Mapping[str, int],
    ) -> tuple[
        dict[str, dict[str, Any]],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        state = self._require_state()
        if state.step >= self.scenario.horizon:
            raise RuntimeError("Episode is complete; call reset() before step()")

        events: list[Event] = []
        pre_masks = {
            role: self.action_mask(role)
            for role in self.possible_agents
        }
        red_action = self._decode_or_sleep(Role.RED, actions.get(Role.RED.value), pre_masks[Role.RED.value], events)
        blue_action = self._decode_or_sleep(Role.BLUE, actions.get(Role.BLUE.value), pre_masks[Role.BLUE.value], events)

        # Activity is a one-step signal. Persistent compromise knowledge lives
        # separately in blue_evidence.
        state.blue_activity.fill(Activity.NONE)

        blue_cost = self._action_cost(Role.BLUE, blue_action)
        red_cost = self._action_cost(Role.RED, red_action)

        # A decoy selected from the pre-transition state is installed before a
        # simultaneous exploit is resolved. Other Blue actions resolve after
        # the Red transition; Restore therefore clears same-step compromise.
        if blue_action.kind is ActionKind.DECOY:
            self._apply_blue(blue_action, events)

        new_user, new_root = self._apply_red(red_action, events)

        if blue_action.kind is not ActionKind.DECOY:
            self._apply_blue(blue_action, events)

        mission_loss = self.mission_loss
        shaping = (
            self.scenario.reward.new_user_access * new_user
            + self.scenario.reward.new_root_access * new_root
        )
        rewards = {
            Role.RED.value: float(mission_loss + shaping - red_cost),
            Role.BLUE.value: float(-mission_loss - shaping - blue_cost),
        }

        state.step += 1
        self._mission_loss_history.append(mission_loss)
        self._last_events = events

        truncated = state.step >= self.scenario.horizon
        terminations = {role: False for role in self.possible_agents}
        truncations = {role: truncated for role in self.possible_agents}
        if truncated:
            self.agents = []

        observations = self._observations()
        event_dicts = [event.to_dict() for event in events]
        infos = {
            Role.RED.value: {
                "action": red_action.label,
                "events": event_dicts,
                "mission_loss": mission_loss,
                "reward_terms": {
                    "mission": mission_loss,
                    "new_access": shaping,
                    "action_cost": -red_cost,
                },
            },
            Role.BLUE.value: {
                "action": blue_action.label,
                "events": event_dicts,
                "mission_loss": mission_loss,
                "reward_terms": {
                    "mission": -mission_loss,
                    "new_access": -shaping,
                    "action_cost": -blue_cost,
                },
            },
        }
        return observations, rewards, terminations, truncations, infos

    def action_mask(self, role: Role | str) -> np.ndarray:
        role = Role(role)
        return self._red_action_mask() if role is Role.RED else self._blue_action_mask()

    def node_cardinalities(self, role: Role | str) -> tuple[int, ...]:
        role = Role(role)
        role_cardinality = len(self.topology.role_names) + 1
        if role is Role.RED:
            return (
                self.topology.num_subnets + 1,
                role_cardinality,
                len(DiscoveryStage),
                len(AccessLevel),
                len(ServiceEvidence),
                2,
            )
        return (
            self.topology.num_subnets,
            role_cardinality,
            len(Activity),
            len(BlueEvidence),
            3,
            self.scenario.max_decoys_per_host + 1,
        )

    def flatten(self, role: Role | str, observation: Mapping[str, Any]) -> np.ndarray:
        return flatten_observation(observation, self.node_cardinalities(role))

    def state_snapshot(self) -> dict[str, Any]:
        """Return simulator truth for reports and tests, never for policy input."""
        state = self._require_state()
        return {
            "step": state.step,
            "access": {
                host: AccessLevel(int(state.access[index])).name.lower()
                for index, host in enumerate(self.topology.host_names)
            },
            "known_subnets": [
                subnet
                for index, subnet in enumerate(self.topology.subnet_names)
                if state.known_subnets[index]
            ],
            "impacted_hosts": [
                host
                for index, host in enumerate(self.topology.host_names)
                if state.impacted[index]
            ],
            "decoys": {
                host: int(state.decoys[index])
                for index, host in enumerate(self.topology.host_names)
                if state.decoys[index]
            },
            "mission_loss": self.mission_loss,
        }

    def _require_state(self) -> WorldState:
        if self._state is None:
            raise RuntimeError("Environment has not been reset")
        return self._state

    def _observations(self) -> dict[str, dict[str, Any]]:
        return {
            Role.RED.value: self._red_observation(),
            Role.BLUE.value: self._blue_observation(),
        }

    def _red_observation(self) -> dict[str, Any]:
        state = self._require_state()
        known = state.discovery > DiscoveryStage.UNKNOWN
        nodes = np.zeros((self.num_hosts, 6), dtype=np.int16)
        nodes[:, 0] = np.where(known, self.topology.host_to_subnet + 1, 0)
        nodes[:, 1] = np.where(known, self.topology.host_roles, 0)
        nodes[:, 2] = state.discovery
        nodes[:, 3] = np.where(known, state.access, AccessLevel.NONE)
        nodes[:, 4] = np.where(known, state.red_service_evidence, ServiceEvidence.UNKNOWN)
        nodes[:, 5] = np.where(known, state.impacted.astype(np.int16), 0)
        visibility = np.outer(known, known).astype(np.int8)
        adjacency = self.topology.host_adjacency * visibility
        return {
            "nodes": nodes,
            "adjacency": adjacency.copy(),
            "action_mask": self._red_action_mask(),
            "time": np.float32(state.step / self.scenario.horizon),
        }

    def _blue_observation(self) -> dict[str, Any]:
        state = self._require_state()
        service_status = np.asarray(
            [1 if host.services else 0 for host in self.scenario.hosts], dtype=np.int16
        )
        service_status[state.impacted] = 2
        nodes = np.column_stack(
            [
                self.topology.host_to_subnet,
                self.topology.host_roles,
                state.blue_activity,
                state.blue_evidence,
                service_status,
                state.decoys,
            ]
        ).astype(np.int16, copy=False)
        return {
            "nodes": nodes,
            "adjacency": self.topology.host_adjacency.copy(),
            "action_mask": self._blue_action_mask(),
            "time": np.float32(state.step / self.scenario.horizon),
        }

    def _red_action_mask(self) -> np.ndarray:
        state = self._require_state()
        space = self.action_spaces[Role.RED.value]
        mask = np.zeros(len(space), dtype=bool)
        mask[0] = True
        for index, action in enumerate(space.actions[1:], start=1):
            if action.kind is ActionKind.DISCOVER_SUBNET:
                subnet = self.topology.subnet_index[action.target]
                mask[index] = not state.known_subnets[subnet] and self._subnet_reachable(subnet)
                continue

            host = self.topology.host_index[action.target]
            if action.kind is ActionKind.SCAN:
                mask[index] = (
                    state.discovery[host] == DiscoveryStage.DISCOVERED
                    and self._host_reachable(host)
                )
            elif action.kind is ActionKind.EXPLOIT:
                mask[index] = (
                    state.discovery[host] == DiscoveryStage.SCANNED
                    and state.access[host] == AccessLevel.NONE
                    and state.red_service_evidence[host] == ServiceEvidence.EXPLOITABLE
                    and bool(self._reachable_services(host))
                )
            elif action.kind is ActionKind.ESCALATE:
                mask[index] = state.access[host] == AccessLevel.USER
            elif action.kind is ActionKind.IMPACT:
                mask[index] = (
                    state.access[host] == AccessLevel.ROOT
                    and self.topology.critical_hosts[host]
                    and not state.impacted[host]
                )
        return mask

    def _blue_action_mask(self) -> np.ndarray:
        state = self._require_state()
        space = self.action_spaces[Role.BLUE.value]
        mask = np.zeros(len(space), dtype=bool)
        mask[0] = True
        for index, action in enumerate(space.actions[1:], start=1):
            host = self.topology.host_index[action.target]
            if not self.topology.blue_managed[host]:
                continue
            if action.kind is ActionKind.ANALYSE:
                mask[index] = True
            elif action.kind is ActionKind.DECOY:
                mask[index] = (
                    bool(self.scenario.hosts[host].services)
                    and state.decoys[host] < self.scenario.max_decoys_per_host
                )
            elif action.kind in (ActionKind.REMOVE, ActionKind.RESTORE):
                mask[index] = state.blue_evidence[host] >= BlueEvidence.SUSPECTED
        return mask

    def _decode_or_sleep(
        self,
        role: Role,
        raw_index: int | None,
        mask: np.ndarray,
        events: list[Event],
    ) -> Action:
        space = self.action_spaces[role.value]
        try:
            index = int(raw_index) if raw_index is not None else 0
            action = space.decode(index)
        except (TypeError, ValueError, IndexError):
            index = 0
            action = space.decode(0)
            events.append(
                Event(
                    step=self._require_state().step,
                    actor=role,
                    kind="invalid_action",
                    target=None,
                    success=False,
                    message=f"Invalid action index {raw_index!r}; treated as sleep",
                )
            )
            return action
        if not mask[index]:
            events.append(
                Event(
                    step=self._require_state().step,
                    actor=role,
                    kind="invalid_action",
                    target=action.target,
                    success=False,
                    message=f"Illegal action {action.label}; treated as sleep",
                )
            )
            return space.decode(0)
        return action

    def _action_cost(self, role: Role, action: Action) -> float:
        costs = (
            self.scenario.reward.red_action_costs
            if role is Role.RED
            else self.scenario.reward.blue_action_costs
        )
        return float(costs.get(action.kind.value, 0.0))

    def _controlled_sources(self) -> np.ndarray:
        state = self._require_state()
        return np.flatnonzero(state.access > AccessLevel.NONE)

    def _subnet_reachable(self, target_subnet: int) -> bool:
        for source in self._controlled_sources():
            source_subnet = int(self.topology.host_to_subnet[source])
            if self.topology.subnet_adjacency[source_subnet, target_subnet]:
                return True
        return False

    def _host_reachable(self, target_host: int) -> bool:
        return any(
            self.topology.any_service_allowed(int(source), target_host)
            for source in self._controlled_sources()
        )

    def _reachable_services(self, target_host: int) -> list[ServiceSpec]:
        services: list[ServiceSpec] = []
        for service in self.scenario.hosts[target_host].services:
            if any(
                self.topology.service_allowed(int(source), target_host, service.name)
                for source in self._controlled_sources()
            ):
                services.append(service)
        return services

    def _apply_red(self, action: Action, events: list[Event]) -> tuple[int, int]:
        state = self._require_state()
        new_user = 0
        new_root = 0

        if action.kind is ActionKind.SLEEP:
            events.append(Event(state.step, Role.RED, action.kind.value, None, True, message="Red slept"))
            return new_user, new_root

        if action.kind is ActionKind.DISCOVER_SUBNET:
            subnet = self.topology.subnet_index[action.target]
            state.known_subnets[subnet] = True
            host_mask = self.topology.host_to_subnet == subnet
            state.discovery[host_mask] = np.maximum(
                state.discovery[host_mask], DiscoveryStage.DISCOVERED
            )
            events.append(
                Event(
                    state.step,
                    Role.RED,
                    action.kind.value,
                    action.target,
                    True,
                    message=f"Red discovered subnet {action.target}",
                )
            )
            return new_user, new_root

        host = self.topology.host_index[action.target]
        host_name = self.topology.host_names[host]

        if action.kind is ActionKind.SCAN:
            state.discovery[host] = DiscoveryStage.SCANNED
            services = self._reachable_services(host)
            state.red_service_evidence[host] = (
                ServiceEvidence.EXPLOITABLE if services else ServiceEvidence.NOT_EXPLOITABLE
            )
            detected = bool(self._rng.random() < self.scenario.dynamics.scan_detection_probability)
            if detected:
                state.blue_activity[host] = Activity.SCAN
            events.append(
                Event(
                    state.step,
                    Role.RED,
                    action.kind.value,
                    host_name,
                    True,
                    detected=detected,
                    message=f"Red scanned {host_name}",
                    details={"services_found": len(services)},
                )
            )
            return new_user, new_root

        if action.kind is ActionKind.EXPLOIT:
            services = self._reachable_services(host)
            if not services:
                events.append(
                    Event(state.step, Role.RED, action.kind.value, host_name, False, message="No reachable service")
                )
                return new_user, new_root
            service = services[int(self._rng.integers(0, len(services)))]

            if state.decoys[host] > 0:
                state.decoys[host] -= 1
                # The decoy reveals that the previous service picture is no
                # longer reliable. Red must scan the host again before a
                # subsequent exploit attempt.
                state.discovery[host] = DiscoveryStage.DISCOVERED
                state.red_service_evidence[host] = ServiceEvidence.DECOY_TRIGGERED
                state.blue_activity[host] = Activity.EXPLOIT
                state.blue_evidence[host] = max(
                    state.blue_evidence[host], BlueEvidence.SUSPECTED
                )
                events.append(
                    Event(
                        state.step,
                        Role.RED,
                        action.kind.value,
                        host_name,
                        False,
                        detected=True,
                        message=f"Exploit on {host_name} was intercepted by a decoy",
                        details={"service": service.name, "exploit": service.exploit},
                    )
                )
                return new_user, new_root

            success = bool(self._rng.random() < service.success_probability)
            detected = bool(self._rng.random() < service.detection_probability)
            if success:
                previous = AccessLevel(int(state.access[host]))
                granted = max(previous, service.grants)
                state.access[host] = int(granted)
                if previous == AccessLevel.NONE and granted == AccessLevel.USER:
                    new_user += 1
                elif previous < AccessLevel.ROOT and granted == AccessLevel.ROOT:
                    new_root += 1
            if detected:
                state.blue_activity[host] = Activity.EXPLOIT
                state.blue_evidence[host] = max(
                    state.blue_evidence[host], BlueEvidence.SUSPECTED
                )
            events.append(
                Event(
                    state.step,
                    Role.RED,
                    action.kind.value,
                    host_name,
                    success,
                    detected=detected,
                    message=f"Exploit on {host_name} {'succeeded' if success else 'failed'}",
                    details={
                        "service": service.name,
                        "exploit": service.exploit,
                        "granted": service.grants.name.lower() if success else "none",
                    },
                )
            )
            return new_user, new_root

        if action.kind is ActionKind.ESCALATE:
            success = bool(
                self._rng.random() < self.scenario.hosts[host].escalation_probability
            )
            detected = bool(
                self._rng.random() < self.scenario.dynamics.escalation_detection_probability
            )
            if success:
                state.access[host] = AccessLevel.ROOT
                new_root += 1
            if detected:
                state.blue_activity[host] = Activity.EXPLOIT
                state.blue_evidence[host] = max(
                    state.blue_evidence[host], BlueEvidence.SUSPECTED
                )
            events.append(
                Event(
                    state.step,
                    Role.RED,
                    action.kind.value,
                    host_name,
                    success,
                    detected=detected,
                    message=f"Privilege escalation on {host_name} {'succeeded' if success else 'failed'}",
                )
            )
            return new_user, new_root

        if action.kind is ActionKind.IMPACT:
            state.impacted[host] = True
            state.blue_activity[host] = Activity.EXPLOIT
            state.blue_evidence[host] = BlueEvidence.CONFIRMED
            events.append(
                Event(
                    state.step,
                    Role.RED,
                    action.kind.value,
                    host_name,
                    True,
                    detected=True,
                    message=f"Critical service on {host_name} was impacted",
                    details={"mission_weight": float(self.topology.mission_weights_by_host[host])},
                )
            )
            return new_user, new_root

        raise RuntimeError(f"Unhandled Red action {action.kind}")

    def _apply_blue(self, action: Action, events: list[Event]) -> None:
        state = self._require_state()
        if action.kind is ActionKind.SLEEP:
            events.append(Event(state.step, Role.BLUE, action.kind.value, None, True, message="Blue slept"))
            return

        host = self.topology.host_index[action.target]
        host_name = self.topology.host_names[host]

        if action.kind is ActionKind.DECOY:
            state.decoys[host] += 1
            events.append(
                Event(
                    state.step,
                    Role.BLUE,
                    action.kind.value,
                    host_name,
                    True,
                    message=f"Blue deployed a decoy on {host_name}",
                    details={"decoy_count": int(state.decoys[host])},
                )
            )
            return

        if action.kind is ActionKind.ANALYSE:
            compromised = state.access[host] > AccessLevel.NONE
            state.blue_evidence[host] = (
                BlueEvidence.CONFIRMED if compromised else BlueEvidence.NO_EVIDENCE
            )
            events.append(
                Event(
                    state.step,
                    Role.BLUE,
                    action.kind.value,
                    host_name,
                    True,
                    message=(
                        f"Blue confirmed compromise on {host_name}"
                        if compromised
                        else f"Blue found no compromise evidence on {host_name}"
                    ),
                    details={"result": "compromised" if compromised else "no_evidence"},
                )
            )
            return

        if action.kind is ActionKind.REMOVE:
            success = state.access[host] == AccessLevel.USER
            if success:
                state.access[host] = AccessLevel.NONE
                state.blue_evidence[host] = BlueEvidence.NO_EVIDENCE
            events.append(
                Event(
                    state.step,
                    Role.BLUE,
                    action.kind.value,
                    host_name,
                    bool(success),
                    message=(
                        f"Blue removed user-level access from {host_name}"
                        if success
                        else f"Remove did not clear access on {host_name}"
                    ),
                )
            )
            return

        if action.kind is ActionKind.RESTORE:
            changed = bool(
                state.access[host] > AccessLevel.NONE
                or state.impacted[host]
                or state.decoys[host] > 0
            )
            state.access[host] = AccessLevel.NONE
            state.impacted[host] = False
            state.decoys[host] = 0
            state.blue_evidence[host] = BlueEvidence.NO_EVIDENCE
            events.append(
                Event(
                    state.step,
                    Role.BLUE,
                    action.kind.value,
                    host_name,
                    changed,
                    message=f"Blue restored {host_name}",
                )
            )
            return

        raise RuntimeError(f"Unhandled Blue action {action.kind}")
