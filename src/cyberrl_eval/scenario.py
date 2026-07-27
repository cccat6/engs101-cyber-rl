"""Scenario schema and YAML loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .model import AccessLevel


class ScenarioError(ValueError):
    """Raised when a scenario file is internally inconsistent."""


@dataclass(frozen=True, slots=True)
class ServiceSpec:
    name: str
    port: int
    exploit: str
    success_probability: float
    detection_probability: float
    grants: AccessLevel


@dataclass(frozen=True, slots=True)
class HostSpec:
    name: str
    subnet: str
    role: str
    services: tuple[ServiceSpec, ...]
    escalation_probability: float
    blue_managed: bool


@dataclass(frozen=True, slots=True)
class SubnetSpec:
    name: str
    connects_to: Mapping[str, frozenset[str]]


@dataclass(frozen=True, slots=True)
class MissionAssetSpec:
    name: str
    host: str
    costs: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class MissionSpec:
    objectives: Mapping[str, float]
    assets: tuple[MissionAssetSpec, ...]


@dataclass(frozen=True, slots=True)
class RewardSpec:
    new_user_access: float
    new_root_access: float
    red_action_costs: Mapping[str, float]
    blue_action_costs: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class DynamicsSpec:
    scan_detection_probability: float
    escalation_detection_probability: float


@dataclass(frozen=True, slots=True)
class RedStartSpec:
    host: str
    access: AccessLevel


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    schema_version: int
    name: str
    horizon: int
    max_decoys_per_host: int
    subnets: tuple[SubnetSpec, ...]
    hosts: tuple[HostSpec, ...]
    red_start: RedStartSpec
    mission: MissionSpec
    reward: RewardSpec
    dynamics: DynamicsSpec


def _probability(value: Any, field_name: str) -> float:
    probability = float(value)
    if not 0.0 <= probability <= 1.0:
        raise ScenarioError(f"{field_name} must be between 0 and 1, got {probability}")
    return probability


def _non_negative(value: Any, field_name: str) -> float:
    number = float(value)
    if number < 0:
        raise ScenarioError(f"{field_name} must be non-negative, got {number}")
    return number


def _parse_access(value: Any, field_name: str) -> AccessLevel:
    try:
        return AccessLevel[str(value).upper()]
    except KeyError as exc:
        choices = ", ".join(level.name.lower() for level in AccessLevel)
        raise ScenarioError(f"{field_name} must be one of: {choices}") from exc


def scenario_from_mapping(data: Mapping[str, Any]) -> ScenarioSpec:
    schema_version = int(data.get("schema_version", 1))
    if schema_version != 1:
        raise ScenarioError(f"Unsupported schema_version {schema_version}; expected 1")

    name = str(data.get("name", "")).strip()
    if not name:
        raise ScenarioError("Scenario name is required")

    horizon = int(data.get("horizon", 100))
    if horizon <= 0:
        raise ScenarioError("horizon must be positive")

    max_decoys = int(data.get("max_decoys_per_host", 2))
    if max_decoys < 0:
        raise ScenarioError("max_decoys_per_host must be non-negative")

    raw_subnets = data.get("subnets")
    if not isinstance(raw_subnets, Mapping) or not raw_subnets:
        raise ScenarioError("subnets must be a non-empty mapping")

    subnets: list[SubnetSpec] = []
    for subnet_name, subnet_data in raw_subnets.items():
        subnet_data = subnet_data or {}
        raw_connections = subnet_data.get("connects_to", {})
        if not isinstance(raw_connections, Mapping):
            raise ScenarioError(f"subnets.{subnet_name}.connects_to must be a mapping")
        connections: dict[str, frozenset[str]] = {}
        for target, allowed_services in raw_connections.items():
            if not isinstance(allowed_services, list) or not allowed_services:
                raise ScenarioError(
                    f"Connection {subnet_name}->{target} must list at least one allowed service"
                )
            connections[str(target)] = frozenset(str(item) for item in allowed_services)
        subnets.append(SubnetSpec(str(subnet_name), connections))

    subnet_names = {subnet.name for subnet in subnets}
    for subnet in subnets:
        unknown = set(subnet.connects_to) - subnet_names
        if unknown:
            raise ScenarioError(f"Subnet {subnet.name} connects to unknown subnets: {sorted(unknown)}")

    raw_hosts = data.get("hosts")
    if not isinstance(raw_hosts, Mapping) or not raw_hosts:
        raise ScenarioError("hosts must be a non-empty mapping")

    hosts: list[HostSpec] = []
    for host_name, host_data in raw_hosts.items():
        if not isinstance(host_data, Mapping):
            raise ScenarioError(f"hosts.{host_name} must be a mapping")
        subnet = str(host_data.get("subnet", ""))
        if subnet not in subnet_names:
            raise ScenarioError(f"Host {host_name} references unknown subnet {subnet!r}")
        role = str(host_data.get("role", "other"))
        services: list[ServiceSpec] = []
        seen_services: set[str] = set()
        for index, service_data in enumerate(host_data.get("services", [])):
            if not isinstance(service_data, Mapping):
                raise ScenarioError(f"hosts.{host_name}.services[{index}] must be a mapping")
            service_name = str(service_data.get("name", "")).strip()
            if not service_name:
                raise ScenarioError(f"hosts.{host_name}.services[{index}].name is required")
            if service_name in seen_services:
                raise ScenarioError(f"Host {host_name} defines service {service_name!r} twice")
            seen_services.add(service_name)
            services.append(
                ServiceSpec(
                    name=service_name,
                    port=int(service_data.get("port", 0)),
                    exploit=str(service_data.get("exploit", "generic")),
                    success_probability=_probability(
                        service_data.get("success_probability", 1.0),
                        f"hosts.{host_name}.services[{index}].success_probability",
                    ),
                    detection_probability=_probability(
                        service_data.get("detection_probability", 0.5),
                        f"hosts.{host_name}.services[{index}].detection_probability",
                    ),
                    grants=_parse_access(
                        service_data.get("grants", "user"),
                        f"hosts.{host_name}.services[{index}].grants",
                    ),
                )
            )
        hosts.append(
            HostSpec(
                name=str(host_name),
                subnet=subnet,
                role=role,
                services=tuple(services),
                escalation_probability=_probability(
                    host_data.get("escalation_probability", 0.75),
                    f"hosts.{host_name}.escalation_probability",
                ),
                blue_managed=bool(host_data.get("blue_managed", True)),
            )
        )

    host_names = {host.name for host in hosts}
    raw_red_start = data.get("red_start")
    if not isinstance(raw_red_start, Mapping):
        raise ScenarioError("red_start must be a mapping")
    red_start_host = str(raw_red_start.get("host", ""))
    if red_start_host not in host_names:
        raise ScenarioError(f"red_start references unknown host {red_start_host!r}")
    red_start_access = _parse_access(raw_red_start.get("access", "root"), "red_start.access")
    if red_start_access == AccessLevel.NONE:
        raise ScenarioError("red_start.access must provide an initial foothold")
    red_start = RedStartSpec(host=red_start_host, access=red_start_access)

    raw_mission = data.get("mission")
    if not isinstance(raw_mission, Mapping):
        raise ScenarioError("mission must be a mapping")
    raw_objectives = raw_mission.get("objectives")
    if not isinstance(raw_objectives, Mapping) or not raw_objectives:
        raise ScenarioError("mission.objectives must be a non-empty mapping")
    objectives = {
        str(key): _non_negative(value, f"mission.objectives.{key}")
        for key, value in raw_objectives.items()
    }
    objective_total = sum(objectives.values())
    if objective_total <= 0:
        raise ScenarioError("At least one mission objective must have positive weight")
    objectives = {key: value / objective_total for key, value in objectives.items()}

    raw_assets = raw_mission.get("assets")
    if not isinstance(raw_assets, Mapping) or not raw_assets:
        raise ScenarioError("mission.assets must be a non-empty mapping")
    assets: list[MissionAssetSpec] = []
    for asset_name, asset_data in raw_assets.items():
        if not isinstance(asset_data, Mapping):
            raise ScenarioError(f"mission.assets.{asset_name} must be a mapping")
        host = str(asset_data.get("host", ""))
        if host not in host_names:
            raise ScenarioError(f"Mission asset {asset_name} references unknown host {host!r}")
        raw_costs = asset_data.get("costs")
        if not isinstance(raw_costs, Mapping) or not raw_costs:
            raise ScenarioError(f"mission.assets.{asset_name}.costs must be a non-empty mapping")
        unknown_objectives = set(raw_costs) - set(objectives)
        if unknown_objectives:
            raise ScenarioError(
                f"Mission asset {asset_name} has unknown objectives: {sorted(unknown_objectives)}"
            )
        costs = {
            objective: _non_negative(raw_costs.get(objective, 0.0), f"mission.assets.{asset_name}.costs.{objective}")
            for objective in objectives
        }
        if not any(costs.values()):
            raise ScenarioError(f"Mission asset {asset_name} must have at least one positive cost")
        assets.append(MissionAssetSpec(str(asset_name), host, costs))

    reward_data = data.get("reward", {}) or {}
    if not isinstance(reward_data, Mapping):
        raise ScenarioError("reward must be a mapping")
    red_costs = {
        str(key): _non_negative(value, f"reward.red_action_costs.{key}")
        for key, value in (reward_data.get("red_action_costs", {}) or {}).items()
    }
    blue_costs = {
        str(key): _non_negative(value, f"reward.blue_action_costs.{key}")
        for key, value in (reward_data.get("blue_action_costs", {}) or {}).items()
    }
    reward = RewardSpec(
        new_user_access=_non_negative(reward_data.get("new_user_access", 0.03), "reward.new_user_access"),
        new_root_access=_non_negative(reward_data.get("new_root_access", 0.10), "reward.new_root_access"),
        red_action_costs=red_costs,
        blue_action_costs=blue_costs,
    )

    dynamics_data = data.get("dynamics", {}) or {}
    if not isinstance(dynamics_data, Mapping):
        raise ScenarioError("dynamics must be a mapping")
    dynamics = DynamicsSpec(
        scan_detection_probability=_probability(
            dynamics_data.get("scan_detection_probability", 0.65),
            "dynamics.scan_detection_probability",
        ),
        escalation_detection_probability=_probability(
            dynamics_data.get("escalation_detection_probability", 0.35),
            "dynamics.escalation_detection_probability",
        ),
    )

    return ScenarioSpec(
        schema_version=schema_version,
        name=name,
        horizon=horizon,
        max_decoys_per_host=max_decoys,
        subnets=tuple(subnets),
        hosts=tuple(hosts),
        red_start=red_start,
        mission=MissionSpec(objectives=objectives, assets=tuple(assets)),
        reward=reward,
        dynamics=dynamics,
    )


def load_scenario(path: str | Path) -> ScenarioSpec:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, Mapping):
        raise ScenarioError(f"Scenario {path} must contain a YAML mapping")
    return scenario_from_mapping(data)
