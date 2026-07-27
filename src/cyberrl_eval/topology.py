"""Compilation of scenario objects into small, fast topology arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scenario import ScenarioSpec


@dataclass(frozen=True, slots=True)
class CompiledTopology:
    scenario: ScenarioSpec
    host_names: tuple[str, ...]
    subnet_names: tuple[str, ...]
    role_names: tuple[str, ...]
    host_index: dict[str, int]
    subnet_index: dict[str, int]
    role_index: dict[str, int]
    host_to_subnet: np.ndarray
    host_roles: np.ndarray
    subnet_adjacency: np.ndarray
    host_adjacency: np.ndarray
    blue_managed: np.ndarray
    mission_weights_by_host: np.ndarray
    critical_hosts: np.ndarray

    @property
    def num_hosts(self) -> int:
        return len(self.host_names)

    @property
    def num_subnets(self) -> int:
        return len(self.subnet_names)

    def service_allowed(self, source_host: int, target_host: int, service_name: str) -> bool:
        source_subnet = int(self.host_to_subnet[source_host])
        target_subnet = int(self.host_to_subnet[target_host])
        if source_subnet == target_subnet:
            return True
        source_spec = self.scenario.subnets[source_subnet]
        allowed = source_spec.connects_to.get(self.subnet_names[target_subnet], frozenset())
        return "*" in allowed or service_name in allowed

    def any_service_allowed(self, source_host: int, target_host: int) -> bool:
        if not self.host_adjacency[source_host, target_host]:
            return False
        target_services = self.scenario.hosts[target_host].services
        if not target_services:
            return True
        return any(self.service_allowed(source_host, target_host, service.name) for service in target_services)


def compile_topology(scenario: ScenarioSpec) -> CompiledTopology:
    host_names = tuple(host.name for host in scenario.hosts)
    subnet_names = tuple(subnet.name for subnet in scenario.subnets)
    role_names = tuple(dict.fromkeys(host.role for host in scenario.hosts))
    host_index = {name: index for index, name in enumerate(host_names)}
    subnet_index = {name: index for index, name in enumerate(subnet_names)}
    # Zero remains the "unknown" role in Red observations.
    role_index = {name: index + 1 for index, name in enumerate(role_names)}

    host_to_subnet = np.asarray(
        [subnet_index[host.subnet] for host in scenario.hosts], dtype=np.int16
    )
    host_roles = np.asarray([role_index[host.role] for host in scenario.hosts], dtype=np.int16)

    subnet_adjacency = np.eye(len(subnet_names), dtype=np.int8)
    for source_idx, subnet in enumerate(scenario.subnets):
        for target in subnet.connects_to:
            subnet_adjacency[source_idx, subnet_index[target]] = 1

    host_adjacency = np.zeros((len(host_names), len(host_names)), dtype=np.int8)
    for source_idx in range(len(host_names)):
        for target_idx in range(len(host_names)):
            source_subnet = host_to_subnet[source_idx]
            target_subnet = host_to_subnet[target_idx]
            host_adjacency[source_idx, target_idx] = subnet_adjacency[source_subnet, target_subnet]

    blue_managed = np.asarray([host.blue_managed for host in scenario.hosts], dtype=np.int8)

    mission_weights = np.zeros(len(host_names), dtype=np.float64)
    for asset in scenario.mission.assets:
        effective_cost = sum(
            scenario.mission.objectives[objective] * cost
            for objective, cost in asset.costs.items()
        )
        mission_weights[host_index[asset.host]] += effective_cost
    total_cost = float(mission_weights.sum())
    if total_cost <= 0:
        raise ValueError("Compiled mission weights must sum to a positive value")
    mission_weights /= total_cost
    critical_hosts = mission_weights > 0

    return CompiledTopology(
        scenario=scenario,
        host_names=host_names,
        subnet_names=subnet_names,
        role_names=role_names,
        host_index=host_index,
        subnet_index=subnet_index,
        role_index=role_index,
        host_to_subnet=host_to_subnet,
        host_roles=host_roles,
        subnet_adjacency=subnet_adjacency,
        host_adjacency=host_adjacency,
        blue_managed=blue_managed,
        mission_weights_by_host=mission_weights,
        critical_hosts=critical_hosts,
    )

