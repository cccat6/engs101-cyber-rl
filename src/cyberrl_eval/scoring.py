"""Episode execution and mission-loss AUC scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable

import numpy as np

from .agents import Agent
from .environment import CyberRLEnv
from .model import Role
from .scenario import ScenarioSpec


class MissionAUCScorer:
    """Convert normalized mission loss over time into symmetric role scores."""

    @staticmethod
    def score(mission_losses: Iterable[float]) -> dict[str, float]:
        values = np.asarray(tuple(mission_losses), dtype=np.float64)
        if values.size == 0:
            raise ValueError("At least one mission-loss value is required")
        if np.any(values < -1e-9) or np.any(values > 1.0 + 1e-9):
            raise ValueError("Mission loss must stay within [0, 1]")
        mean_loss = float(np.clip(values, 0.0, 1.0).mean())
        return {
            "mean_mission_loss": mean_loss,
            "red": 100.0 * mean_loss,
            "blue": 100.0 * (1.0 - mean_loss),
        }


@dataclass(slots=True)
class EpisodeResult:
    scenario: str
    seed: int
    steps: int
    red_return: float
    blue_return: float
    red_score: float
    blue_score: float
    mean_mission_loss: float
    events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self, include_events: bool = True) -> dict[str, Any]:
        result = {
            "scenario": self.scenario,
            "seed": self.seed,
            "steps": self.steps,
            "returns": {"red": self.red_return, "blue": self.blue_return},
            "scores": {"red": self.red_score, "blue": self.blue_score},
            "mean_mission_loss": self.mean_mission_loss,
        }
        if include_events:
            result["events"] = list(self.events)
        return result


@dataclass(slots=True)
class ScenarioSummary:
    scenario: str
    episodes: int
    red_score_mean: float
    blue_score_mean: float
    red_score_std: float
    blue_score_std: float
    mean_mission_loss: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "episodes": self.episodes,
            "scores": {
                "red": {"mean": self.red_score_mean, "std": self.red_score_std},
                "blue": {"mean": self.blue_score_mean, "std": self.blue_score_std},
            },
            "mean_mission_loss": self.mean_mission_loss,
        }


@dataclass(slots=True)
class EvaluationReport:
    red_agent: str
    blue_agent: str
    episodes_per_scenario: int
    scenario_summaries: list[ScenarioSummary]
    episodes: list[EpisodeResult]

    @property
    def red_score(self) -> float:
        return fmean(summary.red_score_mean for summary in self.scenario_summaries)

    @property
    def blue_score(self) -> float:
        return fmean(summary.blue_score_mean for summary in self.scenario_summaries)

    def to_dict(self, include_events: bool = False) -> dict[str, Any]:
        return {
            "agents": {"red": self.red_agent, "blue": self.blue_agent},
            "episodes_per_scenario": self.episodes_per_scenario,
            "overall_scores": {"red": self.red_score, "blue": self.blue_score},
            "scenarios": [summary.to_dict() for summary in self.scenario_summaries],
            "episodes": [episode.to_dict(include_events=include_events) for episode in self.episodes],
        }


def _agent_name(agent: Agent) -> str:
    return str(getattr(agent, "name", agent.__class__.__name__))


def evaluate_agents(
    scenarios: Iterable[ScenarioSpec | str | Path],
    red_agent: Agent,
    blue_agent: Agent,
    *,
    episodes: int = 30,
    seed: int = 0,
    collect_events: bool = False,
) -> EvaluationReport:
    """Evaluate a fixed agent pair with equal weight per scenario."""

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    scenario_list = list(scenarios)
    if not scenario_list:
        raise ValueError("At least one scenario is required")

    all_results: list[EpisodeResult] = []
    summaries: list[ScenarioSummary] = []

    for scenario_index, scenario in enumerate(scenario_list):
        env = CyberRLEnv(scenario)
        scenario_results: list[EpisodeResult] = []

        for episode_index in range(episodes):
            episode_seed = int(seed + scenario_index * 100_000 + episode_index)
            red_rng = np.random.default_rng(episode_seed ^ 0x5A5A5A5A)
            blue_rng = np.random.default_rng(episode_seed ^ 0xA5A5A5A5)
            red_agent.reset()
            blue_agent.reset()
            observations, _ = env.reset(seed=episode_seed)
            returns = {Role.RED.value: 0.0, Role.BLUE.value: 0.0}
            trace: list[dict[str, Any]] = []

            while env.agents:
                actions = {
                    Role.RED.value: red_agent.act(
                        observations[Role.RED.value],
                        env.action_spaces[Role.RED.value],
                        red_rng,
                    ),
                    Role.BLUE.value: blue_agent.act(
                        observations[Role.BLUE.value],
                        env.action_spaces[Role.BLUE.value],
                        blue_rng,
                    ),
                }
                observations, rewards, _, _, infos = env.step(actions)
                returns[Role.RED.value] += rewards[Role.RED.value]
                returns[Role.BLUE.value] += rewards[Role.BLUE.value]
                if collect_events:
                    trace.extend(infos[Role.RED.value]["events"])

            score = MissionAUCScorer.score(env.mission_loss_history)
            result = EpisodeResult(
                scenario=env.scenario.name,
                seed=episode_seed,
                steps=len(env.mission_loss_history),
                red_return=returns[Role.RED.value],
                blue_return=returns[Role.BLUE.value],
                red_score=score[Role.RED.value],
                blue_score=score[Role.BLUE.value],
                mean_mission_loss=score["mean_mission_loss"],
                events=trace,
            )
            scenario_results.append(result)
            all_results.append(result)

        red_scores = [result.red_score for result in scenario_results]
        blue_scores = [result.blue_score for result in scenario_results]
        summaries.append(
            ScenarioSummary(
                scenario=env.scenario.name,
                episodes=episodes,
                red_score_mean=fmean(red_scores),
                blue_score_mean=fmean(blue_scores),
                red_score_std=pstdev(red_scores),
                blue_score_std=pstdev(blue_scores),
                mean_mission_loss=fmean(result.mean_mission_loss for result in scenario_results),
            )
        )

    return EvaluationReport(
        red_agent=_agent_name(red_agent),
        blue_agent=_agent_name(blue_agent),
        episodes_per_scenario=episodes,
        scenario_summaries=summaries,
        episodes=all_results,
    )

