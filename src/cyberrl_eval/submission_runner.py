"""Trusted training, checkpoint, and local evaluation runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping

import torch
from torch import nn

from .devices import resolve_device, seed_everything
from .environment import CyberRLEnv
from .model import Role
from .scoring import MissionAUCScorer
from .single_agent import SingleRoleEnv
from .submission_api import (
    SUBMISSION_API_VERSION,
    AgentConfig,
    AgentFactory,
    Observation,
    StudentAgent,
    SubmissionContractError,
    Transition,
    copy_observation,
    validate_action,
)


CHECKPOINT_FORMAT = "engs101-cyber-rl"
DEFAULT_ENTRYPOINT = "student_agent:create_agent"
TEACHER_BASELINE_DIR = Path(__file__).with_name("teacher_baseline")
MAX_MODEL_PARAMETERS = 1_000_000


@dataclass(frozen=True, slots=True)
class TrainingResult:
    role: str
    steps: int
    episodes: int
    device: str
    checkpoint: Path
    checkpoint_bytes: int
    checkpoint_elements: int

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["checkpoint"] = str(self.checkpoint)
        return result


@dataclass(frozen=True, slots=True)
class ScenarioEvaluation:
    scenario: str
    episodes: int
    score: float


@dataclass(frozen=True, slots=True)
class LocalEvaluationResult:
    role: str
    scenarios: tuple[ScenarioEvaluation, ...]

    @property
    def overall_score(self) -> float:
        return fmean(item.score for item in self.scenarios)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "overall_score": self.overall_score,
            "scenarios": [asdict(item) for item in self.scenarios],
        }


@dataclass(slots=True)
class FrozenSubmission:
    agent: StudentAgent
    config: AgentConfig

    def reset(self, seed: int) -> None:
        self.agent.reset(int(seed))

    def act(self, observation) -> int:
        view = copy_observation(
            observation,
            observation_dim=self.config.observation_dim,
            num_actions=self.config.num_actions,
        )
        action = self.agent.act(view, False)
        return validate_action(action, view, self.config.num_actions)


@dataclass(slots=True)
class FrozenTeacherOpponent:
    role: Role
    policy: FrozenSubmission
    name: str = "teacher-baseline"

    def reset(self, seed: int | None) -> None:
        self.policy.reset(0 if seed is None else int(seed))

    def act(
        self,
        observation: Mapping[str, object],
        environment: CyberRLEnv,
    ) -> int:
        return self.policy.act(
            {
                "features": environment.flatten(self.role, observation),
                "action_mask": observation["action_mask"],
            }
        )


def load_agent_factory() -> AgentFactory:
    module_name, function_name = DEFAULT_ENTRYPOINT.split(":", 1)
    try:
        factory = getattr(import_module(module_name), function_name)
    except Exception as error:
        raise SubmissionContractError(
            f"Could not import {DEFAULT_ENTRYPOINT}: {error}"
        ) from error
    if not callable(factory):
        raise SubmissionContractError(f"{DEFAULT_ENTRYPOINT} is not callable")
    return factory


def _require_agent(agent: object) -> StudentAgent:
    if not isinstance(agent, StudentAgent):
        raise SubmissionContractError(
            "create_agent() must return an object with reset(), act(), and learn()"
        )
    model = getattr(agent, "model", None)
    if not isinstance(model, nn.Module):
        raise SubmissionContractError("The agent must expose model: torch.nn.Module")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count > MAX_MODEL_PARAMETERS:
        raise SubmissionContractError(
            f"Model has {parameter_count:,} parameters; the limit is "
            f"{MAX_MODEL_PARAMETERS:,} per role"
        )
    return agent


def _environments(
    scenarios: Iterable[str | Path],
    role: Role,
    *,
    device: str = "cpu",
) -> list[SingleRoleEnv]:
    paths = [Path(path).resolve() for path in scenarios]
    if not paths:
        raise ValueError("At least one scenario is required")
    opponent_role = Role.BLUE if role is Role.RED else Role.RED
    opponent = FrozenTeacherOpponent(
        opponent_role,
        load_teacher_policy(opponent_role, device=device),
    )
    environments = [
        SingleRoleEnv(
            path,
            role=role,
            opponent_controller=opponent,
        )
        for path in paths
    ]
    first = environments[0]
    for environment in environments[1:]:
        if (
            environment.observation_size != first.observation_size
            or environment.action_count != first.action_count
            or environment.action_labels != first.action_labels
        ):
            raise ValueError("All scenarios must use the same model interface")
    return environments


def _weights(agent: StudentAgent) -> dict[str, torch.Tensor]:
    state = agent.model.state_dict()
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise SubmissionContractError(f"Model weight {key!r} is not finite")
        cleaned[key] = value.detach().cpu().contiguous().clone()
    return cleaned


def _config_payload(config: AgentConfig) -> dict[str, Any]:
    return asdict(config)


def _config_from_payload(payload: Mapping[str, Any], device: str) -> AgentConfig:
    try:
        values = dict(payload)
        values["device"] = device
        return AgentConfig(**values)
    except (TypeError, ValueError) as error:
        raise SubmissionContractError(f"Invalid checkpoint config: {error}") from error


def save_checkpoint(
    agent: StudentAgent,
    config: AgentConfig,
    path: str | Path,
) -> tuple[int, int]:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = _weights(agent)
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "api_version": SUBMISSION_API_VERSION,
            "config": _config_payload(config),
            "model": state,
        },
        destination,
    )
    return destination.stat().st_size, sum(value.numel() for value in state.values())


def _load_payload(path: str | Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(
            Path(path).resolve(),
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise SubmissionContractError(f"Could not load checkpoint: {error}") from error
    if not isinstance(payload, Mapping):
        raise SubmissionContractError("Invalid checkpoint")
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise SubmissionContractError("Unknown checkpoint format")
    if payload.get("api_version") != SUBMISSION_API_VERSION:
        raise SubmissionContractError("Unsupported checkpoint API version")
    if not isinstance(payload.get("config"), Mapping):
        raise SubmissionContractError("Checkpoint is missing config")
    if not isinstance(payload.get("model"), Mapping):
        raise SubmissionContractError("Checkpoint is missing model weights")
    return payload


def load_frozen_submission(
    checkpoint: str | Path,
    *,
    device: str,
    factory: AgentFactory | None = None,
) -> FrozenSubmission:
    resolved_device = resolve_device(device)
    payload = _load_payload(checkpoint)
    config = _config_from_payload(payload["config"], resolved_device)
    seed_everything(config.seed)
    selected_factory = factory if factory is not None else load_agent_factory()
    agent = _require_agent(selected_factory(config))
    try:
        agent.model.load_state_dict(payload["model"], strict=True)
    except Exception as error:
        raise SubmissionContractError(
            f"Model does not match checkpoint: {error}"
        ) from error
    agent.model.eval()
    return FrozenSubmission(agent, config)


def teacher_checkpoint(role: Role | str) -> Path:
    return TEACHER_BASELINE_DIR / f"{Role(role).value}.pt"


def load_teacher_policy(role: Role | str, *, device: str) -> FrozenSubmission:
    role = Role(role)
    checkpoint = teacher_checkpoint(role)
    if not checkpoint.is_file():
        raise RuntimeError(f"Frozen teacher checkpoint is missing for {role.value}")
    module = import_module("cyberrl_eval.teacher_baseline.student_agent")
    return load_frozen_submission(
        checkpoint,
        device=device,
        factory=module.create_agent,
    )


def train_submission(
    scenarios: Iterable[str | Path],
    *,
    role: Role | str,
    total_steps: int,
    seed: int,
    device: str,
    checkpoint: str | Path,
) -> TrainingResult:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    role = Role(role)
    resolved_device = resolve_device(device)
    environments = _environments(scenarios, role, device=resolved_device)
    first = environments[0]
    config = AgentConfig(
        role=role.value,
        observation_dim=first.observation_size,
        num_actions=first.action_count,
        total_steps=int(total_steps),
        seed=int(seed),
        device=resolved_device,
    )
    seed_everything(seed)
    agent = _require_agent(load_agent_factory()(config))
    agent.model.train()

    steps = 0
    episode = 0
    while steps < total_steps:
        environment = environments[episode % len(environments)]
        episode_seed = int(seed) + episode
        raw_observation, _ = environment.reset(seed=episode_seed)
        observation = copy_observation(
            raw_observation,
            observation_dim=config.observation_dim,
            num_actions=config.num_actions,
        )
        agent.reset(episode_seed)
        episode += 1

        while steps < total_steps:
            action = validate_action(
                agent.act(observation, True),
                observation,
                config.num_actions,
            )
            raw_next, reward, terminated, truncated, _ = environment.step(action)
            next_observation = copy_observation(
                raw_next,
                observation_dim=config.observation_dim,
                num_actions=config.num_actions,
            )
            steps += 1
            done = bool(
                terminated
                or truncated
                or (steps == total_steps)
            )
            agent.learn(
                Transition(
                    observation=observation,
                    action=action,
                    reward=float(reward),
                    next_observation=next_observation,
                    done=done,
                )
            )
            observation = next_observation
            if terminated or truncated:
                break

    checkpoint_path = Path(checkpoint).resolve()
    checkpoint_bytes, checkpoint_elements = save_checkpoint(
        agent, config, checkpoint_path
    )
    return TrainingResult(
        role=role.value,
        steps=steps,
        episodes=episode,
        device=resolved_device,
        checkpoint=checkpoint_path,
        checkpoint_bytes=checkpoint_bytes,
        checkpoint_elements=checkpoint_elements,
    )


def evaluate_submission(
    checkpoint: str | Path,
    scenarios: Iterable[str | Path],
    *,
    episodes: int,
    seed: int,
    device: str,
) -> LocalEvaluationResult:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    frozen = load_frozen_submission(checkpoint, device=device)
    role = Role(frozen.config.role)
    environments = _environments(
        scenarios,
        role,
        device=resolve_device(device),
    )
    first = environments[0]
    if (
        first.observation_size != frozen.config.observation_dim
        or first.action_count != frozen.config.num_actions
    ):
        raise SubmissionContractError(
            "Checkpoint dimensions do not match the environment"
        )

    summaries: list[ScenarioEvaluation] = []
    for scenario_index, environment in enumerate(environments):
        scores: list[float] = []
        for episode_index in range(episodes):
            episode_seed = int(seed) + scenario_index * 100_000 + episode_index
            observation, _ = environment.reset(seed=episode_seed)
            frozen.reset(episode_seed)
            while True:
                action = frozen.act(observation)
                observation, _, terminated, truncated, _ = environment.step(action)
                if terminated or truncated:
                    break
            score = MissionAUCScorer.score(environment.mission_loss_history)
            scores.append(float(score[role.value]))
        summaries.append(
            ScenarioEvaluation(
                scenario=(
                    f"{environment.env.scenario.name} "
                    f"vs {environment.opponent.name}"
                ),
                episodes=episodes,
                score=fmean(scores),
            )
        )
    return LocalEvaluationResult(role.value, tuple(summaries))
