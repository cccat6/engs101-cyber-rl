"""Command-line entry points for scenario inspection and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .agents import AGENT_FACTORIES
from .environment import CyberRLEnv
from .model import Role
from .scoring import evaluate_agents


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cyberrl-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect one scenario")
    inspect_parser.add_argument("scenario", type=Path)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a Red/Blue agent pair")
    evaluate_parser.add_argument(
        "--scenario",
        "-s",
        action="append",
        type=Path,
        dest="scenarios",
        help="Scenario YAML; repeat for equal-weight multi-scenario evaluation",
    )
    evaluate_parser.add_argument(
        "--red",
        choices=("sleep", "random", "greedy-red", "focused-red"),
        default="greedy-red",
    )
    evaluate_parser.add_argument(
        "--blue",
        choices=("sleep", "random", "reactive-blue"),
        default="reactive-blue",
    )
    evaluate_parser.add_argument("--episodes", type=int, default=30)
    evaluate_parser.add_argument("--seed", type=int, default=0)
    evaluate_parser.add_argument("--events", action="store_true", help="Include step events in JSON")
    evaluate_parser.add_argument("--output", type=Path)
    return parser


def _default_scenarios() -> list[Path]:
    scenarios = sorted(Path.cwd().glob("scenarios/*.yaml"))
    if not scenarios:
        raise SystemExit("No --scenario supplied and no scenarios/*.yaml files found")
    return scenarios


def _inspect(path: Path) -> int:
    env = CyberRLEnv(path)
    observations, _ = env.reset(seed=0)
    summary = {
        "name": env.scenario.name,
        "horizon": env.scenario.horizon,
        "subnets": list(env.topology.subnet_names),
        "hosts": list(env.topology.host_names),
        "mission_weights_by_host": {
            host: float(env.topology.mission_weights_by_host[index])
            for index, host in enumerate(env.topology.host_names)
            if env.topology.mission_weights_by_host[index] > 0
        },
        "actions": {
            role: len(env.action_spaces[role])
            for role in env.possible_agents
        },
        "flat_observation_size": {
            role: int(env.flatten(role, observations[role]).size)
            for role in env.possible_agents
        },
    }
    print(json.dumps(summary, indent=2))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    scenarios = args.scenarios or _default_scenarios()
    red_agent = AGENT_FACTORIES[args.red]()
    blue_agent = AGENT_FACTORIES[args.blue]()
    report = evaluate_agents(
        scenarios,
        red_agent,
        blue_agent,
        episodes=args.episodes,
        seed=args.seed,
        collect_events=args.events,
    )
    result = report.to_dict(include_events=args.events)
    print(
        f"Red ({report.red_agent}): {report.red_score:.2f} | "
        f"Blue ({report.blue_agent}): {report.blue_score:.2f}"
    )
    for summary in report.scenario_summaries:
        print(
            f"  {summary.scenario}: red={summary.red_score_mean:.2f}±{summary.red_score_std:.2f}, "
            f"blue={summary.blue_score_mean:.2f}±{summary.blue_score_std:.2f}"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote {args.output}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "inspect":
        return _inspect(args.scenario)
    if args.command == "evaluate":
        return _evaluate(args)
    raise AssertionError(f"Unhandled command {args.command}")
