"""Train and test the student agent on the public backend."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .submission_runner import evaluate_submission, train_submission


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cyberrl-submission")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--device", default="auto")
    return parser


def _scenarios() -> list[Path]:
    scenarios = sorted(Path.cwd().glob("scenarios/*.yaml"))
    if not scenarios:
        raise SystemExit("Run this command from the repository root")
    return scenarios


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    scenarios = _scenarios()
    output = Path("checkpoints")

    for role in ("red", "blue"):
        checkpoint = output / f"{role}.pt"
        train_submission(
            scenarios,
            role=role,
            total_steps=args.steps,
            seed=0,
            device=args.device,
            checkpoint=checkpoint,
        )
        evaluation = evaluate_submission(
            checkpoint,
            scenarios,
            episodes=args.eval_episodes,
            seed=10_000,
            device=args.device,
        )
        print(
            f"{role.capitalize()}: score={evaluation.overall_score:.2f}, "
            f"checkpoint={checkpoint}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
