"""Train and test the student agent on the public environment."""

from __future__ import annotations

import argparse
from pathlib import Path

from cyberrl_eval.submission_runner import evaluate_submission, train_submission


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    scenario = Path("scenarios/base.yaml")
    output = Path("checkpoints")

    for role in ("red", "blue"):
        checkpoint = output / f"{role}.pt"
        training = train_submission(
            [scenario],
            role=role,
            total_steps=args.steps,
            seed=0,
            device=args.device,
            checkpoint=checkpoint,
        )
        result = evaluate_submission(
            checkpoint,
            [scenario],
            episodes=args.eval_episodes,
            seed=10_000,
            device=args.device,
        )
        print(
            f"{role.capitalize()}: score={result.overall_score:.2f}, "
            f"parameters={training.checkpoint_elements:,}, "
            f"checkpoint={checkpoint}"
        )


if __name__ == "__main__":
    main()
