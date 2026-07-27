from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from cyberrl_eval import AgentConfig, Transition
from cyberrl_eval.devices import resolve_device
from cyberrl_eval.submission_api import SubmissionContractError
from cyberrl_eval.submission_runner import (
    evaluate_submission,
    load_agent_factory,
    load_frozen_submission,
    train_submission,
)


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "scenarios" / "base.yaml"


class RecordingAgent:
    def __init__(self, config: AgentConfig, invalid: bool = False):
        self.config = config
        self.invalid = invalid
        self.model = nn.Linear(1, 1)
        self.transitions: list[Transition] = []

    def reset(self, seed: int) -> None:
        del seed

    def act(self, observation, training: bool) -> int:
        del training
        if self.invalid:
            return self.config.num_actions
        return int(np.flatnonzero(observation.action_mask)[0])

    def learn(self, transition: Transition) -> None:
        self.transitions.append(transition)


class SubmissionTests(unittest.TestCase):
    def test_device_and_entrypoint(self):
        self.assertEqual(resolve_device("cpu"), "cpu")
        self.assertTrue(callable(load_agent_factory()))
        with self.assertRaises(ValueError):
            resolve_device("not-a-device")

    def test_exact_budget_checkpoint_and_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "red.pt"
            result = train_submission(
                [BASE],
                role="red",
                total_steps=17,
                seed=4,
                device="cpu",
                checkpoint=checkpoint,
            )
            self.assertEqual(result.steps, 17)
            self.assertTrue(checkpoint.is_file())

            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual(payload["api_version"], 1)
            self.assertTrue(
                all(isinstance(value, torch.Tensor) for value in payload["model"].values())
            )

            frozen = load_frozen_submission(checkpoint, device="cpu")
            self.assertEqual(frozen.config.role, "red")
            summary = evaluate_submission(
                checkpoint,
                [BASE],
                episodes=1,
                seed=100,
                device="cpu",
            )
            self.assertGreaterEqual(summary.overall_score, 0.0)
            self.assertLessEqual(summary.overall_score, 100.0)

    def test_invalid_action_is_rejected(self):
        import cyberrl_eval.submission_runner as runner

        original = runner.load_agent_factory
        runner.load_agent_factory = lambda: (
            lambda config: RecordingAgent(config, invalid=True)
        )
        try:
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(SubmissionContractError, "outside"):
                    train_submission(
                        [BASE],
                        role="red",
                        total_steps=1,
                        seed=0,
                        device="cpu",
                        checkpoint=Path(directory) / "invalid.pt",
                    )
        finally:
            runner.load_agent_factory = original

    def test_final_transition_marks_budget_end(self):
        import cyberrl_eval.submission_runner as runner

        agents: list[RecordingAgent] = []

        def factory(config: AgentConfig):
            agent = RecordingAgent(config)
            agents.append(agent)
            return agent

        original = runner.load_agent_factory
        runner.load_agent_factory = lambda: factory
        try:
            with tempfile.TemporaryDirectory() as directory:
                train_submission(
                    [BASE],
                    role="blue",
                    total_steps=7,
                    seed=0,
                    device="cpu",
                    checkpoint=Path(directory) / "blue.pt",
                )
        finally:
            runner.load_agent_factory = original

        self.assertEqual(len(agents[0].transitions), 7)
        self.assertTrue(agents[0].transitions[-1].done)


if __name__ == "__main__":
    unittest.main()
