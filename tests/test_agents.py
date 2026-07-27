from pathlib import Path
import unittest

import numpy as np

from cyberrl_eval.agents import FocusedRedAgent
from cyberrl_eval.environment import CyberRLEnv
from cyberrl_eval.model import ActionKind, Role


ROOT = Path(__file__).resolve().parents[1]


class AgentTests(unittest.TestCase):
    def test_focused_red_expands_before_scanning_entry_hosts(self):
        env = CyberRLEnv(ROOT / "scenarios" / "base.yaml")
        observations, _ = env.reset(seed=1)
        agent = FocusedRedAgent()
        action = agent.act(
            observations[Role.RED.value],
            env.action_spaces[Role.RED.value],
            np.random.default_rng(2),
        )
        self.assertEqual(
            env.action_spaces[Role.RED.value].decode(action).kind,
            ActionKind.DISCOVER_SUBNET,
        )


if __name__ == "__main__":
    unittest.main()
