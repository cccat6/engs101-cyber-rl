from pathlib import Path
import unittest

import numpy as np

from cyberrl_eval import SingleRoleEnv
from cyberrl_eval.agents import GreedyRedAgent, SleepAgent


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "scenarios" / "base.yaml"


class SingleRoleEnvironmentTests(unittest.TestCase):
    def test_red_interface_shapes_and_terminal_flags(self):
        env = SingleRoleEnv(BASE, role="red", opponent=SleepAgent())
        observation, info = env.reset(seed=17)

        self.assertEqual(observation["features"].shape, (289,))
        self.assertEqual(observation["features"].dtype, np.float32)
        self.assertEqual(observation["action_mask"].shape, (40,))
        self.assertEqual(env.observation_size, 289)
        self.assertEqual(env.action_count, 40)
        self.assertEqual(info["role"], "red")

        terminated = truncated = False
        for _ in range(100):
            observation, reward, terminated, truncated, info = env.step(0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertIsInstance(reward, float)
        self.assertEqual(len(env.mission_loss_history), 100)

    def test_blue_interface_runs_fixed_red_opponent(self):
        env = SingleRoleEnv(BASE, role="blue", opponent=GreedyRedAgent())
        observation, _ = env.reset(seed=23)
        action = env.sample_valid_action(observation, np.random.default_rng(4))
        next_observation, _, _, _, info = env.step(action)

        self.assertTrue(observation["action_mask"][action])
        self.assertEqual(next_observation["features"].shape, (289,))
        self.assertIn("opponent_action", info)

    def test_seed_reproduces_opponent_and_transition(self):
        first = SingleRoleEnv(BASE, role="blue", opponent=GreedyRedAgent())
        second = SingleRoleEnv(BASE, role="blue", opponent=GreedyRedAgent())
        first_obs, _ = first.reset(seed=99)
        second_obs, _ = second.reset(seed=99)
        self.assertTrue(np.array_equal(first_obs["features"], second_obs["features"]))

        for _ in range(10):
            first_result = first.step(0)
            second_result = second.step(0)
            self.assertTrue(
                np.array_equal(first_result[0]["features"], second_result[0]["features"])
            )
            self.assertEqual(first_result[1:], second_result[1:])

    def test_convenience_sampling_does_not_change_opponent_rng(self):
        first = SingleRoleEnv(BASE, role="blue", opponent=GreedyRedAgent())
        second = SingleRoleEnv(BASE, role="blue", opponent=GreedyRedAgent())
        first_observation, _ = first.reset(seed=123)
        second.reset(seed=123)

        first.sample_valid_action(first_observation)
        first_result = first.step(0)
        second_result = second.step(0)
        self.assertEqual(first_result[4]["opponent_action"], second_result[4]["opponent_action"])


if __name__ == "__main__":
    unittest.main()
