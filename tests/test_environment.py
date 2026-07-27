from pathlib import Path
import unittest

import numpy as np

from cyberrl_eval.environment import CyberRLEnv
from cyberrl_eval.model import ActionKind, BlueEvidence, DiscoveryStage, Role


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "scenarios" / "base.yaml"


class EnvironmentTests(unittest.TestCase):
    def test_action_and_observation_shapes(self):
        env = CyberRLEnv(BASE)
        observations, infos = env.reset(seed=11)

        self.assertEqual(len(env.action_spaces[Role.RED.value]), 40)
        self.assertEqual(len(env.action_spaces[Role.BLUE.value]), 37)
        self.assertEqual(observations[Role.RED.value]["nodes"].shape, (9, 6))
        self.assertEqual(observations[Role.BLUE.value]["nodes"].shape, (9, 6))
        self.assertEqual(observations[Role.RED.value]["adjacency"].shape, (9, 9))
        self.assertEqual(env.flatten(Role.RED, observations[Role.RED.value]).shape, (289,))
        self.assertEqual(env.flatten(Role.BLUE, observations[Role.BLUE.value]).shape, (289,))
        self.assertIn("action_labels", infos[Role.RED.value])

    def test_blue_does_not_observe_initial_hidden_foothold(self):
        env = CyberRLEnv(BASE)
        observations, _ = env.reset(seed=2)
        start = env.topology.host_index[env.scenario.red_start.host]

        self.assertEqual(
            observations[Role.BLUE.value]["nodes"][start, 3],
            BlueEvidence.NO_EVIDENCE,
        )
        self.assertEqual(env.state_snapshot()["access"][env.scenario.red_start.host], "root")

    def test_discover_subnet_updates_red_observation(self):
        env = CyberRLEnv(BASE)
        observations, _ = env.reset(seed=3)
        action = env.action_spaces[Role.RED.value].encode(
            ActionKind.DISCOVER_SUBNET, "enterprise"
        )
        self.assertTrue(observations[Role.RED.value]["action_mask"][action])

        observations, _, _, _, _ = env.step(
            {Role.RED.value: action, Role.BLUE.value: 0}
        )
        enterprise_hosts = env.topology.host_to_subnet == env.topology.subnet_index["enterprise"]
        self.assertTrue(
            np.all(
                observations[Role.RED.value]["nodes"][enterprise_hosts, 2]
                == DiscoveryStage.DISCOVERED
            )
        )

    def test_sleep_episode_has_no_mission_loss(self):
        env = CyberRLEnv(BASE)
        env.reset(seed=5)
        while env.agents:
            env.step({Role.RED.value: 0, Role.BLUE.value: 0})
        self.assertEqual(len(env.mission_loss_history), 100)
        self.assertTrue(all(loss == 0 for loss in env.mission_loss_history))

    def test_invalid_action_is_safe_noop(self):
        env = CyberRLEnv(BASE)
        env.reset(seed=5)
        _, _, _, _, infos = env.step(
            {Role.RED.value: 9999, Role.BLUE.value: 0}
        )
        kinds = [event["kind"] for event in infos[Role.RED.value]["events"]]
        self.assertIn("invalid_action", kinds)


if __name__ == "__main__":
    unittest.main()

