from pathlib import Path
import unittest

from cyberrl_eval.scenario import ScenarioError, load_scenario, scenario_from_mapping
from cyberrl_eval.topology import compile_topology


ROOT = Path(__file__).resolve().parents[1]


class ScenarioTests(unittest.TestCase):
    def test_base_scenario_compiles_to_expected_course_scale(self):
        scenario = load_scenario(ROOT / "scenarios" / "base.yaml")
        topology = compile_topology(scenario)

        self.assertEqual(scenario.horizon, 100)
        self.assertEqual(topology.num_subnets, 3)
        self.assertEqual(topology.num_hosts, 9)
        self.assertAlmostEqual(float(topology.mission_weights_by_host.sum()), 1.0)
        self.assertEqual(int(topology.critical_hosts.sum()), 3)

    def test_unknown_subnet_is_rejected(self):
        with self.assertRaises(ScenarioError):
            scenario_from_mapping(
                {
                    "name": "invalid",
                    "subnets": {"entry": {"connects_to": {}}},
                    "hosts": {
                        "host": {"subnet": "missing", "role": "client", "services": []}
                    },
                    "red_start": {"host": "host", "access": "root"},
                    "mission": {
                        "objectives": {"availability": 1},
                        "assets": {
                            "asset": {"host": "host", "costs": {"availability": 1}}
                        },
                    },
                }
            )


if __name__ == "__main__":
    unittest.main()

