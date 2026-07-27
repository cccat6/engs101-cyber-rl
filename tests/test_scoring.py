from pathlib import Path
import unittest

from cyberrl_eval.agents import GreedyRedAgent, SleepAgent
from cyberrl_eval.scoring import MissionAUCScorer, evaluate_agents


ROOT = Path(__file__).resolve().parents[1]


class ScoringTests(unittest.TestCase):
    def test_auc_score_is_symmetric_and_normalized(self):
        score = MissionAUCScorer.score([0.0, 0.5, 1.0])
        self.assertAlmostEqual(score["mean_mission_loss"], 0.5)
        self.assertAlmostEqual(score["red"], 50.0)
        self.assertAlmostEqual(score["blue"], 50.0)

    def test_evaluation_is_reproducible(self):
        scenarios = [ROOT / "scenarios" / "base.yaml"]
        first = evaluate_agents(
            scenarios, GreedyRedAgent(), SleepAgent(), episodes=3, seed=19
        )
        second = evaluate_agents(
            scenarios, GreedyRedAgent(), SleepAgent(), episodes=3, seed=19
        )
        self.assertEqual(first.to_dict(), second.to_dict())
        for episode in first.episodes:
            self.assertAlmostEqual(episode.red_score + episode.blue_score, 100.0)

    def test_scenarios_are_equal_weight_in_overall_score(self):
        report = evaluate_agents(
            [
                ROOT / "scenarios" / "base.yaml",
                ROOT / "scenarios" / "segmented.yaml",
            ],
            GreedyRedAgent(),
            SleepAgent(),
            episodes=2,
            seed=23,
        )
        expected = sum(summary.red_score_mean for summary in report.scenario_summaries) / 2
        self.assertAlmostEqual(report.red_score, expected)


if __name__ == "__main__":
    unittest.main()

