import math
import unittest

from homesocial.env import LANGUAGE_NECESSARY_MODE, Action
from homesocial.observations import observation_vector_size
from homesocial.recurrent_ac import RecurrentActorCritic, RecurrentConfig
from homesocial.self_battery import run_consequence_probe


class SelfBatteryTests(unittest.TestCase):
    def test_consequence_probe_returns_finite_metrics(self):
        config = RecurrentConfig(
            condition="grounded",
            include_language_channel=True,
            max_steps=30,
            hidden_size=16,
            randomize_world=False,
            diagnostic_mode=LANGUAGE_NECESSARY_MODE,
        )
        model = RecurrentActorCritic(
            observation_vector_size(include_language=True, include_object_kinds=False),
            hidden_size=16,
            action_size=len(Action),
        )

        result = run_consequence_probe(
            model,
            config,
            teacher_mode="grounded",
            seeds=[1, 2, 3],
            max_samples=5,
            rollout_policy="teacher",
        )

        self.assertGreater(result.samples, 0)
        for value in (
            result.next_need_mse,
            result.reward_mse,
            result.viability_rank_accuracy,
            result.need_direction_accuracy,
            result.action_sensitivity,
        ):
            self.assertTrue(math.isfinite(value))


if __name__ == "__main__":
    unittest.main()
