import math
import unittest

from homesocial.env import LANGUAGE_NECESSARY_MODE, Action
from homesocial.interoception import evaluate_interoception
from homesocial.observations import MASKED_INTEROCEPTION, observation_vector_size
from homesocial.recurrent_ac import RecurrentActorCritic, RecurrentConfig


class InteroceptionTests(unittest.TestCase):
    def test_temporal_interoception_battery_runs(self):
        config = RecurrentConfig(
            condition="grounded",
            include_language_channel=True,
            max_steps=16,
            hidden_size=16,
            randomize_world=False,
            diagnostic_mode=LANGUAGE_NECESSARY_MODE,
            interoception_mode=MASKED_INTEROCEPTION,
        )
        model = RecurrentActorCritic(
            observation_vector_size(
                include_language=True,
                include_object_kinds=False,
            ),
            hidden_size=16,
            action_size=len(Action),
        )

        full = evaluate_interoception(
            model,
            config,
            episodes=2,
            seed=1,
            history_mode="full",
            max_samples=20,
        )
        latest = evaluate_interoception(
            model,
            config,
            episodes=2,
            seed=1,
            history_mode="latest",
            max_samples=20,
        )

        self.assertGreater(full.samples, 0)
        self.assertTrue(math.isfinite(full.next_need_mse))
        self.assertEqual(full.samples, latest.samples)


if __name__ == "__main__":
    unittest.main()
