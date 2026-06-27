import math
import unittest

from homesocial.counterfactual_trend_language import (
    collect_counterfactual_trend_dataset,
)
from homesocial.env import LANGUAGE_NECESSARY_MODE, STOCHASTIC_BODY, Action
from homesocial.observations import MASKED_INTEROCEPTION, observation_vector_size
from homesocial.recurrent_ac import RecurrentActorCritic, RecurrentConfig


class CounterfactualTrendLanguageTests(unittest.TestCase):
    def test_collect_counterfactual_trend_dataset_runs(self):
        config = RecurrentConfig(
            condition="grounded",
            include_language_channel=True,
            max_steps=16,
            hidden_size=16,
            randomize_world=False,
            diagnostic_mode=LANGUAGE_NECESSARY_MODE,
            interoception_mode=MASKED_INTEROCEPTION,
            body_dynamics_mode=STOCHASTIC_BODY,
        )
        base_model = RecurrentActorCritic(
            observation_vector_size(
                include_language=True,
                include_object_kinds=False,
                body_dynamics_mode=STOCHASTIC_BODY,
            ),
            hidden_size=16,
            action_size=len(Action),
        )

        dataset = collect_counterfactual_trend_dataset(
            base_model,
            config,
            episodes=4,
            seed=1,
            balance_target="none",
            max_states=24,
        )

        self.assertEqual(dataset.features.shape[1], 8)
        self.assertEqual(dataset.features.shape[0], dataset.trend_labels.shape[0])
        self.assertLessEqual(int(dataset.dominant_labels.max()), 3)
        self.assertTrue(math.isfinite(float(dataset.needs[0, 0])))


if __name__ == "__main__":
    unittest.main()
