import math
import unittest

import numpy as np

from homesocial.env import LANGUAGE_NECESSARY_MODE, STOCHASTIC_BODY, Action
from homesocial.observations import MASKED_INTEROCEPTION, observation_vector_size
from homesocial.option_counterfactual_language import (
    _balanced_option_trend_indices,
    collect_option_counterfactual_dataset,
)
from homesocial.recurrent_ac import RecurrentActorCritic, RecurrentConfig


class OptionCounterfactualLanguageTests(unittest.TestCase):
    def test_option_trend_balancing_equalizes_trends_inside_options(self):
        option_labels = [0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2]
        trend_labels = [0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1]
        indices = _balanced_option_trend_indices(
            option_labels,
            trend_labels,
            np.random.default_rng(1),
        )

        selected_options = np.asarray([option_labels[int(index)] for index in indices])
        selected_trends = np.asarray([trend_labels[int(index)] for index in indices])

        self.assertEqual(len(indices), 9)
        self.assertEqual(set(selected_options.tolist()), {0, 1})
        expected = {0: [2, 2, 2], 1: [1, 1, 1]}
        for option in (0, 1):
            option_trends = selected_trends[selected_options == option]
            self.assertEqual(
                [int(np.sum(option_trends == trend)) for trend in range(3)],
                expected[option],
            )

    def test_collect_option_counterfactual_dataset_runs(self):
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

        dataset = collect_option_counterfactual_dataset(
            base_model,
            config,
            episodes=2,
            seed=5,
            horizon=3,
            balance_target="none",
            max_states=20,
        )

        self.assertEqual(dataset.features.shape[1], 8)
        self.assertEqual(dataset.features.shape[0], dataset.trend_labels.shape[0])
        self.assertEqual(dataset.features.shape[0], dataset.action_labels.shape[0])
        self.assertLessEqual(int(dataset.dominant_labels.max()), 3)
        self.assertTrue(math.isfinite(float(dataset.needs[0, 0])))

    def test_collect_option_counterfactual_dataset_latent_rollout_runs(self):
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

        dataset = collect_option_counterfactual_dataset(
            base_model,
            config,
            episodes=2,
            seed=6,
            horizon=3,
            balance_target="none",
            rollout_mode="latent",
            max_states=20,
        )

        self.assertEqual(dataset.features.shape[1], 8)
        self.assertGreater(dataset.features.shape[0], 0)
        self.assertTrue(math.isfinite(float(dataset.features[0, 0])))


if __name__ == "__main__":
    unittest.main()
