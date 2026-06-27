import math
import unittest

import numpy as np

from homesocial.counterfactual_trend_language import (
    _balanced_action_trend_indices,
    collect_counterfactual_trend_dataset,
)
from homesocial.env import LANGUAGE_NECESSARY_MODE, STOCHASTIC_BODY, Action
from homesocial.observations import MASKED_INTEROCEPTION, observation_vector_size
from homesocial.recurrent_ac import RecurrentActorCritic, RecurrentConfig


class CounterfactualTrendLanguageTests(unittest.TestCase):
    def test_action_trend_balancing_equalizes_trends_inside_actions(self):
        action_labels = [0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2]
        trend_labels = [0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1]
        indices = _balanced_action_trend_indices(
            action_labels,
            trend_labels,
            np.random.default_rng(1),
        )

        selected_actions = np.asarray([action_labels[int(index)] for index in indices])
        selected_trends = np.asarray([trend_labels[int(index)] for index in indices])

        self.assertEqual(len(indices), 9)
        self.assertEqual(set(selected_actions.tolist()), {0, 1})
        expected = {0: [2, 2, 2], 1: [1, 1, 1]}
        for action in (0, 1):
            action_trends = selected_trends[selected_actions == action]
            self.assertEqual(
                [int(np.sum(action_trends == trend)) for trend in range(3)],
                expected[action],
            )

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
        self.assertEqual(dataset.features.shape[0], dataset.action_labels.shape[0])
        self.assertLessEqual(int(dataset.dominant_labels.max()), 3)
        self.assertTrue(math.isfinite(float(dataset.needs[0, 0])))

    def test_collect_counterfactual_trend_dataset_action_balanced_runs(self):
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
            episodes=8,
            seed=3,
            balance_target="action_trend",
            max_states=64,
        )

        self.assertEqual(dataset.features.shape[1], 8)
        self.assertEqual(dataset.features.shape[0], dataset.action_labels.shape[0])
        self.assertGreater(dataset.features.shape[0], 0)


if __name__ == "__main__":
    unittest.main()
