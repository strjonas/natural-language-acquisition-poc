import unittest

import mlx.core as mx
import numpy as np

from homesocial.option_mediation import (
    OptionMediationDataset,
    _balanced_target_option_indices,
    collect_option_mediation_source,
    intervene_option_mediation_features,
    majority_option_result,
    option_mediation_dataset_from_source,
    target_count_string,
)
from homesocial.env import LANGUAGE_NECESSARY_MODE, STOCHASTIC_BODY, Action
from homesocial.observations import MASKED_INTEROCEPTION, observation_vector_size
from homesocial.recurrent_ac import RecurrentActorCritic, RecurrentConfig


def _dataset(
    targets: list[int],
    values: np.ndarray | None = None,
    feature_width: int = 12,
) -> OptionMediationDataset:
    rows = len(targets)
    features = np.arange(rows * 5 * feature_width, dtype=np.float32).reshape(
        rows,
        5,
        feature_width,
    )
    if values is None:
        values = np.zeros((rows, 5), dtype=np.float32)
        for row, target in enumerate(targets):
            values[row, target] = 1.0
    return OptionMediationDataset(
        features=mx.array(features, dtype=mx.float32),
        option_values=mx.array(values, dtype=mx.float32),
        target_options=mx.array(targets, dtype=mx.int32),
        current_lowest=mx.zeros((rows,), dtype=mx.float32),
    )


class OptionMediationTests(unittest.TestCase):
    def test_balanced_target_option_indices_equalizes_non_empty_targets(self):
        indices = _balanced_target_option_indices(
            [0, 0, 0, 1, 1, 2],
            np.random.default_rng(1),
        )

        selected = np.array([0, 0, 0, 1, 1, 2])[indices]
        self.assertEqual(len(indices), 3)
        self.assertEqual(
            [int(np.sum(selected == option)) for option in range(3)],
            [1, 1, 1],
        )

    def test_grouped_delta_intervention_preserves_values_and_targets(self):
        dataset = _dataset([0, 1])

        intervened = intervene_option_mediation_features(
            dataset,
            intervention="negate_delta",
        )

        features = np.asarray(dataset.features)
        intervened_features = np.asarray(intervened.features)
        np.testing.assert_allclose(intervened_features[:, :, :8], features[:, :, :8])
        np.testing.assert_allclose(
            intervened_features[:, :, 8:12],
            -features[:, :, 8:12],
        )
        np.testing.assert_array_equal(
            np.asarray(intervened.option_values),
            np.asarray(dataset.option_values),
        )
        np.testing.assert_array_equal(
            np.asarray(intervened.target_options),
            np.asarray(dataset.target_options),
        )

    def test_delta_only_intervention_accepts_width_four(self):
        dataset = _dataset([0, 1], feature_width=4)

        intervened = intervene_option_mediation_features(
            dataset,
            intervention="negate_delta",
        )

        np.testing.assert_allclose(
            np.asarray(intervened.features),
            -np.asarray(dataset.features),
        )
        with self.assertRaises(ValueError):
            intervene_option_mediation_features(
                dataset,
                intervention="shuffle_current",
            )

    def test_majority_option_result_reports_choice_value_and_regret(self):
        train = _dataset([1, 1, 2])
        eval_dataset = _dataset(
            [0, 1],
            values=np.array(
                [
                    [0.9, 0.2, 0.1, 0.0, 0.0],
                    [0.1, 0.8, 0.4, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
        )

        result = majority_option_result(train, eval_dataset)

        self.assertEqual(result.model_control, "target_majority")
        self.assertEqual(result.samples, 2)
        self.assertAlmostEqual(result.choice_accuracy, 0.5)
        self.assertAlmostEqual(result.mean_chosen_lowest, 0.5)
        self.assertAlmostEqual(result.mean_oracle_lowest, 0.85)
        self.assertAlmostEqual(result.mean_regret, 0.35)

    def test_target_count_string_uses_stable_option_names(self):
        self.assertEqual(
            target_count_string(_dataset([0, 0, 2, 4])),
            "seek_food=2;seek_water=0;seek_shelter=1;rest=0;wait=1",
        )

    def test_collect_source_and_featurize_runs(self):
        config = RecurrentConfig(
            condition="grounded",
            include_language_channel=True,
            max_steps=12,
            hidden_size=16,
            randomize_world=False,
            diagnostic_mode=LANGUAGE_NECESSARY_MODE,
            interoception_mode=MASKED_INTEROCEPTION,
            body_dynamics_mode=STOCHASTIC_BODY,
        )
        model = RecurrentActorCritic(
            observation_vector_size(
                include_language=True,
                include_object_kinds=False,
                body_dynamics_mode=STOCHASTIC_BODY,
            ),
            hidden_size=16,
            action_size=len(Action),
        )

        source = collect_option_mediation_source(
            config,
            episodes=1,
            seed=13,
            horizon=2,
            balance_target="none",
            max_states=3,
            min_value_gap=0.0,
        )
        dataset = option_mediation_dataset_from_source(source, model)
        delta_dataset = option_mediation_dataset_from_source(
            source,
            model,
            feature_mode="delta",
        )

        self.assertGreater(len(source.samples), 0)
        self.assertEqual(dataset.features.shape[1:], (5, 12))
        self.assertEqual(delta_dataset.features.shape[1:], (5, 4))
        self.assertEqual(dataset.option_values.shape[1], 5)

    def test_option_action_noise_changes_recorded_branch_scripts(self):
        config = RecurrentConfig(
            condition="grounded",
            include_language_channel=True,
            max_steps=12,
            hidden_size=16,
            randomize_world=False,
            diagnostic_mode=LANGUAGE_NECESSARY_MODE,
            interoception_mode=MASKED_INTEROCEPTION,
            body_dynamics_mode=STOCHASTIC_BODY,
        )

        deterministic_source = collect_option_mediation_source(
            config,
            episodes=1,
            seed=17,
            horizon=2,
            balance_target="none",
            max_states=2,
            min_value_gap=0.0,
            option_action_noise=0.0,
        )
        noisy_source = collect_option_mediation_source(
            config,
            episodes=1,
            seed=17,
            horizon=2,
            balance_target="none",
            max_states=2,
            min_value_gap=0.0,
            option_action_noise=1.0,
        )

        self.assertGreater(len(noisy_source.samples), 0)
        self.assertNotEqual(
            deterministic_source.samples[0].option_actions,
            noisy_source.samples[0].option_actions,
        )
        with self.assertRaises(ValueError):
            collect_option_mediation_source(
                config,
                episodes=1,
                seed=17,
                option_action_noise=1.1,
            )


if __name__ == "__main__":
    unittest.main()
