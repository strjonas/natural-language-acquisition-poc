import unittest

import mlx.core as mx
import numpy as np

from homesocial.option_mediation import (
    OptionMediationDataset,
    _balanced_target_option_indices,
    _resampled_target_option_indices,
    _should_keep_option_mediation_sample,
    collect_option_mediation_source,
    intervene_option_mediation_features,
    majority_option_result,
    option_mediation_dataset_with_targets,
    option_mediation_dataset_from_source,
    predicted_future_option_choices,
    predicted_future_option_scores,
    predicted_future_option_result,
    target_count_string,
    staged_option_population_from_mediator,
    train_option_mediator,
    evaluate_option_mediator,
    evaluate_option_population_mediator,
    evaluate_option_population_receiver_transfer,
    evaluate_option_field_use_for_population_sender,
    evaluate_option_receiver_transfer,
    train_option_population_mediator,
    train_option_field_action_receiver_for_population_sender,
    train_option_receiver_for_sender,
    train_option_receiver_for_population_sender,
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

    def test_resampled_target_option_indices_balances_without_shrinking(self):
        targets = np.array([0, 0, 0, 0, 1, 2], dtype=np.int32)
        indices = _resampled_target_option_indices(
            targets.tolist(),
            np.random.default_rng(2),
        )

        selected = targets[indices]
        self.assertEqual(len(indices), 6)
        self.assertEqual(
            [int(np.sum(selected == option)) for option in range(3)],
            [2, 2, 2],
        )
        self.assertGreater(int(np.sum(indices == 4)), 1)

    def test_option_mediation_sample_filter_can_require_positive_delta(self):
        self.assertFalse(
            _should_keep_option_mediation_sample(
                np.array([0.50, 0.51, 0.30], dtype=np.float32),
                current_lowest=0.40,
                min_value_gap=0.02,
                min_positive_delta=None,
            )
        )
        self.assertTrue(
            _should_keep_option_mediation_sample(
                np.array([0.50, 0.53, 0.30], dtype=np.float32),
                current_lowest=0.40,
                min_value_gap=0.02,
                min_positive_delta=None,
            )
        )
        self.assertFalse(
            _should_keep_option_mediation_sample(
                np.array([0.50, 0.53, 0.30], dtype=np.float32),
                current_lowest=0.50,
                min_value_gap=0.02,
                min_positive_delta=0.04,
            )
        )
        self.assertTrue(
            _should_keep_option_mediation_sample(
                np.array([0.50, 0.56, 0.30], dtype=np.float32),
                current_lowest=0.50,
                min_value_gap=0.02,
                min_positive_delta=0.04,
            )
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

    def test_predicted_future_option_result_uses_future_need_block(self):
        dataset = _dataset(
            [1, 0],
            values=np.array(
                [
                    [0.1, 0.9, 0.2, 0.0, 0.0],
                    [0.7, 0.1, 0.4, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
        )
        features = np.zeros((2, 5, 12), dtype=np.float32)
        features[:, :, 4:8] = 0.2
        features[0, 1, 4:8] = 0.8
        features[1, 2, 4:8] = 0.9
        dataset = OptionMediationDataset(
            features=mx.array(features, dtype=mx.float32),
            option_values=dataset.option_values,
            target_options=dataset.target_options,
            current_lowest=dataset.current_lowest,
        )

        result = predicted_future_option_result(dataset)

        self.assertEqual(result.model_control, "predicted_future")
        self.assertEqual(result.intervention, "predicted_future")
        self.assertAlmostEqual(result.choice_accuracy, 0.5)
        self.assertAlmostEqual(result.mean_chosen_lowest, 0.65)
        np.testing.assert_array_equal(predicted_future_option_choices(dataset), [1, 2])
        np.testing.assert_allclose(
            predicted_future_option_scores(dataset),
            np.array(
                [
                    [0.2, 0.8, 0.2, 0.2, 0.2],
                    [0.2, 0.2, 0.9, 0.2, 0.2],
                ],
                dtype=np.float32,
            ),
        )
        with self.assertRaises(ValueError):
            predicted_future_option_result(_dataset([0], feature_width=4))

    def test_option_mediation_dataset_with_targets_replaces_only_targets(self):
        dataset = _dataset([0, 1])

        retargeted = option_mediation_dataset_with_targets(
            dataset,
            np.array([2, 3], dtype=np.int32),
        )

        np.testing.assert_array_equal(np.asarray(retargeted.target_options), [2, 3])
        np.testing.assert_array_equal(
            np.asarray(retargeted.features),
            np.asarray(dataset.features),
        )
        with self.assertRaises(ValueError):
            option_mediation_dataset_with_targets(dataset, np.array([1]))

    def test_train_option_mediator_accepts_custom_message_capacity(self):
        dataset = _dataset([0, 1, 2, 3, 4])

        trained = train_option_mediator(
            dataset,
            hidden_size=12,
            receiver_size=12,
            slots=3,
            vocabulary_size=5,
            epochs=1,
            batch_size=5,
            seed=3,
        )
        result = evaluate_option_mediator(
            trained,
            dataset,
            model_control="trained",
            intervention="original",
        )

        self.assertEqual(trained.model.slots, 3)
        self.assertEqual(trained.model.vocabulary_size, 5)
        self.assertGreaterEqual(result.message_codes_used, 1)
        self.assertGreaterEqual(result.message_code_entropy, 0.0)
        self.assertGreater(result.dominant_message_code_fraction, 0.0)
        self.assertGreaterEqual(result.target_code_mutual_information, 0.0)
        self.assertGreaterEqual(result.choice_code_mutual_information, 0.0)
        self.assertGreaterEqual(result.message_patterns_used, 1)
        self.assertGreaterEqual(result.reused_message_pattern_fraction, 0.0)
        self.assertGreaterEqual(result.target_pattern_mutual_information, 0.0)
        self.assertGreaterEqual(result.choice_pattern_mutual_information, 0.0)

    def test_train_option_mediator_accepts_soft_message_training(self):
        dataset = _dataset([0, 1, 2, 3, 4])

        trained = train_option_mediator(
            dataset,
            hidden_size=12,
            receiver_size=12,
            epochs=1,
            batch_size=5,
            message_temperature=0.9,
            soft_message_training=True,
            seed=4,
        )
        result = evaluate_option_mediator(
            trained,
            dataset,
            model_control="trained",
            intervention="original",
        )

        self.assertEqual(result.samples, 5)
        self.assertGreaterEqual(result.message_codes_used, 1)

    def test_train_option_mediator_accepts_score_reconstruction(self):
        dataset = _dataset([0, 1, 2, 3, 4])
        score_targets = np.asarray(dataset.option_values, dtype=np.float32)

        trained = train_option_mediator(
            dataset,
            hidden_size=12,
            receiver_size=12,
            epochs=1,
            batch_size=5,
            score_targets=score_targets,
            score_reconstruction_weight=0.1,
            seed=5,
        )
        result = evaluate_option_mediator(
            trained,
            dataset,
            model_control="trained",
            intervention="original",
        )

        self.assertEqual(result.samples, 5)
        with self.assertRaises(ValueError):
            train_option_mediator(
                dataset,
                epochs=1,
                score_targets=score_targets[:2],
                score_reconstruction_weight=0.1,
            )

    def test_train_option_mediator_accepts_score_pretraining(self):
        dataset = _dataset([0, 1, 2, 3, 4])
        score_targets = np.asarray(dataset.option_values, dtype=np.float32)

        trained = train_option_mediator(
            dataset,
            hidden_size=12,
            receiver_size=12,
            epochs=1,
            batch_size=5,
            score_targets=score_targets,
            score_pretrain_epochs=1,
            seed=6,
        )
        result = evaluate_option_mediator(
            trained,
            dataset,
            model_control="trained",
            intervention="original",
        )

        self.assertEqual(result.samples, 5)
        with self.assertRaises(ValueError):
            train_option_mediator(
                dataset,
                epochs=1,
                score_pretrain_epochs=1,
            )

    def test_train_option_mediator_accepts_score_rank_loss(self):
        dataset = _dataset([0, 1, 2, 3, 4])
        score_targets = np.asarray(dataset.option_values, dtype=np.float32)

        trained = train_option_mediator(
            dataset,
            hidden_size=12,
            receiver_size=12,
            epochs=1,
            batch_size=5,
            score_targets=score_targets,
            score_pretrain_epochs=1,
            score_rank_weight=0.2,
            seed=7,
        )
        result = evaluate_option_mediator(
            trained,
            dataset,
            model_control="trained",
            intervention="original",
        )

        self.assertEqual(result.samples, 5)
        with self.assertRaises(ValueError):
            train_option_mediator(
                dataset,
                epochs=1,
                score_rank_weight=0.2,
            )

    def test_train_option_mediator_accepts_score_rank_code_loss(self):
        dataset = _dataset([0, 1, 2, 3, 4])
        score_targets = np.asarray(dataset.option_values, dtype=np.float32)

        trained = train_option_mediator(
            dataset,
            hidden_size=12,
            receiver_size=12,
            epochs=1,
            batch_size=5,
            score_targets=score_targets,
            score_rank_code_weight=0.1,
            score_rank_code_slot=0,
            seed=17,
        )
        result = evaluate_option_mediator(
            trained,
            dataset,
            model_control="trained",
            intervention="original",
        )

        self.assertEqual(result.samples, 5)
        with self.assertRaises(ValueError):
            train_option_mediator(
                dataset,
                epochs=1,
                score_targets=score_targets,
                score_rank_code_weight=0.1,
                score_rank_code_slot=3,
            )

    def test_train_option_mediator_accepts_score_value_code_loss(self):
        dataset = _dataset([0, 1, 2, 3, 4])
        score_targets = np.asarray(dataset.option_values, dtype=np.float32)

        trained = train_option_mediator(
            dataset,
            hidden_size=12,
            receiver_size=12,
            epochs=1,
            batch_size=5,
            score_targets=score_targets,
            score_value_code_weight=0.1,
            score_value_code_slot=1,
            seed=18,
        )
        result = evaluate_option_mediator(
            trained,
            dataset,
            model_control="trained",
            intervention="original",
        )

        self.assertEqual(result.samples, 5)
        with self.assertRaises(ValueError):
            train_option_mediator(
                dataset,
                epochs=1,
                score_targets=score_targets,
                score_value_code_weight=0.1,
                score_value_code_slot=3,
            )

    def test_train_option_mediator_accepts_score_distillation(self):
        dataset = _dataset([0, 1, 2, 3, 4])
        score_targets = np.asarray(dataset.option_values, dtype=np.float32)

        trained = train_option_mediator(
            dataset,
            hidden_size=12,
            receiver_size=12,
            epochs=1,
            batch_size=5,
            score_targets=score_targets,
            score_distillation_weight=0.2,
            score_distillation_temperature=0.7,
            seed=8,
        )
        result = evaluate_option_mediator(
            trained,
            dataset,
            model_control="trained",
            intervention="original",
        )

        self.assertEqual(result.samples, 5)
        with self.assertRaises(ValueError):
            train_option_mediator(
                dataset,
                epochs=1,
                score_distillation_weight=0.2,
            )

    def test_train_option_mediator_accepts_frozen_receiver_warmup(self):
        dataset = _dataset([0, 1, 2, 3, 4])

        trained = train_option_mediator(
            dataset,
            hidden_size=12,
            receiver_size=12,
            epochs=1,
            batch_size=5,
            frozen_receiver_epochs=1,
            seed=9,
        )
        result = evaluate_option_mediator(
            trained,
            dataset,
            model_control="trained",
            intervention="original",
        )

        self.assertEqual(result.samples, 5)
        self.assertGreaterEqual(result.message_codes_used, 1)

    def test_train_option_mediator_accepts_message_commitment(self):
        dataset = _dataset([0, 1, 2, 3, 4])

        trained = train_option_mediator(
            dataset,
            hidden_size=12,
            receiver_size=12,
            epochs=1,
            batch_size=5,
            message_commitment_weight=0.05,
            seed=10,
        )
        result = evaluate_option_mediator(
            trained,
            dataset,
            model_control="trained",
            intervention="original",
        )

        self.assertEqual(result.samples, 5)
        self.assertGreaterEqual(result.message_codes_used, 1)

    def test_train_option_mediator_accepts_code_target_loss(self):
        dataset = _dataset([0, 1, 2, 3, 4])

        trained = train_option_mediator(
            dataset,
            hidden_size=12,
            receiver_size=12,
            epochs=1,
            batch_size=5,
            code_target_weight=0.1,
            seed=12,
        )
        result = evaluate_option_mediator(
            trained,
            dataset,
            model_control="trained",
            intervention="original",
        )

        self.assertEqual(result.samples, 5)
        self.assertGreaterEqual(result.message_codes_used, 1)
        self.assertGreaterEqual(result.target_code_mutual_information, 0.0)

    def test_train_option_mediator_accepts_message_replay(self):
        dataset = _dataset([0, 1, 2, 3, 4])
        score_targets = np.asarray(dataset.option_values, dtype=np.float32)

        trained = train_option_mediator(
            dataset,
            hidden_size=12,
            receiver_size=12,
            epochs=1,
            batch_size=5,
            score_targets=score_targets,
            score_pretrain_epochs=1,
            score_pretrain_commitment_weight=0.05,
            message_replay_weight=0.1,
            seed=13,
        )
        result = evaluate_option_mediator(
            trained,
            dataset,
            model_control="trained",
            intervention="original",
        )

        self.assertEqual(result.samples, 5)
        self.assertGreaterEqual(result.message_codes_used, 1)

    def test_train_option_mediator_accepts_receiver_copies(self):
        dataset = _dataset([0, 1, 2, 3, 4])

        trained = train_option_mediator(
            dataset,
            hidden_size=12,
            receiver_size=12,
            receiver_copies=3,
            epochs=1,
            batch_size=5,
            seed=16,
        )
        result = evaluate_option_mediator(
            trained,
            dataset,
            model_control="trained",
            intervention="original",
        )

        self.assertEqual(trained.model.receiver_copies, 3)
        self.assertEqual(result.samples, 5)
        self.assertGreaterEqual(result.message_codes_used, 1)

    def test_train_option_mediator_accepts_receiver_turnover(self):
        dataset = _dataset([0, 1, 2, 3, 4])

        trained = train_option_mediator(
            dataset,
            hidden_size=12,
            receiver_size=12,
            receiver_turnover_interval=1,
            epochs=2,
            batch_size=5,
            seed=19,
        )
        result = evaluate_option_mediator(
            trained,
            dataset,
            model_control="trained",
            intervention="original",
        )

        self.assertEqual(result.samples, 5)
        self.assertEqual(trained.model.receiver_copies, 1)
        self.assertGreaterEqual(result.message_codes_used, 1)

    def test_train_option_population_mediator_runs(self):
        dataset = _dataset([0, 1, 2, 3, 4])
        score_targets = np.asarray(dataset.option_values, dtype=np.float32)

        trained = train_option_population_mediator(
            dataset,
            population_size=2,
            hidden_size=12,
            receiver_size=12,
            epochs=1,
            batch_size=5,
            score_targets=score_targets,
            score_pretrain_epochs=1,
            sender_agreement_weight=0.01,
            seed=20,
        )
        result = evaluate_option_population_mediator(
            trained,
            dataset,
            sender_index=1,
            model_control="population_s1",
            intervention="original",
        )

        self.assertEqual(trained.model.population_size, 2)
        self.assertEqual(result.samples, 5)
        self.assertEqual(result.model_control, "population_s1")
        self.assertGreaterEqual(result.message_codes_used, 1)

    def test_train_option_population_receiver_transfer_runs(self):
        dataset = _dataset([0, 1, 2, 3, 4])
        trained = train_option_population_mediator(
            dataset,
            population_size=2,
            hidden_size=12,
            receiver_size=12,
            epochs=1,
            batch_size=5,
            seed=21,
        )

        receiver = train_option_receiver_for_population_sender(
            trained,
            dataset,
            sender_index=0,
            receiver_size=12,
            epochs=1,
            batch_size=5,
            max_samples=3,
            seed=22,
        )
        result = evaluate_option_population_receiver_transfer(
            trained,
            receiver,
            dataset,
            sender_index=0,
            model_control="population_heldout_receiver_s0",
            intervention="original",
        )

        self.assertEqual(result.samples, 5)
        self.assertEqual(result.model_control, "population_heldout_receiver_s0")
        self.assertGreaterEqual(result.message_codes_used, 1)

    def test_option_field_action_receiver_for_population_sender_runs(self):
        values = np.array(
            [
                [0.6, 0.2, 0.4, 0.1, 0.3],
                [0.1, 0.7, 0.2, 0.5, 0.4],
                [0.2, 0.3, 0.8, 0.1, 0.6],
                [0.5, 0.2, 0.1, 0.9, 0.4],
                [0.3, 0.4, 0.1, 0.2, 0.8],
            ],
            dtype=np.float32,
        )
        dataset = _dataset([0, 1, 2, 3, 4], values=values)
        trained = train_option_population_mediator(
            dataset,
            population_size=2,
            hidden_size=12,
            receiver_size=12,
            epochs=1,
            batch_size=5,
            seed=31,
        )
        receiver = train_option_field_action_receiver_for_population_sender(
            trained,
            dataset,
            sender_index=0,
            receiver_size=12,
            epochs=1,
            batch_size=5,
            positive_opportunity_weight=2.0,
            seed=32,
        )
        result = evaluate_option_field_use_for_population_sender(
            trained,
            receiver,
            dataset,
            sender_index=0,
            model_control="field_use",
        )

        self.assertEqual(result.samples, 5)
        self.assertEqual(result.model_control, "field_use")
        self.assertGreaterEqual(result.mean_accuracy, 0.0)
        self.assertLessEqual(result.mean_accuracy, 1.0)
        self.assertGreaterEqual(result.positive_delta_opportunity_rate, 0.0)
        self.assertLessEqual(result.positive_delta_opportunity_rate, 1.0)
        self.assertGreaterEqual(result.positive_delta_opportunity_accuracy, 0.0)
        self.assertLessEqual(result.positive_delta_opportunity_accuracy, 1.0)
        self.assertTrue(np.isfinite(result.positive_query_mean_selected_delta))
        self.assertTrue(
            np.isfinite(result.positive_opportunity_mean_selected_delta)
        )
        self.assertTrue(np.isfinite(result.positive_opportunity_mean_oracle_delta))
        self.assertTrue(np.isfinite(result.mean_selected_delta))

    def test_staged_option_population_from_mediator_runs(self):
        dataset = _dataset([0, 1, 2, 3, 4])
        base = train_option_mediator(
            dataset,
            hidden_size=12,
            receiver_size=12,
            epochs=1,
            batch_size=5,
            seed=23,
        )

        staged = staged_option_population_from_mediator(
            base,
            dataset,
            population_size=2,
            hidden_size=12,
            receiver_size=12,
            epochs=1,
            batch_size=5,
            transfer_receiver_count=1,
            transfer_receiver_epochs=1,
            transfer_receiver_weight=0.1,
            field_receiver_count=1,
            field_receiver_epochs=1,
            field_receiver_weight=0.1,
            field_action_receiver_count=1,
            field_action_receiver_epochs=1,
            field_action_receiver_weight=0.1,
            field_action_positive_opportunity_weight=2.0,
            sender_imitation_weight=0.1,
            receiver_logit_distillation_weight=0.1,
            receiver_logit_distillation_temperature=0.7,
            score_targets=np.array(dataset.option_values),
            score_reconstruction_weight=0.1,
            score_rank_weight=0.1,
            positive_delta_code_weight=0.1,
            positive_delta_code_slot=0,
            relative_value_code_weight=0.1,
            relative_value_code_slot=1,
            seed=24,
        )
        base_result = evaluate_option_population_mediator(
            staged,
            dataset,
            sender_index=0,
            model_control="staged_s0",
            intervention="original",
        )
        new_sender_result = evaluate_option_population_mediator(
            staged,
            dataset,
            sender_index=1,
            model_control="staged_s1",
            intervention="original",
        )

        self.assertEqual(staged.model.population_size, 2)
        self.assertEqual(base_result.samples, 5)
        self.assertEqual(new_sender_result.samples, 5)
        self.assertGreaterEqual(base_result.message_codes_used, 1)
        self.assertGreaterEqual(new_sender_result.message_codes_used, 1)

    def test_train_option_receiver_for_sender_runs(self):
        dataset = _dataset([0, 1, 2, 3, 4])
        score_targets = np.asarray(dataset.option_values, dtype=np.float32)
        trained = train_option_mediator(
            dataset,
            hidden_size=12,
            receiver_size=12,
            epochs=1,
            batch_size=5,
            seed=14,
        )

        receiver = train_option_receiver_for_sender(
            trained,
            dataset,
            receiver_size=12,
            epochs=1,
            batch_size=5,
            max_samples=3,
            score_targets=score_targets,
            score_distillation_weight=0.2,
            score_distillation_temperature=0.7,
            seed=15,
        )
        result = evaluate_option_receiver_transfer(
            trained,
            receiver,
            dataset,
            model_control="heldout_receiver",
            intervention="original",
        )

        self.assertEqual(result.samples, 5)
        self.assertEqual(result.model_control, "heldout_receiver")
        self.assertGreaterEqual(result.message_codes_used, 1)
        with self.assertRaises(ValueError):
            train_option_receiver_for_sender(
                trained,
                dataset,
                receiver_size=12,
                epochs=1,
                batch_size=5,
                score_distillation_weight=0.2,
            )

    def test_train_option_mediator_accepts_late_message_commitment(self):
        dataset = _dataset([0, 1, 2, 3, 4])
        score_targets = np.asarray(dataset.option_values, dtype=np.float32)

        trained = train_option_mediator(
            dataset,
            hidden_size=12,
            receiver_size=12,
            epochs=1,
            batch_size=5,
            score_targets=score_targets,
            score_pretrain_epochs=1,
            score_pretrain_commitment_weight=0.0,
            message_commitment_weight=0.05,
            seed=11,
        )
        result = evaluate_option_mediator(
            trained,
            dataset,
            model_control="trained",
            intervention="original",
        )

        self.assertEqual(result.samples, 5)
        self.assertGreaterEqual(result.message_codes_used, 1)

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
