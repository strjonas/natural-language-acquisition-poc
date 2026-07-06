import unittest

import mlx.core as mx
import numpy as np

from homesocial.option_dialogue import (
    OptionDialogueResult,
    _forced_proposals,
    _limited_partner_feature_mask,
    _proposal_table_from_values,
    _repair_loss_weights,
    evaluate_option_dialogue,
    format_result,
    train_option_dialogue,
)
from homesocial.option_mediation import (
    OptionMediationDataset,
    intervene_option_mediation_features,
)


def _dataset() -> OptionMediationDataset:
    features = np.zeros((8, 5, 4), dtype=np.float32)
    targets = np.array([0, 1, 2, 3, 4, 0, 1, 2], dtype=np.int32)
    values = np.full((8, 5), 0.1, dtype=np.float32)
    for row, target in enumerate(targets):
        features[row, :, 0] = np.linspace(-1.0, 1.0, 5)
        features[row, target, :] = 1.0
        values[row, target] = 0.9
    return OptionMediationDataset(
        features=mx.array(features, dtype=mx.float32),
        option_values=mx.array(values, dtype=mx.float32),
        target_options=mx.array(targets, dtype=mx.int32),
        current_lowest=mx.zeros((8,), dtype=mx.float32),
    )


class OptionDialogueTests(unittest.TestCase):
    def test_option_dialogue_train_and_evaluate_runs(self):
        dataset = _dataset()

        trained = train_option_dialogue(
            dataset,
            hidden_size=12,
            receiver_size=12,
            epochs=2,
            batch_size=4,
            repair_weight=0.5,
            repair_proposal_modes=(
                "model_runner_up",
                "limited_partner",
                "second_best",
                "worst",
            ),
            repair_only_mistakes=True,
            repair_min_regret=0.05,
            repair_stage_epochs=1,
            limited_partner_epochs=1,
            limited_partner_hidden_size=8,
            seed=5,
        )
        result = evaluate_option_dialogue(
            trained,
            dataset,
            model_control="dialogue",
            intervention="original",
        )
        intervened = intervene_option_mediation_features(
            dataset,
            intervention="reverse_delta_rank",
        )
        intervened_result = evaluate_option_dialogue(
            trained,
            intervened,
            model_control="dialogue",
            intervention="reverse_delta_rank",
            forced_proposal="worst",
        )
        limited_partner_result = evaluate_option_dialogue(
            trained,
            dataset,
            model_control="dialogue",
            intervention="original",
            forced_proposal="limited_partner",
        )

        self.assertEqual(result.samples, 8)
        self.assertGreaterEqual(result.proposal_accuracy, 0.0)
        self.assertLessEqual(result.final_accuracy, 1.0)
        self.assertGreaterEqual(result.changed_fraction, 0.0)
        self.assertEqual(intervened_result.intervention, "reverse_delta_rank")
        self.assertEqual(limited_partner_result.samples, 8)

    def test_limited_partner_can_drive_main_training_context(self):
        dataset = _dataset()
        trained = train_option_dialogue(
            dataset,
            hidden_size=12,
            receiver_size=12,
            epochs=2,
            batch_size=4,
            proposal_weight=0.0,
            train_proposal_mode="limited_partner",
            limited_partner_epochs=1,
            limited_partner_hidden_size=8,
            seed=8,
        )
        result = evaluate_option_dialogue(
            trained,
            dataset,
            model_control="dialogue",
            intervention="original",
            forced_proposal="limited_partner",
        )

        self.assertEqual(result.samples, 8)
        self.assertGreaterEqual(result.final_accuracy, 0.0)

    def test_forced_proposals_use_dataset_values(self):
        dataset = _dataset()
        logits = mx.array(
            np.tile(np.array([0.0, 3.0, 1.0, 2.0, -1.0], dtype=np.float32), (8, 1)),
            dtype=mx.float32,
        )

        model_runner_up = _forced_proposals(logits, dataset, mode="model_runner_up")
        worst = _forced_proposals(logits, dataset, mode="worst")
        second_best = _forced_proposals(logits, dataset, mode="second_best")

        self.assertTrue(np.all(model_runner_up == 3))
        self.assertEqual(worst.shape, (8,))
        self.assertTrue(np.all(worst != np.asarray(dataset.target_options)))
        self.assertTrue(np.all(second_best != np.asarray(dataset.target_options)))
        table = _proposal_table_from_values(
            dataset.option_values,
            modes=("second_best", "worst"),
        )
        self.assertEqual(table.shape, (8, 2))
        self.assertTrue(np.all(table[:, 0] == second_best))
        self.assertTrue(np.all(table[:, 1] == worst))
        with self.assertRaises(ValueError):
            _forced_proposals(logits, dataset, mode="unknown")

    def test_unknown_repair_mode_fails(self):
        with self.assertRaises(ValueError):
            train_option_dialogue(
                _dataset(),
                hidden_size=12,
                receiver_size=12,
                epochs=1,
                repair_proposal_modes=("unknown",),
            )
        with self.assertRaises(ValueError):
            train_option_dialogue(
                _dataset(),
                hidden_size=12,
                receiver_size=12,
                epochs=1,
                repair_min_regret=-0.1,
            )
        with self.assertRaises(ValueError):
            train_option_dialogue(
                _dataset(),
                hidden_size=12,
                receiver_size=12,
                epochs=1,
                repair_stage_epochs=-1,
            )
        with self.assertRaises(ValueError):
            train_option_dialogue(
                _dataset(),
                hidden_size=12,
                receiver_size=12,
                epochs=1,
                train_proposal_mode="unknown",
            )
        with self.assertRaises(ValueError):
            train_option_dialogue(
                _dataset(),
                hidden_size=12,
                receiver_size=12,
                epochs=1,
                train_proposal_mode="limited_partner",
            )

    def test_repair_loss_weights_filter_mistakes_and_regret(self):
        proposals = mx.array([0, 1, 2, 3], dtype=mx.int32)
        targets = mx.array([0, 2, 2, 1], dtype=mx.int32)
        values = mx.array(
            [
                [0.9, 0.1, 0.0, 0.0],
                [0.1, 0.7, 0.9, 0.0],
                [0.0, 0.1, 0.8, 0.2],
                [0.1, 0.9, 0.0, 0.4],
            ],
            dtype=mx.float32,
        )

        mistake_weights = np.asarray(
            _repair_loss_weights(
                proposals,
                targets,
                values,
                only_mistakes=True,
                min_regret=0.0,
            )
        )
        regret_weights = np.asarray(
            _repair_loss_weights(
                proposals,
                targets,
                values,
                only_mistakes=True,
                min_regret=0.25,
            )
        )

        np.testing.assert_array_equal(
            mistake_weights,
            np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            regret_weights,
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )

    def test_limited_partner_requires_training_and_masks_features(self):
        dataset = _dataset()
        trained = train_option_dialogue(
            dataset,
            hidden_size=12,
            receiver_size=12,
            epochs=1,
            seed=7,
        )
        with self.assertRaises(ValueError):
            evaluate_option_dialogue(
                trained,
                dataset,
                model_control="dialogue",
                intervention="original",
                forced_proposal="limited_partner",
            )
        with self.assertRaises(ValueError):
            train_option_dialogue(
                dataset,
                hidden_size=12,
                receiver_size=12,
                epochs=1,
                repair_proposal_modes=("limited_partner",),
            )

        np.testing.assert_array_equal(
            _limited_partner_feature_mask(4, mode="first_half"),
            np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            _limited_partner_feature_mask(4, mode="even_features"),
            np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
        )
        with self.assertRaises(ValueError):
            _limited_partner_feature_mask(4, mode="unknown")

    def test_format_result_is_stable(self):
        result = OptionDialogueResult(
            model_control="dialogue",
            intervention="original",
            samples=3,
            proposal_accuracy=0.25,
            final_accuracy=0.5,
            changed_fraction=0.75,
            mean_chosen_delta=0.123456,
            mean_oracle_delta=0.2,
            mean_regret=0.076544,
            target_counts="seek_food=1",
        )

        self.assertEqual(
            format_result(result),
            "dialogue,original,3,0.2500,0.5000,0.7500,"
            "0.123456,0.200000,0.076544,seek_food=1",
        )


if __name__ == "__main__":
    unittest.main()
