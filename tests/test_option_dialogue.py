import unittest

import mlx.core as mx
import numpy as np

from homesocial.option_dialogue import (
    OptionDialogueResult,
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
        )

        self.assertEqual(result.samples, 8)
        self.assertGreaterEqual(result.proposal_accuracy, 0.0)
        self.assertLessEqual(result.final_accuracy, 1.0)
        self.assertGreaterEqual(result.changed_fraction, 0.0)
        self.assertEqual(intervened_result.intervention, "reverse_delta_rank")

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
