import unittest
from math import nan

import mlx.core as mx
import numpy as np

from homesocial.option_seed_replication import (
    OptionSeedReplicationRow,
    format_row,
    option_majority_trend_accuracy,
)
from homesocial.self_state_language import SelfStateDataset


def _dataset(option_labels: list[int], trend_labels: list[int]) -> SelfStateDataset:
    rows = len(option_labels)
    return SelfStateDataset(
        features=mx.zeros((rows, 12), dtype=mx.float32),
        needs=mx.ones((rows, 4), dtype=mx.float32),
        low_flags=mx.zeros((rows, 4), dtype=mx.int32),
        dominant_labels=mx.zeros((rows,), dtype=mx.int32),
        severity_labels=mx.ones((rows,), dtype=mx.int32),
        trend_labels=mx.array(trend_labels, dtype=mx.int32),
        action_labels=mx.array(option_labels, dtype=mx.int32),
    )


class OptionSeedReplicationTests(unittest.TestCase):
    def test_option_majority_trend_accuracy_uses_train_option_mapping(self):
        train = _dataset(
            option_labels=[0, 0, 0, 1, 1, 1, 2],
            trend_labels=[2, 2, 1, 0, 0, 1, 1],
        )
        eval_dataset = _dataset(
            option_labels=[0, 0, 1, 1, 2, 3],
            trend_labels=[2, 1, 0, 2, 1, 2],
        )

        accuracy = option_majority_trend_accuracy(train, eval_dataset)

        self.assertAlmostEqual(accuracy, 3 / 6)

    def test_format_row_emits_stable_nan_and_precisions(self):
        row = OptionSeedReplicationRow(
            seed=12,
            model_control="option_majority",
            intervention="original",
            samples=9,
            trend_accuracy=1 / 3,
            trend_drop_from_original=0.0,
            need_mse=nan,
            dominant_accuracy=nan,
            severity_accuracy=nan,
            low_flag_accuracy=nan,
            exact_discrete_accuracy=nan,
            world_final_need_mse=0.01234567,
            world_trend_accuracy=0.75,
            trained_world_final_need_mse_before=0.1234567,
            trained_world_final_need_mse_after=0.01234567,
            trained_world_trend_before=0.25,
            trained_world_trend_after=0.75,
        )

        self.assertEqual(
            format_row(row),
            "12,option_majority,original,9,0.3333,0.0000,nan,nan,nan,"
            "nan,nan,0.012346,0.7500,0.123457,0.012346,0.2500,0.7500",
        )


if __name__ == "__main__":
    unittest.main()
