import unittest

import mlx.core as mx
import numpy as np

from homesocial.option_feature_intervention import intervene_option_features
from homesocial.self_state_language import SelfStateDataset


def _dataset(features: np.ndarray) -> SelfStateDataset:
    rows = features.shape[0]
    return SelfStateDataset(
        features=mx.array(features, dtype=mx.float32),
        needs=mx.ones((rows, 4), dtype=mx.float32),
        low_flags=mx.zeros((rows, 4), dtype=mx.int32),
        dominant_labels=mx.zeros((rows,), dtype=mx.int32),
        severity_labels=mx.ones((rows,), dtype=mx.int32),
        trend_labels=mx.array(np.arange(rows) % 3, dtype=mx.int32),
        action_labels=mx.array(np.arange(rows) % 5, dtype=mx.int32),
    )


class OptionFeatureInterventionTests(unittest.TestCase):
    def test_zero_delta_preserves_targets_and_labels(self):
        features = np.arange(36, dtype=np.float32).reshape(3, 12)
        dataset = _dataset(features)

        intervened = intervene_option_features(dataset, intervention="zero_delta")

        intervened_features = np.array(intervened.features)
        np.testing.assert_allclose(intervened_features[:, :8], features[:, :8])
        np.testing.assert_allclose(intervened_features[:, 8:12], 0.0)
        np.testing.assert_array_equal(
            np.array(intervened.trend_labels),
            np.array(dataset.trend_labels),
        )
        np.testing.assert_array_equal(
            np.array(intervened.action_labels),
            np.array(dataset.action_labels),
        )

    def test_shuffle_current_changes_current_block_only(self):
        features = np.arange(48, dtype=np.float32).reshape(4, 12)
        dataset = _dataset(features)

        intervened = intervene_option_features(
            dataset,
            intervention="shuffle_current",
            seed=4,
        )

        intervened_features = np.array(intervened.features)
        np.testing.assert_allclose(intervened_features[:, 4:12], features[:, 4:12])
        self.assertCountEqual(
            [tuple(row) for row in intervened_features[:, :4]],
            [tuple(row) for row in features[:, :4]],
        )

    def test_negate_delta_flips_only_delta_block(self):
        features = np.arange(24, dtype=np.float32).reshape(2, 12)
        dataset = _dataset(features)

        intervened = intervene_option_features(dataset, intervention="negate_delta")

        intervened_features = np.array(intervened.features)
        np.testing.assert_allclose(intervened_features[:, :8], features[:, :8])
        np.testing.assert_allclose(intervened_features[:, 8:12], -features[:, 8:12])

    def test_rejects_non_latent_current_width(self):
        dataset = _dataset(np.zeros((2, 8), dtype=np.float32))

        with self.assertRaises(ValueError):
            intervene_option_features(dataset, intervention="zero_delta")


if __name__ == "__main__":
    unittest.main()
