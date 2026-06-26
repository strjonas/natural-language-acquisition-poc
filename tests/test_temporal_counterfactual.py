import unittest

import numpy as np

from homesocial.self_state_language import SelfStateDataset
from homesocial.temporal_counterfactual import intervene_delta

import mlx.core as mx


class TemporalCounterfactualTests(unittest.TestCase):
    def test_delta_interventions_modify_only_delta_half(self):
        dataset = SelfStateDataset(
            features=mx.array(
                [
                    [0.1, 0.2, 0.3, 0.4, 0.01, 0.02, 0.03, 0.04],
                    [0.5, 0.6, 0.7, 0.8, -0.01, -0.02, -0.03, -0.04],
                ],
                dtype=mx.float32,
            ),
            needs=mx.zeros((2, 4)),
            low_flags=mx.zeros((2, 4), dtype=mx.int32),
            dominant_labels=mx.zeros((2,), dtype=mx.int32),
            severity_labels=mx.zeros((2,), dtype=mx.int32),
            trend_labels=mx.zeros((2,), dtype=mx.int32),
        )

        zeroed = np.asarray(intervene_delta(dataset, intervention="zero_delta").features)
        negated = np.asarray(intervene_delta(dataset, intervention="negate_delta").features)

        np.testing.assert_allclose(zeroed[:, :4], np.asarray(dataset.features)[:, :4])
        np.testing.assert_allclose(zeroed[:, 4:], 0.0)
        np.testing.assert_allclose(negated[:, :4], np.asarray(dataset.features)[:, :4])
        np.testing.assert_allclose(negated[:, 4:], -np.asarray(dataset.features)[:, 4:])


if __name__ == "__main__":
    unittest.main()
