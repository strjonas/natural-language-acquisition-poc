import unittest

import mlx.core as mx
import numpy as np

from homesocial.attribution import attribution_feature_layout
from homesocial.report_calibration import (
    majority_report_baseline,
    shuffled_report_dataset,
)
from homesocial.report_head import (
    ReportDataset,
    project_report_dataset,
    report_feature_indices,
)


class ReportCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.dataset = ReportDataset(
            features=mx.array(np.arange(48, dtype=np.float32).reshape(4, 12)),
            need_labels=mx.array([0, 0, 1, 0], dtype=mx.int32),
            cause_labels=mx.array([1, 1, 0, 1], dtype=mx.int32),
            consequence_labels=mx.array([2, 2, 2, 1], dtype=mx.int32),
        )

    def test_feature_layout_matches_attribution_width(self):
        layout = attribution_feature_layout(16)
        self.assertEqual(layout.size, 44)
        self.assertEqual(layout.hidden, slice(0, 16))
        self.assertEqual(layout.teacher_utterance_present, slice(43, 44))
        self.assertEqual(len(report_feature_indices(16, "hidden")), 16)
        self.assertEqual(len(report_feature_indices(16, "internal")), 21)
        self.assertEqual(len(report_feature_indices(16, "full")), 44)

    def test_majority_baseline_and_projection(self):
        baseline = majority_report_baseline(self.dataset, self.dataset)
        projected = project_report_dataset(self.dataset, [1, 4, 7])

        self.assertEqual(projected.features.shape, (4, 3))
        self.assertAlmostEqual(baseline.need_accuracy, 0.75)
        self.assertAlmostEqual(baseline.cause_accuracy, 0.75)
        self.assertAlmostEqual(baseline.consequence_accuracy, 0.75)

    def test_shuffling_preserves_joint_label_rows(self):
        shuffled = shuffled_report_dataset(self.dataset, seed=7)
        original_rows = sorted(
            zip(
                np.asarray(self.dataset.need_labels),
                np.asarray(self.dataset.cause_labels),
                np.asarray(self.dataset.consequence_labels),
            )
        )
        shuffled_rows = sorted(
            zip(
                np.asarray(shuffled.need_labels),
                np.asarray(shuffled.cause_labels),
                np.asarray(shuffled.consequence_labels),
            )
        )

        self.assertEqual(original_rows, shuffled_rows)
        np.testing.assert_array_equal(shuffled.features, self.dataset.features)


if __name__ == "__main__":
    unittest.main()
