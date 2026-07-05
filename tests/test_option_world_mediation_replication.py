import unittest
from argparse import Namespace

import numpy as np

from homesocial.option_mediation import OptionMediationSourceSample
from homesocial.option_world_mediation_replication import (
    OptionWorldMediationReplicationRow,
    _cyclic_sample_slice,
    format_row,
    header,
    pad_option_rank_samples,
    resolved_mediation_balance_targets,
    resolved_option_action_noises,
)


class OptionWorldMediationReplicationTests(unittest.TestCase):
    def test_header_and_row_format_are_stable(self):
        row = OptionWorldMediationReplicationRow(
            seed=7,
            model_control="trained",
            intervention="original",
            samples=11,
            choice_accuracy=0.5,
            mean_regret=0.0123456,
            mean_chosen_delta=-0.1,
            mean_oracle_delta=0.2,
            message_codes_used=3,
            message_code_entropy=0.8,
            dominant_message_code_fraction=0.6,
            target_code_mutual_information=0.03,
            choice_code_mutual_information=0.04,
            positive_delta_code_mutual_information=0.07,
            relative_value_code_mutual_information=0.08,
            message_patterns_used=5,
            reused_message_pattern_fraction=0.7,
            target_pattern_mutual_information=0.05,
            choice_pattern_mutual_information=0.06,
            positive_delta_pattern_mutual_information=0.09,
            relative_value_pattern_mutual_information=0.10,
            world_final_need_mse=0.0345678,
            world_trend_accuracy=0.75,
            trained_world_final_need_mse_before=0.12,
            trained_world_final_need_mse_after=0.03,
            trained_world_trend_before=0.25,
            trained_world_trend_after=0.75,
            field_use_mean_accuracy=0.81,
            field_use_positive_delta_accuracy=0.82,
            field_use_positive_delta_opportunity_rate=0.30,
            field_use_positive_delta_opportunity_accuracy=0.40,
            field_use_relative_value_accuracy=0.80,
            field_use_positive_query_mean_selected_delta=0.05,
            field_use_positive_opportunity_mean_selected_delta=0.06,
            field_use_positive_opportunity_mean_oracle_delta=0.09,
            field_use_mean_selected_delta=0.11,
        )

        self.assertIn("message_code_entropy", header())
        self.assertIn("positive_delta_code_mutual_information", header())
        self.assertIn("field_use_mean_accuracy", header())
        self.assertIn("trained_world_trend_after", header())
        self.assertEqual(
            format_row(row),
            "7,trained,original,11,0.5000,0.012346,-0.100000,"
            "0.200000,3,0.800000,0.600000,0.030000,0.040000,"
            "0.070000,0.080000,5,0.700000,0.050000,0.060000,"
            "0.090000,0.100000,0.034568,0.7500,0.120000,0.030000,"
            "0.2500,0.7500,0.8100,0.8200,0.3000,0.4000,0.8000,"
            "0.050000,0.060000,0.090000,0.110000",
        )

    def test_noise_overrides_fall_back_to_shared_value(self):
        self.assertEqual(
            resolved_option_action_noises(
                Namespace(
                    option_action_noise=0.2,
                    world_option_action_noise=None,
                    mediation_option_action_noise=None,
                )
            ),
            (0.2, 0.2),
        )
        self.assertEqual(
            resolved_option_action_noises(
                Namespace(
                    option_action_noise=0.2,
                    world_option_action_noise=0.3,
                    mediation_option_action_noise=0.1,
                )
            ),
            (0.3, 0.1),
        )

    def test_balance_overrides_fall_back_to_shared_value(self):
        self.assertEqual(
            resolved_mediation_balance_targets(
                Namespace(
                    mediation_balance_target="target_option",
                    mediation_train_balance_target=None,
                    mediation_eval_balance_target=None,
                )
            ),
            ("target_option", "target_option"),
        )
        self.assertEqual(
            resolved_mediation_balance_targets(
                Namespace(
                    mediation_balance_target="target_option",
                    mediation_train_balance_target="target_option_resample",
                    mediation_eval_balance_target="none",
                )
            ),
            ("target_option_resample", "none"),
        )

    def test_pad_option_rank_samples_preserves_grouped_actions_and_targets(self):
        sample = OptionMediationSourceSample(
            history=(
                np.ones((3,), dtype=np.float32),
                np.ones((3,), dtype=np.float32) * 2,
            ),
            option_actions=((1, 2), (3,), (4, 5, 6)),
            option_values=np.array([0.2, 0.1, 0.3], dtype=np.float32),
            target_option=2,
            current_lowest=0.1,
        )

        batch = pad_option_rank_samples([sample])

        self.assertEqual(batch.observations.shape, (1, 2, 3))
        self.assertEqual(batch.actions.shape, (1, 3, 3))
        np.testing.assert_array_equal(np.asarray(batch.actions)[0, 0, :2], [1, 2])
        np.testing.assert_array_equal(np.asarray(batch.action_masks)[0, 1], [1, 0, 0])
        self.assertEqual(int(np.asarray(batch.target_options)[0]), 2)

    def test_cyclic_sample_slice_wraps_replay_batches(self):
        self.assertEqual(_cyclic_sample_slice([1, 2, 3], 2, 5), [3, 1, 2, 3, 1])


if __name__ == "__main__":
    unittest.main()
