import unittest

from homesocial.option_world_mediation_replication import (
    OptionWorldMediationReplicationRow,
    format_row,
    header,
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
            world_final_need_mse=0.0345678,
            world_trend_accuracy=0.75,
            trained_world_final_need_mse_before=0.12,
            trained_world_final_need_mse_after=0.03,
            trained_world_trend_before=0.25,
            trained_world_trend_after=0.75,
        )

        self.assertIn("trained_world_trend_after", header())
        self.assertEqual(
            format_row(row),
            "7,trained,original,11,0.5000,0.012346,-0.100000,"
            "0.200000,0.034568,0.7500,0.120000,0.030000,0.2500,0.7500",
        )


if __name__ == "__main__":
    unittest.main()
