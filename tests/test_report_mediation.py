import math
import unittest

from homesocial.env import LANGUAGE_NECESSARY_MODE, Action, HomeostaticSocialGrid
from homesocial.observations import observation_vector_size
from homesocial.recurrent_ac import RecurrentActorCritic, RecurrentConfig
from homesocial.report_head import NEED_LABELS
from homesocial.report_mediation import (
    collect_mediation_report_dataset,
    evaluate_report_mediation,
    teacher_response_to_report,
)


class ReportMediationTests(unittest.TestCase):
    def test_teacher_response_depends_on_reported_need_and_hidden_kind(self):
        env = HomeostaticSocialGrid(
            seed=1,
            randomize_world=False,
            diagnostic_mode=LANGUAGE_NECESSARY_MODE,
        )
        env.reset(seed=1)
        water = next(obj for obj in env.objects if obj.kind == "water")

        self.assertEqual(
            teacher_response_to_report(NEED_LABELS.index("need_water"), water),
            "drink water",
        )
        self.assertEqual(
            teacher_response_to_report(NEED_LABELS.index("need_food"), water),
            "avoid danger",
        )

    def test_oracle_reporter_runs_in_closed_loop(self):
        config = RecurrentConfig(
            condition="grounded",
            include_language_channel=True,
            max_steps=24,
            hidden_size=16,
            randomize_world=False,
            diagnostic_mode=LANGUAGE_NECESSARY_MODE,
        )
        model = RecurrentActorCritic(
            observation_vector_size(
                include_language=True,
                include_object_kinds=False,
            ),
            hidden_size=16,
            action_size=len(Action),
        )
        dataset = collect_mediation_report_dataset(
            model,
            config,
            trials=12,
            seed=1,
        )
        result = evaluate_report_mediation(
            model,
            config,
            report_mode="oracle",
            reporter=None,
            majority_need=NEED_LABELS.index("stable"),
            trials=12,
            seed=10,
        )

        self.assertEqual(result.report_accuracy, 1.0)
        self.assertEqual(result.decision_accuracy, 1.0)
        self.assertEqual(result.helpful_use_rate, 1.0)
        self.assertEqual(result.irrelevant_use_rate, 0.0)
        self.assertTrue(math.isfinite(result.target_need_delta))
        self.assertEqual(dataset.features.shape[0], 12)


if __name__ == "__main__":
    unittest.main()
