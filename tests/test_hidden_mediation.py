import math
import unittest

from homesocial.env import LANGUAGE_NECESSARY_MODE, Action
from homesocial.hidden_mediation import (
    evaluate_hidden_mediation,
    fit_need_calibrator,
)
from homesocial.observations import MASKED_INTEROCEPTION, observation_vector_size
from homesocial.recurrent_ac import RecurrentActorCritic, RecurrentConfig


class HiddenMediationTests(unittest.TestCase):
    def test_hidden_mediation_battery_runs(self):
        config = RecurrentConfig(
            condition="grounded",
            include_language_channel=True,
            max_steps=20,
            hidden_size=16,
            randomize_world=False,
            diagnostic_mode=LANGUAGE_NECESSARY_MODE,
            interoception_mode=MASKED_INTEROCEPTION,
        )
        model = RecurrentActorCritic(
            observation_vector_size(
                include_language=True,
                include_object_kinds=False,
            ),
            hidden_size=16,
            action_size=len(Action),
        )
        calibrator = fit_need_calibrator(
            model,
            config,
            episodes=2,
            seed=20,
            max_samples=20,
        )
        result = evaluate_hidden_mediation(
            model,
            config,
            episodes=3,
            seed=1,
            history_mode="full",
            max_samples=10,
            examples=1,
            calibrator=calibrator,
        )

        self.assertGreater(result.samples, 0)
        self.assertTrue(math.isfinite(result.report_accuracy))
        self.assertLessEqual(result.irrelevant_use_rate, 1.0)
        self.assertLessEqual(result.helpful_use_precision, 1.0)


if __name__ == "__main__":
    unittest.main()
