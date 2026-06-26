import math
import unittest

from homesocial.env import LANGUAGE_NECESSARY_MODE, STOCHASTIC_BODY, Action
from homesocial.observations import MASKED_INTEROCEPTION, observation_vector_size
from homesocial.recurrent_ac import RecurrentActorCritic, RecurrentConfig
from homesocial.self_state_language import (
    collect_self_state_dataset,
    evaluate_self_state_communication,
    train_self_state_communication,
)


class SelfStateLanguageTests(unittest.TestCase):
    def test_self_state_training_and_evaluation_runs(self):
        config = RecurrentConfig(
            condition="grounded",
            include_language_channel=True,
            max_steps=16,
            hidden_size=16,
            randomize_world=False,
            diagnostic_mode=LANGUAGE_NECESSARY_MODE,
            interoception_mode=MASKED_INTEROCEPTION,
            body_dynamics_mode=STOCHASTIC_BODY,
        )
        base_model = RecurrentActorCritic(
            observation_vector_size(
                include_language=True,
                include_object_kinds=False,
                body_dynamics_mode=STOCHASTIC_BODY,
            ),
            hidden_size=16,
            action_size=len(Action),
        )
        dataset = collect_self_state_dataset(
            base_model,
            config,
            episodes=3,
            seed=1,
            feature_mode="self_estimate",
            balance_intents=True,
            max_states=20,
        )
        trained = train_self_state_communication(
            dataset,
            population_size=2,
            hidden_size=16,
            receiver_size=16,
            epochs=2,
            batch_size=16,
            agreement_weight=0.05,
            seed=1,
        )
        result = evaluate_self_state_communication(
            trained,
            dataset,
            model_control="random",
            feature_mode="self_estimate",
            history_mode="full",
        )

        self.assertEqual(result.population_size, 2)
        self.assertEqual(result.samples_per_sender, dataset.features.shape[0])
        self.assertTrue(math.isfinite(result.need_mse))
        self.assertGreaterEqual(result.message_codes_used, 1)


if __name__ == "__main__":
    unittest.main()
