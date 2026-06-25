import math
import unittest

from homesocial.emergent_language import (
    collect_communication_dataset,
    evaluate_communication,
    train_communication,
)
from homesocial.env import LANGUAGE_NECESSARY_MODE, STOCHASTIC_BODY, Action
from homesocial.observations import MASKED_INTEROCEPTION, observation_vector_size
from homesocial.recurrent_ac import RecurrentActorCritic, RecurrentConfig


class EmergentLanguageTests(unittest.TestCase):
    def test_sender_receiver_training_and_evaluation_runs(self):
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
        dataset = collect_communication_dataset(
            base_model,
            config,
            episodes=3,
            seed=1,
            feature_mode="self_estimate",
            max_states=20,
        )
        trained = train_communication(
            dataset,
            hidden_size=16,
            receiver_size=12,
            epochs=2,
            batch_size=32,
            seed=1,
        )
        result = evaluate_communication(
            trained,
            dataset,
            model_control="random",
            feature_mode="self_estimate",
            history_mode="full",
            seed=1,
        )

        self.assertEqual(dataset.features.shape[0] % 4, 0)
        self.assertTrue(math.isfinite(result.decision_accuracy))
        self.assertGreaterEqual(result.message_pairs_used, 1)


if __name__ == "__main__":
    unittest.main()
