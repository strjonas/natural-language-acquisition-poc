import math
import unittest

from homesocial.convention_transfer import (
    evaluate_receiver_transfer,
    evaluate_sender_receiver_pair,
    train_receiver_for_sender,
)
from homesocial.emergent_language import (
    collect_communication_dataset,
    train_communication,
)
from homesocial.env import LANGUAGE_NECESSARY_MODE, STOCHASTIC_BODY, Action
from homesocial.observations import MASKED_INTEROCEPTION, observation_vector_size
from homesocial.recurrent_ac import RecurrentActorCritic, RecurrentConfig


class ConventionTransferTests(unittest.TestCase):
    def test_frozen_sender_receiver_transfer_runs(self):
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
        sender = train_communication(
            dataset,
            hidden_size=16,
            receiver_size=12,
            epochs=2,
            batch_size=32,
            seed=1,
        )
        target = train_communication(
            dataset,
            hidden_size=16,
            receiver_size=12,
            epochs=2,
            batch_size=32,
            seed=2,
        )
        receiver = train_receiver_for_sender(
            sender,
            dataset,
            train_samples=12,
            receiver_size=12,
            epochs=2,
            batch_size=16,
            seed=3,
        )
        transfer = evaluate_receiver_transfer(
            sender,
            receiver,
            dataset,
            condition="test_transfer",
            train_samples=12,
        )
        cross = evaluate_sender_receiver_pair(
            sender,
            target,
            dataset,
            condition="test_cross",
        )

        self.assertEqual(transfer.samples, dataset.decisions.shape[0])
        self.assertTrue(math.isfinite(transfer.balanced_accuracy))
        self.assertTrue(math.isfinite(cross.decision_accuracy))


if __name__ == "__main__":
    unittest.main()
