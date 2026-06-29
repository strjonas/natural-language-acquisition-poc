import math
import unittest

import numpy as np

from homesocial.env import LANGUAGE_NECESSARY_MODE, STOCHASTIC_BODY, Action
from homesocial.observations import MASKED_INTEROCEPTION, observation_vector_size
from homesocial.option_world_model import (
    collect_option_branch_dataset,
    evaluate_option_world_model,
    pad_option_branch_samples,
    train_option_world_model,
)
from homesocial.recurrent_ac import RecurrentActorCritic, RecurrentConfig


class OptionWorldModelTests(unittest.TestCase):
    def test_collect_pad_evaluate_and_train_option_world_model(self):
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
        model = RecurrentActorCritic(
            observation_vector_size(
                include_language=True,
                include_object_kinds=False,
                body_dynamics_mode=STOCHASTIC_BODY,
            ),
            hidden_size=16,
            action_size=len(Action),
        )

        dataset = collect_option_branch_dataset(
            config,
            episodes=2,
            seed=7,
            horizon=3,
            state_policy="mixed",
            max_samples=18,
        )
        self.assertEqual(len(dataset.samples), 18)

        batch = pad_option_branch_samples(list(dataset.samples[:5]))
        self.assertEqual(batch.actions.shape[0], 5)
        self.assertEqual(batch.next_needs.shape[-1], 4)
        self.assertGreaterEqual(int(batch.action_lengths.min()), 1)

        before = evaluate_option_world_model(model, dataset, batch_size=6)
        self.assertTrue(math.isfinite(before.final_need_mse))

        trained = train_option_world_model(
            model,
            config,
            dataset,
            epochs=1,
            batch_size=6,
            learning_rate=1e-3,
            seed=7,
        )
        self.assertEqual(trained.samples, len(dataset.samples))
        self.assertTrue(math.isfinite(trained.final_loss))

        after = evaluate_option_world_model(model, dataset, batch_size=6)
        self.assertTrue(math.isfinite(after.step_need_mse))

    def test_option_action_noise_changes_option_world_branch_actions(self):
        config = RecurrentConfig(
            condition="grounded",
            include_language_channel=True,
            max_steps=12,
            hidden_size=16,
            randomize_world=False,
            diagnostic_mode=LANGUAGE_NECESSARY_MODE,
            interoception_mode=MASKED_INTEROCEPTION,
            body_dynamics_mode=STOCHASTIC_BODY,
        )

        deterministic = collect_option_branch_dataset(
            config,
            episodes=1,
            seed=11,
            horizon=2,
            max_samples=5,
            option_action_noise=0.0,
        )
        noisy = collect_option_branch_dataset(
            config,
            episodes=1,
            seed=11,
            horizon=2,
            max_samples=5,
            option_action_noise=1.0,
        )

        deterministic_actions = tuple(
            int(value) for value in np.asarray(deterministic.samples[0].actions)
        )
        noisy_actions = tuple(
            int(value) for value in np.asarray(noisy.samples[0].actions)
        )
        self.assertNotEqual(deterministic_actions, noisy_actions)
        with self.assertRaises(ValueError):
            collect_option_branch_dataset(
                config,
                episodes=1,
                seed=11,
                option_action_noise=-0.1,
            )


if __name__ == "__main__":
    unittest.main()
