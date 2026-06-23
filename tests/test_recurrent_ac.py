import unittest

import mlx.core as mx
import numpy as np

from homesocial.env import HomeostaticSocialGrid
from homesocial.observations import observation_vector, observation_vector_size
from homesocial.recurrent_ac import (
    RecurrentActorCritic,
    RecurrentConfig,
    action_mask,
    choose_action,
    train_condition,
)


class RecurrentActorCriticTests(unittest.TestCase):
    def test_observation_vector_matches_declared_size(self):
        env = HomeostaticSocialGrid(seed=1)
        obs = env.reset()

        vector = observation_vector(
            obs,
            width=env.width,
            height=env.height,
            include_language=True,
            include_object_kinds=False,
        )

        self.assertEqual(
            vector.shape[0],
            observation_vector_size(include_language=True, include_object_kinds=False),
        )

    def test_model_forward_shapes(self):
        input_size = observation_vector_size()
        model = RecurrentActorCritic(input_size, hidden_size=16, action_size=8)
        observations = mx.zeros((5, input_size))

        logits, values = model(observations)

        self.assertEqual(logits.shape, (5, 8))
        self.assertEqual(values.shape, (5,))

    def test_consequence_prediction_shapes(self):
        input_size = observation_vector_size()
        model = RecurrentActorCritic(input_size, hidden_size=16, action_size=8)
        observations = mx.zeros((5, input_size))
        actions = mx.array([0, 1, 2, 3, 4], dtype=mx.int32)

        needs, rewards, utterances = model.predict_consequences(observations, actions)

        self.assertEqual(needs.shape, (5, 4))
        self.assertEqual(rewards.shape, (5,))
        self.assertEqual(utterances.shape[0], 5)

    def test_choose_action_returns_valid_action_index(self):
        input_size = observation_vector_size()
        model = RecurrentActorCritic(input_size, hidden_size=16, action_size=8)
        sequence = np.zeros((3, input_size), dtype=np.float32)

        action = choose_action(
            model,
            sequence,
            np.ones((3, 8), dtype=np.float32),
            rng=np.random.default_rng(1),
            sample=False,
        )

        self.assertGreaterEqual(action, 0)
        self.assertLess(action, 8)

    def test_action_mask_disables_social_actions_without_object_ahead(self):
        env = HomeostaticSocialGrid(seed=1)
        obs = env.reset()
        env.agent_pos = (6, 6)
        obs = env._observe(None, None)

        mask = action_mask(obs)

        self.assertEqual(mask[3], 0.0)
        self.assertEqual(mask[4], 0.0)
        self.assertEqual(mask[5], 0.0)

    def test_tiny_training_run_completes(self):
        result = train_condition(
            RecurrentConfig(
                condition="grounded_teacher",
                episodes=2,
                eval_episodes=1,
                seed=1,
                hidden_size=16,
                max_steps=20,
            )
        )

        self.assertEqual(result.condition, "grounded_teacher")
        self.assertGreater(result.train_stats.steps, 0)
        self.assertGreater(result.eval_stats.steps, 0)


if __name__ == "__main__":
    unittest.main()
