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
    collect_episode,
    pad_trajectories,
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

        next_observations, needs, rewards, utterances = model.predict_consequences(
            observations, actions
        )

        self.assertEqual(next_observations.shape, (5, input_size))
        self.assertEqual(needs.shape, (5, 4))
        self.assertEqual(rewards.shape, (5,))
        self.assertEqual(utterances.shape[0], 5)

    def test_batched_model_forward_shapes(self):
        input_size = observation_vector_size()
        model = RecurrentActorCritic(input_size, hidden_size=16, action_size=8)
        observations = mx.zeros((2, 5, input_size))
        actions = mx.zeros((2, 5), dtype=mx.int32)

        logits, values = model(observations)
        next_observations, needs, rewards, utterances = model.predict_consequences(
            observations, actions
        )

        self.assertEqual(logits.shape, (2, 5, 8))
        self.assertEqual(values.shape, (2, 5))
        self.assertEqual(next_observations.shape, (2, 5, input_size))
        self.assertEqual(needs.shape, (2, 5, 4))
        self.assertEqual(rewards.shape, (2, 5))
        self.assertEqual(utterances.shape[:2], (2, 5))

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

    def test_pad_trajectories_adds_step_mask(self):
        env = HomeostaticSocialGrid(seed=1, max_steps=8)
        input_size = observation_vector_size()
        model = RecurrentActorCritic(input_size, hidden_size=16, action_size=8)
        rng = np.random.default_rng(1)
        first = collect_episode(
            env,
            model,
            seed=1,
            rng=rng,
            include_language=True,
            include_object_kinds=False,
            viability_reward_weight=0.05,
            train=True,
        )[0]
        env.max_steps = 5
        second = collect_episode(
            env,
            model,
            seed=2,
            rng=rng,
            include_language=True,
            include_object_kinds=False,
            viability_reward_weight=0.05,
            train=True,
        )[0]

        batch = pad_trajectories([first, second], discount=0.99, gae_lambda=0.95)

        self.assertEqual(batch.observations.shape[0], 2)
        self.assertEqual(batch.next_observations.shape, batch.observations.shape)
        self.assertEqual(batch.step_masks.shape[0], 2)
        self.assertEqual(batch.advantages.shape, batch.step_masks.shape)
        self.assertEqual(batch.value_targets.shape, batch.step_masks.shape)
        self.assertEqual(batch.old_action_log_probs.shape, batch.step_masks.shape)
        self.assertEqual(
            float(mx.sum(batch.step_masks)),
            len(first.rewards) + len(second.rewards),
        )

    def test_tiny_training_run_completes(self):
        result = train_condition(
            RecurrentConfig(
                condition="grounded_teacher",
                episodes=2,
                eval_episodes=1,
                seed=1,
                hidden_size=16,
                max_steps=20,
                batch_size=2,
            )
        )

        self.assertEqual(result.condition, "grounded_teacher")
        self.assertGreater(result.train_stats.steps, 0)
        self.assertGreater(result.eval_stats.steps, 0)


if __name__ == "__main__":
    unittest.main()
