import tempfile
import unittest
from pathlib import Path

import numpy as np

from homesocial.env import LANGUAGE_NECESSARY_MODE, Action
from homesocial.imitation import (
    collect_expert_episodes,
    evaluate_checkpoint,
    pad_expert_episodes,
    train_bc,
)
from homesocial.observations import MASKED_INTEROCEPTION
from homesocial.recurrent_ac import RecurrentConfig


class ImitationTests(unittest.TestCase):
    def test_expert_dataset_can_hide_exact_interoception(self):
        config = RecurrentConfig(
            condition="grounded",
            include_language_channel=True,
            max_steps=12,
            hidden_size=16,
            randomize_world=False,
            diagnostic_mode=LANGUAGE_NECESSARY_MODE,
            interoception_mode=MASKED_INTEROCEPTION,
        )
        dataset = collect_expert_episodes(config, n=1, seed=1, noise=0.0)
        observations = np.asarray(dataset.episodes[0].observations)

        np.testing.assert_array_equal(observations[:, 6:10], 0.0)
        self.assertGreater(
            float(np.mean(np.asarray(dataset.episodes[0].next_needs))),
            0.0,
        )

    def test_collect_expert_episodes_contains_ask_then_act_data(self):
        config = RecurrentConfig(
            condition="grounded",
            include_language_channel=True,
            max_steps=40,
            hidden_size=16,
            randomize_world=False,
            diagnostic_mode=LANGUAGE_NECESSARY_MODE,
        )

        dataset = collect_expert_episodes(config, n=4, seed=1, noise=0.0)
        actions = np.concatenate(
            [np.asarray(episode.actions) for episode in dataset.episodes]
        )

        self.assertIn(tuple(Action).index(Action.ASK), actions)
        self.assertGreater(dataset.episodes[0].stats.teacher_utterances, 0)

    def test_pad_expert_episodes_adds_masks(self):
        config = RecurrentConfig(
            condition="grounded",
            include_language_channel=True,
            max_steps=20,
            hidden_size=16,
            randomize_world=False,
            diagnostic_mode=LANGUAGE_NECESSARY_MODE,
        )
        dataset = collect_expert_episodes(config, n=3, seed=1, noise=0.0)

        batch = pad_expert_episodes(list(dataset.episodes))

        self.assertEqual(batch.observations.shape[0], 3)
        self.assertEqual(batch.next_observations.shape, batch.observations.shape)
        self.assertEqual(batch.actions.shape, batch.step_masks.shape)

    def test_train_bc_saves_and_evaluates_checkpoint(self):
        config = RecurrentConfig(
            condition="grounded",
            include_language_channel=True,
            max_steps=20,
            hidden_size=16,
            learning_rate=1e-3,
            randomize_world=False,
            diagnostic_mode=LANGUAGE_NECESSARY_MODE,
        )
        dataset = collect_expert_episodes(config, n=4, seed=1, noise=0.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = str(Path(tmpdir) / "bc.weights.npz")
            result = train_bc(
                config,
                dataset,
                checkpoint_path=checkpoint,
                epochs=1,
                batch_size=2,
                seed=1,
            )
            stats = evaluate_checkpoint(checkpoint, "grounded", seeds=[20])

            self.assertTrue(Path(checkpoint).exists())
            self.assertTrue(Path(checkpoint + ".json").exists())
            self.assertGreater(result.final_loss, 0.0)
            self.assertEqual(len(stats), 1)
            self.assertGreater(stats[0].steps, 0)


if __name__ == "__main__":
    unittest.main()
