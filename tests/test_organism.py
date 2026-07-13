import math
import unittest

import numpy as np

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover
    mx = None

if mx is not None:
    from homesocial.island.world import IslandConfig
    from homesocial.organism.model import OrganismModel
    from homesocial.organism.train import (
        OrganismConfig,
        OrganismTrainer,
        compute_gae,
        evaluate_organism,
    )


@unittest.skipIf(mx is None, "MLX is unavailable")
class GaeTest(unittest.TestCase):
    def test_single_step_matches_td_error(self):
        advantages, returns = compute_gae(
            np.array([1.0], dtype=np.float32),
            np.array([0.5], dtype=np.float32),
            0.25,
            np.array([0.0], dtype=np.float32),
            discount=0.9,
            gae_lambda=0.95,
        )
        self.assertAlmostEqual(float(advantages[0]), 1.0 + 0.9 * 0.25 - 0.5, places=5)
        self.assertAlmostEqual(float(returns[0]), float(advantages[0]) + 0.5, places=5)

    def test_done_blocks_bootstrap(self):
        advantages, _ = compute_gae(
            np.array([1.0], dtype=np.float32),
            np.array([0.0], dtype=np.float32),
            10.0,
            np.array([1.0], dtype=np.float32),
            discount=0.9,
            gae_lambda=0.95,
        )
        self.assertAlmostEqual(float(advantages[0]), 1.0, places=5)


@unittest.skipIf(mx is None, "MLX is unavailable")
class ModelTest(unittest.TestCase):
    def _model(self) -> "OrganismModel":
        return OrganismModel(
            vector_size=10,
            vocab_size=12,
            tokens_per_utterance=4,
            action_size=5,
            hidden_size=16,
            token_embed_size=8,
        )

    def test_stepwise_matches_sequence(self):
        model = self._model()
        vectors = mx.random.normal((1, 3, 10))
        tokens = mx.random.randint(0, 12, (1, 3, 4))
        sequence_states, sequence_carry = model.core_states(vectors, tokens)
        hidden = None
        step_states = []
        for t in range(3):
            states, hidden = model.core_states(
                vectors[:, t : t + 1, :], tokens[:, t : t + 1, :], hidden
            )
            step_states.append(states[0, 0])
        mx.eval(sequence_states, sequence_carry, hidden, *step_states)
        for t in range(3):
            difference = float(mx.abs(sequence_states[0, t] - step_states[t]).max())
            self.assertLess(difference, 1e-4)
        carry_difference = float(mx.abs(sequence_carry - hidden).max())
        self.assertLess(carry_difference, 1e-4)

    def test_consequence_head_shapes(self):
        model = self._model()
        states, _ = model.core_states(
            mx.random.normal((1, 3, 10)), mx.random.randint(0, 12, (1, 3, 4))
        )
        actions = mx.array(np.asarray([[0, 2, 4]], dtype=np.int32))
        vectors, needs, rewards, token_logits = model.predict_consequences(
            states, actions
        )
        self.assertEqual(vectors.shape, (1, 3, 10))
        self.assertEqual(needs.shape, (1, 3, 4))
        self.assertEqual(rewards.shape, (1, 3))
        self.assertEqual(token_logits.shape, (1, 3, 4, 12))


@unittest.skipIf(mx is None, "MLX is unavailable")
class TrainingSmokeTest(unittest.TestCase):
    def _config(self, language_mode: str = "grounded") -> "OrganismConfig":
        return OrganismConfig(
            language_mode=language_mode,
            total_steps=192,
            segment_length=32,
            hidden_size=32,
            token_embed_size=8,
            seed=5,
            max_steps=120,
            log_every_lives=1000,
            island=IslandConfig(max_steps=120),
        )

    def test_training_runs_and_losses_are_finite(self):
        trainer = OrganismTrainer(self._config())
        trainer.train()
        self.assertGreater(len(trainer.loss_log), 0)
        for entry in trainer.loss_log:
            self.assertTrue(math.isfinite(entry["loss"]))

    def test_evaluation_reports_stats(self):
        trainer = OrganismTrainer(self._config())
        stats = evaluate_organism(
            trainer.model,
            language_mode="grounded",
            episodes=2,
            base_seed=990,
            max_steps=60,
        )
        self.assertIn("survival_rate", stats)
        self.assertGreaterEqual(stats["mean_steps"], 1.0)

    def test_silent_mode_trains_with_matched_shapes(self):
        trainer = OrganismTrainer(self._config("silent"))
        segment, hidden, bootstrap = trainer.collect_segment()
        self.assertGreater(len(segment), 0)
        loss = trainer.update(segment, hidden, bootstrap)
        self.assertTrue(math.isfinite(loss))

    def test_metabolism_curriculum_anneals_to_adult(self):
        from dataclasses import replace as dc_replace

        config = dc_replace(
            self._config(),
            metabolism_curriculum_start=0.25,
            metabolism_curriculum_steps=1000,
        )
        trainer = OrganismTrainer(config)
        self.assertAlmostEqual(trainer.metabolism_factor(), 0.25, places=5)
        self.assertAlmostEqual(
            trainer.world.grid.water_metabolism, 0.014 * 0.25, places=6
        )
        trainer.global_steps = 500
        self.assertAlmostEqual(trainer.metabolism_factor(), 0.625, places=5)
        trainer.global_steps = 5000
        self.assertAlmostEqual(trainer.metabolism_factor(), 1.0, places=5)
        trainer._finish_life(survived=False)
        self.assertAlmostEqual(trainer.world.grid.water_metabolism, 0.014, places=6)


if __name__ == "__main__":
    unittest.main()
