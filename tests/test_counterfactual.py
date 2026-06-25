import tempfile
import unittest
from pathlib import Path

from homesocial.counterfactual import (
    collect_counterfactual_branches,
    collect_counterfactual_decisions,
    pad_counterfactual_samples,
    train_counterfactual_ranked_consequences,
    train_counterfactual_consequences,
)
from homesocial.env import LANGUAGE_NECESSARY_MODE
from homesocial.imitation import collect_expert_episodes, train_bc
from homesocial.recurrent_ac import RecurrentConfig
from homesocial.self_battery import run_consequence_probe


class CounterfactualTests(unittest.TestCase):
    def test_collect_and_train_counterfactual_branches(self):
        config = RecurrentConfig(
            condition="grounded",
            include_language_channel=True,
            max_steps=20,
            hidden_size=16,
            learning_rate=1e-3,
            randomize_world=False,
            diagnostic_mode=LANGUAGE_NECESSARY_MODE,
        )
        expert = collect_expert_episodes(config, n=3, seed=1, noise=0.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            bc_checkpoint = str(Path(tmpdir) / "bc.weights.npz")
            train_bc(
                config,
                expert,
                checkpoint_path=bc_checkpoint,
                epochs=1,
                batch_size=2,
                seed=1,
            )
            from homesocial.imitation import load_checkpoint

            model, loaded_config = load_checkpoint(bc_checkpoint)
            dataset = collect_counterfactual_branches(
                model,
                loaded_config,
                episodes=2,
                seed=10,
                max_samples=12,
            )
            batch = pad_counterfactual_samples(list(dataset.samples[:4]))
            cf_checkpoint = str(Path(tmpdir) / "cf.weights.npz")
            result = train_counterfactual_consequences(
                model,
                loaded_config,
                dataset,
                checkpoint_path=cf_checkpoint,
                epochs=1,
                batch_size=4,
                learning_rate=1e-3,
                seed=1,
            )
            probe = run_consequence_probe(
                model,
                loaded_config,
                teacher_mode="grounded",
                seeds=[20, 21],
                max_samples=4,
                rollout_policy="teacher",
            )

            self.assertEqual(batch.actions.shape[0], 4)
            self.assertGreater(result.samples, 0)
            self.assertGreater(result.final_loss, 0.0)
            self.assertTrue(Path(cf_checkpoint).exists())
            self.assertGreater(probe.samples, 0)

    def test_train_ranked_counterfactual_decisions(self):
        config = RecurrentConfig(
            condition="grounded",
            include_language_channel=True,
            max_steps=20,
            hidden_size=16,
            learning_rate=1e-3,
            randomize_world=False,
            diagnostic_mode=LANGUAGE_NECESSARY_MODE,
        )
        expert = collect_expert_episodes(config, n=3, seed=1, noise=0.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            bc_checkpoint = str(Path(tmpdir) / "bc.weights.npz")
            train_bc(
                config,
                expert,
                checkpoint_path=bc_checkpoint,
                epochs=1,
                batch_size=2,
                seed=1,
            )
            from homesocial.imitation import load_checkpoint

            model, loaded_config = load_checkpoint(bc_checkpoint)
            dataset = collect_counterfactual_decisions(
                model,
                loaded_config,
                episodes=2,
                seed=10,
                max_decisions=4,
            )
            checkpoint = str(Path(tmpdir) / "ranked.weights.npz")
            result = train_counterfactual_ranked_consequences(
                model,
                loaded_config,
                dataset,
                checkpoint_path=checkpoint,
                epochs=1,
                batch_size=2,
                learning_rate=1e-3,
                seed=1,
                rank_weight=0.5,
            )

            self.assertGreater(len(dataset.decisions), 0)
            self.assertGreater(result.samples, 0)
            self.assertGreater(result.final_loss, 0.0)
            self.assertTrue(Path(checkpoint).exists())


if __name__ == "__main__":
    unittest.main()
