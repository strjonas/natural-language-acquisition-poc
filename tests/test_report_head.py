import math
import unittest

from homesocial.env import LANGUAGE_NECESSARY_MODE
from homesocial.imitation import collect_expert_episodes, load_checkpoint, train_bc
from homesocial.recurrent_ac import RecurrentConfig
from homesocial.report_head import (
    collect_report_dataset,
    render_report,
    train_and_evaluate_report_head,
)


class ReportHeadTests(unittest.TestCase):
    def test_collect_train_and_render_report_head(self):
        config = RecurrentConfig(
            condition="grounded",
            include_language_channel=True,
            max_steps=24,
            hidden_size=16,
            learning_rate=1e-3,
            randomize_world=False,
            diagnostic_mode=LANGUAGE_NECESSARY_MODE,
        )
        expert = collect_expert_episodes(config, n=4, seed=1, noise=0.0)

        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = str(Path(tmpdir) / "bc.weights.npz")
            train_bc(
                config,
                expert,
                checkpoint_path=checkpoint,
                epochs=1,
                batch_size=2,
                seed=1,
            )
            model, loaded_config = load_checkpoint(checkpoint)
            train = collect_report_dataset(
                model,
                loaded_config,
                episodes=4,
                seed=10,
                max_samples=80,
            )
            evaluation = collect_report_dataset(
                model,
                loaded_config,
                episodes=3,
                seed=30,
                max_samples=60,
            )
            result = train_and_evaluate_report_head(
                train,
                evaluation,
                epochs=3,
                batch_size=16,
                hidden_size=16,
                seed=1,
                examples=2,
            )

            self.assertGreater(result.train_samples, 0)
            self.assertGreater(result.eval_samples, 0)
            self.assertTrue(math.isfinite(result.exact_match_accuracy))
            self.assertIn("need=", render_report(0, 0, 0))
            self.assertGreater(len(result.rendered_examples), 0)


if __name__ == "__main__":
    unittest.main()
