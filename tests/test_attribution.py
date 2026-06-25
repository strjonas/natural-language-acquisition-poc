import math
import unittest

import numpy as np

from homesocial.attribution import (
    CAUSE_LABELS,
    collect_attribution_dataset,
    train_and_evaluate_attribution,
)
from homesocial.env import LANGUAGE_NECESSARY_MODE
from homesocial.imitation import collect_expert_episodes, load_checkpoint, train_bc
from homesocial.recurrent_ac import RecurrentConfig


class AttributionTests(unittest.TestCase):
    def test_collect_and_train_attribution_probe(self):
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
            train = collect_attribution_dataset(
                model,
                loaded_config,
                episodes=4,
                seed=10,
                max_samples=80,
            )
            evaluation = collect_attribution_dataset(
                model,
                loaded_config,
                episodes=3,
                seed=30,
                max_samples=60,
            )
            result = train_and_evaluate_attribution(
                train,
                evaluation,
                epochs=3,
                batch_size=16,
                hidden_size=16,
                seed=1,
            )

            labels = set(int(label) for label in np.asarray(train.labels))
            self.assertEqual(labels, set(range(len(CAUSE_LABELS))))
            self.assertGreater(result.train_samples, 0)
            self.assertGreater(result.eval_samples, 0)
            self.assertTrue(math.isfinite(result.eval_accuracy))


if __name__ == "__main__":
    unittest.main()
