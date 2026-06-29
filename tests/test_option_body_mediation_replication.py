import unittest

from homesocial.option_body_mediation_replication import (
    body_header,
    iter_checkpoint_seed_pairs,
)


class OptionBodyMediationReplicationTests(unittest.TestCase):
    def test_body_header_prefixes_option_world_mediation_header(self):
        self.assertTrue(body_header().startswith("body_checkpoint,seed,"))

    def test_single_seed_broadcasts_to_all_checkpoints(self):
        self.assertEqual(
            iter_checkpoint_seed_pairs(["a", "b"], [7]),
            [("a", 7), ("b", 7)],
        )

    def test_matching_seeds_pair_with_checkpoints(self):
        self.assertEqual(
            iter_checkpoint_seed_pairs(["a", "b"], [7, 8]),
            [("a", 7), ("b", 8)],
        )

    def test_rejects_mismatched_seed_count(self):
        with self.assertRaises(ValueError):
            iter_checkpoint_seed_pairs(["a", "b"], [7, 8, 9])


if __name__ == "__main__":
    unittest.main()
