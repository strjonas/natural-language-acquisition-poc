import unittest
from collections import Counter
from dataclasses import replace

from homesocial.creole.bank import load_default_bank
from homesocial.creole.situations import Situation
from homesocial.creole.vocab import PAD_TOKEN, TOKEN_TO_ID, validate_utterance
from homesocial.env import Action, Direction
from homesocial.island.oracle import OraclePolicy, RandomPolicy
from homesocial.island.world import (
    CONSUMABLE_KIND_QUOTA,
    CONSUMABLE_SURFACES,
    Caregiver,
    IslandConfig,
    IslandGrid,
    IslandWorld,
    ObsPacket,
)


class IslandGridTest(unittest.TestCase):
    def test_kind_assignment_has_exact_quota_and_uniform_surface_marginals(self):
        sample_size = 2000
        expected_quota = Counter(CONSUMABLE_KIND_QUOTA)
        marginals = {surface: Counter() for surface in CONSUMABLE_SURFACES}
        grid = IslandGrid(seed=0)
        for seed in range(sample_size):
            grid.reset(seed)
            kinds = grid.kind_by_surface
            assigned = Counter(kinds[surface] for surface in CONSUMABLE_SURFACES)
            self.assertEqual(assigned, expected_quota)
            for surface in CONSUMABLE_SURFACES:
                marginals[surface][kinds[surface]] += 1
            self.assertEqual(kinds["thorn"], "danger")

        expected_rates = {"food": 0.4, "water": 0.4, "poison": 0.2}
        for surface in CONSUMABLE_SURFACES:
            for kind, expected_rate in expected_rates.items():
                self.assertAlmostEqual(
                    marginals[surface][kind] / sample_size,
                    expected_rate,
                    delta=0.035,
                    msg=f"biased marginal for {surface=} {kind=}",
                )
        for kind in expected_rates:
            counts = [marginals[surface][kind] for surface in CONSUMABLE_SURFACES]
            self.assertLess(max(counts) - min(counts), sample_size * 0.05)

    def test_reset_is_seed_deterministic(self):
        first = IslandGrid(seed=5)
        first.reset(5)
        second = IslandGrid(seed=5)
        second.reset(5)
        self.assertEqual(first.kind_by_surface, second.kind_by_surface)
        self.assertEqual(
            [(obj.name, obj.pos) for obj in first.objects],
            [(obj.name, obj.pos) for obj in second.objects],
        )

    def test_instances_of_same_surface_share_kind(self):
        grid = IslandGrid(seed=11)
        grid.reset(11)
        for obj in grid.objects:
            self.assertEqual(obj.kind, grid.kind_by_surface[obj.name])


class ObsPacketTest(unittest.TestCase):
    def test_vector_shape_matches_declared_size(self):
        world = IslandWorld(seed=3)
        packet = world.reset(3)
        self.assertEqual(len(packet.vector()), ObsPacket.vector_size())

    def test_kind_is_hidden_from_observation(self):
        world = IslandWorld(seed=7)
        world.reset(7)
        observation = world.grid._observe(None, None)
        baseline = world._packet(observation, None).vector()
        world.grid.objects = [
            replace(obj, kind="poison" if obj.kind == "food" else obj.kind)
            for obj in world.grid.objects
        ]
        mutated_observation = world.grid._observe(None, None)
        mutated = world._packet(mutated_observation, None).vector()
        self.assertTrue((baseline == mutated).all())

    def test_language_modes_have_identical_observation_shapes(self):
        shapes = []
        for mode in ("grounded", "silent", "shuffled"):
            world = IslandWorld(IslandConfig(language_mode=mode), seed=17)
            packet = world.reset(17)
            shapes.append((packet.vector().shape, len(packet.tokens)))
        self.assertEqual(len(set(shapes)), 1)


class CaregiverTest(unittest.TestCase):
    def _world_facing(self, surface: str, language_mode: str = "grounded"):
        world = IslandWorld(IslandConfig(language_mode=language_mode), seed=9)
        world.reset(9)
        target = next(obj for obj in world.grid.objects if obj.name == surface)
        x, y = target.pos
        stand = (x - 1, y) if x > 0 else (x + 1, y)
        world.grid.agent_pos = stand
        world.grid.direction = Direction.EAST if x > 0 else Direction.WEST
        return world, target

    def test_point_labels_world_assigned_kind(self):
        world, target = self._world_facing("berry")
        _, _, _, _, info = world.step(Action.POINT)
        situation = Situation.from_key(info["situation"])
        self.assertEqual(situation.act, "label")
        self.assertEqual(situation.slot("surface"), "berry")
        expected = "danger" if target.kind == "poison" else target.kind
        self.assertEqual(situation.slot("kind"), expected)
        self.assertIsNotNone(info["utterance"])
        validate_utterance(info["utterance"])

    def test_ask_into_empty_space_answers_where(self):
        world = IslandWorld(seed=21)
        world.reset(21)
        world.grid.objects = [
            obj for obj in world.grid.objects if obj.pos != world.grid._ahead_pos()
        ]
        _, _, _, _, info = world.step(Action.ASK)
        situation = Situation.from_key(info["situation"])
        self.assertEqual(situation.act, "answer_where")
        self.assertIn(situation.slot("kind"), ("food", "water", "shelter"))

    def test_silent_mode_emits_only_padding(self):
        world, _ = self._world_facing("berry", language_mode="silent")
        packet, _, _, _, info = world.step(Action.POINT)
        self.assertIsNone(info["utterance"])
        self.assertEqual(set(packet.tokens), {TOKEN_TO_ID[PAD_TOKEN]})
        self.assertIsNotNone(info["situation"])

    def test_shuffled_labels_are_per_event_and_true_label_independent(self):
        bank = load_default_bank()
        caregiver = Caregiver(bank, language_mode="shuffled", seed=2)
        food = Situation("label", (("surface", "berry"), ("kind", "food")))
        water = Situation("label", (("surface", "berry"), ("kind", "water")))

        caregiver.reset(83)
        food_sequence = [caregiver.utter_situation(food) for _ in range(64)]
        caregiver.reset(83)
        water_sequence = [caregiver.utter_situation(water) for _ in range(64)]

        # With the same event RNG, changing the true kind cannot change any
        # shuffled draw. Repeated events still receive fresh surface forms.
        self.assertEqual(food_sequence, water_sequence)
        self.assertGreater(len(set(food_sequence)), 8)
        emitted_kind_words = {
            word
            for utterance in food_sequence
            for word in utterance
            if word in {"food", "water", "danger"}
        }
        self.assertEqual(emitted_kind_words, {"food", "water", "danger"})

    def test_shuffled_event_sequence_replays_after_seeded_reset(self):
        world = IslandWorld(IslandConfig(language_mode="shuffled"), seed=19)
        events = (
            Situation("label", (("surface", "water"), ("kind", "food"))),
            Situation("warn", (("place", "there"),)),
            Situation("label", (("surface", "roots"), ("kind", "danger"))),
            Situation("offer", (("kind", "water"),)),
        ) * 6

        world.reset(191)
        first = [world.caregiver.utter_situation(event) for event in events]
        world.reset(191)
        second = [world.caregiver.utter_situation(event) for event in events]
        self.assertEqual(first, second)

    def test_grounded_tokens_reach_observation(self):
        world, _ = self._world_facing("water")
        packet, _, _, _, info = world.step(Action.POINT)
        self.assertIsNotNone(info["utterance"])
        self.assertNotEqual(set(packet.tokens), {TOKEN_TO_ID[PAD_TOKEN]})

    def test_offer_is_visible_actionable_and_spoken(self):
        world = IslandWorld(
            IslandConfig(caregiver_offer_threshold=1.0), seed=41
        )
        world.reset(41)
        packet, _, _, _, info = world.step(Action.WAIT)
        self.assertEqual(Situation.from_key(info["situation"]).act, "offer")
        self.assertIsNotNone(info["utterance"])
        self.assertIsNotNone(world.grid.offered_kind)
        self.assertIn((0, 0), {(dx, dy) for dx, dy, _ in packet.visible})
        offered_kind = world.grid.offered_kind
        before = getattr(world.grid.needs, offered_kind)
        _, _, _, _, consume_info = world.step(Action.CONSUME)
        self.assertEqual(consume_info["event"], f"consumed_{offered_kind}")
        self.assertGreater(getattr(world.grid.needs, offered_kind), before)
        self.assertTrue(consume_info["offered_consumed"])

    def test_silent_offer_has_matched_visible_resource_without_tokens(self):
        world = IslandWorld(
            IslandConfig(
                language_mode="silent", caregiver_offer_threshold=1.0
            ),
            seed=42,
        )
        world.reset(42)
        packet, _, _, _, info = world.step(Action.WAIT)
        self.assertEqual(Situation.from_key(info["situation"]).act, "offer")
        self.assertIsNone(info["utterance"])
        self.assertIn((0, 0), {(dx, dy) for dx, dy, _ in packet.visible})

    def test_distal_offer_requires_approach(self):
        world = IslandWorld(
            IslandConfig(
                caregiver_offer_threshold=1.0,
                caregiver_offer_distance=2,
            ),
            seed=43,
        )
        world.reset(43)
        packet, _, _, _, _ = world.step(Action.WAIT)
        offered = next(
            (dx, dy) for dx, dy, _ in packet.visible if abs(dx) + abs(dy) == 2
        )
        self.assertEqual(abs(offered[0]) + abs(offered[1]), 2)
        _, _, _, _, info = world.step(Action.CONSUME)
        self.assertFalse(info["offered_consumed"])


class OracleTest(unittest.TestCase):
    def test_oracle_survives_full_lives(self):
        for seed in (101, 202, 303):
            world = IslandWorld(IslandConfig(max_steps=400), seed=seed)
            world.reset(seed)
            policy = OraclePolicy()
            terminated = False
            while True:
                _, _, terminated, truncated, _ = world.step(policy.act(world.grid))
                if terminated or truncated:
                    break
            self.assertFalse(terminated, f"oracle died on seed {seed}")

    def test_random_policy_dies_quickly(self):
        deaths = 0
        for seed in (11, 22, 33):
            world = IslandWorld(IslandConfig(max_steps=400), seed=seed)
            world.reset(seed)
            policy = RandomPolicy(seed=seed)
            while True:
                _, _, terminated, truncated, _ = world.step(policy.act(world.grid))
                if terminated or truncated:
                    deaths += int(terminated)
                    break
        self.assertGreaterEqual(deaths, 2)


if __name__ == "__main__":
    unittest.main()
