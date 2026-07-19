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
    SURFACE_INDEX,
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


class SemanticChoiceTrialTest(unittest.TestCase):
    @staticmethod
    def _face(world: IslandWorld, target) -> None:
        ax, ay = target.pos
        if ax > 0:
            world.grid.agent_pos = (ax - 1, ay)
            world.grid.direction = Direction.EAST
        else:
            world.grid.agent_pos = (ax + 1, ay)
            world.grid.direction = Direction.WEST

    def test_reset_builds_only_visible_need_matched_and_poison_pair(self):
        config = IslandConfig(semantic_choice_trial=True)
        expected_quota = Counter(CONSUMABLE_KIND_QUOTA)
        for seed in range(100):
            world = IslandWorld(config, seed=seed)
            packet = world.reset(seed)
            grid = world.grid

            self.assertEqual(
                Counter(
                    grid.kind_by_surface[surface]
                    for surface in CONSUMABLE_SURFACES
                ),
                expected_quota,
            )
            self.assertIn(grid.choice_need, ("food", "water"))
            self.assertAlmostEqual(getattr(grid.needs, grid.choice_need), 0.35)
            other_need = "water" if grid.choice_need == "food" else "food"
            self.assertAlmostEqual(getattr(grid.needs, other_need), 0.75)

            self.assertEqual(len(grid.objects), 2)
            self.assertEqual(
                Counter(obj.kind for obj in grid.objects),
                Counter((grid.choice_need, "poison")),
            )
            self.assertTrue(all(obj.consumable for obj in grid.objects))
            self.assertTrue(
                all(
                    abs(obj.pos[0] - grid.agent_pos[0])
                    + abs(obj.pos[1] - grid.agent_pos[1])
                    == 2
                    for obj in grid.objects
                )
            )
            self.assertEqual(
                {surface_index for _, _, surface_index in packet.visible},
                {SURFACE_INDEX[obj.name] for obj in grid.objects},
            )

    def test_seed_replays_pair_body_pose_and_packet(self):
        config = IslandConfig(semantic_choice_trial=True)
        first = IslandWorld(config, seed=71)
        second = IslandWorld(config, seed=71)
        first_packet = first.reset(71)
        second_packet = second.reset(71)

        self.assertEqual(first_packet, second_packet)
        self.assertEqual(first.grid.kind_by_surface, second.grid.kind_by_surface)
        self.assertEqual(first.grid.choice_need, second.grid.choice_need)
        self.assertEqual(first.grid.direction, second.grid.direction)
        self.assertEqual(first.grid.objects, second.grid.objects)

    def test_need_and_resource_position_are_balanced_over_seeds(self):
        needs = Counter()
        resource_offsets = Counter()
        config = IslandConfig(semantic_choice_trial=True)
        for seed in range(1000):
            world = IslandWorld(config, seed=seed)
            world.reset(seed)
            grid = world.grid
            needs[grid.choice_need] += 1
            resource = next(
                obj for obj in grid.objects if obj.kind == grid.choice_need
            )
            resource_offsets[
                (
                    resource.pos[0] - grid.agent_pos[0],
                    resource.pos[1] - grid.agent_pos[1],
                )
            ] += 1

        self.assertAlmostEqual(needs["food"] / 1000, 0.5, delta=0.04)
        self.assertEqual(
            set(resource_offsets), {(-2, 0), (2, 0), (0, -2), (0, 2)}
        )
        for count in resource_offsets.values():
            self.assertAlmostEqual(count / 1000, 0.25, delta=0.04)

    def test_real_consumption_truncates_and_audits_without_shaping_reward(self):
        config = IslandConfig(semantic_choice_trial=True)

        correct_world = IslandWorld(config, seed=82)
        correct_world.reset(82)
        correct_target = next(
            obj
            for obj in correct_world.grid.objects
            if obj.kind == correct_world.grid.choice_need
        )
        self._face(correct_world, correct_target)
        need = correct_world.grid.choice_need
        before = getattr(correct_world.grid.needs, need)
        _, correct_reward, terminated, truncated, info = correct_world.step(
            Action.CONSUME
        )
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertGreater(getattr(correct_world.grid.needs, need), before)
        self.assertGreater(correct_reward, 0.0)
        self.assertTrue(info["semantic_choice_trial"])
        self.assertTrue(info["correct"])
        self.assertFalse(info["poison"])
        self.assertEqual(info["chosen_kind"], need)
        self.assertEqual(info["chosen_surface"], correct_target.name)
        self.assertFalse(info["chosen_surface_inspected"])
        self.assertFalse(info["timeout"])

        poison_world = IslandWorld(config, seed=83)
        poison_world.reset(83)
        poison_target = next(
            obj for obj in poison_world.grid.objects if obj.kind == "poison"
        )
        self._face(poison_world, poison_target)
        safety_before = poison_world.grid.needs.safety
        _, poison_reward, terminated, truncated, info = poison_world.step(
            Action.CONSUME
        )
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertLess(poison_world.grid.needs.safety, safety_before)
        self.assertLess(poison_reward, 0.0)
        self.assertFalse(info["correct"])
        self.assertTrue(info["poison"])
        self.assertEqual(info["chosen_kind"], "poison")
        self.assertEqual(info["chosen_surface"], poison_target.name)

    def test_ask_and_point_label_interactions_mark_prior_inspection(self):
        for inspect_action in (Action.ASK, Action.POINT):
            with self.subTest(inspect_action=inspect_action):
                world = IslandWorld(
                    IslandConfig(semantic_choice_trial=True), seed=91
                )
                world.reset(91)
                target = next(
                    obj
                    for obj in world.grid.objects
                    if obj.kind == world.grid.choice_need
                )
                self._face(world, target)
                _, _, _, truncated, inspect_info = world.step(inspect_action)
                self.assertFalse(truncated)
                self.assertEqual(
                    Situation.from_key(inspect_info["situation"]).act, "label"
                )
                expected_kind = (
                    "danger" if target.kind == "poison" else target.kind
                )
                self.assertEqual(
                    inspect_info["utterance"], f"this {expected_kind}"
                )

                _, _, _, truncated, choice_info = world.step(Action.CONSUME)
                self.assertTrue(truncated)
                self.assertEqual(choice_info["chosen_surface"], target.name)
                self.assertTrue(choice_info["chosen_surface_inspected"])

    def test_shuffled_choice_labels_match_length_but_not_true_slots(self):
        emitted = Counter()
        matches = 0
        for seed in range(200):
            world = IslandWorld(
                IslandConfig(
                    semantic_choice_trial=True,
                    language_mode="shuffled",
                ),
                seed=seed,
            )
            world.reset(seed)
            target = world.grid.objects[0]
            self._face(world, target)
            _, _, _, _, info = world.step(Action.ASK)
            words = info["utterance"].split()
            self.assertEqual(words[0], "this")
            self.assertEqual(len(words), 2)
            self.assertIn(words[1], {"food", "water", "danger"})
            emitted[words[1]] += 1
            true_kind = "danger" if target.kind == "poison" else target.kind
            matches += int(words[1] == true_kind)
        self.assertAlmostEqual(emitted["danger"] / 200, 0.5, delta=0.1)
        self.assertAlmostEqual(emitted["food"] / 200, 0.25, delta=0.1)
        self.assertAlmostEqual(emitted["water"] / 200, 0.25, delta=0.1)
        self.assertAlmostEqual(matches / 200, 0.375, delta=0.1)

    def test_empty_consume_does_not_end_trial_and_horizon_times_out(self):
        world = IslandWorld(
            IslandConfig(
                semantic_choice_trial=True,
                semantic_choice_horizon=3,
            ),
            seed=101,
        )
        world.reset(101)
        # The initial heading is perpendicular to the pair, so ahead is empty.
        _, _, _, truncated, info = world.step(Action.CONSUME)
        self.assertFalse(truncated)
        self.assertIsNone(info["chosen_surface"])
        self.assertFalse(info["timeout"])

        _, _, _, truncated, info = world.step(Action.WAIT)
        self.assertFalse(truncated)
        self.assertFalse(info["timeout"])
        _, _, _, truncated, info = world.step(Action.WAIT)
        self.assertTrue(truncated)
        self.assertTrue(info["timeout"])
        self.assertFalse(info["correct"])
        self.assertFalse(info["poison"])

    def test_empty_ask_cannot_bypass_object_label_inspection(self):
        for mode in ("grounded", "silent", "shuffled"):
            with self.subTest(mode=mode):
                world = IslandWorld(
                    IslandConfig(
                        semantic_choice_trial=True,
                        language_mode=mode,
                    ),
                    seed=106,
                )
                world.reset(106)
                # Initial heading is perpendicular to the pair by construction.
                self.assertIsNone(world.grid.object_ahead())
                packet, _, terminated, truncated, info = world.step(Action.ASK)
                self.assertFalse(terminated)
                self.assertFalse(truncated)
                self.assertIsNone(info["situation"])
                self.assertIsNone(info["utterance"])
                self.assertEqual(set(packet.tokens), {TOKEN_TO_ID[PAD_TOKEN]})

    def test_language_modes_have_identical_pair_body_motion_and_reward(self):
        worlds = [
            IslandWorld(
                IslandConfig(
                    semantic_choice_trial=True,
                    language_mode=mode,
                    caregiver_offer_threshold=1.0,
                ),
                seed=111,
            )
            for mode in ("grounded", "silent", "shuffled")
        ]
        packets = [world.reset(111) for world in worlds]
        first = worlds[0].grid
        for world, packet in zip(worlds[1:], packets[1:]):
            self.assertEqual(world.grid.kind_by_surface, first.kind_by_surface)
            self.assertEqual(world.grid.objects, first.objects)
            self.assertEqual(world.grid.needs, first.needs)
            self.assertEqual(world.grid.agent_pos, first.agent_pos)
            self.assertEqual(world.grid.direction, first.direction)
            self.assertEqual(packet.vector().tolist(), packets[0].vector().tolist())
            self.assertEqual(packet.vector().shape, packets[0].vector().shape)
            self.assertEqual(len(packet.tokens), len(packets[0].tokens))

        target = first.objects[0]
        for world in worlds:
            matching_target = next(
                obj for obj in world.grid.objects if obj.name == target.name
            )
            self._face(world, matching_target)
        results = [world.step(Action.POINT) for world in worlds]
        rewards = [result[1] for result in results]
        self.assertEqual(rewards, [rewards[0]] * len(rewards))
        self.assertEqual(
            [world.grid.agent_pos for world in worlds],
            [worlds[0].grid.agent_pos] * len(worlds),
        )
        self.assertEqual(
            [world.grid.needs for world in worlds],
            [worlds[0].grid.needs] * len(worlds),
        )
        self.assertTrue(all(len(world.grid.objects) == 2 for world in worlds))
        self.assertTrue(all(world.grid.offered_kind is None for world in worlds))

    def test_horizon_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            IslandConfig(semantic_choice_horizon=0)

    def test_choice_trial_requires_two_visible_slots(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            IslandConfig(
                semantic_choice_trial=True,
                max_visible_slots=1,
            )


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
