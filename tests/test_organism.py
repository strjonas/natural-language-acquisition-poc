import math
import unittest

import numpy as np

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover
    mx = None

if mx is not None:
    from homesocial.creole.vocab import PAD_TOKEN, TOKEN_TO_ID
    from homesocial.env import Action
    from homesocial.island.world import IslandConfig, SURFACE_INDEX
    from homesocial.organism.model import OrganismModel
    from homesocial.organism.train import (
        ACTIONS,
        OrganismConfig,
        OrganismTrainer,
        audit_binding_consequence_geometry,
        audit_label_referent_binding,
        audit_label_to_self_model,
        audit_observation_branching_planner,
        audit_protocol_return_origin,
        audit_real_vs_explicit_event_transfer,
        audit_terminal_consume_value_calibration,
        audit_cross_round_label_reuse,
        audit_persistent_choice_environment,
        audit_persistent_mapping_information_rent,
        audit_semantic_choice_information_upper_bound,
        audit_self_model_actions,
        DRIFT_ERROR_SCALE,
        DRIFT_REGIME_THRESHOLD,
        EVENT_ERROR_SCALE,
        audit_metabolic_drift_forecast,
        bodily_delta_prediction_loss,
        _bodily_terminal_score,
        _directed_resource_swap_probability_shift,
        _terminal_consume_scores,
        available_action_mask,
        compute_gae,
        consumption_action_mask,
        decode_object_option,
        evaluate_organism,
        evaluate_semantic_choice,
        execute_agent_action,
        _is_voluntary_inspection_event,
        _paired_gradient_geometry,
        _population_gradient_geometry,
        object_option_action_index,
        observation_branching_action_scores,
        observation_branching_inspect_values,
        planned_policy_logits,
        predict_all_action_consequences,
        semantic_choice_label_candidates,
        two_step_action_scores,
    )


@unittest.skipIf(mx is None, "MLX is unavailable")
class GaeTest(unittest.TestCase):
    def test_paired_gradient_geometry_reports_direction_and_scale(self):
        cosine, left_norm, right_norm, ratio = _paired_gradient_geometry(
            {"weight": mx.array([1.0, 0.0])},
            {"weight": mx.array([0.0, 2.0])},
            lexical_only=False,
        )
        self.assertAlmostEqual(cosine, 0.0)
        self.assertAlmostEqual(left_norm, 1.0)
        self.assertAlmostEqual(right_norm, 2.0)
        self.assertAlmostEqual(ratio, 2.0)

    def test_paired_gradient_geometry_can_select_shared_prefixes(self):
        cosine, left_norm, right_norm, ratio = _paired_gradient_geometry(
            {
                "shared.weight": mx.array([1.0, 0.0]),
                "private.weight": mx.array([100.0]),
            },
            {
                "shared.weight": mx.array([0.0, 2.0]),
                "private.weight": mx.array([100.0]),
            },
            lexical_only=False,
            parameter_prefixes=("shared.",),
        )
        self.assertAlmostEqual(cosine, 0.0)
        self.assertAlmostEqual(left_norm, 1.0)
        self.assertAlmostEqual(right_norm, 2.0)
        self.assertAlmostEqual(ratio, 2.0)

    def test_population_gradient_geometry_reports_stable_conflict(self):
        food = np.asarray([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
        water = np.asarray([[-1.0, 0.0], [-2.0, 0.0]], dtype=np.float32)
        result = _population_gradient_geometry(
            food,
            water,
            np.ones(2, dtype=np.float32),
            np.ones(2, dtype=np.float32),
            bootstrap_samples=16,
            seed=7,
        )
        self.assertAlmostEqual(result["aggregate_cosine"], -1.0)
        self.assertEqual(result["bootstrap_negative_fraction"], 1.0)
        self.assertAlmostEqual(
            result["aggregate_water_to_food_sensitivity"], 1.0
        )

    def test_resource_swap_direction_tracks_current_body_need(self):
        self.assertAlmostEqual(
            _directed_resource_swap_probability_shift(
                true_kind="food",
                choice_need="food",
                true_probability=0.8,
                swapped_probability=0.2,
            ),
            0.6,
        )
        self.assertAlmostEqual(
            _directed_resource_swap_probability_shift(
                true_kind="food",
                choice_need="water",
                true_probability=0.2,
                swapped_probability=0.8,
            ),
            0.6,
        )
        self.assertAlmostEqual(
            _directed_resource_swap_probability_shift(
                true_kind="water",
                choice_need="food",
                true_probability=0.75,
                swapped_probability=0.25,
            ),
            -0.5,
        )

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

    def test_duration_discounts_bootstrap_as_semi_markov_step(self):
        advantages, _ = compute_gae(
            np.array([1.0], dtype=np.float32),
            np.array([0.5], dtype=np.float32),
            0.25,
            np.array([0.0], dtype=np.float32),
            np.array([3.0], dtype=np.float32),
            discount=0.9,
            gae_lambda=0.95,
        )
        self.assertAlmostEqual(
            float(advantages[0]), 1.0 + 0.9**3 * 0.25 - 0.5, places=5
        )


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

    def test_zero_scale_planner_preserves_policy_logits(self):
        model = self._model()
        states, _ = model.core_states(
            mx.random.normal((1, 3, 10)), mx.random.randint(0, 12, (1, 3, 4))
        )
        logits = model.policy(states)
        planned = planned_policy_logits(
            model,
            states,
            mx.full((1, 3, 10), 0.5),
            logits,
            mx.zeros((1, 3)),
            reward_weight=0.5,
        )
        mx.eval(logits, planned)
        self.assertLess(float(mx.abs(logits - planned).max()), 1e-7)

    def test_reversed_planner_inverts_self_model_bias(self):
        model = self._model()
        states, _ = model.core_states(
            mx.random.normal((1, 3, 10)), mx.random.randint(0, 12, (1, 3, 4))
        )
        logits = model.policy(states)
        normal = planned_policy_logits(
            model,
            states,
            mx.full((1, 3, 10), 0.5),
            logits,
            mx.ones((1, 3)),
            reward_weight=0.5,
        )
        reversed_planning = planned_policy_logits(
            model,
            states,
            mx.full((1, 3, 10), 0.5),
            logits,
            mx.ones((1, 3)),
            reward_weight=0.5,
            score_sign=-1.0,
        )
        mx.eval(logits, normal, reversed_planning)
        self.assertLess(
            float(mx.abs((normal - logits) + (reversed_planning - logits)).max()),
            1e-6,
        )

    def test_two_step_planner_preserves_shape_and_zero_scale(self):
        model = self._model()
        vectors = mx.full((1, 3, 10), 0.5)
        states, _ = model.core_states(
            vectors, mx.random.randint(0, 12, (1, 3, 4))
        )
        logits = model.policy(states)
        planned = planned_policy_logits(
            model,
            states,
            vectors,
            logits,
            mx.zeros((1, 3)),
            reward_weight=0.5,
            horizon=2,
        )
        mx.eval(logits, planned)
        self.assertEqual(planned.shape, logits.shape)
        self.assertLess(float(mx.abs(logits - planned).max()), 1e-7)

    def test_object_option_consequences_bind_to_selected_slot(self):
        model = OrganismModel(
            vector_size=6,
            vocab_size=12,
            tokens_per_utterance=4,
            action_size=7,
            hidden_size=8,
            token_embed_size=4,
            primitive_action_size=5,
            visible_slots=2,
            object_feature_offset=0,
            object_feature_size=3,
        )
        states = mx.random.normal((1, 1, 8))
        feature_x = [1.0, 0.25, -0.5]
        feature_y = [1.0, -0.75, 0.5]
        first = mx.array([[feature_x + feature_y]])
        swapped = mx.array([[feature_y + feature_x]])
        select_first = model.predict_consequences(
            states, mx.array([[5]]), first
        )
        select_second = model.predict_consequences(
            states, mx.array([[6]]), swapped
        )
        mx.eval(*select_first, *select_second)
        for left, right in zip(select_first, select_second):
            self.assertLess(float(mx.abs(left - right).max()), 1e-7)

    def test_consume_and_inspect_have_distinct_transition_action_identity(self):
        model = OrganismModel(
            vector_size=6,
            vocab_size=12,
            tokens_per_utterance=4,
            action_size=9,
            hidden_size=8,
            token_embed_size=4,
            primitive_action_size=5,
            visible_slots=2,
            object_feature_offset=0,
            object_feature_size=3,
            object_option_types=2,
        )
        vectors = mx.array([[[1.0, 0.25, -0.5, 1.0, -0.75, 0.5]]])
        states = mx.zeros((1, 2, model.state_size))
        action_features, object_features, binding_features = (
            model._transition_features(
                states,
                mx.array([[5, 7]]),
                mx.broadcast_to(vectors, (1, 2, 6)),
            )
        )
        mx.eval(action_features, object_features, binding_features)
        self.assertEqual(action_features.shape, (1, 2, 7))
        self.assertEqual(int(mx.argmax(action_features[0, 0]).item()), 5)
        self.assertEqual(int(mx.argmax(action_features[0, 1]).item()), 6)
        self.assertLess(
            float(mx.abs(object_features[0, 0] - object_features[0, 1]).max()),
            1e-7,
        )


@unittest.skipIf(mx is None, "MLX is unavailable")
class EpisodicBindingMemoryTest(unittest.TestCase):
    def _trainer(self, *, writes: bool = True) -> "OrganismTrainer":
        return OrganismTrainer(
            OrganismConfig(
                total_steps=1,
                segment_length=8,
                hidden_size=16,
                token_embed_size=8,
                semantic_choice_childhood_steps=1,
                consume_options=True,
                inspect_options=True,
                episodic_binding_size=4,
                episodic_binding_writes=writes,
                seed=311,
                log_every_lives=1000,
                island=IslandConfig(
                    semantic_choice_horizon=20,
                    semantic_choice_objects=3,
                ),
            )
        )

    @staticmethod
    def _model_step(model, packet, hidden=None):
        vector = mx.array(packet.vector()[None, None, :])
        tokens = mx.array(np.asarray(packet.tokens, dtype=np.int32)[None, None, :])
        return (*model.core_states(vector, tokens, hidden), vector)

    def _inspect_first_object(self):
        trainer = self._trainer()
        packet = trainer.packet
        model = trainer.model
        _, hidden, initial_vector = self._model_step(model, packet)
        target = trainer.world.grid.objects[0]
        target_slot = next(
            slot
            for slot, (_, _, surface) in enumerate(packet.visible)
            if surface == SURFACE_INDEX[target.name]
        )
        inspect_action = object_option_action_index(
            "inspect",
            target_slot,
            consume_options=True,
            inspect_options=True,
            visible_slots=model.visible_slots,
        )
        post_packet, _, terminated, truncated, info = execute_agent_action(
            trainer.world,
            packet,
            inspect_action,
            consume_options=True,
            inspect_options=True,
        )
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(str(info["situation"]).startswith("label|"))
        post_states, carry, post_vector = self._model_step(
            model, post_packet, hidden
        )
        return (
            trainer,
            packet,
            initial_vector,
            hidden,
            target,
            post_packet,
            post_vector,
            post_states,
            carry,
        )

    def test_label_writes_only_attended_surface_and_persists_through_padding(self):
        from dataclasses import replace as dc_replace

        (
            trainer,
            _,
            _,
            hidden,
            target,
            post_packet,
            post_vector,
            _,
            carry,
        ) = self._inspect_first_object()
        model = trainer.model
        _, memory, valid = model.split_carry(carry, post_vector)
        mx.eval(memory, valid)
        target_index = SURFACE_INDEX[target.name]
        self.assertEqual(float(valid.sum().item()), 1.0)
        self.assertEqual(float(valid[0, target_index].item()), 1.0)
        self.assertGreater(float(mx.abs(memory[0, target_index]).max().item()), 0.0)

        padding = (TOKEN_TO_ID[PAD_TOKEN],) * model.tokens_per_utterance
        padded_packet = dc_replace(
            post_packet,
            tokens=padding,
            last_action_index=ACTIONS.index(Action.WAIT),
        )
        _, padded_carry, padded_vector = self._model_step(
            model, padded_packet, carry
        )
        _, padded_memory, padded_valid = model.split_carry(
            padded_carry, padded_vector
        )
        mx.eval(padded_memory, padded_valid)
        np.testing.assert_allclose(
            np.asarray(padded_memory), np.asarray(memory), atol=1e-7
        )
        np.testing.assert_allclose(
            np.asarray(padded_valid), np.asarray(valid), atol=1e-7
        )

        reordered_packet = dc_replace(
            padded_packet,
            visible=tuple(reversed(padded_packet.visible)),
        )
        reordered_vector = mx.array(reordered_packet.vector()[None, :])
        original_reads = model._binding_reads(
            padded_vector[:, 0, :], padded_memory, padded_valid
        ).reshape(1, model.visible_slots, model.episodic_binding_size + 1)
        reordered_reads = model._binding_reads(
            reordered_vector, padded_memory, padded_valid
        ).reshape(1, model.visible_slots, model.episodic_binding_size + 1)
        original_target_slot = next(
            slot
            for slot, (_, _, surface) in enumerate(padded_packet.visible)
            if surface == target_index
        )
        reordered_target_slot = next(
            slot
            for slot, (_, _, surface) in enumerate(reordered_packet.visible)
            if surface == target_index
        )
        mx.eval(original_reads, reordered_reads)
        np.testing.assert_allclose(
            np.asarray(original_reads[0, original_target_slot]),
            np.asarray(reordered_reads[0, reordered_target_slot]),
            atol=1e-7,
        )

        nonreferential = dc_replace(
            post_packet,
            last_action_index=ACTIONS.index(Action.MOVE_FORWARD),
        )
        _, no_write_carry, no_write_vector = self._model_step(
            model, nonreferential, hidden
        )
        _, _, no_write_valid = model.split_carry(
            no_write_carry, no_write_vector
        )
        mx.eval(no_write_valid)
        self.assertEqual(float(no_write_valid.sum().item()), 0.0)

    def test_hypothetical_observation_writes_only_selected_visible_surface(self):
        trainer = self._trainer()
        model = trainer.model
        vector = mx.array(trainer.packet.vector()[None, None, :])
        tokens = mx.array(
            np.asarray(trainer.packet.tokens, dtype=np.int32)[None, None, :]
        )
        states, _ = model.core_states(vector, tokens)
        slots = model._visible_object_features(vector)
        selected_surface = slots[..., 1, 3:]
        label_tokens = mx.broadcast_to(
            semantic_choice_label_candidates()[0][None, None, :],
            (1, 1, model.tokens_per_utterance),
        )

        observed = model.observe_from_state(
            states,
            vector,
            label_tokens,
            binding_surfaces=selected_surface,
        )
        _, memory, valid = model._state_memory_parts(observed)
        selected_index = int(mx.argmax(selected_surface[0, 0]).item())
        mx.eval(memory, valid)
        self.assertEqual(float(valid.sum().item()), 1.0)
        self.assertEqual(float(valid[0, 0, selected_index].item()), 1.0)
        self.assertGreater(
            float(mx.abs(memory[0, 0, selected_index]).max().item()),
            0.0,
        )

        model.episodic_binding_writes = False
        suppressed = model.observe_from_state(
            states,
            vector,
            label_tokens,
            binding_surfaces=selected_surface,
        )
        _, _, suppressed_valid = model._state_memory_parts(suppressed)
        mx.eval(suppressed_valid)
        self.assertEqual(float(suppressed_valid.sum().item()), 0.0)

    def test_protocol_branch_writes_memory_without_touching_the_core(self):
        trainer = self._trainer()
        model = trainer.model
        packet = trainer.packet
        vector = mx.array(packet.vector()[None, None, :])
        tokens = mx.array(
            np.asarray(packet.tokens, dtype=np.int32)[None, None, :]
        )
        states, _ = model.core_states(vector, tokens)
        surface = mx.array(
            np.eye(len(SURFACE_INDEX), dtype=np.float32)[
                packet.visible[0][2]
            ][None, None, :]
        )
        label_tokens = mx.broadcast_to(
            semantic_choice_label_candidates()[0][None, None, :],
            (1, 1, model.tokens_per_utterance),
        )
        written = model.write_binding_into_state(states, surface, label_tokens)
        core_before, memory_before, valid_before = model._state_memory_parts(
            states
        )
        core_after, memory_after, valid_after = model._state_memory_parts(
            written
        )
        mx.eval(core_before, core_after, memory_after, valid_after)
        # The recurrent core is untouched: no sensory scene was fabricated.
        np.testing.assert_allclose(
            np.asarray(core_before), np.asarray(core_after), atol=0.0
        )
        # The addressed surface row is now valid and nonzero.
        target = packet.visible[0][2]
        self.assertGreater(float(np.asarray(valid_after[0, 0])[target]), 0.5)
        self.assertEqual(float(np.asarray(valid_before[0, 0])[target]), 0.0)
        self.assertGreater(
            float(np.abs(np.asarray(memory_after[0, 0, target])).max()), 0.0
        )
        # Every other surface row is untouched.
        others = [s for s in range(model.surface_count) if s != target]
        np.testing.assert_allclose(
            np.asarray(memory_after[0, 0])[others],
            np.asarray(memory_before[0, 0])[others],
            atol=0.0,
        )

    def test_protocol_branch_changes_inspect_values(self):
        trainer = self._trainer()
        model = trainer.model
        vector = mx.array(trainer.packet.vector()[None, None, :])
        tokens = mx.array(
            np.asarray(trainer.packet.tokens, dtype=np.int32)[None, None, :]
        )
        states, _ = model.core_states(vector, tokens)
        reconstructive, _, _, _ = observation_branching_inspect_values(
            model, states, vector, persistent_information_reuses=2
        )
        protocol, _, probabilities, choices = (
            observation_branching_inspect_values(
                model,
                states,
                vector,
                persistent_information_reuses=2,
                protocol_branch=True,
            )
        )
        chained, _, _, _ = observation_branching_inspect_values(
            model,
            states,
            vector,
            persistent_information_reuses=2,
            protocol_branch=True,
            protocol_return_from_inspect_state=True,
        )
        mx.eval(reconstructive, protocol, chained, probabilities, choices)
        self.assertEqual(protocol.shape, reconstructive.shape)
        self.assertEqual(chained.shape, protocol.shape)
        self.assertTrue(np.all(np.isfinite(np.asarray(protocol))))
        self.assertFalse(
            np.allclose(
                np.asarray(protocol), np.asarray(reconstructive), atol=1e-6
            )
        )
        self.assertFalse(
            np.allclose(
                np.asarray(chained), np.asarray(protocol), atol=1e-6
            )
        )

    def test_inspect_state_return_origin_requires_protocol_branch(self):
        trainer = self._trainer()
        model = trainer.model
        vector = mx.array(trainer.packet.vector()[None, None, :])
        tokens = mx.array(
            np.asarray(trainer.packet.tokens, dtype=np.int32)[None, None, :]
        )
        states, _ = model.core_states(vector, tokens)
        with self.assertRaises(ValueError):
            observation_branching_inspect_values(
                model,
                states,
                vector,
                protocol_return_from_inspect_state=True,
            )

    def test_urgent_deficit_utility_reads_the_lowest_observed_need(self):
        needs = mx.array([[[[0.9, 0.4, 0.7, 0.8], [0.9, 0.4, 0.7, 0.8]]]])
        # The second candidate is better on its worst dimension but worse on
        # the dimension the body currently needs most.
        terminal = mx.array([[[[0.9, 0.95, 0.2, 0.8], [0.9, 0.45, 0.6, 0.8]]]])
        legacy = _bodily_terminal_score(
            terminal, needs, urgent_deficit_utility=False
        )
        urgent = _bodily_terminal_score(
            terminal, needs, urgent_deficit_utility=True
        )
        mx.eval(legacy, urgent)
        np.testing.assert_allclose(
            np.asarray(legacy[0, 0]), [0.2, 0.45], atol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(urgent[0, 0]), [0.95, 0.45], atol=1e-6
        )

    def test_urgent_deficit_utility_changes_terminal_consume_ranking(self):
        trainer = self._trainer()
        model = trainer.model
        vector = mx.array(trainer.packet.vector()[None, None, :])
        tokens = mx.array(
            np.asarray(trainer.packet.tokens, dtype=np.int32)[None, None, :]
        )
        states, _ = model.core_states(vector, tokens)
        legacy = _terminal_consume_scores(model, states, vector)
        urgent = _terminal_consume_scores(
            model, states, vector, urgent_deficit_utility=True
        )
        mx.eval(legacy, urgent)
        legacy = np.asarray(legacy[0, 0], dtype=np.float64)
        urgent = np.asarray(urgent[0, 0], dtype=np.float64)
        # Absent slots stay masked out under both rules.
        np.testing.assert_array_equal(legacy <= -1e8, urgent <= -1e8)
        # The urgent rule can only equal or exceed the minimum rule, since the
        # minimum is a lower bound on any single need.
        visible = legacy > -1e8
        self.assertTrue(np.all(urgent[visible] >= legacy[visible] - 1e-6))

    def test_persistent_self_query_backs_up_future_bodily_contexts(self):
        trainer = self._trainer()
        model = trainer.model
        vector = mx.array(trainer.packet.vector()[None, None, :])
        tokens = mx.array(
            np.asarray(trainer.packet.tokens, dtype=np.int32)[None, None, :]
        )
        states, _ = model.core_states(vector, tokens)

        one_shot, _, _, _ = observation_branching_inspect_values(
            model, states, vector
        )
        reuse_one, _, _, _ = observation_branching_inspect_values(
            model,
            states,
            vector,
            persistent_information_reuses=1,
        )
        reuse_two, _, _, _ = observation_branching_inspect_values(
            model,
            states,
            vector,
            persistent_information_reuses=2,
        )
        mx.eval(one_shot, reuse_one, reuse_two)
        one_shot = np.asarray(one_shot[0, 0], dtype=np.float64)
        reuse_one = np.asarray(reuse_one[0, 0], dtype=np.float64)
        reuse_two = np.asarray(reuse_two[0, 0], dtype=np.float64)

        # Reuse adds one identical future-context term per count, so the
        # increments are linear in the reuse count.
        np.testing.assert_allclose(
            reuse_two - reuse_one,
            reuse_one - one_shot,
            atol=1e-5,
        )
        self.assertTrue(np.all(np.isfinite(reuse_two)))

    def test_persistent_self_query_reverts_known_surfaces_to_costly_rollout(self):
        trainer = self._trainer()
        model = trainer.model
        packet = trainer.packet
        vector = mx.array(packet.vector()[None, None, :])
        tokens = mx.array(
            np.asarray(packet.tokens, dtype=np.int32)[None, None, :]
        )
        states, _ = model.core_states(vector, tokens)

        # Write a label for the first visible surface, so its lexical row is
        # already valid and re-inspection cannot add a new fact.
        label_tokens = semantic_choice_label_candidates()[0]
        label_tokens = mx.broadcast_to(
            label_tokens[None, None, :],
            (1, 1, model.tokens_per_utterance),
        )
        surface = mx.array(
            np.eye(len(SURFACE_INDEX), dtype=np.float32)[
                packet.visible[0][2]
            ][None, None, :]
        )
        known_states = model.observe_from_state(
            states,
            vector,
            label_tokens,
            binding_surfaces=surface,
        )
        _, _, valid = model._state_memory_parts(known_states)
        mx.eval(valid)
        self.assertGreater(
            float(np.asarray(valid[0, 0])[packet.visible[0][2]]),
            0.5,
        )

        scores = observation_branching_action_scores(
            model,
            known_states,
            vector,
            persistent_information_reuses=7,
        )
        ordinary = two_step_action_scores(
            model,
            known_states,
            vector,
            reward_weight=0.0,
            force_return_after_inspect=True,
        )
        mx.eval(scores, ordinary)
        inspect_start = model.primitive_action_size + model.visible_slots
        known_slot = 0
        self.assertAlmostEqual(
            float(np.asarray(scores[0, 0, inspect_start + known_slot])),
            float(np.asarray(ordinary[0, 0, inspect_start + known_slot])),
            places=5,
        )

    def test_observation_branching_has_normalized_three_way_traces(self):
        trainer = self._trainer()
        model = trainer.model
        vector = mx.array(trainer.packet.vector()[None, None, :])
        tokens = mx.array(
            np.asarray(trainer.packet.tokens, dtype=np.int32)[None, None, :]
        )
        states, _ = model.core_states(vector, tokens)
        values, probabilities, branch_values, choices = (
            observation_branching_inspect_values(
                model,
                states,
                vector,
            )
        )
        scores = observation_branching_action_scores(
            model,
            states,
            vector,
        )
        collapsed = observation_branching_inspect_values(
            model,
            states,
            vector,
            collapsed_labels=True,
        )
        mx.eval(
            values,
            probabilities,
            branch_values,
            choices,
            scores,
            *collapsed,
        )
        self.assertEqual(values.shape, (1, 1, model.visible_slots))
        self.assertEqual(
            probabilities.shape,
            (1, 1, model.visible_slots, 3),
        )
        self.assertEqual(branch_values.shape, probabilities.shape)
        self.assertEqual(choices.shape, probabilities.shape)
        self.assertEqual(scores.shape, (1, 1, model.action_size))
        np.testing.assert_allclose(
            np.asarray(probabilities.sum(axis=-1)),
            1.0,
            atol=1e-6,
        )
        collapsed_probabilities = np.asarray(collapsed[1])
        collapsed_values = np.asarray(collapsed[2])
        collapsed_choices = np.asarray(collapsed[3])
        np.testing.assert_allclose(
            collapsed_probabilities,
            1.0 / 3.0,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            collapsed_values[..., 0],
            collapsed_values[..., 1],
            atol=1e-6,
        )
        np.testing.assert_array_equal(
            collapsed_choices[..., 0],
            collapsed_choices[..., 2],
        )

    def test_binding_value_is_token_only_and_transition_read_is_object_local(self):
        from dataclasses import replace as dc_replace

        (
            trainer,
            _,
            _,
            hidden,
            target,
            post_packet,
            post_vector,
            post_states,
            carry,
        ) = self._inspect_first_object()
        model = trainer.model
        swapped_needs = (
            post_packet.needs[1],
            post_packet.needs[0],
            post_packet.needs[2],
            post_packet.needs[3],
        )
        swapped_packet = dc_replace(post_packet, needs=swapped_needs)
        _, swapped_carry, swapped_vector = self._model_step(
            model, swapped_packet, hidden
        )
        _, memory, _ = model.split_carry(carry, post_vector)
        _, swapped_memory, _ = model.split_carry(swapped_carry, swapped_vector)
        mx.eval(memory, swapped_memory)
        np.testing.assert_allclose(
            np.asarray(memory), np.asarray(swapped_memory), atol=1e-7
        )

        target_surface = SURFACE_INDEX[target.name]
        target_slot = next(
            slot
            for slot, (_, _, surface) in enumerate(post_packet.visible)
            if surface == target_surface
        )
        other_slot = next(
            slot
            for slot, (_, _, surface) in enumerate(post_packet.visible)
            if surface != target_surface
        )
        target_action = object_option_action_index(
            "consume",
            target_slot,
            consume_options=True,
            inspect_options=True,
            visible_slots=model.visible_slots,
        )
        other_action = object_option_action_index(
            "consume",
            other_slot,
            consume_options=True,
            inspect_options=True,
            visible_slots=model.visible_slots,
        )
        actions = mx.array([[target_action, other_action]])
        states = mx.broadcast_to(post_states, (1, 2, model.state_size))
        vectors = mx.broadcast_to(post_vector, (1, 2, model.vector_size))
        _, _, binding_features = model._transition_features(
            states, actions, vectors
        )
        mx.eval(binding_features)
        self.assertEqual(float(binding_features[0, 0, -1].item()), 1.0)
        self.assertEqual(float(binding_features[0, 1, -1].item()), 0.0)
        self.assertGreater(
            float(mx.abs(binding_features[0, 0, :-1]).max().item()), 0.0
        )
        self.assertEqual(
            float(mx.abs(binding_features[0, 1, :-1]).max().item()), 0.0
        )

        # The consequence transition may use the selected object's lexical
        # row, but no unrelated row from the external bank.  This guards
        # against an accidental global-memory bypass of referent locality.
        altered = np.asarray(post_states).copy()
        memory_start = model.hidden_size
        memory_stop = memory_start + model.binding_bank_size
        altered_memory = altered[..., memory_start:memory_stop].reshape(
            1,
            1,
            model.surface_count,
            model.episodic_binding_size,
        )
        other_surface = post_packet.visible[other_slot][2]
        altered_memory[..., other_surface, :] += 7.0
        altered[..., memory_stop + other_surface] = 1.0
        base_outputs = model.predict_consequences(
            post_states,
            mx.array([[target_action]]),
            post_vector,
        )
        altered_outputs = model.predict_consequences(
            mx.array(altered),
            mx.array([[target_action]]),
            post_vector,
        )
        mx.eval(*base_outputs, *altered_outputs)
        for base, changed in zip(base_outputs, altered_outputs, strict=True):
            np.testing.assert_allclose(
                np.asarray(base), np.asarray(changed), atol=0.0, rtol=0.0
            )

    def test_whole_sequence_and_stepwise_memory_execution_match_and_reset(self):
        from dataclasses import replace as dc_replace

        (
            trainer,
            initial_packet,
            _,
            _,
            _,
            post_packet,
            _,
            _,
            carry,
        ) = self._inspect_first_object()
        model = trainer.model
        padding = (TOKEN_TO_ID[PAD_TOKEN],) * model.tokens_per_utterance
        padded_packet = dc_replace(
            post_packet,
            tokens=padding,
            last_action_index=ACTIONS.index(Action.WAIT),
        )
        packets = (initial_packet, post_packet, padded_packet)
        vectors = mx.array(np.stack([packet.vector() for packet in packets])[None])
        tokens = mx.array(
            np.asarray([packet.tokens for packet in packets], dtype=np.int32)[None]
        )
        whole_states, whole_carry = model.core_states(vectors, tokens)
        step_hidden = None
        step_states = []
        for packet in packets:
            states, step_hidden, _ = self._model_step(
                model, packet, step_hidden
            )
            step_states.append(states)
        concatenated = mx.concatenate(step_states, axis=1)
        mx.eval(whole_states, whole_carry, concatenated, step_hidden)
        np.testing.assert_allclose(
            np.asarray(whole_states), np.asarray(concatenated), atol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(whole_carry), np.asarray(step_hidden), atol=1e-6
        )

        trainer.hidden = carry
        trainer._finish_life(survived=True)
        self.assertIsNone(trainer.hidden)
        reset_vector = mx.array(trainer.packet.vector()[None, None, :])
        _, _, reset_valid = model.split_carry(None, reset_vector)
        mx.eval(reset_valid)
        self.assertEqual(float(reset_valid.sum().item()), 0.0)

    def test_write_disabled_control_keeps_matched_bank_empty(self):
        trainer = self._trainer(writes=False)
        packet = trainer.packet
        model = trainer.model
        _, hidden, _ = self._model_step(model, packet)
        target = trainer.world.grid.objects[0]
        slot = next(
            slot
            for slot, (_, _, surface) in enumerate(packet.visible)
            if surface == SURFACE_INDEX[target.name]
        )
        action = object_option_action_index(
            "inspect",
            slot,
            consume_options=True,
            inspect_options=True,
            visible_slots=model.visible_slots,
        )
        post_packet, _, _, _, _ = execute_agent_action(
            trainer.world,
            packet,
            action,
            consume_options=True,
            inspect_options=True,
        )
        _, carry, vector = self._model_step(model, post_packet, hidden)
        _, memory, valid = model.split_carry(carry, vector)
        mx.eval(memory, valid)
        self.assertEqual(float(mx.abs(memory).max().item()), 0.0)
        self.assertEqual(float(valid.sum().item()), 0.0)

    def test_write_ablation_has_identical_parameters_and_initialization(self):
        from mlx.utils import tree_flatten

        enabled = self._trainer(writes=True)
        disabled = self._trainer(writes=False)
        enabled_parameters = tree_flatten(enabled.model.parameters())
        disabled_parameters = tree_flatten(disabled.model.parameters())
        self.assertEqual(
            [name for name, _ in enabled_parameters],
            [name for name, _ in disabled_parameters],
        )
        for (name, left), (_, right) in zip(
            enabled_parameters, disabled_parameters
        ):
            mx.eval(left, right)
            self.assertEqual(left.shape, right.shape, name)
            self.assertEqual(float(mx.abs(left - right).max().item()), 0.0, name)

    def test_locked_binding_architecture_parameter_count(self):
        from dataclasses import replace as dc_replace

        from mlx.utils import tree_flatten

        trainer = OrganismTrainer(
            dc_replace(
                self._trainer().config,
                hidden_size=64,
                token_embed_size=32,
                episodic_binding_size=16,
            )
        )
        count = sum(
            int(parameter.size)
            for _, parameter in tree_flatten(trainer.model.parameters())
        )
        self.assertEqual(count, 105_930)

    def test_delayed_bodily_loss_reaches_earlier_lexical_value(self):
        from copy import deepcopy

        import mlx.nn as nn
        from mlx.utils import tree_flatten

        (
            trainer,
            initial_packet,
            _,
            _,
            target,
            post_packet,
            _,
            _,
            _,
        ) = self._inspect_first_object()
        wait_packet, _, _, _, _ = trainer.world.step(Action.WAIT)
        model = trainer.model
        target_slot = next(
            slot
            for slot, (_, _, surface) in enumerate(wait_packet.visible)
            if surface == SURFACE_INDEX[target.name]
        )
        consume_action = object_option_action_index(
            "consume",
            target_slot,
            consume_options=True,
            inspect_options=True,
            visible_slots=model.visible_slots,
        )
        outcome_world = deepcopy(trainer.world)
        outcome_packet, _, _, _, _ = execute_agent_action(
            outcome_world,
            wait_packet,
            consume_action,
            consume_options=True,
            inspect_options=True,
        )
        target_delta = np.asarray(outcome_packet.needs, dtype=np.float32) - np.asarray(
            wait_packet.needs, dtype=np.float32
        )
        packets = (initial_packet, post_packet, wait_packet)
        vectors = mx.array(np.stack([packet.vector() for packet in packets])[None])
        tokens = mx.array(
            np.asarray([packet.tokens for packet in packets], dtype=np.int32)[None]
        )

        def delayed_loss(active_model):
            states, _ = active_model.core_states(vectors, tokens, None)
            _, predicted_delta, _, _ = active_model.predict_consequences(
                states[:, -1:, :],
                mx.array([[consume_action]], dtype=mx.int32),
                vectors[:, -1:, :],
            )
            return ((predicted_delta[0, 0] - mx.array(target_delta)) ** 2).mean()

        loss, grads = nn.value_and_grad(model, delayed_loss)(model)
        flat_grads = dict(tree_flatten(grads))
        mx.eval(loss, grads)
        self.assertGreater(
            float(mx.abs(flat_grads["binding_value.weight"]).sum().item()),
            0.0,
        )
        self.assertGreater(
            float(mx.abs(flat_grads["token_embedding.weight"]).sum().item()),
            0.0,
        )


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

    def test_segment_keeps_audit_metadata_aligned(self):
        trainer = OrganismTrainer(self._config())
        segment, _, _ = trainer.collect_segment()
        self.assertEqual(len(segment.events), len(segment))
        self.assertEqual(len(segment.chosen_kinds), len(segment))
        self.assertEqual(len(segment.chosen_surfaces), len(segment))

    def test_bodily_event_audit_writes_observational_report(self):
        import json
        from dataclasses import replace as dc_replace
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            audit_path = f"{directory}/event_path.json"
            trainer = OrganismTrainer(
                dc_replace(
                    self._config(),
                    total_steps=64,
                    bodily_event_audit_json=audit_path,
                    world_model_replay_capacity=2,
                )
            )
            trainer.train()
            report = json.loads(Path(audit_path).read_text())
        self.assertEqual(report["audit"], "bodily_event_training_path")
        self.assertEqual(report["total_steps"], 64)
        self.assertIn("online", report)
        self.assertIn("final_replay", report)
        self.assertIn("gradient_samples", report)
        self.assertFalse(report["need_balanced_replay"]["enabled"])

    def test_need_balanced_replay_requires_an_active_reservoir(self):
        from dataclasses import replace as dc_replace

        with self.assertRaisesRegex(ValueError, "positive replay capacity"):
            OrganismTrainer(
                dc_replace(self._config(), need_balanced_replay=True)
            )

    def test_need_balanced_replay_alternates_and_selects_requested_need(self):
        from dataclasses import replace as dc_replace
        from homesocial.organism.train import _ReplayItem, _Segment

        trainer = OrganismTrainer(
            dc_replace(
                self._config(),
                world_model_replay_capacity=4,
                world_model_replay_updates=1,
                need_balanced_replay=True,
            )
        )
        food = _ReplayItem(_Segment(), None, frozenset({0}))
        water = _ReplayItem(_Segment(), None, frozenset({1}))
        trainer.world_model_replay = [food, water]

        selected = [trainer._sample_world_model_replay() for _ in range(6)]
        self.assertEqual(selected, [food, water, food, water, food, water])
        accounting = trainer._need_balanced_replay_accounting()
        self.assertEqual(
            accounting["requested_need_counts"], {"food": 3, "water": 3}
        )
        self.assertEqual(accounting["uniform_fallback_count"], 0)
        self.assertEqual(
            accounting["post_both_pool_eligible_selection_rate"], 1.0
        )

    def test_replay_need_classifier_uses_visible_binding_and_lived_delta(self):
        from dataclasses import replace as dc_replace
        from unittest.mock import patch
        from homesocial.organism.train import _Segment

        trainer = OrganismTrainer(
            dc_replace(
                self._config(),
                consume_options=True,
                inspect_options=True,
                episodic_binding_size=4,
                world_model_replay_capacity=2,
                world_model_replay_updates=1,
                need_balanced_replay=True,
            )
        )
        segment = _Segment()
        segment.vectors = [
            np.zeros(trainer.model.vector_size, dtype=np.float32)
            for _ in range(3)
        ]
        padding = tuple(
            [TOKEN_TO_ID[PAD_TOKEN]] * trainer.model.tokens_per_utterance
        )
        segment.tokens = [padding, padding, padding]
        segment.actions = [0, 0, 0]
        segment.next_needs = [
            (0.30, 0.00, 0.00, 0.00),
            (0.00, 0.25, 0.00, 0.00),
            (0.40, 0.00, 0.00, 0.00),
        ]
        # Intentionally contradictory simulator-only fields: selection must
        # not consult them.
        segment.events = ["consumed_poison"] * 3
        segment.chosen_kinds = ["poison"] * 3
        segment.chosen_surfaces = ["unknown"] * 3
        states = mx.zeros((1, 3, trainer.model.state_size))
        with patch.object(
            trainer.model, "core_states", return_value=(states, states[:, -1])
        ), patch(
            "homesocial.organism.train.bound_consumption_action_mask",
            return_value=mx.array([[True, True, False]]),
        ):
            eligible = trainer._learner_visible_replay_needs(segment, None)

        self.assertEqual(eligible, frozenset({0, 1}))

    def test_need_balanced_replay_falls_back_only_for_an_empty_need_pool(self):
        from dataclasses import replace as dc_replace
        from homesocial.organism.train import _ReplayItem, _Segment

        trainer = OrganismTrainer(
            dc_replace(
                self._config(),
                world_model_replay_capacity=2,
                world_model_replay_updates=1,
                need_balanced_replay=True,
            )
        )
        food = _ReplayItem(_Segment(), None, frozenset({0}))
        trainer.world_model_replay = [food]

        self.assertIs(trainer._sample_world_model_replay(), food)
        self.assertIs(trainer._sample_world_model_replay(), food)
        accounting = trainer._need_balanced_replay_accounting()
        self.assertEqual(accounting["uniform_fallback_count"], 1)
        self.assertEqual(
            accounting["requested_eligible_selection_counts"],
            {"food": 1, "water": 0},
        )

    def test_uniform_replay_keeps_the_seeded_choice_path(self):
        from random import Random
        from dataclasses import replace as dc_replace
        from homesocial.organism.train import _ReplayItem, _Segment

        trainer = OrganismTrainer(
            dc_replace(
                self._config(),
                world_model_replay_capacity=3,
                world_model_replay_updates=1,
            )
        )
        items = [
            _ReplayItem(_Segment(), None, frozenset({need}))
            for need in (0, 1, 0)
        ]
        trainer.world_model_replay = items
        reference = Random(trainer.config.seed + 1_000_003)

        selected = [trainer._sample_world_model_replay() for _ in range(8)]
        expected = [reference.choice(items) for _ in range(8)]
        self.assertEqual(selected, expected)

    def test_binding_consequence_geometry_is_finite_and_slot_matched(self):
        from dataclasses import replace as dc_replace

        trainer = OrganismTrainer(
            dc_replace(
                self._config(),
                consume_options=True,
                inspect_options=True,
                episodic_binding_size=8,
                island=IslandConfig(
                    semantic_choice_horizon=40,
                    semantic_choice_objects=3,
                    semantic_choice_return_duration=6,
                    semantic_choice_rounds=2,
                ),
            )
        )
        audit = audit_binding_consequence_geometry(
            trainer.model,
            contexts=2,
            semantic_choice_rounds=2,
        )
        self.assertEqual(audit["audited_slot_contexts"], 6.0)
        self.assertIn("lexical_min_to_max_distance_ratio", audit)
        self.assertIn("settled_water_intended_label_rate", audit)
        for value in audit.values():
            self.assertTrue(math.isfinite(value))

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
        self.assertIn("consume_attempts_per_episode", stats)
        self.assertIn("resource_consumes_per_episode", stats)
        self.assertIn("option_decisions_per_episode", stats)
        self.assertGreaterEqual(stats["mean_steps"], 1.0)

    def test_semantic_choice_evaluation_reports_coverage_and_accuracy(self):
        from dataclasses import replace as dc_replace

        trainer = OrganismTrainer(
            dc_replace(
                self._config(),
                consume_options=True,
                inspect_options=True,
                island=IslandConfig(semantic_choice_horizon=12),
            )
        )
        stats = evaluate_semantic_choice(
            trainer.model,
            language_mode="grounded",
            episodes=4,
            base_seed=1_090,
            semantic_choice_horizon=12,
            consume_options=True,
            inspect_options=True,
        )
        self.assertEqual(stats["trials"], 4.0)
        self.assertAlmostEqual(
            stats["choice_rate"] + stats["timeout_rate"], 1.0
        )
        self.assertLessEqual(stats["correct_choices"], stats["choices_made"])
        self.assertLessEqual(stats["inspected_choices"], stats["choices_made"])
        for value in stats.values():
            self.assertTrue(math.isfinite(value))

    def test_semantic_choice_evaluation_reports_persistent_rounds(self):
        from dataclasses import replace as dc_replace

        trainer = OrganismTrainer(
            dc_replace(
                self._config(),
                semantic_choice_childhood_steps=1,
                consume_options=True,
                inspect_options=True,
                episodic_binding_size=4,
                island=IslandConfig(
                    semantic_choice_horizon=40,
                    semantic_choice_objects=3,
                    semantic_choice_low_need=0.55,
                    semantic_choice_rounds=2,
                    semantic_choice_return_duration=6,
                ),
            )
        )
        stats = evaluate_semantic_choice(
            trainer.model,
            language_mode="grounded",
            episodes=2,
            base_seed=1_095,
            semantic_choice_horizon=40,
            semantic_choice_objects=3,
            semantic_choice_low_need=0.55,
            semantic_choice_rounds=2,
            semantic_choice_return_duration=6,
            consume_options=True,
            inspect_options=True,
        )
        self.assertEqual(stats["trials"], 4.0)
        self.assertEqual(stats["lives"], 2.0)
        self.assertIn("round_1_correct_rate", stats)
        self.assertIn("round_2_inspection_rate", stats)
        for value in stats.values():
            self.assertTrue(math.isfinite(value))

    def test_semantic_choice_childhood_switches_without_resetting_model(self):
        from dataclasses import replace as dc_replace

        trainer = OrganismTrainer(
            dc_replace(
                self._config(),
                semantic_choice_childhood_steps=10,
                consume_options=True,
                inspect_options=True,
                island=IslandConfig(semantic_choice_horizon=8),
            )
        )
        model_id = id(trainer.model)
        self.assertTrue(trainer.world.config.semantic_choice_trial)
        self.assertEqual(trainer.world.grid.choice_need in {"food", "water"}, True)
        trainer.global_steps = 10
        trainer._finish_life(survived=True)
        self.assertTrue(trainer.life_stats[-1].semantic_choice_trial)
        self.assertFalse(trainer.world.config.semantic_choice_trial)
        self.assertIsNone(trainer.world.grid.choice_need)
        self.assertEqual(id(trainer.model), model_id)

    def test_choice_policy_targets_preserve_raw_delayed_returns(self):
        from homesocial.organism.train import _Segment

        trainer = OrganismTrainer(self._config())
        segment = _Segment()
        segment.semantic_choice_trial = True
        segment.values = [0.0, 0.0]
        segment.shaped_rewards = [0.002, 0.104]
        segment.dones = [False, False]
        segment.durations = [3, 3]
        advantages, returns = trainer._policy_targets(segment, bootstrap=0.0)
        expected, expected_returns = compute_gae(
            np.asarray(segment.shaped_rewards, dtype=np.float32),
            np.asarray(segment.values, dtype=np.float32),
            0.0,
            np.asarray(segment.dones, dtype=np.float32),
            np.asarray(segment.durations, dtype=np.float32),
            discount=trainer.config.discount,
            gae_lambda=trainer.config.gae_lambda,
        )
        np.testing.assert_allclose(advantages, expected)
        np.testing.assert_allclose(returns, expected_returns)
        self.assertGreater(advantages[0], 0.0)

    def test_choice_training_uses_exact_primitive_tick_budget(self):
        from dataclasses import replace as dc_replace

        trainer = OrganismTrainer(
            dc_replace(
                self._config(),
                total_steps=37,
                semantic_choice_childhood_steps=37,
                consume_options=True,
                inspect_options=True,
                island=IslandConfig(semantic_choice_horizon=20),
            )
        )
        trainer.train()
        self.assertEqual(trainer.loss_log[-1]["steps"], 37.0)
        self.assertEqual(trainer.global_steps, 37)

    def test_counterfactual_self_model_audit_reports_finite_metrics(self):
        trainer = OrganismTrainer(self._config())
        stats = audit_self_model_actions(
            trainer.model,
            language_mode="grounded",
            episodes=1,
            base_seed=991,
            max_decisions=2,
            max_steps=60,
        )
        self.assertEqual(stats["audited_decisions"], 2.0)
        self.assertGreater(stats["candidate_outcomes"], 0.0)
        for value in stats.values():
            self.assertTrue(math.isfinite(value))

    def test_label_to_self_model_audit_runs_without_training_leakage(self):
        from dataclasses import replace as dc_replace

        trainer = OrganismTrainer(
            dc_replace(
                self._config(),
                consume_options=True,
                inspect_options=True,
            )
        )
        stats = audit_label_to_self_model(
            trainer.model,
            episodes=4,
            base_seed=1_991,
            max_inspections=2,
            max_steps=120,
            semantic_choice_trial=True,
            semantic_choice_horizon=20,
        )
        self.assertEqual(stats["audited_inspections"], 2.0)
        self.assertEqual(stats["unique_episodes"], 2.0)
        self.assertEqual(stats["unique_target_contexts"], 2.0)
        self.assertEqual(stats["resource_cases"], 1.0)
        self.assertEqual(stats["danger_cases"], 1.0)
        self.assertEqual(stats["resource_swap_cases"], 1.0)
        self.assertIn("resource_swap_counterfactual_kind_accuracy", stats)
        for value in stats.values():
            self.assertTrue(math.isfinite(value))

    def test_delayed_referent_binding_audit_runs_on_three_way_memory(self):
        from dataclasses import replace as dc_replace

        trainer = OrganismTrainer(
            dc_replace(
                self._config(),
                semantic_choice_childhood_steps=1,
                consume_options=True,
                inspect_options=True,
                episodic_binding_size=4,
                island=IslandConfig(
                    semantic_choice_horizon=40,
                    semantic_choice_objects=3,
                    semantic_choice_return_duration=6,
                ),
            )
        )
        stats = audit_label_referent_binding(
            trainer.model,
            episodes=6,
            base_seed=3_090,
            max_contexts=6,
            semantic_choice_horizon=40,
            semantic_choice_objects=3,
            semantic_choice_return_duration=6,
        )
        self.assertEqual(stats["audited_contexts"], 6.0)
        self.assertEqual(stats["episodic_memory_available"], 1.0)
        self.assertIn(
            "final_state_controlled_direct_key_reassignment_hit_rate",
            stats,
        )
        for value in stats.values():
            self.assertTrue(math.isfinite(value))

    def test_cross_round_reuse_audit_measures_every_later_round(self):
        from dataclasses import replace as dc_replace

        trainer = OrganismTrainer(
            dc_replace(
                self._config(),
                semantic_choice_childhood_steps=1,
                consume_options=True,
                inspect_options=True,
                episodic_binding_size=4,
                island=IslandConfig(
                    semantic_choice_horizon=40,
                    semantic_choice_objects=3,
                    semantic_choice_low_need=0.55,
                    semantic_choice_rounds=4,
                    semantic_choice_return_duration=6,
                ),
            )
        )
        stats = audit_cross_round_label_reuse(
            trainer.model,
            lives=3,
            base_seed=1_994_000,
            rounds=4,
        )
        self.assertEqual(stats["audited_lives"], 3.0)
        self.assertEqual(stats["completed_lives"], 3.0)
        # Two acquisition rounds, so rounds 3 and 4 are measured per life.
        self.assertEqual(stats["measured_rounds"], 6.0)
        self.assertIn("round_3_reuse_accuracy", stats)
        self.assertIn("round_4_reuse_accuracy", stats)
        self.assertNotIn("round_2_reuse_accuracy", stats)
        self.assertEqual(stats["mean_valid_memory_rows"], 2.0)
        for value in stats.values():
            self.assertTrue(math.isfinite(value))

    def test_observation_branching_feasibility_audit_is_finite(self):
        from dataclasses import replace as dc_replace

        trainer = OrganismTrainer(
            dc_replace(
                self._config(),
                semantic_choice_childhood_steps=1,
                consume_options=True,
                inspect_options=True,
                episodic_binding_size=4,
                island=IslandConfig(
                    semantic_choice_horizon=40,
                    semantic_choice_objects=3,
                    semantic_choice_low_need=0.55,
                    semantic_choice_return_duration=6,
                ),
            )
        )
        stats = audit_observation_branching_planner(
            trainer.model,
            episodes=2,
            base_seed=1_991_000,
        )
        self.assertEqual(stats["audited_contexts"], 2.0)
        self.assertIn("intact_mean_best_inspect_advantage", stats)
        self.assertIn("write_causal_advantage_drop", stats)
        self.assertIn("collapsed_label_contingency_drop", stats)
        for value in stats.values():
            self.assertTrue(math.isfinite(value))

    def test_protocol_return_origin_audit_is_paired_and_finite(self):
        from dataclasses import replace as dc_replace

        trainer = OrganismTrainer(
            dc_replace(
                self._config(),
                semantic_choice_childhood_steps=1,
                consume_options=True,
                inspect_options=True,
                episodic_binding_size=4,
                island=IslandConfig(
                    semantic_choice_horizon=40,
                    semantic_choice_objects=3,
                    semantic_choice_low_need=0.55,
                    semantic_choice_return_duration=6,
                ),
            )
        )
        stats = audit_protocol_return_origin(
            trainer.model,
            contexts=2,
            base_seed=1_991_000,
        )
        self.assertEqual(stats["audited_contexts"], 2.0)
        self.assertIn("aliased_worst_return_absolute_error", stats)
        self.assertIn("chained_worst_return_absolute_error", stats)
        self.assertIn("chained_lower_error_context_rate", stats)
        for value in stats.values():
            self.assertTrue(math.isfinite(value))

    def test_terminal_consume_value_calibration_is_grouped_and_finite(self):
        from dataclasses import replace as dc_replace

        trainer = OrganismTrainer(
            dc_replace(
                self._config(),
                semantic_choice_childhood_steps=1,
                consume_options=True,
                inspect_options=True,
                episodic_binding_size=4,
                island=IslandConfig(
                    semantic_choice_horizon=40,
                    semantic_choice_objects=3,
                    semantic_choice_low_need=0.55,
                    semantic_choice_return_duration=6,
                ),
            )
        )
        stats = audit_terminal_consume_value_calibration(
            trainer.model,
            contexts=6,
            base_seed=1_991_000,
        )
        self.assertEqual(stats["all_contexts"], 6.0)
        self.assertGreater(stats["food_contexts"], 0.0)
        self.assertGreater(stats["water_contexts"], 0.0)
        self.assertIn(
            "all_demanded_minus_best_wrong_predicted_realized_ratio",
            stats,
        )
        for value in stats.values():
            self.assertTrue(math.isfinite(value))

    def test_real_explicit_event_transfer_is_paired_and_finite(self):
        from dataclasses import replace as dc_replace

        trainer = OrganismTrainer(
            dc_replace(
                self._config(),
                semantic_choice_childhood_steps=1,
                consume_options=True,
                inspect_options=True,
                episodic_binding_size=4,
                island=IslandConfig(
                    semantic_choice_horizon=40,
                    semantic_choice_objects=3,
                    semantic_choice_low_need=0.55,
                    semantic_choice_return_duration=6,
                ),
            )
        )
        stats = audit_real_vs_explicit_event_transfer(
            trainer.model,
            contexts=6,
            base_seed=1_991_000,
        )
        self.assertEqual(stats["all_real_immediate_contexts"], 6.0)
        self.assertEqual(stats["all_real_settled_contexts"], 6.0)
        self.assertEqual(stats["all_explicit_settled_contexts"], 6.0)
        self.assertIn(
            "all_real_settled_minus_explicit_settled_margin",
            stats,
        )
        for value in stats.values():
            self.assertTrue(math.isfinite(value))

    def test_semantic_choice_information_upper_bound_uses_exact_protocol(self):
        stats = audit_semantic_choice_information_upper_bound(
            episodes=6,
            base_seed=1_992_000,
        )
        self.assertEqual(stats["audited_contexts"], 6.0)
        self.assertEqual(stats["blind_immediate_mean_ticks"], 4.0)
        self.assertEqual(stats["one_inspection_mean_ticks"], 14.0)
        self.assertEqual(stats["clairvoyant_immediate_correct_rate"], 1.0)
        self.assertEqual(stats["clairvoyant_immediate_poison_rate"], 0.0)
        self.assertGreaterEqual(
            stats["two_inspections_mean_ticks"],
            stats["one_inspection_mean_ticks"],
        )
        for value in stats.values():
            self.assertTrue(math.isfinite(value))

    def test_persistent_mapping_rent_audit_reuses_at_most_two_labels(self):
        stats = audit_persistent_mapping_information_rent(
            lives=4,
            base_seed=1_993_000,
            round_counts=(1, 2),
        )
        self.assertEqual(stats["audited_lives"], 4.0)
        for rounds in (1, 2):
            prefix = f"rounds_{rounds}"
            self.assertEqual(
                stats[f"{prefix}_persistent_correct_rate"],
                1.0,
            )
            self.assertLessEqual(
                stats[
                    f"{prefix}_persistent_mean_inspections_per_life"
                ],
                2.0,
            )
            self.assertEqual(
                stats[f"{prefix}_clairvoyant_correct_rate"],
                1.0,
            )
            self.assertEqual(
                stats[f"{prefix}_blind_mean_ticks_per_round"],
                4.0,
            )
        for value in stats.values():
            self.assertTrue(math.isfinite(value))

    def test_implemented_persistent_environment_completes_all_rounds(self):
        stats = audit_persistent_choice_environment(
            lives=3,
            base_seed=1_994_000,
            rounds=2,
        )
        self.assertEqual(stats["audited_lives"], 3.0)
        for policy in ("blind", "persistent", "clairvoyant"):
            self.assertEqual(
                stats[f"{policy}_completed_rounds_per_life"],
                2.0,
            )
        self.assertEqual(stats["persistent_correct_rate"], 1.0)
        self.assertLessEqual(
            stats["persistent_mean_inspections_per_life"],
            2.0,
        )
        self.assertEqual(stats["clairvoyant_correct_rate"], 1.0)
        for value in stats.values():
            self.assertTrue(math.isfinite(value))

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

    def test_caregiver_offer_curriculum_fades_to_zero(self):
        from dataclasses import replace as dc_replace

        trainer = OrganismTrainer(
            dc_replace(
                self._config(),
                caregiver_offer_threshold_start=0.75,
                caregiver_offer_curriculum_steps=1000,
                caregiver_offer_distance_end=2,
            )
        )
        self.assertAlmostEqual(trainer.caregiver_offer_threshold(), 0.75)
        self.assertAlmostEqual(trainer.world.caregiver_offer_threshold, 0.75)
        trainer.global_steps = 500
        self.assertAlmostEqual(trainer.caregiver_offer_threshold(), 0.375)
        trainer._finish_life(survived=False)
        self.assertEqual(trainer.world.caregiver_offer_distance, 1)
        trainer.global_steps = 1000
        self.assertAlmostEqual(trainer.caregiver_offer_threshold(), 0.0)
        trainer._finish_life(survived=False)
        self.assertAlmostEqual(trainer.world.caregiver_offer_threshold, 0.0)
        self.assertEqual(trainer.world.caregiver_offer_distance, 2)

    def test_bc_demonstrations_mask_language(self):
        from dataclasses import replace as dc_replace

        from homesocial.creole.vocab import PAD_TOKEN, TOKEN_TO_ID

        trainer = OrganismTrainer(
            dc_replace(
                self._config(),
                bc_warmstart_lives=1,
                max_steps=40,
                island=IslandConfig(max_steps=40),
            )
        )
        demonstrations = trainer.collect_oracle_demonstrations()
        self.assertEqual(len(demonstrations), 1)
        self.assertGreater(len(demonstrations[0]), 0)
        self.assertEqual(
            {token for utterance in demonstrations[0].tokens for token in utterance},
            {TOKEN_TO_ID[PAD_TOKEN]},
        )

    def test_bc_warmstart_runs_and_reports_metrics(self):
        from dataclasses import replace as dc_replace

        trainer = OrganismTrainer(
            dc_replace(
                self._config(),
                bc_warmstart_lives=1,
                bc_epochs=1,
                max_steps=40,
                island=IslandConfig(max_steps=40),
            )
        )
        stats = trainer.behavior_clone()
        self.assertIsNotNone(stats)
        assert stats is not None
        self.assertEqual(stats.lives, 1)
        self.assertGreater(stats.transitions, 0)
        self.assertTrue(math.isfinite(stats.final_epoch_loss))
        self.assertGreaterEqual(stats.final_accuracy, 0.0)
        self.assertLessEqual(stats.final_accuracy, 1.0)

    def test_model_initialization_is_seeded(self):
        first = OrganismTrainer(self._config()).model
        mx.eval(first.policy.weight)
        second = OrganismTrainer(self._config()).model
        mx.eval(second.policy.weight)
        self.assertLess(
            float(mx.abs(first.policy.weight - second.policy.weight).max()), 1e-7
        )

    def test_consume_option_reaches_visible_offer_without_kind_access(self):
        from homesocial.env import Action
        from homesocial.island.world import IslandWorld

        world = IslandWorld(
            IslandConfig(
                caregiver_offer_threshold=1.0,
                caregiver_offer_distance=2,
            ),
            seed=71,
        )
        packet = world.reset(71)
        packet, _, _, _, _ = world.step(Action.WAIT)
        assert world.grid.offered_pos is not None
        offered_relative = (
            world.grid.offered_pos[0] - packet.position[0],
            world.grid.offered_pos[1] - packet.position[1],
        )
        slot = next(
            index
            for index, (dx, dy, _) in enumerate(packet.visible)
            if (dx, dy) == offered_relative
        )
        _, _, terminated, truncated, info = execute_agent_action(
            world,
            packet,
            len(ACTIONS) + slot,
            consume_options=True,
        )
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["offered_consumed"])
        self.assertIn(info["event"], {"consumed_food", "consumed_water"})
        self.assertGreater(int(info["duration"]), 1)

    def test_option_model_has_one_action_per_visible_slot(self):
        from dataclasses import replace as dc_replace

        trainer = OrganismTrainer(
            dc_replace(self._config(), consume_options=True)
        )
        self.assertEqual(
            trainer.model.action_size,
            len(ACTIONS) + trainer.world.config.max_visible_slots,
        )
        mask = available_action_mask(trainer.packet, trainer.model.action_size)
        self.assertTrue(mask[: len(ACTIONS)].all())
        self.assertEqual(
            int(mask[len(ACTIONS) :].sum()), len(trainer.packet.visible)
        )

    def test_consume_and_inspect_have_separate_visible_slot_blocks(self):
        from dataclasses import replace as dc_replace

        trainer = OrganismTrainer(
            dc_replace(
                self._config(),
                consume_options=True,
                inspect_options=True,
            )
        )
        slots = trainer.world.config.max_visible_slots
        self.assertEqual(trainer.model.action_size, len(ACTIONS) + 2 * slots)
        mask = available_action_mask(
            trainer.packet,
            trainer.model.action_size,
        )
        visible = len(trainer.packet.visible)
        self.assertEqual(int(mask[len(ACTIONS) : len(ACTIONS) + slots].sum()), visible)
        self.assertEqual(int(mask[len(ACTIONS) + slots :].sum()), visible)
        self.assertEqual(
            decode_object_option(
                len(ACTIONS) + slots,
                consume_options=True,
                inspect_options=True,
                visible_slots=slots,
            )[0],
            "inspect",
        )

    def test_inspect_option_approaches_labels_and_does_not_consume(self):
        from dataclasses import replace as dc_replace

        from homesocial.env import Direction
        from homesocial.island.world import IslandWorld

        world = IslandWorld(seed=73)
        world.reset(73)
        target = next(obj for obj in world.grid.objects if obj.consumable)
        world.grid.agent_pos = (3, 3)
        world.grid.direction = Direction.NORTH
        world.grid.objects = [dc_replace(target, pos=(3, 1))]
        packet = world._packet(world.grid._observe(None, None), None)
        slot = next(
            index
            for index, (dx, dy, _) in enumerate(packet.visible)
            if (dx, dy) == (0, -2)
        )
        slots = world.config.max_visible_slots
        next_packet, _, terminated, truncated, info = execute_agent_action(
            world,
            packet,
            len(ACTIONS) + slots + slot,
            consume_options=True,
            inspect_options=True,
        )
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["option_kind"], "inspect")
        self.assertGreaterEqual(int(info["duration"]), 2)
        self.assertTrue(str(info["situation"]).startswith("label|"))
        self.assertTrue(_is_voluntary_inspection_event(info))
        self.assertFalse(any(str(event).startswith("consumed_") for event in info["events"]))
        self.assertEqual(len(world.grid.objects), 1)
        self.assertNotEqual(packet.tokens, next_packet.tokens)

    def test_delayed_choice_options_and_return_are_geometry_neutral(self):
        from copy import deepcopy
        from dataclasses import replace as dc_replace

        from homesocial.env import Direction

        trainer = OrganismTrainer(
            dc_replace(
                self._config(),
                semantic_choice_childhood_steps=1,
                consume_options=True,
                inspect_options=True,
                island=IslandConfig(
                    semantic_choice_horizon=40,
                    semantic_choice_objects=3,
                    semantic_choice_return_duration=6,
                ),
            )
        )
        packet = trainer.packet
        returned_packets = []
        for slot in range(len(packet.visible)):
            world = deepcopy(trainer.world)
            inspect_action = object_option_action_index(
                "inspect",
                slot,
                consume_options=True,
                inspect_options=True,
                visible_slots=trainer.model.visible_slots,
            )
            labeled, _, terminated, truncated, inspect_info = execute_agent_action(
                world,
                packet,
                inspect_action,
                consume_options=True,
                inspect_options=True,
            )
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertEqual(inspect_info["duration"], 4)
            self.assertTrue(_is_voluntary_inspection_event(inspect_info))
            self.assertTrue(world.semantic_choice_return_pending)

            pending_mask = available_action_mask(
                labeled,
                trainer.model.action_size,
                visible_slots=trainer.model.visible_slots,
                semantic_choice_delayed=True,
                return_pending=True,
            )
            self.assertEqual(int(pending_mask.sum()), 1)
            self.assertTrue(pending_mask[ACTIONS.index(Action.WAIT)])

            returned, _, terminated, truncated, return_info = execute_agent_action(
                world,
                labeled,
                ACTIONS.index(Action.WAIT),
                consume_options=True,
                inspect_options=True,
            )
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertEqual(return_info["duration"], 6)
            self.assertTrue(return_info["forced_return"])
            self.assertFalse(world.semantic_choice_return_pending)
            self.assertEqual(returned.position, world.semantic_choice_center)
            self.assertEqual(world.grid.direction, Direction.NORTH)
            self.assertEqual(
                returned.last_action_index, ACTIONS.index(Action.WAIT)
            )
            self.assertEqual(
                set(returned.tokens), {TOKEN_TO_ID[PAD_TOKEN]}
            )
            returned_packets.append(returned)

        for returned in returned_packets[1:]:
            np.testing.assert_allclose(
                returned.needs, returned_packets[0].needs, atol=0.0, rtol=0.0
            )
            self.assertEqual(returned.position, returned_packets[0].position)
            self.assertEqual(
                returned.direction_index, returned_packets[0].direction_index
            )

        trainer.packet = execute_agent_action(
            trainer.world,
            packet,
            object_option_action_index(
                "inspect",
                0,
                consume_options=True,
                inspect_options=True,
                visible_slots=trainer.model.visible_slots,
            ),
            consume_options=True,
            inspect_options=True,
        )[0]
        _, _, _, forced_mask, decision_weight, _ = trainer._act()
        self.assertEqual(decision_weight, 0.0)
        self.assertEqual(int(forced_mask.sum()), 1)

    def test_delayed_choice_planner_stops_after_terminal_consume(self):
        from dataclasses import replace as dc_replace

        trainer = OrganismTrainer(
            dc_replace(
                self._config(),
                semantic_choice_childhood_steps=1,
                consume_options=True,
                inspect_options=True,
                island=IslandConfig(
                    semantic_choice_horizon=40,
                    semantic_choice_objects=3,
                    semantic_choice_low_need=0.55,
                    semantic_choice_return_duration=6,
                ),
            )
        )
        vector = mx.array(trainer.packet.vector()[None, None, :])
        tokens = mx.array(
            np.asarray(trainer.packet.tokens, dtype=np.int32)[None, None, :]
        )
        states, _ = trainer.model.core_states(vector, tokens, None)
        predicted_needs, predicted_rewards = predict_all_action_consequences(
            trainer.model, states, vector
        )
        one_step = mx.min(predicted_needs, axis=-1) + 0.5 * predicted_rewards
        two_step = two_step_action_scores(
            trainer.model,
            states,
            vector,
            reward_weight=0.5,
            force_return_after_inspect=True,
        )
        consume_start = len(ACTIONS)
        consume_stop = consume_start + trainer.model.visible_slots
        mx.eval(one_step, two_step)
        np.testing.assert_allclose(
            np.asarray(two_step[..., consume_start:consume_stop]),
            np.asarray(one_step[..., consume_start:consume_stop]),
            atol=0.0,
            rtol=0.0,
        )

    def test_persistent_choice_round_exposes_consequence_before_demand_reset(self):
        from dataclasses import replace as dc_replace

        trainer = OrganismTrainer(
            dc_replace(
                self._config(),
                semantic_choice_childhood_steps=1,
                consume_options=True,
                inspect_options=True,
                episodic_binding_size=4,
                island=IslandConfig(
                    semantic_choice_horizon=40,
                    semantic_choice_objects=3,
                    semantic_choice_low_need=0.55,
                    semantic_choice_rounds=2,
                    semantic_choice_return_duration=6,
                ),
            )
        )
        world = trainer.world
        packet = trainer.packet
        first_need = world.grid.choice_need
        fixed_mapping = dict(world.grid.kind_by_surface)
        fixed_surfaces = {
            surface for _, _, surface in packet.visible
        }
        consumed_packet, _, terminated, truncated, info = execute_agent_action(
            world,
            packet,
            object_option_action_index(
                "consume",
                0,
                consume_options=True,
                inspect_options=True,
                visible_slots=trainer.model.visible_slots,
            ),
            consume_options=True,
            inspect_options=True,
        )
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["semantic_choice_round_complete"])
        self.assertTrue(world.semantic_choice_round_pending)
        self.assertNotEqual(consumed_packet.position, world.semantic_choice_center)

        next_packet, _, terminated, truncated, transition_info = (
            execute_agent_action(
                world,
                consumed_packet,
                ACTIONS.index(Action.WAIT),
                consume_options=True,
                inspect_options=True,
            )
        )
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(transition_info["forced_round_transition"])
        self.assertEqual(transition_info["duration"], 1)
        self.assertFalse(world.semantic_choice_round_pending)
        self.assertEqual(world.grid.choice_round_index, 1)
        self.assertNotEqual(world.grid.choice_need, first_need)
        self.assertEqual(world.grid.kind_by_surface, fixed_mapping)
        self.assertEqual(
            {surface for _, _, surface in next_packet.visible},
            fixed_surfaces,
        )
        self.assertEqual(next_packet.position, world.semantic_choice_center)
        self.assertEqual(next_packet.direction_index, 0)
        low_index = 0 if world.grid.choice_need == "food" else 1
        self.assertAlmostEqual(next_packet.needs[low_index], 0.55)

        correct_slot = next(
            slot
            for slot, (_, _, surface) in enumerate(next_packet.visible)
            if world.grid.kind_by_surface[
                next(
                    name
                    for name, index in SURFACE_INDEX.items()
                    if index == surface
                )
            ]
            == world.grid.choice_need
        )
        _, _, _, final_truncated, final_info = execute_agent_action(
            world,
            next_packet,
            object_option_action_index(
                "consume",
                correct_slot,
                consume_options=True,
                inspect_options=True,
                visible_slots=trainer.model.visible_slots,
            ),
            consume_options=True,
            inspect_options=True,
        )
        self.assertTrue(final_truncated)
        self.assertTrue(final_info["correct"])

    def test_primitive_inspection_label_uses_same_evaluation_criterion(self):
        from dataclasses import replace as dc_replace

        from homesocial.env import Direction
        from homesocial.island.world import IslandWorld

        world = IslandWorld(seed=731)
        world.reset(731)
        target = next(obj for obj in world.grid.objects if obj.consumable)
        world.grid.agent_pos = (3, 3)
        world.grid.direction = Direction.NORTH
        world.grid.objects = [dc_replace(target, pos=(3, 2))]
        packet = world._packet(world.grid._observe(None, None), None)
        _, _, terminated, truncated, info = execute_agent_action(
            world,
            packet,
            ACTIONS.index(Action.ASK),
            consume_options=True,
            inspect_options=True,
        )
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(_is_voluntary_inspection_event(info))

    def test_option_routes_off_selected_under_agent_object(self):
        from dataclasses import replace as dc_replace

        from homesocial.env import Direction
        from homesocial.island.world import IslandWorld, SURFACE_INDEX

        world = IslandWorld(seed=74)
        world.reset(74)
        target = next(obj for obj in world.grid.objects if obj.name == "mushroom")
        target = dc_replace(target, pos=(3, 3))
        world.grid.objects = [target]
        world.grid.agent_pos = target.pos
        world.grid.direction = Direction.NORTH
        packet = world._packet(world.grid._observe(None, None), None)
        slot = next(
            index
            for index, (dx, dy, surface) in enumerate(packet.visible)
            if (dx, dy, surface) == (0, 0, SURFACE_INDEX[target.name])
        )
        slots = world.config.max_visible_slots
        inspected, _, terminated, truncated, info = execute_agent_action(
            world,
            packet,
            len(ACTIONS) + slots + slot,
            consume_options=True,
            inspect_options=True,
        )
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(str(info["situation"]).startswith("label|"))
        self.assertEqual(info["situation"].split("|")[1], f"surface={target.name}")
        self.assertNotEqual(world.grid.agent_pos, target.pos)
        self.assertEqual(world.grid.object_ahead().name, target.name)

        consume_slot = next(
            index
            for index, (dx, dy, surface) in enumerate(inspected.visible)
            if (
                inspected.position[0] + dx,
                inspected.position[1] + dy,
                surface,
            )
            == (target.pos[0], target.pos[1], SURFACE_INDEX[target.name])
        )
        _, _, _, _, consume_info = execute_agent_action(
            world,
            inspected,
            len(ACTIONS) + consume_slot,
            consume_options=True,
            inspect_options=True,
        )
        self.assertEqual(consume_info["event"], f"consumed_{target.kind}")

    def test_inspect_option_routes_around_visible_blocker(self):
        from dataclasses import replace as dc_replace

        from homesocial.env import Direction
        from homesocial.island.world import IslandWorld, SURFACE_INDEX

        world = IslandWorld(seed=75)
        world.reset(75)
        target = next(obj for obj in world.grid.objects if obj.name == "mushroom")
        blocker = next(obj for obj in world.grid.objects if obj.name == "rock")
        target = dc_replace(target, pos=(3, 1))
        blocker = dc_replace(blocker, pos=(3, 2))
        world.grid.objects = [target, blocker]
        world.grid.agent_pos = (3, 3)
        world.grid.direction = Direction.NORTH
        packet = world._packet(world.grid._observe(None, None), None)
        slot = next(
            index
            for index, (_, _, surface) in enumerate(packet.visible)
            if surface == SURFACE_INDEX[target.name]
        )
        _, _, terminated, truncated, info = execute_agent_action(
            world,
            packet,
            len(ACTIONS) + world.config.max_visible_slots + slot,
            consume_options=True,
            inspect_options=True,
        )
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(str(info["situation"]).startswith("label|"))
        self.assertEqual(world.grid.object_ahead().name, target.name)
        self.assertLessEqual(info["duration"], 12)

    def test_nondefault_visible_slot_layout_is_consistent(self):
        from dataclasses import replace as dc_replace

        trainer = OrganismTrainer(
            dc_replace(
                self._config(),
                consume_options=True,
                inspect_options=True,
                island=IslandConfig(max_visible_slots=2),
            )
        )
        self.assertEqual(len(trainer.packet.vector()), trainer.model.vector_size)
        self.assertEqual(trainer.model.visible_slots, 2)
        stats = evaluate_semantic_choice(
            trainer.model,
            language_mode="grounded",
            episodes=2,
            base_seed=2_090,
            semantic_choice_horizon=12,
            consume_options=True,
            inspect_options=True,
        )
        self.assertEqual(stats["trials"], 2.0)

    def test_planned_option_training_smoke(self):
        from dataclasses import replace as dc_replace

        trainer = OrganismTrainer(
            dc_replace(
                self._config(),
                total_steps=64,
                consume_options=True,
                self_model_planning_scale=2.0,
                self_model_planning_start_steps=0,
            )
        )
        trainer.train()
        self.assertGreater(len(trainer.loss_log), 0)
        self.assertTrue(math.isfinite(trainer.loss_log[-1]["loss"]))

    def test_replay_multistep_training_smoke(self):
        from dataclasses import replace as dc_replace

        trainer = OrganismTrainer(
            dc_replace(
                self._config(),
                total_steps=96,
                consume_options=True,
                self_model_planning_scale=2.0,
                self_model_planning_start_steps=0,
                self_model_planning_horizon=2,
                multi_step_model_horizon=2,
                multi_step_model_weight=1.0,
                world_model_replay_capacity=2,
                world_model_replay_updates=1,
            )
        )
        trainer.train()
        self.assertGreater(len(trainer.world_model_replay), 0)
        self.assertLessEqual(len(trainer.world_model_replay), 2)
        self.assertTrue(math.isfinite(trainer.loss_log[-1]["loss"]))
        self.assertTrue(math.isfinite(trainer.loss_log[-1]["replay_loss"]))

    def test_three_way_episodic_binding_training_smoke(self):
        from dataclasses import replace as dc_replace

        trainer = OrganismTrainer(
            dc_replace(
                self._config(),
                total_steps=83,
                semantic_choice_childhood_steps=83,
                consume_options=True,
                inspect_options=True,
                episodic_binding_size=4,
                self_model_planning_scale=2.0,
                self_model_planning_start_steps=40,
                self_model_planning_horizon=2,
                multi_step_model_horizon=2,
                multi_step_model_weight=1.0,
                world_model_replay_capacity=2,
                world_model_replay_updates=1,
                island=IslandConfig(
                    semantic_choice_horizon=40,
                    semantic_choice_objects=3,
                    semantic_choice_return_duration=6,
                ),
            )
        )
        trainer.train()
        self.assertEqual(trainer.global_steps, 83)
        self.assertTrue(math.isfinite(trainer.loss_log[-1]["loss"]))
        self.assertTrue(math.isfinite(trainer.loss_log[-1]["replay_loss"]))


@unittest.skipIf(mx is None, "MLX is unavailable")
@unittest.skipIf(mx is None, "mlx is required")
class BodilyDriftLossTest(unittest.TestCase):
    """The drift term is off by default and stratifies per need, not per row."""

    def _batch(self):
        # One consumption transition (food jumps) and one drift-only
        # transition. The water entry of the consumption row is itself drift.
        target = mx.array(
            [
                [0.50, -0.014, -0.015, -0.002],
                [-0.010, -0.014, -0.015, -0.002],
            ]
        )
        predicted = mx.array(
            [
                [0.50, -0.114, -0.015, -0.002],
                [-0.110, -0.014, -0.015, -0.002],
            ]
        )
        return predicted, target

    def test_zero_weight_reproduces_the_sealed_loss(self):
        predicted, target = self._batch()
        legacy = (
            ((predicted - target) ** 2).mean(axis=-1)
            * (1.0 + 20.0 * mx.max(mx.abs(target), axis=-1))
        ).sum() / (1.0 + 20.0 * mx.max(mx.abs(target), axis=-1)).sum()
        current = bodily_delta_prediction_loss(
            predicted, target, change_boost=20.0
        )
        mx.eval(legacy, current)
        self.assertAlmostEqual(float(legacy), float(current), places=9)

    def test_drift_term_is_stratified_per_need_not_per_transition(self):
        """The water residual on a consumption row must count as drift."""

        predicted, target = self._batch()
        # Only the water entry of the consumption row carries a residual.
        only_event_row = mx.array([[0.50, -0.114, -0.015, -0.002]])
        with_weight = bodily_delta_prediction_loss(
            only_event_row,
            mx.array([[0.50, -0.014, -0.015, -0.002]]),
            change_boost=20.0,
            drift_weight=0.1,
        )
        without_weight = bodily_delta_prediction_loss(
            only_event_row,
            mx.array([[0.50, -0.014, -0.015, -0.002]]),
            change_boost=20.0,
        )
        mx.eval(with_weight, without_weight)
        self.assertGreater(float(with_weight), float(without_weight))

    def test_event_entries_are_excluded_from_the_drift_term(self):
        """A residual on a jump entry alone must leave the drift term at zero."""

        target = mx.array([[0.50, -0.014, -0.015, -0.002]])
        predicted = mx.array([[0.20, -0.014, -0.015, -0.002]])
        plain = bodily_delta_prediction_loss(
            predicted, target, change_boost=20.0
        )
        weighted = bodily_delta_prediction_loss(
            predicted, target, change_boost=20.0, drift_weight=0.3
        )
        mx.eval(plain, weighted)
        self.assertAlmostEqual(float(plain), float(weighted), places=9)

    def test_drift_term_is_scaled_relative_to_the_metabolic_scale(self):
        target = mx.zeros((1, 4))
        predicted = mx.full((1, 4), DRIFT_ERROR_SCALE)
        weighted = bodily_delta_prediction_loss(
            predicted, target, change_boost=20.0, drift_weight=1.0
        )
        plain = bodily_delta_prediction_loss(
            predicted, target, change_boost=20.0
        )
        mx.eval(weighted, plain)
        # A residual of exactly one metabolic scale contributes exactly one.
        self.assertAlmostEqual(float(weighted) - float(plain), 1.0, places=6)

    def test_event_term_is_scaled_relative_to_the_bodily_event_scale(self):
        target = mx.array([[EVENT_ERROR_SCALE, 0.0, 0.0, 0.0]])
        predicted = mx.zeros((1, 4))
        weighted = bodily_delta_prediction_loss(
            predicted,
            target,
            change_boost=20.0,
            event_weight=1.0,
        )
        plain = bodily_delta_prediction_loss(
            predicted,
            target,
            change_boost=20.0,
        )
        mx.eval(weighted, plain)
        # One event-scale residual contributes exactly one.
        self.assertAlmostEqual(float(weighted) - float(plain), 1.0, places=6)

    def test_consumption_event_term_excludes_unselected_event_rows(self):
        target = mx.array(
            [
                [EVENT_ERROR_SCALE, 0.0, 0.0, 0.0],
                [EVENT_ERROR_SCALE, 0.0, 0.0, 0.0],
            ]
        )
        predicted = mx.zeros((2, 4))
        plain = bodily_delta_prediction_loss(
            predicted, target, change_boost=20.0
        )
        scoped = bodily_delta_prediction_loss(
            predicted,
            target,
            change_boost=20.0,
            consumption_event_weight=1.0,
            consumption_valid=mx.array([1.0, 0.0]),
        )
        mx.eval(plain, scoped)
        self.assertAlmostEqual(float(scoped) - float(plain), 1.0, places=6)

    def test_consumption_event_term_requires_action_mask(self):
        with self.assertRaises(ValueError):
            bodily_delta_prediction_loss(
                mx.zeros((1, 4)),
                mx.array([[EVENT_ERROR_SCALE, 0.0, 0.0, 0.0]]),
                change_boost=20.0,
                consumption_event_weight=1.0,
            )

    def test_bound_consumption_term_balances_active_needs(self):
        target = mx.array(
            [
                [EVENT_ERROR_SCALE, 0.0, 0.0, 0.0],
                [EVENT_ERROR_SCALE, 0.0, 0.0, 0.0],
                [0.0, EVENT_ERROR_SCALE, 0.0, 0.0],
            ]
        )
        predicted = mx.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, EVENT_ERROR_SCALE / 2.0, 0.0, 0.0],
            ]
        )
        plain = bodily_delta_prediction_loss(
            predicted, target, change_boost=20.0
        )
        balanced = bodily_delta_prediction_loss(
            predicted,
            target,
            change_boost=20.0,
            bound_consumption_event_weight=1.0,
            bound_consumption_valid=mx.ones((3,)),
        )
        mx.eval(plain, balanced)
        # Food contributes 1.0 once after within-need averaging; water 0.25.
        self.assertAlmostEqual(float(balanced) - float(plain), 0.625, places=6)

    def test_bound_consumption_term_requires_binding_mask(self):
        with self.assertRaises(ValueError):
            bodily_delta_prediction_loss(
                mx.zeros((1, 4)),
                mx.array([[EVENT_ERROR_SCALE, 0.0, 0.0, 0.0]]),
                change_boost=20.0,
                bound_consumption_event_weight=1.0,
            )

    def test_consumption_action_mask_selects_primitive_and_option_only(self):
        visible_slots = 3
        primitive = ACTIONS.index(Action.CONSUME)
        consume_option = object_option_action_index(
            "consume",
            1,
            consume_options=True,
            inspect_options=True,
            visible_slots=visible_slots,
        )
        inspect_option = object_option_action_index(
            "inspect",
            1,
            consume_options=True,
            inspect_options=True,
            visible_slots=visible_slots,
        )
        mask = consumption_action_mask(
            mx.array([primitive, consume_option, inspect_option]),
            consume_options=True,
            inspect_options=True,
            visible_slots=visible_slots,
        )
        mx.eval(mask)
        np.testing.assert_array_equal(
            np.asarray(mask), np.asarray([True, True, False])
        )

    def test_drift_entries_are_excluded_from_the_event_term(self):
        target = mx.zeros((1, 4))
        predicted = mx.full((1, 4), EVENT_ERROR_SCALE)
        weighted = bodily_delta_prediction_loss(
            predicted,
            target,
            change_boost=20.0,
            event_weight=1.0,
        )
        plain = bodily_delta_prediction_loss(
            predicted,
            target,
            change_boost=20.0,
        )
        mx.eval(weighted, plain)
        self.assertAlmostEqual(float(weighted), float(plain), places=9)

    def test_threshold_separates_the_two_regimes_of_the_task(self):
        self.assertGreater(DRIFT_REGIME_THRESHOLD, 0.14)
        self.assertLess(DRIFT_REGIME_THRESHOLD, 0.225)


@unittest.skipIf(mx is None, "mlx is required")
class SplitDriftHeadTest(unittest.TestCase):
    """The drift path is range-limited and absent unless asked for."""

    def _model(self, *, split):
        return OrganismModel(
            vector_size=10,
            vocab_size=len(TOKEN_TO_ID),
            tokens_per_utterance=4,
            action_size=len(ACTIONS),
            hidden_size=16,
            split_drift_head=split,
        )

    def test_default_builds_no_extra_layer(self):
        model = self._model(split=False)
        self.assertFalse(model.has_split_drift_head)
        self.assertNotIn("drift_needs", model.parameters())

    def test_split_builds_the_extra_layer(self):
        model = self._model(split=True)
        self.assertTrue(model.has_split_drift_head)
        self.assertIn("drift_needs", model.parameters())

    def test_drift_path_cannot_express_a_consumption_jump(self):
        """Saturating both paths must stay inside 0.5 + the drift bound."""

        model = self._model(split=True)
        states = mx.random.normal((1, 3, 16)) * 50.0
        _, need_deltas, _, _ = model.decode_transition(states)
        mx.eval(need_deltas)
        bound = 0.5 + DRIFT_REGIME_THRESHOLD
        self.assertLessEqual(float(mx.max(mx.abs(need_deltas))), bound + 1e-6)

    def test_drift_path_alone_is_bounded_by_the_regime_threshold(self):
        model = self._model(split=True)
        # Zero the event path so only the drift path can contribute.
        model.next_needs.weight = mx.zeros_like(model.next_needs.weight)
        model.next_needs.bias = mx.zeros_like(model.next_needs.bias)
        states = mx.random.normal((1, 8, 16)) * 50.0
        _, need_deltas, _, _ = model.decode_transition(states)
        mx.eval(need_deltas)
        self.assertLessEqual(
            float(mx.max(mx.abs(need_deltas))), DRIFT_REGIME_THRESHOLD + 1e-6
        )


@unittest.skipIf(mx is None, "mlx is required")
class MetabolicDriftAuditTest(unittest.TestCase):
    def test_audit_reports_both_steps_and_the_oracle_bound(self):
        from dataclasses import replace as dc_replace

        trainer = OrganismTrainer(
            dc_replace(
                TrainingSmokeTest._config(TrainingSmokeTest),
                semantic_choice_childhood_steps=1,
                consume_options=True,
                inspect_options=True,
                episodic_binding_size=4,
                island=IslandConfig(
                    semantic_choice_horizon=40,
                    semantic_choice_objects=3,
                    semantic_choice_low_need=0.55,
                    semantic_choice_rounds=8,
                    semantic_choice_return_duration=6,
                ),
            )
        )
        result = audit_metabolic_drift_forecast(trainer.model, contexts=4)
        self.assertEqual(result["audited_contexts"], 4.0)
        for step in ("inspect", "return"):
            for need in ("food", "water", "energy", "health"):
                self.assertIn(f"{step}_{need}_bias", result)
                self.assertIn(f"{step}_{need}_absolute_error", result)
                self.assertGreaterEqual(
                    result[f"{step}_{need}_absolute_error"], 0.0
                )
        # The real observation and the oracle chain are properties of the
        # world, not of the model, so they hold for an untrained model too.
        self.assertEqual(result["urgent_index_survives_real_observation"], 1.0)
        self.assertEqual(result["urgent_index_survives_oracle_post_return"], 1.0)
        self.assertAlmostEqual(result["min_urgency_margin"], 0.2, places=6)


class HarnessArgumentTest(unittest.TestCase):
    @staticmethod
    def _parse(argv: list[str]):
        import sys
        from unittest import mock

        from homesocial.organism.harness import _parse_args

        with mock.patch.object(sys, "argv", ["harness", *argv]):
            return _parse_args()

    def test_branching_planner_flags_reach_the_namespace(self):
        args = self._parse(
            [
                "--self-model-planning",
                "--planning-horizon",
                "2",
                "--observation-branching-planning",
                "--persistent-information-reuses",
                "7",
                "--semantic-choice-return-duration",
                "6",
                "--episodic-binding-size",
                "16",
            ]
        )
        self.assertTrue(args.observation_branching_planning)
        self.assertEqual(args.persistent_information_reuses, 7)

    def test_branching_planner_requires_two_step_delayed_mechanics(self):
        for argv in (
            ["--observation-branching-planning", "--planning-horizon", "1"],
            [
                "--observation-branching-planning",
                "--planning-horizon",
                "2",
                "--episodic-binding-size",
                "16",
            ],
            [
                "--observation-branching-planning",
                "--planning-horizon",
                "2",
                "--semantic-choice-return-duration",
                "6",
            ],
        ):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                self._parse(argv)


if __name__ == "__main__":
    unittest.main()
