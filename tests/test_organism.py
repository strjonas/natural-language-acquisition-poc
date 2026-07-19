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
        ACTIONS,
        OrganismConfig,
        OrganismTrainer,
        audit_self_model_actions,
        available_action_mask,
        compute_gae,
        evaluate_organism,
        execute_agent_action,
        planned_policy_logits,
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
        self.assertIn("consume_attempts_per_episode", stats)
        self.assertIn("resource_consumes_per_episode", stats)
        self.assertIn("option_decisions_per_episode", stats)
        self.assertGreaterEqual(stats["mean_steps"], 1.0)

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


if __name__ == "__main__":
    unittest.main()
