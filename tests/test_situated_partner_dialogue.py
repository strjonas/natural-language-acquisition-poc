import unittest

import mlx.core as mx
import numpy as np

from homesocial.option_world_model import OptionBranchDataset, OptionBranchSample
from homesocial.observations import observation_vector_size
from homesocial.recurrent_ac import RecurrentActorCritic, RecurrentConfig
from homesocial.situated_partner_dialogue import (
    EXTENDED_SITUATED_OPTION_NAMES,
    SituatedPartnerDataset,
    collect_online_adaptation_risk_samples,
    collect_situated_partner_dataset,
    collect_situated_self_model_rank_samples,
    _option_value,
    _partner_proposals,
    evaluate_online_partner_dialogue,
    evaluate_situated_partner_dialogue,
    fit_situated_self_model_calibrator,
    format_online_result,
    format_result,
    intervene_situated_partner_features,
    situated_partner_dataset_from_branches,
    train_online_recovery_policy,
    train_situated_self_model_rank,
    train_situated_partner_dialogue,
)


def _branch_dataset() -> OptionBranchDataset:
    groups = [
        (
            np.array([0.4, 0.8, 0.8, 0.8], dtype=np.float32),
            np.array(
                [
                    [0.9, 0.7, 0.4, 0.4],
                    [0.5, 0.9, 0.9, 0.9],
                    [0.65, 0.65, 0.95, 0.95],
                    [0.36, 0.75, 0.85, 0.85],
                    [0.35, 0.75, 0.75, 0.75],
                ],
                dtype=np.float32,
            ),
        ),
        (
            np.array([0.8, 0.8, 0.35, 0.8], dtype=np.float32),
            np.array(
                [
                    [0.85, 0.7, 0.4, 0.75],
                    [0.8, 0.85, 0.4, 0.75],
                    [0.7, 0.7, 0.95, 0.95],
                    [0.78, 0.78, 0.45, 0.85],
                    [0.75, 0.75, 0.3, 0.75],
                ],
                dtype=np.float32,
            ),
        ),
    ]
    samples = []
    for current, finals in groups:
        for option_label, final in enumerate(finals):
            samples.append(
                OptionBranchSample(
                    observations=mx.zeros((1, 3), dtype=mx.float32),
                    actions=mx.array([0], dtype=mx.int32),
                    next_observations=mx.zeros((1, 3), dtype=mx.float32),
                    next_needs=mx.array(final[None, :], dtype=mx.float32),
                    rewards=mx.array([0.0], dtype=mx.float32),
                    current_needs=mx.array(current, dtype=mx.float32),
                    option_label=option_label,
                    trend_label=0,
                )
            )
    return OptionBranchDataset(
        samples=tuple(samples),
        config=RecurrentConfig(),
        horizon=1,
    )


def _trainable_dataset(repeats: int = 8) -> SituatedPartnerDataset:
    base = situated_partner_dataset_from_branches(_branch_dataset())
    features = np.asarray(base.features)
    current = np.asarray(base.current_needs)
    final = np.asarray(base.final_needs)
    values = np.asarray(base.option_values)
    targets = np.asarray(base.target_options)
    lowest = np.asarray(base.current_lowest)
    return SituatedPartnerDataset(
        features=mx.array(np.tile(features, (repeats, 1, 1)), dtype=mx.float32),
        current_needs=mx.array(np.tile(current, (repeats, 1)), dtype=mx.float32),
        final_needs=mx.array(np.tile(final, (repeats, 1, 1)), dtype=mx.float32),
        option_values=mx.array(np.tile(values, (repeats, 1)), dtype=mx.float32),
        target_options=mx.array(np.tile(targets, repeats), dtype=mx.int32),
        current_lowest=mx.array(np.tile(lowest, repeats), dtype=mx.float32),
    )


class SituatedPartnerDialogueTests(unittest.TestCase):
    def test_dataset_groups_environment_branches_by_state(self):
        dataset = situated_partner_dataset_from_branches(_branch_dataset())

        self.assertEqual(dataset.features.shape, (2, 5, 17))
        self.assertEqual(dataset.option_values.shape, (2, 5))
        np.testing.assert_array_equal(
            np.asarray(dataset.target_options),
            np.array([2, 2], dtype=np.int32),
        )

    def test_dataset_filters_non_positive_oracle_opportunities(self):
        dataset = situated_partner_dataset_from_branches(
            _branch_dataset(),
            min_oracle_delta=0.3,
        )

        self.assertEqual(dataset.features.shape[0], 1)
        np.testing.assert_array_equal(
            np.asarray(dataset.target_options),
            np.array([2], dtype=np.int32),
        )

    def test_dataset_supports_trajectory_value_modes(self):
        dataset = situated_partner_dataset_from_branches(
            _branch_dataset(),
            value_mode="trajectory_mean",
        )

        self.assertEqual(dataset.option_values.shape, (2, 5))
        self.assertAlmostEqual(
            _option_value(
                np.array([[0.4, 0.8], [0.6, 0.7]], dtype=np.float32),
                value_mode="trajectory_min",
            ),
            0.4,
        )
        self.assertAlmostEqual(
            _option_value(
                np.array([[0.4, 0.8], [0.6, 0.7]], dtype=np.float32),
                value_mode="trajectory_mean",
            ),
            0.5,
        )
        with self.assertRaises(ValueError):
            situated_partner_dataset_from_branches(
                _branch_dataset(),
                value_mode="unknown",
            )

    def test_collect_extended_situated_option_dataset_runs(self):
        config = RecurrentConfig(max_steps=6, randomize_world=False)
        dataset = collect_situated_partner_dataset(
            config,
            episodes=1,
            seed=19,
            horizon=2,
            max_states=3,
            option_set="extended",
            value_mode="trajectory_mean",
        )

        self.assertEqual(dataset.option_names, EXTENDED_SITUATED_OPTION_NAMES)
        self.assertEqual(dataset.features.shape[-1], 20)
        self.assertEqual(
            dataset.option_values.shape[1],
            len(EXTENDED_SITUATED_OPTION_NAMES),
        )
        with self.assertRaises(ValueError):
            collect_situated_partner_dataset(
                config,
                episodes=1,
                seed=19,
                option_set="unknown",
            )

    def test_situated_self_model_rank_finetune_runs(self):
        config = RecurrentConfig(max_steps=5, randomize_world=False)
        samples = collect_situated_self_model_rank_samples(
            config,
            episodes=1,
            seed=23,
            horizon=2,
            max_samples=2,
            option_set="extended",
            value_mode="trajectory_mean",
        )
        model = RecurrentActorCritic(
            observation_vector_size(
                include_language=config.include_language_channel,
                include_object_kinds=config.include_object_kinds,
                body_dynamics_mode=config.body_dynamics_mode,
            ),
            hidden_size=8,
            action_size=8,
        )
        loss = train_situated_self_model_rank(
            model,
            samples,
            epochs=1,
            batch_size=2,
            seed=29,
        )

        self.assertEqual(len(samples), 2)
        self.assertEqual(
            samples[0].option_final_needs.shape,
            (len(EXTENDED_SITUATED_OPTION_NAMES), 4),
        )
        self.assertEqual(
            samples[0].option_values.shape,
            (len(EXTENDED_SITUATED_OPTION_NAMES),),
        )
        self.assertGreaterEqual(loss, 0.0)
        calibrator = fit_situated_self_model_calibrator(
            model,
            samples,
            option_names=EXTENDED_SITUATED_OPTION_NAMES,
            value_mode="trajectory_mean",
        )

        self.assertEqual(
            calibrator.final_scale.shape,
            (len(EXTENDED_SITUATED_OPTION_NAMES), 4),
        )
        self.assertEqual(
            calibrator.final_rmse.shape,
            (len(EXTENDED_SITUATED_OPTION_NAMES), 4),
        )
        self.assertEqual(
            calibrator.value_scale.shape,
            (len(EXTENDED_SITUATED_OPTION_NAMES),),
        )
        self.assertEqual(
            calibrator.value_rmse.shape,
            (len(EXTENDED_SITUATED_OPTION_NAMES),),
        )
        self.assertEqual(
            calibrator.value_reference_predictions.shape,
            (len(samples), len(EXTENDED_SITUATED_OPTION_NAMES)),
        )
        self.assertEqual(
            calibrator.value_reference_errors.shape,
            (len(samples), len(EXTENDED_SITUATED_OPTION_NAMES)),
        )

    def test_partner_proposals_use_partial_body_views(self):
        dataset = situated_partner_dataset_from_branches(_branch_dataset())

        partial = _partner_proposals(dataset, mode="partial_body")
        energy_safety = _partner_proposals(dataset, mode="partial_energy_safety")
        worst = _partner_proposals(dataset, mode="worst")

        np.testing.assert_array_equal(partial, np.array([0, 1], dtype=np.int32))
        np.testing.assert_array_equal(energy_safety, np.array([2, 2], dtype=np.int32))
        np.testing.assert_array_equal(worst, np.array([4, 4], dtype=np.int32))
        with self.assertRaises(ValueError):
            _partner_proposals(dataset, mode="unknown")

    def test_interventions_keep_targets_but_change_features(self):
        dataset = situated_partner_dataset_from_branches(_branch_dataset())
        intervened = intervene_situated_partner_features(
            dataset,
            intervention="reverse_delta_rank",
        )

        self.assertEqual(intervened.features.shape, dataset.features.shape)
        self.assertEqual(intervened.option_names, dataset.option_names)
        np.testing.assert_array_equal(
            np.asarray(intervened.target_options),
            np.asarray(dataset.target_options),
        )
        self.assertFalse(
            np.allclose(np.asarray(intervened.features), np.asarray(dataset.features))
        )
        with self.assertRaises(ValueError):
            intervene_situated_partner_features(dataset, intervention="unknown")

    def test_train_and_evaluate_runs(self):
        dataset = _trainable_dataset()
        trained = train_situated_partner_dialogue(
            dataset,
            hidden_size=12,
            receiver_size=12,
            epochs=3,
            batch_size=4,
            vocabulary_size=3,
            seed=11,
        )
        result = evaluate_situated_partner_dialogue(
            trained,
            dataset,
            model_control="situated_partial_body",
            intervention="original",
            partner_mode="partial_body",
        )

        self.assertEqual(result.samples, int(dataset.features.shape[0]))
        self.assertGreaterEqual(result.final_accuracy, 0.0)
        self.assertLessEqual(result.final_accuracy, 1.0)
        self.assertIn("situated_partial_body,original", format_result(result))

    def test_online_partner_dialogue_runs(self):
        dataset = _trainable_dataset()
        trained = train_situated_partner_dialogue(
            dataset,
            hidden_size=12,
            receiver_size=12,
            epochs=2,
            batch_size=4,
            vocabulary_size=3,
            seed=13,
        )
        config = RecurrentConfig(max_steps=4, randomize_world=False)
        result = evaluate_online_partner_dialogue(
            trained,
            config,
            episodes=2,
            seed=17,
            horizon=1,
            model_control="dialogue",
            partner_mode="partial_body",
        )

        self.assertEqual(result.episodes, 2)
        self.assertGreaterEqual(result.mean_steps, 1.0)
        self.assertGreaterEqual(result.override_rate, 0.0)
        self.assertGreaterEqual(result.termination_rate, 0.0)
        self.assertIn("dialogue,original,2", format_online_result(result))
        adaptive = evaluate_online_partner_dialogue(
            trained,
            config,
            episodes=2,
            seed=17,
            horizon=1,
            model_control="adaptive_dialogue",
            partner_mode="partial_body",
            online_adaptation_steps=1,
        )

        self.assertEqual(adaptive.episodes, 2)
        self.assertIn("adaptive_dialogue,original,2", format_online_result(adaptive))
        gated_adaptive = evaluate_online_partner_dialogue(
            trained,
            config,
            episodes=1,
            seed=17,
            horizon=1,
            model_control="adaptive_dialogue",
            partner_mode="partial_body",
            online_adaptation_steps=1,
            online_adaptation_kl_weight=0.5,
            online_adaptation_min_value_gap=10.0,
            online_adaptation_choice_guard=True,
            online_adaptation_local=True,
        )

        self.assertEqual(gated_adaptive.episodes, 1)
        self_model = RecurrentActorCritic(
            observation_vector_size(
                include_language=config.include_language_channel,
                include_object_kinds=config.include_object_kinds,
                body_dynamics_mode=config.body_dynamics_mode,
            ),
            hidden_size=8,
            action_size=8,
        )
        learned = evaluate_online_partner_dialogue(
            trained,
            config,
            episodes=1,
            seed=17,
            horizon=1,
            model_control="dialogue",
            partner_mode="partial_body",
            online_self_model_source="learned",
            online_self_model=self_model,
            online_self_model_config=config,
        )
        calibration_samples = collect_situated_self_model_rank_samples(
            config,
            episodes=1,
            seed=31,
            horizon=1,
            max_samples=2,
            option_set="base",
        )
        calibrated = evaluate_online_partner_dialogue(
            trained,
            config,
            episodes=1,
            seed=17,
            horizon=1,
            model_control="dialogue",
            partner_mode="partial_body",
            online_self_model_source="learned",
            online_self_model=self_model,
            online_self_model_config=config,
            online_self_model_calibrator=fit_situated_self_model_calibrator(
                self_model,
                calibration_samples,
                option_names=trained.option_names,
                value_mode="final_lowest",
            ),
            online_self_calibration_mode="value_knn_lcb",
            online_self_calibration_uncertainty_scale=0.5,
            online_self_calibration_knn=1,
        )

        self.assertEqual(learned.episodes, 1)
        self.assertEqual(calibrated.episodes, 1)
        self.assertIn("dialogue,original,1", format_online_result(learned))
        risk_calibrator = collect_online_adaptation_risk_samples(
            trained,
            config,
            episodes=1,
            seed=37,
            horizon=1,
            max_samples=4,
            partner_mode="partial_body",
            online_self_model_source="learned",
            online_self_model=self_model,
            online_self_model_config=config,
        )
        self.assertEqual(risk_calibrator.features.shape[0], 4)
        self.assertEqual(risk_calibrator.risks.shape, (4,))
        self.assertEqual(risk_calibrator.risk_label, "branch")
        rollout_risk_calibrator = collect_online_adaptation_risk_samples(
            trained,
            config,
            episodes=1,
            seed=37,
            horizon=1,
            max_samples=4,
            partner_mode="partial_body",
            online_self_model_source="learned",
            online_self_model=self_model,
            online_self_model_config=config,
            online_risk_label="rollout_min",
            online_risk_rollout_steps=2,
            online_risk_option_commit_steps=1,
        )
        self.assertEqual(rollout_risk_calibrator.risks.shape, (4,))
        self.assertEqual(rollout_risk_calibrator.risk_label, "rollout_min")
        self.assertEqual(rollout_risk_calibrator.rollout_steps, 2)
        self.assertTrue(np.all(rollout_risk_calibrator.risks >= 0.0))
        ridge_risk_calibrator = collect_online_adaptation_risk_samples(
            trained,
            config,
            episodes=1,
            seed=37,
            horizon=1,
            max_samples=4,
            partner_mode="partial_body",
            online_self_model_source="learned",
            online_self_model=self_model,
            online_self_model_config=config,
            online_risk_model="ridge",
            online_risk_ridge=0.01,
        )
        self.assertIsNotNone(ridge_risk_calibrator.ridge_weights)
        self.assertEqual(
            ridge_risk_calibrator.ridge_weights.shape,
            (ridge_risk_calibrator.features.shape[1] + 1,),
        )
        mlp_risk_calibrator = collect_online_adaptation_risk_samples(
            trained,
            config,
            episodes=1,
            seed=37,
            horizon=1,
            max_samples=4,
            partner_mode="partial_body",
            online_self_model_source="learned",
            online_self_model=self_model,
            online_self_model_config=config,
            online_risk_model="mlp",
            online_risk_hidden_size=4,
            online_risk_epochs=1,
            online_risk_batch_size=4,
        )
        self.assertIsNotNone(mlp_risk_calibrator.risk_critic)
        risk_gated = evaluate_online_partner_dialogue(
            trained,
            config,
            episodes=1,
            seed=17,
            horizon=1,
            model_control="adaptive_dialogue",
            partner_mode="partial_body",
            online_adaptation_steps=1,
            online_adaptation_local=True,
            online_self_model_source="learned",
            online_self_model=self_model,
            online_self_model_config=config,
            online_adaptation_risk_calibrator=risk_calibrator,
            online_adaptation_risk_threshold=0.0,
            online_adaptation_risk_knn=1,
            online_adaptation_risk_penalty=0.5,
            online_adaptation_risk_fallback="need_recovery",
        )
        self.assertEqual(risk_gated.episodes, 1)
        message_recovery = evaluate_online_partner_dialogue(
            trained,
            config,
            episodes=1,
            seed=17,
            horizon=1,
            model_control="adaptive_dialogue",
            partner_mode="partial_body",
            online_adaptation_steps=1,
            online_adaptation_local=True,
            online_self_model_source="learned",
            online_self_model=self_model,
            online_self_model_config=config,
            online_adaptation_risk_calibrator=risk_calibrator,
            online_adaptation_risk_threshold=0.0,
            online_adaptation_risk_knn=1,
            online_adaptation_risk_fallback="message_recovery",
        )
        self.assertEqual(message_recovery.episodes, 1)
        recovery_policy = train_online_recovery_policy(
            trained,
            config,
            self_model,
            episodes=1,
            seed=41,
            horizon=1,
            max_samples=4,
            rollout_steps=2,
            hidden_size=4,
            epochs=1,
            batch_size=4,
        )
        self.assertEqual(recovery_policy.option_names, trained.option_names)
        self.assertEqual(recovery_policy.source, "state_policy")
        self.assertEqual(recovery_policy.feature_mode, "history")
        self.assertEqual(recovery_policy.regret_weight, 1.0)
        self.assertEqual(recovery_policy.floor_weight, 1.0)
        outcome_recovery_policy = train_online_recovery_policy(
            trained,
            config,
            self_model,
            episodes=1,
            seed=42,
            horizon=1,
            max_samples=4,
            rollout_steps=2,
            feature_mode="history_outcome",
            hidden_size=4,
            epochs=1,
            batch_size=4,
        )
        self.assertEqual(outcome_recovery_policy.feature_mode, "history_outcome")
        regret_recovery_policy = train_online_recovery_policy(
            trained,
            config,
            self_model,
            episodes=1,
            seed=43,
            horizon=1,
            max_samples=4,
            rollout_steps=2,
            feature_mode="history_outcome",
            label="rollout_mean_regret",
            regret_weight=0.5,
            hidden_size=4,
            epochs=1,
            batch_size=4,
        )
        self.assertEqual(regret_recovery_policy.label, "rollout_mean_regret")
        self.assertEqual(regret_recovery_policy.regret_weight, 0.5)
        floor_recovery_policy = train_online_recovery_policy(
            trained,
            config,
            self_model,
            episodes=1,
            seed=44,
            horizon=1,
            max_samples=4,
            rollout_steps=2,
            feature_mode="history_outcome",
            label="rollout_mean_floor_regret",
            regret_weight=0.1,
            floor_weight=0.5,
            hidden_size=4,
            epochs=1,
            batch_size=4,
        )
        self.assertEqual(floor_recovery_policy.label, "rollout_mean_floor_regret")
        self.assertEqual(floor_recovery_policy.regret_weight, 0.1)
        self.assertEqual(floor_recovery_policy.floor_weight, 0.5)
        temporal_recovery = evaluate_online_partner_dialogue(
            trained,
            config,
            episodes=1,
            seed=17,
            horizon=1,
            model_control="adaptive_dialogue",
            partner_mode="partial_body",
            online_adaptation_steps=1,
            online_adaptation_local=True,
            online_self_model_source="learned",
            online_self_model=self_model,
            online_self_model_config=config,
            online_adaptation_risk_calibrator=risk_calibrator,
            online_adaptation_risk_threshold=0.0,
            online_adaptation_risk_knn=1,
            online_adaptation_risk_fallback="temporal_recovery",
            online_recovery_policy=recovery_policy,
        )
        self.assertEqual(temporal_recovery.episodes, 1)
        outcome_temporal_recovery = evaluate_online_partner_dialogue(
            trained,
            config,
            episodes=1,
            seed=18,
            horizon=1,
            intervention="zero_outcome",
            model_control="adaptive_dialogue",
            partner_mode="partial_body",
            online_adaptation_steps=1,
            online_adaptation_local=True,
            online_self_model_source="learned",
            online_self_model=self_model,
            online_self_model_config=config,
            online_adaptation_risk_calibrator=risk_calibrator,
            online_adaptation_risk_threshold=0.0,
            online_adaptation_risk_knn=1,
            online_adaptation_risk_fallback="temporal_recovery",
            online_recovery_policy=outcome_recovery_policy,
        )
        self.assertEqual(outcome_temporal_recovery.episodes, 1)
        constrained_temporal_recovery = evaluate_online_partner_dialogue(
            trained,
            config,
            episodes=1,
            seed=19,
            horizon=1,
            intervention="zero_outcome",
            model_control="adaptive_dialogue",
            partner_mode="partial_body",
            online_adaptation_steps=1,
            online_adaptation_local=True,
            online_self_model_source="learned",
            online_self_model=self_model,
            online_self_model_config=config,
            online_adaptation_risk_calibrator=risk_calibrator,
            online_adaptation_risk_threshold=0.0,
            online_adaptation_risk_knn=1,
            online_adaptation_risk_fallback="constrained_temporal_recovery",
            online_recovery_policy=outcome_recovery_policy,
            online_recovery_floor_margin=0.1,
        )
        self.assertEqual(constrained_temporal_recovery.episodes, 1)
        with self.assertRaises(ValueError):
            evaluate_online_partner_dialogue(
                trained,
                config,
                episodes=1,
                seed=17,
                model_control="unknown",
            )
        with self.assertRaises(ValueError):
            evaluate_online_partner_dialogue(
                trained,
                config,
                episodes=1,
                seed=17,
                option_commit_steps=0,
            )
        with self.assertRaises(ValueError):
            evaluate_online_partner_dialogue(
                trained,
                config,
                episodes=1,
                seed=17,
                value_mode="unknown",
            )
        with self.assertRaises(ValueError):
            evaluate_online_partner_dialogue(
                trained,
                config,
                episodes=1,
                seed=17,
                model_control="adaptive_dialogue",
            )
        with self.assertRaises(ValueError):
            evaluate_online_partner_dialogue(
                trained,
                config,
                episodes=1,
                seed=17,
                online_adaptation_steps=-1,
            )
        with self.assertRaises(ValueError):
            evaluate_online_partner_dialogue(
                trained,
                config,
                episodes=1,
                seed=17,
                online_adaptation_kl_weight=-0.1,
            )
        with self.assertRaises(ValueError):
            evaluate_online_partner_dialogue(
                trained,
                config,
                episodes=1,
                seed=17,
                online_adaptation_min_value_gap=-0.1,
            )
        with self.assertRaises(ValueError):
            evaluate_online_partner_dialogue(
                trained,
                config,
                episodes=1,
                seed=17,
                online_adaptation_choice_guard_margin=-0.1,
            )
        with self.assertRaises(ValueError):
            evaluate_online_partner_dialogue(
                trained,
                config,
                episodes=1,
                seed=17,
                online_self_calibration_mode="unknown",
            )
        with self.assertRaises(ValueError):
            evaluate_online_partner_dialogue(
                trained,
                config,
                episodes=1,
                seed=17,
                online_self_calibration_uncertainty_scale=-0.1,
            )
        with self.assertRaises(ValueError):
            evaluate_online_partner_dialogue(
                trained,
                config,
                episodes=1,
                seed=17,
                online_self_calibration_knn=0,
            )
        with self.assertRaises(ValueError):
            evaluate_online_partner_dialogue(
                trained,
                config,
                episodes=1,
                seed=17,
                online_adaptation_risk_threshold=-0.1,
            )
        with self.assertRaises(ValueError):
            evaluate_online_partner_dialogue(
                trained,
                config,
                episodes=1,
                seed=17,
                online_adaptation_risk_knn=0,
            )
        with self.assertRaises(ValueError):
            collect_online_adaptation_risk_samples(
                trained,
                config,
                episodes=1,
                seed=37,
                online_risk_model="unknown",
            )
        with self.assertRaises(ValueError):
            collect_online_adaptation_risk_samples(
                trained,
                config,
                episodes=1,
                seed=37,
                online_risk_ridge=-0.1,
            )
        with self.assertRaises(ValueError):
            collect_online_adaptation_risk_samples(
                trained,
                config,
                episodes=1,
                seed=37,
                online_risk_hidden_size=0,
            )
        with self.assertRaises(ValueError):
            collect_online_adaptation_risk_samples(
                trained,
                config,
                episodes=1,
                seed=37,
                online_risk_learning_rate=0.0,
            )
        with self.assertRaises(ValueError):
            collect_online_adaptation_risk_samples(
                trained,
                config,
                episodes=1,
                seed=37,
                online_risk_batch_size=0,
            )
        with self.assertRaises(ValueError):
            collect_online_adaptation_risk_samples(
                trained,
                config,
                episodes=1,
                seed=37,
                online_risk_label="unknown",
            )
        with self.assertRaises(ValueError):
            collect_online_adaptation_risk_samples(
                trained,
                config,
                episodes=1,
                seed=37,
                online_risk_rollout_steps=0,
            )
        with self.assertRaises(ValueError):
            collect_online_adaptation_risk_samples(
                trained,
                config,
                episodes=1,
                seed=37,
                online_risk_option_commit_steps=0,
            )
        with self.assertRaises(ValueError):
            evaluate_online_partner_dialogue(
                trained,
                config,
                episodes=1,
                seed=17,
                online_adaptation_risk_penalty=-0.1,
            )
        with self.assertRaises(ValueError):
            evaluate_online_partner_dialogue(
                trained,
                config,
                episodes=1,
                seed=17,
                model_control="adaptive_dialogue",
                online_adaptation_steps=1,
                online_adaptation_risk_penalty=0.5,
            )
        with self.assertRaises(ValueError):
            evaluate_online_partner_dialogue(
                trained,
                config,
                episodes=1,
                seed=17,
                online_adaptation_risk_fallback="unknown",
            )
        with self.assertRaises(ValueError):
            evaluate_online_partner_dialogue(
                trained,
                config,
                episodes=1,
                seed=17,
                model_control="adaptive_dialogue",
                online_adaptation_steps=1,
                online_adaptation_risk_fallback="temporal_recovery",
            )
        with self.assertRaises(ValueError):
            evaluate_online_partner_dialogue(
                trained,
                config,
                episodes=1,
                seed=17,
                model_control="adaptive_dialogue",
                online_adaptation_steps=1,
                online_recovery_floor_margin=-0.1,
            )
        with self.assertRaises(ValueError):
            train_online_recovery_policy(
                trained,
                config,
                self_model,
                episodes=1,
                seed=41,
                max_samples=0,
            )
        with self.assertRaises(ValueError):
            train_online_recovery_policy(
                trained,
                config,
                self_model,
                episodes=1,
                seed=41,
                source="unknown",
            )
        with self.assertRaises(ValueError):
            train_online_recovery_policy(
                trained,
                config,
                self_model,
                episodes=1,
                seed=41,
                feature_mode="unknown",
            )
        with self.assertRaises(ValueError):
            train_online_recovery_policy(
                trained,
                config,
                self_model,
                episodes=1,
                seed=41,
                label="unknown",
            )
        with self.assertRaises(ValueError):
            train_online_recovery_policy(
                trained,
                config,
                self_model,
                episodes=1,
                seed=41,
                regret_weight=-0.1,
            )
        with self.assertRaises(ValueError):
            train_online_recovery_policy(
                trained,
                config,
                self_model,
                episodes=1,
                seed=41,
                floor_weight=-0.1,
            )
        with self.assertRaises(ValueError):
            train_online_recovery_policy(
                trained,
                config,
                self_model,
                episodes=1,
                seed=41,
                source="risky_adaptive",
            )
        with self.assertRaises(ValueError):
            evaluate_online_partner_dialogue(
                trained,
                config,
                episodes=1,
                seed=17,
                online_self_model_source="learned",
            )

    def test_unknown_train_partner_mode_fails(self):
        with self.assertRaises(ValueError):
            train_situated_partner_dialogue(
                _trainable_dataset(repeats=1),
                hidden_size=12,
                receiver_size=12,
                epochs=1,
                partner_mode="unknown",
            )
        with self.assertRaises(ValueError):
            train_situated_partner_dialogue(
                _trainable_dataset(repeats=1),
                hidden_size=12,
                receiver_size=12,
                epochs=1,
                target_weight=-0.1,
            )


if __name__ == "__main__":
    unittest.main()
