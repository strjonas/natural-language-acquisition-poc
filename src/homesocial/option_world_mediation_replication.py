from __future__ import annotations

import argparse
from dataclasses import dataclass, replace

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .env import RESOURCE_ECOLOGIES, Action
from .imitation import load_checkpoint
from .observations import observation_vector_size
from .option_counterfactual_language import STATE_POLICIES
from .option_feature_intervention import OPTION_FEATURE_INTERVENTIONS
from .option_mediation import (
    OPTION_MEDIATION_BALANCE_TARGETS,
    OPTION_MEDIATION_FEATURE_MODES,
    OPTION_MEDIATION_SLOTS,
    OPTION_MEDIATION_TARGET_MODES,
    OPTION_MEDIATION_VOCABULARY,
    collect_option_mediation_source,
    evaluate_option_mediator,
    evaluate_option_population_mediator,
    evaluate_option_population_receiver_transfer,
    evaluate_option_receiver_transfer,
    intervene_option_mediation_features,
    majority_option_result,
    option_mediation_dataset_with_targets,
    option_mediation_dataset_from_source,
    predicted_future_option_choices,
    predicted_future_option_scores,
    predicted_future_option_result,
    staged_option_population_from_mediator,
    train_option_population_mediator,
    train_option_receiver_for_sender,
    train_option_receiver_for_population_sender,
    train_option_mediator,
)
from .option_world_model import (
    _last_valid_hidden,
    _option_world_loss,
    collect_option_branch_dataset,
    evaluate_option_world_model,
    pad_option_branch_samples,
    train_option_world_model,
)
from .recurrent_ac import RecurrentActorCritic


@dataclass(frozen=True)
class OptionWorldMediationReplicationRow:
    seed: int
    model_control: str
    intervention: str
    samples: int
    choice_accuracy: float
    mean_regret: float
    mean_chosen_delta: float
    mean_oracle_delta: float
    message_codes_used: int
    message_code_entropy: float
    dominant_message_code_fraction: float
    target_code_mutual_information: float
    choice_code_mutual_information: float
    positive_delta_code_mutual_information: float
    relative_value_code_mutual_information: float
    message_patterns_used: int
    reused_message_pattern_fraction: float
    target_pattern_mutual_information: float
    choice_pattern_mutual_information: float
    positive_delta_pattern_mutual_information: float
    relative_value_pattern_mutual_information: float
    world_final_need_mse: float
    world_trend_accuracy: float
    trained_world_final_need_mse_before: float
    trained_world_final_need_mse_after: float
    trained_world_trend_before: float
    trained_world_trend_after: float


@dataclass(frozen=True)
class OptionRankBatch:
    observations: mx.array
    step_masks: mx.array
    observation_lengths: mx.array
    actions: mx.array
    action_masks: mx.array
    target_options: mx.array


def header() -> str:
    return (
        "seed,model_control,intervention,samples,choice_accuracy,mean_regret,"
        "mean_chosen_delta,mean_oracle_delta,message_codes_used,"
        "message_code_entropy,dominant_message_code_fraction,"
        "target_code_mutual_information,choice_code_mutual_information,"
        "positive_delta_code_mutual_information,relative_value_code_mutual_information,"
        "message_patterns_used,reused_message_pattern_fraction,"
        "target_pattern_mutual_information,choice_pattern_mutual_information,"
        "positive_delta_pattern_mutual_information,"
        "relative_value_pattern_mutual_information,"
        "world_final_need_mse,world_trend_accuracy,trained_world_final_need_mse_before,"
        "trained_world_final_need_mse_after,trained_world_trend_before,"
        "trained_world_trend_after"
    )


def format_row(row: OptionWorldMediationReplicationRow) -> str:
    return ",".join(
        [
            str(row.seed),
            row.model_control,
            row.intervention,
            str(row.samples),
            f"{row.choice_accuracy:.4f}",
            f"{row.mean_regret:.6f}",
            f"{row.mean_chosen_delta:.6f}",
            f"{row.mean_oracle_delta:.6f}",
            str(row.message_codes_used),
            f"{row.message_code_entropy:.6f}",
            f"{row.dominant_message_code_fraction:.6f}",
            f"{row.target_code_mutual_information:.6f}",
            f"{row.choice_code_mutual_information:.6f}",
            f"{row.positive_delta_code_mutual_information:.6f}",
            f"{row.relative_value_code_mutual_information:.6f}",
            str(row.message_patterns_used),
            f"{row.reused_message_pattern_fraction:.6f}",
            f"{row.target_pattern_mutual_information:.6f}",
            f"{row.choice_pattern_mutual_information:.6f}",
            f"{row.positive_delta_pattern_mutual_information:.6f}",
            f"{row.relative_value_pattern_mutual_information:.6f}",
            f"{row.world_final_need_mse:.6f}",
            f"{row.world_trend_accuracy:.4f}",
            f"{row.trained_world_final_need_mse_before:.6f}",
            f"{row.trained_world_final_need_mse_after:.6f}",
            f"{row.trained_world_trend_before:.4f}",
            f"{row.trained_world_trend_after:.4f}",
        ]
    )


def resolved_option_action_noises(args: argparse.Namespace) -> tuple[float, float]:
    shared_noise = float(args.option_action_noise)
    world_noise = getattr(args, "world_option_action_noise", None)
    mediation_noise = getattr(args, "mediation_option_action_noise", None)
    return (
        shared_noise if world_noise is None else float(world_noise),
        shared_noise if mediation_noise is None else float(mediation_noise),
    )


def train_option_rank_finetune(
    model: RecurrentActorCritic,
    source,
    *,
    dynamics_samples=None,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    temperature: float,
    dynamics_weight: float = 0.0,
    step_needs_weight: float = 2.0,
    final_needs_weight: float = 10.0,
    observation_prediction_weight: float = 0.03,
    reward_prediction_weight: float = 0.4,
    seed: int,
) -> float:
    if epochs <= 0:
        return 0.0
    if not source.samples:
        raise ValueError("Cannot rank-finetune on an empty option source.")
    rng = np.random.default_rng(seed)
    mx.random.seed(seed)
    optimizer = optim.Adam(learning_rate=learning_rate)
    samples = list(source.samples)
    replay_samples = list(dynamics_samples or [])
    if dynamics_weight > 0.0 and not replay_samples:
        raise ValueError("Dynamics replay requires non-empty dynamics samples.")
    final_loss = 0.0

    def loss_fn(
        observations: mx.array,
        step_masks: mx.array,
        observation_lengths: mx.array,
        actions: mx.array,
        action_masks: mx.array,
        target_options: mx.array,
    ) -> mx.array:
        scores = _grouped_predicted_lowest_scores(
            model,
            observations,
            step_masks,
            observation_lengths,
            actions,
            action_masks,
        )
        logits = scores / max(temperature, 1e-6)
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        selected = mx.sum(log_probs * mx.eye(scores.shape[1])[target_options], axis=-1)
        return -mx.mean(selected)

    def replay_loss_fn(
        observations: mx.array,
        step_masks: mx.array,
        observation_lengths: mx.array,
        actions: mx.array,
        action_masks: mx.array,
        action_lengths: mx.array,
        next_observations: mx.array,
        next_needs: mx.array,
        rewards: mx.array,
    ) -> mx.array:
        return dynamics_weight * _option_world_loss(
            model,
            observations,
            step_masks,
            observation_lengths,
            actions,
            action_masks,
            action_lengths,
            next_observations,
            next_needs,
            rewards,
            step_needs_weight,
            final_needs_weight,
            observation_prediction_weight,
            reward_prediction_weight,
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    replay_loss_and_grad = nn.value_and_grad(model, replay_loss_fn)
    for _epoch in range(max(1, epochs)):
        rng.shuffle(samples)
        if replay_samples:
            rng.shuffle(replay_samples)
        for start in range(0, len(samples), max(1, batch_size)):
            batch = pad_option_rank_samples(samples[start : start + max(1, batch_size)])
            loss, grads = loss_and_grad(
                batch.observations,
                batch.step_masks,
                batch.observation_lengths,
                batch.actions,
                batch.action_masks,
                batch.target_options,
            )
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            final_loss = float(loss)
            if dynamics_weight > 0.0:
                replay_start = start % len(replay_samples)
                replay_batch = _cyclic_sample_slice(
                    replay_samples,
                    replay_start,
                    max(1, batch_size),
                )
                dynamics_batch = pad_option_branch_samples(replay_batch)
                replay_loss, replay_grads = replay_loss_and_grad(
                    dynamics_batch.observations,
                    dynamics_batch.step_masks,
                    dynamics_batch.observation_lengths,
                    dynamics_batch.actions,
                    dynamics_batch.action_masks,
                    dynamics_batch.action_lengths,
                    dynamics_batch.next_observations,
                    dynamics_batch.next_needs,
                    dynamics_batch.rewards,
                )
                optimizer.update(model, replay_grads)
                mx.eval(model.parameters(), optimizer.state, replay_loss)
    return final_loss


def pad_option_rank_samples(samples) -> OptionRankBatch:
    if not samples:
        raise ValueError("Cannot pad an empty option-rank batch.")
    batch_size = len(samples)
    option_count = len(samples[0].option_actions)
    max_observation_length = max(len(sample.history) for sample in samples)
    max_action_length = max(
        len(actions)
        for sample in samples
        for actions in sample.option_actions
    )
    input_size = samples[0].history[0].shape[-1]
    observations = np.zeros(
        (batch_size, max_observation_length, input_size),
        dtype=np.float32,
    )
    step_masks = np.zeros((batch_size, max_observation_length), dtype=np.float32)
    observation_lengths = np.zeros(batch_size, dtype=np.int32)
    actions = np.zeros(
        (batch_size, option_count, max_action_length),
        dtype=np.int32,
    )
    action_masks = np.zeros(
        (batch_size, option_count, max_action_length),
        dtype=np.float32,
    )
    target_options = np.zeros(batch_size, dtype=np.int32)

    for index, sample in enumerate(samples):
        observation_length = len(sample.history)
        observations[index, :observation_length] = np.stack(sample.history)
        step_masks[index, :observation_length] = 1.0
        observation_lengths[index] = observation_length
        for option_index, option_actions in enumerate(sample.option_actions):
            action_length = len(option_actions)
            actions[index, option_index, :action_length] = np.asarray(
                option_actions,
                dtype=np.int32,
            )
            action_masks[index, option_index, :action_length] = 1.0
        target_options[index] = sample.target_option

    return OptionRankBatch(
        observations=mx.array(observations, dtype=mx.float32),
        step_masks=mx.array(step_masks, dtype=mx.float32),
        observation_lengths=mx.array(observation_lengths, dtype=mx.int32),
        actions=mx.array(actions, dtype=mx.int32),
        action_masks=mx.array(action_masks, dtype=mx.float32),
        target_options=mx.array(target_options, dtype=mx.int32),
    )


def _cyclic_sample_slice(samples: list, start: int, count: int) -> list:
    return [samples[(start + offset) % len(samples)] for offset in range(count)]


def _grouped_predicted_lowest_scores(
    model: RecurrentActorCritic,
    observations: mx.array,
    step_masks: mx.array,
    observation_lengths: mx.array,
    actions: mx.array,
    action_masks: mx.array,
) -> mx.array:
    hidden = model.hidden_states(observations)
    initial = mx.stop_gradient(
        _last_valid_hidden(hidden, observation_lengths, step_masks)
    )
    option_scores = []
    for option_index in range(actions.shape[1]):
        state = initial
        final_needs = mx.sigmoid(model.next_needs(state))
        for step in range(actions.shape[2]):
            action_features = mx.eye(model.action_size)[actions[:, option_index, step]]
            x = mx.concatenate([state, action_features], axis=-1)
            x = nn.relu(model.transition_norm(model.transition(x)))
            next_state = x + nn.relu(model.transition_state(x))
            next_needs = mx.sigmoid(model.next_needs(next_state))
            mask = action_masks[:, option_index, step : step + 1]
            state = mx.where(mask > 0.0, next_state, state)
            final_needs = mx.where(mask > 0.0, next_needs, final_needs)
        option_scores.append(mx.min(final_needs, axis=-1))
    return mx.stack(option_scores, axis=1)


def main() -> None:
    args = _parse_args()
    print(header())
    for seed in args.seeds:
        for row in run_replication_seed(args, seed=seed):
            print(format_row(row))


def run_replication_seed(
    args: argparse.Namespace,
    *,
    seed: int,
) -> list[OptionWorldMediationReplicationRow]:
    trained_base, config = load_checkpoint(args.checkpoint)
    if args.renewable_resources:
        config = replace(config, renewable_resources=True)
    if args.resource_ecology is not None:
        config = replace(config, resource_ecology=args.resource_ecology)
    world_noise, mediation_noise = resolved_option_action_noises(args)

    train_branches = collect_option_branch_dataset(
        config,
        episodes=args.world_train_episodes,
        seed=seed,
        teacher_mode=args.teacher_mode,
        horizon=args.horizon,
        state_policy=args.state_policy,
        max_samples=args.max_world_train_samples,
        option_action_noise=world_noise,
    )
    eval_branches = collect_option_branch_dataset(
        config,
        episodes=args.world_eval_episodes,
        seed=seed + 10_000,
        teacher_mode=args.teacher_mode,
        horizon=args.horizon,
        state_policy=args.state_policy,
        max_samples=args.max_world_eval_samples,
        option_action_noise=world_noise,
    )
    before_world = evaluate_option_world_model(
        trained_base,
        eval_branches,
        batch_size=args.world_eval_batch_size,
    )
    train_option_world_model(
        trained_base,
        config,
        train_branches,
        checkpoint_path=None,
        epochs=args.world_epochs,
        batch_size=args.world_batch_size,
        learning_rate=args.world_learning_rate,
        seed=seed,
        step_needs_weight=args.step_needs_weight,
        final_needs_weight=args.final_needs_weight,
        observation_prediction_weight=args.observation_prediction_weight,
        reward_prediction_weight=args.reward_prediction_weight,
    )
    after_world = evaluate_option_world_model(
        trained_base,
        eval_branches,
        batch_size=args.world_eval_batch_size,
    )

    train_source = collect_option_mediation_source(
        config,
        episodes=args.mediation_train_episodes,
        seed=seed,
        teacher_mode=args.teacher_mode,
        horizon=args.horizon,
        state_policy=args.state_policy,
        balance_target=args.mediation_balance_target,
        max_states=args.max_mediation_train_states,
        min_value_gap=args.min_value_gap,
        option_action_noise=mediation_noise,
    )
    eval_source = collect_option_mediation_source(
        config,
        episodes=args.mediation_eval_episodes,
        seed=seed + 10_000,
        teacher_mode=args.teacher_mode,
        horizon=args.horizon,
        state_policy=args.state_policy,
        balance_target=args.mediation_balance_target,
        max_states=args.max_mediation_eval_states,
        min_value_gap=args.min_value_gap,
        option_action_noise=mediation_noise,
    )
    train_option_rank_finetune(
        trained_base,
        train_source,
        dynamics_samples=train_branches.samples,
        epochs=args.rank_finetune_epochs,
        batch_size=args.rank_finetune_batch_size,
        learning_rate=args.rank_finetune_learning_rate,
        temperature=args.rank_finetune_temperature,
        dynamics_weight=args.rank_finetune_dynamics_weight,
        step_needs_weight=args.step_needs_weight,
        final_needs_weight=args.final_needs_weight,
        observation_prediction_weight=args.observation_prediction_weight,
        reward_prediction_weight=args.reward_prediction_weight,
        seed=seed + 40_000,
    )
    if args.rank_finetune_epochs > 0:
        after_world = evaluate_option_world_model(
            trained_base,
            eval_branches,
            batch_size=args.world_eval_batch_size,
        )

    rows = _mediation_rows(
        trained_base,
        train_source,
        eval_source,
        seed=seed,
        model_control="trained",
        args=args,
        world_final_need_mse=after_world.final_need_mse,
        world_trend_accuracy=after_world.trend_accuracy,
        before_world_final_need_mse=before_world.final_need_mse,
        after_world_final_need_mse=after_world.final_need_mse,
        before_world_trend_accuracy=before_world.trend_accuracy,
        after_world_trend_accuracy=after_world.trend_accuracy,
        include_target_majority=True,
    )

    if args.random_model_control:
        mx.random.seed(seed)
        random_base = RecurrentActorCritic(
            observation_vector_size(
                include_language=config.include_language_channel,
                include_object_kinds=config.include_object_kinds,
                body_dynamics_mode=config.body_dynamics_mode,
            ),
            config.hidden_size,
            len(Action),
        )
        random_world = evaluate_option_world_model(
            random_base,
            eval_branches,
            batch_size=args.world_eval_batch_size,
        )
        rows.extend(
            _mediation_rows(
                random_base,
                train_source,
                eval_source,
                seed=seed,
                model_control="random",
                args=args,
                world_final_need_mse=random_world.final_need_mse,
                world_trend_accuracy=random_world.trend_accuracy,
                before_world_final_need_mse=before_world.final_need_mse,
                after_world_final_need_mse=after_world.final_need_mse,
                before_world_trend_accuracy=before_world.trend_accuracy,
                after_world_trend_accuracy=after_world.trend_accuracy,
                include_target_majority=False,
            )
        )
    return rows


def _mediation_rows(
    base_model: RecurrentActorCritic,
    train_source,
    eval_source,
    *,
    seed: int,
    model_control: str,
    args: argparse.Namespace,
    world_final_need_mse: float,
    world_trend_accuracy: float,
    before_world_final_need_mse: float,
    after_world_final_need_mse: float,
    before_world_trend_accuracy: float,
    after_world_trend_accuracy: float,
    include_target_majority: bool,
) -> list[OptionWorldMediationReplicationRow]:
    train_dataset = option_mediation_dataset_from_source(
        train_source,
        base_model,
        feature_mode=args.feature_mode,
    )
    eval_dataset = option_mediation_dataset_from_source(
        eval_source,
        base_model,
        feature_mode=args.feature_mode,
    )
    training_dataset = train_dataset
    if args.mediation_target_mode == "self_model":
        rank_train_dataset = (
            train_dataset
            if args.feature_mode == "latent_current"
            else option_mediation_dataset_from_source(
                train_source,
                base_model,
                feature_mode="latent_current",
            )
        )
        training_dataset = option_mediation_dataset_with_targets(
            train_dataset,
            predicted_future_option_choices(rank_train_dataset),
        )
    score_targets = None
    if (
        args.score_reconstruction_weight > 0.0
        or args.score_pretrain_epochs > 0
        or args.score_distillation_weight > 0.0
        or args.score_rank_weight > 0.0
        or args.score_rank_code_weight > 0.0
        or args.score_value_code_weight > 0.0
        or args.heldout_receiver_score_distillation_weight > 0.0
    ):
        rank_score_dataset = (
            train_dataset
            if args.feature_mode == "latent_current"
            else option_mediation_dataset_from_source(
                train_source,
                base_model,
                feature_mode="latent_current",
            )
        )
        score_targets = predicted_future_option_scores(rank_score_dataset)
    if args.population_size > 1:
        if args.staged_population_epochs > 0:
            trained_base = train_option_mediator(
                training_dataset,
                hidden_size=args.hidden_size,
                receiver_size=args.receiver_size,
                receiver_copies=args.receiver_copies,
                receiver_turnover_interval=args.receiver_turnover_interval,
                slots=args.message_slots,
                vocabulary_size=args.message_vocabulary,
                epochs=args.mediation_epochs,
                batch_size=args.mediation_batch_size,
                learning_rate=args.mediation_learning_rate,
                balance_weight=args.mediation_balance_weight,
                entropy_weight=args.mediation_entropy_weight,
                message_commitment_weight=args.message_commitment_weight,
                message_temperature=args.message_temperature,
                soft_message_training=args.soft_message_training,
                score_targets=score_targets,
                score_pretrain_epochs=args.score_pretrain_epochs,
                score_pretrain_commitment_weight=(
                    args.score_pretrain_commitment_weight
                ),
                frozen_receiver_epochs=args.frozen_receiver_epochs,
                score_distillation_weight=args.score_distillation_weight,
                score_distillation_temperature=args.score_distillation_temperature,
                score_rank_weight=args.score_rank_weight,
                score_reconstruction_weight=args.score_reconstruction_weight,
                code_target_weight=args.code_target_weight,
                message_replay_weight=args.message_replay_weight,
                score_rank_code_weight=args.score_rank_code_weight,
                score_rank_code_slot=args.score_rank_code_slot,
                score_value_code_weight=args.score_value_code_weight,
                score_value_code_slot=args.score_value_code_slot,
                seed=seed,
            )
            trained_population = staged_option_population_from_mediator(
                trained_base,
                training_dataset,
                population_size=args.population_size,
                hidden_size=args.hidden_size,
                receiver_size=args.receiver_size,
                epochs=args.staged_population_epochs,
                batch_size=args.mediation_batch_size,
                learning_rate=args.mediation_learning_rate,
                message_temperature=args.message_temperature,
                balance_weight=args.mediation_balance_weight,
                entropy_weight=args.mediation_entropy_weight,
                message_commitment_weight=args.message_commitment_weight,
                transfer_receiver_count=args.staged_transfer_receiver_count,
                transfer_receiver_epochs=args.staged_transfer_receiver_epochs,
                transfer_receiver_weight=args.staged_transfer_receiver_weight,
                field_receiver_count=args.staged_field_receiver_count,
                field_receiver_epochs=args.staged_field_receiver_epochs,
                field_receiver_weight=args.staged_field_receiver_weight,
                sender_imitation_weight=args.staged_sender_imitation_weight,
                receiver_logit_distillation_weight=(
                    args.staged_receiver_logit_distillation_weight
                ),
                receiver_logit_distillation_temperature=(
                    args.staged_receiver_logit_distillation_temperature
                ),
                score_targets=score_targets,
                score_reconstruction_weight=(
                    args.staged_score_reconstruction_weight
                ),
                score_rank_weight=args.staged_score_rank_weight,
                positive_delta_code_weight=(
                    args.staged_positive_delta_code_weight
                ),
                positive_delta_code_slot=args.staged_positive_delta_code_slot,
                relative_value_code_weight=(
                    args.staged_relative_value_code_weight
                ),
                relative_value_code_slot=args.staged_relative_value_code_slot,
                seed=seed + 80_000,
            )
        else:
            trained_population = train_option_population_mediator(
                training_dataset,
                population_size=args.population_size,
                hidden_size=args.hidden_size,
                receiver_size=args.receiver_size,
                slots=args.message_slots,
                vocabulary_size=args.message_vocabulary,
                epochs=args.mediation_epochs,
                batch_size=args.mediation_batch_size,
                learning_rate=args.mediation_learning_rate,
                balance_weight=args.mediation_balance_weight,
                entropy_weight=args.mediation_entropy_weight,
                message_commitment_weight=args.message_commitment_weight,
                message_temperature=args.message_temperature,
                score_targets=score_targets,
                score_pretrain_epochs=args.score_pretrain_epochs,
                score_pretrain_commitment_weight=args.score_pretrain_commitment_weight,
                sender_agreement_weight=args.sender_agreement_weight,
                seed=seed,
            )
        rows: list[OptionWorldMediationReplicationRow] = []
        for sender_index in range(args.population_size):
            for intervention in args.interventions:
                intervened = intervene_option_mediation_features(
                    eval_dataset,
                    intervention=intervention,
                    seed=seed + 20_000,
                )
                result = evaluate_option_population_mediator(
                    trained_population,
                    intervened,
                    sender_index=sender_index,
                    model_control=f"{model_control}_population_s{sender_index}",
                    intervention=intervention,
                )
                rows.append(
                    _result_row(
                        seed,
                        result,
                        world_final_need_mse=world_final_need_mse,
                        world_trend_accuracy=world_trend_accuracy,
                        before_world_final_need_mse=before_world_final_need_mse,
                        after_world_final_need_mse=after_world_final_need_mse,
                        before_world_trend_accuracy=before_world_trend_accuracy,
                        after_world_trend_accuracy=after_world_trend_accuracy,
                    )
                )
            if args.heldout_receiver_epochs > 0:
                heldout_receiver = train_option_receiver_for_population_sender(
                    trained_population,
                    training_dataset,
                    sender_index=sender_index,
                    receiver_size=args.receiver_size,
                    epochs=args.heldout_receiver_epochs,
                    batch_size=args.mediation_batch_size,
                    learning_rate=args.mediation_learning_rate,
                    max_samples=args.heldout_receiver_samples,
                    seed=seed + 70_000 + sender_index,
                )
                for intervention in args.interventions:
                    intervened = intervene_option_mediation_features(
                        eval_dataset,
                        intervention=intervention,
                        seed=seed + 20_000,
                    )
                    result = evaluate_option_population_receiver_transfer(
                        trained_population,
                        heldout_receiver,
                        intervened,
                        sender_index=sender_index,
                        model_control=(
                            f"{model_control}_population_heldout_receiver_s"
                            f"{sender_index}"
                        ),
                        intervention=intervention,
                    )
                    rows.append(
                        _result_row(
                            seed,
                            result,
                            world_final_need_mse=world_final_need_mse,
                            world_trend_accuracy=world_trend_accuracy,
                            before_world_final_need_mse=before_world_final_need_mse,
                            after_world_final_need_mse=after_world_final_need_mse,
                            before_world_trend_accuracy=before_world_trend_accuracy,
                            after_world_trend_accuracy=after_world_trend_accuracy,
                        )
                    )
        if include_target_majority:
            rows.append(
                _result_row(
                    seed,
                    majority_option_result(train_dataset, eval_dataset),
                    world_final_need_mse=world_final_need_mse,
                    world_trend_accuracy=world_trend_accuracy,
                    before_world_final_need_mse=before_world_final_need_mse,
                    after_world_final_need_mse=after_world_final_need_mse,
                    before_world_trend_accuracy=before_world_trend_accuracy,
                    after_world_trend_accuracy=after_world_trend_accuracy,
                )
            )
        if args.self_model_rank_control:
            rank_dataset = (
                eval_dataset
                if args.feature_mode == "latent_current"
                else option_mediation_dataset_from_source(
                    eval_source,
                    base_model,
                    feature_mode="latent_current",
                )
            )
            rows.append(
                _result_row(
                    seed,
                    predicted_future_option_result(
                        rank_dataset,
                        model_control=f"{model_control}_self_model",
                    ),
                    world_final_need_mse=world_final_need_mse,
                    world_trend_accuracy=world_trend_accuracy,
                    before_world_final_need_mse=before_world_final_need_mse,
                    after_world_final_need_mse=after_world_final_need_mse,
                    before_world_trend_accuracy=before_world_trend_accuracy,
                    after_world_trend_accuracy=after_world_trend_accuracy,
                )
            )
        return rows
    trained = train_option_mediator(
        training_dataset,
        hidden_size=args.hidden_size,
        receiver_size=args.receiver_size,
        receiver_copies=args.receiver_copies,
        receiver_turnover_interval=args.receiver_turnover_interval,
        slots=args.message_slots,
        vocabulary_size=args.message_vocabulary,
        epochs=args.mediation_epochs,
        batch_size=args.mediation_batch_size,
        learning_rate=args.mediation_learning_rate,
        balance_weight=args.mediation_balance_weight,
        entropy_weight=args.mediation_entropy_weight,
        message_commitment_weight=args.message_commitment_weight,
        message_temperature=args.message_temperature,
        soft_message_training=args.soft_message_training,
        score_targets=score_targets,
        score_pretrain_epochs=args.score_pretrain_epochs,
        score_pretrain_commitment_weight=args.score_pretrain_commitment_weight,
        frozen_receiver_epochs=args.frozen_receiver_epochs,
        score_distillation_weight=args.score_distillation_weight,
        score_distillation_temperature=args.score_distillation_temperature,
        score_rank_weight=args.score_rank_weight,
        score_reconstruction_weight=args.score_reconstruction_weight,
        code_target_weight=args.code_target_weight,
        message_replay_weight=args.message_replay_weight,
        score_rank_code_weight=args.score_rank_code_weight,
        score_rank_code_slot=args.score_rank_code_slot,
        score_value_code_weight=args.score_value_code_weight,
        score_value_code_slot=args.score_value_code_slot,
        seed=seed,
    )

    rows: list[OptionWorldMediationReplicationRow] = []
    for intervention in args.interventions:
        intervened = intervene_option_mediation_features(
            eval_dataset,
            intervention=intervention,
            seed=seed + 20_000,
        )
        result = evaluate_option_mediator(
            trained,
            intervened,
            model_control=model_control,
            intervention=intervention,
        )
        rows.append(
            _result_row(
                seed,
                result,
                world_final_need_mse=world_final_need_mse,
                world_trend_accuracy=world_trend_accuracy,
                before_world_final_need_mse=before_world_final_need_mse,
                after_world_final_need_mse=after_world_final_need_mse,
                before_world_trend_accuracy=before_world_trend_accuracy,
                after_world_trend_accuracy=after_world_trend_accuracy,
            )
        )

    if args.heldout_receiver_epochs > 0:
        heldout_receiver = train_option_receiver_for_sender(
            trained,
            training_dataset,
            receiver_size=args.receiver_size,
            epochs=args.heldout_receiver_epochs,
            batch_size=args.mediation_batch_size,
            learning_rate=args.mediation_learning_rate,
            max_samples=args.heldout_receiver_samples,
            score_targets=score_targets,
            score_distillation_weight=(
                args.heldout_receiver_score_distillation_weight
            ),
            score_distillation_temperature=(
                args.heldout_receiver_score_distillation_temperature
            ),
            seed=seed + 70_000,
        )
        for intervention in args.interventions:
            intervened = intervene_option_mediation_features(
                eval_dataset,
                intervention=intervention,
                seed=seed + 20_000,
            )
            result = evaluate_option_receiver_transfer(
                trained,
                heldout_receiver,
                intervened,
                model_control=f"{model_control}_heldout_receiver",
                intervention=intervention,
            )
            rows.append(
                _result_row(
                    seed,
                    result,
                    world_final_need_mse=world_final_need_mse,
                    world_trend_accuracy=world_trend_accuracy,
                    before_world_final_need_mse=before_world_final_need_mse,
                    after_world_final_need_mse=after_world_final_need_mse,
                    before_world_trend_accuracy=before_world_trend_accuracy,
                    after_world_trend_accuracy=after_world_trend_accuracy,
                )
            )

    if include_target_majority:
        majority = majority_option_result(train_dataset, eval_dataset)
        rows.append(
            _result_row(
                seed,
                majority,
                world_final_need_mse=world_final_need_mse,
                world_trend_accuracy=world_trend_accuracy,
                before_world_final_need_mse=before_world_final_need_mse,
                after_world_final_need_mse=after_world_final_need_mse,
                before_world_trend_accuracy=before_world_trend_accuracy,
                after_world_trend_accuracy=after_world_trend_accuracy,
            )
        )
    if args.self_model_rank_control:
        rank_dataset = (
            eval_dataset
            if args.feature_mode == "latent_current"
            else option_mediation_dataset_from_source(
                eval_source,
                base_model,
                feature_mode="latent_current",
            )
        )
        rows.append(
            _result_row(
                seed,
                predicted_future_option_result(
                    rank_dataset,
                    model_control=f"{model_control}_self_model",
                ),
                world_final_need_mse=world_final_need_mse,
                world_trend_accuracy=world_trend_accuracy,
                before_world_final_need_mse=before_world_final_need_mse,
                after_world_final_need_mse=after_world_final_need_mse,
                before_world_trend_accuracy=before_world_trend_accuracy,
                after_world_trend_accuracy=after_world_trend_accuracy,
            )
        )
    return rows


def _result_row(
    seed: int,
    result,
    *,
    world_final_need_mse: float,
    world_trend_accuracy: float,
    before_world_final_need_mse: float,
    after_world_final_need_mse: float,
    before_world_trend_accuracy: float,
    after_world_trend_accuracy: float,
) -> OptionWorldMediationReplicationRow:
    return OptionWorldMediationReplicationRow(
        seed=seed,
        model_control=result.model_control,
        intervention=result.intervention,
        samples=result.samples,
        choice_accuracy=result.choice_accuracy,
        mean_regret=result.mean_regret,
        mean_chosen_delta=result.mean_chosen_delta,
        mean_oracle_delta=result.mean_oracle_delta,
        message_codes_used=result.message_codes_used,
        message_code_entropy=result.message_code_entropy,
        dominant_message_code_fraction=result.dominant_message_code_fraction,
        target_code_mutual_information=result.target_code_mutual_information,
        choice_code_mutual_information=result.choice_code_mutual_information,
        positive_delta_code_mutual_information=(
            result.positive_delta_code_mutual_information
        ),
        relative_value_code_mutual_information=(
            result.relative_value_code_mutual_information
        ),
        message_patterns_used=result.message_patterns_used,
        reused_message_pattern_fraction=result.reused_message_pattern_fraction,
        target_pattern_mutual_information=result.target_pattern_mutual_information,
        choice_pattern_mutual_information=result.choice_pattern_mutual_information,
        positive_delta_pattern_mutual_information=(
            result.positive_delta_pattern_mutual_information
        ),
        relative_value_pattern_mutual_information=(
            result.relative_value_pattern_mutual_information
        ),
        world_final_need_mse=world_final_need_mse,
        world_trend_accuracy=world_trend_accuracy,
        trained_world_final_need_mse_before=before_world_final_need_mse,
        trained_world_final_need_mse_after=after_world_final_need_mse,
        trained_world_trend_before=before_world_trend_accuracy,
        trained_world_trend_after=after_world_trend_accuracy,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[9901, 9902])
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument(
        "--option-action-noise",
        type=float,
        default=0.0,
        help="Probability of replacing a scripted option step with another valid body action.",
    )
    parser.add_argument(
        "--world-option-action-noise",
        type=float,
        default=None,
        help="Override option action noise for option-world branch training/eval.",
    )
    parser.add_argument(
        "--mediation-option-action-noise",
        type=float,
        default=None,
        help="Override option action noise for mediation source collection.",
    )
    parser.add_argument("--renewable-resources", action="store_true")
    parser.add_argument(
        "--resource-ecology",
        choices=RESOURCE_ECOLOGIES,
        default=None,
    )
    parser.add_argument(
        "--state-policy",
        choices=STATE_POLICIES,
        default="cycle",
    )
    parser.add_argument("--teacher-mode", default="grounded")
    parser.add_argument(
        "--mediation-balance-target",
        choices=OPTION_MEDIATION_BALANCE_TARGETS,
        default="target_option",
    )
    parser.add_argument(
        "--interventions",
        nargs="+",
        choices=OPTION_FEATURE_INTERVENTIONS,
        default=["original", "shuffle_delta", "negate_delta"],
    )
    parser.add_argument("--random-model-control", action="store_true")
    parser.add_argument("--self-model-rank-control", action="store_true")
    parser.add_argument(
        "--mediation-target-mode",
        choices=OPTION_MEDIATION_TARGET_MODES,
        default="oracle",
    )

    parser.add_argument("--world-train-episodes", type=int, default=1000)
    parser.add_argument("--world-eval-episodes", type=int, default=500)
    parser.add_argument("--max-world-train-samples", type=int, default=14000)
    parser.add_argument("--max-world-eval-samples", type=int, default=7000)
    parser.add_argument("--world-epochs", type=int, default=8)
    parser.add_argument("--world-batch-size", type=int, default=128)
    parser.add_argument("--world-eval-batch-size", type=int, default=128)
    parser.add_argument("--world-learning-rate", type=float, default=4e-4)
    parser.add_argument("--step-needs-weight", type=float, default=2.0)
    parser.add_argument("--final-needs-weight", type=float, default=10.0)
    parser.add_argument("--observation-prediction-weight", type=float, default=0.03)
    parser.add_argument("--reward-prediction-weight", type=float, default=0.4)
    parser.add_argument("--rank-finetune-epochs", type=int, default=0)
    parser.add_argument("--rank-finetune-batch-size", type=int, default=128)
    parser.add_argument("--rank-finetune-learning-rate", type=float, default=1e-4)
    parser.add_argument("--rank-finetune-temperature", type=float, default=0.05)
    parser.add_argument("--rank-finetune-dynamics-weight", type=float, default=0.0)

    parser.add_argument("--mediation-train-episodes", type=int, default=1000)
    parser.add_argument("--mediation-eval-episodes", type=int, default=500)
    parser.add_argument("--max-mediation-train-states", type=int, default=12000)
    parser.add_argument("--max-mediation-eval-states", type=int, default=6000)
    parser.add_argument("--min-value-gap", type=float, default=0.005)
    parser.add_argument(
        "--feature-mode",
        choices=OPTION_MEDIATION_FEATURE_MODES,
        default="latent_current",
    )
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--receiver-size", type=int, default=96)
    parser.add_argument("--receiver-copies", type=int, default=1)
    parser.add_argument("--receiver-turnover-interval", type=int, default=0)
    parser.add_argument("--population-size", type=int, default=1)
    parser.add_argument("--sender-agreement-weight", type=float, default=0.0)
    parser.add_argument("--staged-population-epochs", type=int, default=0)
    parser.add_argument("--staged-transfer-receiver-count", type=int, default=0)
    parser.add_argument("--staged-transfer-receiver-epochs", type=int, default=40)
    parser.add_argument("--staged-transfer-receiver-weight", type=float, default=0.0)
    parser.add_argument("--staged-field-receiver-count", type=int, default=0)
    parser.add_argument("--staged-field-receiver-epochs", type=int, default=40)
    parser.add_argument("--staged-field-receiver-weight", type=float, default=0.0)
    parser.add_argument("--staged-sender-imitation-weight", type=float, default=0.0)
    parser.add_argument(
        "--staged-receiver-logit-distillation-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--staged-receiver-logit-distillation-temperature",
        type=float,
        default=1.0,
    )
    parser.add_argument("--staged-score-reconstruction-weight", type=float, default=0.0)
    parser.add_argument("--staged-score-rank-weight", type=float, default=0.0)
    parser.add_argument(
        "--staged-positive-delta-code-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument("--staged-positive-delta-code-slot", type=int, default=0)
    parser.add_argument(
        "--staged-relative-value-code-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument("--staged-relative-value-code-slot", type=int, default=1)
    parser.add_argument("--message-slots", type=int, default=OPTION_MEDIATION_SLOTS)
    parser.add_argument(
        "--message-vocabulary",
        type=int,
        default=OPTION_MEDIATION_VOCABULARY,
    )
    parser.add_argument("--mediation-epochs", type=int, default=100)
    parser.add_argument("--mediation-batch-size", type=int, default=128)
    parser.add_argument("--mediation-learning-rate", type=float, default=1e-3)
    parser.add_argument("--mediation-balance-weight", type=float, default=0.02)
    parser.add_argument("--mediation-entropy-weight", type=float, default=0.0)
    parser.add_argument("--message-commitment-weight", type=float, default=0.0)
    parser.add_argument("--message-temperature", type=float, default=0.6)
    parser.add_argument("--soft-message-training", action="store_true")
    parser.add_argument("--score-pretrain-epochs", type=int, default=0)
    parser.add_argument("--score-pretrain-commitment-weight", type=float, default=None)
    parser.add_argument("--frozen-receiver-epochs", type=int, default=0)
    parser.add_argument("--score-distillation-weight", type=float, default=0.0)
    parser.add_argument("--score-distillation-temperature", type=float, default=1.0)
    parser.add_argument("--score-rank-weight", type=float, default=0.0)
    parser.add_argument("--score-reconstruction-weight", type=float, default=0.0)
    parser.add_argument("--code-target-weight", type=float, default=0.0)
    parser.add_argument("--message-replay-weight", type=float, default=0.0)
    parser.add_argument("--score-rank-code-weight", type=float, default=0.0)
    parser.add_argument("--score-rank-code-slot", type=int, default=0)
    parser.add_argument("--score-value-code-weight", type=float, default=0.0)
    parser.add_argument("--score-value-code-slot", type=int, default=1)
    parser.add_argument("--heldout-receiver-epochs", type=int, default=0)
    parser.add_argument("--heldout-receiver-samples", type=int, default=None)
    parser.add_argument(
        "--heldout-receiver-score-distillation-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--heldout-receiver-score-distillation-temperature",
        type=float,
        default=1.0,
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
