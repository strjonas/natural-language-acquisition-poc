from __future__ import annotations

import argparse
from dataclasses import dataclass, replace

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .env import RESOURCE_ECOLOGIES
from .imitation import load_checkpoint
from .option_counterfactual_language import STATE_POLICIES
from .option_mediation import (
    OPTION_MEDIATION_BALANCE_TARGETS,
    OPTION_MEDIATION_FEATURE_INTERVENTIONS,
    OPTION_MEDIATION_FEATURE_MODES,
    OPTION_MEDIATION_VOCABULARY,
    OptionMediationDataset,
    collect_option_mediation_source,
    intervene_option_mediation_features,
    option_mediation_dataset_from_source,
    target_count_string,
)
from .option_world_mediation_replication import train_option_rank_finetune
from .option_world_model import collect_option_branch_dataset, train_option_world_model

FORCED_PROPOSAL_MODES = (
    "model",
    "model_runner_up",
    "limited_partner",
    "second_best",
    "worst",
)
REPAIR_PROPOSAL_MODES = (
    "model_runner_up",
    "limited_partner",
    "second_best",
    "worst",
)
VALUE_BASED_REPAIR_PROPOSAL_MODES = ("second_best", "worst")
LIMITED_PARTNER_MASK_MODES = ("first_half", "even_features", "odd_features")
TRAIN_PROPOSAL_MODES = ("model", "limited_partner")


@dataclass(frozen=True)
class TrainedOptionDialogue:
    model: "OptionDialogueMediator"
    feature_mean: mx.array
    feature_std: mx.array
    limited_partner: "LimitedPartnerProposal | None" = None
    limited_partner_mask: mx.array | None = None


@dataclass(frozen=True)
class OptionDialogueResult:
    model_control: str
    intervention: str
    samples: int
    proposal_accuracy: float
    final_accuracy: float
    changed_fraction: float
    mean_chosen_delta: float
    mean_oracle_delta: float
    mean_regret: float
    target_counts: str


class OptionDialogueMediator(nn.Module):
    def __init__(
        self,
        input_size: int,
        option_count: int,
        *,
        hidden_size: int = 96,
        receiver_size: int = 96,
        vocabulary_size: int = OPTION_MEDIATION_VOCABULARY,
    ) -> None:
        super().__init__()
        self.option_count = option_count
        self.vocabulary_size = vocabulary_size
        self.first_sender = nn.Linear(input_size, hidden_size)
        self.first_sender_hidden = nn.Linear(hidden_size, hidden_size)
        self.first_token = nn.Linear(hidden_size, vocabulary_size)
        self.proposal_receiver = nn.Linear(option_count * vocabulary_size, receiver_size)
        self.proposal_hidden = nn.Linear(receiver_size, receiver_size)
        self.proposal = nn.Linear(receiver_size, option_count)
        self.reply_sender = nn.Linear(input_size + 1, hidden_size)
        self.reply_sender_hidden = nn.Linear(hidden_size, hidden_size)
        self.reply_token = nn.Linear(hidden_size, vocabulary_size)
        final_input = option_count * vocabulary_size * 2 + option_count
        self.final_receiver = nn.Linear(final_input, receiver_size)
        self.final_hidden = nn.Linear(receiver_size, receiver_size)
        self.final_choice = nn.Linear(receiver_size, option_count)

    def first_message(
        self,
        features: mx.array,
        *,
        temperature: float = 0.6,
        hard: bool = True,
    ) -> tuple[mx.array, mx.array]:
        batch_size, option_count, feature_size = features.shape
        flat = mx.reshape(features, (batch_size * option_count, feature_size))
        hidden = nn.relu(self.first_sender(flat))
        hidden = hidden + nn.relu(self.first_sender_hidden(hidden))
        logits = self.first_token(hidden)
        probs = mx.softmax(logits / temperature, axis=-1)
        if hard:
            hard_message = mx.eye(self.vocabulary_size)[mx.argmax(probs, axis=-1)]
            message = hard_message + probs - mx.stop_gradient(probs)
        else:
            message = probs
        return (
            mx.reshape(message, (batch_size, option_count, self.vocabulary_size)),
            mx.reshape(probs, (batch_size, option_count, self.vocabulary_size)),
        )

    def propose(self, first_message: mx.array) -> mx.array:
        inputs = mx.reshape(first_message, (first_message.shape[0], -1))
        hidden = nn.relu(self.proposal_receiver(inputs))
        hidden = hidden + nn.relu(self.proposal_hidden(hidden))
        return self.proposal(hidden)

    def reply_message(
        self,
        features: mx.array,
        proposal_signal: mx.array,
        *,
        temperature: float = 0.6,
        hard: bool = True,
    ) -> tuple[mx.array, mx.array]:
        batch_size, option_count, feature_size = features.shape
        reply_features = mx.concatenate([features, proposal_signal[:, :, None]], axis=-1)
        flat = mx.reshape(reply_features, (batch_size * option_count, feature_size + 1))
        hidden = nn.relu(self.reply_sender(flat))
        hidden = hidden + nn.relu(self.reply_sender_hidden(hidden))
        logits = self.reply_token(hidden)
        probs = mx.softmax(logits / temperature, axis=-1)
        if hard:
            hard_message = mx.eye(self.vocabulary_size)[mx.argmax(probs, axis=-1)]
            message = hard_message + probs - mx.stop_gradient(probs)
        else:
            message = probs
        return (
            mx.reshape(message, (batch_size, option_count, self.vocabulary_size)),
            mx.reshape(probs, (batch_size, option_count, self.vocabulary_size)),
        )

    def final(self, first_message: mx.array, reply_message: mx.array, proposal_signal: mx.array) -> mx.array:
        inputs = mx.concatenate(
            [
                mx.reshape(first_message, (first_message.shape[0], -1)),
                mx.reshape(reply_message, (reply_message.shape[0], -1)),
                proposal_signal,
            ],
            axis=-1,
        )
        hidden = nn.relu(self.final_receiver(inputs))
        hidden = hidden + nn.relu(self.final_hidden(hidden))
        return self.final_choice(hidden)


class LimitedPartnerProposal(nn.Module):
    def __init__(
        self,
        input_size: int,
        option_count: int,
        *,
        hidden_size: int = 48,
    ) -> None:
        super().__init__()
        self.option_count = option_count
        self.receiver = nn.Linear(option_count * input_size, hidden_size)
        self.receiver_hidden = nn.Linear(hidden_size, hidden_size)
        self.choice = nn.Linear(hidden_size, option_count)

    def __call__(self, features: mx.array) -> mx.array:
        inputs = mx.reshape(features, (features.shape[0], -1))
        hidden = nn.relu(self.receiver(inputs))
        hidden = hidden + nn.relu(self.receiver_hidden(hidden))
        return self.choice(hidden)


def train_option_dialogue(
    dataset: OptionMediationDataset,
    *,
    hidden_size: int = 96,
    receiver_size: int = 96,
    vocabulary_size: int = OPTION_MEDIATION_VOCABULARY,
    epochs: int = 80,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    proposal_weight: float = 0.2,
    train_proposal_mode: str = "model",
    repair_weight: float = 0.0,
    repair_proposal_modes: tuple[str, ...] = (),
    repair_only_mistakes: bool = False,
    repair_min_regret: float = 0.0,
    repair_stage_epochs: int = 0,
    repair_stage_learning_rate: float | None = None,
    limited_partner_epochs: int = 0,
    limited_partner_hidden_size: int = 48,
    limited_partner_mask_mode: str = "first_half",
    balance_weight: float = 0.02,
    entropy_weight: float = 0.0,
    message_temperature: float = 0.6,
    seed: int = 1,
) -> TrainedOptionDialogue:
    rng = np.random.default_rng(seed)
    mx.random.seed(seed)
    feature_mean = mx.mean(dataset.features, axis=(0, 1), keepdims=True)
    feature_std = mx.sqrt(
        mx.mean((dataset.features - feature_mean) ** 2, axis=(0, 1), keepdims=True)
        + 1e-6
    )
    features = (dataset.features - feature_mean) / feature_std
    sample_count = int(features.shape[0])
    option_count = int(features.shape[1])
    model = OptionDialogueMediator(
        int(features.shape[-1]),
        option_count,
        hidden_size=hidden_size,
        receiver_size=receiver_size,
        vocabulary_size=vocabulary_size,
    )
    optimizer = optim.Adam(learning_rate=learning_rate)
    indices = np.arange(sample_count)
    unknown_repair_modes = set(repair_proposal_modes) - set(REPAIR_PROPOSAL_MODES)
    if unknown_repair_modes:
        raise ValueError(f"Unknown repair proposal modes: {sorted(unknown_repair_modes)}.")
    if train_proposal_mode not in TRAIN_PROPOSAL_MODES:
        raise ValueError(f"Unknown train proposal mode: {train_proposal_mode}.")
    if repair_min_regret < 0.0:
        raise ValueError("repair_min_regret must be non-negative.")
    if repair_stage_epochs < 0:
        raise ValueError("repair_stage_epochs must be non-negative.")
    needs_limited_partner = (
        "limited_partner" in repair_proposal_modes
        or train_proposal_mode == "limited_partner"
    )
    if needs_limited_partner and limited_partner_epochs <= 0:
        raise ValueError("limited_partner repair requires limited_partner_epochs > 0.")
    limited_partner_mask = None
    limited_partner = None
    if limited_partner_epochs > 0:
        limited_partner_mask = mx.array(
            _limited_partner_feature_mask(
                int(features.shape[-1]),
                mode=limited_partner_mask_mode,
            )[None, None, :],
            dtype=mx.float32,
        )
        limited_partner = _train_limited_partner(
            features,
            dataset.target_options,
            feature_mask=limited_partner_mask,
            hidden_size=limited_partner_hidden_size,
            epochs=limited_partner_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed + 10_101,
        )
    repair_proposals = mx.array(
        _proposal_table_from_values(
            dataset.option_values,
            modes=tuple(
                mode
                for mode in repair_proposal_modes
                if mode in VALUE_BASED_REPAIR_PROPOSAL_MODES
            ),
        ),
        dtype=mx.int32,
    )

    def ce_examples(logits: mx.array, targets: mx.array) -> mx.array:
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        selected = mx.sum(log_probs * mx.eye(option_count)[targets], axis=-1)
        return -selected

    def ce_loss(logits: mx.array, targets: mx.array) -> mx.array:
        return mx.mean(ce_examples(logits, targets))

    def weighted_examples_loss(examples: mx.array, weights: mx.array) -> mx.array:
        return mx.sum(examples * weights) / (mx.sum(weights) + 1e-6)

    def token_regularizer(probabilities: list[mx.array]) -> mx.array:
        uniform = 1.0 / vocabulary_size
        balance = mx.array(0.0)
        entropy = mx.array(0.0)
        for probs in probabilities:
            flat = mx.reshape(probs, (-1, probs.shape[-1]))
            balance = balance + mx.sum((mx.mean(flat, axis=0) - uniform) ** 2)
            entropy = entropy - mx.mean(mx.sum(flat * mx.log(flat + 1e-8), axis=-1))
        return balance_weight * balance / len(probabilities) + entropy_weight * entropy / len(probabilities)

    def final_loss_for_proposals(
        batch_features: mx.array,
        batch_targets: mx.array,
        first_message: mx.array,
        proposal_signal: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        reply_message, reply_probs = model.reply_message(
            batch_features,
            proposal_signal,
            temperature=message_temperature,
            hard=True,
        )
        final_logits = model.final(first_message, reply_message, proposal_signal)
        example_losses = ce_examples(final_logits, batch_targets)
        return mx.mean(example_losses), reply_probs, example_losses

    def repair_loss_for_batch(
        batch_features: mx.array,
        batch_targets: mx.array,
        batch_value_repair_proposals: mx.array,
        batch_option_values: mx.array,
        first_message: mx.array,
        proposal_logits: mx.array,
    ) -> tuple[mx.array, list[mx.array]]:
        if len(repair_proposal_modes) == 0:
            return mx.array(0.0), []
        value_repair_index = 0
        repair_loss = mx.array(0.0)
        repair_probabilities: list[mx.array] = []
        for repair_mode in repair_proposal_modes:
            if repair_mode == "model_runner_up":
                repair_indices = _runner_up_from_logits(proposal_logits)
            elif repair_mode == "limited_partner":
                if limited_partner is None or limited_partner_mask is None:
                    raise ValueError("limited_partner repair requires a trained partner.")
                partner_logits = limited_partner(batch_features * limited_partner_mask)
                repair_indices = mx.argmax(partner_logits, axis=-1)
            else:
                repair_indices = batch_value_repair_proposals[:, value_repair_index]
                value_repair_index += 1
            repair_signal = mx.eye(option_count)[repair_indices]
            single_repair_loss, repair_reply_probs, repair_examples = final_loss_for_proposals(
                batch_features,
                batch_targets,
                first_message,
                repair_signal,
            )
            if repair_only_mistakes or repair_min_regret > 0.0:
                weights = _repair_loss_weights(
                    repair_indices,
                    batch_targets,
                    batch_option_values,
                    only_mistakes=repair_only_mistakes,
                    min_regret=repair_min_regret,
                )
                single_repair_loss = weighted_examples_loss(repair_examples, weights)
            repair_loss = repair_loss + single_repair_loss
            repair_probabilities.append(repair_reply_probs)
        return repair_loss / len(repair_proposal_modes), repair_probabilities

    def loss_fn(
        batch_features: mx.array,
        batch_targets: mx.array,
        batch_value_repair_proposals: mx.array,
        batch_option_values: mx.array,
    ) -> mx.array:
        first_message, first_probs = model.first_message(
            batch_features,
            temperature=message_temperature,
            hard=True,
        )
        proposal_logits = model.propose(first_message)
        if train_proposal_mode == "limited_partner":
            if limited_partner is None or limited_partner_mask is None:
                raise ValueError("limited_partner proposal training requires a trained partner.")
            partner_logits = limited_partner(batch_features * limited_partner_mask)
            proposal_signal = mx.eye(option_count)[mx.argmax(partner_logits, axis=-1)]
        else:
            proposal_probs = mx.softmax(proposal_logits, axis=-1)
            hard_proposal = mx.eye(option_count)[mx.argmax(proposal_probs, axis=-1)]
            proposal_signal = hard_proposal + proposal_probs - mx.stop_gradient(
                proposal_probs
            )
        final_loss, reply_probs, _final_examples = final_loss_for_proposals(
            batch_features,
            batch_targets,
            first_message,
            proposal_signal,
        )
        stage_base_repair_weight = 0.0 if repair_stage_epochs > 0 else repair_weight
        repair_loss, repair_probs = repair_loss_for_batch(
            batch_features,
            batch_targets,
            batch_value_repair_proposals,
            batch_option_values,
            first_message,
            proposal_logits,
        )
        return (
            final_loss
            + stage_base_repair_weight * repair_loss
            + proposal_weight * ce_loss(proposal_logits, batch_targets)
            + token_regularizer([first_probs, reply_probs] + repair_probs)
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    for _epoch in range(max(1, epochs)):
        rng.shuffle(indices)
        for start in range(0, sample_count, max(1, batch_size)):
            batch = mx.array(indices[start : start + max(1, batch_size)], dtype=mx.int32)
            loss, grads = loss_and_grad(
                features[batch],
                dataset.target_options[batch],
                repair_proposals[batch],
                dataset.option_values[batch],
            )
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

    if repair_stage_epochs > 0 and repair_weight > 0.0 and len(repair_proposal_modes) > 0:
        model.freeze()
        for module in (
            model.reply_sender,
            model.reply_sender_hidden,
            model.reply_token,
            model.final_receiver,
            model.final_hidden,
            model.final_choice,
        ):
            module.unfreeze()
        stage_optimizer = optim.Adam(
            learning_rate=(
                learning_rate
                if repair_stage_learning_rate is None
                else repair_stage_learning_rate
            )
        )

        def stage_loss_fn(
            batch_features: mx.array,
            batch_targets: mx.array,
            batch_value_repair_proposals: mx.array,
            batch_option_values: mx.array,
        ) -> mx.array:
            first_message, _first_probs = model.first_message(
                batch_features,
                temperature=message_temperature,
                hard=True,
            )
            proposal_logits = model.propose(first_message)
            repair_loss, repair_probs = repair_loss_for_batch(
                batch_features,
                batch_targets,
                batch_value_repair_proposals,
                batch_option_values,
                first_message,
                proposal_logits,
            )
            return repair_weight * repair_loss + token_regularizer(repair_probs)

        stage_loss_and_grad = nn.value_and_grad(model, stage_loss_fn)
        for _epoch in range(repair_stage_epochs):
            rng.shuffle(indices)
            for start in range(0, sample_count, max(1, batch_size)):
                batch = mx.array(indices[start : start + max(1, batch_size)], dtype=mx.int32)
                loss, grads = stage_loss_and_grad(
                    features[batch],
                    dataset.target_options[batch],
                    repair_proposals[batch],
                    dataset.option_values[batch],
                )
                stage_optimizer.update(model, grads)
                mx.eval(model.parameters(), stage_optimizer.state, loss)
        model.unfreeze()

    return TrainedOptionDialogue(
        model=model,
        feature_mean=feature_mean,
        feature_std=feature_std,
        limited_partner=limited_partner,
        limited_partner_mask=limited_partner_mask,
    )


def evaluate_option_dialogue(
    trained: TrainedOptionDialogue,
    dataset: OptionMediationDataset,
    *,
    model_control: str,
    intervention: str,
    forced_proposal: str = "model",
) -> OptionDialogueResult:
    features = (dataset.features - trained.feature_mean) / trained.feature_std
    first_message, _first_probs = trained.model.first_message(features, hard=True)
    proposal_logits = trained.model.propose(first_message)
    proposals = _dialogue_proposals(trained, proposal_logits, dataset, features, mode=forced_proposal)
    proposal_signal = mx.eye(trained.model.option_count)[mx.array(proposals, dtype=mx.int32)]
    reply_message, _reply_probs = trained.model.reply_message(
        features,
        proposal_signal,
        hard=True,
    )
    final_logits = trained.model.final(first_message, reply_message, proposal_signal)
    choices = np.asarray(mx.argmax(final_logits, axis=-1), dtype=np.int32)
    targets = np.asarray(dataset.target_options, dtype=np.int32)
    values = np.asarray(dataset.option_values, dtype=np.float32)
    current = np.asarray(dataset.current_lowest, dtype=np.float32)
    chosen_values = values[np.arange(values.shape[0]), choices]
    oracle_values = np.max(values, axis=1)
    return OptionDialogueResult(
        model_control=model_control,
        intervention=intervention,
        samples=int(values.shape[0]),
        proposal_accuracy=float(np.mean(proposals == targets)),
        final_accuracy=float(np.mean(choices == targets)),
        changed_fraction=float(np.mean(choices != proposals)),
        mean_chosen_delta=float(np.mean(chosen_values - current)),
        mean_oracle_delta=float(np.mean(oracle_values - current)),
        mean_regret=float(np.mean(oracle_values - chosen_values)),
        target_counts=target_count_string(dataset),
    )


def _forced_proposals(
    proposal_logits: mx.array,
    dataset: OptionMediationDataset,
    *,
    mode: str,
) -> np.ndarray:
    if mode == "model":
        return np.asarray(mx.argmax(proposal_logits, axis=-1), dtype=np.int32)
    if mode == "model_runner_up":
        return np.asarray(_runner_up_from_logits(proposal_logits), dtype=np.int32)
    return _proposal_indices_from_values(dataset.option_values, mode=mode)


def _dialogue_proposals(
    trained: TrainedOptionDialogue,
    proposal_logits: mx.array,
    dataset: OptionMediationDataset,
    features: mx.array,
    *,
    mode: str,
) -> np.ndarray:
    if mode == "limited_partner":
        if trained.limited_partner is None or trained.limited_partner_mask is None:
            raise ValueError("limited_partner proposals require a trained limited partner.")
        logits = trained.limited_partner(features * trained.limited_partner_mask)
        return np.asarray(mx.argmax(logits, axis=-1), dtype=np.int32)
    return _forced_proposals(proposal_logits, dataset, mode=mode)


def _train_limited_partner(
    features: mx.array,
    targets: mx.array,
    *,
    feature_mask: mx.array,
    hidden_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> LimitedPartnerProposal:
    rng = np.random.default_rng(seed)
    mx.random.seed(seed)
    sample_count = int(features.shape[0])
    option_count = int(features.shape[1])
    partner = LimitedPartnerProposal(
        int(features.shape[-1]),
        option_count,
        hidden_size=hidden_size,
    )
    optimizer = optim.Adam(learning_rate=learning_rate)
    indices = np.arange(sample_count)

    def loss_fn(batch_features: mx.array, batch_targets: mx.array) -> mx.array:
        logits = partner(batch_features * feature_mask)
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        selected = mx.sum(log_probs * mx.eye(option_count)[batch_targets], axis=-1)
        return -mx.mean(selected)

    loss_and_grad = nn.value_and_grad(partner, loss_fn)
    for _epoch in range(max(1, epochs)):
        rng.shuffle(indices)
        for start in range(0, sample_count, max(1, batch_size)):
            batch = mx.array(indices[start : start + max(1, batch_size)], dtype=mx.int32)
            loss, grads = loss_and_grad(features[batch], targets[batch])
            optimizer.update(partner, grads)
            mx.eval(partner.parameters(), optimizer.state, loss)
    return partner


def _repair_loss_weights(
    proposal_indices: mx.array,
    targets: mx.array,
    option_values: mx.array,
    *,
    only_mistakes: bool,
    min_regret: float,
) -> mx.array:
    weights = mx.ones(targets.shape, dtype=mx.float32)
    if only_mistakes:
        weights = weights * (proposal_indices != targets).astype(mx.float32)
    if min_regret > 0.0:
        option_count = int(option_values.shape[1])
        proposal_values = mx.sum(
            option_values * mx.eye(option_count)[proposal_indices],
            axis=-1,
        )
        regret = mx.max(option_values, axis=-1) - proposal_values
        weights = weights * (regret >= min_regret).astype(mx.float32)
    return weights


def _limited_partner_feature_mask(width: int, *, mode: str) -> np.ndarray:
    if mode not in LIMITED_PARTNER_MASK_MODES:
        raise ValueError(f"Unknown limited partner mask mode: {mode}.")
    mask = np.zeros((width,), dtype=np.float32)
    if mode == "first_half":
        mask[: max(1, width // 2)] = 1.0
    elif mode == "even_features":
        mask[::2] = 1.0
    elif mode == "odd_features":
        mask[1::2] = 1.0
        if not np.any(mask):
            mask[0] = 1.0
    return mask


def _runner_up_from_logits(proposal_logits: mx.array) -> mx.array:
    order = mx.argsort(proposal_logits, axis=-1)
    return order[:, -2].astype(mx.int32)


def _proposal_table_from_values(
    option_values: mx.array,
    *,
    modes: tuple[str, ...],
) -> np.ndarray:
    if len(modes) == 0:
        return np.zeros((int(option_values.shape[0]), 0), dtype=np.int32)
    return np.stack(
        [_proposal_indices_from_values(option_values, mode=mode) for mode in modes],
        axis=1,
    ).astype(np.int32)


def _proposal_indices_from_values(option_values: mx.array, *, mode: str) -> np.ndarray:
    values = np.asarray(option_values, dtype=np.float32)
    if mode == "worst":
        return np.asarray(np.argmin(values, axis=1), dtype=np.int32)
    if mode == "second_best":
        order = np.argsort(values, axis=1)
        return np.asarray(order[:, -2], dtype=np.int32)
    raise ValueError(f"Unknown forced proposal mode: {mode}.")


def format_result(result: OptionDialogueResult) -> str:
    return ",".join(
        [
            result.model_control,
            result.intervention,
            str(result.samples),
            f"{result.proposal_accuracy:.4f}",
            f"{result.final_accuracy:.4f}",
            f"{result.changed_fraction:.4f}",
            f"{result.mean_chosen_delta:.6f}",
            f"{result.mean_oracle_delta:.6f}",
            f"{result.mean_regret:.6f}",
            result.target_counts,
        ]
    )


def main() -> None:
    args = _parse_args()
    base_model, config = load_checkpoint(args.checkpoint)
    if args.renewable_resources:
        config = replace(config, renewable_resources=True)
    if args.resource_ecology is not None:
        config = replace(config, resource_ecology=args.resource_ecology)
    train_branches = None
    if args.world_epochs > 0 or args.rank_finetune_epochs > 0:
        train_branches = collect_option_branch_dataset(
            config,
            episodes=args.world_train_episodes,
            seed=args.seed,
            teacher_mode=args.teacher_mode,
            horizon=args.horizon,
            state_policy=args.state_policy,
            max_samples=args.max_world_train_samples,
            option_action_noise=args.option_action_noise,
        )
    if args.world_epochs > 0:
        train_option_world_model(
            base_model,
            config,
            train_branches,
            checkpoint_path=None,
            epochs=args.world_epochs,
            batch_size=args.world_batch_size,
            learning_rate=args.world_learning_rate,
            seed=args.seed,
            step_needs_weight=args.step_needs_weight,
            final_needs_weight=args.final_needs_weight,
            observation_prediction_weight=args.observation_prediction_weight,
            reward_prediction_weight=args.reward_prediction_weight,
        )
    train_source = collect_option_mediation_source(
        config,
        episodes=args.train_episodes,
        seed=args.seed,
        teacher_mode=args.teacher_mode,
        horizon=args.horizon,
        state_policy=args.state_policy,
        balance_target=args.train_balance_target,
        max_states=args.max_train_states,
        min_value_gap=args.min_value_gap,
        min_positive_delta=args.min_positive_delta,
        option_action_noise=args.option_action_noise,
    )
    eval_source = collect_option_mediation_source(
        config,
        episodes=args.eval_episodes,
        seed=args.seed + 10_000,
        teacher_mode=args.teacher_mode,
        horizon=args.horizon,
        state_policy=args.state_policy,
        balance_target=args.eval_balance_target,
        max_states=args.max_eval_states,
        min_value_gap=args.min_value_gap,
        min_positive_delta=args.min_positive_delta,
        option_action_noise=args.option_action_noise,
    )
    train_option_rank_finetune(
        base_model,
        train_source,
        dynamics_samples=None if train_branches is None else train_branches.samples,
        epochs=args.rank_finetune_epochs,
        batch_size=args.rank_finetune_batch_size,
        learning_rate=args.rank_finetune_learning_rate,
        temperature=args.rank_finetune_temperature,
        dynamics_weight=args.rank_finetune_dynamics_weight,
        step_needs_weight=args.step_needs_weight,
        final_needs_weight=args.final_needs_weight,
        observation_prediction_weight=args.observation_prediction_weight,
        reward_prediction_weight=args.reward_prediction_weight,
        seed=args.seed + 40_000,
    )
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
    trained = train_option_dialogue(
        train_dataset,
        hidden_size=args.hidden_size,
        receiver_size=args.receiver_size,
        vocabulary_size=args.vocabulary_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        proposal_weight=args.proposal_weight,
        train_proposal_mode=args.train_proposal_mode,
        repair_weight=args.repair_weight,
        repair_proposal_modes=tuple(args.repair_proposal_modes),
        repair_only_mistakes=args.repair_only_mistakes,
        repair_min_regret=args.repair_min_regret,
        repair_stage_epochs=args.repair_stage_epochs,
        repair_stage_learning_rate=args.repair_stage_learning_rate,
        limited_partner_epochs=args.limited_partner_epochs,
        limited_partner_hidden_size=args.limited_partner_hidden_size,
        limited_partner_mask_mode=args.limited_partner_mask_mode,
        balance_weight=args.balance_weight,
        entropy_weight=args.entropy_weight,
        message_temperature=args.message_temperature,
        seed=args.seed,
    )
    print(
        "model_control,intervention,samples,proposal_accuracy,final_accuracy,"
        "changed_fraction,mean_chosen_delta,mean_oracle_delta,mean_regret,"
        "target_counts"
    )
    for intervention in args.interventions:
        intervened = intervene_option_mediation_features(
            eval_dataset,
            intervention=intervention,
            seed=args.seed + 20_000,
        )
        for forced_proposal in args.forced_proposals:
            print(
                format_result(
                    evaluate_option_dialogue(
                        trained,
                        intervened,
                        model_control=f"dialogue_{forced_proposal}",
                        intervention=intervention,
                        forced_proposal=forced_proposal,
                    )
                )
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--train-episodes", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=250)
    parser.add_argument("--max-train-states", type=int, default=6000)
    parser.add_argument("--max-eval-states", type=int, default=3000)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--teacher-mode", default="grounded")
    parser.add_argument("--state-policy", choices=STATE_POLICIES, default="cycle")
    parser.add_argument("--renewable-resources", action="store_true")
    parser.add_argument("--resource-ecology", choices=RESOURCE_ECOLOGIES, default=None)
    parser.add_argument("--option-action-noise", type=float, default=0.0)
    parser.add_argument("--min-value-gap", type=float, default=0.005)
    parser.add_argument("--min-positive-delta", type=float, default=None)
    parser.add_argument("--world-train-episodes", type=int, default=500)
    parser.add_argument("--max-world-train-samples", type=int, default=7000)
    parser.add_argument("--world-epochs", type=int, default=0)
    parser.add_argument("--world-batch-size", type=int, default=128)
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
    parser.add_argument(
        "--feature-mode",
        choices=OPTION_MEDIATION_FEATURE_MODES,
        default="delta",
    )
    parser.add_argument(
        "--train-balance-target",
        choices=OPTION_MEDIATION_BALANCE_TARGETS,
        default="target_option_resample",
    )
    parser.add_argument(
        "--eval-balance-target",
        choices=OPTION_MEDIATION_BALANCE_TARGETS,
        default="none",
    )
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--receiver-size", type=int, default=96)
    parser.add_argument("--vocabulary-size", type=int, default=OPTION_MEDIATION_VOCABULARY)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--proposal-weight", type=float, default=0.2)
    parser.add_argument(
        "--train-proposal-mode",
        choices=TRAIN_PROPOSAL_MODES,
        default="model",
    )
    parser.add_argument("--repair-weight", type=float, default=0.0)
    parser.add_argument("--repair-only-mistakes", action="store_true")
    parser.add_argument("--repair-min-regret", type=float, default=0.0)
    parser.add_argument("--repair-stage-epochs", type=int, default=0)
    parser.add_argument("--repair-stage-learning-rate", type=float, default=None)
    parser.add_argument("--limited-partner-epochs", type=int, default=0)
    parser.add_argument("--limited-partner-hidden-size", type=int, default=48)
    parser.add_argument(
        "--limited-partner-mask-mode",
        choices=LIMITED_PARTNER_MASK_MODES,
        default="first_half",
    )
    parser.add_argument(
        "--repair-proposal-modes",
        nargs="*",
        choices=REPAIR_PROPOSAL_MODES,
        default=[],
    )
    parser.add_argument("--balance-weight", type=float, default=0.02)
    parser.add_argument("--entropy-weight", type=float, default=0.0)
    parser.add_argument("--message-temperature", type=float, default=0.6)
    parser.add_argument(
        "--interventions",
        nargs="+",
        choices=OPTION_MEDIATION_FEATURE_INTERVENTIONS,
        default=["original", "shuffle_delta", "reverse_delta_rank"],
    )
    parser.add_argument(
        "--forced-proposals",
        nargs="+",
        choices=FORCED_PROPOSAL_MODES,
        default=["model"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
