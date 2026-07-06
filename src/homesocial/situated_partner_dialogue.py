from __future__ import annotations

import argparse
from dataclasses import dataclass, replace

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .env import RESOURCE_ECOLOGIES
from .imitation import load_checkpoint
from .option_counterfactual_language import OPTION_NAMES, STATE_POLICIES
from .option_world_model import (
    OptionBranchDataset,
    collect_option_branch_dataset,
)
from .teachers import TEACHER_MODES


PARTNER_PROPOSAL_MODES = (
    "partial_body",
    "partial_food_water",
    "partial_energy_safety",
    "second_best",
    "worst",
)
SITUATED_INTERVENTIONS = (
    "original",
    "shuffle_delta",
    "reverse_delta_rank",
    "shuffle_outcome",
    "reverse_outcome_rank",
    "zero_outcome",
)


@dataclass(frozen=True)
class SituatedPartnerDataset:
    features: mx.array
    current_needs: mx.array
    final_needs: mx.array
    option_values: mx.array
    target_options: mx.array
    current_lowest: mx.array


@dataclass(frozen=True)
class TrainedSituatedPartnerDialogue:
    model: "SituatedPartnerDialogue"
    feature_mean: mx.array
    feature_std: mx.array


@dataclass(frozen=True)
class SituatedPartnerResult:
    model_control: str
    intervention: str
    samples: int
    proposal_accuracy: float
    final_accuracy: float
    changed_fraction: float
    mean_proposal_delta: float
    mean_chosen_delta: float
    mean_oracle_delta: float
    mean_regret: float
    target_counts: str


class SituatedPartnerDialogue(nn.Module):
    def __init__(
        self,
        input_size: int,
        option_count: int,
        *,
        hidden_size: int = 96,
        receiver_size: int = 96,
        vocabulary_size: int = 4,
    ) -> None:
        super().__init__()
        self.option_count = option_count
        self.vocabulary_size = vocabulary_size
        self.reply_sender = nn.Linear(input_size + 1, hidden_size)
        self.reply_sender_hidden = nn.Linear(hidden_size, hidden_size)
        self.reply_token = nn.Linear(hidden_size, vocabulary_size)
        final_input = option_count * vocabulary_size + option_count
        self.final_receiver = nn.Linear(final_input, receiver_size)
        self.final_hidden = nn.Linear(receiver_size, receiver_size)
        self.final_choice = nn.Linear(receiver_size, option_count)

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

    def final(self, reply_message: mx.array, proposal_signal: mx.array) -> mx.array:
        inputs = mx.concatenate(
            [
                mx.reshape(reply_message, (reply_message.shape[0], -1)),
                proposal_signal,
            ],
            axis=-1,
        )
        hidden = nn.relu(self.final_receiver(inputs))
        hidden = hidden + nn.relu(self.final_hidden(hidden))
        return self.final_choice(hidden)


def situated_partner_dataset_from_branches(
    branches: OptionBranchDataset,
    *,
    min_value_gap: float = 0.0,
    min_oracle_delta: float | None = None,
) -> SituatedPartnerDataset:
    option_count = len(OPTION_NAMES)
    samples = list(branches.samples)
    grouped_features: list[np.ndarray] = []
    grouped_current: list[np.ndarray] = []
    grouped_final: list[np.ndarray] = []
    grouped_values: list[np.ndarray] = []
    targets: list[int] = []
    current_lowest: list[float] = []

    for start in range(0, len(samples) - option_count + 1, option_count):
        group = samples[start : start + option_count]
        labels = [sample.option_label for sample in group]
        if labels != list(range(option_count)):
            continue
        current = np.asarray(group[0].current_needs, dtype=np.float32)
        final_needs = np.stack(
            [np.asarray(sample.next_needs, dtype=np.float32)[-1] for sample in group]
        ).astype(np.float32)
        values = np.min(final_needs, axis=1).astype(np.float32)
        current_low = float(np.min(current))
        if float(np.max(values) - np.min(values)) < min_value_gap:
            continue
        if (
            min_oracle_delta is not None
            and float(np.max(values) - current_low) < min_oracle_delta
        ):
            continue
        option_ids = np.eye(option_count, dtype=np.float32)
        features = np.concatenate(
            [
                np.repeat(current[None, :], option_count, axis=0),
                final_needs,
                final_needs - current[None, :],
                option_ids,
            ],
            axis=1,
        )
        grouped_features.append(features.astype(np.float32))
        grouped_current.append(current)
        grouped_final.append(final_needs)
        grouped_values.append(values)
        targets.append(int(np.argmax(values)))
        current_lowest.append(current_low)

    if not grouped_features:
        raise ValueError("No complete situated partner states were collected.")

    return SituatedPartnerDataset(
        features=mx.array(np.stack(grouped_features), dtype=mx.float32),
        current_needs=mx.array(np.stack(grouped_current), dtype=mx.float32),
        final_needs=mx.array(np.stack(grouped_final), dtype=mx.float32),
        option_values=mx.array(np.stack(grouped_values), dtype=mx.float32),
        target_options=mx.array(targets, dtype=mx.int32),
        current_lowest=mx.array(current_lowest, dtype=mx.float32),
    )


def train_situated_partner_dialogue(
    dataset: SituatedPartnerDataset,
    *,
    hidden_size: int = 96,
    receiver_size: int = 96,
    vocabulary_size: int = 4,
    epochs: int = 80,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    partner_mode: str = "partial_body",
    balance_weight: float = 0.02,
    entropy_weight: float = 0.0,
    message_temperature: float = 0.6,
    seed: int = 1,
) -> TrainedSituatedPartnerDialogue:
    if partner_mode not in PARTNER_PROPOSAL_MODES:
        raise ValueError(f"Unknown partner proposal mode: {partner_mode}.")

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
    model = SituatedPartnerDialogue(
        int(features.shape[-1]),
        option_count,
        hidden_size=hidden_size,
        receiver_size=receiver_size,
        vocabulary_size=vocabulary_size,
    )
    optimizer = optim.Adam(learning_rate=learning_rate)
    indices = np.arange(sample_count)
    partner_proposals = mx.array(
        _partner_proposals(dataset, mode=partner_mode),
        dtype=mx.int32,
    )

    def outcome_regret(logits: mx.array, option_values: mx.array) -> mx.array:
        choice_probs = mx.softmax(logits, axis=-1)
        expected_value = mx.sum(choice_probs * option_values, axis=-1)
        return mx.max(option_values, axis=-1) - expected_value

    def token_regularizer(probs: mx.array) -> mx.array:
        flat = mx.reshape(probs, (-1, probs.shape[-1]))
        uniform = 1.0 / vocabulary_size
        balance = mx.sum((mx.mean(flat, axis=0) - uniform) ** 2)
        entropy = -mx.mean(mx.sum(flat * mx.log(flat + 1e-8), axis=-1))
        return balance_weight * balance + entropy_weight * entropy

    def loss_fn(
        batch_features: mx.array,
        batch_values: mx.array,
        batch_partner_proposals: mx.array,
    ) -> mx.array:
        proposal_signal = mx.eye(option_count)[batch_partner_proposals]
        reply_message, reply_probs = model.reply_message(
            batch_features,
            proposal_signal,
            temperature=message_temperature,
            hard=True,
        )
        final_logits = model.final(reply_message, proposal_signal)
        return mx.mean(outcome_regret(final_logits, batch_values)) + token_regularizer(
            reply_probs
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    for _epoch in range(max(1, epochs)):
        rng.shuffle(indices)
        for start in range(0, sample_count, max(1, batch_size)):
            batch = mx.array(indices[start : start + max(1, batch_size)], dtype=mx.int32)
            loss, grads = loss_and_grad(
                features[batch],
                dataset.option_values[batch],
                partner_proposals[batch],
            )
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

    return TrainedSituatedPartnerDialogue(
        model=model,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )


def evaluate_situated_partner_dialogue(
    trained: TrainedSituatedPartnerDialogue,
    dataset: SituatedPartnerDataset,
    *,
    model_control: str,
    intervention: str,
    partner_mode: str = "partial_body",
) -> SituatedPartnerResult:
    if partner_mode not in PARTNER_PROPOSAL_MODES:
        raise ValueError(f"Unknown partner proposal mode: {partner_mode}.")
    features = (dataset.features - trained.feature_mean) / trained.feature_std
    proposals = _partner_proposals(dataset, mode=partner_mode)
    proposal_signal = mx.eye(trained.model.option_count)[
        mx.array(proposals, dtype=mx.int32)
    ]
    reply_message, _reply_probs = trained.model.reply_message(
        features,
        proposal_signal,
        hard=True,
    )
    final_logits = trained.model.final(reply_message, proposal_signal)
    choices = np.asarray(mx.argmax(final_logits, axis=-1), dtype=np.int32)
    targets = np.asarray(dataset.target_options, dtype=np.int32)
    values = np.asarray(dataset.option_values, dtype=np.float32)
    current = np.asarray(dataset.current_lowest, dtype=np.float32)
    proposal_values = values[np.arange(values.shape[0]), proposals]
    chosen_values = values[np.arange(values.shape[0]), choices]
    oracle_values = np.max(values, axis=1)
    return SituatedPartnerResult(
        model_control=model_control,
        intervention=intervention,
        samples=int(values.shape[0]),
        proposal_accuracy=float(np.mean(proposals == targets)),
        final_accuracy=float(np.mean(choices == targets)),
        changed_fraction=float(np.mean(choices != proposals)),
        mean_proposal_delta=float(np.mean(proposal_values - current)),
        mean_chosen_delta=float(np.mean(chosen_values - current)),
        mean_oracle_delta=float(np.mean(oracle_values - current)),
        mean_regret=float(np.mean(oracle_values - chosen_values)),
        target_counts=target_count_string(dataset),
    )


def intervene_situated_partner_features(
    dataset: SituatedPartnerDataset,
    *,
    intervention: str,
    seed: int = 1,
) -> SituatedPartnerDataset:
    if intervention not in SITUATED_INTERVENTIONS:
        raise ValueError(f"Unknown situated partner intervention: {intervention}.")
    if intervention == "original":
        return dataset
    features = np.asarray(dataset.features).copy()
    current_width = 4
    final_width = 4
    final_slice = slice(current_width, current_width + final_width)
    delta_start = current_width + final_width
    delta_slice = slice(delta_start, delta_start + 4)
    rng = np.random.default_rng(seed)
    if intervention == "shuffle_delta":
        flat = features[:, :, delta_slice].reshape((-1, 4))
        rng.shuffle(flat)
        features[:, :, delta_slice] = flat.reshape(features[:, :, delta_slice].shape)
    elif intervention == "reverse_delta_rank":
        values = np.asarray(dataset.option_values, dtype=np.float32)
        order = np.argsort(values, axis=1, kind="stable")
        reverse_order = order[:, ::-1]
        rows = np.arange(features.shape[0])[:, None]
        original = features[:, :, delta_slice].copy()
        features[rows, order, delta_slice] = original[rows, reverse_order]
    elif intervention == "shuffle_outcome":
        flat_final = features[:, :, final_slice].reshape((-1, 4))
        flat_delta = features[:, :, delta_slice].reshape((-1, 4))
        permutation = rng.permutation(flat_final.shape[0])
        features[:, :, final_slice] = flat_final[permutation].reshape(
            features[:, :, final_slice].shape
        )
        features[:, :, delta_slice] = flat_delta[permutation].reshape(
            features[:, :, delta_slice].shape
        )
    elif intervention == "reverse_outcome_rank":
        values = np.asarray(dataset.option_values, dtype=np.float32)
        order = np.argsort(values, axis=1, kind="stable")
        reverse_order = order[:, ::-1]
        rows = np.arange(features.shape[0])[:, None]
        original_final = features[:, :, final_slice].copy()
        original_delta = features[:, :, delta_slice].copy()
        features[rows, order, final_slice] = original_final[rows, reverse_order]
        features[rows, order, delta_slice] = original_delta[rows, reverse_order]
    elif intervention == "zero_outcome":
        features[:, :, final_slice] = 0.0
        features[:, :, delta_slice] = 0.0
    return SituatedPartnerDataset(
        features=mx.array(features, dtype=mx.float32),
        current_needs=dataset.current_needs,
        final_needs=dataset.final_needs,
        option_values=dataset.option_values,
        target_options=dataset.target_options,
        current_lowest=dataset.current_lowest,
    )


def _partner_proposals(
    dataset: SituatedPartnerDataset,
    *,
    mode: str,
) -> np.ndarray:
    if mode not in PARTNER_PROPOSAL_MODES:
        raise ValueError(f"Unknown partner proposal mode: {mode}.")
    values = np.asarray(dataset.option_values, dtype=np.float32)
    final_needs = np.asarray(dataset.final_needs, dtype=np.float32)
    if mode == "partial_body":
        partial_scores = np.min(final_needs[:, :, :2], axis=2)
        return np.argmax(partial_scores, axis=1).astype(np.int32)
    if mode == "partial_food_water":
        partial_scores = np.mean(final_needs[:, :, :2], axis=2)
        return np.argmax(partial_scores, axis=1).astype(np.int32)
    if mode == "partial_energy_safety":
        partial_scores = np.mean(final_needs[:, :, 2:], axis=2)
        return np.argmax(partial_scores, axis=1).astype(np.int32)
    if mode == "second_best":
        order = np.argsort(values, axis=1, kind="stable")
        return order[:, -2].astype(np.int32)
    if mode == "worst":
        return np.argmin(values, axis=1).astype(np.int32)
    raise ValueError(f"Unknown partner proposal mode: {mode}.")


def target_count_string(dataset: SituatedPartnerDataset) -> str:
    targets = np.asarray(dataset.target_options, dtype=np.int32)
    return ";".join(
        f"{name}={int(np.sum(targets == index))}"
        for index, name in enumerate(OPTION_NAMES)
    )


def format_result(result: SituatedPartnerResult) -> str:
    return ",".join(
        [
            result.model_control,
            result.intervention,
            str(result.samples),
            f"{result.proposal_accuracy:.4f}",
            f"{result.final_accuracy:.4f}",
            f"{result.changed_fraction:.4f}",
            f"{result.mean_proposal_delta:.6f}",
            f"{result.mean_chosen_delta:.6f}",
            f"{result.mean_oracle_delta:.6f}",
            f"{result.mean_regret:.6f}",
            result.target_counts,
        ]
    )


def main() -> None:
    args = _parse_args()
    _model, config = load_checkpoint(args.checkpoint)
    if args.renewable_resources:
        config = replace(config, renewable_resources=True)
    if args.resource_ecology is not None:
        config = replace(config, resource_ecology=args.resource_ecology)
    train_branches = collect_option_branch_dataset(
        config,
        episodes=args.train_episodes,
        seed=args.seed,
        teacher_mode=args.teacher_mode,
        horizon=args.horizon,
        state_policy=args.state_policy,
        max_samples=args.max_train_samples,
        option_action_noise=args.option_action_noise,
    )
    eval_branches = collect_option_branch_dataset(
        config,
        episodes=args.eval_episodes,
        seed=args.seed + 10_000,
        teacher_mode=args.teacher_mode,
        horizon=args.horizon,
        state_policy=args.state_policy,
        max_samples=args.max_eval_samples,
        option_action_noise=args.option_action_noise,
    )
    train_dataset = situated_partner_dataset_from_branches(
        train_branches,
        min_value_gap=args.min_value_gap,
        min_oracle_delta=args.min_oracle_delta,
    )
    eval_dataset = situated_partner_dataset_from_branches(
        eval_branches,
        min_value_gap=args.min_value_gap,
        min_oracle_delta=args.min_oracle_delta,
    )
    trained = train_situated_partner_dialogue(
        train_dataset,
        hidden_size=args.hidden_size,
        receiver_size=args.receiver_size,
        vocabulary_size=args.vocabulary_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        partner_mode=args.train_partner_mode,
        balance_weight=args.balance_weight,
        entropy_weight=args.entropy_weight,
        message_temperature=args.message_temperature,
        seed=args.seed,
    )
    print(
        "model_control,intervention,samples,proposal_accuracy,final_accuracy,"
        "changed_fraction,mean_proposal_delta,mean_chosen_delta,"
        "mean_oracle_delta,mean_regret,target_counts"
    )
    for intervention in args.interventions:
        intervened = intervene_situated_partner_features(
            eval_dataset,
            intervention=intervention,
            seed=args.seed + 20_000,
        )
        for partner_mode in args.partner_modes:
            print(
                format_result(
                    evaluate_situated_partner_dialogue(
                        trained,
                        intervened,
                        model_control=f"situated_{partner_mode}",
                        intervention=intervention,
                        partner_mode=partner_mode,
                    )
                )
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--train-episodes", type=int, default=300)
    parser.add_argument("--eval-episodes", type=int, default=150)
    parser.add_argument("--max-train-samples", type=int, default=6000)
    parser.add_argument("--max-eval-samples", type=int, default=3000)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--teacher-mode", choices=TEACHER_MODES, default="grounded")
    parser.add_argument("--state-policy", choices=STATE_POLICIES, default="cycle")
    parser.add_argument("--renewable-resources", action="store_true")
    parser.add_argument("--resource-ecology", choices=RESOURCE_ECOLOGIES, default=None)
    parser.add_argument("--option-action-noise", type=float, default=0.0)
    parser.add_argument("--min-value-gap", type=float, default=0.0)
    parser.add_argument("--min-oracle-delta", type=float, default=None)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--receiver-size", type=int, default=96)
    parser.add_argument("--vocabulary-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--train-partner-mode",
        choices=PARTNER_PROPOSAL_MODES,
        default="partial_body",
    )
    parser.add_argument("--balance-weight", type=float, default=0.02)
    parser.add_argument("--entropy-weight", type=float, default=0.0)
    parser.add_argument("--message-temperature", type=float, default=0.6)
    parser.add_argument(
        "--interventions",
        nargs="+",
        choices=SITUATED_INTERVENTIONS,
        default=["original", "shuffle_outcome", "reverse_outcome_rank"],
    )
    parser.add_argument(
        "--partner-modes",
        nargs="+",
        choices=PARTNER_PROPOSAL_MODES,
        default=[
            "partial_body",
            "partial_food_water",
            "partial_energy_safety",
            "second_best",
            "worst",
        ],
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
