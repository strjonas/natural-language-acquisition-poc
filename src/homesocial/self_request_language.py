from __future__ import annotations

import argparse
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .emergent_language import (
    FEATURE_MODES,
    INTENT_LABELS,
    MESSAGE_VOCABULARY,
    CommunicationDataset,
    _codebook,
    _intent,
    _purity,
    collect_communication_dataset,
)
from .imitation import load_checkpoint
from .population_language import (
    PopulationSender,
    _agreement_loss,
    _dominant_intent_pairs,
    _intent_distinctness,
    _intent_pair_agreement,
    _intent_pair_table,
)


@dataclass(frozen=True)
class RequestDataset:
    features: mx.array
    intents: mx.array


@dataclass(frozen=True)
class TrainedSelfRequestPopulation:
    model: SelfRequestPopulation
    feature_mean: mx.array
    feature_std: mx.array
    agreement_weight: float


@dataclass(frozen=True)
class SelfRequestResult:
    agreement_weight: float
    population_size: int
    samples_per_sender: int
    mean_request_accuracy: float
    min_request_accuracy: float
    pooled_pair_intent_purity: float
    pair_conflict_rate: float
    intent_pair_agreement: float
    intent_distinctness: float
    message_pairs_used: int
    sender_codebooks: str
    intent_pair_table: str


class SelfRequestPopulation(nn.Module):
    def __init__(
        self,
        input_size: int,
        *,
        population_size: int = 4,
        hidden_size: int = 64,
        receiver_size: int = 48,
        vocabulary_size: int = MESSAGE_VOCABULARY,
    ) -> None:
        super().__init__()
        self.population_size = population_size
        self.vocabulary_size = vocabulary_size
        self.senders = [
            PopulationSender(
                input_size,
                hidden_size=hidden_size,
                vocabulary_size=vocabulary_size,
            )
            for _ in range(population_size)
        ]
        receiver_input = 2 * vocabulary_size
        self.receiver = nn.Linear(receiver_input, receiver_size)
        self.receiver_hidden = nn.Linear(receiver_size, receiver_size)
        self.request = nn.Linear(receiver_size, len(INTENT_LABELS))

    def message(
        self,
        sender_index: int,
        features: mx.array,
        *,
        temperature: float = 0.6,
        hard: bool = True,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        return self.senders[sender_index](
            features,
            temperature=temperature,
            hard=hard,
        )

    def receive(self, message1: mx.array, message2: mx.array) -> mx.array:
        inputs = mx.concatenate([message1, message2], axis=-1)
        hidden = nn.relu(self.receiver(inputs))
        hidden = hidden + nn.relu(self.receiver_hidden(hidden))
        return self.request(hidden)

    def __call__(
        self,
        sender_index: int,
        features: mx.array,
        *,
        temperature: float = 0.6,
    ) -> tuple[mx.array, mx.array, mx.array]:
        message1, message2, probs1, probs2 = self.message(
            sender_index,
            features,
            temperature=temperature,
            hard=True,
        )
        return self.receive(message1, message2), probs1, probs2


def request_dataset_from_communication(
    dataset: CommunicationDataset,
) -> RequestDataset:
    state_rows = np.asarray(dataset.objects) == 0
    features = np.asarray(dataset.features)[state_rows]
    intents = np.asarray(
        [_intent(label) for label in np.asarray(dataset.need_labels)[state_rows]]
    )
    return RequestDataset(
        features=mx.array(features, dtype=mx.float32),
        intents=mx.array(intents, dtype=mx.int32),
    )


def train_self_request_population(
    dataset: RequestDataset,
    *,
    population_size: int = 4,
    hidden_size: int = 64,
    receiver_size: int = 48,
    epochs: int = 45,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    agreement_weight: float = 0.0,
    balance_weight: float = 0.1,
    entropy_weight: float = 0.01,
    seed: int = 1,
) -> TrainedSelfRequestPopulation:
    rng = np.random.default_rng(seed)
    mx.random.seed(seed)
    model = SelfRequestPopulation(
        int(dataset.features.shape[-1]),
        population_size=population_size,
        hidden_size=hidden_size,
        receiver_size=receiver_size,
    )
    optimizer = optim.Adam(learning_rate=learning_rate)
    feature_mean = mx.mean(dataset.features, axis=0, keepdims=True)
    feature_std = mx.sqrt(
        mx.mean((dataset.features - feature_mean) ** 2, axis=0, keepdims=True)
        + 1e-6
    )
    features = (dataset.features - feature_mean) / feature_std
    sample_count = int(features.shape[0])
    indices = np.arange(sample_count)

    def loss_fn(batch_features: mx.array, batch_intents: mx.array) -> mx.array:
        request_loss = mx.array(0.0)
        balance_loss = mx.array(0.0)
        entropy = mx.array(0.0)
        all_probs1: list[mx.array] = []
        all_probs2: list[mx.array] = []

        for sender_index in range(population_size):
            logits, probs1, probs2 = model(sender_index, batch_features)
            log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            selected = mx.sum(
                log_probs * mx.eye(len(INTENT_LABELS))[batch_intents],
                axis=-1,
            )
            request_loss = request_loss - mx.mean(selected)

            uniform = 1.0 / MESSAGE_VOCABULARY
            balance_loss = balance_loss + mx.sum(
                (mx.mean(probs1, axis=0) - uniform) ** 2
            )
            balance_loss = balance_loss + mx.sum(
                (mx.mean(probs2, axis=0) - uniform) ** 2
            )
            entropy = entropy - mx.mean(
                mx.sum(probs1 * mx.log(probs1 + 1e-8), axis=-1)
            )
            entropy = entropy - mx.mean(
                mx.sum(probs2 * mx.log(probs2 + 1e-8), axis=-1)
            )
            all_probs1.append(probs1)
            all_probs2.append(probs2)

        request_loss = request_loss / population_size
        balance_loss = balance_loss / population_size
        entropy = entropy / population_size
        agreement_loss = _agreement_loss(all_probs1, all_probs2)
        return (
            request_loss
            + balance_weight * balance_loss
            + entropy_weight * entropy
            + agreement_weight * agreement_loss
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    for _epoch in range(max(1, epochs)):
        rng.shuffle(indices)
        for start in range(0, sample_count, max(1, batch_size)):
            batch = mx.array(
                indices[start : start + max(1, batch_size)],
                dtype=mx.int32,
            )
            loss, grads = loss_and_grad(features[batch], dataset.intents[batch])
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

    return TrainedSelfRequestPopulation(
        model=model,
        feature_mean=feature_mean,
        feature_std=feature_std,
        agreement_weight=agreement_weight,
    )


def evaluate_self_request_population(
    trained: TrainedSelfRequestPopulation,
    dataset: RequestDataset,
) -> SelfRequestResult:
    features = (dataset.features - trained.feature_mean) / trained.feature_std
    intents = np.asarray(dataset.intents)
    accuracies: list[float] = []
    pooled_pairs: list[np.ndarray] = []
    pooled_intents: list[np.ndarray] = []
    sender_codebooks: list[str] = []
    intent_pair_rows: list[np.ndarray] = []

    for sender_index in range(trained.model.population_size):
        message1, message2, _soft1, _soft2 = trained.model.message(
            sender_index,
            features,
            hard=True,
        )
        logits = trained.model.receive(message1, message2)
        predictions = np.asarray(mx.argmax(logits, axis=-1))
        token1 = np.asarray(mx.argmax(message1, axis=-1))
        token2 = np.asarray(mx.argmax(message2, axis=-1))
        pairs = token1 * MESSAGE_VOCABULARY + token2

        accuracies.append(float(np.mean(predictions == intents)))
        pooled_pairs.append(pairs)
        pooled_intents.append(intents)
        sender_codebooks.append(f"s{sender_index}:{_codebook(pairs, intents)}")
        intent_pair_rows.append(_dominant_intent_pairs(pairs, intents))

    all_pairs = np.concatenate(pooled_pairs)
    all_intents = np.concatenate(pooled_intents)
    intent_pair_matrix = np.stack(intent_pair_rows)
    pooled_purity = _purity(all_pairs, all_intents)
    return SelfRequestResult(
        agreement_weight=trained.agreement_weight,
        population_size=trained.model.population_size,
        samples_per_sender=len(intents),
        mean_request_accuracy=float(np.mean(accuracies)),
        min_request_accuracy=float(np.min(accuracies)),
        pooled_pair_intent_purity=pooled_purity,
        pair_conflict_rate=1.0 - pooled_purity,
        intent_pair_agreement=_intent_pair_agreement(intent_pair_matrix),
        intent_distinctness=_intent_distinctness(intent_pair_matrix),
        message_pairs_used=int(len(np.unique(all_pairs))),
        sender_codebooks=";".join(sender_codebooks),
        intent_pair_table=_intent_pair_table(intent_pair_matrix),
    )


def _result_row(result: SelfRequestResult) -> str:
    return ",".join(
        [
            f"{result.agreement_weight:.4f}",
            str(result.population_size),
            str(result.samples_per_sender),
            f"{result.mean_request_accuracy:.4f}",
            f"{result.min_request_accuracy:.4f}",
            f"{result.pooled_pair_intent_purity:.4f}",
            f"{result.pair_conflict_rate:.4f}",
            f"{result.intent_pair_agreement:.4f}",
            f"{result.intent_distinctness:.4f}",
            str(result.message_pairs_used),
            result.sender_codebooks,
            result.intent_pair_table,
        ]
    )


def main() -> None:
    args = _parse_args()
    base_model, config = load_checkpoint(args.checkpoint)
    train_source = collect_communication_dataset(
        base_model,
        config,
        episodes=args.train_episodes,
        seed=args.seed,
        teacher_mode=args.teacher_mode,
        history_mode="full",
        feature_mode=args.feature_mode,
        balance_intents=True,
        max_states=args.max_train_states,
    )
    eval_source = collect_communication_dataset(
        base_model,
        config,
        episodes=args.eval_episodes,
        seed=args.seed + 10_000,
        teacher_mode=args.teacher_mode,
        history_mode="full",
        feature_mode=args.feature_mode,
        balance_intents=True,
        max_states=args.max_eval_states,
    )
    train_dataset = request_dataset_from_communication(train_source)
    eval_dataset = request_dataset_from_communication(eval_source)

    print(
        "agreement_weight,population_size,samples_per_sender,"
        "mean_request_accuracy,min_request_accuracy,pooled_pair_intent_purity,"
        "pair_conflict_rate,intent_pair_agreement,intent_distinctness,"
        "message_pairs_used,sender_codebooks,intent_pair_table"
    )
    for agreement_weight in args.agreement_weights:
        trained = train_self_request_population(
            train_dataset,
            population_size=args.population_size,
            hidden_size=args.hidden_size,
            receiver_size=args.receiver_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            agreement_weight=agreement_weight,
            balance_weight=args.balance_weight,
            entropy_weight=args.entropy_weight,
            seed=args.seed,
        )
        result = evaluate_self_request_population(trained, eval_dataset)
        print(_result_row(result))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=6301)
    parser.add_argument("--train-episodes", type=int, default=220)
    parser.add_argument("--eval-episodes", type=int, default=120)
    parser.add_argument("--max-train-states", type=int, default=3600)
    parser.add_argument("--max-eval-states", type=int, default=2000)
    parser.add_argument("--population-size", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--receiver-size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--balance-weight", type=float, default=0.1)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument(
        "--agreement-weights",
        nargs="+",
        type=float,
        default=[0.0, 0.05, 0.2],
    )
    parser.add_argument("--teacher-mode", default="grounded")
    parser.add_argument(
        "--feature-mode",
        choices=FEATURE_MODES,
        default="self_estimate",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
