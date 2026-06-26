from __future__ import annotations

import argparse
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .emergent_language import (
    FEATURE_MODES,
    MESSAGE_VOCABULARY,
    OBJECT_KINDS,
    CommunicationDataset,
    TrainedCommunication,
    _balanced_accuracy,
    _precision,
    _recall,
    collect_communication_dataset,
    evaluate_communication,
    train_communication,
)
from .imitation import load_checkpoint


@dataclass(frozen=True)
class ReceiverTransferResult:
    condition: str
    train_samples: int
    samples: int
    decision_accuracy: float
    balanced_accuracy: float
    use_precision: float
    use_recall: float


class MessageReceiver(nn.Module):
    def __init__(
        self,
        receiver_size: int = 48,
        vocabulary_size: int = MESSAGE_VOCABULARY,
    ) -> None:
        super().__init__()
        self.vocabulary_size = vocabulary_size
        receiver_input = 2 * vocabulary_size + len(OBJECT_KINDS)
        self.receiver = nn.Linear(receiver_input, receiver_size)
        self.receiver_hidden = nn.Linear(receiver_size, receiver_size)
        self.decision = nn.Linear(receiver_size, 2)

    def __call__(
        self,
        token1: mx.array,
        token2: mx.array,
        objects: mx.array,
    ) -> mx.array:
        message1 = mx.eye(self.vocabulary_size)[token1]
        message2 = mx.eye(self.vocabulary_size)[token2]
        object_features = mx.eye(len(OBJECT_KINDS))[objects]
        inputs = mx.concatenate([message1, message2, object_features], axis=-1)
        hidden = nn.relu(self.receiver(inputs))
        hidden = hidden + nn.relu(self.receiver_hidden(hidden))
        return self.decision(hidden)


def train_receiver_for_sender(
    sender: TrainedCommunication,
    dataset: CommunicationDataset,
    *,
    train_samples: int,
    receiver_size: int = 48,
    epochs: int = 40,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    seed: int = 1,
) -> MessageReceiver:
    rng = np.random.default_rng(seed)
    mx.random.seed(seed)
    token1, token2 = sender_message_tokens(sender, dataset)
    sample_count = int(dataset.decisions.shape[0])
    indices = np.arange(sample_count)
    rng.shuffle(indices)
    train_indices = indices[: min(max(1, train_samples), sample_count)]

    model = MessageReceiver(receiver_size=receiver_size)
    optimizer = optim.Adam(learning_rate=learning_rate)

    token1_array = mx.array(token1, dtype=mx.int32)
    token2_array = mx.array(token2, dtype=mx.int32)

    def loss_fn(
        batch_token1: mx.array,
        batch_token2: mx.array,
        batch_objects: mx.array,
        batch_decisions: mx.array,
    ) -> mx.array:
        logits = model(batch_token1, batch_token2, batch_objects)
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        selected = mx.sum(log_probs * mx.eye(2)[batch_decisions], axis=-1)
        weights = mx.where(batch_decisions == 1, 3.0, 1.0)
        return -mx.sum(selected * weights) / mx.sum(weights)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    for _epoch in range(max(1, epochs)):
        rng.shuffle(train_indices)
        for start in range(0, len(train_indices), max(1, batch_size)):
            batch_np = train_indices[start : start + max(1, batch_size)]
            batch = mx.array(batch_np, dtype=mx.int32)
            loss, grads = loss_and_grad(
                token1_array[batch],
                token2_array[batch],
                dataset.objects[batch],
                dataset.decisions[batch],
            )
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
    return model


def evaluate_receiver_transfer(
    sender: TrainedCommunication,
    receiver: MessageReceiver,
    dataset: CommunicationDataset,
    *,
    condition: str,
    train_samples: int,
) -> ReceiverTransferResult:
    token1, token2 = sender_message_tokens(sender, dataset)
    logits = receiver(
        mx.array(token1, dtype=mx.int32),
        mx.array(token2, dtype=mx.int32),
        dataset.objects,
    )
    predictions = np.asarray(mx.argmax(logits, axis=-1))
    targets = np.asarray(dataset.decisions)
    return ReceiverTransferResult(
        condition=condition,
        train_samples=train_samples,
        samples=len(targets),
        decision_accuracy=float(np.mean(predictions == targets)),
        balanced_accuracy=_balanced_accuracy(targets, predictions),
        use_precision=_precision(targets, predictions),
        use_recall=_recall(targets, predictions),
    )


def evaluate_sender_receiver_pair(
    sender: TrainedCommunication,
    receiver: TrainedCommunication,
    dataset: CommunicationDataset,
    *,
    condition: str,
) -> ReceiverTransferResult:
    normalized = (dataset.features - sender.feature_mean) / sender.feature_std
    message1, message2, _soft1, _soft2 = sender.model.message(
        normalized,
        hard=True,
    )
    logits = receiver.model.receive(message1, message2, dataset.objects)
    predictions = np.asarray(mx.argmax(logits, axis=-1))
    targets = np.asarray(dataset.decisions)
    return ReceiverTransferResult(
        condition=condition,
        train_samples=0,
        samples=len(targets),
        decision_accuracy=float(np.mean(predictions == targets)),
        balanced_accuracy=_balanced_accuracy(targets, predictions),
        use_precision=_precision(targets, predictions),
        use_recall=_recall(targets, predictions),
    )


def sender_message_tokens(
    sender: TrainedCommunication,
    dataset: CommunicationDataset,
) -> tuple[np.ndarray, np.ndarray]:
    normalized = (dataset.features - sender.feature_mean) / sender.feature_std
    message1, message2, _soft1, _soft2 = sender.model.message(
        normalized,
        hard=True,
    )
    return (
        np.asarray(mx.argmax(message1, axis=-1)),
        np.asarray(mx.argmax(message2, axis=-1)),
    )


def _result_row(
    result: ReceiverTransferResult,
    *,
    source_codebook: str,
    target_codebook: str,
) -> str:
    return ",".join(
        [
            result.condition,
            str(result.train_samples),
            str(result.samples),
            f"{result.decision_accuracy:.4f}",
            f"{result.balanced_accuracy:.4f}",
            f"{result.use_precision:.4f}",
            f"{result.use_recall:.4f}",
            source_codebook,
            target_codebook,
        ]
    )


def main() -> None:
    args = _parse_args()
    base_model, config = load_checkpoint(args.checkpoint)
    protocol_dataset = collect_communication_dataset(
        base_model,
        config,
        episodes=args.protocol_train_episodes,
        seed=args.source_seed,
        teacher_mode=args.teacher_mode,
        history_mode="full",
        feature_mode=args.feature_mode,
        balance_intents=True,
        max_states=args.max_protocol_train_states,
    )
    source = train_communication(
        protocol_dataset,
        hidden_size=args.hidden_size,
        receiver_size=args.receiver_size,
        epochs=args.protocol_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.source_seed,
    )
    target = train_communication(
        protocol_dataset,
        hidden_size=args.hidden_size,
        receiver_size=args.receiver_size,
        epochs=args.protocol_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.target_seed,
    )
    eval_dataset = collect_communication_dataset(
        base_model,
        config,
        episodes=args.eval_episodes,
        seed=args.source_seed + 10_000,
        teacher_mode=args.teacher_mode,
        history_mode="full",
        feature_mode=args.feature_mode,
        balance_intents=True,
        max_states=args.max_eval_states,
    )
    receiver_train_dataset = collect_communication_dataset(
        base_model,
        config,
        episodes=args.receiver_train_episodes,
        seed=args.source_seed + 20_000,
        teacher_mode=args.teacher_mode,
        history_mode="full",
        feature_mode=args.feature_mode,
        balance_intents=True,
        max_states=args.max_receiver_train_states,
    )
    source_eval = evaluate_communication(
        source,
        eval_dataset,
        model_control="trained",
        feature_mode=args.feature_mode,
        history_mode="full",
        seed=args.source_seed + 30_000,
    )
    target_eval = evaluate_communication(
        target,
        eval_dataset,
        model_control="trained",
        feature_mode=args.feature_mode,
        history_mode="full",
        seed=args.target_seed + 30_000,
    )

    print(
        "condition,train_samples,samples,decision_accuracy,balanced_accuracy,"
        "use_precision,use_recall,source_codebook,target_codebook"
    )
    baseline_results = [
        evaluate_sender_receiver_pair(
            source,
            source,
            eval_dataset,
            condition="source_own_pair",
        ),
        evaluate_sender_receiver_pair(
            target,
            target,
            eval_dataset,
            condition="target_own_pair",
        ),
        evaluate_sender_receiver_pair(
            source,
            target,
            eval_dataset,
            condition="source_sender_target_receiver",
        ),
        evaluate_sender_receiver_pair(
            target,
            source,
            eval_dataset,
            condition="target_sender_source_receiver",
        ),
    ]
    for result in baseline_results:
        print(
            _result_row(
                result,
                source_codebook=source_eval.codebook,
                target_codebook=target_eval.codebook,
            )
        )

    for train_samples in args.transfer_samples:
        receiver = train_receiver_for_sender(
            source,
            receiver_train_dataset,
            train_samples=train_samples,
            receiver_size=args.receiver_size,
            epochs=args.receiver_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.source_seed + train_samples,
        )
        result = evaluate_receiver_transfer(
            source,
            receiver,
            eval_dataset,
            condition="frozen_sender_new_receiver",
            train_samples=train_samples,
        )
        print(
            _result_row(
                result,
                source_codebook=source_eval.codebook,
                target_codebook=target_eval.codebook,
            )
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-seed", type=int, default=4301)
    parser.add_argument("--target-seed", type=int, default=4302)
    parser.add_argument("--protocol-train-episodes", type=int, default=220)
    parser.add_argument("--receiver-train-episodes", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=120)
    parser.add_argument("--max-protocol-train-states", type=int, default=3600)
    parser.add_argument("--max-receiver-train-states", type=int, default=2000)
    parser.add_argument("--max-eval-states", type=int, default=2000)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--receiver-size", type=int, default=48)
    parser.add_argument("--protocol-epochs", type=int, default=45)
    parser.add_argument("--receiver-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--teacher-mode", default="grounded")
    parser.add_argument(
        "--feature-mode",
        choices=FEATURE_MODES,
        default="self_estimate",
    )
    parser.add_argument(
        "--transfer-samples",
        nargs="+",
        type=int,
        default=[32, 64, 128, 256, 512, 1024],
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
