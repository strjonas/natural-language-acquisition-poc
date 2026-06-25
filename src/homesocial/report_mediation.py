from __future__ import annotations

import argparse
from dataclasses import dataclass, replace

import mlx.core as mx
import numpy as np

from .attribution import WORLD_CAUSED, _needs_array
from .env import (
    Action,
    Direction,
    HomeostaticSocialGrid,
    Needs,
    SilentTeacher,
    WorldObject,
)
from .imitation import load_checkpoint
from .observations import observation_vector, observation_vector_size
from .recurrent_ac import RecurrentActorCritic
from .report_calibration import shuffled_report_dataset
from .report_head import (
    CONSEQUENCE_LABELS,
    NEED_LABELS,
    ReportDataset,
    TrainedSelfReport,
    _need_label,
    internal_report_features,
    predict_report,
    train_report_head,
)


REPORT_MODES = ("oracle", "learned", "shuffled", "majority")
TRIAGE_NEEDS = ("need_food", "need_water", "need_rest")
TRIAGE_OBJECTS = ("food", "water", "shelter", "danger")


@dataclass(frozen=True)
class MediatedReportResult:
    report_mode: str
    trials: int
    report_accuracy: float
    decision_accuracy: float
    helpful_use_rate: float
    irrelevant_use_rate: float
    target_need_delta: float
    mean_reward: float
    danger_hits: int


def collect_mediation_report_dataset(
    model,
    config,
    *,
    trials: int,
    seed: int,
) -> ReportDataset:
    features: list[np.ndarray] = []
    need_labels: list[int] = []
    for trial in range(trials):
        urgent_label = TRIAGE_NEEDS[trial % len(TRIAGE_NEEDS)]
        object_kind = TRIAGE_OBJECTS[trial % len(TRIAGE_OBJECTS)]
        env = _make_triage_env(
            config,
            urgent_label=urgent_label,
            object_kind=object_kind,
            seed=seed + trial,
        )
        observation = env._observe(None, None)
        vector = observation_vector(
            observation,
            width=env.width,
            height=env.height,
            include_language=config.include_language_channel,
            include_object_kinds=config.include_object_kinds,
            interoception_mode=config.interoception_mode,
            body_dynamics_mode=config.body_dynamics_mode,
        )
        features.append(
            internal_report_features(
                model,
                [vector],
                tuple(Action).index(Action.WAIT),
                current_needs=_needs_array(observation.needs),
            )
        )
        need_labels.append(_need_label(_needs_array(observation.needs)))

    sample_count = len(features)
    return ReportDataset(
        features=mx.array(np.stack(features), dtype=mx.float32),
        need_labels=mx.array(need_labels, dtype=mx.int32),
        cause_labels=mx.full((sample_count,), WORLD_CAUSED, dtype=mx.int32),
        consequence_labels=mx.full(
            (sample_count,),
            CONSEQUENCE_LABELS.index("no_major_change"),
            dtype=mx.int32,
        ),
    )


def teacher_response_to_report(need_index: int, obj: WorldObject) -> str:
    need = NEED_LABELS[need_index]
    desired_kind = {
        "need_food": "food",
        "need_water": "water",
        "need_rest": "shelter",
    }.get(need)
    if desired_kind == obj.kind:
        return {
            "food": "eat food",
            "water": "drink water",
            "shelter": "rest at shelter",
        }[obj.kind]
    return "avoid danger"


def evaluate_report_mediation(
    model,
    config,
    *,
    report_mode: str,
    reporter: TrainedSelfReport | None,
    majority_need: int,
    trials: int,
    seed: int,
) -> MediatedReportResult:
    if report_mode not in REPORT_MODES:
        raise ValueError(f"Unknown report mode: {report_mode}.")
    if report_mode in {"learned", "shuffled"} and reporter is None:
        raise ValueError(f"{report_mode} evaluation requires a trained reporter.")

    report_correct = 0
    decisions_correct = 0
    helpful_trials = 0
    helpful_uses = 0
    irrelevant_trials = 0
    irrelevant_uses = 0
    target_deltas: list[float] = []
    rewards: list[float] = []
    danger_hits = 0

    for trial in range(trials):
        urgent_label = TRIAGE_NEEDS[trial % len(TRIAGE_NEEDS)]
        object_kind = TRIAGE_OBJECTS[trial % len(TRIAGE_OBJECTS)]
        env = _make_triage_env(
            config,
            urgent_label=urgent_label,
            object_kind=object_kind,
            seed=seed + trial,
        )
        obj = env.object_ahead()
        observation = env._observe(None, None)
        obs_vectors = [
            observation_vector(
                observation,
                width=env.width,
                height=env.height,
                include_language=config.include_language_channel,
                include_object_kinds=config.include_object_kinds,
                interoception_mode=config.interoception_mode,
                body_dynamics_mode=config.body_dynamics_mode,
            )
        ]

        true_need = _need_label(_needs_array(observation.needs))
        reported_need = _reported_need(
            report_mode,
            reporter,
            model,
            obs_vectors,
            true_need=true_need,
            majority_need=majority_need,
            current_needs=_needs_array(observation.needs),
        )
        report_correct += int(reported_need == true_need)
        response = teacher_response_to_report(reported_need, obj)
        action = {
            "eat food": Action.CONSUME,
            "drink water": Action.CONSUME,
            "rest at shelter": Action.REST,
            "avoid danger": Action.WAIT,
        }[response]

        before_target = _target_need_value(observation.needs, urgent_label)
        next_observation, reward, _terminated, _truncated, info = env.step(action)
        after_target = _target_need_value(next_observation.needs, urgent_label)
        target_deltas.append(after_target - before_target)
        rewards.append(float(reward))

        helpful = _need_to_object(urgent_label) == object_kind
        used = info["event"] in {
            "consumed_food",
            "consumed_water",
            "rested_shelter",
        }
        decisions_correct += int(used == helpful)
        if helpful:
            helpful_trials += 1
            helpful_uses += int(used)
        else:
            irrelevant_trials += 1
            irrelevant_uses += int(used)
        danger_hits += int(
            info["event"] in {"hit_danger", "consumed_danger", "rested_danger"}
        )

    return MediatedReportResult(
        report_mode=report_mode,
        trials=trials,
        report_accuracy=report_correct / trials,
        decision_accuracy=decisions_correct / trials,
        helpful_use_rate=helpful_uses / max(1, helpful_trials),
        irrelevant_use_rate=irrelevant_uses / max(1, irrelevant_trials),
        target_need_delta=float(np.mean(target_deltas)),
        mean_reward=float(np.mean(rewards)),
        danger_hits=danger_hits,
    )


def _reported_need(
    report_mode: str,
    reporter: TrainedSelfReport | None,
    model,
    obs_vectors: list[np.ndarray],
    *,
    true_need: int,
    majority_need: int,
    current_needs: np.ndarray,
) -> int:
    if report_mode == "oracle":
        return true_need
    if report_mode == "majority":
        return majority_need
    wait_index = tuple(Action).index(Action.WAIT)
    features = internal_report_features(
        model,
        obs_vectors,
        wait_index,
        current_needs=current_needs,
    )
    need, _cause, _consequence = predict_report(reporter, features)
    return need


def _make_triage_env(
    config,
    *,
    urgent_label: str,
    object_kind: str,
    seed: int,
) -> HomeostaticSocialGrid:
    env = HomeostaticSocialGrid(
        width=config.width,
        height=config.height,
        seed=seed,
        max_steps=config.max_steps,
        teacher=SilentTeacher(),
        randomize_world=config.randomize_world,
        diagnostic_mode=config.diagnostic_mode,
        body_dynamics_mode=config.body_dynamics_mode,
    )
    env.reset(seed=seed)
    target_index = next(
        index for index, obj in enumerate(env.objects) if obj.kind == object_kind
    )
    target = replace(env.objects[target_index], pos=(2, 1))
    env.objects = [
        target if index == target_index else obj
        for index, obj in enumerate(env.objects)
        if index == target_index or obj.pos not in {(1, 1), (2, 1)}
    ]
    env.agent_pos = (1, 1)
    env.direction = Direction.EAST
    _set_urgent_need(env, urgent_label)
    return env


def _set_urgent_need(env: HomeostaticSocialGrid, urgent_label: str) -> None:
    env.needs = Needs(
        food=0.45 if urgent_label == "need_food" else 0.95,
        water=0.45 if urgent_label == "need_water" else 0.95,
        energy=0.45 if urgent_label == "need_rest" else 0.95,
        safety=0.95,
    )


def _need_to_object(need_label: str) -> str:
    return {
        "need_food": "food",
        "need_water": "water",
        "need_rest": "shelter",
    }[need_label]


def _target_need_value(needs: Needs, need_label: str) -> float:
    return {
        "need_food": needs.food,
        "need_water": needs.water,
        "need_rest": needs.energy,
    }[need_label]


def _majority_need(dataset: ReportDataset) -> int:
    labels = np.asarray(dataset.need_labels, dtype=np.int32)
    return int(np.argmax(np.bincount(labels)))


def main() -> None:
    args = _parse_args()
    model, config = load_checkpoint(args.checkpoint)
    model_control = "trained"
    if args.random_model_control:
        mx.random.seed(args.seed)
        model = RecurrentActorCritic(
            observation_vector_size(
                include_language=config.include_language_channel,
                include_object_kinds=config.include_object_kinds,
                body_dynamics_mode=config.body_dynamics_mode,
            ),
            config.hidden_size,
            len(Action),
        )
        model_control = "random"
    train_dataset = collect_mediation_report_dataset(
        model,
        config,
        trials=args.train_trials,
        seed=args.seed,
    )
    learned_reporter = train_report_head(
        train_dataset,
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    shuffled_train = shuffled_report_dataset(
        train_dataset,
        seed=args.seed + 77_000,
    )
    shuffled_reporter = train_report_head(
        shuffled_train,
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    majority_need = _majority_need(train_dataset)

    print(
        "model_control,report_mode,trials,report_accuracy,decision_accuracy,helpful_use_rate,"
        "irrelevant_use_rate,target_need_delta,mean_reward,danger_hits"
    )
    for report_mode in args.report_modes:
        reporter = {
            "learned": learned_reporter,
            "shuffled": shuffled_reporter,
        }.get(report_mode)
        result = evaluate_report_mediation(
            model,
            config,
            report_mode=report_mode,
            reporter=reporter,
            majority_need=majority_need,
            trials=args.eval_trials,
            seed=args.seed + 10_000,
        )
        print(
            ",".join(
                [
                    model_control,
                    report_mode,
                    str(result.trials),
                    f"{result.report_accuracy:.4f}",
                    f"{result.decision_accuracy:.4f}",
                    f"{result.helpful_use_rate:.4f}",
                    f"{result.irrelevant_use_rate:.4f}",
                    f"{result.target_need_delta:.4f}",
                    f"{result.mean_reward:.4f}",
                    str(result.danger_hits),
                ]
            )
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=1901)
    parser.add_argument("--train-trials", type=int, default=1200)
    parser.add_argument("--eval-trials", type=int, default=240)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--random-model-control",
        action="store_true",
        help="Use the checkpoint configuration with freshly initialized model weights.",
    )
    parser.add_argument(
        "--report-modes",
        nargs="+",
        choices=REPORT_MODES,
        default=list(REPORT_MODES),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
