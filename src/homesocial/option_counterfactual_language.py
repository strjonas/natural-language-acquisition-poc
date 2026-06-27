from __future__ import annotations

import argparse
from copy import deepcopy

import mlx.core as mx
import numpy as np

from .agents import TeacherFollowingAgent
from .attribution import _needs_array
from .emergent_language import _intent
from .env import Action, HomeostaticSocialGrid, Observation, WorldObject
from .imitation import load_checkpoint
from .observations import observation_vector, observation_vector_size
from .recurrent_ac import RecurrentActorCritic, RecurrentConfig, action_mask
from .report_head import _need_label
from .self_state_language import (
    SelfStateDataset,
    TREND_LABELS,
    _balanced_label_indices,
    _severity,
    _trend,
    collect_self_state_dataset,
    evaluate_self_state_communication,
    train_self_state_communication,
)
from .teachers import build_teacher, masks_language, normalize_teacher_mode


OPTION_NAMES = ("seek_food", "seek_water", "seek_shelter", "rest", "wait")
OPTION_BALANCE_TARGETS = ("none", "trend", "option_trend")


def collect_option_counterfactual_dataset(
    base_model: RecurrentActorCritic,
    config: RecurrentConfig,
    *,
    episodes: int,
    seed: int,
    teacher_mode: str = "grounded",
    horizon: int = 6,
    balance_target: str = "option_trend",
    max_states: int = 3000,
) -> SelfStateDataset:
    if balance_target not in OPTION_BALANCE_TARGETS:
        raise ValueError(f"Unknown balance target: {balance_target}.")
    normalized_teacher = normalize_teacher_mode(teacher_mode)
    mask_language = (
        masks_language(normalized_teacher) or not config.include_language_channel
    )
    env = HomeostaticSocialGrid(
        width=config.width,
        height=config.height,
        seed=seed,
        max_steps=config.max_steps,
        teacher=build_teacher(normalized_teacher, seed=seed),
        randomize_world=config.randomize_world,
        diagnostic_mode=config.diagnostic_mode,
        body_dynamics_mode=config.body_dynamics_mode,
    )
    rng = np.random.default_rng(seed + 2_710_000)
    features: list[np.ndarray] = []
    needs: list[np.ndarray] = []
    low_flags: list[np.ndarray] = []
    dominant_labels: list[int] = []
    severity_labels: list[int] = []
    trend_labels: list[int] = []
    option_labels: list[int] = []

    for episode in range(episodes):
        observation = env.reset(seed=seed + episode)
        agent = TeacherFollowingAgent()
        history: list[np.ndarray] = []
        terminated = False
        truncated = False
        while not terminated and not truncated and len(features) < max_states:
            history.append(
                observation_vector(
                    observation,
                    width=env.width,
                    height=env.height,
                    include_language=config.include_language_channel,
                    mask_language=mask_language,
                    include_object_kinds=config.include_object_kinds,
                    interoception_mode=config.interoception_mode,
                    body_dynamics_mode=config.body_dynamics_mode,
                )
            )
            before_needs = _needs_array(observation.needs)
            for option_index, option_name in enumerate(OPTION_NAMES):
                branch = deepcopy(env)
                branch_observation = observation
                option_actions: list[int] = []
                branch_terminated = False
                branch_truncated = False
                for _step in range(max(1, horizon)):
                    option_action = _option_action(option_name, branch_observation)
                    mask = action_mask(branch_observation)
                    option_action_index = tuple(Action).index(option_action)
                    if mask[option_action_index] <= 0.0:
                        option_action = Action.MOVE_FORWARD
                        option_action_index = tuple(Action).index(option_action)
                    option_actions.append(option_action_index)
                    (
                        branch_observation,
                        _reward,
                        branch_terminated,
                        branch_truncated,
                        _info,
                    ) = branch.step(option_action)
                    if branch_terminated or branch_truncated:
                        break

                after_needs = _needs_array(branch_observation.needs)
                features.append(
                    _option_counterfactual_features(
                        base_model,
                        history,
                        option_actions=option_actions,
                    )
                )
                needs.append(after_needs)
                low_flags.append((after_needs < 0.60).astype(np.int32))
                dominant_labels.append(_intent(_need_label(after_needs)))
                severity_labels.append(_severity(float(np.min(after_needs))))
                trend_labels.append(_trend(before_needs, after_needs))
                option_labels.append(option_index)
                if len(features) >= max_states:
                    break

            policy_observation = observation
            if mask_language:
                policy_observation = observation.__class__(
                    step_count=observation.step_count,
                    position=observation.position,
                    direction=observation.direction,
                    needs=observation.needs,
                    visible=observation.visible,
                    object_ahead=observation.object_ahead,
                    teacher_utterance=None,
                    last_event=observation.last_event,
                )
            action = agent.act(policy_observation)
            action_index = tuple(Action).index(action)
            if action_mask(observation)[action_index] <= 0.0:
                action = Action.MOVE_FORWARD
            observation, _reward, terminated, truncated, _info = env.step(action)
        if len(features) >= max_states:
            break

    indices = np.arange(len(features))
    if balance_target == "trend":
        indices = _balanced_label_indices(trend_labels, rng)
    elif balance_target == "option_trend":
        indices = _balanced_option_trend_indices(option_labels, trend_labels, rng)

    return SelfStateDataset(
        features=mx.array(
            np.stack([features[int(index)] for index in indices]),
            dtype=mx.float32,
        ),
        needs=mx.array(
            np.stack([needs[int(index)] for index in indices]),
            dtype=mx.float32,
        ),
        low_flags=mx.array(
            np.stack([low_flags[int(index)] for index in indices]),
            dtype=mx.int32,
        ),
        dominant_labels=mx.array(
            [dominant_labels[int(index)] for index in indices],
            dtype=mx.int32,
        ),
        severity_labels=mx.array(
            [severity_labels[int(index)] for index in indices],
            dtype=mx.int32,
        ),
        trend_labels=mx.array(
            [trend_labels[int(index)] for index in indices],
            dtype=mx.int32,
        ),
        action_labels=mx.array(
            [option_labels[int(index)] for index in indices],
            dtype=mx.int32,
        ),
    )


def _option_action(option_name: str, observation: Observation) -> Action:
    if option_name == "rest":
        return Action.REST
    if option_name == "wait":
        return Action.WAIT
    if option_name == "seek_shelter":
        return _seek_kind_action("shelter", observation)
    if option_name == "seek_food":
        return _seek_kind_action("food", observation)
    if option_name == "seek_water":
        return _seek_kind_action("water", observation)
    raise ValueError(f"Unknown option: {option_name}.")


def _seek_kind_action(kind: str, observation: Observation) -> Action:
    ahead = observation.object_ahead
    if ahead is not None:
        if ahead.kind == kind:
            return Action.REST if kind == "shelter" else Action.CONSUME
        if ahead.kind == "danger":
            return Action.TURN_RIGHT

    target = _nearest_visible_kind(kind, observation)
    if target is None:
        if observation.last_event in {"bumped_wall", "blocked"}:
            return Action.TURN_RIGHT
        return Action.MOVE_FORWARD

    desired_direction = _desired_direction(observation.position, target.pos)
    if observation.direction.value != desired_direction:
        return Action.TURN_RIGHT
    return Action.MOVE_FORWARD


def _nearest_visible_kind(kind: str, observation: Observation) -> WorldObject | None:
    candidates = [obj for obj in observation.visible if obj.kind == kind]
    if not candidates:
        return None
    ax, ay = observation.position
    return min(candidates, key=lambda obj: abs(obj.pos[0] - ax) + abs(obj.pos[1] - ay))


def _desired_direction(current: tuple[int, int], target: tuple[int, int]) -> str:
    ax, ay = current
    tx, ty = target
    if abs(tx - ax) >= abs(ty - ay):
        return "east" if tx > ax else "west"
    return "south" if ty > ay else "north"


def _option_counterfactual_features(
    base_model: RecurrentActorCritic,
    history: list[np.ndarray],
    *,
    option_actions: list[int],
) -> np.ndarray:
    option_needs = _imagined_rollout_needs(base_model, history, option_actions)
    wait_action = tuple(Action).index(Action.WAIT)
    wait_needs = _imagined_rollout_needs(
        base_model,
        history,
        [wait_action for _ in option_actions],
    )
    return np.concatenate([option_needs, option_needs - wait_needs]).astype(np.float32)


def _imagined_rollout_needs(
    base_model: RecurrentActorCritic,
    history: list[np.ndarray],
    actions: list[int],
) -> np.ndarray:
    sequence = [np.asarray(item, dtype=np.float32) for item in history]
    final_needs: np.ndarray | None = None
    for action_index in actions:
        observations = mx.array(np.stack(sequence), dtype=mx.float32)
        rollout_actions = np.zeros(len(sequence), dtype=np.int32)
        rollout_actions[-1] = action_index
        predicted_observations, predicted_needs, _rewards, _utterances = (
            base_model.predict_consequences(
                observations,
                mx.array(rollout_actions, dtype=mx.int32),
            )
        )
        final_needs = np.asarray(predicted_needs[-1], dtype=np.float32)
        sequence.append(np.asarray(predicted_observations[-1], dtype=np.float32))
    if final_needs is None:
        raise ValueError("Cannot imagine an empty option rollout.")
    return final_needs


def _balanced_option_trend_indices(
    option_labels: list[int],
    trend_labels: list[int],
    rng: np.random.Generator,
) -> np.ndarray:
    sampled: list[np.ndarray] = []
    for option_index in sorted(set(option_labels)):
        option_buckets = []
        for trend_index in range(len(TREND_LABELS)):
            bucket = np.array(
                [
                    index
                    for index, (option, trend) in enumerate(
                        zip(option_labels, trend_labels, strict=True)
                    )
                    if option == option_index and trend == trend_index
                ],
                dtype=np.int64,
            )
            if len(bucket) == 0:
                option_buckets = []
                break
            option_buckets.append(bucket)
        if option_buckets:
            target = min(len(bucket) for bucket in option_buckets)
            sampled.extend(
                rng.choice(bucket, size=target, replace=False)
                for bucket in option_buckets
            )
    if not sampled:
        return _balanced_label_indices(trend_labels, rng)
    indices = np.concatenate(sampled)
    rng.shuffle(indices)
    return indices


def _print_counts(label: str, dataset: SelfStateDataset) -> None:
    trends = np.asarray(dataset.trend_labels)
    counts = [int(np.sum(trends == index)) for index in range(len(TREND_LABELS))]
    print(f"# {label}_trend_counts=" + "/".join(str(value) for value in counts))
    if dataset.action_labels is None:
        return
    options = np.asarray(dataset.action_labels)
    parts = []
    for option_index in sorted(int(value) for value in np.unique(options)):
        option_trends = trends[options == option_index]
        option_counts = [
            int(np.sum(option_trends == trend_index))
            for trend_index in range(len(TREND_LABELS))
        ]
        parts.append(
            f"{OPTION_NAMES[option_index]}:"
            + "/".join(str(value) for value in option_counts)
        )
    print(f"# {label}_option_trend_counts=" + ";".join(parts))


def main() -> None:
    args = _parse_args()
    trained_base, config = load_checkpoint(args.checkpoint)
    base_models = [("trained", trained_base)]
    if args.random_model_control:
        mx.random.seed(args.seed)
        base_models.append(
            (
                "random",
                RecurrentActorCritic(
                    observation_vector_size(
                        include_language=config.include_language_channel,
                        include_object_kinds=config.include_object_kinds,
                        body_dynamics_mode=config.body_dynamics_mode,
                    ),
                    config.hidden_size,
                    len(Action),
                ),
            )
        )

    print(
        "model_control,dataset,samples,need_mse,low_flag_accuracy,"
        "dominant_accuracy,severity_accuracy,trend_accuracy,"
        "exact_discrete_accuracy,dominant_code_agreement,"
        "dominant_code_distinctness,message_codes_used"
    )
    for model_control, base_model in base_models:
        train_dataset = collect_option_counterfactual_dataset(
            base_model,
            config,
            episodes=args.train_episodes,
            seed=args.seed,
            teacher_mode=args.teacher_mode,
            horizon=args.horizon,
            balance_target=args.balance_target,
            max_states=args.max_train_states,
        )
        _print_counts(f"{model_control}_train", train_dataset)
        trained = train_self_state_communication(
            train_dataset,
            population_size=args.population_size,
            hidden_size=args.hidden_size,
            receiver_size=args.receiver_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            agreement_weight=args.agreement_weight,
            balance_weight=args.balance_weight,
            entropy_weight=args.entropy_weight,
            need_weight=args.need_weight,
            low_weight=args.low_weight,
            dominant_weight=args.dominant_weight,
            severity_weight=args.severity_weight,
            trend_weight=args.trend_weight,
            seed=args.seed,
        )

        option_eval = collect_option_counterfactual_dataset(
            base_model,
            config,
            episodes=args.eval_episodes,
            seed=args.seed + 10_000,
            teacher_mode=args.teacher_mode,
            horizon=args.horizon,
            balance_target=args.balance_target,
            max_states=args.max_eval_states,
        )
        _print_counts(f"{model_control}_option_eval", option_eval)
        option_result = evaluate_self_state_communication(
            trained,
            option_eval,
            model_control=model_control,
            feature_mode="option_counterfactual_delta",
            history_mode=f"horizon_{args.horizon}",
        )
        print(_row(model_control, "option", option_result))

        trajectory_balance_target = (
            "trend" if args.balance_target == "option_trend" else args.balance_target
        )
        trajectory_eval = collect_self_state_dataset(
            base_model,
            config,
            episodes=args.eval_episodes,
            seed=args.seed + 10_000,
            teacher_mode=args.teacher_mode,
            history_mode="full",
            feature_mode="self_estimate_delta",
            balance_target=trajectory_balance_target,
            max_states=args.max_eval_states,
        )
        _print_counts(f"{model_control}_trajectory_eval", trajectory_eval)
        trajectory_result = evaluate_self_state_communication(
            trained,
            trajectory_eval,
            model_control=model_control,
            feature_mode="option_counterfactual_delta",
            history_mode="trajectory",
        )
        print(_row(model_control, "trajectory", trajectory_result))


def _row(model_control: str, dataset_name: str, result) -> str:
    return ",".join(
        [
            model_control,
            dataset_name,
            str(result.samples_per_sender),
            f"{result.need_mse:.6f}",
            f"{result.low_flag_accuracy:.4f}",
            f"{result.dominant_accuracy:.4f}",
            f"{result.severity_accuracy:.4f}",
            f"{result.trend_accuracy:.4f}",
            f"{result.exact_discrete_accuracy:.4f}",
            f"{result.dominant_code_agreement:.4f}",
            f"{result.dominant_code_distinctness:.4f}",
            str(result.message_codes_used),
        ]
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=9501)
    parser.add_argument("--train-episodes", type=int, default=800)
    parser.add_argument("--eval-episodes", type=int, default=400)
    parser.add_argument("--max-train-states", type=int, default=9000)
    parser.add_argument("--max-eval-states", type=int, default=4500)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--population-size", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--receiver-size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--agreement-weight", type=float, default=0.05)
    parser.add_argument("--balance-weight", type=float, default=0.02)
    parser.add_argument("--entropy-weight", type=float, default=0.0)
    parser.add_argument("--need-weight", type=float, default=2.0)
    parser.add_argument("--low-weight", type=float, default=1.0)
    parser.add_argument("--dominant-weight", type=float, default=1.0)
    parser.add_argument("--severity-weight", type=float, default=0.5)
    parser.add_argument("--trend-weight", type=float, default=0.5)
    parser.add_argument(
        "--balance-target",
        choices=OPTION_BALANCE_TARGETS,
        default="option_trend",
    )
    parser.add_argument("--teacher-mode", default="grounded")
    parser.add_argument("--random-model-control", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
