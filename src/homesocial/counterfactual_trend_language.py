from __future__ import annotations

import argparse
from copy import deepcopy

import mlx.core as mx
import numpy as np

from .agents import TeacherFollowingAgent
from .attribution import _needs_array
from .emergent_language import _intent
from .env import Action, HomeostaticSocialGrid
from .imitation import load_checkpoint
from .observations import observation_vector, observation_vector_size
from .recurrent_ac import RecurrentActorCritic, RecurrentConfig, action_mask
from .report_head import _need_label
from .self_state_language import (
    BALANCE_TARGETS,
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


BRANCH_ACTIONS = (
    Action.MOVE_FORWARD,
    Action.CONSUME,
    Action.REST,
    Action.WAIT,
)


def collect_counterfactual_trend_dataset(
    base_model: RecurrentActorCritic,
    config: RecurrentConfig,
    *,
    episodes: int,
    seed: int,
    teacher_mode: str = "grounded",
    balance_target: str = "trend",
    max_states: int = 3000,
) -> SelfStateDataset:
    if balance_target not in BALANCE_TARGETS:
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
    rng = np.random.default_rng(seed + 2_510_000)
    features: list[np.ndarray] = []
    needs: list[np.ndarray] = []
    low_flags: list[np.ndarray] = []
    dominant_labels: list[int] = []
    severity_labels: list[int] = []
    trend_labels: list[int] = []

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
            valid_mask = action_mask(observation)
            for action in BRANCH_ACTIONS:
                action_index = tuple(Action).index(action)
                if valid_mask[action_index] <= 0.0:
                    continue
                branch = deepcopy(env)
                branch_observation, _reward, _terminated, _truncated, _info = (
                    branch.step(action)
                )
                after_needs = _needs_array(branch_observation.needs)
                features.append(
                    _counterfactual_features(
                        base_model,
                        history,
                        action_index=action_index,
                    )
                )
                needs.append(after_needs)
                low_flags.append((after_needs < 0.60).astype(np.int32))
                dominant_labels.append(_intent(_need_label(after_needs)))
                severity_labels.append(_severity(float(np.min(after_needs))))
                trend_labels.append(_trend(before_needs, after_needs))
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
            if valid_mask[action_index] <= 0.0:
                action = Action.MOVE_FORWARD
            observation, _reward, terminated, truncated, _info = env.step(action)
        if len(features) >= max_states:
            break

    indices = np.arange(len(features))
    if balance_target == "trend":
        indices = _balanced_label_indices(trend_labels, rng)
    elif balance_target == "intent":
        indices = _balanced_label_indices(dominant_labels, rng)

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
    )


def _counterfactual_features(
    base_model: RecurrentActorCritic,
    history: list[np.ndarray],
    *,
    action_index: int,
) -> np.ndarray:
    observations = mx.array(np.stack(history), dtype=mx.float32)
    wait_actions = np.zeros(len(history), dtype=np.int32)
    wait_actions[-1] = tuple(Action).index(Action.WAIT)
    branch_actions = np.zeros(len(history), dtype=np.int32)
    branch_actions[-1] = action_index
    _wait_obs, current_needs, _wait_rewards, _wait_utterances = (
        base_model.predict_consequences(
            observations,
            mx.array(wait_actions, dtype=mx.int32),
        )
    )
    _branch_obs, branch_needs, _branch_rewards, _branch_utterances = (
        base_model.predict_consequences(
            observations,
            mx.array(branch_actions, dtype=mx.int32),
        )
    )
    current = np.asarray(current_needs[-1], dtype=np.float32)
    branch = np.asarray(branch_needs[-1], dtype=np.float32)
    return np.concatenate([branch, branch - current]).astype(np.float32)


def _print_counts(label: str, dataset: SelfStateDataset) -> None:
    trends = np.asarray(dataset.trend_labels)
    counts = [int(np.sum(trends == index)) for index in range(len(TREND_LABELS))]
    print(f"# {label}_trend_counts=" + "/".join(str(value) for value in counts))


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
        train_dataset = collect_counterfactual_trend_dataset(
            base_model,
            config,
            episodes=args.train_episodes,
            seed=args.seed,
            teacher_mode=args.teacher_mode,
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
        branch_eval = collect_counterfactual_trend_dataset(
            base_model,
            config,
            episodes=args.eval_episodes,
            seed=args.seed + 10_000,
            teacher_mode=args.teacher_mode,
            balance_target=args.balance_target,
            max_states=args.max_eval_states,
        )
        _print_counts(f"{model_control}_branch_eval", branch_eval)
        branch_result = evaluate_self_state_communication(
            trained,
            branch_eval,
            model_control=model_control,
            feature_mode="counterfactual_delta",
            history_mode="branch",
        )
        print(_row(model_control, "branch", branch_result))

        trajectory_eval = collect_self_state_dataset(
            base_model,
            config,
            episodes=args.eval_episodes,
            seed=args.seed + 10_000,
            teacher_mode=args.teacher_mode,
            history_mode="full",
            feature_mode="self_estimate_delta",
            balance_target=args.balance_target,
            max_states=args.max_eval_states,
        )
        _print_counts(f"{model_control}_trajectory_eval", trajectory_eval)
        trajectory_result = evaluate_self_state_communication(
            trained,
            trajectory_eval,
            model_control=model_control,
            feature_mode="counterfactual_delta",
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
    parser.add_argument("--seed", type=int, default=9301)
    parser.add_argument("--train-episodes", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--max-train-states", type=int, default=18000)
    parser.add_argument("--max-eval-states", type=int, default=8000)
    parser.add_argument("--population-size", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--receiver-size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=120)
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
        choices=BALANCE_TARGETS,
        default="trend",
    )
    parser.add_argument("--teacher-mode", default="grounded")
    parser.add_argument("--random-model-control", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
