from __future__ import annotations

import argparse
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from .agents import TeacherFollowingAgent
from .env import Action, HomeostaticSocialGrid
from .imitation import load_checkpoint
from .observations import observation_vector, observation_vector_size
from .recurrent_ac import RecurrentActorCritic, RecurrentConfig, action_mask
from .teachers import build_teacher, masks_language, normalize_teacher_mode


HISTORY_MODES = ("full", "latest", "shuffled", "reversed")


@dataclass(frozen=True)
class InteroceptionResult:
    model_control: str
    history_mode: str
    samples: int
    next_need_mse: float
    food_mse: float
    water_mse: float
    energy_mse: float
    safety_mse: float
    lowest_need_accuracy: float


def evaluate_interoception(
    model: RecurrentActorCritic,
    config: RecurrentConfig,
    *,
    episodes: int,
    seed: int,
    history_mode: str = "full",
    teacher_mode: str = "grounded",
    model_control: str = "trained",
    max_samples: int | None = None,
) -> InteroceptionResult:
    if history_mode not in HISTORY_MODES:
        raise ValueError(f"Unknown history mode: {history_mode}.")
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
    rng = np.random.default_rng(seed + 910_000)
    predicted: list[np.ndarray] = []
    actual: list[np.ndarray] = []

    for episode in range(episodes):
        observation = env.reset(seed=seed + episode)
        agent = TeacherFollowingAgent()
        history: list[np.ndarray] = []
        terminated = False
        truncated = False

        while not terminated and not truncated:
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
            mask = action_mask(observation)
            if mask[action_index] <= 0.0:
                action = Action.MOVE_FORWARD
                action_index = tuple(Action).index(action)

            model_history = _history_control(history, history_mode, rng)
            observations = mx.array(np.stack(model_history), dtype=mx.float32)
            actions = np.zeros(len(model_history), dtype=np.int32)
            actions[-1] = action_index
            _next_obs, predicted_needs, _reward, _utterance = (
                model.predict_consequences(
                    observations,
                    mx.array(actions, dtype=mx.int32),
                )
            )
            next_observation, _reward_value, terminated, truncated, _info = env.step(
                action
            )
            predicted.append(np.asarray(predicted_needs[-1], dtype=np.float32))
            actual.append(
                np.asarray(
                    [
                        next_observation.needs.food,
                        next_observation.needs.water,
                        next_observation.needs.energy,
                        next_observation.needs.safety,
                    ],
                    dtype=np.float32,
                )
            )
            observation = next_observation
            if max_samples is not None and len(actual) >= max_samples:
                return _summarize(
                    predicted,
                    actual,
                    model_control=model_control,
                    history_mode=history_mode,
                )

    return _summarize(
        predicted,
        actual,
        model_control=model_control,
        history_mode=history_mode,
    )


def _history_control(
    history: list[np.ndarray],
    mode: str,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    if mode == "full":
        return history
    if mode == "latest" or len(history) <= 2:
        return [history[-1]]
    prior = list(history[:-1])
    if mode == "reversed":
        return list(reversed(prior)) + [history[-1]]
    permutation = rng.permutation(len(prior))
    return [prior[index] for index in permutation] + [history[-1]]


def _summarize(
    predicted: list[np.ndarray],
    actual: list[np.ndarray],
    *,
    model_control: str,
    history_mode: str,
) -> InteroceptionResult:
    if not actual:
        raise ValueError("Interoception evaluation produced no samples.")
    predicted_array = np.stack(predicted)
    actual_array = np.stack(actual)
    squared_error = (predicted_array - actual_array) ** 2
    return InteroceptionResult(
        model_control=model_control,
        history_mode=history_mode,
        samples=len(actual),
        next_need_mse=float(np.mean(squared_error)),
        food_mse=float(np.mean(squared_error[:, 0])),
        water_mse=float(np.mean(squared_error[:, 1])),
        energy_mse=float(np.mean(squared_error[:, 2])),
        safety_mse=float(np.mean(squared_error[:, 3])),
        lowest_need_accuracy=float(
            np.mean(
                np.argmin(predicted_array, axis=1)
                == np.argmin(actual_array, axis=1)
            )
        ),
    )


def main() -> None:
    args = _parse_args()
    trained_model, config = load_checkpoint(args.checkpoint)
    models = [("trained", trained_model)]
    if args.random_model_control:
        mx.random.seed(args.seed)
        models.append(
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
        "model_control,history_mode,samples,next_need_mse,food_mse,water_mse,"
        "energy_mse,safety_mse,lowest_need_accuracy"
    )
    for model_control, model in models:
        for history_mode in args.history_modes:
            result = evaluate_interoception(
                model,
                config,
                episodes=args.eval_episodes,
                seed=args.seed + 10_000,
                history_mode=history_mode,
                teacher_mode=args.teacher_mode,
                model_control=model_control,
                max_samples=args.max_samples,
            )
            print(
                ",".join(
                    [
                        result.model_control,
                        result.history_mode,
                        str(result.samples),
                        f"{result.next_need_mse:.6f}",
                        f"{result.food_mse:.6f}",
                        f"{result.water_mse:.6f}",
                        f"{result.energy_mse:.6f}",
                        f"{result.safety_mse:.6f}",
                        f"{result.lowest_need_accuracy:.4f}",
                    ]
                )
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=2101)
    parser.add_argument("--eval-episodes", type=int, default=80)
    parser.add_argument("--max-samples", type=int, default=4000)
    parser.add_argument(
        "--history-modes",
        nargs="+",
        choices=HISTORY_MODES,
        default=list(HISTORY_MODES),
    )
    parser.add_argument("--teacher-mode", default="grounded")
    parser.add_argument("--random-model-control", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
