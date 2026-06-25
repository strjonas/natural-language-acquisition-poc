from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from .agents import TeacherFollowingAgent
from .attribution import _needs_array
from .env import Action, HomeostaticSocialGrid
from .imitation import load_checkpoint
from .interoception import HISTORY_MODES, _history_control
from .observations import observation_vector, observation_vector_size
from .recurrent_ac import RecurrentActorCritic, RecurrentConfig, action_mask
from .report_head import NEED_LABELS, _need_label, render_report
from .report_mediation import teacher_response_to_report
from .teachers import build_teacher, masks_language, normalize_teacher_mode


@dataclass(frozen=True)
class HiddenMediationResult:
    model_control: str
    history_mode: str
    samples: int
    report_accuracy: float
    decision_accuracy: float
    helpful_use_rate: float
    irrelevant_use_rate: float
    helpful_use_precision: float
    target_need_delta: float
    rendered_examples: tuple[str, ...]


@dataclass(frozen=True)
class NeedCalibrator:
    weights: np.ndarray
    bias: np.ndarray

    def apply(self, predicted_needs: np.ndarray) -> np.ndarray:
        return np.clip(predicted_needs @ self.weights + self.bias, 0.0, 1.0)


def fit_need_calibrator(
    model: RecurrentActorCritic,
    config: RecurrentConfig,
    *,
    episodes: int,
    seed: int,
    teacher_mode: str = "grounded",
    max_samples: int = 4000,
) -> NeedCalibrator:
    predicted, actual = _collect_need_pairs(
        model,
        config,
        episodes=episodes,
        seed=seed,
        teacher_mode=teacher_mode,
        max_samples=max_samples,
    )
    design = np.concatenate(
        [predicted, np.ones((predicted.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    coefficients, _residuals, _rank, _singular = np.linalg.lstsq(
        design,
        actual,
        rcond=1e-5,
    )
    return NeedCalibrator(
        weights=coefficients[:-1],
        bias=coefficients[-1],
    )


def evaluate_hidden_mediation(
    model: RecurrentActorCritic,
    config: RecurrentConfig,
    *,
    episodes: int,
    seed: int,
    history_mode: str = "full",
    teacher_mode: str = "grounded",
    model_control: str = "trained",
    max_samples: int = 1200,
    examples: int = 3,
    calibrator: NeedCalibrator | None = None,
) -> HiddenMediationResult:
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
    rng = np.random.default_rng(seed + 1_010_000)
    report_correct = 0
    decision_correct = 0
    helpful_trials = 0
    helpful_uses = 0
    irrelevant_trials = 0
    irrelevant_uses = 0
    target_deltas: list[float] = []
    rendered: list[str] = []
    samples = 0

    for episode in range(episodes):
        observation = env.reset(seed=seed + episode)
        agent = TeacherFollowingAgent()
        history: list[np.ndarray] = []
        terminated = False
        truncated = False

        while not terminated and not truncated and samples < max_samples:
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
            if observation.object_ahead is not None:
                controlled_history = _history_control(history, history_mode, rng)
                reported_need = _predicted_need_report(
                    model,
                    controlled_history,
                    calibrator=calibrator,
                )
                true_need = _need_label(_needs_array(observation.needs))
                response = teacher_response_to_report(
                    reported_need,
                    observation.object_ahead,
                )
                action = {
                    "eat food": Action.CONSUME,
                    "drink water": Action.CONSUME,
                    "rest at shelter": Action.REST,
                    "avoid danger": Action.WAIT,
                }[response]
                branch = deepcopy(env)
                before_needs = _needs_array(observation.needs)
                branch_observation, _reward, _terminated, _truncated, info = (
                    branch.step(action)
                )
                after_needs = _needs_array(branch_observation.needs)
                helpful_kind = _need_to_object(NEED_LABELS[true_need])
                helpful = helpful_kind == observation.object_ahead.kind
                used = info["event"] in {
                    "consumed_food",
                    "consumed_water",
                    "rested_shelter",
                }

                report_correct += int(reported_need == true_need)
                decision_correct += int(used == helpful)
                if helpful:
                    helpful_trials += 1
                    helpful_uses += int(used)
                    target_index = {
                        "food": 0,
                        "water": 1,
                        "shelter": 2,
                    }[helpful_kind]
                    target_deltas.append(
                        float(after_needs[target_index] - before_needs[target_index])
                    )
                else:
                    irrelevant_trials += 1
                    irrelevant_uses += int(used)
                if len(rendered) < examples:
                    rendered.append(
                        render_report(
                            reported_need,
                            1,
                            4,
                        )
                    )
                samples += 1

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
            mask = action_mask(observation)
            action_index = tuple(Action).index(action)
            if mask[action_index] <= 0.0:
                action = Action.MOVE_FORWARD
            observation, _reward, terminated, truncated, _info = env.step(action)

        if samples >= max_samples:
            break

    if samples == 0:
        raise ValueError("Hidden mediation evaluation produced no samples.")
    return HiddenMediationResult(
        model_control=model_control,
        history_mode=history_mode,
        samples=samples,
        report_accuracy=report_correct / samples,
        decision_accuracy=decision_correct / samples,
        helpful_use_rate=helpful_uses / max(1, helpful_trials),
        irrelevant_use_rate=irrelevant_uses / max(1, irrelevant_trials),
        helpful_use_precision=helpful_uses / max(
            1,
            helpful_uses + irrelevant_uses,
        ),
        target_need_delta=float(np.mean(target_deltas)) if target_deltas else 0.0,
        rendered_examples=tuple(rendered),
    )


def _predicted_need_report(
    model: RecurrentActorCritic,
    history: list[np.ndarray],
    *,
    calibrator: NeedCalibrator | None = None,
) -> int:
    predicted = _predict_needs(model, history)
    if calibrator is not None:
        predicted = calibrator.apply(predicted)
    return _need_label(predicted)


def _predict_needs(
    model: RecurrentActorCritic,
    history: list[np.ndarray],
) -> np.ndarray:
    observations = mx.array(np.stack(history), dtype=mx.float32)
    actions = np.zeros(len(history), dtype=np.int32)
    actions[-1] = tuple(Action).index(Action.WAIT)
    _next_observations, predicted_needs, _rewards, _utterances = (
        model.predict_consequences(
            observations,
            mx.array(actions, dtype=mx.int32),
        )
    )
    return np.asarray(predicted_needs[-1], dtype=np.float32)


def _collect_need_pairs(
    model: RecurrentActorCritic,
    config: RecurrentConfig,
    *,
    episodes: int,
    seed: int,
    teacher_mode: str,
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
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
    predicted: list[np.ndarray] = []
    actual: list[np.ndarray] = []
    for episode in range(episodes):
        observation = env.reset(seed=seed + episode)
        agent = TeacherFollowingAgent()
        history: list[np.ndarray] = []
        terminated = False
        truncated = False
        while not terminated and not truncated and len(actual) < max_samples:
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
            predicted.append(_predict_needs(model, history))
            actual.append(_needs_array(observation.needs))
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
            mask = action_mask(observation)
            action_index = tuple(Action).index(action)
            if mask[action_index] <= 0.0:
                action = Action.MOVE_FORWARD
            observation, _reward, terminated, truncated, _info = env.step(action)
        if len(actual) >= max_samples:
            break
    return np.stack(predicted), np.stack(actual)


def _need_to_object(need_label: str) -> str | None:
    return {
        "need_food": "food",
        "need_water": "water",
        "need_rest": "shelter",
    }.get(need_label)


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
        "model_control,history_mode,samples,report_accuracy,decision_accuracy,"
        "helpful_use_rate,irrelevant_use_rate,helpful_use_precision,"
        "target_need_delta,examples"
    )
    for model_control, model in models:
        calibrator = fit_need_calibrator(
            model,
            config,
            episodes=args.calibration_episodes,
            seed=args.seed,
            teacher_mode=args.teacher_mode,
            max_samples=args.calibration_samples,
        )
        for history_mode in args.history_modes:
            result = evaluate_hidden_mediation(
                model,
                config,
                episodes=args.eval_episodes,
                seed=args.seed + 10_000,
                history_mode=history_mode,
                teacher_mode=args.teacher_mode,
                model_control=model_control,
                max_samples=args.max_samples,
                examples=args.examples,
                calibrator=calibrator,
            )
            print(
                ",".join(
                    [
                        result.model_control,
                        result.history_mode,
                        str(result.samples),
                        f"{result.report_accuracy:.4f}",
                        f"{result.decision_accuracy:.4f}",
                        f"{result.helpful_use_rate:.4f}",
                        f"{result.irrelevant_use_rate:.4f}",
                        f"{result.helpful_use_precision:.4f}",
                        f"{result.target_need_delta:.4f}",
                        "|".join(result.rendered_examples),
                    ]
                )
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=2301)
    parser.add_argument("--eval-episodes", type=int, default=120)
    parser.add_argument("--max-samples", type=int, default=1200)
    parser.add_argument("--calibration-episodes", type=int, default=80)
    parser.add_argument("--calibration-samples", type=int, default=3000)
    parser.add_argument(
        "--history-modes",
        nargs="+",
        choices=HISTORY_MODES,
        default=list(HISTORY_MODES),
    )
    parser.add_argument("--teacher-mode", default="grounded")
    parser.add_argument("--random-model-control", action="store_true")
    parser.add_argument("--examples", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    main()
