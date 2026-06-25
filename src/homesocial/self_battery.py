from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, replace
from math import isfinite
from typing import Literal

import mlx.core as mx
import numpy as np

from .agents import TeacherFollowingAgent
from .env import DIAGNOSTIC_MODES, STANDARD_MODE, Action, HomeostaticSocialGrid, Needs
from .imitation import load_checkpoint
from .observations import observation_vector
from .recurrent_ac import (
    RecurrentActorCritic,
    RecurrentConfig,
    action_mask,
    choose_action_info,
)
from .teachers import TEACHER_MODES, build_teacher, masks_language, normalize_teacher_mode


RolloutPolicy = Literal["model", "teacher"]


@dataclass(frozen=True)
class ConsequenceProbeResult:
    samples: int
    next_need_mse: float
    reward_mse: float
    viability_rank_accuracy: float
    need_direction_accuracy: float
    action_sensitivity: float


def run_consequence_probe(
    model: RecurrentActorCritic,
    config: RecurrentConfig,
    *,
    teacher_mode: str = "grounded",
    seeds: list[int] | None = None,
    max_samples: int = 200,
    rollout_policy: RolloutPolicy = "model",
) -> ConsequenceProbeResult:
    normalized_mode = normalize_teacher_mode(teacher_mode)
    mask_language = masks_language(normalized_mode) or not config.include_language_channel
    env = HomeostaticSocialGrid(
        width=config.width,
        height=config.height,
        seed=config.seed,
        max_steps=config.max_steps,
        teacher=build_teacher(normalized_mode, seed=config.seed),
        randomize_world=config.randomize_world,
        diagnostic_mode=config.diagnostic_mode,
        body_dynamics_mode=config.body_dynamics_mode,
    )
    rng = np.random.default_rng(config.seed + 200_000)
    teacher_agent = TeacherFollowingAgent()
    seeds = seeds if seeds is not None else [config.seed + index for index in range(20)]

    need_errors: list[float] = []
    reward_errors: list[float] = []
    rank_hits: list[float] = []
    direction_hits: list[float] = []
    sensitivities: list[float] = []

    for seed in seeds:
        observation = env.reset(seed=seed)
        teacher_agent = TeacherFollowingAgent()
        obs_vectors: list[np.ndarray] = []
        action_masks: list[np.ndarray] = []
        terminated = False
        truncated = False

        while not terminated and not truncated and len(need_errors) < max_samples:
            vector = observation_vector(
                observation,
                width=env.width,
                height=env.height,
                include_language=config.include_language_channel,
                mask_language=mask_language,
                include_object_kinds=config.include_object_kinds,
                interoception_mode=config.interoception_mode,
                body_dynamics_mode=config.body_dynamics_mode,
            )
            obs_vectors.append(vector)
            mask = action_mask(observation)
            action_masks.append(mask)

            if _is_probe_state(observation):
                sample = _probe_state(model, env, obs_vectors, mask)
                if sample is not None:
                    (
                        need_error,
                        reward_error,
                        rank_hit,
                        direction_hit,
                        sensitivity,
                    ) = sample
                    need_errors.append(need_error)
                    reward_errors.append(reward_error)
                    rank_hits.append(rank_hit)
                    direction_hits.append(direction_hit)
                    sensitivities.append(sensitivity)

            if rollout_policy == "teacher":
                policy_observation = (
                    replace(observation, teacher_utterance=None)
                    if mask_language
                    else observation
                )
                action = teacher_agent.act(policy_observation)
                action_index = tuple(Action).index(action)
                if mask[action_index] <= 0.0:
                    action = Action.MOVE_FORWARD
            else:
                action_index, _log_prob, _value = choose_action_info(
                    model,
                    np.stack(obs_vectors),
                    np.stack(action_masks),
                    rng=rng,
                    sample=False,
                )
                action = tuple(Action)[action_index]

            observation, _reward, terminated, truncated, _info = env.step(action)

        if len(need_errors) >= max_samples:
            break

    if not need_errors:
        raise ValueError("No consequence-probe samples collected.")

    return ConsequenceProbeResult(
        samples=len(need_errors),
        next_need_mse=float(np.mean(need_errors)),
        reward_mse=float(np.mean(reward_errors)),
        viability_rank_accuracy=float(np.mean(rank_hits)),
        need_direction_accuracy=float(np.mean(direction_hits)),
        action_sensitivity=float(np.mean(sensitivities)),
    )


def _is_probe_state(observation) -> bool:
    return observation.object_ahead is not None and observation.teacher_utterance is not None


def _probe_state(
    model: RecurrentActorCritic,
    env: HomeostaticSocialGrid,
    obs_vectors: list[np.ndarray],
    mask: np.ndarray,
) -> tuple[float, float, float, float, float] | None:
    candidate_indices = [index for index, allowed in enumerate(mask) if allowed > 0.0]
    if len(candidate_indices) < 2:
        return None

    predicted_needs: list[np.ndarray] = []
    actual_needs: list[np.ndarray] = []
    predicted_rewards: list[float] = []
    actual_rewards: list[float] = []
    current_needs = _needs_array(env.needs)

    for action_index in candidate_indices:
        needs, reward = _predict_last_consequence(model, obs_vectors, action_index)
        predicted_needs.append(needs)
        predicted_rewards.append(reward)

        env_copy = deepcopy(env)
        next_observation, actual_reward, _terminated, _truncated, _info = env_copy.step(
            tuple(Action)[action_index]
        )
        actual_needs.append(_needs_array(next_observation.needs))
        actual_rewards.append(float(actual_reward))

    predicted_needs_array = np.stack(predicted_needs)
    actual_needs_array = np.stack(actual_needs)
    predicted_rewards_array = np.asarray(predicted_rewards, dtype=np.float32)
    actual_rewards_array = np.asarray(actual_rewards, dtype=np.float32)
    predicted_viability = np.mean(predicted_needs_array, axis=1)
    actual_viability = np.mean(actual_needs_array, axis=1)

    need_error = float(np.mean((predicted_needs_array - actual_needs_array) ** 2))
    reward_error = float(np.mean((predicted_rewards_array - actual_rewards_array) ** 2))
    actual_best = np.flatnonzero(actual_viability >= np.max(actual_viability) - 1e-6)
    rank_hit = float(int(int(np.argmax(predicted_viability)) in set(actual_best)))
    direction_hit = _direction_accuracy(
        predicted_needs_array - current_needs,
        actual_needs_array - current_needs,
    )
    sensitivity = float(np.std(predicted_viability))

    if not all(
        isfinite(value)
        for value in (need_error, reward_error, rank_hit, direction_hit, sensitivity)
    ):
        return None
    return need_error, reward_error, rank_hit, direction_hit, sensitivity


def _predict_last_consequence(
    model: RecurrentActorCritic,
    obs_vectors: list[np.ndarray],
    action_index: int,
) -> tuple[np.ndarray, float]:
    actions = np.zeros(len(obs_vectors), dtype=np.int32)
    actions[-1] = action_index
    _next_obs, needs, rewards, _utterances = model.predict_consequences(
        mx.array(np.stack(obs_vectors), dtype=mx.float32),
        mx.array(actions, dtype=mx.int32),
    )
    return np.asarray(needs[-1]), float(np.asarray(rewards[-1]))


def _needs_array(needs: Needs) -> np.ndarray:
    return np.asarray([needs.food, needs.water, needs.energy, needs.safety], dtype=np.float32)


def _direction_accuracy(predicted_delta: np.ndarray, actual_delta: np.ndarray) -> float:
    active = np.abs(actual_delta) > 0.01
    if not np.any(active):
        return 1.0
    predicted_sign = np.sign(predicted_delta[active])
    actual_sign = np.sign(actual_delta[active])
    return float(np.mean(predicted_sign == actual_sign))


def main() -> None:
    args = _parse_args()
    model, config = load_checkpoint(args.checkpoint)
    seeds = list(range(args.seed, args.seed + args.eval_episodes))
    print(
        "teacher_mode,rollout_policy,samples,next_need_mse,reward_mse,viability_rank_accuracy,need_direction_accuracy,action_sensitivity"
    )
    for mode in args.teacher_modes:
        result = run_consequence_probe(
            model,
            config,
            teacher_mode=mode,
            seeds=seeds,
            max_samples=args.max_samples,
            rollout_policy=args.rollout_policy,
        )
        print(
            ",".join(
                [
                    mode,
                    args.rollout_policy,
                    str(result.samples),
                    f"{result.next_need_mse:.6f}",
                    f"{result.reward_mse:.6f}",
                    f"{result.viability_rank_accuracy:.4f}",
                    f"{result.need_direction_accuracy:.4f}",
                    f"{result.action_sensitivity:.6f}",
                ]
            )
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=421)
    parser.add_argument("--eval-episodes", type=int, default=40)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument(
        "--teacher-modes",
        nargs="+",
        choices=TEACHER_MODES,
        default=["grounded", "masked", "shuffled", "wrong"],
    )
    parser.add_argument(
        "--rollout-policy",
        choices=["model", "teacher"],
        default="model",
    )
    parser.add_argument(
        "--diagnostic-mode",
        choices=DIAGNOSTIC_MODES,
        default=STANDARD_MODE,
        help="Accepted for CLI symmetry; checkpoint config controls the environment.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
