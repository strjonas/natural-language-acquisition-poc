from __future__ import annotations

from dataclasses import fields, replace
from statistics import mean
from typing import Literal

from .agents import TeacherFollowingAgent
from .env import LANGUAGE_NECESSARY_MODE, Action, HomeostaticSocialGrid, Observation
from .qlearning import EpisodeStats
from .teachers import TeacherMode, build_teacher, masks_language, normalize_teacher_mode


BlindPolicy = Literal["consume", "rest", "avoid", "ask_then_consume"]


def run_teacher_following_diagnostic(
    mode: TeacherMode | str,
    *,
    seed: int = 1,
    eval_episodes: int = 20,
    randomize_world: bool = False,
    include_language_channel: bool = True,
) -> EpisodeStats:
    teacher_mode = normalize_teacher_mode(mode)
    env = HomeostaticSocialGrid(
        seed=seed,
        teacher=build_teacher(teacher_mode, seed=seed),
        randomize_world=randomize_world,
        diagnostic_mode=LANGUAGE_NECESSARY_MODE,
    )
    stats = [
        _run_episode(
            env,
            TeacherFollowingAgent().act,
            seed=seed + index,
            mask_language=(not include_language_channel) or masks_language(teacher_mode),
        )
        for index in range(eval_episodes)
    ]
    return average_stats(stats)


def run_blind_diagnostic(
    policy: BlindPolicy,
    *,
    seed: int = 1,
    eval_episodes: int = 20,
    randomize_world: bool = False,
) -> EpisodeStats:
    env = HomeostaticSocialGrid(
        seed=seed,
        teacher=build_teacher("silent", seed=seed),
        randomize_world=randomize_world,
        diagnostic_mode=LANGUAGE_NECESSARY_MODE,
    )
    stats = [
        _run_episode(env, lambda observation: _blind_action(policy, observation), seed=seed + index)
        for index in range(eval_episodes)
    ]
    return average_stats(stats)


def average_stats(stats: list[EpisodeStats]) -> EpisodeStats:
    values = {
        field.name: mean(getattr(stat, field.name) for stat in stats)
        for field in fields(EpisodeStats)
    }
    return EpisodeStats(**values)


def _run_episode(
    env: HomeostaticSocialGrid,
    policy,
    *,
    seed: int,
    mask_language: bool = False,
) -> EpisodeStats:
    observation = env.reset(seed=seed)
    total_reward = 0.0
    viability_sum = observation.needs.mean_viability()
    min_viability = observation.needs.viability()
    resource_uses = 0
    danger_hits = 0
    teacher_utterances = 0
    steps = 0
    terminated = False
    truncated = False

    while not terminated and not truncated:
        policy_observation = (
            replace(observation, teacher_utterance=None) if mask_language else observation
        )
        action = policy(policy_observation)
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1
        viability_sum += observation.needs.mean_viability()
        min_viability = min(min_viability, observation.needs.viability())

        event = info["event"]
        if event in {"consumed_water", "consumed_food", "rested_shelter"}:
            resource_uses += 1
        if event in {"hit_danger", "consumed_danger", "rested_danger"}:
            danger_hits += 1
        if observation.teacher_utterance:
            teacher_utterances += 1

    return EpisodeStats(
        total_reward=total_reward,
        steps=steps,
        terminated=terminated,
        truncated=truncated,
        mean_viability=viability_sum / (steps + 1),
        min_viability=min_viability,
        resource_uses=resource_uses,
        danger_hits=danger_hits,
        teacher_utterances=teacher_utterances,
    )


def _blind_action(policy: BlindPolicy, observation: Observation) -> Action:
    if observation.last_event in {"bumped_wall", "blocked"}:
        return Action.TURN_RIGHT
    if observation.object_ahead is None:
        return Action.MOVE_FORWARD
    if policy == "consume":
        return Action.CONSUME
    if policy == "rest":
        return Action.REST
    if policy == "avoid":
        return Action.TURN_RIGHT
    if policy == "ask_then_consume":
        return Action.CONSUME if observation.last_event == "asked" else Action.ASK
    raise ValueError(f"Unknown blind policy: {policy}")
