from __future__ import annotations

import argparse
from dataclasses import fields
from statistics import mean

from .agents import TeacherFollowingAgent
from .env import (
    DIAGNOSTIC_MODES,
    STANDARD_MODE,
    HomeostaticSocialGrid,
)
from .qlearning import EpisodeStats, QLearningAgent, run_episode
from .teachers import TEACHER_MODES, build_teacher, masks_language, normalize_teacher_mode


def main() -> None:
    args = _parse_args()
    results = []
    for condition in args.conditions:
        stats = run_condition(
            condition=condition,
            episodes=args.episodes,
            eval_episodes=args.eval_episodes,
            seed=args.seed,
            randomize_world=args.randomize_world,
            include_language_channel=args.include_language_channel,
            include_object_kinds=args.include_object_kinds,
            diagnostic_mode=args.diagnostic_mode,
        )
        results.append((condition, stats))
    if args.scripted_probe:
        for condition in args.conditions:
            stats = run_scripted_condition(
                condition=condition,
                eval_episodes=args.eval_episodes,
                seed=args.seed + args.episodes,
                randomize_world=args.randomize_world,
                include_language_channel=args.include_language_channel,
                diagnostic_mode=args.diagnostic_mode,
            )
            results.append((f"{condition}_scripted_probe", stats))

    print("condition,total_reward,steps,mean_viability,min_viability,resource_uses,danger_hits,teacher_utterances")
    for condition, stats in results:
        print(
            ",".join(
                [
                    condition,
                    f"{stats.total_reward:.4f}",
                    f"{stats.steps:.2f}",
                    f"{stats.mean_viability:.4f}",
                    f"{stats.min_viability:.4f}",
                    f"{stats.resource_uses:.2f}",
                    f"{stats.danger_hits:.2f}",
                    f"{stats.teacher_utterances:.2f}",
                ]
            )
        )


def run_condition(
    *,
    condition: str,
    episodes: int,
    eval_episodes: int,
    seed: int,
    randomize_world: bool = True,
    include_language_channel: bool = True,
    include_object_kinds: bool = False,
    diagnostic_mode: str = STANDARD_MODE,
) -> EpisodeStats:
    teacher_mode = normalize_teacher_mode(condition)
    teacher = build_teacher(teacher_mode, seed=seed)
    env = HomeostaticSocialGrid(
        seed=seed,
        teacher=teacher,
        randomize_world=randomize_world,
        diagnostic_mode=diagnostic_mode,
    )
    agent = QLearningAgent(
        seed=seed,
        include_language=include_language_channel,
        mask_language=masks_language(teacher_mode),
        include_object_kinds=include_object_kinds,
    )

    for episode in range(episodes):
        run_episode(env, agent, seed=seed + episode, train=True)

    eval_stats = [
        run_episode(env, agent, seed=seed + episodes + idx, train=False)
        for idx in range(eval_episodes)
    ]
    return _average_stats(eval_stats)


def run_scripted_condition(
    *,
    condition: str,
    eval_episodes: int,
    seed: int,
    randomize_world: bool = True,
    include_language_channel: bool = True,
    diagnostic_mode: str = STANDARD_MODE,
) -> EpisodeStats:
    teacher_mode = normalize_teacher_mode(condition)
    teacher = build_teacher(teacher_mode, seed=seed)
    env = HomeostaticSocialGrid(
        seed=seed,
        teacher=teacher,
        randomize_world=randomize_world,
        diagnostic_mode=diagnostic_mode,
    )
    eval_stats = [
        _run_teacher_following_episode(
            env,
            seed=seed + idx,
            mask_language=(not include_language_channel) or masks_language(teacher_mode),
        )
        for idx in range(eval_episodes)
    ]
    return _average_stats(eval_stats)


def _run_teacher_following_episode(
    env: HomeostaticSocialGrid,
    *,
    seed: int,
    mask_language: bool = False,
) -> EpisodeStats:
    agent = TeacherFollowingAgent()
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
        if mask_language:
            observation = replace_observation_language(observation, None)
        action = agent.act(observation)
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


def replace_observation_language(observation, utterance):
    return observation.__class__(
        step_count=observation.step_count,
        position=observation.position,
        direction=observation.direction,
        needs=observation.needs,
        visible=observation.visible,
        object_ahead=observation.object_ahead,
        teacher_utterance=utterance,
        last_event=observation.last_event,
    )


def _average_stats(stats: list[EpisodeStats]) -> EpisodeStats:
    values = {
        field.name: mean(getattr(stat, field.name) for stat in stats)
        for field in fields(EpisodeStats)
    }
    return EpisodeStats(**values)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--fixed-world",
        action="store_false",
        dest="randomize_world",
        help="Disable randomized object placement across episodes.",
    )
    parser.add_argument(
        "--include-object-kinds",
        action="store_true",
        help="Expose object kinds directly to the learner instead of forcing language/experience to carry semantics.",
    )
    parser.add_argument(
        "--no-language-channel",
        action="store_false",
        dest="include_language_channel",
        help="Remove the teacher-language channel from the learner input.",
    )
    parser.add_argument(
        "--diagnostic-mode",
        choices=DIAGNOSTIC_MODES,
        default=STANDARD_MODE,
        help="Environment diagnostic mode. language_necessary hides exploitable object identity shortcuts.",
    )
    parser.add_argument(
        "--scripted-probe",
        action="store_true",
        help="Also evaluate the deterministic ask-then-act teacher-following probe.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["silent", "grounded"],
        choices=[*TEACHER_MODES, "silent_teacher", "grounded_teacher"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
