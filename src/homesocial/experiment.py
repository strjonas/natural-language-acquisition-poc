from __future__ import annotations

import argparse
from dataclasses import fields
from statistics import mean

from .env import HomeostaticSocialGrid, SilentTeacher, SituatedTeacher
from .qlearning import EpisodeStats, QLearningAgent, run_episode


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
            include_object_kinds=args.include_object_kinds,
        )
        results.append((condition, stats))

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
    include_object_kinds: bool = False,
) -> EpisodeStats:
    teacher = SituatedTeacher() if condition == "grounded_teacher" else SilentTeacher()
    include_language = condition == "grounded_teacher"
    env = HomeostaticSocialGrid(
        seed=seed, teacher=teacher, randomize_world=randomize_world
    )
    agent = QLearningAgent(
        seed=seed,
        include_language=include_language,
        include_object_kinds=include_object_kinds,
    )

    for episode in range(episodes):
        run_episode(env, agent, seed=seed + episode, train=True)

    eval_stats = [
        run_episode(env, agent, seed=seed + episodes + idx, train=False)
        for idx in range(eval_episodes)
    ]
    return _average_stats(eval_stats)


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
        "--conditions",
        nargs="+",
        default=["silent_teacher", "grounded_teacher"],
        choices=["silent_teacher", "grounded_teacher"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
