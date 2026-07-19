"""Gate B1: measure island difficulty against scripted policies.

Pass condition: oracle survives >= 95% of full-length lives; random policy
survives <= 10%. Run:

    PYTHONPATH=src python3 -m homesocial.island.calibrate --episodes 40
"""

from __future__ import annotations

import argparse

from homesocial.island.oracle import OraclePolicy, RandomPolicy
from homesocial.island.world import IslandConfig, IslandWorld

HARM_EVENTS = {"hit_danger", "consumed_poison"}


def run_policy(
    policy_name: str,
    *,
    episodes: int,
    config: IslandConfig,
    base_seed: int,
) -> dict[str, float]:
    survived = 0
    viability_sum = 0.0
    min_viability_sum = 0.0
    steps_sum = 0
    harm_sum = 0
    consume_attempt_sum = 0
    resource_consume_sum = 0
    food_consume_sum = 0
    water_consume_sum = 0
    for episode in range(episodes):
        seed = base_seed + episode
        world = IslandWorld(config, seed=seed)
        world.reset(seed)
        policy = (
            OraclePolicy() if policy_name == "oracle" else RandomPolicy(seed=seed)
        )
        min_viability = 1.0
        viability_running = 0.0
        steps = 0
        while True:
            action = policy.act(world.grid)
            _, _, terminated, truncated, info = world.step(action)
            steps += 1
            viability = float(info["viability"])
            min_viability = min(min_viability, viability)
            viability_running += float(info["mean_viability"])
            if info.get("event") in HARM_EVENTS:
                harm_sum += 1
            if str(info.get("event", "")).startswith("consumed_"):
                consume_attempt_sum += 1
            if info.get("event") in {"consumed_food", "consumed_water"}:
                resource_consume_sum += 1
            food_consume_sum += int(info.get("event") == "consumed_food")
            water_consume_sum += int(info.get("event") == "consumed_water")
            if terminated or truncated:
                if truncated and not terminated:
                    survived += 1
                break
        viability_sum += viability_running / max(1, steps)
        min_viability_sum += min_viability
        steps_sum += steps
    return {
        "survival_rate": survived / episodes,
        "mean_viability": viability_sum / episodes,
        "mean_min_viability": min_viability_sum / episodes,
        "mean_steps": steps_sum / episodes,
        "harm_events_per_episode": harm_sum / episodes,
        "consume_attempts_per_episode": consume_attempt_sum / episodes,
        "resource_consumes_per_episode": resource_consume_sum / episodes,
        "food_consumes_per_episode": food_consume_sum / episodes,
        "water_consumes_per_episode": water_consume_sum / episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--policies", nargs="+", default=["oracle", "random"],
        choices=["oracle", "random"],
    )
    parser.add_argument("--language-mode", default="grounded")
    args = parser.parse_args()

    config = IslandConfig(max_steps=args.max_steps, language_mode=args.language_mode)
    results = {}
    for policy_name in args.policies:
        stats = run_policy(
            policy_name,
            episodes=args.episodes,
            config=config,
            base_seed=args.seed,
        )
        results[policy_name] = stats
        formatted = ", ".join(f"{key}={value:.4f}" for key, value in stats.items())
        print(f"{policy_name}: {formatted}")

    if "oracle" in results and "random" in results:
        gate = (
            results["oracle"]["survival_rate"] >= 0.95
            and results["random"]["survival_rate"] <= 0.10
        )
        print(f"gate B1: {'PASS' if gate else 'FAIL'}")


if __name__ == "__main__":
    main()
