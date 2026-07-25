"""Feasibility gates F1-F5 for the report island.

These are environment-validation instruments, not learners. They read the true
body directly (or refuse to read it at all) and exist to answer one question
before any training happens: does staying alive in this world actually require
knowing your own state?

The recovery-line post-mortem in `DIRECTION_2026-07-12.md` is the reason this
file runs first. A substrate whose best body-blind schedule matches its oracle
cannot teach self-knowledge, and no learner-side result on it would mean
anything.

    PYTHONPATH=src python3 -m homesocial.island.report_calibrate --lives 400
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from random import Random

from homesocial.creole.vocab import PAD_TOKEN, TOKEN_TO_ID
from homesocial.env import Action
from homesocial.island.report import (
    NEED_TO_REPORT_WORD,
    REPORT_NEEDS,
    ReportConfig,
    ReportWorld,
)
from homesocial.island.world import IslandConfig

PAD_ID = TOKEN_TO_ID[PAD_TOKEN]
NEED_UTTERANCE = {
    need: (TOKEN_TO_ID[word], PAD_ID)
    for need, word in NEED_TO_REPORT_WORD.items()
}
SILENCE = (PAD_ID, PAD_ID)


@dataclass
class PolicyResult:
    name: str
    lives: int
    survival: float
    mean_viability: float
    mean_steps: float
    report_fidelity: float
    grants: float
    wasted_grants: float

    def row(self) -> dict[str, object]:
        return {
            "policy": self.name,
            "lives": self.lives,
            "survival": round(self.survival, 4),
            "mean_viability": round(self.mean_viability, 4),
            "mean_steps": round(self.mean_steps, 2),
            "report_fidelity": round(self.report_fidelity, 4),
            "grants_per_life": round(self.grants, 2),
            "wasted_grants_per_life": round(self.wasted_grants, 2),
        }


class TruthfulSpeaker:
    """Reads the true body and names its worst need. The upper bound."""

    body_blind = False

    def utterance(self, world: ReportWorld) -> tuple[int, ...]:
        return NEED_UTTERANCE[world.lowest_need()]


class MuteSpeaker:
    body_blind = True

    def utterance(self, world: ReportWorld) -> tuple[int, ...]:
        return SILENCE


class RandomSpeaker:
    body_blind = True

    def __init__(self, seed: int) -> None:
        self._rng = Random(seed)

    def utterance(self, world: ReportWorld) -> tuple[int, ...]:
        return NEED_UTTERANCE[self._rng.choice(REPORT_NEEDS)]


class RhythmSpeaker:
    """A fixed cyclic schedule, advanced once per help window, body-blind."""

    body_blind = True

    def __init__(self, pattern: tuple[str, ...], help_period: int) -> None:
        self.pattern = pattern
        self.help_period = help_period

    def utterance(self, world: ReportWorld) -> tuple[int, ...]:
        window = world.grid.step_count // self.help_period
        return NEED_UTTERANCE[self.pattern[window % len(self.pattern)]]


def uptake_action(world: ReportWorld) -> Action:
    """Take whatever help is in hand; otherwise stand still.

    Every policy here shares this body: only the utterance differs, so any
    survival difference is attributable to what was said.
    """

    pending = world.pending_help_need()
    if pending is None:
        return Action.WAIT
    return Action.REST if pending == "energy" else Action.CONSUME


def run_speaker(
    speaker: object,
    *,
    name: str,
    lives: int,
    base_seed: int,
    report: ReportConfig,
    island: IslandConfig,
) -> PolicyResult:
    survived = 0
    viability_sum = 0.0
    steps_sum = 0
    truthful = 0
    spoken = 0
    grants = 0
    wasted = 0
    for life in range(lives):
        seed = base_seed + life
        world = ReportWorld(island, report=report, seed=seed)
        world.reset(seed)
        if isinstance(speaker, RandomSpeaker):
            speaker = RandomSpeaker(seed)
        life_viability = 0.0
        steps = 0
        while True:
            tokens = speaker.utterance(world)
            if any(token != PAD_ID for token in tokens):
                spoken += 1
                truthful += int(
                    tokens == NEED_UTTERANCE[world.lowest_need()]
                )
            world.hear(tokens)
            action = uptake_action(world)
            _, _, terminated, truncated, info = world.step(action)
            steps += 1
            life_viability += float(info["mean_viability"])
            if info.get("granted_need") is not None:
                grants += 1
            elif world.grid.step_count % report.help_period == 0:
                wasted += 1
            if terminated or truncated:
                survived += int(not terminated)
                break
        # Averaged over the whole locked horizon, not over the ticks survived:
        # a policy that dies at tick 40 with a full body must not outscore one
        # that lives to the end.
        viability_sum += life_viability / report.life_steps
        steps_sum += steps
    return PolicyResult(
        name=name,
        lives=lives,
        survival=survived / lives,
        mean_viability=viability_sum / lives,
        mean_steps=steps_sum / lives,
        report_fidelity=truthful / max(1, spoken),
        grants=grants / lives,
        wasted_grants=wasted / lives,
    )


def rhythm_family(max_length: int) -> list[tuple[str, ...]]:
    """All cyclic schedules up to ``max_length``, deduplicated by rotation."""

    seen: set[tuple[str, ...]] = set()
    patterns: list[tuple[str, ...]] = []
    for length in range(1, max_length + 1):
        for pattern in itertools.product(REPORT_NEEDS, repeat=length):
            canonical = min(
                tuple(pattern[shift:] + pattern[:shift])
                for shift in range(length)
            )
            if canonical in seen:
                continue
            seen.add(canonical)
            patterns.append(canonical)
    return patterns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lives", type=int, default=400)
    parser.add_argument("--rhythm-lives", type=int, default=100)
    parser.add_argument("--rhythm-max-length", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    frozen = ReportConfig()
    parser.add_argument("--help-period", type=int, default=frozen.help_period)
    parser.add_argument("--life-steps", type=int, default=frozen.life_steps)
    parser.add_argument(
        "--portion-small", type=float, default=frozen.portion_small
    )
    parser.add_argument(
        "--portion-large", type=float, default=frozen.portion_large
    )
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    report = ReportConfig(
        help_period=args.help_period,
        life_steps=args.life_steps,
        portion_small=args.portion_small,
        portion_large=args.portion_large,
    )
    island = IslandConfig(max_steps=args.life_steps, max_visible_slots=2)

    results = [
        run_speaker(
            TruthfulSpeaker(),
            name="truthful_oracle",
            lives=args.lives,
            base_seed=args.seed,
            report=report,
            island=island,
        ),
        run_speaker(
            MuteSpeaker(),
            name="mute",
            lives=args.lives,
            base_seed=args.seed,
            report=report,
            island=island,
        ),
        run_speaker(
            RandomSpeaker(args.seed),
            name="random_word",
            lives=args.lives,
            base_seed=args.seed,
            report=report,
            island=island,
        ),
    ]

    best: PolicyResult | None = None
    rhythm_rows: list[dict[str, object]] = []
    for pattern in rhythm_family(args.rhythm_max_length):
        result = run_speaker(
            RhythmSpeaker(pattern, report.help_period),
            name="rhythm_" + "".join(need[0] for need in pattern),
            lives=args.rhythm_lives,
            base_seed=args.seed,
            report=report,
            island=island,
        )
        rhythm_rows.append(result.row())
        if best is None or result.survival > best.survival or (
            result.survival == best.survival
            and result.mean_viability > best.mean_viability
        ):
            best = result
    assert best is not None
    results.append(best)

    oracle = results[0]
    mute = results[1]
    random_word = results[2]
    gates = {
        "F1_oracle_survival_at_least_0.95": oracle.survival >= 0.95,
        "F2_mute_survival_at_most_0.05": mute.survival <= 0.05,
        "F3_random_word_20_points_below_oracle": (
            oracle.survival - random_word.survival >= 0.20
        ),
        "F4_best_rhythm_20_points_below_oracle": (
            oracle.survival - best.survival >= 0.20
        ),
        "F5_oracle_viability_margin_at_least_0.05": (
            oracle.mean_viability - max(
                mute.mean_viability, random_word.mean_viability, best.mean_viability
            )
            >= 0.05
        ),
    }

    print(
        f"{'policy':<24}{'survival':>10}{'viability':>11}{'steps':>9}"
        f"{'fidelity':>10}{'grants':>9}{'wasted':>8}"
    )
    for result in results:
        row = result.row()
        print(
            f"{row['policy']:<24}{row['survival']:>10}{row['mean_viability']:>11}"
            f"{row['mean_steps']:>9}{row['report_fidelity']:>10}"
            f"{row['grants_per_life']:>9}{row['wasted_grants_per_life']:>8}"
        )
    print()
    for gate, passed in gates.items():
        print(f"{'PASS' if passed else 'FAIL'}  {gate}")

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "config": {
                        "help_period": report.help_period,
                        "life_steps": report.life_steps,
                        "portion_small": report.portion_small,
                        "portion_large": report.portion_large,
                        "large_portion_probability": (
                            report.large_portion_probability
                        ),
                        "birth_levels": list(report.birth_levels),
                        "food_metabolism": report.food_metabolism,
                        "water_metabolism": report.water_metabolism,
                        "energy_metabolism": report.energy_metabolism,
                        "lives": args.lives,
                        "rhythm_lives": args.rhythm_lives,
                        "rhythm_max_length": args.rhythm_max_length,
                        "seed": args.seed,
                    },
                    "results": [result.row() for result in results],
                    "rhythms": rhythm_rows,
                    "gates": gates,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
