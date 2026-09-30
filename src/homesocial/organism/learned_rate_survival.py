"""Probe73: does the learned individual rate improve survival?

Preregistration:
``docs/decisions/2026-08-24-learned-rate-survival-preregistration.md``.

Every cell runs the same recursive body calibrator.  The only intervention is
which metabolic-rate vector the fixed future-request rule is allowed to read:
species, learned, or true.  Long runs checkpoint after every chunk and resume
without spending completed lives again.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
from random import Random
from typing import Callable, Literal

import numpy as np

from homesocial.creole.vocab import PAD_TOKEN, TOKEN_TO_ID
from homesocial.island.report import NEED_TO_REPORT_WORD, REPORT_NEEDS
from homesocial.organism.future_request import (
    RequestLedger,
    answer_horizon,
    choose_need,
    expected_portion,
    need_scores,
    true_need,
    true_rates,
)
from homesocial.organism.individual_self import _true_body
from homesocial.organism.model import OrganismModel
from homesocial.organism.portion_request import (
    MoveFraction,
    SelfModelTier,
    species_rate,
)
from homesocial.organism.report_audit import FIDELITY_WARMUP, make_report_world
from homesocial.organism.self_belief import _sample_motor_action
from homesocial.organism.train import (
    OrganismConfig,
    execute_agent_action,
    load_organism_checkpoint,
)


PAD_ID = TOKEN_TO_ID[PAD_TOKEN]
DELAY = 24
SEEDS = 5
LIVES_PER_SEED = 140
CHUNK_LIVES = 20
SEED_STRIDE = 2_000_000
DIAGNOSTIC_SEED_BASE = 2_900_000_000
TREATMENT_SEED_BASE = 3_000_000_000
VIABILITY_FLOOR = 0.15

RateMode = Literal["species", "learned", "true"]

# A complete paired replicate lands before the next seed begins.  Chunking is
# only a durable-progress concern and never creates additional statistical
# units.
CELL_SPECS: tuple[dict[str, str], ...] = (
    {"label": "species_rate", "rate_mode": "species"},
    {"label": "learned_rate", "rate_mode": "learned"},
    {"label": "true_rate", "rate_mode": "true"},
)


def _paired_interval(values: list[float]) -> dict[str, float | int]:
    """Mean and two-sided 95% Student-t interval over paired seed blocks."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("an interval needs at least one value")
    mean = float(array.mean())
    if array.size == 1:
        return {"mean": mean, "low": mean, "high": mean, "n": 1}
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
    }
    critical = table.get(int(array.size - 1), 1.96)
    half = critical * float(array.std(ddof=1)) / float(np.sqrt(array.size))
    return {
        "mean": mean,
        "low": mean - half,
        "high": mean + half,
        "n": int(array.size),
    }


def _contrast(a: list[float], b: list[float]) -> dict[str, object]:
    """Paired ``a - b`` with direction counts."""

    if len(a) != len(b):
        raise ValueError("paired contrasts must have the same number of seeds")
    values = [float(left) - float(right) for left, right in zip(a, b)]
    return _summarize_values(values)


def _summarize_values(values: list[float]) -> dict[str, object]:
    interval = _paired_interval(values)
    mean = float(interval["mean"])
    return {
        **interval,
        "values": values,
        "positive": int(sum(value > 0.0 for value in values)),
        "negative": int(sum(value < 0.0 for value in values)),
        "same_sign": int(sum(value * mean > 0.0 for value in values)),
    }


def _positive_gate(summary: dict[str, object]) -> bool:
    return (
        float(summary["mean"]) > 0.0
        and float(summary["low"]) > 0.0
        and int(summary["same_sign"]) >= 4
    )


def grade(cells: list[dict[str, object]]) -> dict[str, object]:
    """Score only the five gates fixed in the preregistration."""

    index = {str(cell["label"]): cell for cell in cells}
    expected = {str(spec["label"]) for spec in CELL_SPECS}
    if set(index) != expected:
        missing = sorted(expected - set(index))
        extra = sorted(set(index) - expected)
        raise ValueError(f"wrong cells; missing={missing}, extra={extra}")

    def per_seed(label: str, key: str) -> list[float]:
        return [float(value) for value in index[label][key]]

    learned_value = _contrast(
        per_seed("learned_rate", "per_seed_survival"),
        per_seed("species_rate", "per_seed_survival"),
    )
    true_value = _contrast(
        per_seed("true_rate", "per_seed_survival"),
        per_seed("species_rate", "per_seed_survival"),
    )
    regret_value = _summarize_values(
        per_seed("learned_rate", "per_seed_matched_regret_value")
    )
    rate_recovery = _summarize_values(
        per_seed("learned_rate", "per_seed_rate_mae_value")
    )

    true_survival = float(index["true_rate"]["survival"])
    g5_pass = true_survival >= VIABILITY_FLOOR
    gates: dict[str, object] = {
        "G1_learned_rate_reaches_survival": {
            **learned_value,
            "verdict": "pass" if _positive_gate(learned_value) else "fail",
        },
        "G2_matched_physical_ceiling": {
            **true_value,
            "verdict": "pass" if _positive_gate(true_value) else "fail",
        },
        "G3_matched_regret_predicts_sign": {
            **regret_value,
            "verdict": "pass" if _positive_gate(regret_value) else "fail",
        },
        "G4_rate_was_learned": {
            **rate_recovery,
            "verdict": "pass" if _positive_gate(rate_recovery) else "fail",
        },
        "G5_construction_is_viable": {
            "true_rate_survival": true_survival,
            "floor": VIABILITY_FLOOR,
            "verdict": "pass" if g5_pass else "fail",
        },
    }
    gates["all_pass"] = all(
        gate["verdict"] == "pass"
        for gate in gates.values()
        if isinstance(gate, dict) and "verdict" in gate
    )
    gates["contrasts"] = {
        "learned_rate_value": learned_value,
        "true_rate_value": true_value,
        "matched_regret_value": regret_value,
        "rate_mae_value": rate_recovery,
    }
    return gates


def _species_rates(report, move_fraction: float) -> np.ndarray:
    return np.asarray(
        [species_rate(report, need, move_fraction) for need in REPORT_NEEDS],
        dtype=np.float64,
    )


def _learned_rates(tier: SelfModelTier, move_fraction: float) -> np.ndarray:
    return np.asarray(
        [tier.believed_rate(need, move_fraction) for need in REPORT_NEEDS],
        dtype=np.float64,
    )


def _word_for_rates(
    tier: SelfModelTier,
    rates: np.ndarray,
    ledger: RequestLedger,
    *,
    now: int,
    delay: int,
    report,
) -> str:
    """One fixed state and uptake belief, with only the rate vector supplied."""

    uptake = tier.believed_uptake()
    return choose_need(
        believed_levels=tier.point(),
        believed_rates=rates,
        believed_uptake=uptake,
        arriving=ledger.arriving(now, uptake, delay=delay),
        horizon=answer_horizon(delay),
        report=report,
    )


def run_closed_loop_rate_mode(
    model: OrganismModel,
    organism: OrganismConfig,
    *,
    rate_mode: RateMode,
    delay: int,
    lives: int,
    seed_base: int,
    interoception: float = 0.03,
) -> dict[str, object]:
    """One matched recursive tier living with one selected rate vector."""

    if rate_mode not in ("species", "learned", "true"):
        raise ValueError(f"unknown rate mode: {rate_mode}")
    world_kwargs = {
        "metabolic_spread": 0.60,
        "uptake_spread": 0.00,
        "interoception_probability": interoception,
        "help_delay": delay,
        "caregiver_store": 0.0,
    }
    report = replace(organism.report, **world_kwargs)

    survived = 0
    truthful = 0
    future_truthful = 0
    said = 0
    steps = 0
    matched_word_differences = 0
    regret_learned = 0.0
    regret_species = 0.0
    mae_learned = 0.0
    mae_species = 0.0
    deaths: defaultdict[str, int] = defaultdict(int)

    for life in range(lives):
        seed = seed_base + life
        world = make_report_world(organism, seed=seed, **world_kwargs)
        packet = world.reset(seed)
        tier = SelfModelTier("recursive", organism, report)
        tier.reset(packet, world)
        ledger = RequestLedger(
            help_period=report.help_period,
            delay=delay,
            expected_portion=expected_portion(report),
        )
        ledger.reset()
        moves = MoveFraction(organism, report)
        moves.reset(packet)
        hidden = None
        rng = Random(seed + 59_000_003)

        while True:
            fraction = moves.fraction
            now = int(packet.step_count)
            learned = _learned_rates(tier, fraction)
            species = _species_rates(report, fraction)
            truth = true_rates(world, report, fraction)
            learned_word = _word_for_rates(
                tier, learned, ledger, now=now, delay=delay, report=report
            )
            species_word = _word_for_rates(
                tier, species, ledger, now=now, delay=delay, report=report
            )
            if rate_mode == "learned":
                spoken = learned_word
            elif rate_mode == "species":
                spoken = species_word
            else:
                spoken = _word_for_rates(
                    tier, truth, ledger, now=now, delay=delay, report=report
                )

            if packet.step_count >= FIDELITY_WARMUP:
                said += 1
                truthful += int(spoken == world.lowest_need())
                future_truthful += int(
                    spoken == true_need(world, report, fraction, delay, ledger, now)
                )
                matched_word_differences += int(learned_word != species_word)
                mae_learned += float(np.abs(learned - truth).mean())
                mae_species += float(np.abs(species - truth).mean())

                true_scores = need_scores(
                    believed_levels=_true_body(world),
                    believed_rates=truth,
                    believed_uptake=world.uptake_scale,
                    arriving=ledger.arriving(
                        now, world.uptake_scale, delay=delay
                    ),
                    horizon=answer_horizon(delay),
                    report=report,
                )
                best = float(true_scores.max())
                regret_learned += best - float(
                    true_scores[REPORT_NEEDS.index(learned_word)]
                )
                regret_species += best - float(
                    true_scores[REPORT_NEEDS.index(species_word)]
                )

            ledger.record(now, spoken)
            action, hidden = _sample_motor_action(model, packet, hidden, rng)
            world.hear((TOKEN_TO_ID[NEED_TO_REPORT_WORD[spoken]], PAD_ID))
            before = packet
            packet, _, terminated, truncated, info = execute_agent_action(
                world,
                packet,
                action,
                consume_options=organism.consume_options,
                inspect_options=organism.inspect_options,
            )
            steps += int(info["duration"])
            if terminated or truncated:
                survived += int(not terminated)
                if terminated:
                    deaths[str(info.get("death_need"))] += 1
                break
            moves.observe(before, action, packet)
            tier.update(before, action, packet, world, info.get("interoception"))

    denominator = max(1, said)
    learned_regret = regret_learned / denominator
    species_regret = regret_species / denominator
    learned_mae = mae_learned / denominator
    species_mae = mae_species / denominator
    return {
        "rate_mode": rate_mode,
        "world": "metabolic",
        "help_delay": delay,
        "lives": lives,
        "seed_base": seed_base,
        "survival": survived / lives,
        "report_fidelity": truthful / denominator,
        "future_fidelity": future_truthful / denominator,
        "mean_life_steps": steps / lives,
        "scored_ticks": said,
        "matched_word_differences": matched_word_differences,
        "matched_word_difference_share": matched_word_differences / denominator,
        "matched_regret": {
            "learned": learned_regret,
            "species": species_regret,
            "value": species_regret - learned_regret,
        },
        "rate_mae": {
            "learned": learned_mae,
            "species": species_mae,
            "value": species_mae - learned_mae,
        },
        "deaths_by_need": dict(deaths),
    }


def _validate_run(
    *, seed_base: int, seeds: int, lives_per_seed: int, chunk_lives: int
) -> None:
    if seeds < 1 or lives_per_seed < 1 or chunk_lives < 1:
        raise ValueError("seeds and lives must be positive")
    if lives_per_seed % chunk_lives:
        raise ValueError("lives_per_seed must be divisible by chunk_lives")
    if lives_per_seed > SEED_STRIDE:
        raise ValueError("a seed block exceeds the fixed seed stride")
    last_seed = seed_base + (seeds - 1) * SEED_STRIDE + lives_per_seed - 1
    if last_seed >= 2**32:
        raise ValueError("the requested life seeds exceed the 32-bit RNG range")
    if seed_base < DIAGNOSTIC_SEED_BASE:
        raise ValueError("Probe73 must not reuse an earlier experiment's seed band")


def _configuration(
    *, seed_base: int, seeds: int, lives_per_seed: int, chunk_lives: int
) -> dict[str, object]:
    return {
        "seed_base": seed_base,
        "seeds": seeds,
        "lives_per_seed": lives_per_seed,
        "chunk_lives": chunk_lives,
        "seed_stride": SEED_STRIDE,
        "delay": DELAY,
        "cells": list(CELL_SPECS),
    }


def _write_progress(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float))
    temporary.replace(path)


def _summarize(
    chunks: list[dict[str, object]], *, seeds: int, lives_per_seed: int
) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    for spec in CELL_SPECS:
        label = str(spec["label"])
        per_seed_survival: list[float] = []
        per_seed_steps: list[float] = []
        per_seed_regret_learned: list[float] = []
        per_seed_regret_species: list[float] = []
        per_seed_regret_value: list[float] = []
        per_seed_mae_learned: list[float] = []
        per_seed_mae_species: list[float] = []
        per_seed_mae_value: list[float] = []
        per_seed_word_difference: list[float] = []
        deaths: defaultdict[str, int] = defaultdict(int)

        for seed_index in range(seeds):
            records = [
                record
                for record in chunks
                if str(record["cell"]) == label
                and int(record["seed_index"]) == seed_index
            ]
            lives = sum(int(record["lives"]) for record in records)
            if lives != lives_per_seed:
                raise ValueError(
                    f"{label} seed {seed_index} has {lives}/{lives_per_seed} lives"
                )
            survived = sum(
                int(round(float(record["survival"]) * int(record["lives"])))
                for record in records
            )
            steps = sum(
                float(record["mean_life_steps"]) * int(record["lives"])
                for record in records
            )
            ticks = sum(int(record["scored_ticks"]) for record in records)
            if ticks <= 0:
                raise ValueError(f"{label} seed {seed_index} has no scored ticks")

            def tick_mean(section: str, key: str) -> float:
                total = sum(
                    float(record[section][key]) * int(record["scored_ticks"])
                    for record in records
                )
                return total / ticks

            learned_regret = tick_mean("matched_regret", "learned")
            species_regret = tick_mean("matched_regret", "species")
            learned_mae = tick_mean("rate_mae", "learned")
            species_mae = tick_mean("rate_mae", "species")
            differences = sum(
                int(record["matched_word_differences"]) for record in records
            )

            per_seed_survival.append(survived / lives)
            per_seed_steps.append(steps / lives)
            per_seed_regret_learned.append(learned_regret)
            per_seed_regret_species.append(species_regret)
            per_seed_regret_value.append(species_regret - learned_regret)
            per_seed_mae_learned.append(learned_mae)
            per_seed_mae_species.append(species_mae)
            per_seed_mae_value.append(species_mae - learned_mae)
            per_seed_word_difference.append(differences / ticks)
            for record in records:
                for need, count in dict(record["deaths_by_need"]).items():
                    deaths[str(need)] += int(count)

        cells.append(
            {
                **spec,
                "lives": seeds * lives_per_seed,
                "survival": float(np.mean(per_seed_survival)),
                "per_seed_survival": per_seed_survival,
                "mean_life_steps": float(np.mean(per_seed_steps)),
                "per_seed_mean_life_steps": per_seed_steps,
                "matched_word_difference_share": float(
                    np.mean(per_seed_word_difference)
                ),
                "per_seed_matched_word_difference_share": per_seed_word_difference,
                "matched_regret": {
                    "learned": float(np.mean(per_seed_regret_learned)),
                    "species": float(np.mean(per_seed_regret_species)),
                    "value": float(np.mean(per_seed_regret_value)),
                },
                "per_seed_matched_regret_value": per_seed_regret_value,
                "rate_mae": {
                    "learned": float(np.mean(per_seed_mae_learned)),
                    "species": float(np.mean(per_seed_mae_species)),
                    "value": float(np.mean(per_seed_mae_value)),
                },
                "per_seed_rate_mae_value": per_seed_mae_value,
                "deaths_by_need": dict(deaths),
            }
        )
    return cells


def run_experiment(
    model: OrganismModel,
    organism: OrganismConfig,
    *,
    seed_base: int = TREATMENT_SEED_BASE,
    seeds: int = SEEDS,
    lives_per_seed: int = LIVES_PER_SEED,
    chunk_lives: int = CHUNK_LIVES,
    out: Path | None = None,
    runner: Callable[..., dict[str, object]] = run_closed_loop_rate_mode,
) -> dict[str, object]:
    """Run or resume the three cells, persisting progress after each chunk."""

    _validate_run(
        seed_base=seed_base,
        seeds=seeds,
        lives_per_seed=lives_per_seed,
        chunk_lives=chunk_lives,
    )
    config = _configuration(
        seed_base=seed_base,
        seeds=seeds,
        lives_per_seed=lives_per_seed,
        chunk_lives=chunk_lives,
    )
    chunks: list[dict[str, object]] = []
    if out is not None and out.exists():
        prior = json.loads(out.read_text())
        if prior.get("configuration") != config:
            raise ValueError("existing progress artifact has a different configuration")
        chunks = list(prior.get("chunks", []))

    completed = {
        (int(record["seed_index"]), str(record["cell"]), int(record["chunk_index"]))
        for record in chunks
    }
    chunks_per_seed = lives_per_seed // chunk_lives
    total_units = seeds * len(CELL_SPECS) * chunks_per_seed

    for seed_index in range(seeds):
        block_base = seed_base + seed_index * SEED_STRIDE
        for spec in CELL_SPECS:
            label = str(spec["label"])
            for chunk_index in range(chunks_per_seed):
                key = (seed_index, label, chunk_index)
                if key in completed:
                    continue
                life_base = block_base + chunk_index * chunk_lives
                record = runner(
                    model,
                    organism,
                    rate_mode=str(spec["rate_mode"]),
                    delay=DELAY,
                    lives=chunk_lives,
                    seed_base=life_base,
                )
                chunks.append(
                    {
                        "seed_index": seed_index,
                        "cell": label,
                        "chunk_index": chunk_index,
                        "seed_base": life_base,
                        **record,
                    }
                )
                completed.add(key)
                progress = {
                    "configuration": config,
                    "complete": False,
                    "completed_units": len(completed),
                    "total_units": total_units,
                    "chunks": chunks,
                }
                if out is not None:
                    _write_progress(out, progress)
                print(
                    f"seed {seed_index + 1}/{seeds} {label:<13} "
                    f"chunk {chunk_index + 1}/{chunks_per_seed} "
                    f"survival {float(record['survival']):.3f} "
                    f"[{len(completed)}/{total_units}]",
                    flush=True,
                )

    cells = _summarize(chunks, seeds=seeds, lives_per_seed=lives_per_seed)
    payload = {
        "configuration": config,
        "complete": True,
        "completed_units": len(completed),
        "total_units": total_units,
        "chunks": chunks,
        "cells": cells,
        "gates": grade(cells),
    }
    if out is not None:
        _write_progress(out, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=(
            "runs/organism/probe52_guided_report_lexicon/adult/"
            "organism_report_seed1.npz"
        ),
    )
    parser.add_argument("--seed-base", type=int, default=TREATMENT_SEED_BASE)
    parser.add_argument("--seeds", type=int, default=SEEDS)
    parser.add_argument("--lives-per-seed", type=int, default=LIVES_PER_SEED)
    parser.add_argument("--chunk-lives", type=int, default=CHUNK_LIVES)
    parser.add_argument(
        "--out",
        default="runs/organism/probe73_learned_rate_survival/treatment.json",
    )
    args = parser.parse_args()

    model, organism = load_organism_checkpoint(Path(args.checkpoint))
    result = run_experiment(
        model,
        organism,
        seed_base=args.seed_base,
        seeds=args.seeds,
        lives_per_seed=args.lives_per_seed,
        chunk_lives=args.chunk_lives,
        out=Path(args.out),
    )

    print("\n| cell | survival | per seed | regret value | rate-MAE value |")
    print("|---|---:|---|---:|---:|")
    for cell in result["cells"]:
        per_seed = ", ".join(f"{value:.3f}" for value in cell["per_seed_survival"])
        print(
            f"| {cell['label']} | {cell['survival']:.4f} | {per_seed} | "
            f"{cell['matched_regret']['value']:+.6f} | "
            f"{cell['rate_mae']['value']:+.6f} |"
        )
    print()
    for name, gate in result["gates"].items():
        if isinstance(gate, dict) and "verdict" in gate:
            print(f"{name}: {gate['verdict']}")
    print(f"all_pass: {result['gates']['all_pass']}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
