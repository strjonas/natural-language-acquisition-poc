"""Probe72: does decision granularity carry individual-rate value to survival?

Preregistration:
``docs/decisions/2026-08-22-granularity-survival-preregistration.md``.

This module adds no mechanism. It gives Probe68's clean ``oracle -
state_oracle`` rate contrast a closed-loop survival endpoint in the baseline and
fine-quantum ecologies. Long runs are checkpointed after every chunk and can be
resumed without spending completed lives again.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Callable

import numpy as np

from homesocial.organism.future_request import run_closed_loop
from homesocial.organism.model import OrganismModel
from homesocial.organism.train import OrganismConfig, load_organism_checkpoint


DELAY = 18
SEEDS = 5
LIVES_PER_SEED = 140
CHUNK_LIVES = 20
SEED_STRIDE = 2_000_000
PILOT_SEED_BASE = 2_700_000_000
TREATMENT_SEED_BASE = 2_800_000_000
VIABILITY_FLOOR = 0.30

# The order is deliberate: one complete paired replicate lands before the next
# seed begins. Each cell is divided into chunks only for durable progress.
CELL_SPECS: tuple[dict[str, object], ...] = (
    {
        "label": "baseline_oracle",
        "ecology": "baseline",
        "arm": "oracle",
        "portion_scale": 1.0,
        "help_period": 6,
    },
    {
        "label": "baseline_state_oracle",
        "ecology": "baseline",
        "arm": "state_oracle",
        "portion_scale": 1.0,
        "help_period": 6,
    },
    {
        "label": "baseline_population",
        "ecology": "baseline",
        "arm": "population",
        "portion_scale": 1.0,
        "help_period": 6,
    },
    {
        "label": "fine_oracle",
        "ecology": "fine",
        "arm": "oracle",
        "portion_scale": 0.5,
        "help_period": 3,
    },
    {
        "label": "fine_state_oracle",
        "ecology": "fine",
        "arm": "state_oracle",
        "portion_scale": 0.5,
        "help_period": 3,
    },
    {
        "label": "fine_population",
        "ecology": "fine",
        "arm": "population",
        "portion_scale": 0.5,
        "help_period": 3,
    },
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


def _difference(a: list[float], b: list[float]) -> dict[str, object]:
    if len(a) != len(b):
        raise ValueError("paired contrasts must have the same number of seeds")
    values = [float(left) - float(right) for left, right in zip(a, b)]
    interval = _paired_interval(values)
    mean = float(interval["mean"])
    return {
        **interval,
        "values": values,
        "positive": int(sum(value > 0.0 for value in values)),
        "negative": int(sum(value < 0.0 for value in values)),
        "same_sign": int(sum(value * mean > 0.0 for value in values)),
    }


def grade(cells: list[dict[str, object]]) -> dict[str, object]:
    """Score only the five gates locked in the preregistration."""

    index = {str(cell["label"]): cell for cell in cells}
    expected = {str(spec["label"]) for spec in CELL_SPECS}
    if set(index) != expected:
        missing = sorted(expected - set(index))
        extra = sorted(set(index) - expected)
        raise ValueError(f"wrong cells; missing={missing}, extra={extra}")

    def survival(label: str) -> list[float]:
        return [float(value) for value in index[label]["per_seed_survival"]]

    baseline_rate = _difference(
        survival("baseline_oracle"), survival("baseline_state_oracle")
    )
    fine_rate = _difference(
        survival("fine_oracle"), survival("fine_state_oracle")
    )
    baseline_state = _difference(
        survival("baseline_state_oracle"), survival("baseline_population")
    )
    fine_state = _difference(
        survival("fine_state_oracle"), survival("fine_population")
    )
    rate_step = _difference(fine_rate["values"], baseline_rate["values"])
    state_step = _difference(fine_state["values"], baseline_state["values"])

    g1_pass = (
        float(rate_step["mean"]) > 0.0
        and float(rate_step["low"]) > 0.0
        and int(rate_step["same_sign"]) >= 4
    )
    g2_pass = abs(float(state_step["mean"])) < float(rate_step["mean"])
    g3_pass = (
        float(fine_rate["mean"]) > 0.0
        and float(fine_rate["low"]) > 0.0
        and int(fine_rate["same_sign"]) >= 4
    )

    oracle_survival = {
        "baseline": float(index["baseline_oracle"]["survival"]),
        "fine": float(index["fine_oracle"]["survival"]),
    }
    g4_pass = all(value >= VIABILITY_FLOOR for value in oracle_survival.values())

    grant_rates = {
        "baseline": float(index["baseline_oracle"]["grant_rate"]),
        "fine": float(index["fine_oracle"]["grant_rate"]),
    }
    grant_rate_difference = grant_rates["fine"] - grant_rates["baseline"]
    g5_pass = abs(grant_rate_difference) <= 1e-12

    gates: dict[str, object] = {
        "G1_grain_carries_rate_value": {
            **rate_step,
            "verdict": "pass" if g1_pass else "fail",
        },
        "G2_state_specificity": {
            "rate_step": rate_step,
            "state_step": state_step,
            "verdict": "pass" if g2_pass else "fail",
        },
        "G3_physical_rate_value_exists": {
            **fine_rate,
            "verdict": "pass" if g3_pass else "fail",
        },
        "G4_both_ecologies_are_viable": {
            "oracle_survival": oracle_survival,
            "floor": VIABILITY_FLOOR,
            "verdict": "pass" if g4_pass else "fail",
        },
        "G5_nutrition_is_fixed": {
            "grant_rates": grant_rates,
            "difference": grant_rate_difference,
            "tolerance": 1e-12,
            "verdict": "pass" if g5_pass else "fail",
        },
    }
    gates["all_pass"] = all(
        gate["verdict"] == "pass" for gate in gates.values() if isinstance(gate, dict)
    )
    gates["contrasts"] = {
        "baseline_rate_value": baseline_rate,
        "fine_rate_value": fine_rate,
        "baseline_state_value": baseline_state,
        "fine_state_value": fine_state,
    }
    return gates


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
    if seed_base < 2_700_000_000:
        raise ValueError("Probe72 must not reuse an earlier experiment's seed band")


def _summarize(
    chunks: list[dict[str, object]],
    *,
    seeds: int,
    lives_per_seed: int,
    base_expected_grant: float,
) -> list[dict[str, object]]:
    """Collapse checkpoint chunks into the five preregistered seed blocks."""

    cells: list[dict[str, object]] = []
    for spec in CELL_SPECS:
        label = str(spec["label"])
        per_seed_survival: list[float] = []
        per_seed_mean_steps: list[float] = []
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
            per_seed_survival.append(survived / lives)
            per_seed_mean_steps.append(steps / lives)
            for record in records:
                for need, count in dict(record["deaths_by_need"]).items():
                    deaths[str(need)] += int(count)

        scale = float(spec["portion_scale"])
        period = int(spec["help_period"])
        cells.append(
            {
                **spec,
                "lives": seeds * lives_per_seed,
                "expected_grant": base_expected_grant * scale,
                "grant_rate": base_expected_grant * scale / period,
                "survival": float(np.mean(per_seed_survival)),
                "per_seed_survival": per_seed_survival,
                "mean_life_steps": float(np.mean(per_seed_mean_steps)),
                "per_seed_mean_life_steps": per_seed_mean_steps,
                "deaths_by_need": dict(deaths),
            }
        )
    return cells


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


def run_experiment(
    model: OrganismModel,
    organism: OrganismConfig,
    *,
    seed_base: int = TREATMENT_SEED_BASE,
    seeds: int = SEEDS,
    lives_per_seed: int = LIVES_PER_SEED,
    chunk_lives: int = CHUNK_LIVES,
    out: Path | None = None,
    runner: Callable[..., dict[str, object]] = run_closed_loop,
) -> dict[str, object]:
    """Run or resume all cells, writing durable progress after every chunk."""

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
                    arm_name=str(spec["arm"]),
                    world_name="metabolic",
                    delay=DELAY,
                    lives=chunk_lives,
                    seed_base=life_base,
                    portion_scale=float(spec["portion_scale"]),
                    help_period=int(spec["help_period"]),
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
                    f"seed {seed_index + 1}/{seeds} {label:<24} "
                    f"chunk {chunk_index + 1}/{chunks_per_seed} "
                    f"survival {float(record['survival']):.3f} "
                    f"[{len(completed)}/{total_units}]",
                    flush=True,
                )

    base_expected_grant = (
        float(organism.report.portion_small) + float(organism.report.portion_large)
    ) / 2.0
    cells = _summarize(
        chunks,
        seeds=seeds,
        lives_per_seed=lives_per_seed,
        base_expected_grant=base_expected_grant,
    )
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
        default="runs/organism/probe72_granularity_survival/treatment.json",
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

    print("\n| cell | survival | per seed | mean life steps |")
    print("|---|---:|---|---:|")
    for cell in result["cells"]:
        per_seed = ", ".join(f"{value:.3f}" for value in cell["per_seed_survival"])
        print(
            f"| {cell['label']} | {cell['survival']:.4f} | {per_seed} | "
            f"{cell['mean_life_steps']:.1f} |"
        )
    print()
    for name, gate in result["gates"].items():
        if isinstance(gate, dict) and "verdict" in gate:
            print(f"{name}: {gate['verdict']}")
    print(f"all_pass: {result['gates']['all_pass']}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
