"""Parameter-free first-use occupancy law for Probe71.

An enumerative message model must observe a complete need-by-size cell before
using it.  A factorized model can compose already-grounded parts on that cell's
first natural use.  This module states the resulting cold-start rent in closed
form and supplies order-sensitive audits.  It fits no coefficient and imports no
learner.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from random import Random
from typing import Hashable, Iterable, Mapping, Sequence, TypeVar

import numpy as np

from homesocial.island.expanded_body import (
    EXPANDED_BODY_SCHEMA,
    LEGACY_BODY_SCHEMA,
    BodySchema,
    OracleLife,
    run_oracle_life,
)


Combination = TypeVar("Combination", bound=Hashable)

SURVEY_SEED_BASE = 2_500_000_000
TREATMENT_SEED_BASE = 2_600_000_000
SEED_STRIDE = 2_000_000
SURVEY_SEEDS = 5
SURVEY_LIVES = 100


@dataclass(frozen=True)
class CompositionRent:
    """Formula and ordered-stream readouts for one request history."""

    opportunities: int
    combinations: int
    occupied_combinations: int
    predicted_rate: float
    empirical_rate: float
    permuted_rate: float
    iid_resampled_rate: float
    serial_residual: float


def expected_first_use_rent(
    probabilities: Mapping[Combination, float],
    consequential_probabilities: Mapping[Combination, float],
    *,
    opportunities: int,
) -> float:
    """Expected consequential first-use events per opportunity.

    For cell ``c``, ``1 - (1 - p_c) ** T`` is the probability that it appears at
    least once in ``T`` exchangeable i.i.d. opportunities.  It can charge the
    deliberately smallest compositional rent once, on that first appearance,
    and does so with probability ``q_c``.
    """

    if opportunities <= 0:
        raise ValueError("opportunities must be positive.")
    if not probabilities:
        raise ValueError("At least one combination probability is required.")
    unknown = set(consequential_probabilities).difference(probabilities)
    if unknown:
        raise KeyError(f"Consequential probabilities have unknown cells: {unknown!r}.")
    total = 0.0
    for combination, probability in probabilities.items():
        _validate_probability(probability, label=f"p[{combination!r}]")
        consequential = float(consequential_probabilities.get(combination, 0.0))
        _validate_probability(consequential, label=f"q[{combination!r}]")
        total += consequential * (1.0 - (1.0 - probability) ** opportunities)
    if abs(sum(float(value) for value in probabilities.values()) - 1.0) > 1e-9:
        raise ValueError("Combination probabilities must sum to one.")
    return total / opportunities


def empirical_first_use_rent(
    combinations: Sequence[Combination],
    consequential: Sequence[bool],
) -> float:
    """Observed consequential first uses on an ordered request stream."""

    if len(combinations) != len(consequential):
        raise ValueError("Combination and consequence streams must align.")
    if not combinations:
        raise ValueError("The request stream must be non-empty.")
    seen: set[Combination] = set()
    rent = 0
    for combination, matters in zip(combinations, consequential, strict=True):
        if combination not in seen:
            rent += int(bool(matters))
            seen.add(combination)
    return rent / len(combinations)


def estimate_cell_probabilities(
    combinations: Sequence[Combination],
    consequential: Sequence[bool],
) -> tuple[dict[Combination, float], dict[Combination, float]]:
    """Raw cell frequencies and within-cell consequence rates; no fitted terms."""

    if len(combinations) != len(consequential):
        raise ValueError("Combination and consequence streams must align.")
    if not combinations:
        raise ValueError("The request stream must be non-empty.")
    counts = Counter(combinations)
    consequential_counts: dict[Combination, int] = defaultdict(int)
    for combination, matters in zip(combinations, consequential, strict=True):
        consequential_counts[combination] += int(bool(matters))
    total = len(combinations)
    probabilities = {
        combination: count / total for combination, count in counts.items()
    }
    consequence_rates = {
        combination: consequential_counts[combination] / count
        for combination, count in counts.items()
    }
    return probabilities, consequence_rates


def permute_requests(
    combinations: Sequence[Combination],
    consequential: Sequence[bool],
    *,
    seed: int,
) -> tuple[tuple[Combination, ...], tuple[bool, ...]]:
    """Remove serial order while preserving every cell/consequence pair."""

    if len(combinations) != len(consequential):
        raise ValueError("Combination and consequence streams must align.")
    order = list(range(len(combinations)))
    Random(seed).shuffle(order)
    return (
        tuple(combinations[index] for index in order),
        tuple(bool(consequential[index]) for index in order),
    )


def resample_requests(
    combinations: Sequence[Combination],
    consequential: Sequence[bool],
    *,
    seed: int,
) -> tuple[tuple[Combination, ...], tuple[bool, ...]]:
    """Draw an i.i.d. stream from the empirical pair distribution.

    Unlike :func:`permute_requests`, sampling is with replacement. Rare cells
    can therefore be absent, which is the sampling process described by
    :func:`expected_first_use_rent`.
    """

    if len(combinations) != len(consequential):
        raise ValueError("Combination and consequence streams must align.")
    if not combinations:
        raise ValueError("The request stream must be non-empty.")
    rng = Random(seed)
    order = [rng.randrange(len(combinations)) for _ in combinations]
    return (
        tuple(combinations[index] for index in order),
        tuple(bool(consequential[index]) for index in order),
    )


def summarize_composition_rent(
    combinations: Sequence[Combination],
    consequential: Sequence[bool],
    *,
    possible_combinations: int,
    permutation_seed: int,
) -> CompositionRent:
    """Compute the preregistered formula, live stream and permuted control."""

    if possible_combinations <= 0:
        raise ValueError("possible_combinations must be positive.")
    probabilities, consequence_rates = estimate_cell_probabilities(
        combinations, consequential
    )
    if len(probabilities) > possible_combinations:
        raise ValueError("Observed more cells than the declared message space.")
    predicted = expected_first_use_rent(
        probabilities,
        consequence_rates,
        opportunities=len(combinations),
    )
    empirical = empirical_first_use_rent(combinations, consequential)
    permuted_combinations, permuted_consequential = permute_requests(
        combinations, consequential, seed=permutation_seed
    )
    permuted = empirical_first_use_rent(
        permuted_combinations, permuted_consequential
    )
    resampled_combinations, resampled_consequential = resample_requests(
        combinations, consequential, seed=permutation_seed + 1_000_003
    )
    resampled = empirical_first_use_rent(
        resampled_combinations, resampled_consequential
    )
    return CompositionRent(
        opportunities=len(combinations),
        combinations=possible_combinations,
        occupied_combinations=len(probabilities),
        predicted_rate=predicted,
        empirical_rate=empirical,
        permuted_rate=permuted,
        iid_resampled_rate=resampled,
        serial_residual=empirical - permuted,
    )


def pool_life_summaries(rows: Iterable[CompositionRent]) -> dict[str, float]:
    """Opportunity-weighted aggregation without treating lives as equal length."""

    records = tuple(rows)
    if not records:
        raise ValueError("At least one life summary is required.")
    opportunities = sum(row.opportunities for row in records)
    return {
        "lives": float(len(records)),
        "opportunities": float(opportunities),
        "predicted_rate": sum(
            row.predicted_rate * row.opportunities for row in records
        )
        / opportunities,
        "empirical_rate": sum(
            row.empirical_rate * row.opportunities for row in records
        )
        / opportunities,
        "permuted_rate": sum(
            row.permuted_rate * row.opportunities for row in records
        )
        / opportunities,
        "iid_resampled_rate": sum(
            row.iid_resampled_rate * row.opportunities for row in records
        )
        / opportunities,
    }


def run_construction_survey(
    *,
    lives: int = SURVEY_LIVES,
    seeds: int = SURVEY_SEEDS,
    seed_base: int = SURVEY_SEED_BASE,
    progress_path: Path | None = None,
) -> dict[str, object]:
    """Run Probe71's locked K3/K5 oracle construction survey.

    Seed is the outer loop so one complete paired replicate lands before the
    next, matching the repository's long-run discipline. The progress artifact
    is replaced after every seed and is safe to inspect mid-run.
    """

    _check_seed_isolation(lives=lives, seeds=seeds, seed_base=seed_base)
    if lives <= 0 or seeds <= 0:
        raise ValueError("lives and seeds must be positive.")
    schemas = {"K3": LEGACY_BODY_SCHEMA, "K5": EXPANDED_BODY_SCHEMA}
    per_condition: dict[str, list[dict[str, object]]] = {
        name: [] for name in schemas
    }
    for seed_index in range(seeds):
        for name, schema in schemas.items():
            rows = [
                run_oracle_life(
                    schema,
                    seed=seed_base + seed_index * SEED_STRIDE + life,
                )
                for life in range(lives)
            ]
            per_condition[name].append(
                _summarize_seed(
                    rows,
                    schema=schema,
                    seed_index=seed_index,
                    permutation_seed=(
                        seed_base + seed_index * SEED_STRIDE + len(schema) * 10_000
                    ),
                )
            )
        partial = _survey_record(
            per_condition,
            lives=lives,
            seeds_completed=seed_index + 1,
            seed_base=seed_base,
        )
        if progress_path is not None:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            progress_path.write_text(json.dumps(partial, indent=2) + "\n")
    return _survey_record(
        per_condition,
        lives=lives,
        seeds_completed=seeds,
        seed_base=seed_base,
    )


def score_continuation(record: Mapping[str, object]) -> dict[str, object]:
    """Apply the six continuation clauses fixed in the preregistration."""

    conditions = record["conditions"]
    if not isinstance(conditions, Mapping):
        raise TypeError("record.conditions must be a mapping.")
    k3 = conditions["K3"]
    k5 = conditions["K5"]
    if not isinstance(k3, Mapping) or not isinstance(k5, Mapping):
        raise TypeError("K3 and K5 summaries must be mappings.")

    k5_seeds = k5["per_seed"]
    if not isinstance(k5_seeds, Sequence):
        raise TypeError("K5 per_seed must be a sequence.")
    viable_seeds = sum(
        float(row["survival"]) >= 0.70  # type: ignore[index]
        for row in k5_seeds
    )
    viability = float(k5["survival"]) >= 0.75 and viable_seeds >= 4
    health_real = (
        float(k5["request_share"].get("health", 0.0)) >= 0.05  # type: ignore[union-attr]
        or float(k5["death_share"].get("health", 0.0)) >= 0.05  # type: ignore[union-attr]
    )
    instrument = all(
        float(condition["iid_formula_mae"]) <= 0.015
        for condition in (k3, k5)
    )
    natural_range = float(k5["predicted_rent_p90_p10"]) >= 0.03
    gap_step = float(k5["median_axis_gap"]) - float(k3["median_axis_gap"])
    gap_moved = abs(gap_step) >= 0.02
    legacy_schema = (
        LEGACY_BODY_SCHEMA.names == ("food", "water", "energy")
        and len(LEGACY_BODY_SCHEMA.surfaces) == 9
    )
    clauses = {
        "C1_legacy_schema": legacy_schema,
        "C2_five_axis_viability": viability,
        "C3_health_is_real": health_real,
        "C4_instrument_validity": instrument,
        "C5_natural_range": natural_range,
        "C6_axis_gap_moved": gap_moved,
    }
    return {
        "clauses": clauses,
        "viable_seeds": viable_seeds,
        "required_viable_seeds": 4,
        "axis_gap_step_K5_minus_K3": gap_step,
        # Full-suite inertness is an external construction check and is reported
        # beside this score; it is never inferred from a schema assertion.
        "requires_full_suite_inertness": True,
        "licensed_subject_to_full_suite": all(clauses.values()),
    }


def _summarize_seed(
    lives: Sequence[OracleLife],
    *,
    schema: BodySchema,
    seed_index: int,
    permutation_seed: int,
) -> dict[str, object]:
    rent_rows: list[CompositionRent] = []
    gaps: list[float] = []
    margins: list[float] = []
    request_counts = Counter({need: 0 for need in schema.names})
    shock_counts = Counter({need: 0 for need in schema.names})
    death_counts = Counter({need: 0 for need in schema.names})
    formula_errors: list[float] = []
    predicted_life_rates: list[float] = []
    for life_index, life in enumerate(lives):
        combinations = tuple(event.combination for event in life.events)
        consequential = tuple(
            event.consequential_first_use for event in life.events
        )
        if combinations:
            rent = summarize_composition_rent(
                combinations,
                consequential,
                possible_combinations=len(schema) * 2,
                permutation_seed=permutation_seed + life_index,
            )
            rent_rows.append(rent)
            formula_errors.append(
                abs(rent.predicted_rate - rent.iid_resampled_rate)
            )
            predicted_life_rates.append(rent.predicted_rate)
        for event in life.events:
            request_counts[event.need] += 1
            gaps.append(event.axis_gap)
            margins.append(event.margin)
        shock_counts.update(life.shock_counts)
        if life.death_need is not None:
            death_counts[life.death_need] += 1
    pooled = pool_life_summaries(rent_rows)
    total_requests = sum(request_counts.values())
    deaths = sum(death_counts.values())
    return {
        "seed": seed_index,
        "lives": len(lives),
        "survival": sum(life.survived for life in lives) / len(lives),
        "mean_steps": float(np.mean([life.steps for life in lives])),
        "requests": total_requests,
        "request_share": {
            need: request_counts[need] / max(1, total_requests)
            for need in schema.names
        },
        "shock_counts": dict(shock_counts),
        "death_counts": dict(death_counts),
        "death_share": {
            need: death_counts[need] / max(1, deaths) for need in schema.names
        },
        "median_axis_gap": float(np.median(gaps)),
        "median_margin": float(np.median(margins)),
        "iid_formula_mae": float(np.mean(formula_errors)),
        "predicted_rent_p10": float(np.quantile(predicted_life_rates, 0.10)),
        "predicted_rent_p90": float(np.quantile(predicted_life_rates, 0.90)),
        "rent": pooled,
        # Internal pooling inputs. `_pool_seed_condition` consumes and removes
        # them so the scored aggregate follows the preregistered pooled-life
        # wording rather than taking quantiles of per-seed quantiles.
        "_axis_gaps": gaps,
        "_margins": margins,
        "_predicted_life_rates": predicted_life_rates,
    }


def _survey_record(
    per_condition: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    lives: int,
    seeds_completed: int,
    seed_base: int,
) -> dict[str, object]:
    conditions = {
        name: _pool_seed_condition(rows) for name, rows in per_condition.items()
    }
    record: dict[str, object] = {
        "probe": 71,
        "status": "construction_survey",
        "seed_base": seed_base,
        "seed_stride": SEED_STRIDE,
        "seeds_completed": seeds_completed,
        "lives_per_seed": lives,
        "conditions": conditions,
    }
    if all(per_condition.values()):
        record["continuation"] = score_continuation(record)
    return record


def _pool_seed_condition(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not rows:
        return {"per_seed": []}
    pooled_gaps = [
        float(value) for row in rows for value in row["_axis_gaps"]  # type: ignore[union-attr]
    ]
    pooled_margins = [
        float(value) for row in rows for value in row["_margins"]  # type: ignore[union-attr]
    ]
    pooled_predicted = [
        float(value)
        for row in rows
        for value in row["_predicted_life_rates"]  # type: ignore[union-attr]
    ]
    public_rows = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in rows
    ]
    needs = tuple(rows[0]["request_share"].keys())  # type: ignore[union-attr]
    request_totals = {
        need: sum(
            int(row["requests"]) * float(row["request_share"][need])  # type: ignore[index]
            for row in rows
        )
        for need in needs
    }
    death_totals = {
        need: sum(int(row["death_counts"].get(need, 0)) for row in rows)  # type: ignore[union-attr]
        for need in needs
    }
    total_requests = sum(request_totals.values())
    total_deaths = sum(death_totals.values())
    return {
        "per_seed": public_rows,
        "survival": float(np.mean([float(row["survival"]) for row in rows])),
        "mean_steps": float(np.mean([float(row["mean_steps"]) for row in rows])),
        "request_share": {
            need: request_totals[need] / max(1.0, total_requests) for need in needs
        },
        "death_counts": death_totals,
        "death_share": {
            need: death_totals[need] / max(1, total_deaths) for need in needs
        },
        "median_axis_gap": float(np.median(pooled_gaps)),
        "median_margin": float(np.median(pooled_margins)),
        "iid_formula_mae": float(
            np.mean([float(row["iid_formula_mae"]) for row in rows])
        ),
        "predicted_rent_p90_p10": float(
            np.quantile(pooled_predicted, 0.90)
            - np.quantile(pooled_predicted, 0.10)
        ),
    }


def _check_seed_isolation(*, lives: int, seeds: int, seed_base: int) -> None:
    if lives > SEED_STRIDE:
        raise ValueError("Lives exceed the seed stride and would overlap.")
    survey_span = (seed_base, seed_base + seeds * SEED_STRIDE)
    treatment_span = (
        TREATMENT_SEED_BASE,
        TREATMENT_SEED_BASE + SURVEY_SEEDS * SEED_STRIDE,
    )
    if survey_span[0] < treatment_span[1] and treatment_span[0] < survey_span[1]:
        raise ValueError("Construction and treatment seed bands overlap.")


def _validate_probability(value: float, *, label: str) -> None:
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{label} must lie in [0, 1].")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lives", type=int, default=SURVEY_LIVES)
    parser.add_argument("--seeds", type=int, default=SURVEY_SEEDS)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "runs/organism/probe71_expanded_body/construction_survey.json"
        ),
    )
    args = parser.parse_args()
    record = run_construction_survey(
        lives=args.lives,
        seeds=args.seeds,
        progress_path=args.out,
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
