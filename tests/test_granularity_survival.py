"""Guards for Probe72's locked survival endpoint and resumable harness."""

from __future__ import annotations

from pathlib import Path

import pytest

from homesocial.organism.granularity_survival import (
    CELL_SPECS,
    _summarize,
    _validate_run,
    grade,
    run_experiment,
)


def _cells(
    *,
    baseline_oracle=(0.40,) * 5,
    baseline_state=(0.38,) * 5,
    baseline_population=(0.30,) * 5,
    fine_oracle=(0.50,) * 5,
    fine_state=(0.42,) * 5,
    fine_population=(0.34,) * 5,
    baseline_rate=1.0 / 15.0,
    fine_rate=1.0 / 15.0,
):
    values = {
        "baseline_oracle": baseline_oracle,
        "baseline_state_oracle": baseline_state,
        "baseline_population": baseline_population,
        "fine_oracle": fine_oracle,
        "fine_state_oracle": fine_state,
        "fine_population": fine_population,
    }
    out = []
    for spec in CELL_SPECS:
        label = str(spec["label"])
        per_seed = list(values[label])
        out.append(
            {
                **spec,
                "survival": sum(per_seed) / len(per_seed),
                "per_seed_survival": per_seed,
                "grant_rate": fine_rate
                if spec["ecology"] == "fine"
                else baseline_rate,
            }
        )
    return out


def test_all_locked_gates_pass_on_the_preregistered_pattern():
    gates = grade(_cells())

    assert gates["G1_grain_carries_rate_value"]["verdict"] == "pass"
    assert gates["G2_state_specificity"]["verdict"] == "pass"
    assert gates["G3_physical_rate_value_exists"]["verdict"] == "pass"
    assert gates["G4_both_ecologies_are_viable"]["verdict"] == "pass"
    assert gates["G5_nutrition_is_fixed"]["verdict"] == "pass"
    assert gates["all_pass"] is True


def test_grain_gate_needs_an_interval_clear_of_zero_and_four_seeds():
    cells = _cells(
        fine_oracle=(0.50, 0.50, 0.50, 0.40, 0.40),
        fine_state=(0.42, 0.42, 0.42, 0.42, 0.42),
    )

    gate = grade(cells)["G1_grain_carries_rate_value"]
    assert gate["positive"] == 3
    assert gate["same_sign"] == 3
    assert gate["verdict"] == "fail"


def test_state_step_is_the_specificity_lesion():
    cells = _cells(fine_population=(0.16,) * 5)

    assert grade(cells)["G2_state_specificity"]["verdict"] == "fail"


def test_positive_interaction_cannot_hide_a_negative_fine_rate_value():
    cells = _cells(
        baseline_oracle=(0.30,) * 5,
        baseline_state=(0.40,) * 5,
        fine_oracle=(0.38,) * 5,
        fine_state=(0.42,) * 5,
    )

    gates = grade(cells)
    assert gates["G1_grain_carries_rate_value"]["verdict"] == "pass"
    assert gates["G3_physical_rate_value_exists"]["verdict"] == "fail"
    assert gates["all_pass"] is False


def test_viability_and_exact_nutrition_are_binding_controls():
    collapsed = grade(_cells(baseline_oracle=(0.29,) * 5))
    moved_food = grade(_cells(fine_rate=0.067))

    assert collapsed["G4_both_ecologies_are_viable"]["verdict"] == "fail"
    assert moved_food["G5_nutrition_is_fixed"]["verdict"] == "fail"


def test_chunk_summaries_reconstruct_seed_survival_exactly():
    chunks = []
    for seed in range(2):
        for spec in CELL_SPECS:
            for chunk in range(2):
                survival = 0.2 + 0.4 * seed + 0.2 * chunk
                chunks.append(
                    {
                        "seed_index": seed,
                        "cell": spec["label"],
                        "chunk_index": chunk,
                        "lives": 5,
                        "survival": survival,
                        "mean_life_steps": 100.0 + seed,
                        "deaths_by_need": {"water": 5 - int(5 * survival)},
                    }
                )

    cells = _summarize(
        chunks, seeds=2, lives_per_seed=10, base_expected_grant=0.4
    )
    assert cells[0]["per_seed_survival"] == pytest.approx([0.3, 0.7])
    assert cells[0]["survival"] == 0.5
    assert cells[0]["grant_rate"] == pytest.approx(0.4 / 6.0)


def test_seed_and_chunk_boundaries_are_guarded():
    with pytest.raises(ValueError, match="divisible"):
        _validate_run(
            seed_base=2_800_000_000,
            seeds=5,
            lives_per_seed=141,
            chunk_lives=20,
        )
    with pytest.raises(ValueError, match="reuse"):
        _validate_run(
            seed_base=2_600_000_000,
            seeds=5,
            lives_per_seed=140,
            chunk_lives=20,
        )


def test_resume_skips_chunks_already_in_the_progress_artifact(tmp_path: Path):
    calls = []

    def runner(_model, _organism, **kwargs):
        calls.append(kwargs["seed_base"])
        return {
            "lives": kwargs["lives"],
            "survival": 0.5,
            "mean_life_steps": 200.0,
            "deaths_by_need": {"water": kwargs["lives"] // 2},
        }

    class Report:
        portion_small = 0.2
        portion_large = 0.6

    class Organism:
        report = Report()

    out = tmp_path / "progress.json"
    first = run_experiment(
        object(),
        Organism(),
        seed_base=2_900_000_000,
        seeds=1,
        lives_per_seed=2,
        chunk_lives=1,
        out=out,
        runner=runner,
    )
    assert first["complete"] is True
    assert len(calls) == 12
    assert not out.with_suffix(out.suffix + ".tmp").exists()

    second = run_experiment(
        object(),
        Organism(),
        seed_base=2_900_000_000,
        seeds=1,
        lives_per_seed=2,
        chunk_lives=1,
        out=out,
        runner=runner,
    )
    assert second["complete"] is True
    assert len(calls) == 12
