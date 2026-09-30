"""Guards for Probe73's one-factor rate lesion and resumable treatment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from homesocial.island.report import ReportConfig
from homesocial.organism.future_request import RequestLedger
from homesocial.organism.learned_rate_survival import (
    CELL_SPECS,
    _summarize,
    _validate_run,
    _word_for_rates,
    grade,
    run_experiment,
)


def _cells(
    *,
    species=(0.30,) * 5,
    learned=(0.36,) * 5,
    true=(0.40,) * 5,
    regret=(0.003,) * 5,
    recovery=(0.002,) * 5,
):
    survival = {
        "species_rate": list(species),
        "learned_rate": list(learned),
        "true_rate": list(true),
    }
    out = []
    for spec in CELL_SPECS:
        label = str(spec["label"])
        values = survival[label]
        out.append(
            {
                **spec,
                "survival": sum(values) / len(values),
                "per_seed_survival": values,
                "per_seed_matched_regret_value": list(regret),
                "per_seed_rate_mae_value": list(recovery),
            }
        )
    return out


def test_all_preregistered_gates_pass_on_the_locked_pattern():
    gates = grade(_cells())

    assert gates["G1_learned_rate_reaches_survival"]["verdict"] == "pass"
    assert gates["G2_matched_physical_ceiling"]["verdict"] == "pass"
    assert gates["G3_matched_regret_predicts_sign"]["verdict"] == "pass"
    assert gates["G4_rate_was_learned"]["verdict"] == "pass"
    assert gates["G5_construction_is_viable"]["verdict"] == "pass"
    assert gates["all_pass"] is True


def test_survival_gate_needs_a_clear_interval_and_four_positive_blocks():
    cells = _cells(learned=(0.36, 0.36, 0.36, 0.28, 0.28))

    gate = grade(cells)["G1_learned_rate_reaches_survival"]
    assert gate["positive"] == 3
    assert gate["same_sign"] == 3
    assert gate["verdict"] == "fail"


def test_each_control_gate_is_binding():
    no_ceiling = grade(_cells(true=(0.30,) * 5))
    bad_regret = grade(_cells(regret=(-0.001,) * 5))
    no_recovery = grade(_cells(recovery=(0.0,) * 5))
    collapsed = grade(_cells(true=(0.14,) * 5))

    assert no_ceiling["G2_matched_physical_ceiling"]["verdict"] == "fail"
    assert bad_regret["G3_matched_regret_predicts_sign"]["verdict"] == "fail"
    assert no_recovery["G4_rate_was_learned"]["verdict"] == "fail"
    assert collapsed["G5_construction_is_viable"]["verdict"] == "fail"


def test_rate_lesion_holds_state_uptake_ledger_and_horizon_fixed():
    class Tier:
        def __init__(self):
            self.levels = np.asarray([0.50, 0.44, 0.60])
            self.point_calls = 0

        def point(self):
            self.point_calls += 1
            return self.levels

        def believed_uptake(self):
            return {"food": 1.0, "water": 1.0, "energy": 1.0}

    tier = Tier()
    report = ReportConfig()
    ledger = RequestLedger(help_period=6, delay=24, expected_portion=0.4)
    ledger.reset()
    species = np.asarray([0.004, 0.006, 0.008])
    learned = np.asarray([0.020, 0.006, 0.008])

    species_word = _word_for_rates(
        tier, species, ledger, now=0, delay=24, report=report
    )
    learned_word = _word_for_rates(
        tier, learned, ledger, now=0, delay=24, report=report
    )

    assert species_word == "water"
    assert learned_word == "food"
    assert tier.point_calls == 2
    assert tier.levels.tolist() == [0.50, 0.44, 0.60]
    assert ledger.counts(0) == (0, 0, 0)


def test_chunk_summaries_weight_tick_endpoints_and_reconstruct_survival():
    chunks = []
    for seed in range(2):
        for spec in CELL_SPECS:
            for chunk in range(2):
                survival = 0.2 + 0.4 * seed + 0.2 * chunk
                ticks = 10 if chunk == 0 else 30
                chunks.append(
                    {
                        "seed_index": seed,
                        "cell": spec["label"],
                        "chunk_index": chunk,
                        "lives": 5,
                        "survival": survival,
                        "mean_life_steps": 100.0 + seed,
                        "scored_ticks": ticks,
                        "matched_word_differences": ticks // 2,
                        "matched_regret": {
                            "learned": 0.001 + 0.002 * chunk,
                            "species": 0.005,
                            "value": 0.0,
                        },
                        "rate_mae": {
                            "learned": 0.002,
                            "species": 0.006 + 0.002 * chunk,
                            "value": 0.0,
                        },
                        "deaths_by_need": {"water": 5 - int(5 * survival)},
                    }
                )

    cells = _summarize(chunks, seeds=2, lives_per_seed=10)
    first = cells[0]
    assert first["per_seed_survival"] == pytest.approx([0.3, 0.7])
    assert first["survival"] == 0.5
    assert first["matched_regret"]["learned"] == pytest.approx(0.0025)
    assert first["matched_regret"]["value"] == pytest.approx(0.0025)
    assert first["rate_mae"]["value"] == pytest.approx(0.0055)
    assert first["matched_word_difference_share"] == 0.5


def test_seed_and_chunk_boundaries_are_guarded():
    with pytest.raises(ValueError, match="divisible"):
        _validate_run(
            seed_base=3_000_000_000,
            seeds=5,
            lives_per_seed=141,
            chunk_lives=20,
        )
    with pytest.raises(ValueError, match="reuse"):
        _validate_run(
            seed_base=2_800_000_000,
            seeds=5,
            lives_per_seed=140,
            chunk_lives=20,
        )


def test_resume_skips_completed_chunks(tmp_path: Path):
    calls = []

    def runner(_model, _organism, **kwargs):
        calls.append((kwargs["rate_mode"], kwargs["seed_base"]))
        return {
            "rate_mode": kwargs["rate_mode"],
            "lives": kwargs["lives"],
            "survival": 0.5,
            "mean_life_steps": 200.0,
            "scored_ticks": 10,
            "matched_word_differences": 2,
            "matched_regret": {
                "learned": 0.002,
                "species": 0.005,
                "value": 0.003,
            },
            "rate_mae": {
                "learned": 0.001,
                "species": 0.004,
                "value": 0.003,
            },
            "deaths_by_need": {"water": kwargs["lives"] // 2},
        }

    out = tmp_path / "progress.json"
    first = run_experiment(
        object(),
        object(),
        seed_base=3_100_000_000,
        seeds=1,
        lives_per_seed=2,
        chunk_lives=1,
        out=out,
        runner=runner,
    )
    assert first["complete"] is True
    assert len(calls) == 6
    assert not out.with_suffix(out.suffix + ".tmp").exists()

    second = run_experiment(
        object(),
        object(),
        seed_base=3_100_000_000,
        seeds=1,
        lives_per_seed=2,
        chunk_lives=1,
        out=out,
        runner=runner,
    )
    assert second["complete"] is True
    assert len(calls) == 6
