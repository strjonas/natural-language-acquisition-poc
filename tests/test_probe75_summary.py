"""Independent artifact reconstruction and tamper guards; no MLX imports."""

from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/summarize_probe75.py"
spec = importlib.util.spec_from_file_location("probe75_summary", SCRIPT)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def _episode(seed, *, survived=1, ticks=10, calibration=False):
    return {
        "episode_seed": seed, "survived": survived,
        "steps": 40 if calibration else 400,
        "death_need": None if survived else "water",
        "scored_ticks": ticks, "readings": 1 if calibration else 2,
        "evidence_ticks": 39 if calibration else 399,
        "scores": {
            cell: {"regret": regret, "rate_mae": 0.002}
            for cell, regret in zip(summary.CELLS, (0.2, 0.1, 0.0, 0.3))
        },
        "birth_rate_mae": dict(zip(summary.CELLS, (0.003, 0.001, 0.0, 0.004))),
        "word_differences": {"reset": 2, "swapped": 3},
    }


@pytest.fixture
def artifact():
    configuration = {
        "schema": 2, "seeds": 5, "identities": 28, "test_episodes": 5,
        "seed_stride": summary.BLOCK_STRIDE, "episode_stride": summary.EPISODE_STRIDE,
        "identity_seed_base": summary.IDENTITY_BASE, "episode_seed_base": summary.EPISODE_BASE,
        "parent_fingerprint": "a" * 64, "mechanism_fingerprint": "b" * 64,
        "world_kwargs": dict(summary.WORLD), "cells": list(summary.CELLS),
        "organism": {"report": {"life_steps": 400}},
    }
    calibrations, units = [], []
    for block in range(5):
        for identity in range(28):
            base = summary.EPISODE_BASE + block * summary.BLOCK_STRIDE + identity * summary.EPISODE_STRIDE
            calibrations.append({"seed_index": block, "identity": identity,
                                 **_episode(base, calibration=True)})
            for cell, survivors in zip(summary.CELLS, (4, 5, 6, 3)):
                units.append({
                    "seed_index": block, "identity": identity, "cell": cell,
                    "episodes": [_episode(base + episode, survived=int(identity < survivors),
                                          ticks=10 if episode % 2 else 100)
                                 for episode in range(1, 6)],
                    "final_evidence_ticks": 39 + 5 * 399,
                    "final_readings": 1 + 5 * 2,
                })
    result = {"configuration": configuration, "complete": True,
              "completed_units": 700, "total_units": 700,
              "calibration": calibrations, "units": units}
    independent = summary.reconstruct(result)
    result["cells"] = independent["cells"]
    result["gates"] = independent["gates"]
    return result


def test_exact_original_t4_interval():
    result = summary.paired_interval([0.01, 0.02, 0.03, 0.04, 0.05])
    half = 2.776 * math.sqrt(0.00025) / math.sqrt(5)
    assert result["mean"] == pytest.approx(0.03)
    assert result["low"] == pytest.approx(0.03 - half)
    assert result["high"] == pytest.approx(0.03 + half)
    with pytest.raises(ValueError, match="five"):
        summary.paired_interval([0.1] * 4)


def test_raw_counts_exclude_calibration_and_regret_is_tick_weighted(artifact):
    result = summary.reconstruct(artifact)
    assert not result["mismatches"]
    assert result["gates"]["all_pass"]
    retained = next(cell for cell in result["cells"] if cell["cell"] == "persistent_rate")
    assert retained["survival"] == pytest.approx(125 / 700)
    assert retained["matched_regret_value"] == pytest.approx(0.5 / 230)
    # Averaging five episode ratios would incorrectly produce 0.0064.
    assert retained["matched_regret_value"] != pytest.approx(0.0064)
    assert retained["birth_rate_mae_value"] == pytest.approx(0.002)
    assert retained["birth_identity_mae_value"] == pytest.approx(0.003)
    for block in result["blocks"]:
        assert block["scored_ticks"] == 28 * 230
        assert block["means"]["matched_regret_value"] == pytest.approx(
            (block["regret_sums"]["reset_rate"] - block["regret_sums"]["persistent_rate"])
            / block["scored_ticks"]
        )


@pytest.mark.parametrize("tamper, message", [
    (lambda a: a.update(complete=False), "incomplete"),
    (lambda a: a["configuration"].update(episode_seed_base=3_500_000_000), "episode_seed_base"),
    (lambda a: a["units"][0]["episodes"][0].update(episode_seed=0), "wrong episode seed"),
    (lambda a: a["calibration"][0].update(episode_seed=0), "wrong episode seed"),
    (lambda a: a["units"].__setitem__(1, deepcopy(a["units"][0])), "duplicate"),
    (lambda a: a["calibration"].__setitem__(1, deepcopy(a["calibration"][0])), "duplicate"),
    (lambda a: a["units"][0]["episodes"].pop(), "five complete episodes"),
    (lambda a: a["units"][0].update(final_readings=500), "calibration plus test evidence"),
    (lambda a: a["units"][0]["episodes"][0]["scores"]["persistent_rate"].update(regret=float("nan")), "finite"),
])
def test_invalid_raw_evidence_cannot_be_scored(artifact, tamper, message):
    tamper(artifact)
    with pytest.raises(ValueError, match=message):
        summary.reconstruct(artifact)


def test_stored_summary_and_gate_tampering_is_reported(artifact):
    artifact["cells"][0]["survival"] += 0.01
    artifact["gates"]["C2_retained_evidence_reaches_survival"]["low"] -= 0.02
    result = summary.reconstruct(artifact)
    assert result["stored_summary_matches"] is False
    assert result["stored_gates_match"] is False
    assert any("cells.reset_rate.survival" in issue for issue in result["mismatches"])
    assert any("gates.C2_retained_evidence_reaches_survival.low" in issue for issue in result["mismatches"])


def test_old_survey_is_only_accepted_as_explicit_diagnostic(artifact):
    artifact["configuration"].update(identity_seed_base=3_200_000_000,
                                     episode_seed_base=3_300_000_000, schema=1)
    artifact["configuration"].pop("mechanism_fingerprint")
    for row in artifact["calibration"]:
        row["episode_seed"] -= 400_000_000
    for unit in artifact["units"]:
        for row in unit["episodes"]:
            row["episode_seed"] -= 400_000_000
    with pytest.raises(ValueError, match="identity_seed_base"):
        summary.reconstruct(artifact)
    result = summary.reconstruct(artifact, diagnostic_probe74=True)
    assert result["status"] == "diagnostic-only-probe74"
    assert not result["mismatches"]


def test_compact_underlying_sums_reproduce_raw_results(artifact):
    raw = summary.reconstruct(artifact)
    compact = summary.reconstruct_compact(raw)
    assert compact["cells"] == raw["cells"]
    assert compact["gates"] == raw["gates"]
    assert not compact["mismatches"]
    raw["blocks"][0]["means"]["survival"] += 0.01
    tampered = summary.reconstruct_compact(raw)
    assert not tampered["stored_summary_matches"]
    assert any("blocks[0,reset_rate].means.survival" in issue for issue in tampered["mismatches"])


@pytest.mark.parametrize("tamper, message", [
    (lambda a: a["blocks"].__setitem__(1, deepcopy(a["blocks"][0])), "duplicate"),
    (lambda a: a["blocks"][0].update(survivors=0), "inconsistent.*survival counts"),
    (lambda a: a["blocks"][0].update(matched_regret_sum=0.0), "inconsistent compact matched_regret_sum"),
])
def test_compact_tampering_fails(artifact, tamper, message):
    compact = summary.reconstruct(artifact)
    tamper(compact)
    with pytest.raises(ValueError, match=message):
        summary.reconstruct_compact(compact)


def test_default_cli_falls_back_to_committed_compact_evidence(artifact, tmp_path, monkeypatch, capsys):
    compact_path = tmp_path / "evidence.json"
    compact_path.write_text(json.dumps(summary.reconstruct(artifact)))
    monkeypatch.setattr(summary, "DEFAULT_INPUT", tmp_path / "absent-raw.json")
    monkeypatch.setattr(summary, "COMPACT_INPUT", compact_path)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT)])
    assert summary.main() == 0
    printed = capsys.readouterr().out
    assert "compact block sums" in printed
    assert "all_pass = True" in printed
