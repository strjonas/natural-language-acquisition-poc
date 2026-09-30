#!/usr/bin/env python3
"""Independently reconstruct Probe75 from raw units, using only the standard library.

The committed compact evidence output contains per-block integer counts and
underlying sums sufficient to reproduce the means and five-block intervals.
Probe74 may be checked only with --diagnostic-probe74; it is never confirmation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import mean, stdev
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "runs/organism/probe75_cross_life_replication/replication.json"
COMPACT_INPUT = DEFAULT_INPUT.with_name("evidence.json")
CELLS = ("reset_rate", "persistent_rate", "true_rate", "swapped_rate")
BLOCKS, IDENTITIES, EPISODES = 5, 28, 5
BLOCK_STRIDE, EPISODE_STRIDE = 2_000_000, 16
IDENTITY_BASE, EPISODE_BASE = 3_600_000_000, 3_700_000_000
T_CRITICAL_95_DF4 = 2.776
WORLD = {
    "metabolic_spread": 0.60, "uptake_spread": 0.00,
    "interoception_probability": 0.03, "help_delay": 24,
    "help_period": 6, "caregiver_store": 0.0,
}


def integer(value, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def paired_interval(values: list[float]) -> dict:
    if len(values) != BLOCKS:
        raise ValueError("intervals require exactly five paired blocks")
    center = mean(values)
    width = T_CRITICAL_95_DF4 * stdev(values) / math.sqrt(BLOCKS)
    return {
        "mean": center, "low": center - width, "high": center + width,
        "n": BLOCKS, "values": values,
        "positive": sum(value > 0 for value in values),
        "negative": sum(value < 0 for value in values),
        "same_sign": sum(value * center > 0 for value in values),
    }


def _check_record(row: dict, expected_seed: int, label: str) -> None:
    if integer(row.get("episode_seed"), f"{label}.episode_seed") != expected_seed:
        raise ValueError(f"{label} has the wrong episode seed")
    if integer(row.get("survived"), f"{label}.survived") not in (0, 1):
        raise ValueError(f"{label}.survived must be zero or one")
    for key in ("steps", "scored_ticks", "readings", "evidence_ticks"):
        integer(row.get(key), f"{label}.{key}")
    if row["steps"] > 400 or row["evidence_ticks"] > row["steps"]:
        raise ValueError(f"{label} exceeds its physical evidence budget")
    if row["readings"] > row["evidence_ticks"] or row["scored_ticks"] > row["steps"]:
        raise ValueError(f"{label} has impossible readings or scored decisions")
    if set(row.get("scores", {})) != set(CELLS) or set(row.get("birth_rate_mae", {})) != set(CELLS):
        raise ValueError(f"{label} must score all four rate readouts")
    for cell in CELLS:
        for key in ("regret", "rate_mae"):
            value = number(row["scores"][cell].get(key), f"{label}.{cell}.{key}")
            if value < -1e-12:
                raise ValueError(f"{label} has negative {key}")
        if number(row["birth_rate_mae"][cell], f"{label}.{cell}.birth_mae") < 0:
            raise ValueError(f"{label} has negative birth error")
    if set(row.get("word_differences", {})) != {"reset", "swapped"}:
        raise ValueError(f"{label} has incomplete word differences")
    for key, value in row["word_differences"].items():
        if integer(value, f"{label}.word_differences.{key}") > row["scored_ticks"]:
            raise ValueError(f"{label} has impossible word differences")


def _validate_configuration(config: dict, *, diagnostic_probe74: bool = False) -> dict:
    expected = {
        "seeds": BLOCKS, "identities": IDENTITIES, "test_episodes": EPISODES,
        "seed_stride": BLOCK_STRIDE, "episode_stride": EPISODE_STRIDE,
        "identity_seed_base": 3_200_000_000 if diagnostic_probe74 else IDENTITY_BASE,
        "episode_seed_base": 3_300_000_000 if diagnostic_probe74 else EPISODE_BASE,
    }
    for key, value in expected.items():
        if type(config.get(key)) is not int or config[key] != value:
            raise ValueError(f"configuration.{key} must be {value}")
    if config.get("cells") != list(CELLS) or config.get("world_kwargs") != WORLD:
        raise ValueError("configuration does not match the four cells and locked ecology")
    if config.get("organism", {}).get("report", {}).get("life_steps") != 400:
        raise ValueError("the locked episode duration must be 400 ticks")
    for key in ("parent_fingerprint",) if diagnostic_probe74 else ("parent_fingerprint", "mechanism_fingerprint"):
        fingerprint = config.get(key)
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise ValueError(f"missing or invalid {key}")
        try:
            int(fingerprint, 16)
        except ValueError as exc:
            raise ValueError(f"invalid {key}") from exc
    if not diagnostic_probe74 and config.get("schema") != 2:
        raise ValueError("confirmation requires source-pinned schema 2")
    return expected


def validate(artifact: dict, *, diagnostic_probe74: bool = False) -> tuple[dict, dict]:
    expected = _validate_configuration(artifact.get("configuration", {}),
                                       diagnostic_probe74=diagnostic_probe74)
    if artifact.get("complete") is not True:
        raise ValueError("artifact is incomplete; no confirmation can be scored")
    expected_total = BLOCKS * IDENTITIES * (1 + len(CELLS))
    if artifact.get("completed_units") != expected_total or artifact.get("total_units") != expected_total:
        raise ValueError("artifact has an incomplete completed-unit count")
    calibration = artifact.get("calibration", [])
    units = artifact.get("units", [])
    if len(calibration) != BLOCKS * IDENTITIES or len(units) != BLOCKS * IDENTITIES * len(CELLS):
        raise ValueError("expected 140 calibrations and 560 complete identity/cell units")

    def key(row, label):
        block = integer(row.get("seed_index"), f"{label}.seed_index")
        identity = integer(row.get("identity"), f"{label}.identity")
        if block >= BLOCKS or identity >= IDENTITIES:
            raise ValueError(f"{label} has an unexpected block or identity")
        return block, identity

    warm = {}
    for row in calibration:
        block, identity = key(row, "calibration")
        if (block, identity) in warm:
            raise ValueError("duplicate calibration identity")
        _check_record(row, expected["episode_seed_base"] + block * BLOCK_STRIDE + identity * EPISODE_STRIDE,
                      f"calibration[{block},{identity}]")
        warm[block, identity] = row
    indexed = {}
    for unit in units:
        block, identity = key(unit, "unit")
        cell = unit.get("cell")
        if cell not in CELLS or (block, identity, cell) in indexed:
            raise ValueError("duplicate or unexpected identity/cell unit")
        episodes = unit.get("episodes", [])
        if len(episodes) != EPISODES:
            raise ValueError("every identity/cell unit needs five complete episodes")
        for episode, row in enumerate(episodes, 1):
            _check_record(row, expected["episode_seed_base"] + block * BLOCK_STRIDE + identity * EPISODE_STRIDE + episode,
                          f"unit[{block},{identity},{cell}].episode[{episode}]")
        for final, increment in (("final_evidence_ticks", "evidence_ticks"), ("final_readings", "readings")):
            expected_final = warm[block, identity][increment] + sum(row[increment] for row in episodes)
            if integer(unit.get(final), final) != expected_final:
                raise ValueError(f"{final} does not match calibration plus test evidence")
        indexed[block, identity, cell] = unit
    return warm, indexed


def _block_values(sums: dict) -> dict:
    count, ticks = IDENTITIES * EPISODES, sums["scored_ticks"]
    values = {
        "survival": sums["survivors"] / count,
        "mean_life_steps": sums["life_steps_sum"] / count,
        "matched_regret_value": sums["matched_regret_sum"] / ticks,
        "birth_rate_mae_value": sums["birth_rate_mae_difference_sum"] / count,
        "birth_identity_mae_value": sums["birth_identity_mae_difference_sum"] / count,
        "word_difference_share": sums["word_differences"]["reset"] / ticks,
        "mean_readings": sums["readings_sum"] / count,
        "mean_final_evidence_ticks": sums["final_evidence_ticks_sum"] / IDENTITIES,
        "mean_final_readings": sums["final_readings_sum"] / IDENTITIES,
    }
    for name in CELLS:
        values[f"regret_{name}"] = sums["regret_sums"][name] / ticks
        values[f"rate_mae_{name}"] = sums["rate_mae_sums"][name] / ticks
        values[f"birth_mae_{name}"] = sums["birth_mae_sums"][name] / count
    return values


def _cells_from_blocks(evidence_blocks: list[dict]) -> list[dict]:
    cells = []
    for cell in CELLS:
        rows = sorted((row for row in evidence_blocks if row["cell"] == cell),
                      key=lambda row: row["seed_index"])
        values = [_block_values(row) for row in rows]
        deaths = Counter()
        for row in rows:
            deaths.update(row["deaths_by_need"])
        cells.append({
            "cell": cell, "test_episodes": BLOCKS * IDENTITIES * EPISODES,
            **{key: mean(block[key] for block in values) for key in values[0]},
            **{f"per_seed_{key}": [block[key] for block in values] for key in values[0]},
            "per_episode_survival": [mean(row["episode_survivors"][i] / IDENTITIES for row in rows) for i in range(EPISODES)],
            "deaths_by_need": dict(deaths),
        })
    return cells


def _gates(cells: list[dict]) -> dict:
    index = {row["cell"]: row for row in cells}
    persistent = index["persistent_rate"]
    survival = lambda name: index[name]["per_seed_survival"]
    delta = lambda a, b: [left - right for left, right in zip(survival(a), survival(b))]
    contrasts = {
        "C1_persistent_identity_physical_ceiling": delta("true_rate", "reset_rate"),
        "C2_retained_evidence_reaches_survival": delta("persistent_rate", "reset_rate"),
        "C3_matched_regret_bridge": persistent["per_seed_matched_regret_value"],
        "C4_less_rediscovery_at_birth": persistent["per_seed_birth_rate_mae_value"],
        "C5_memory_belongs_to_the_individual": persistent["per_seed_birth_identity_mae_value"],
    }
    gates = {}
    for name, values in contrasts.items():
        result = paired_interval(values)
        passed = result["mean"] > 0 and result["low"] > 0 and result["positive"] >= 4
        gates[name] = {**result, "verdict": "pass" if passed else "fail"}
    truth = index["true_rate"]["survival"]
    gates["C6_viable_construction"] = {
        "true_rate_survival": truth, "floor": 0.15,
        "verdict": "pass" if truth >= 0.15 else "fail",
    }
    gates["all_pass"] = all(row["verdict"] == "pass" for row in gates.values())
    gates["ungated_persistent_minus_swapped_survival"] = paired_interval(delta("persistent_rate", "swapped_rate"))
    return gates


def _compare(expected, actual, path: str, mismatches: list[str]) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            mismatches.append(f"{path}: expected an object")
            return
        for key in expected:
            _compare(expected[key], actual.get(key), f"{path}.{key}", mismatches)
        for key in actual.keys() - expected.keys():
            mismatches.append(f"{path}.{key}: unexpected stored field")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            mismatches.append(f"{path}: wrong list length or missing list")
            return
        for i, (left, right) in enumerate(zip(expected, actual)):
            _compare(left, right, f"{path}[{i}]", mismatches)
    elif isinstance(expected, float):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isclose(expected, actual, rel_tol=1e-10, abs_tol=1e-12):
            mismatches.append(f"{path}: reconstructed {expected!r}, stored {actual!r}")
    elif type(expected) is not type(actual) or expected != actual:
        mismatches.append(f"{path}: reconstructed {expected!r}, stored {actual!r}")


def reconstruct(artifact: dict, *, diagnostic_probe74: bool = False) -> dict:
    warm, units = validate(artifact, diagnostic_probe74=diagnostic_probe74)
    evidence_blocks, cells = [], []
    for cell in CELLS:
        blocks, by_episode, deaths = [], [], Counter()
        for block in range(BLOCKS):
            group = [units[block, identity, cell] for identity in range(IDENTITIES)]
            rows = [row for unit in group for row in unit["episodes"]]
            count = IDENTITIES * EPISODES
            ticks = sum(row["scored_ticks"] for row in rows)
            if ticks <= 0:
                raise ValueError(f"{cell} block {block} has no scored decisions")
            sums = {
                "survivors": sum(row["survived"] for row in rows),
                "life_steps_sum": sum(row["steps"] for row in rows),
                "readings_sum": sum(row["readings"] for row in rows),
                "evidence_ticks_sum": sum(row["evidence_ticks"] for row in rows),
                "final_evidence_ticks_sum": sum(unit["final_evidence_ticks"] for unit in group),
                "final_readings_sum": sum(unit["final_readings"] for unit in group),
                "scored_ticks": ticks,
                "regret_sums": {name: sum(row["scores"][name]["regret"] for row in rows) for name in CELLS},
                "rate_mae_sums": {name: sum(row["scores"][name]["rate_mae"] for row in rows) for name in CELLS},
                "birth_mae_sums": {name: sum(row["birth_rate_mae"][name] for row in rows) for name in CELLS},
                "word_differences": {name: sum(row["word_differences"][name] for row in rows) for name in ("reset", "swapped")},
                "episode_survivors": [sum(unit["episodes"][i]["survived"] for unit in group) for i in range(EPISODES)],
                "matched_regret_sum": sum(row["scores"]["reset_rate"]["regret"] - row["scores"]["persistent_rate"]["regret"] for row in rows),
                "birth_rate_mae_difference_sum": sum(row["birth_rate_mae"]["reset_rate"] - row["birth_rate_mae"]["persistent_rate"] for row in rows),
                "birth_identity_mae_difference_sum": sum(row["birth_rate_mae"]["swapped_rate"] - row["birth_rate_mae"]["persistent_rate"] for row in rows),
                "deaths_by_need": dict(Counter(row["death_need"] for row in rows if row["death_need"] is not None)),
            }
            block_values = _block_values(sums)
            blocks.append(block_values)
            by_episode.append([value / IDENTITIES for value in sums["episode_survivors"]])
            for row in rows:
                if row["death_need"] is not None:
                    deaths[row["death_need"]] += 1
            evidence_blocks.append({"cell": cell, "seed_index": block,
                                    "identities": IDENTITIES, "test_episodes": count,
                                    **sums, "means": block_values})
        cells.append({
            "cell": cell, "test_episodes": BLOCKS * IDENTITIES * EPISODES,
            **{key: mean(values[key] for values in blocks) for key in blocks[0]},
            **{f"per_seed_{key}": [values[key] for values in blocks] for key in blocks[0]},
            "per_episode_survival": [mean(values[i] for values in by_episode) for i in range(EPISODES)],
            "deaths_by_need": dict(deaths),
        })
    return _finish(artifact, cells, evidence_blocks, [{
        "seed_index": block, "identities": IDENTITIES,
        **{f"{key}_sum": sum(warm[block, identity][key] for identity in range(IDENTITIES))
           for key in ("steps", "readings", "evidence_ticks")},
    } for block in range(BLOCKS)], diagnostic_probe74=diagnostic_probe74)


def _finish(artifact: dict, cells: list[dict], evidence_blocks: list[dict],
            calibration: list[dict], *, diagnostic_probe74: bool = False) -> dict:
    gates = _gates(cells)
    mismatches = []
    stored_cells = artifact.get("cells")
    if not isinstance(stored_cells, list) or len(stored_cells) != len(CELLS) or {row.get("cell") for row in stored_cells} != set(CELLS):
        mismatches.append("cells: stored summary lacks the four distinct cells")
    else:
        stored_index = {row["cell"]: row for row in stored_cells}
        for row in cells:
            _compare(row, stored_index[row["cell"]], f"cells.{row['cell']}", mismatches)
    _compare(gates, artifact.get("gates"), "gates", mismatches)
    index = {row["cell"]: row for row in cells}
    true_minus_retained = paired_interval([
        truth - retained for truth, retained in zip(index["true_rate"]["per_seed_survival"], index["persistent_rate"]["per_seed_survival"])
    ])
    return {
        "schema": 2, "complete": True, "calibration_episodes": 140,
        "identity_cell_units": 560, "test_episodes_per_cell": 700,
        "status": "diagnostic-only-probe74" if diagnostic_probe74 else "probe75-confirmation",
        "configuration": artifact["configuration"],
        "calibration_blocks": calibration, "blocks": evidence_blocks,
        "cells": cells, "gates": gates,
        "ungated_true_minus_retained_survival": true_minus_retained,
        "stored_summary_matches": not any(value.startswith("cells") for value in mismatches),
        "stored_gates_match": not any(value.startswith("gates") for value in mismatches),
        "mismatches": mismatches,
    }


def reconstruct_compact(artifact: dict, *, diagnostic_probe74: bool = False) -> dict:
    """Recalculate from committed block sums, without using their stored means.

    Individual episode seeds were checked when these sums were produced from
    the raw artifact. This reduced record permits aggregate recalculation, not
    a fresh inspection of the omitted individual episode records.
    """
    _validate_configuration(artifact.get("configuration", {}),
                            diagnostic_probe74=diagnostic_probe74)
    status = "diagnostic-only-probe74" if diagnostic_probe74 else "probe75-confirmation"
    required = {"schema": 2, "complete": True, "status": status,
                "calibration_episodes": 140, "identity_cell_units": 560,
                "test_episodes_per_cell": 700}
    if any(type(artifact.get(key)) is not type(value) or artifact[key] != value
           for key, value in required.items()):
        raise ValueError("compact evidence lacks the complete expected experiment metadata")
    calibration = artifact.get("calibration_blocks", [])
    blocks = artifact.get("blocks", [])
    if len(calibration) != BLOCKS or len(blocks) != BLOCKS * len(CELLS):
        raise ValueError("compact evidence needs five calibration blocks and twenty cell blocks")
    warm = {}
    for row in calibration:
        block = integer(row.get("seed_index"), "calibration.seed_index")
        if block >= BLOCKS or block in warm or row.get("identities") != IDENTITIES:
            raise ValueError("duplicate or incomplete compact calibration block")
        for key in ("steps_sum", "readings_sum", "evidence_ticks_sum"):
            integer(row.get(key), key)
        if not row["readings_sum"] <= row["evidence_ticks_sum"] <= row["steps_sum"] <= IDENTITIES * 400:
            raise ValueError("impossible compact calibration evidence budget")
        warm[block] = row
    seen, mean_mismatches = set(), []
    for row in blocks:
        block = integer(row.get("seed_index"), "block.seed_index")
        cell = row.get("cell")
        if block >= BLOCKS or cell not in CELLS or (block, cell) in seen:
            raise ValueError("duplicate or unexpected compact cell block")
        seen.add((block, cell))
        if row.get("identities") != IDENTITIES or row.get("test_episodes") != IDENTITIES * EPISODES:
            raise ValueError("incomplete compact identity/episode count")
        for key in ("survivors", "life_steps_sum", "readings_sum", "evidence_ticks_sum",
                    "final_evidence_ticks_sum", "final_readings_sum", "scored_ticks"):
            integer(row.get(key), key)
        if row["survivors"] > IDENTITIES * EPISODES or row["scored_ticks"] == 0:
            raise ValueError("impossible compact survival count or no scored decisions")
        if not row["readings_sum"] <= row["evidence_ticks_sum"] <= row["life_steps_sum"] <= IDENTITIES * EPISODES * 400:
            raise ValueError("impossible compact test evidence budget")
        if row["scored_ticks"] > row["life_steps_sum"]:
            raise ValueError("impossible compact scored decision count")
        if row["final_evidence_ticks_sum"] != warm[block]["evidence_ticks_sum"] + row["evidence_ticks_sum"] or row["final_readings_sum"] != warm[block]["readings_sum"] + row["readings_sum"]:
            raise ValueError("compact cumulative evidence differs from calibration plus tests")
        episode_counts = row.get("episode_survivors", [])
        if len(episode_counts) != EPISODES or sum(episode_counts) != row["survivors"]:
            raise ValueError("incomplete or inconsistent compact episode survival counts")
        for value in episode_counts:
            if integer(value, "episode survivor count") > IDENTITIES:
                raise ValueError("impossible compact episode survivor count")
        for field in ("regret_sums", "rate_mae_sums", "birth_mae_sums"):
            if set(row.get(field, {})) != set(CELLS):
                raise ValueError(f"incomplete compact {field}")
            for value in row[field].values():
                if number(value, field) < -1e-12:
                    raise ValueError(f"negative compact {field}")
        for paired, field, left, right in (
            ("matched_regret_sum", "regret_sums", "reset_rate", "persistent_rate"),
            ("birth_rate_mae_difference_sum", "birth_mae_sums", "reset_rate", "persistent_rate"),
            ("birth_identity_mae_difference_sum", "birth_mae_sums", "swapped_rate", "persistent_rate"),
        ):
            value = number(row.get(paired), paired)
            if not math.isclose(value, row[field][left] - row[field][right], rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError(f"inconsistent compact {paired}")
        if set(row.get("word_differences", {})) != {"reset", "swapped"}:
            raise ValueError("incomplete compact word differences")
        for value in row["word_differences"].values():
            if integer(value, "word differences") > row["scored_ticks"]:
                raise ValueError("impossible compact word differences")
        for value in row.get("deaths_by_need", {}).values():
            integer(value, "death count")
        if sum(row.get("deaths_by_need", {}).values()) != IDENTITIES * EPISODES - row["survivors"]:
            raise ValueError("compact death counts disagree with survival")
        _compare(_block_values(row), row.get("means"), f"blocks[{block},{cell}].means", mean_mismatches)
    result = _finish(artifact, _cells_from_blocks(blocks), blocks, calibration,
                     diagnostic_probe74=diagnostic_probe74)
    _compare(result["ungated_true_minus_retained_survival"],
             artifact.get("ungated_true_minus_retained_survival"),
             "gates.ungated_true_minus_retained_survival", result["mismatches"])
    result["stored_gates_match"] = not any(value.startswith("gates") for value in result["mismatches"])
    result["mismatches"].extend(mean_mismatches)
    if mean_mismatches:
        result["stored_summary_matches"] = False
    if "source_artifact_sha256" in artifact:
        result["source_artifact_sha256"] = artifact["source_artifact_sha256"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="defaults to full raw artifact, or committed evidence.json when raw is absent")
    parser.add_argument("--out", type=Path, help="write compact reconstructed evidence JSON")
    parser.add_argument("--diagnostic-probe74", action="store_true", help="check the old survey only; never confirmation")
    args = parser.parse_args()
    try:
        path = args.input or (DEFAULT_INPUT if DEFAULT_INPUT.exists() else COMPACT_INPUT)
        source = path.read_bytes()
        artifact = json.loads(source)
        compact = "blocks" in artifact and "units" not in artifact
        rebuild = reconstruct_compact if compact else reconstruct
        result = rebuild(artifact, diagnostic_probe74=args.diagnostic_probe74)
    except (ValueError, KeyError, TypeError, AttributeError, OSError) as exc:
        print(f"Cannot score artifact: {exc}", file=sys.stderr)
        return 2
    result["input_evidence_sha256" if compact else "source_artifact_sha256"] = sha256(source).hexdigest()
    title = "Probe74 diagnostic reconstruction ONLY (not confirmation)" if args.diagnostic_probe74 else "Probe75 independent confirmation reconstruction"
    print(f"{title}: 5 blocks x 28 identities x 5 test episodes per cell")
    print("Input: compact block sums; individual seed checks were performed at extraction" if compact
          else "Input: full raw units; every individual episode seed validated")
    print("Calibration: 140 episodes, excluded from every survival gate")
    for cell in CELLS:
        survivors = sum(block["survivors"] for block in result["blocks"] if block["cell"] == cell)
        row = next(row for row in result["cells"] if row["cell"] == cell)
        print(f"{cell:<16} {survivors}/700 survivors ({row['survival']:.6f}); blocks {row['per_seed_survival']}")
    print("Locked gates (paired Student-t, df=4, critical=2.776):")
    for name, gate in result["gates"].items():
        if name == "all_pass" or name.startswith("ungated"):
            continue
        if "mean" in gate:
            print(f"  {gate['verdict'].upper():4} {name}: {gate['mean']:+.8f} [{gate['low']:+.8f}, {gate['high']:+.8f}], {gate['positive']}/5 positive")
        else:
            print(f"  {gate['verdict'].upper():4} {name}: survival {gate['true_rate_survival']:.6f}, floor {gate['floor']}")
    print(f"all_pass = {result['gates']['all_pass']}")
    for name, contrast in (
        ("retained minus swapped survival (ungated)", result["gates"]["ungated_persistent_minus_swapped_survival"]),
        ("true minus retained survival (ungated)", result["ungated_true_minus_retained_survival"]),
    ):
        print(f"{name}: {contrast['mean']:+.8f} [{contrast['low']:+.8f}, {contrast['high']:+.8f}]")
    print(f"Stored summary matches: {result['stored_summary_matches']}; stored gates match: {result['stored_gates_match']}")
    for mismatch in result["mismatches"]:
        print(f"MISMATCH: {mismatch}", file=sys.stderr)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return 1 if result["mismatches"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
