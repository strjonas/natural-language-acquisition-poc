"""Probe74: rate memory across episodes of one genuinely persistent body.

See docs/decisions/2026-09-30-cross-life-self-preregistration.md. Current-state
belief always resets. Only sufficient statistics in a separate rate estimator
cross episode boundaries; every cell computes all three estimators.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from random import Random
from time import perf_counter
from typing import Callable

import numpy as np

from homesocial.creole.vocab import PAD_TOKEN, TOKEN_TO_ID
from homesocial.island.report import BODY_NEEDS, NEED_TO_REPORT_WORD, REPORT_NEEDS
from homesocial.organism.future_request import (
    RequestLedger, answer_horizon, expected_portion, need_scores, true_rates,
)
from homesocial.organism.individual_self import _true_body
from homesocial.organism.learned_rate_survival import (
    _contrast, _learned_rates, _positive_gate, _summarize_values,
    _word_for_rates, _write_progress,
)
from homesocial.organism.portion_request import (
    MIN_BELIEVED_RATE, MoveFraction, SelfModelTier, species_rate,
)
from homesocial.organism.report_audit import FIDELITY_WARMUP, make_report_world
from homesocial.organism.self_belief import _sample_motor_action
from homesocial.organism.self_calibration import RecursiveSelfCalibration
from homesocial.organism.train import execute_agent_action, load_organism_checkpoint


SEEDS = 5
IDENTITIES = 28
TEST_EPISODES = 5
DELAY = 24
IDENTITY_SEED_BASE = 3_200_000_000
EPISODE_SEED_BASE = 3_300_000_000
SEED_STRIDE = 2_000_000
EPISODE_STRIDE = 16
CELLS = ("reset_rate", "persistent_rate", "true_rate", "swapped_rate")
PAD_ID = TOKEN_TO_ID[PAD_TOKEN]
WORLD_KWARGS = {
    "metabolic_spread": 0.60,
    "uptake_spread": 0.00,
    "interoception_probability": 0.03,
    "help_delay": DELAY,
    "help_period": 6,
    "caregiver_store": 0.0,
}


class RateMemory(RecursiveSelfCalibration):
    """Existing RLS, with an explicit boundary between identity and episode.

    Snapshots contain parameters and their evidence only. They contain no
    Jacobian, current body, true constants, packet, ledger or motor state.
    """

    def __init__(self, organism, report):
        super().__init__(organism, report)
        self.evidence_ticks = 0

    def start_episode(self, packet, world, prior: dict | None = None) -> None:
        super().reset(packet, world)
        self.evidence_ticks = 0
        if prior is None:
            return
        for name in ("a", "covariance", "burn_species", "burn_corrected"):
            current = getattr(self, "_" + name)
            value = np.asarray(prior[name], dtype=np.float64)
            if value.shape != current.shape or not np.isfinite(value).all():
                raise ValueError(f"invalid rate-memory {name}")
            setattr(self, "_" + name, value.copy())
        if np.any(self._a <= -1.0):
            raise ValueError("invalid rate-memory amplitude")
        self._m = np.log1p(self._a)
        self.readings = int(prior["readings"])
        self.skipped = int(prior["skipped"])
        self.evidence_ticks = int(prior["evidence_ticks"])
        if min(self.readings, self.skipped, self.evidence_ticks) < 0:
            raise ValueError("negative rate-memory evidence count")

    def snapshot(self) -> dict[str, object]:
        return {
            "a": self._a.tolist(),
            "covariance": self._covariance.tolist(),
            "burn_species": self._burn_species.tolist(),
            "burn_corrected": self._burn_corrected.tolist(),
            "readings": self.readings,
            "skipped": self.skipped,
            "evidence_ticks": self.evidence_ticks,
        }

    def update(self, before, action_index, after, world, reading) -> None:
        super().update(before, action_index, after, world, reading)
        self.evidence_ticks += int(after.step_count - before.step_count)


def identity_constants(identity_seed: int, report) -> dict[str, float]:
    """Environment-only draw, matching ReportWorld's four-axis distribution."""

    rng = Random(identity_seed + 9_900_023)
    return {
        need: rng.uniform(1.0 - report.metabolic_spread, 1.0 + report.metabolic_spread)
        for need in BODY_NEEDS
    }


def reset_identity_world(organism, *, identity_seed: int, episode_seed: int):
    """Fresh episode randomness, unchanged physical constants for this identity."""

    report = replace(organism.report, **WORLD_KWARGS)
    world = make_report_world(organism, seed=episode_seed, **WORLD_KWARGS)
    packet = world.reset(episode_seed)
    # Constants belong to the simulator. Neither the state tier nor memory
    # reset reads this property; only the true-rate arm and scoring can do so.
    world.grid.metabolic_scale = identity_constants(identity_seed, report)
    return world, packet, report


def memory_rates(memory: RateMemory, report, fraction: float) -> np.ndarray:
    scale = memory.recovered_scale()
    return np.asarray([
        max(MIN_BELIEVED_RATE, species_rate(report, need, fraction) * scale[need])
        for need in REPORT_NEEDS
    ], dtype=np.float64)


def run_episode(model, organism, *, cell: str, identity_seed: int,
                episode_seed: int, own_prior: dict | None = None,
                donor_prior: dict | None = None) -> dict[str, object]:
    """One episode; rate interventions share the episodic state machinery."""

    if cell not in CELLS:
        raise ValueError(f"unknown cell: {cell}")
    world, packet, report = reset_identity_world(
        organism, identity_seed=identity_seed, episode_seed=episode_seed,
    )
    tier = SelfModelTier("recursive", organism, report)
    tier.reset(packet, world)
    memory = RateMemory(organism, report)
    memory.start_episode(packet, world, own_prior)
    donor = RateMemory(organism, report)
    donor.start_episode(packet, world, donor_prior)
    ledger = RequestLedger(help_period=report.help_period, delay=DELAY,
                           expected_portion=expected_portion(report))
    ledger.reset()
    moves = MoveFraction(organism, report)
    moves.reset(packet)
    hidden = None
    rng = Random(episode_seed + 59_000_003)
    initial_readings = memory.readings
    initial_ticks = memory.evidence_ticks
    scores = {name: {"regret": 0.0, "rate_mae": 0.0} for name in CELLS}
    differences = {"reset": 0, "swapped": 0}
    birth_mae = {}
    scored_ticks = steps = 0

    while True:
        fraction = moves.fraction
        now = int(packet.step_count)
        rates = {
            "reset_rate": _learned_rates(tier, fraction),
            "persistent_rate": memory_rates(memory, report, fraction),
            "true_rate": true_rates(world, report, fraction),
            "swapped_rate": memory_rates(donor, report, fraction),
        }
        words = {name: _word_for_rates(tier, vector, ledger, now=now,
                                     delay=DELAY, report=report)
                 for name, vector in rates.items()}
        if now == 0:
            birth_mae = {
                name: float(np.abs(vector - rates["true_rate"]).mean())
                for name, vector in rates.items()
            }
        if now >= FIDELITY_WARMUP:
            scored_ticks += 1
            true_scores = need_scores(
                believed_levels=_true_body(world),
                believed_rates=rates["true_rate"],
                believed_uptake=world.uptake_scale,
                arriving=ledger.arriving(now, world.uptake_scale, delay=DELAY),
                horizon=answer_horizon(DELAY), report=report,
            )
            best = float(true_scores.max())
            for name in CELLS:
                scores[name]["regret"] += best - float(
                    true_scores[REPORT_NEEDS.index(words[name])]
                )
                scores[name]["rate_mae"] += float(
                    np.abs(rates[name] - rates["true_rate"]).mean()
                )
            differences["reset"] += int(words["reset_rate"] != words["persistent_rate"])
            differences["swapped"] += int(words["swapped_rate"] != words["persistent_rate"])

        # In the swapped cell, donor memory is its retained self prior. The
        # own-memory estimator remains a same-history counterfactual there.
        spoken = words[cell]
        ledger.record(now, spoken)
        action, hidden = _sample_motor_action(model, packet, hidden, rng)
        world.hear((TOKEN_TO_ID[NEED_TO_REPORT_WORD[spoken]], PAD_ID))
        before = packet
        packet, _, terminated, truncated, info = execute_agent_action(
            world, packet, action, consume_options=organism.consume_options,
            inspect_options=organism.inspect_options,
        )
        steps += int(info["duration"])
        if terminated or truncated:
            break
        moves.observe(before, action, packet)
        reading = info.get("interoception")
        tier.update(before, action, packet, world, reading)
        memory.update(before, action, packet, world, reading)
        donor.update(before, action, packet, world, reading)

    return {
        "episode_seed": episode_seed,
        "survived": int(not terminated),
        "steps": steps,
        "death_need": str(info.get("death_need")) if terminated else None,
        "scored_ticks": scored_ticks,
        "scores": scores,
        "word_differences": differences,
        "birth_rate_mae": birth_mae,
        "readings": memory.readings - initial_readings,
        "evidence_ticks": memory.evidence_ticks - initial_ticks,
        "own_memory": memory.snapshot(),
        "donor_memory": donor.snapshot(),
    }


def run_identity_cell(model, organism, *, cell: str, identity_seed: int,
                      episode_base: int, episodes: int, own_prior: dict,
                      donor_prior: dict) -> dict[str, object]:
    records = []
    own, donor = own_prior, donor_prior
    for episode in range(1, episodes + 1):
        record = run_episode(
            model, organism, cell=cell, identity_seed=identity_seed,
            episode_seed=episode_base + episode, own_prior=own, donor_prior=donor,
        )
        own = record.pop("own_memory")
        donor = record.pop("donor_memory")
        records.append(record)
    return {"episodes": records,
            "final_evidence_ticks": own["evidence_ticks"],
            "final_readings": own["readings"]}


def summarize(units: list[dict], *, seeds: int, identities: int,
              episodes: int) -> list[dict[str, object]]:
    cells = []
    for cell in CELLS:
        blocks = []
        by_episode = []
        deaths: defaultdict[str, int] = defaultdict(int)
        for seed in range(seeds):
            group = [unit for unit in units if unit["seed_index"] == seed
                     and unit["cell"] == cell]
            if len(group) != identities or len({u["identity"] for u in group}) != identities:
                raise ValueError(f"incomplete or duplicate identities: {cell}, block {seed}")
            rows = [row for unit in group for row in unit["episodes"]]
            if any(len(unit["episodes"]) != episodes for unit in group):
                raise ValueError("incomplete identity episode sequence")
            ticks = sum(row["scored_ticks"] for row in rows)
            if ticks <= 0:
                raise ValueError("a seed block has no scored decisions")
            block = {
                "survival": float(np.mean([row["survived"] for row in rows])),
                "mean_life_steps": float(np.mean([row["steps"] for row in rows])),
                "matched_regret_value": sum(
                    row["scores"]["reset_rate"]["regret"]
                    - row["scores"]["persistent_rate"]["regret"] for row in rows
                ) / ticks,
                "birth_rate_mae_value": float(np.mean([
                    row["birth_rate_mae"]["reset_rate"]
                    - row["birth_rate_mae"]["persistent_rate"] for row in rows
                ])),
                "birth_identity_mae_value": float(np.mean([
                    row["birth_rate_mae"]["swapped_rate"]
                    - row["birth_rate_mae"]["persistent_rate"] for row in rows
                ])),
                "word_difference_share": sum(row["word_differences"]["reset"]
                                             for row in rows) / ticks,
                "mean_readings": float(np.mean([row["readings"] for row in rows])),
                "mean_final_evidence_ticks": float(np.mean([
                    unit["final_evidence_ticks"] for unit in group
                ])),
                "mean_final_readings": float(np.mean([
                    unit["final_readings"] for unit in group
                ])),
            }
            for name in CELLS:
                block[f"regret_{name}"] = sum(row["scores"][name]["regret"]
                                              for row in rows) / ticks
                block[f"rate_mae_{name}"] = sum(row["scores"][name]["rate_mae"]
                                               for row in rows) / ticks
                block[f"birth_mae_{name}"] = float(np.mean([
                    row["birth_rate_mae"][name] for row in rows
                ]))
            blocks.append(block)
            by_episode.append([
                float(np.mean([unit["episodes"][i]["survived"] for unit in group]))
                for i in range(episodes)
            ])
            for row in rows:
                if row["death_need"] is not None:
                    deaths[row["death_need"]] += 1
        cells.append({
            "cell": cell, "test_episodes": seeds * identities * episodes,
            **{key: float(np.mean([b[key] for b in blocks])) for key in blocks[0]},
            **{f"per_seed_{key}": [b[key] for b in blocks] for key in blocks[0]},
            "per_episode_survival": np.mean(by_episode, axis=0).tolist(),
            "deaths_by_need": dict(deaths),
        })
    return cells


def grade(cells: list[dict]) -> dict[str, object]:
    index = {row["cell"]: row for row in cells}
    if len(cells) != len(CELLS) or set(index) != set(CELLS):
        raise ValueError("the four distinct preregistered cells are required")
    for row in cells:
        for key in ("survival", "matched_regret_value", "birth_rate_mae_value",
                    "birth_identity_mae_value"):
            if len(row[f"per_seed_{key}"]) != SEEDS:
                raise ValueError("gates require exactly five paired seed blocks")
    persistent = index["persistent_rate"]
    contrasts = {
        "C1_persistent_identity_physical_ceiling": _contrast(
            index["true_rate"]["per_seed_survival"],
            index["reset_rate"]["per_seed_survival"],
        ),
        "C2_retained_evidence_reaches_survival": _contrast(
            persistent["per_seed_survival"], index["reset_rate"]["per_seed_survival"],
        ),
        "C3_matched_regret_bridge": _summarize_values(
            persistent["per_seed_matched_regret_value"]),
        "C4_less_rediscovery_at_birth": _summarize_values(
            persistent["per_seed_birth_rate_mae_value"]),
        "C5_memory_belongs_to_the_individual": _summarize_values(
            persistent["per_seed_birth_identity_mae_value"]),
    }
    gates = {name: {**value, "verdict": "pass" if _positive_gate(value) else "fail"}
             for name, value in contrasts.items()}
    gates["C6_viable_construction"] = {
        "true_rate_survival": index["true_rate"]["survival"], "floor": 0.15,
        "verdict": "pass" if index["true_rate"]["survival"] >= 0.15 else "fail",
    }
    gates["all_pass"] = all(value["verdict"] == "pass" for value in gates.values())
    gates["ungated_persistent_minus_swapped_survival"] = _contrast(
        persistent["per_seed_survival"], index["swapped_rate"]["per_seed_survival"],
    )
    return gates


def checkpoint_fingerprint(path: Path) -> str:
    digest = sha256()
    for source in (path, path.with_suffix(path.suffix + ".json")):
        digest.update(source.read_bytes())
    return digest.hexdigest()


def mechanism_fingerprint(source_root: Path | None = None) -> str:
    """Pin local source, including the simulator and estimator dependencies.

    Hashing the whole package conservatively rejects even unrelated source
    edits rather than silently mixing episode mechanisms in a resumed run.
    """

    root = Path(source_root) if source_root is not None else Path(__file__).resolve().parents[1]
    sources = sorted(root.rglob("*.py"))
    if not sources:
        raise ValueError("the mechanism source package contains no Python files")
    digest = sha256()
    for source in sources:
        digest.update(source.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def validate_run(*, seeds: int, identities: int, episodes: int,
                 identity_seed_base: int = IDENTITY_SEED_BASE,
                 episode_seed_base: int = EPISODE_SEED_BASE) -> None:
    values = (seeds, identities, episodes, identity_seed_base, episode_seed_base)
    if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
        raise ValueError("sample counts and seed bases must be integers")
    if seeds < 1 or identities < 2 or not 1 <= episodes < EPISODE_STRIDE:
        raise ValueError("need positive blocks, at least two identities, and 1–15 test episodes")
    if min(identity_seed_base, episode_seed_base) < 0:
        raise ValueError("seed bases must be nonnegative unsigned 32-bit values")
    if identities * EPISODE_STRIDE > SEED_STRIDE:
        raise ValueError("identity episode streams overlap the next seed block")
    identity_last = identity_seed_base + (seeds - 1) * SEED_STRIDE + identities - 1
    episode_last = episode_seed_base + (seeds - 1) * SEED_STRIDE + (identities - 1) * EPISODE_STRIDE + episodes
    bands_overlap = identity_seed_base <= episode_last and episode_seed_base <= identity_last
    if (bands_overlap or episode_last + 59_000_003 >= 2**32
            or identity_last + 9_900_023 >= 2**32):
        raise ValueError("seed bands overlap or exceed the 32-bit motor RNG range")


def run_survey(model, organism, *, parent_fingerprint: str, seeds: int = SEEDS,
               identities: int = IDENTITIES, episodes: int = TEST_EPISODES,
               identity_seed_base: int = IDENTITY_SEED_BASE,
               episode_seed_base: int = EPISODE_SEED_BASE,
               out: Path | None = None, episode_runner: Callable = run_episode,
               cell_runner: Callable = run_identity_cell) -> dict[str, object]:
    validate_run(seeds=seeds, identities=identities, episodes=episodes,
                 identity_seed_base=identity_seed_base, episode_seed_base=episode_seed_base)
    if not parent_fingerprint:
        raise ValueError("a parent checkpoint fingerprint is required")
    config = {
        "schema": 2, "parent_fingerprint": parent_fingerprint,
        "mechanism_fingerprint": mechanism_fingerprint(),
        "organism": asdict(organism),
        "seeds": seeds, "identities": identities, "test_episodes": episodes,
        "identity_seed_base": identity_seed_base, "episode_seed_base": episode_seed_base,
        "seed_stride": SEED_STRIDE, "episode_stride": EPISODE_STRIDE,
        "world_kwargs": WORLD_KWARGS, "cells": list(CELLS),
    }
    # Normalize tuples just as JSON serialization does, before resume equality.
    config = json.loads(json.dumps(config))
    calibration, units = [], []
    legacy = False
    if out is not None and out.exists():
        saved = json.loads(out.read_text())
        saved_config = saved.get("configuration")
        legacy_config = {key: value for key, value in config.items()
                         if key != "mechanism_fingerprint"}
        legacy_config["schema"] = 1
        legacy = saved_config == legacy_config
        if saved_config != config and not legacy:
            raise ValueError("resume configuration, mechanism, or parent checkpoint changed")
        if legacy and (not saved.get("complete")
                       or identity_seed_base != IDENTITY_SEED_BASE
                       or episode_seed_base != EPISODE_SEED_BASE):
            raise ValueError("legacy progress lacks a mechanism fingerprint; cannot resume unfinished work")
        calibration, units = saved["calibration"], saved["units"]
    warm = {(row["seed_index"], row["identity"]): row for row in calibration}
    done = {(row["seed_index"], row["identity"], row["cell"]) for row in units}
    valid_warm = {(seed, identity) for seed in range(seeds) for identity in range(identities)}
    valid_done = {(seed, identity, cell) for seed, identity in valid_warm for cell in CELLS}
    if (len(warm) != len(calibration) or len(done) != len(units)
            or not set(warm) <= valid_warm or not done <= valid_done
            or any((seed, identity) not in warm or (seed, (identity + 1) % identities) not in warm
                   for seed, identity, _ in done)):
        raise ValueError("invalid or duplicate completed checkpoint units")
    if legacy:
        if set(warm) != valid_warm or done != valid_done:
            raise ValueError("legacy progress lacks complete calibration and identity/cell units")
        cells = summarize(units, seeds=seeds, identities=identities, episodes=episodes)
        gates = grade(cells) if (seeds, identities, episodes) == (SEEDS, IDENTITIES, TEST_EPISODES) else None
        # Reading an old completed run performs no new episodes and never
        # stamps unverified historical records with the current source hash.
        return {**saved, "cells": cells, "gates": gates}
    total = seeds * identities * (1 + len(CELLS))

    def save(complete=False, **extra):
        payload = {
            "configuration": config, "complete": complete,
            "completed_units": len(calibration) + len(units), "total_units": total,
            "calibration": calibration, "units": units, **extra,
        }
        if out is not None:
            _write_progress(out, payload)
        return payload

    for seed in range(seeds):
        for identity in range(identities):
            if (seed, identity) in warm:
                continue
            started = perf_counter()
            record = episode_runner(
                model, organism, cell="reset_rate",
                identity_seed=identity_seed_base + seed * SEED_STRIDE + identity,
                episode_seed=episode_seed_base + seed * SEED_STRIDE + identity * EPISODE_STRIDE,
            )
            record.pop("donor_memory")
            row = {"seed_index": seed, "identity": identity, **record}
            calibration.append(row)
            warm[seed, identity] = row
            save()
            print(f"block {seed + 1}/{seeds} calibration {identity + 1}/{identities} "
                  f"{perf_counter() - started:.2f}s [{len(calibration) + len(units)}/{total}]", flush=True)
        for identity in range(identities):
            for cell in CELLS:
                if (seed, identity, cell) in done:
                    continue
                started = perf_counter()
                record = cell_runner(
                    model, organism, cell=cell,
                    identity_seed=identity_seed_base + seed * SEED_STRIDE + identity,
                    episode_base=episode_seed_base + seed * SEED_STRIDE + identity * EPISODE_STRIDE,
                    episodes=episodes, own_prior=warm[seed, identity]["own_memory"],
                    donor_prior=warm[seed, (identity + 1) % identities]["own_memory"],
                )
                units.append({"seed_index": seed, "identity": identity, "cell": cell, **record})
                done.add((seed, identity, cell))
                save()
                survival = np.mean([row["survived"] for row in record["episodes"]])
                print(f"block {seed + 1}/{seeds} identity {identity + 1}/{identities} "
                      f"{cell:<15} survival {survival:.3f} {perf_counter() - started:.2f}s "
                      f"[{len(calibration) + len(units)}/{total}]", flush=True)
    cells = summarize(units, seeds=seeds, identities=identities, episodes=episodes)
    # Smoke-sized runs never get promoted into a preregistered gate result.
    gates = grade(cells) if (seeds, identities, episodes) == (SEEDS, IDENTITIES, TEST_EPISODES) else None
    return save(True, cells=cells, gates=gates)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/organism/probe52_guided_report_lexicon/adult/organism_report_seed1.npz")
    parser.add_argument("--seeds", type=int, default=SEEDS)
    parser.add_argument("--identities", type=int, default=IDENTITIES)
    parser.add_argument("--episodes", type=int, default=TEST_EPISODES)
    parser.add_argument("--identity-seed-base", type=int, default=IDENTITY_SEED_BASE)
    parser.add_argument("--episode-seed-base", type=int, default=EPISODE_SEED_BASE)
    parser.add_argument("--out", default="runs/organism/probe74_cross_life_self/survey.json")
    args = parser.parse_args()
    validate_run(seeds=args.seeds, identities=args.identities, episodes=args.episodes,
                 identity_seed_base=args.identity_seed_base,
                 episode_seed_base=args.episode_seed_base)
    checkpoint = Path(args.checkpoint)
    model, organism = load_organism_checkpoint(checkpoint)
    result = run_survey(model, organism, parent_fingerprint=checkpoint_fingerprint(checkpoint),
                        seeds=args.seeds, identities=args.identities, episodes=args.episodes,
                        identity_seed_base=args.identity_seed_base,
                        episode_seed_base=args.episode_seed_base,
                        out=Path(args.out))
    for cell in result["cells"]:
        print(f"{cell['cell']}: survival {cell['survival']:.4f}; blocks {cell['per_seed_survival']}")
    print(json.dumps(result["gates"], indent=2))


if __name__ == "__main__":
    main()
