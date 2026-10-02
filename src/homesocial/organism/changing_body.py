"""Probe78: oracle-cued evidence reset after hidden bodily changes.

This is a feasibility ceiling, not a learned change detector. All rate
readouts share a physical history within each episode. Only one drives speech.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from random import Random
from time import perf_counter

import numpy as np

from homesocial.creole.vocab import PAD_TOKEN, TOKEN_TO_ID
from homesocial.island.report import BODY_NEEDS, NEED_TO_REPORT_WORD, REPORT_NEEDS
from homesocial.organism.cross_life_self import (
    DELAY, RateMemory, WORLD_KWARGS, checkpoint_fingerprint, mechanism_fingerprint,
    memory_rates, reset_identity_world, run_episode as calibration_episode,
)
from homesocial.organism.future_request import (
    RequestLedger, answer_horizon, expected_portion, need_scores, true_rates,
)
from homesocial.organism.individual_self import _true_body
from homesocial.organism.learned_rate_survival import (
    _contrast, _positive_gate, _summarize_values, _word_for_rates, _write_progress,
)
from homesocial.organism.portion_request import MoveFraction, SelfModelTier
from homesocial.organism.report_audit import FIDELITY_WARMUP
from homesocial.organism.self_belief import _sample_motor_action
from homesocial.organism.train import execute_agent_action, load_organism_checkpoint

CELLS = ("retained", "episodic", "forgetting", "oracle_reset", "true_rate")
MEMORIES = CELLS[:-1]
COHORTS = ("stable", "changed")
SEEDS, IDENTITIES, EPISODES = 5, 28, 5
IDENTITY_BASE, EPISODE_BASE = 4_000_000_000, 4_100_000_000
DIAGNOSTIC_IDENTITY, DIAGNOSTIC_EPISODE = 4_200_000_000, 4_220_000_000
STRIDE = 2_000_000
FORGETTING = 0.9
PAD_ID = TOKEN_TO_ID[PAD_TOKEN]
PREREG = Path("docs/decisions/2026-09-30-changing-body-ceiling-preregistration.md")


def domain_seed(seed: int, domain: str) -> int:
    return int.from_bytes(sha256(f"probe78:{domain}:{seed}".encode()).digest(), "big")


def change_schedule(identity_seed: int) -> dict:
    timing = Random(domain_seed(identity_seed, "timing"))
    rates = Random(domain_seed(identity_seed, "rates"))
    return {"episode": timing.choice((2, 3, 4)), "tick": timing.choice((24, 48, 72)),
            "constants": {need: rates.uniform(.4, 1.6) for need in BODY_NEEDS}}


class ForgettingMemory(RateMemory):
    def __init__(self, organism, report):
        super().__init__(organism, report)
        self._forgetting = FORGETTING

    def update(self, before, action_index, after, world, reading):
        if reading is not None:
            self._burn_species *= FORGETTING
            self._burn_corrected *= FORGETTING
        super().update(before, action_index, after, world, reading)


def clear_rate_evidence(memory: RateMemory) -> None:
    """Cue gives no bodily state: preserve Jacobian, counts and episodic belief."""
    memory._a.fill(0.0)
    memory._m.fill(0.0)
    memory._covariance[:] = np.eye(memory._a.shape[1])[None, :, :] / memory._ridge
    memory._burn_species.fill(0.0)
    memory._burn_corrected.fill(0.0)


def run_episode(model, organism, *, cell: str, cohort: str, identity_seed: int,
                episode_seed: int, episode: int, priors: dict, cue_seen: bool) -> dict:
    if cell not in CELLS or cohort not in COHORTS:
        raise ValueError("unknown cell/cohort")
    world, packet, report = reset_identity_world(
        organism, identity_seed=identity_seed, episode_seed=episode_seed)
    schedule = change_schedule(identity_seed)
    changed_at_birth = cohort == "changed" and episode > schedule["episode"]
    if changed_at_birth:
        world.grid.metabolic_scale = dict(schedule["constants"])
    tier = SelfModelTier("recursive", organism, report)
    tier.reset(packet, None)
    memories = {name: (ForgettingMemory if name == "forgetting" else RateMemory)(organism, report)
                for name in MEMORIES}
    for name, memory in memories.items():
        memory.start_episode(packet, None, None if name == "episodic" else priors[name])
    reset_tick = None
    if changed_at_birth and not cue_seen:
        clear_rate_evidence(memories["oracle_reset"])
        cue_seen, reset_tick = True, 0
    ledger = RequestLedger(help_period=report.help_period, delay=DELAY,
                           expected_portion=expected_portion(report))
    ledger.reset()
    moves = MoveFraction(organism, report)
    moves.reset(packet)
    hidden = None
    rng = Random(episode_seed + 59_000_003)
    scores = {panel: {name: {"regret": 0., "rate_mae": 0., "word_changes": 0}
                      for name in CELLS} for panel in ("all", "after_change")}
    counts = {"all": 0, "after_change": 0}
    physically_changed = changed_at_birth
    starting_readings = memories["retained"].readings
    steps = 0
    while True:
        now = int(packet.step_count)
        if (cohort == "changed" and episode == schedule["episode"]
                and not physically_changed and now >= schedule["tick"]):
            world.grid.metabolic_scale = dict(schedule["constants"])
            physically_changed = True
            clear_rate_evidence(memories["oracle_reset"])
            cue_seen, reset_tick = True, now
        fraction = moves.fraction
        rates = {name: memory_rates(memory, report, fraction) for name, memory in memories.items()}
        rates["true_rate"] = true_rates(world, report, fraction)
        words = {name: _word_for_rates(tier, vector, ledger, now=now, delay=DELAY, report=report)
                 for name, vector in rates.items()}
        if cohort == "stable":
            if not np.array_equal(rates["retained"], rates["oracle_reset"]):
                raise AssertionError("sham cue changed oracle-reset rates")
        if now >= FIDELITY_WARMUP:
            true_scores = need_scores(
                believed_levels=_true_body(world), believed_rates=rates["true_rate"],
                believed_uptake=world.uptake_scale,
                arriving=ledger.arriving(now, world.uptake_scale, delay=DELAY),
                horizon=answer_horizon(DELAY), report=report)
            best = float(true_scores.max())
            for panel in (("all", "after_change") if physically_changed else ("all",)):
                counts[panel] += 1
                for name in CELLS:
                    scores[panel][name]["regret"] += best - float(true_scores[REPORT_NEEDS.index(words[name])])
                    scores[panel][name]["rate_mae"] += float(np.abs(rates[name] - rates["true_rate"]).mean())
                    scores[panel][name]["word_changes"] += int(words[name] != words["retained"])
        spoken = words[cell]
        ledger.record(now, spoken)
        action, hidden = _sample_motor_action(model, packet, hidden, rng)
        world.hear((TOKEN_TO_ID[NEED_TO_REPORT_WORD[spoken]], PAD_ID))
        before = packet
        packet, _, terminated, truncated, info = execute_agent_action(
            world, packet, action, consume_options=organism.consume_options,
            inspect_options=organism.inspect_options)
        steps += int(info["duration"])
        if terminated or truncated:
            break
        moves.observe(before, action, packet)
        reading = info.get("interoception")
        tier.update(before, action, packet, None, reading)
        for memory in memories.values():
            memory.update(before, action, packet, None, reading)
    return {"episode": episode, "episode_seed": episode_seed, "survived": int(not terminated),
            "steps": steps, "counts": counts, "scores": scores,
            "reset_tick": reset_tick, "physical_change_exposed": physically_changed,
            "readings": memories["retained"].readings - starting_readings,
            "priors": {name: memory.snapshot() for name, memory in memories.items()},
            "cue_seen": cue_seen}


def run_unit(model, organism, *, cell: str, cohort: str, identity_seed: int,
             episode_base: int, warm: dict) -> dict:
    priors = {name: warm for name in MEMORIES}
    records, cue_seen = [], False
    for episode in range(1, EPISODES + 1):
        record = run_episode(model, organism, cell=cell, cohort=cohort,
                             identity_seed=identity_seed, episode_seed=episode_base + episode,
                             episode=episode, priors=priors, cue_seen=cue_seen)
        priors, cue_seen = record.pop("priors"), record.pop("cue_seen")
        records.append(record)
    return {"episodes": records, "schedule": change_schedule(identity_seed),
            "final_readings": priors["retained"]["readings"],
            "final_evidence_ticks": priors["retained"]["evidence_ticks"]}


def summarize(units: list[dict], *, seeds: int, identities: int) -> dict:
    expected = {(b, i, c, a) for b in range(seeds) for i in range(identities)
                for c in COHORTS for a in CELLS}
    actual = {(u["block"], u["identity"], u["cohort"], u["cell"]) for u in units}
    if len(units) != len(expected) or actual != expected:
        raise ValueError("incomplete or duplicate units")
    if any(len(u["episodes"]) != EPISODES for u in units):
        raise ValueError("incomplete episodes")
    # A stable oracle must be bit-identical in the real closed loop, not only
    # in same-history counterfactual scores.
    keyed = {(u["block"], u["identity"], u["cohort"], u["cell"]): u for u in units}
    parity = all(keyed[b, i, "stable", "retained"]["episodes"] == keyed[b, i, "stable", "oracle_reset"]["episodes"]
                 for b in range(seeds) for i in range(identities))
    if not parity:
        raise ValueError("stable oracle reset differs from retained: invalid construction")
    cohorts = {}
    for cohort in COHORTS:
        cells = {}
        for cell in CELLS:
            blocks = []
            for block in range(seeds):
                selected = [u for u in units if (u["block"], u["cohort"], u["cell"]) == (block, cohort, cell)]
                rows = [r for u in selected for r in u["episodes"]]
                n = len(rows)
                panels = {}
                for panel in ("all", "after_change"):
                    ticks = sum(r["counts"][panel] for r in rows)
                    totals = {name: {metric: sum(r["scores"][panel][name][metric] for r in rows)
                                      for metric in ("regret", "rate_mae", "word_changes")}
                              for name in CELLS}
                    panels[panel] = {"ticks": ticks, "totals": totals}
                blocks.append({"survived": sum(r["survived"] for r in rows), "episodes": n,
                               "survival": sum(r["survived"] for r in rows) / n,
                               "steps": sum(r["steps"] for r in rows),
                               "readings": sum(r["readings"] for r in rows),
                               "exposed_episodes": sum(r["physical_change_exposed"] for r in rows),
                               "within_life_cues": sum(r["reset_tick"] is not None and r["reset_tick"] > 0 for r in rows),
                               "birth_cues": sum(r["reset_tick"] == 0 for r in rows),
                               "panels": panels})
            cells[cell] = {"survival": float(np.mean([b["survival"] for b in blocks])),
                           "survivors": sum(b["survived"] for b in blocks),
                           "blocks": blocks}
        cohorts[cohort] = cells
    changed = cohorts["changed"]
    retained = changed["retained"]["blocks"]
    contrasts = {
        "C1_true_rate_survival": _contrast([b["survival"] for b in changed["true_rate"]["blocks"]], [b["survival"] for b in retained]),
        "C2_oracle_reset_survival": _contrast([b["survival"] for b in changed["oracle_reset"]["blocks"]], [b["survival"] for b in retained]),
    }
    for metric in ("rate_mae", "regret"):
        values = []
        for b in retained:
            panel = b["panels"]["all"]
            if panel["ticks"] <= 0:
                raise ValueError("no scored decisions")
            values.append((panel["totals"]["retained"][metric] - panel["totals"]["oracle_reset"][metric]) / panel["ticks"])
        contrasts[f"C3_oracle_reset_{metric}"] = _summarize_values(values)
    gates = {k: _positive_gate(v) for k, v in contrasts.items()} if (seeds, identities) == (SEEDS, IDENTITIES) else None
    return {"cohorts": cohorts, "contrasts": contrasts, "stable_exact_parity": parity,
            "gates": gates, "all_pass": all(gates.values()) if gates else None}


def run(model, organism, *, checkpoint: Path, out: Path, diagnostic: bool) -> dict:
    seeds, identities = (1, 2) if diagnostic else (SEEDS, IDENTITIES)
    identity_base, episode_base = ((DIAGNOSTIC_IDENTITY, DIAGNOSTIC_EPISODE) if diagnostic
                                  else (IDENTITY_BASE, EPISODE_BASE))
    manifest = json.loads(json.dumps({
        "schema": 1, "diagnostic": diagnostic, "seeds": seeds, "identities": identities,
        "episodes": EPISODES, "cells": CELLS, "cohorts": COHORTS,
        "identity_base": identity_base, "episode_base": episode_base,
        "forgetting": FORGETTING, "organism": asdict(organism), "world": WORLD_KWARGS,
        "parent": checkpoint_fingerprint(checkpoint), "source": mechanism_fingerprint(),
        "prereg": sha256(PREREG.read_bytes()).hexdigest(),
    }))
    warm, units = [], []
    if out.exists():
        saved = json.loads(out.read_text())
        if saved["manifest"] != manifest:
            raise ValueError("source, parent or configuration changed")
        warm, units = saved["calibration"], saved["units"]
        if saved["complete"]:
            summarize(units, seeds=seeds, identities=identities)
            return saved
    warm_index = {(r["block"], r["identity"]): r for r in warm}
    done = {(r["block"], r["identity"], r["cohort"], r["cell"]) for r in units}
    expected_warm = {(b, i) for b in range(seeds) for i in range(identities)}
    expected = {(b, i, c, a) for b, i in expected_warm for c in COHORTS for a in CELLS}
    if (len(warm_index) != len(warm) or len(done) != len(units)
            or not set(warm_index) <= expected_warm or not done <= expected
            or any((b, i) not in warm_index for b, i, _, _ in done)):
        raise ValueError("invalid resume units")
    start = perf_counter()
    def save(complete=False):
        payload = {"manifest": manifest, "complete": complete, "calibration": warm,
                   "units": units, "completed_units": len(warm) + len(units),
                   "total_units": len(expected_warm) + len(expected),
                   "elapsed_this_run": perf_counter() - start}
        if complete:
            payload["summary"] = summarize(units, seeds=seeds, identities=identities)
        _write_progress(out, payload)
        return payload
    for block in range(seeds):
        for identity in range(identities):
            id_seed = identity_base + block * STRIDE + identity
            ep_base = episode_base + block * STRIDE + identity * 16
            if (block, identity) not in warm_index:
                record = calibration_episode(model, organism, cell="reset_rate", identity_seed=id_seed, episode_seed=ep_base)
                record.pop("donor_memory")
                record.update(block=block, identity=identity)
                warm.append(record)
                warm_index[block, identity] = record
                save()
            for cohort in COHORTS:
                for cell in CELLS:
                    if (block, identity, cohort, cell) in done:
                        continue
                    began = perf_counter()
                    record = run_unit(model, organism, cell=cell, cohort=cohort,
                                      identity_seed=id_seed, episode_base=ep_base,
                                      warm=warm_index[block, identity]["own_memory"])
                    units.append({"block": block, "identity": identity,
                                  "cohort": cohort, "cell": cell, **record})
                    done.add((block, identity, cohort, cell))
                    save()
                    print(f"block {block+1}/{seeds} body {identity+1}/{identities} {cohort} {cell}: "
                          f"{perf_counter()-began:.2f}s [{len(warm)+len(units)}/{len(expected_warm)+len(expected)}]", flush=True)
    return save(True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/organism/probe52_guided_report_lexicon/adult/organism_report_seed1.npz"))
    parser.add_argument("--out", type=Path, default=Path("runs/organism/probe78_changing_body/survey.json"))
    args = parser.parse_args()
    model, organism = load_organism_checkpoint(args.checkpoint)
    result = run(model, organism, checkpoint=args.checkpoint, out=args.out, diagnostic=args.diagnostic)
    print(json.dumps({k: v for k, v in result["summary"].items() if k != "cohorts"}, indent=2))


if __name__ == "__main__":
    main()
