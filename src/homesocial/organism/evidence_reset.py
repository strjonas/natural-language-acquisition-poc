"""Probe79: matched-history screen of an observable prediction-error reset."""
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
from homesocial.island.report import NEED_TO_REPORT_WORD, REPORT_NEEDS
from homesocial.organism.changing_body import change_schedule, clear_rate_evidence
from homesocial.organism.cross_life_self import (
    RateMemory, WORLD_KWARGS, checkpoint_fingerprint, mechanism_fingerprint,
    memory_rates, reset_identity_world, run_episode as calibration_episode,
)
from homesocial.organism.future_request import RequestLedger, answer_horizon, expected_portion, need_scores, true_rates
from homesocial.organism.individual_self import BOUND_EPSILON, _true_body
from homesocial.organism.learned_rate_survival import _positive_gate, _summarize_values, _word_for_rates, _write_progress
from homesocial.organism.portion_request import MoveFraction, SelfModelTier
from homesocial.organism.report_audit import FIDELITY_WARMUP
from homesocial.organism.self_belief import _sample_motor_action
from homesocial.organism.self_calibration import MAX_LOG_CORRECTION, _METABOLIC_INDEX
from homesocial.organism.train import execute_agent_action, load_organism_checkpoint

READOUTS = ("retained", "adaptive", "muted", "scrambled", "oracle_reset", "true_rate")
DETECTORS = ("adaptive", "muted", "scrambled")
MEMORIES = READOUTS[:-1]
COHORTS = ("stable", "changed")
SEEDS, IDENTITIES, EPISODES = 5, 28, 5
IDENTITY_BASE, EPISODE_BASE = 4_400_000_000, 4_500_000_000
DIAGNOSTIC_IDENTITY, DIAGNOSTIC_EPISODE = 4_600_000_000, 4_700_000_000
STRIDE, DELAY, THRESHOLD = 2_000_000, 24, .05
PAD_ID = TOKEN_TO_ID[PAD_TOKEN]
PREREG = Path("docs/decisions/2026-10-02-evidence-reset-preregistration.md")


def rng_seed(seed: int, mode: str) -> int:
    return int.from_bytes(sha256(f"probe79:{mode}:{seed}".encode()).digest(), "big")


class EvidenceResetMemory(RateMemory):
    """Existing RLS arithmetic with a pre-fit hook; no simulator access."""
    def __init__(self, organism, report, *, mode: str, random_seed: int):
        if mode not in DETECTORS:
            raise ValueError("unknown detector mode")
        super().__init__(organism, report)
        self.mode = mode
        self.rng = Random(random_seed)
        self.last_sign = np.zeros(len(REPORT_NEEDS), dtype=int)
        self.reset_ticks: list[int] = []
        self.monitored_readings = 0

    def start_episode(self, packet, world, prior=None):
        super().start_episode(packet, world, prior)
        self.last_sign.fill(0)
        self.reset_ticks = []
        self.monitored_readings = 0

    def observe_error(self, reading, prediction, *, tick: int) -> bool:
        actual = np.asarray(reading, dtype=float)
        errors = actual - np.asarray(prediction, dtype=float)
        eligible = (actual > BOUND_EPSILON) & (actual < 1 - BOUND_EPSILON)
        signs = np.where(eligible & (np.abs(errors) > THRESHOLD), np.sign(errors), 0).astype(int)
        if self.mode == "scrambled":
            # Corruption affects only detection, never the real fitting target.
            for i in range(len(signs)):
                if eligible[i]:
                    random_sign = self.rng.choice((-1, 1))
                    signs[i] = random_sign if signs[i] != 0 else 0
        trigger = bool(np.any((signs != 0) & (signs == self.last_sign)))
        self.last_sign = signs
        self.monitored_readings += 1
        if trigger and self.mode != "muted":
            clear_rate_evidence(self)
            self.last_sign.fill(0)
            self.reset_ticks.append(int(tick))
            return True
        return False

    def update(self, before, action_index, after, world, reading):
        # Keep the original RLS update order and arithmetic. The added hook is
        # between public prediction and fitting, so no target leakage or second
        # history update can erase the innovation being monitored.
        self._jacobian.update(before, action_index, after)
        self.evidence_ticks += int(after.step_count - before.step_count)
        if reading is None:
            return
        actual = np.asarray(reading, dtype=np.float64)
        self.observe_error(actual, self.point(), tick=after.step_count)
        base = self._jacobian.belief
        matrix = self._jacobian.matrix()
        self.readings += 1
        for index in range(len(REPORT_NEEDS)):
            if not BOUND_EPSILON < actual[index] < 1 - BOUND_EPSILON:
                self.skipped += 1
                continue
            regressor = matrix[index]
            target = float(actual[index] - base[index])
            covariance = self._covariance[index]
            projected = covariance @ regressor
            denominator = self._forgetting + float(regressor @ projected)
            if denominator <= 0.0:
                self.skipped += 1
                continue
            gain = projected / denominator
            self._a[index] += gain * (target - float(regressor @ self._a[index]))
            self._covariance[index] = (covariance - np.outer(gain, projected)) / self._forgetting
        np.clip(self._a, -.99, np.expm1(MAX_LOG_CORRECTION), out=self._a)
        self._m = np.log1p(self._a)
        for index in range(len(REPORT_NEEDS)):
            species = matrix[index, _METABOLIC_INDEX].sum()
            corrected = ((1 + self._a[index, _METABOLIC_INDEX]) * matrix[index, _METABOLIC_INDEX]).sum()
            self._burn_species[index] += species
            self._burn_corrected[index] += corrected
        self._jacobian.snap(actual)


def run_episode(model, organism, *, cohort: str, identity_seed: int, episode_seed: int,
                episode: int, priors: dict, oracle_seen: bool) -> dict:
    if cohort not in COHORTS:
        raise ValueError("unknown cohort")
    world, packet, report = reset_identity_world(organism, identity_seed=identity_seed, episode_seed=episode_seed)
    schedule = change_schedule(identity_seed)
    changed = cohort == "changed" and episode > schedule["episode"]
    if changed:
        world.grid.metabolic_scale = dict(schedule["constants"])
    tier = SelfModelTier("recursive", organism, report)
    tier.reset(packet, None)
    memories = {name: EvidenceResetMemory(organism, report, mode=name, random_seed=rng_seed(episode_seed, name))
                if name in DETECTORS else RateMemory(organism, report) for name in MEMORIES}
    for name, memory in memories.items():
        memory.start_episode(packet, None, priors[name])
    cue_tick = None
    if changed and not oracle_seen:
        clear_rate_evidence(memories["oracle_reset"])
        oracle_seen, cue_tick = True, 0
    ledger = RequestLedger(help_period=report.help_period, delay=DELAY, expected_portion=expected_portion(report))
    ledger.reset()
    moves = MoveFraction(organism, report)
    moves.reset(packet)
    hidden, rng = None, Random(episode_seed + 59_000_003)
    scores = {p: {n: {"rate_mae": 0., "regret": 0., "word_changes": 0} for n in READOUTS}
              for p in ("all", "after_change")}
    counts = {"all": 0, "after_change": 0}
    steps = 0
    start_readings = memories["retained"].readings
    while True:
        now = int(packet.step_count)
        if (cohort == "changed" and episode == schedule["episode"] and not changed and now >= schedule["tick"]):
            world.grid.metabolic_scale = dict(schedule["constants"])
            changed = True
            clear_rate_evidence(memories["oracle_reset"])
            oracle_seen, cue_tick = True, now
        rates = {name: memory_rates(memory, report, moves.fraction) for name, memory in memories.items()}
        rates["true_rate"] = true_rates(world, report, moves.fraction)
        words = {name: _word_for_rates(tier, vector, ledger, now=now, delay=DELAY, report=report)
                 for name, vector in rates.items()}
        if not np.array_equal(rates["muted"], rates["retained"]) or words["muted"] != words["retained"]:
            raise AssertionError("muted detector differs from original RLS")
        if now >= FIDELITY_WARMUP:
            true_scores = need_scores(believed_levels=_true_body(world), believed_rates=rates["true_rate"],
                believed_uptake=world.uptake_scale, arriving=ledger.arriving(now, world.uptake_scale, delay=DELAY),
                horizon=answer_horizon(DELAY), report=report)
            best = float(true_scores.max())
            for panel in (("all", "after_change") if changed else ("all",)):
                counts[panel] += 1
                for name in READOUTS:
                    scores[panel][name]["regret"] += best - float(true_scores[REPORT_NEEDS.index(words[name])])
                    scores[panel][name]["rate_mae"] += float(np.abs(rates[name] - rates["true_rate"]).mean())
                    scores[panel][name]["word_changes"] += int(words[name] != words["retained"])
        spoken = words["retained"]
        ledger.record(now, spoken)
        action, hidden = _sample_motor_action(model, packet, hidden, rng)
        world.hear((TOKEN_TO_ID[NEED_TO_REPORT_WORD[spoken]], PAD_ID))
        before = packet
        packet, _, terminated, truncated, info = execute_agent_action(world, packet, action,
            consume_options=organism.consume_options, inspect_options=organism.inspect_options)
        steps += int(info["duration"])
        if terminated or truncated:
            break
        moves.observe(before, action, packet)
        reading = info.get("interoception")
        tier.update(before, action, packet, None, reading)
        for memory in memories.values():
            memory.update(before, action, packet, None, reading)
    return {"episode": episode, "episode_seed": episode_seed, "survived": int(not terminated),
            "steps": steps, "counts": counts, "scores": scores, "oracle_cue_tick": cue_tick,
            "readings": memories["retained"].readings - start_readings,
            "detector_resets": {n: memories[n].reset_ticks for n in DETECTORS},
            "priors": {n: m.snapshot() for n, m in memories.items()}, "oracle_seen": oracle_seen}


def run_unit(model, organism, *, cohort, identity_seed, episode_base, warm):
    priors = {n: warm for n in MEMORIES}
    rows, oracle_seen = [], False
    for episode in range(1, EPISODES + 1):
        row = run_episode(model, organism, cohort=cohort, identity_seed=identity_seed,
                          episode_seed=episode_base + episode, episode=episode, priors=priors, oracle_seen=oracle_seen)
        priors, oracle_seen = row.pop("priors"), row.pop("oracle_seen")
        rows.append(row)
    return {"episodes": rows, "schedule": change_schedule(identity_seed)}


def summarize(units: list[dict], *, seeds: int, identities: int) -> dict:
    expected = {(b,i,c) for b in range(seeds) for i in range(identities) for c in COHORTS}
    if len(units) != len(expected) or {(u["block"],u["identity"],u["cohort"]) for u in units} != expected:
        raise ValueError("incomplete or duplicate units")
    if any(len(u["episodes"]) != EPISODES for u in units):
        raise ValueError("incomplete episodes")
    blocks = {}
    for cohort in COHORTS:
        blocks[cohort] = []
        for b in range(seeds):
            group = [u for u in units if u["block"] == b and u["cohort"] == cohort]
            rows = [r for u in group for r in u["episodes"]]
            panels = {}
            for panel in ("all", "after_change"):
                ticks = sum(r["counts"][panel] for r in rows)
                totals = {n: {m: sum(r["scores"][panel][n][m] for r in rows)
                              for m in ("rate_mae", "regret", "word_changes")} for n in READOUTS}
                panels[panel] = {"ticks": ticks, "totals": totals}
            blocks[cohort].append({"identities": identities, "episodes":len(rows),
                "survivors":sum(r["survived"] for r in rows), "steps":sum(r["steps"] for r in rows),
                "readings":sum(r["readings"] for r in rows), "panels":panels,
                "reset_identities":{n:sum(any(r["detector_resets"][n] for r in u["episodes"]) for u in group) for n in DETECTORS},
                "resets":{n:sum(len(r["detector_resets"][n]) for r in rows) for n in DETECTORS}})
    def values(cohort, name, metric):
        return [b["panels"]["all"]["totals"][name][metric]/max(1,b["panels"]["all"]["ticks"]) for b in blocks[cohort]]
    contrasts, reductions = {}, {}
    for metric in ("rate_mae", "regret"):
        adaptive = values("changed", "adaptive", metric)
        for baseline in ("retained", "scrambled"):
            reference = values("changed", baseline, metric)
            contrasts[f"{baseline}_minus_adaptive_{metric}"] = _summarize_values([x-y for x,y in zip(reference,adaptive)])
        base = float(np.mean(values("changed","retained",metric)))
        reductions[metric] = (base-float(np.mean(adaptive)))/base if base>0 else 0.
    stable_mae = float(np.mean(values("stable","adaptive","rate_mae")))
    stable_base = float(np.mean(values("stable","retained","rate_mae")))
    stable_regret_cost = float(np.mean(values("stable","adaptive","regret")) - np.mean(values("stable","retained","regret")))
    stable_false_alarm = sum(b["reset_identities"]["adaptive"] for b in blocks["stable"])/(seeds*identities)
    muted_exact = all(r["scores"][p]["muted"] == r["scores"][p]["retained"]
                      for u in units for r in u["episodes"] for p in ("all","after_change"))
    gates = {
        "G1_changed_improvement": all(_positive_gate(contrasts[f"retained_minus_adaptive_{m}"]) and reductions[m]>=.2 for m in ("rate_mae","regret")),
        "G2_stable_preservation": stable_mae<=1.1*stable_base and stable_regret_cost<=.001 and stable_false_alarm<=.1,
        "G3_controls": muted_exact and all(_positive_gate(contrasts[f"scrambled_minus_adaptive_{m}"]) for m in ("rate_mae","regret")),
    } if (seeds,identities)==(SEEDS,IDENTITIES) else None
    return {"blocks":blocks,"contrasts":contrasts,"changed_reductions":reductions,
            "stable_mae":stable_mae,"stable_baseline_mae":stable_base,
            "stable_regret_cost":stable_regret_cost,"stable_false_alarm_share":stable_false_alarm,
            "muted_exact":muted_exact,"gates":gates,"all_pass":all(gates.values()) if gates else None}


def run(model, organism, *, checkpoint: Path, out: Path, diagnostic: bool) -> dict:
    seeds, identities = (1,2) if diagnostic else (SEEDS,IDENTITIES)
    id_base, ep_base = (DIAGNOSTIC_IDENTITY,DIAGNOSTIC_EPISODE) if diagnostic else (IDENTITY_BASE,EPISODE_BASE)
    manifest = json.loads(json.dumps({"diagnostic":diagnostic,"seeds":seeds,"identities":identities,
        "episodes":EPISODES,"identity_base":id_base,"episode_base":ep_base,"threshold":THRESHOLD,
        "world":WORLD_KWARGS,"organism":asdict(organism),"parent":checkpoint_fingerprint(checkpoint),
        "source":mechanism_fingerprint(),"prereg":sha256(PREREG.read_bytes()).hexdigest()}))
    calibration, units = [], []
    if out.exists():
        saved = json.loads(out.read_text())
        if saved["manifest"] != manifest:
            raise ValueError("source, parent or configuration changed")
        calibration, units = saved["calibration"], saved["units"]
        if saved["complete"]:
            summarize(units,seeds=seeds,identities=identities)
            return saved
    warm = {(r["block"],r["identity"]):r for r in calibration}
    done = {(u["block"],u["identity"],u["cohort"]) for u in units}
    expected_warm = {(b,i) for b in range(seeds) for i in range(identities)}
    expected = {(b,i,c) for b,i in expected_warm for c in COHORTS}
    if (len(warm)!=len(calibration) or len(done)!=len(units) or not set(warm)<=expected_warm
            or not done<=expected or any((b,i) not in warm for b,i,c in done)):
        raise ValueError("invalid resume units")
    start = perf_counter()
    def save(complete=False):
        payload = {"manifest":manifest,"complete":complete,"calibration":calibration,"units":units,
                   "completed_units":len(calibration)+len(units),"total_units":len(expected_warm)+len(expected),
                   "elapsed_this_run":perf_counter()-start}
        if complete:
            payload["summary"] = summarize(units,seeds=seeds,identities=identities)
        _write_progress(out,payload)
        return payload
    for b in range(seeds):
        for i in range(identities):
            identity_seed, episode_base = id_base+b*STRIDE+i, ep_base+b*STRIDE+i*16
            if (b,i) not in warm:
                record = calibration_episode(model,organism,cell="reset_rate",identity_seed=identity_seed,episode_seed=episode_base)
                record.pop("donor_memory")
                record.update(block=b,identity=i)
                calibration.append(record)
                warm[b,i] = record
                save()
            for cohort in COHORTS:
                if (b,i,cohort) in done:
                    continue
                began = perf_counter()
                record = run_unit(model,organism,cohort=cohort,identity_seed=identity_seed,
                                  episode_base=episode_base,warm=warm[b,i]["own_memory"])
                units.append({"block":b,"identity":i,"cohort":cohort,**record})
                done.add((b,i,cohort))
                save()
                print(f"block {b+1}/{seeds} body {i+1}/{identities} {cohort}: {perf_counter()-began:.2f}s "
                      f"[{len(calibration)+len(units)}/{len(expected_warm)+len(expected)}]",flush=True)
    return save(True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic",action="store_true")
    parser.add_argument("--checkpoint",type=Path,default=Path("runs/organism/probe52_guided_report_lexicon/adult/organism_report_seed1.npz"))
    parser.add_argument("--out",type=Path,default=Path("runs/organism/probe79_evidence_reset/survey.json"))
    args = parser.parse_args()
    model,organism = load_organism_checkpoint(args.checkpoint)
    result = run(model,organism,checkpoint=args.checkpoint,out=args.out,diagnostic=args.diagnostic)
    print(json.dumps({k:v for k,v in result["summary"].items() if k!="blocks"},indent=2))


if __name__ == "__main__":
    main()
