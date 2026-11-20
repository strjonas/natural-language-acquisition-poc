"""Probe80: a self-posterior that discovers its own body while it lives in it.

See docs/decisions/2026-10-04-active-self-discovery-preregistration.md.

Substrate: Probe77's lived, sequential version of Probe76's body family -- ten
opaque actuators, K1-K5 hidden bounded resources, two scalar sensations (the
sum and the minimum of the five sensor channels), shocks, one actuator choice
every six ticks absorbed on the next tick, 400-tick lives, six lives per
persistent identity. What changes from Probe77 is who chooses the actuator:
here the organism's own self-model does, and survival depends on it.

Arms share identities, birth levels, individual rates and shock streams:

    random            Probe77's uniform actuator choice; no self-model.
    bandit            structure-free learner: optimistic action values on the
                      change in the observed minimum.
    posterior         the self-posterior, retained across lives; acts by
                      posterior sampling; checks itself against its own
                      sensations and rebuilds when falsified.   <- treatment
    uncritical        the same posterior without self-criticism (lesion).
    reset             the same mechanism, re-initialized at every birth.
    certainty         the same posterior, acting on its single most probable
                      self (uncertainty lesion).
    yoked             the same mechanism fed by a donor body's sensations under
                      the same actions, acting on its own body (self/other).
    oracle_structure  true structure, sizes and rates; tracks levels from the
                      same two sensations (structure ceiling).
    oracle            true state and structure (ceiling).

Physics is private to ``LivedBody``. Every self-model receives only the
actuator it chose and the two scalars that followed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from random import Random
from statistics import mean
from time import perf_counter

import numpy as np

from homesocial.island.expanded_body import BodyState, ExpandedBody, ExpandedBodyConfig
from homesocial.organism.saturation_structure import ACTUATORS, PhysicalWorld, make_physical_world
from homesocial.organism.self_posterior import (
    CHANNELS, DIM, PosteriorConfig, SelfPosterior, _P, _R, _S,
)

ARMS = ("random", "bandit", "posterior", "uncritical", "reset", "certainty", "yoked", "oracle_structure", "oracle")
POSTERIOR_ARMS = ("posterior", "uncritical", "reset", "certainty", "yoked")
CORE_KS = (1, 2, 3, 4)
BLOCKS, IDENTITIES, LIVES = 5, 12, 6
PERMUTATION_IDENTITIES = 4
SURVEY_BASE = 5_000_000_000
DIAGNOSTIC_BASE = 5_100_000_000
DEVELOPMENT_BASE = 5_200_000_000
BLOCK_STRIDE, K_STRIDE, IDENTITY_STRIDE = 2_000_000, 100_000, 100
PHYSICS = ExpandedBodyConfig()
POSTERIOR = PosteriorConfig(particles=1000, hazard=0.0, shock_attribution="sample",
                            falsification_nats=30.0, rebuild_attempts=3)
PREREG = Path("docs/decisions/2026-10-04-active-self-discovery-preregistration.md")


def domain_seed(seed: int, domain: str) -> int:
    return int.from_bytes(sha256(f"probe80:{domain}:{seed}".encode()).digest(), "big")


def identity_seed(base: int, block: int, k: int, identity: int) -> int:
    return base + block * BLOCK_STRIDE + k * K_STRIDE + identity * IDENTITY_STRIDE


def true_partition(world: PhysicalWorld) -> tuple[tuple[int, ...], ...]:
    groups = {}
    for a, assignment in enumerate(world.actuators):
        if world.actionable(a):
            groups.setdefault(assignment.target, []).append(a)
    return tuple(sorted(tuple(sorted(g)) for g in groups.values()))


# ---------------------------------------------------------------- physics
class LivedBody:
    """One life of one identity. The only public outputs are ``sense()`` and
    whether the body is alive; nothing here is passed to a self-model."""

    def __init__(self, world: PhysicalWorld, *, identity: int, life: int) -> None:
        self.world = world
        body = ExpandedBody(world.schema(), config=replace(PHYSICS, shock_probability=0.0))
        body.reset()
        birth = Random(domain_seed(identity * 16 + life, "birth"))
        scale = Random(domain_seed(identity, "scale"))
        self._shock = Random(domain_seed(identity * 16 + life, "shock"))
        body.state = BodyState(body.schema, tuple(
            birth.choice(PHYSICS.birth_levels) if slot in world.active_slots else 1.0
            for slot in range(CHANNELS)))
        body.metabolic_scale = {axis.name: scale.uniform(0.4, 1.6) for axis in body.schema.axes}
        self.body = body
        self.alive = True

    @property
    def ticks(self) -> int:
        return self.body.step_count

    def sense(self) -> tuple[float, float]:
        state = self.body.state
        return CHANNELS * state.mean_viability(), state.viability()

    def tick(self, actuator: int | None) -> bool:
        body, world = self.body, self.world
        if actuator is not None and world.actionable(actuator):
            assignment = world.actuators[actuator]
            body.grant(body.schema.axes[assignment.target].name, assignment.size)
        body.step()
        if self._shock.random() < PHYSICS.shock_probability:
            slot = self._shock.randrange(CHANNELS)
            if slot in world.active_slots:
                body.state = body.state.add(body.schema.axes[slot].name, -PHYSICS.shock_size)
        if body.state.viability() <= 0.0:
            self.alive = False
        return self.alive

    def true_rates(self) -> np.ndarray:
        rates = self.body.rates()
        return np.array([rates[axis.name] if slot in self.world.active_slots else 0.0
                         for slot, axis in enumerate(self.body.schema.axes)])

    def oracle_choice(self) -> int:
        world = self.world
        rates, levels = self.true_rates(), np.array(self.body.state.values)
        live = [s for s in world.active_slots if s not in world.blocked_slots]
        urgent = min(live, key=lambda s: levels[s] / max(rates[s], 1e-9))
        for a, assignment in enumerate(world.actuators):
            if assignment.target == urgent and assignment.size == "large":
                return a
        raise AssertionError("every active resource has a large actuator")


# ----------------------------------------------------------- non-posterior
class MinBandit:
    """Optimistic running mean of each actuator's absorption-tick change in the
    observed minimum, net of the median of the last six ordinary changes.
    Untried actuators first; then epsilon-greedy (0.1). No hidden variables."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.total = np.zeros(ACTUATORS)
        self.count = np.zeros(ACTUATORS)

    def birth(self, total: float, minimum: float, *, retain: bool) -> None:
        if not retain:
            self.total[:] = 0.0
            self.count[:] = 0.0
        self.last = minimum
        self.drift: list[float] = []

    def observe(self, grant: int | None, total: float, minimum: float) -> None:
        change = minimum - self.last
        self.last = minimum
        if grant is None:
            self.drift = (self.drift + [change])[-6:]
        else:
            baseline = float(np.median(self.drift)) if self.drift else 0.0
            self.total[grant] += change - baseline
            self.count[grant] += 1

    def action(self) -> int:
        untried = np.flatnonzero(self.count == 0)
        if len(untried):
            return int(self.rng.choice(untried))
        if self.rng.uniform() < 0.1:
            return int(self.rng.integers(0, ACTUATORS))
        return int(np.argmax(self.total / self.count))


def oracle_structure_posterior(world: PhysicalWorld, rates: np.ndarray, rng) -> SelfPosterior:
    """The same posterior with every particle given the true structure, sizes,
    rates and shock size (variance 1e-10). Only levels remain to be tracked."""

    active = np.array([slot in world.active_slots for slot in range(CHANNELS)])
    drives = np.array([assignment.target for assignment in world.actuators])
    sizes = np.array([PHYSICS.portion_large if a.size == "large" else PHYSICS.portion_small
                      for a in world.actuators])
    # A rebuild would discard the given truth, so self-criticism is off here.
    posterior = SelfPosterior(replace(POSTERIOR, falsification_nats=0.0), rng)

    def fresh(count: int):
        mean_ = np.zeros((count, DIM))
        cov = np.zeros((count, DIM, DIM))
        mean_[:, _R] = np.where(active, rates, 0.0)
        mean_[:, _P] = sizes
        mean_[:, _S] = PHYSICS.shock_size
        for d in list(_R) + list(_P) + [_S]:
            cov[:, d, d] = 1e-10
        return (np.broadcast_to(active, (count, CHANNELS)).copy(),
                np.broadcast_to(drives, (count, ACTUATORS)).copy(), mean_, cov)

    posterior._fresh_structure = fresh
    return posterior


# ------------------------------------------------------------------ runs
def _report_record(posterior: SelfPosterior, truth) -> dict:
    report = posterior.structure_report()
    return {"partition": [list(g) for g in report.partition], "confidence": report.confidence,
            "resource_count": report.resource_count, "count_posterior": list(report.count_posterior),
            "tried": list(report.tried), "exact": report.partition == truth,
            "falsified": report.falsified, "rebuilds": posterior.rebuilds,
            "evidence": posterior.evidence, "collapses": posterior.collapses}


def run_arm(arm: str, world: PhysicalWorld, donor: PhysicalWorld, *, ident: int, donor_ident: int,
            lives: int | None = None) -> dict:
    lives = LIVES if lives is None else lives
    rng = np.random.default_rng(domain_seed(ident, "agent-" + arm))
    truth = true_partition(world)
    agent = None
    if arm in POSTERIOR_ARMS:
        config = replace(POSTERIOR, falsification_nats=0.0) if arm == "uncritical" else POSTERIOR
        agent = SelfPosterior(config, rng)
    elif arm == "oracle_structure":
        agent = oracle_structure_posterior(world, LivedBody(world, identity=ident, life=0).true_rates(), rng)
    elif arm == "bandit":
        agent = MinBandit(rng)
    uniform = Random(domain_seed(ident, "uniform-actions"))
    records, history = [], []
    for life in range(lives):
        body = LivedBody(world, identity=ident, life=life)
        other = LivedBody(donor, identity=donor_ident, life=life) if arm == "yoked" else None
        retain = life > 0 and arm != "reset"
        if agent is not None:
            source = other if other is not None else body
            agent.birth(*source.sense(), retain=retain)
            history.append(("birth", retain, *source.sense()))
        pending, grants, worst = None, 0, 0.0
        for _ in range(PHYSICS.life_steps):
            absorbed, pending = pending, None
            alive = body.tick(absorbed)
            if other is not None and other.alive:
                other.tick(absorbed)
            if not alive:
                break
            if agent is not None:
                before = getattr(agent, "evidence", 0.0)
                if other is None:
                    agent.observe(absorbed, *body.sense())
                    history.append(("tick", absorbed, *body.sense()))
                elif other.alive:
                    agent.observe(absorbed, *other.sense())
                worst = min(worst, getattr(agent, "evidence", 0.0) - before)
            if body.ticks % PHYSICS.help_period == 0:
                grants += 1
                if arm == "random":
                    pending = uniform.randrange(ACTUATORS)
                elif arm == "oracle":
                    pending = body.oracle_choice()
                elif arm == "bandit":
                    pending = agent.action()
                elif arm == "certainty":
                    pending = agent.certainty_action()
                else:
                    pending = agent.sample_action()
        record = {"life": life, "survived": body.alive, "ticks": body.ticks, "grants": grants}
        if arm in POSTERIOR_ARMS:
            record.update(_report_record(agent, truth))
            record["worst_tick_evidence"] = worst
        records.append(record)
    out = {"arm": arm, "lives": records}
    if arm == "posterior":
        out["_history"] = history
    return out


def permutation_control(history: list, ident: int) -> dict:
    """Replay the posterior arm's own recorded lives offline twice through a
    fresh posterior with one fixed random stream: once as lived, once with
    every actuator ID relabeled consistently. The inferred partition must
    transport exactly; the random stream never depends on an actuator's ID."""

    permutation = list(range(ACTUATORS))
    Random(domain_seed(ident, "permutation")).shuffle(permutation)

    def infer(relabel):
        posterior = SelfPosterior(POSTERIOR, np.random.default_rng(domain_seed(ident, "offline")))
        for kind, value, total, minimum in history:
            if kind == "birth":
                posterior.birth(total, minimum, retain=value)
            else:
                posterior.observe(None if value is None else relabel[value], total, minimum)
        return posterior.structure_report().partition

    plain = infer(list(range(ACTUATORS)))
    moved = infer(permutation)
    transported = tuple(sorted(tuple(sorted(permutation[a] for a in g)) for g in plain))
    return {"plain": [list(g) for g in plain], "relabeled": [list(g) for g in moved],
            "transports": moved == transported}


def run_identity(*, base: int, block: int, k: int, identity: int, arms=ARMS) -> dict:
    ident = identity_seed(base, block, k, identity)
    world = make_physical_world(seed=domain_seed(ident, "map"), k=k)
    donor_ident = ident + 50
    donor = make_physical_world(seed=domain_seed(donor_ident, "donor-map"), k=k)
    start = perf_counter()
    row = {"block": block, "k": k, "identity": identity, "identity_seed": ident,
           "truth": [list(g) for g in true_partition(world)],
           "donor_truth": [list(g) for g in true_partition(donor)], "arms": {}}
    for arm in arms:
        row["arms"][arm] = run_arm(arm, world, donor, ident=ident, donor_ident=donor_ident)
    history = row["arms"]["posterior"].pop("_history")
    if identity < PERMUTATION_IDENTITIES:
        row["permutation"] = permutation_control(history, ident)
    row["seconds"] = perf_counter() - start
    return row


# ------------------------------------------------------------ summaries
def _interval(values: list[float]) -> dict:
    """Mean and two-sided 95% Student-t interval over paired blocks."""

    array = np.asarray(values, dtype=float)
    centre = float(array.mean())
    if array.size < 2:
        return {"mean": centre, "low": centre, "high": centre, "n": int(array.size), "values": list(values)}
    critical = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}.get(array.size - 1, 1.96)
    half = critical * float(array.std(ddof=1)) / float(np.sqrt(array.size))
    return {"mean": centre, "low": centre - half, "high": centre + half, "n": int(array.size),
            "values": [float(v) for v in values],
            "positive": int((array > 0).sum()), "negative": int((array < 0).sum())}


def _block_metric(rows, block, arm, metric, ks=(1, 2, 3, 4, 5), lives=range(LIVES)):
    values = []
    for row in rows:
        if row["block"] != block or row["k"] not in ks:
            continue
        for record in row["arms"][arm]["lives"]:
            if record["life"] in lives:
                values.append(float(metric(record)))
    return mean(values)


def summarize(rows: list[dict], *, blocks: int, identities: int) -> dict:
    expected = {(b, k, i) for b in range(blocks) for k in range(1, 6) for i in range(identities)}
    observed = {(r["block"], r["k"], r["identity"]) for r in rows}
    if observed != expected or len(rows) != len(expected):
        raise ValueError("incomplete or duplicated survey")
    survived = lambda record: record["survived"]
    exact = lambda record: record["exact"]
    brier = lambda record: (record["confidence"] - record["exact"]) ** 2
    last = (LIVES - 1,)
    table = {}
    for k in range(1, 6):
        table[str(k)] = {}
        for arm in ARMS:
            entry = {"survival": mean(_block_metric(rows, b, arm, survived, (k,)) for b in range(blocks))}
            if arm in POSTERIOR_ARMS:
                entry["exact_after_life1"] = mean(_block_metric(rows, b, arm, exact, (k,), (0,)) for b in range(blocks))
                entry["exact_after_life6"] = mean(_block_metric(rows, b, arm, exact, (k,), last) for b in range(blocks))
                entry["mean_count_after_life6"] = mean(
                    r["arms"][arm]["lives"][-1]["resource_count"] for r in rows if r["k"] == k)
                entry["brier"] = mean(_block_metric(rows, b, arm, brier, (k,)) for b in range(blocks))
            table[str(k)][arm] = entry

    def contrast(a, b, metric, ks=CORE_KS, lives=range(LIVES)):
        return _interval([_block_metric(rows, blk, a, metric, ks, lives) - _block_metric(rows, blk, b, metric, ks, lives)
                          for blk in range(blocks)])

    contrasts = {
        "posterior_minus_random_survival_by_k": {str(k): contrast("posterior", "random", survived, (k,)) for k in range(1, 6)},
        "posterior_minus_bandit_survival_by_k": {str(k): contrast("posterior", "bandit", survived, (k,)) for k in range(1, 6)},
        "posterior_minus_yoked_survival_k1to4": contrast("posterior", "yoked", survived),
        "posterior_minus_reset_survival_lives2to6_k1to4": contrast("posterior", "reset", survived, lives=range(1, LIVES)),
        "posterior_minus_certainty_exact_life6_k1to4": contrast("posterior", "certainty", exact, lives=last),
        "posterior_minus_certainty_survival_k1to4": contrast("posterior", "certainty", survived),
        "uncritical_minus_posterior_brier_k1to4": contrast("uncritical", "posterior", brier),
        "posterior_minus_uncritical_exact_life6_k1to4": contrast("posterior", "uncritical", exact, lives=last),
        "posterior_minus_uncritical_survival_k1to4": contrast("posterior", "uncritical", survived),
    }
    calibration = {}
    for arm in ("posterior", "uncritical"):
        reports = [(rec["confidence"], rec["exact"], r["k"]) for r in rows for rec in r["arms"][arm]["lives"]]
        for name, ks in (("k1to4", CORE_KS), ("k5", (5,))):
            chosen = [(c, e) for c, e, k in reports if k in ks]
            high = [e for c, e in chosen if c >= 0.9]
            low = [e for c, e in chosen if c <= 0.1]
            calibration[f"{arm}_{name}"] = {
                "reports": len(chosen),
                "brier": mean((c - e) ** 2 for c, e in chosen),
                "mean_confidence": mean(c for c, _ in chosen),
                "accuracy": mean(e for _, e in chosen),
                "high_confidence_reports": len(high),
                "accuracy_when_confidence_at_least_0_9": mean(high) if high else None,
                "low_confidence_reports": len(low),
                "accuracy_when_confidence_at_most_0_1": mean(low) if low else None,
            }
    yoked_own_exact = mean(r["arms"]["yoked"]["lives"][-1]["exact"] for r in rows if r["k"] in CORE_KS)
    permuted = [r["permutation"]["transports"] for r in rows if "permutation" in r]
    return {"table": table, "contrasts": contrasts, "calibration": calibration,
            "yoked_exact_on_own_body_k1to4": yoked_own_exact,
            "permutation_controls": len(permuted),
            "permutation_transport_rate": mean(permuted) if permuted else None}


def grade(summary: dict) -> dict:
    """The nine locked gates, copied from the preregistration."""

    table, contrasts, cal = summary["table"], summary["contrasts"], summary["calibration"]["posterior_k1to4"]
    positive = lambda c: c["mean"] > 0.0 and c["low"] > 0.0 and c.get("positive", 0) >= 4
    counts = [table[str(k)]["posterior"]["mean_count_after_life6"] for k in range(1, 6)]
    gates = {
        "G1_discovery": all(table[str(k)]["posterior"]["exact_after_life6"] >= 0.80 for k in (1, 2, 3))
        and table["4"]["posterior"]["exact_after_life6"] >= 0.40
        and all(a < b for a, b in zip(counts, counts[1:])),
        "G2_action_uses_self": all(positive(contrasts["posterior_minus_random_survival_by_k"][str(k)]) for k in CORE_KS),
        "G3_structure_not_action_values": all(positive(contrasts["posterior_minus_bandit_survival_by_k"][str(k)]) for k in (2, 3, 4)),
        "G4_own_body": positive(contrasts["posterior_minus_yoked_survival_k1to4"])
        and summary["yoked_exact_on_own_body_k1to4"] <= 0.10,
        "G5_uncertainty_drives_discovery": positive(contrasts["posterior_minus_certainty_exact_life6_k1to4"])
        and contrasts["posterior_minus_certainty_exact_life6_k1to4"]["mean"] >= 0.30,
        "G6_persistence": positive(contrasts["posterior_minus_reset_survival_lives2to6_k1to4"]),
        "G7_calibrated_self_report": cal["brier"] <= 0.10
        and cal["accuracy_when_confidence_at_least_0_9"] is not None
        and cal["accuracy_when_confidence_at_least_0_9"] >= 0.90
        and (cal["accuracy_when_confidence_at_most_0_1"] is None or cal["accuracy_when_confidence_at_most_0_1"] <= 0.10),
        "G8_self_criticism_load_bearing": positive(contrasts["uncritical_minus_posterior_brier_k1to4"]),
        "G9_permutation": summary["permutation_transport_rate"] == 1.0,
    }
    return {"gates": gates, "all_pass": all(gates.values())}


# ----------------------------------------------------------- execution
def manifest(*, base: int, blocks: int, identities: int) -> dict:
    files = [Path(__file__), Path(__file__).with_name("self_posterior.py"),
             Path(__file__).with_name("saturation_structure.py"),
             Path(__file__).parent.parent / "island/expanded_body.py"]
    pins = {p.name: sha256(p.read_bytes()).hexdigest() for p in files}
    if PREREG.exists():
        pins[PREREG.name] = sha256(PREREG.read_bytes()).hexdigest()
    return json.loads(json.dumps({"base": base, "blocks": blocks, "identities": identities, "lives": LIVES,
                                  "arms": list(ARMS), "physics": asdict(PHYSICS), "posterior": asdict(POSTERIOR),
                                  "pins": pins}))


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":"), default=float))
    temporary.replace(path)


def run_block(*, base: int, block: int, identities: int, blocks: int, out: Path,
              ks: tuple[int, ...] = (1, 2, 3, 4, 5)) -> dict:
    """Run one block (or some of its K), resumably, in ``for k: for identity`` order."""

    man = manifest(base=base, blocks=blocks, identities=identities)
    rows = []
    if out.exists():
        saved = json.loads(out.read_text())
        if saved["manifest"] != man:
            raise ValueError("resume refused: configuration or source changed")
        rows = saved["rows"]
    done = {(r["k"], r["identity"]) for r in rows}
    for k in ks:
        for identity in range(identities):
            if (k, identity) in done:
                continue
            rows.append(run_identity(base=base, block=block, k=k, identity=identity))
            _write(out, {"manifest": man, "block": block, "rows": rows})
            print(f"block {block} K{k} identity {identity}: {rows[-1]['seconds']:.1f}s", flush=True)
    return {"manifest": man, "block": block, "rows": rows}


def merge_and_grade(paths: list[Path], *, blocks: int, identities: int, out: Path) -> dict:
    rows, manifests = [], []
    for path in paths:
        saved = json.loads(path.read_text())
        manifests.append(saved["manifest"])
        rows.extend(saved["rows"])
    if any(m != manifests[0] for m in manifests):
        raise ValueError("blocks were run under different configurations")
    summary = summarize(rows, blocks=blocks, identities=identities)
    result = {"manifest": manifests[0], "summary": summary, "grade": grade(summary)}
    _write(out, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("survey", "diagnostic", "development"), default="survey")
    parser.add_argument("--block", type=int)
    parser.add_argument("--k", type=int, help="run only this K of the block (its own progress file)")
    parser.add_argument("--identities", type=int)
    parser.add_argument("--grade", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/organism/probe80_active_self"))
    args = parser.parse_args()
    base = {"survey": SURVEY_BASE, "diagnostic": DIAGNOSTIC_BASE, "development": DEVELOPMENT_BASE}[args.mode]
    identities = args.identities or {"survey": IDENTITIES, "diagnostic": 1, "development": 4}[args.mode]
    blocks = BLOCKS if args.mode != "diagnostic" else 1
    if args.grade:
        paths = sorted(args.out_dir.glob(f"{args.mode}_block*_k*.json")) or \
            [args.out_dir / f"{args.mode}_block{b}.json" for b in range(blocks)]
        result = merge_and_grade(paths, blocks=blocks, identities=identities, out=args.out_dir / f"{args.mode}_graded.json")
        print(json.dumps({"grade": result["grade"], "calibration": result["summary"]["calibration"]}, indent=2))
        return
    todo = [args.block] if args.block is not None else range(blocks)
    for block in todo:
        ks = (args.k,) if args.k is not None else (1, 2, 3, 4, 5)
        for k in ks:
            out = args.out_dir / f"{args.mode}_block{block}_k{k}.json"
            run_block(base=base, block=block, identities=identities, blocks=blocks, out=out, ks=(k,))


if __name__ == "__main__":
    main()
