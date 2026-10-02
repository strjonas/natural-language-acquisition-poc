"""Probe77: bounded sequential randomized exploration, without simulator forks."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from random import Random
from statistics import mean, median, variance
from time import perf_counter

from homesocial.island.expanded_body import BodyState, ExpandedBody, ExpandedBodyConfig
from homesocial.organism.saturation_structure import (
    ACTUATORS, PhysicalWorld, make_physical_world, remap_small_alias, unreachable_world,
)

BASE = 3_900_000_000
DIAGNOSTIC_BASE = 3_950_000_000
CONFIG = ExpandedBodyConfig()
PREREG = Path("docs/decisions/2026-09-30-sequential-structure-preregistration.md")


def seed_for(seed: int, domain: str) -> int:
    return int.from_bytes(sha256(f"probe77:{domain}:{seed}".encode()).digest(), "big")


def estimate_effect(current_delta: float, preceding_neutral: list[float]) -> float | None:
    if len(preceding_neutral) < 3:
        return None
    return current_delta - median(preceding_neutral[-5:])


def run_episode(world: PhysicalWorld, *, identity_seed: int, episode_seed: int) -> dict:
    # Physics is private to this runner. The inference functions below receive
    # only (action, predecessor, scalar effect), never this object or its seed.
    body = ExpandedBody(world.schema(), config=replace(CONFIG, shock_probability=0.0))
    body.reset()
    birth = Random(seed_for(episode_seed, "birth"))
    scale = Random(seed_for(identity_seed, "scale"))
    shock = Random(seed_for(episode_seed, "shock"))
    actions = Random(seed_for(episode_seed, "actions"))
    body.state = BodyState(body.schema, tuple(
        birth.choice(CONFIG.birth_levels) if i in world.active_slots else 1.0
        for i in range(5)
    ))
    body.metabolic_scale = {a.name: scale.uniform(0.4, 1.6) for a in body.schema.axes}
    pending = previous = None
    neutral: list[float] = []
    records = []
    for _ in range(CONFIG.life_steps):
        before = body.state.mean_viability()
        absorbed = pending
        if absorbed is not None and world.actionable(absorbed):
            assignment = world.actuators[absorbed]
            body.grant(body.schema.axes[assignment.target].name, assignment.size)
        pending = None
        transition = body.step()
        if not transition.terminated and shock.random() < CONFIG.shock_probability:
            slot = shock.randrange(5)
            if slot in world.active_slots:
                body.state = body.state.add(body.schema.axes[slot].name, -CONFIG.shock_size)
        delta = body.state.mean_viability() - before
        if absorbed is None:
            neutral.append(delta)
        else:
            effect = estimate_effect(delta, neutral)
            if effect is not None:
                records.append({"tick": body.step_count, "action": absorbed,
                                "previous": previous, "effect": effect})
            previous = absorbed
        if body.state.viability() <= 0.0:
            break
        if body.step_count % CONFIG.help_period == 0:
            pending = actions.randrange(ACTUATORS)
    return {"episode_seed": episode_seed, "steps": body.step_count,
            "survived": body.state.viability() > 0.0 and body.step_count == CONFIG.life_steps,
            "records": records}


def infer(records: list[dict]) -> dict:
    by_action = {a: [r["effect"] for r in records if r["action"] == a]
                 for a in range(ACTUATORS)}
    active = [a for a, values in by_action.items()
              if len(values) >= 3 and median(values) > 1e-6]
    adjacency = {a: {a} for a in active}
    supports, suppressed = [], []
    # All predictor means are trained here, including means used on test data.
    baselines = {a: mean(v) if v else 0.0 for a, v in by_action.items()}
    conditional = {}
    for b in active:
        for a in active:
            if a == b:
                continue
            paired = [r["effect"] for r in records if r["action"] == b and r["previous"] == a]
            other = [r["effect"] for r in records if r["action"] == b
                     and r["previous"] is not None and r["previous"] != a]
            if min(len(paired), len(other)) < 3:
                continue
            supports.append([a, b])
            conditional[f"{a},{b}"] = mean(paired)
            difference = mean(other) - mean(paired)
            se = (max(variance(paired), 1e-12) / len(paired)
                  + max(variance(other), 1e-12) / len(other)) ** 0.5
            if difference > max(1e-6, 2 * se):
                adjacency[a].add(b)
                adjacency[b].add(a)
                suppressed.append([a, b])
    groups, remaining = [], set(active)
    while remaining:
        seen, todo = set(), [min(remaining)]
        while todo:
            a = todo.pop()
            if a not in seen:
                seen.add(a)
                todo.extend(adjacency[a] - seen)
        groups.append(sorted(seen))
        remaining -= seen
    return {"groups": sorted(groups), "active": active, "count": len(groups),
            "supports": supports, "suppressed": suppressed,
            "baselines": baselines, "conditional": conditional,
            "total_dimension": "unresolved"}


def actual_groups(world: PhysicalWorld) -> list[list[int]]:
    return sorted(sorted(a for a in range(ACTUATORS)
                         if world.actionable(a) and world.actuators[a].target == slot)
                  for slot in world.active_slots if slot not in world.blocked_slots)


def score(inferred: dict, world: PhysicalWorld) -> dict:
    truth = actual_groups(world)
    truth_active = {a for group in truth for a in group}
    predicted = {a for group in inferred["groups"] for a in group}
    confusion = dict(tp=0, tn=0, fp=0, fn=0)
    for a in sorted(truth_active):
        for b in sorted(truth_active):
            if a >= b:
                continue
            same = any(a in g and b in g for g in truth)
            both = a in predicted and b in predicted
            guessed = any(a in g and b in g for g in inferred["groups"])
            # Missing active IDs are errors even on a true nonoverlap pair.
            key = ("tp" if guessed and both else "fn") if same else (
                "tn" if both and not guessed else "fp")
            confusion[key] += 1
    recall = confusion["tp"] / max(1, confusion["tp"] + confusion["fn"])
    specificity = confusion["tn"] / max(1, confusion["tn"] + confusion["fp"])
    return {"exact": inferred["groups"] == truth, "count": inferred["count"],
            "count_exact": inferred["count"] == len(truth),
            "balanced_accuracy": (recall + specificity) / 2,
            "active_recall": len(truth_active & predicted) / len(truth_active),
            "active_precision": len(truth_active & predicted) / max(1, len(predicted)),
            "supported_fraction": len(inferred["supports"]) / max(1, len(truth_active) * (len(truth_active) - 1)),
            "confusion": confusion}


def prediction_score(model: dict, records: list[dict]) -> dict:
    base_error = learned_error = 0.0
    for r in records:
        baseline = model["baselines"][r["action"]]
        predicted = model["conditional"].get(f"{r['previous']},{r['action']}", baseline)
        base_error += (r["effect"] - baseline) ** 2
        learned_error += (r["effect"] - predicted) ** 2
    return {"n": len(records), "baseline_sse": base_error, "model_sse": learned_error}


def permute(records: list[dict], permutation: list[int]) -> list[dict]:
    return [{**r, "action": permutation[r["action"]],
             "previous": None if r["previous"] is None else permutation[r["previous"]]}
            for r in records]


def experiment(*, block: int, k: int, identity: int, base: int) -> dict:
    identity_seed = base + block * 2_000_000 + k * 100_000 + identity * 100
    world = make_physical_world(seed=seed_for(identity_seed, "map"), k=k)
    def episodes(physical, offsets):
        return [run_episode(physical, identity_seed=identity_seed,
                            episode_seed=identity_seed + offset) for offset in offsets]
    dev = episodes(world, range(6))
    heldout = episodes(world, range(6, 8))
    test_records = [r for e in heldout for r in e["records"]]
    budgets = {}
    for budget in (1, 6):
        records = [r for e in dev[:budget] for r in e["records"]]
        model = infer(records)
        shuffled = [dict(r) for r in records]
        effects = [r["effect"] for r in records]
        Random(seed_for(identity_seed, f"shuffle{budget}")).shuffle(effects)
        for r, value in zip(shuffled, effects, strict=True):
            r["effect"] = value
        permutation = list(range(ACTUATORS))
        Random(seed_for(identity_seed, "permutation")).shuffle(permutation)
        transported = sorted(sorted(permutation[a] for a in g) for g in model["groups"])
        budgets[str(budget)] = {
            **score(model, world), "groups": model["groups"],
            "shuffled": score(infer(shuffled), world),
            "permutation_exact": infer(permute(records, permutation))["groups"] == transported,
            "heldout": prediction_score(model, test_records),
            "actual_ticks": sum(e["steps"] for e in dev[:budget]),
            "observations": len(records),
            "survival": mean(e["survived"] for e in dev[:budget]),
        }
    challenges = {}
    challenge_episodes = []
    if k >= 2:
        moved, moved_id = remap_small_alias(world, seed=seed_for(identity_seed, "remap"))
        lives = episodes(moved, range(10, 16))
        challenge_episodes.extend(lives)
        model = infer([r for e in lives for r in e["records"]])
        challenges["remap"] = {**score(model, moved), "moved_id": moved_id,
                               "stale_wrong": budgets["6"]["groups"] != actual_groups(moved)}
    if k <= 4:
        unreachable = unreachable_world(world, seed=seed_for(identity_seed, "unreachable"))
        lives = episodes(unreachable, range(20, 26))
        challenge_episodes.extend(lives)
        model = infer([r for e in lives for r in e["records"]])
        challenges["unreachable"] = {**score(model, unreachable),
                                     "total_dimension": model["total_dimension"]}
    return {"block": block, "k": k, "identity": identity, "identity_seed": identity_seed,
            "budgets": budgets, "challenges": challenges,
            "total_ticks": sum(e["steps"] for e in dev + heldout + challenge_episodes),
            "episodes": dev + heldout + challenge_episodes}


def summarize(rows: list[dict], *, blocks: int, identities: int) -> dict:
    expected = {(b, k, i) for b in range(blocks) for k in range(1, 6) for i in range(identities)}
    observed = {(r["block"], r["k"], r["identity"]) for r in rows}
    if observed != expected or len(rows) != len(expected):
        raise ValueError("incomplete/duplicate survey")
    output = {}
    for budget in ("1", "6"):
        by_k = {}
        for k in range(1, 6):
            chosen = [r for r in rows if r["k"] == k]
            values = [r["budgets"][budget] for r in chosen]
            by_k[str(k)] = {
                key: mean(v[key] for v in values) for key in (
                    "exact", "count", "count_exact", "balanced_accuracy", "active_recall", "active_precision",
                    "supported_fraction", "actual_ticks", "observations", "survival")}
            by_k[str(k)]["shuffled_exact"] = mean(v["shuffled"]["exact"] for v in values)
            by_k[str(k)]["per_block_exact"] = [mean(r["budgets"][budget]["exact"]
                for r in chosen if r["block"] == b) for b in range(blocks)]
        reductions = []
        for b in range(blocks):
            scores = [r["budgets"][budget]["heldout"] for r in rows if r["block"] == b]
            baseline = sum(s["baseline_sse"] for s in scores)
            reductions.append((baseline - sum(s["model_sse"] for s in scores)) / baseline if baseline else 0.0)
        remap = {str(k): mean(r["challenges"]["remap"]["exact"] for r in rows if r["k"] == k)
                 for k in range(2, 6)}
        stale = {str(k): mean(r["challenges"]["remap"]["stale_wrong"] for r in rows if r["k"] == k)
                 for k in range(2, 6)}
        gates = {
            "G1_partition": all(v["exact"] >= .8 and sum(x >= .7 for x in v["per_block_exact"]) >= 4 for v in by_k.values()),
            "G2_structure": all(by_k[str(k)]["balanced_accuracy"] >= .8 for k in range(2, 6)) and all(by_k[str(k)]["count"] < by_k[str(k+1)]["count"] for k in range(1, 5)),
            "G3_controls": all(v["exact"] - v["shuffled_exact"] >= .2 for v in by_k.values()) and all(r["budgets"][budget]["permutation_exact"] for r in rows),
            "G4_prediction": sum(x > 0 for x in reductions) >= 4 and mean(reductions) >= .1,
            "G5_remapping": all(x >= .8 for x in remap.values()) and all(x >= .8 for x in stale.values()),
        } if (blocks, identities) == (5, 20) else None
        output[budget] = {"by_k": by_k, "prediction_reduction_per_block": reductions,
                          "remap_exact": remap, "stale_wrong": stale,
                          "gates": gates, "all_pass": all(gates.values()) if gates else None}
    return output


def run(*, diagnostic: bool, out: Path) -> dict:
    blocks, identities = (1, 1) if diagnostic else (5, 20)
    base = DIAGNOSTIC_BASE if diagnostic else BASE
    source = Path(__file__)
    files = [source, source.with_name("saturation_structure.py"),
             source.parent.parent / "island/expanded_body.py", PREREG]
    pins = {p.name: sha256(p.read_bytes()).hexdigest() for p in files}
    manifest = {"diagnostic": diagnostic, "blocks": blocks, "identities": identities,
                "base": base, "configuration": asdict(CONFIG), "pins": pins}
    manifest = json.loads(json.dumps(manifest))
    rows = []
    if out.exists():
        saved = json.loads(out.read_text())
        if saved["manifest"] != manifest:
            raise ValueError("resume configuration or source changed")
        rows = saved["rows"]
        if saved["complete"]:
            summarize(rows, blocks=blocks, identities=identities)
            return saved
    done = {(r["block"], r["k"], r["identity"]) for r in rows}
    expected = {(b, k, i) for b in range(blocks) for k in range(1, 6) for i in range(identities)}
    if len(done) != len(rows) or not done <= expected:
        raise ValueError("invalid completed world units")
    start = perf_counter()
    def save(complete=False):
        payload = {"manifest": manifest, "complete": complete, "rows": rows,
                   "total_worlds": len(expected), "elapsed_this_run": perf_counter() - start}
        if complete:
            payload["summary"] = summarize(rows, blocks=blocks, identities=identities)
        out.parent.mkdir(parents=True, exist_ok=True)
        temporary = out.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, separators=(",", ":")))
        temporary.replace(out)
        return payload
    for b in range(blocks):
        for k in range(1, 6):
            for i in range(identities):
                if (b, k, i) in done:
                    continue
                rows.append(experiment(block=b, k=k, identity=i, base=base))
                save()
            print(f"block {b+1}/{blocks} K{k}: {len(rows)}/{len(expected)} worlds", flush=True)
    return save(True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("runs/organism/probe77_sequential_structure/survey.json"))
    args = parser.parse_args()
    result = run(diagnostic=args.diagnostic, out=args.out)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
