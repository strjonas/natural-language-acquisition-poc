"""Probe76: identify reachable restoration groups from scalar interventions.

This is a forked, shock-free identification instrument, not an online organism.
Physics and audit identities are kept outside ``infer_groups``: its only input
is scalar branch transcripts with the same opaque actuator catalog in all worlds.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from random import Random
import time
from typing import Sequence

from homesocial.island.expanded_body import (
    EXPANDED_BODY_SCHEMA,
    BodySchema,
    BodyState,
    ExpandedBody,
    ExpandedBodyConfig,
)

ACTUATORS = 10
TOLERANCE = 1e-8
BRANCH_TICKS = 13
BRANCHES_PER_CONTEXT = 1 + 2 * ACTUATORS + ACTUATORS**2
SURVEY_BASE = 3_800_000_000
DIAGNOSTIC_BASE = 3_850_000_000
BLOCK_STRIDE = 2_000_000
WORLD_STRIDE = 100_000
OFFSETS = {
    "mapping": 0,
    "development": 1_000,
    "held_out": 10_000,
    "permutation": 20_000,
    "shuffled": 30_000,
    "remap_development": 40_000,
    "remap_held_out": 41_000,
    "unreachable": 50_000,
}


@dataclass(frozen=True)
class SurveyConfig:
    diagnostic: bool = False
    tolerance: float = TOLERANCE

    def __post_init__(self) -> None:
        if self.tolerance != TOLERANCE:
            raise ValueError("The preregistered numerical tolerance is locked.")

    @property
    def blocks(self) -> int:
        return 1 if self.diagnostic else 5

    @property
    def contexts(self) -> int:
        return 2 if self.diagnostic else 40

    @property
    def seed_base(self) -> int:
        return DIAGNOSTIC_BASE if self.diagnostic else SURVEY_BASE

    def manifest(self) -> dict:
        manifest = {
            **asdict(self),
            "blocks": self.blocks,
            "development_contexts": self.contexts,
            "held_out_contexts": self.contexts,
            "unreachable_contexts": self.contexts,
            "seed_base": self.seed_base,
            "block_stride": BLOCK_STRIDE,
            "world_stride": WORLD_STRIDE,
            "offsets": OFFSETS,
            "actuators": ACTUATORS,
            "physical_config": asdict(ExpandedBodyConfig(shock_probability=0.0)),
            "branch_ticks": BRANCH_TICKS,
            "unique_branches_per_context": BRANCHES_PER_CONTEXT,
        }
        return json.loads(json.dumps(manifest))


@dataclass(frozen=True)
class Actuator:
    """Scorer/instrument metadata. Never supplied to inference."""

    target: int
    size: str


@dataclass(frozen=True)
class PhysicalWorld:
    active_slots: tuple[int, ...]
    actuators: tuple[Actuator, ...]
    blocked_slots: tuple[int, ...] = ()

    def actionable(self, actuator: int) -> bool:
        target = self.actuators[actuator].target
        return target in self.active_slots and target not in self.blocked_slots

    def schema(self) -> BodySchema:
        return BodySchema(
            tuple(
                axis
                if slot in self.active_slots
                else replace(axis, resting_rate=0.0, moving_rate=0.0)
                for slot, axis in enumerate(EXPANDED_BODY_SCHEMA.axes)
            )
        )


def make_physical_world(*, seed: int, k: int) -> PhysicalWorld:
    if not 1 <= k <= len(EXPANDED_BODY_SCHEMA):
        raise ValueError("K must be in the preregistered 1..5 sweep.")
    rng = Random(domain_seed(seed, "mapping"))
    active = tuple(sorted(rng.sample(range(len(EXPANDED_BODY_SCHEMA)), k)))
    catalog = [
        Actuator(slot, size)
        for slot in range(len(EXPANDED_BODY_SCHEMA))
        for size in ("small", "large")
    ]
    rng.shuffle(catalog)
    return PhysicalWorld(active, tuple(catalog))


def domain_seed(context_seed: int, domain: str) -> int:
    """Separate physical RNG domains without additive-offset cross-block aliases."""

    return int.from_bytes(hashlib.sha256(f"probe76:{domain}:{context_seed}".encode()).digest(), "big")


def remap_small_alias(world: PhysicalWorld, *, seed: int) -> tuple[PhysicalWorld, int]:
    if len(world.active_slots) < 2:
        raise ValueError("A physical remapping needs two active targets.")
    rng = Random(domain_seed(seed, "remapping"))
    options = [
        actuator
        for actuator, assignment in enumerate(world.actuators)
        if assignment.size == "small" and world.actionable(actuator)
    ]
    moved = rng.choice(options)
    assignment = world.actuators[moved]
    target = rng.choice([slot for slot in world.active_slots if slot != assignment.target])
    catalog = list(world.actuators)
    catalog[moved] = replace(assignment, target=target)
    return replace(world, actuators=tuple(catalog)), moved


def unreachable_world(world: PhysicalWorld, *, seed: int) -> PhysicalWorld:
    inactive = [slot for slot in range(len(EXPANDED_BODY_SCHEMA)) if slot not in world.active_slots]
    if not inactive:
        raise ValueError("The challenge needs an additional inactive slot.")
    hidden = Random(domain_seed(seed, "unreachable_selection")).choice(inactive)
    return replace(
        world,
        active_slots=tuple(sorted((*world.active_slots, hidden))),
        blocked_slots=tuple(sorted((*world.blocked_slots, hidden))),
    )


@dataclass(frozen=True)
class Sensation:
    mean: float
    minimum: float


@dataclass(frozen=True)
class ContextTranscript:
    """Only sensations indexed by the organism's own intervention choices."""

    empty: Sensation
    first: tuple[Sensation, ...]
    second: tuple[Sensation, ...]
    joint: tuple[tuple[Sensation, ...], ...]

    def interaction(self, first: int, second: int) -> float:
        return (
            self.joint[first][second].mean
            - self.first[first].mean
            - self.second[second].mean
            + self.empty.mean
        )

    def outputs(self) -> list[Sensation]:
        return [self.empty, *self.first, *self.second, *(value for row in self.joint for value in row)]

    @classmethod
    def from_outputs(cls, outputs: Sequence[Sensation]) -> ContextTranscript:
        if len(outputs) != BRANCHES_PER_CONTEXT:
            raise ValueError("A transcript requires all 121 branch outputs.")
        start = 1 + 2 * ACTUATORS
        return cls(
            outputs[0],
            tuple(outputs[1 : 1 + ACTUATORS]),
            tuple(outputs[1 + ACTUATORS : start]),
            tuple(
                tuple(outputs[start + row * ACTUATORS : start + (row + 1) * ACTUATORS])
                for row in range(ACTUATORS)
            ),
        )


def branch_sensations(
    world: PhysicalWorld,
    *,
    seed: int,
    first: int | None = None,
    second: int | None = None,
    schema: BodySchema | None = None,
    trace: bool = False,
) -> Sensation | tuple[Sensation, ...]:
    """Request6/12, absorb7/13, then deplete and read after each full tick."""

    body = ExpandedBody(
        schema or world.schema(),
        config=ExpandedBodyConfig(shock_probability=0.0),
        seed=seed,
    )
    # Preserve ExpandedBody's distributions and axis draw order, replacing its
    # additive RNG salts before they can alias between independent blocks.
    birth_rng = Random(domain_seed(seed, "birth"))
    scale_rng = Random(domain_seed(seed, "scale"))
    births = tuple(birth_rng.choice(body.config.birth_levels) for _ in body.schema.axes)
    body.state = BodyState(body.schema, tuple(value if slot in world.active_slots else 1.0 for slot, value in enumerate(births)))
    spread = body.config.metabolic_spread
    body.metabolic_scale = {axis.name: scale_rng.uniform(1.0 - spread, 1.0 + spread) for axis in body.schema.axes}
    body._shock_rng = Random(domain_seed(seed, "shock"))
    pending = None
    sensations = []
    for tick in range(1, BRANCH_TICKS + 1):
        if pending is not None and world.actionable(pending):
            assignment = world.actuators[pending]
            body.grant(body.schema.axes[assignment.target].name, assignment.size)
        pending = None
        transition = body.step(moved=False)
        if transition.terminated or body.state.viability() <= 0.0:
            raise AssertionError("The locked 13-tick instrument reached a dead body.")
        sensation = Sensation(body.state.mean_viability(), body.state.viability())
        if trace:
            sensations.append(sensation)
        if tick == 6:
            pending = first
        elif tick == 12:
            pending = second
    return tuple(sensations) if trace else sensation


def collect_context(world: PhysicalWorld, *, seed: int) -> ContextTranscript:
    schema = world.schema()

    def branch(first: int | None = None, second: int | None = None) -> Sensation:
        return branch_sensations(world, seed=seed, first=first, second=second, schema=schema)

    return ContextTranscript(
        branch(),
        tuple(branch(first=actuator) for actuator in range(ACTUATORS)),
        tuple(branch(second=actuator) for actuator in range(ACTUATORS)),
        tuple(
            tuple(branch(first=first, second=second) for second in range(ACTUATORS))
            for first in range(ACTUATORS)
        ),
    )


def collect_contexts(world: PhysicalWorld, *, seed_base: int, contexts: int) -> tuple[ContextTranscript, ...]:
    return tuple(collect_context(world, seed=seed_base + index) for index in range(contexts))


@dataclass(frozen=True)
class GroupInference:
    active: tuple[int, ...]
    groups: tuple[tuple[int, ...], ...]
    representatives: tuple[int, ...]

    @property
    def count(self) -> int:
        return len(self.groups)

    def overlaps(self, first: int, second: int) -> bool:
        return any(first in group and second in group for group in self.groups)


def infer_groups(transcripts: Sequence[ContextTranscript]) -> GroupInference:
    """No axis metadata, true states, dimensions, portions or target map input."""

    if not transcripts:
        raise ValueError("Inference requires developmental transcripts.")
    active = tuple(
        actuator
        for actuator in range(ACTUATORS)
        if any(
            max(context.first[actuator].mean, context.second[actuator].mean)
            - context.empty.mean > TOLERANCE
            for context in transcripts
        )
    )
    edges = {actuator: set() for actuator in active}
    for first in active:
        for second in active:
            if any(
                min(context.interaction(first, second), context.interaction(second, first))
                < -TOLERANCE
                for context in transcripts
            ):
                edges[first].add(second)
                edges[second].add(first)
    unseen = set(active)
    groups = []
    while unseen:
        frontier = [min(unseen)]
        group = set()
        while frontier:
            actuator = frontier.pop()
            if actuator in group:
                continue
            group.add(actuator)
            frontier.extend(edges[actuator] - group)
        unseen.difference_update(group)
        groups.append(tuple(sorted(group)))
    groups = tuple(sorted(groups))

    def mean_effect(actuator: int) -> float:
        return sum(
            context.first[actuator].mean + context.second[actuator].mean - 2 * context.empty.mean
            for context in transcripts
        ) / (2 * len(transcripts))

    representatives = tuple(max(group, key=lambda actuator: (mean_effect(actuator), -actuator)) for group in groups)
    return GroupInference(active, groups, representatives)


def truth_partition(world: PhysicalWorld) -> tuple[tuple[int, ...], ...]:
    return tuple(sorted(
        tuple(actuator for actuator in range(ACTUATORS) if world.actionable(actuator) and world.actuators[actuator].target == target)
        for target in world.active_slots
        if target not in world.blocked_slots
    ))


def partition_matches(inference: GroupInference, world: PhysicalWorld) -> bool:
    active = tuple(actuator for actuator in range(ACTUATORS) if world.actionable(actuator))
    return inference.active == active and inference.groups == truth_partition(world)


def score_overlap(inference: GroupInference, transcripts: Sequence[ContextTranscript]) -> dict:
    confusion = {"true_positive": 0, "true_negative": 0, "false_positive": 0, "false_negative": 0}
    for actuator in range(ACTUATORS):
        for representative in inference.representatives:
            observed = any(
                min(context.interaction(actuator, representative), context.interaction(representative, actuator))
                < -TOLERANCE
                for context in transcripts
            )
            predicted = inference.overlaps(actuator, representative)
            key = "true_positive" if observed and predicted else "false_negative" if observed else "false_positive" if predicted else "true_negative"
            confusion[key] += 1
    tp, tn, fp, fn = (confusion[key] for key in ("true_positive", "true_negative", "false_positive", "false_negative"))
    accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn else 0.0
    positive_recall = tp / (tp + fn) if tp + fn else 0.0
    negative_recall = tn / (tn + fp) if tn + fp else 0.0
    return {
        **confusion,
        "accuracy": accuracy,
        "occluding_recall": positive_recall,
        "non_occluding_recall": negative_recall,
        "passes": accuracy >= 0.95 and positive_recall >= 0.90 and negative_recall >= 0.90,
    }


def relabel_transcripts(transcripts: Sequence[ContextTranscript], permutation: Sequence[int]) -> tuple[ContextTranscript, ...]:
    if sorted(permutation) != list(range(ACTUATORS)):
        raise ValueError("ID relabeling must be a complete permutation.")
    inverse = [permutation.index(actuator) for actuator in range(ACTUATORS)]
    return tuple(ContextTranscript(
        context.empty,
        tuple(context.first[old] for old in inverse),
        tuple(context.second[old] for old in inverse),
        tuple(tuple(context.joint[a][b] for b in inverse) for a in inverse),
    ) for context in transcripts)


def shuffled_transcripts(transcripts: Sequence[ContextTranscript], *, seed: int) -> tuple[ContextTranscript, ...]:
    shuffled = []
    for index, context in enumerate(transcripts):
        outputs = context.outputs()
        Random(domain_seed(seed + index, "shuffled_pairing")).shuffle(outputs)
        shuffled.append(ContextTranscript.from_outputs(outputs))
    return tuple(shuffled)


def _permutation_control(inference: GroupInference, development, held_out, *, seed: int) -> dict:
    permutation = list(range(ACTUATORS))
    Random(domain_seed(seed, "id_permutation")).shuffle(permutation)
    fitted = infer_groups(relabel_transcripts(development, permutation))
    expected_active = tuple(sorted(permutation[actuator] for actuator in inference.active))
    expected_groups = tuple(sorted(tuple(sorted(permutation[actuator] for actuator in group)) for group in inference.groups))
    return {
        "old_to_new": permutation,
        "count": fitted.count,
        "transport_exact": fitted.active == expected_active and fitted.groups == expected_groups and fitted.count == inference.count,
        "held_out_prediction": score_overlap(fitted, relabel_transcripts(held_out, permutation)),
    }


def _raw_transcripts(transcripts) -> list:
    return [[[sensation.mean, sensation.minimum] for sensation in context.outputs()] for context in transcripts]


def _physics_audit(world: PhysicalWorld) -> dict:
    return {"active_slots": world.active_slots, "blocked_slots": world.blocked_slots, "actuators": [asdict(actuator) for actuator in world.actuators], "truth_partition": truth_partition(world)}


def run_world(*, block: int, k: int, config: SurveyConfig) -> dict:
    started = time.monotonic()
    base = config.seed_base + block * BLOCK_STRIDE + (k - 1) * WORLD_STRIDE
    world = make_physical_world(seed=base + OFFSETS["mapping"], k=k)
    development = collect_contexts(world, seed_base=base + OFFSETS["development"], contexts=config.contexts)
    held_out = collect_contexts(world, seed_base=base + OFFSETS["held_out"], contexts=config.contexts)
    inference = infer_groups(development)
    shuffled = infer_groups(shuffled_transcripts(development, seed=base + OFFSETS["shuffled"]))
    record = {
        "block": block, "k": k, "seed_base": base,
        "inference": {**asdict(inference), "count": inference.count},
        "count_exact": inference.count == k,
        "partition_exact": partition_matches(inference, world),
        "prediction": score_overlap(inference, held_out),
        "permutation": _permutation_control(inference, development, held_out, seed=base + OFFSETS["permutation"]),
        "shuffled": {**asdict(shuffled), "count": shuffled.count, "partition_exact": partition_matches(shuffled, world)},
        "audit": _physics_audit(world),
        "raw": {"development": _raw_transcripts(development), "held_out": _raw_transcripts(held_out)},
        "remap": None, "unreachable": None,
    }
    context_sets = 2
    if k >= 2:
        remapped, moved = remap_small_alias(world, seed=base + OFFSETS["remap_development"])
        remap_dev = collect_contexts(remapped, seed_base=base + OFFSETS["remap_development"], contexts=config.contexts)
        remap_test = collect_contexts(remapped, seed_base=base + OFFSETS["remap_held_out"], contexts=config.contexts)
        updated = infer_groups(remap_dev)
        old_target = world.actuators[moved].target
        new_target = remapped.actuators[moved].target
        representatives = [representative for representative in inference.representatives if world.actuators[representative].target in (old_target, new_target)]
        mismatches = []
        for representative in representatives:
            observed = any(min(context.interaction(moved, representative), context.interaction(representative, moved)) < -TOLERANCE for context in remap_test)
            if observed != inference.overlaps(moved, representative):
                mismatches.append(representative)
        record["remap"] = {
            "moved_actuator": moved, "old_target": old_target, "new_target": new_target,
            "stale_mismatch_representatives": mismatches,
            "stale_invalidated": bool(mismatches),
            "partition_exact": partition_matches(updated, remapped),
            "inference": {**asdict(updated), "count": updated.count},
            "prediction": score_overlap(updated, remap_test),
            "audit": _physics_audit(remapped),
            "raw": {"development": _raw_transcripts(remap_dev), "held_out": _raw_transcripts(remap_test)},
        }
        context_sets += 2
    if k <= 4:
        hidden = unreachable_world(world, seed=base + OFFSETS["unreachable"])
        hidden_dev = collect_contexts(hidden, seed_base=base + OFFSETS["unreachable"], contexts=config.contexts)
        reachable = infer_groups(hidden_dev)
        record["unreachable"] = {
            "discovered_reachable_count": reachable.count,
            "actual_reachable_count": k,
            "actual_bodily_count": len(hidden.active_slots),
            "total_dimension_status": "unresolved",
            "partition_exact": partition_matches(reachable, hidden),
            "audit": _physics_audit(hidden),
            "raw": {"development": _raw_transcripts(hidden_dev)},
        }
        context_sets += 1
    contexts = context_sets * config.contexts
    record["budget"] = {
        "physical_contexts": contexts,
        "unique_branches": contexts * BRANCHES_PER_CONTEXT,
        "actual_ticks": contexts * BRANCHES_PER_CONTEXT * BRANCH_TICKS,
        "four_branch_contrasts": contexts * ACTUATORS**2,
        "reused_control_references": contexts * (4 * ACTUATORS**2 - BRANCHES_PER_CONTEXT),
    }
    record["elapsed_seconds"] = time.monotonic() - started
    return record


def score_gates(records: Sequence[dict]) -> dict:
    if {(record["block"], record["k"]) for record in records} != {(block, k) for block in range(5) for k in range(1, 6)} or len(records) != 25:
        raise ValueError("Locked gates require the complete five-block, five-K survey.")
    by_k = {k: [record for record in records if record["k"] == k] for k in range(1, 6)}
    counts = [sum(record["inference"]["count"] for record in by_k[k]) / 5 for k in range(1, 6)]
    gates = {
        "G1_count": all(sum(record["count_exact"] for record in cells) >= 4 for cells in by_k.values()) and all(right > left for left, right in zip(counts, counts[1:])),
        "G2_partition": all(sum(record["partition_exact"] for record in cells) >= 4 for cells in by_k.values()),
        "G3_prediction": all(sum(record["prediction"]["passes"] for record in cells) >= 4 for cells in by_k.values()),
        "G4_permutation": all(record["permutation"]["transport_exact"] for record in records),
        "G5_shuffled_pairing": all(sum(not record["shuffled"]["partition_exact"] for record in cells) >= 4 for cells in by_k.values()),
        "G6_moved_mapping": all(sum(record["remap"]["stale_invalidated"] and record["remap"]["partition_exact"] for record in by_k[k]) >= 4 for k in range(2, 6)),
    }
    return {"gates": gates, "all_pass": all(gates.values()), "mean_recovered_counts": counts}


def source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[3]
    paths = (
        "src/homesocial/organism/saturation_structure.py",
        "src/homesocial/island/expanded_body.py",
        "docs/decisions/2026-09-30-saturation-structure-preregistration.md",
    )
    return {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths}


def validate_seed_isolation() -> dict:
    """Audit all production/diagnostic contexts and randomization domains.

    Forks intentionally share a physical context's draws. Distinct contexts or
    control domains must not share a derived RNG seed, even across diagnostics.
    """

    context_seeds = {}
    derived_seeds = {}

    def register(raw_seed: int, domain: str, identity: str) -> None:
        derived = domain_seed(raw_seed, domain)
        if derived in derived_seeds:
            raise ValueError(f"Derived RNG seed collision: {identity} and {derived_seeds[derived]}.")
        derived_seeds[derived] = identity

    for config in (SurveyConfig(), SurveyConfig(diagnostic=True)):
        for block in range(config.blocks):
            for k in range(1, 6):
                base = config.seed_base + block * BLOCK_STRIDE + (k - 1) * WORLD_STRIDE
                prefix = f"diagnostic={config.diagnostic}/block={block}/K={k}"
                for offset, domain in (("mapping", "mapping"), ("permutation", "id_permutation")):
                    register(base + OFFSETS[offset], domain, f"{prefix}/{domain}")
                if k >= 2:
                    register(base + OFFSETS["remap_development"], "remapping", f"{prefix}/remapping")
                if k <= 4:
                    register(base + OFFSETS["unreachable"], "unreachable_selection", f"{prefix}/unreachable_selection")
                for index in range(config.contexts):
                    register(base + OFFSETS["shuffled"] + index, "shuffled_pairing", f"{prefix}/shuffled/{index}")
                sets = ["development", "held_out"]
                if k >= 2:
                    sets += ["remap_development", "remap_held_out"]
                if k <= 4:
                    sets += ["unreachable"]
                for context_set in sets:
                    for index in range(config.contexts):
                        raw_seed = base + OFFSETS[context_set] + index
                        identity = f"{prefix}/{context_set}/{index}"
                        if raw_seed in context_seeds:
                            raise ValueError(f"Physical context seed collision: {identity} and {context_seeds[raw_seed]}.")
                        context_seeds[raw_seed] = identity
                        for domain in ("birth", "scale", "shock"):
                            register(raw_seed, domain, f"{identity}/{domain}")
    return {"physical_context_seeds": len(context_seeds), "derived_domain_seeds": len(derived_seeds), "fork_reuse_intentional": True}


def _atomic_json(path: Path, artifact: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(artifact, separators=(",", ":")) + "\n")
    temporary.replace(path)


def run_survey(*, output: Path, config: SurveyConfig) -> dict:
    seed_audit = validate_seed_isolation()
    pins = {"configuration": config.manifest(), "source_hashes": source_hashes()}
    if output.exists():
        artifact = json.loads(output.read_text())
        if artifact.get("pins") != pins:
            raise ValueError("Resume configuration/source mismatch; use a different output file.")
    else:
        artifact = {"probe": 76, "pins": pins, "seed_audit": seed_audit, "records": [], "complete": False, "gates_evaluated": False}
    completed = {(record["block"], record["k"]) for record in artifact["records"]}
    if len(completed) != len(artifact["records"]):
        raise ValueError("Resume contains duplicate completed worlds.")
    expected = {(block, k) for block in range(config.blocks) for k in range(1, 6)}
    if not completed <= expected:
        raise ValueError("Resume contains unexpected worlds.")
    for block in range(config.blocks):
        for k in range(1, 6):
            if (block, k) in completed:
                continue
            if source_hashes() != pins["source_hashes"]:
                raise ValueError("Source changed during the survey; refusing to mix implementations.")
            record = run_world(block=block, k=k, config=config)
            artifact["records"].append(record)
            _atomic_json(output, artifact)
            print(f"block={block} K={k} count={record['inference']['count']} partition={record['partition_exact']} elapsed={record['elapsed_seconds']:.2f}s", flush=True)
    artifact["complete"] = True
    artifact["gates_evaluated"] = not config.diagnostic
    if not config.diagnostic:
        artifact.update(score_gates(artifact["records"]))
    artifact["budget"] = {key: sum(record["budget"][key] for record in artifact["records"]) for key in artifact["records"][0]["budget"]}
    artifact["elapsed_seconds"] = sum(record["elapsed_seconds"] for record in artifact["records"])
    _atomic_json(output, artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", "--diagnostic2contexts", action="store_true", help="Separate diagnostic band: one block, two contexts per set, no locked gates.")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    config = SurveyConfig(diagnostic=args.diagnostic)
    output = args.out or Path("runs/organism/probe76_saturation_structure") / ("diagnostic.json" if args.diagnostic else "survey.json")
    artifact = run_survey(output=output, config=config)
    print(json.dumps({"complete": artifact["complete"], "gates_evaluated": artifact["gates_evaluated"], "gates": artifact.get("gates"), "elapsed_seconds": artifact["elapsed_seconds"], "budget": artifact["budget"]}), flush=True)


if __name__ == "__main__":
    main()
