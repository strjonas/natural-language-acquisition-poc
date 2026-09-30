"""Clock, information-boundary and causal guards for the Probe76 instrument."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, replace
import json
from random import Random

import pytest

from homesocial.island.expanded_body import EXPANDED_BODY_SCHEMA, ExpandedBodyConfig
from homesocial.organism.saturation_structure import (
    ACTUATORS,
    BLOCK_STRIDE,
    BRANCHES_PER_CONTEXT,
    BRANCH_TICKS,
    ContextTranscript,
    DIAGNOSTIC_BASE,
    OFFSETS,
    SURVEY_BASE,
    Sensation,
    SurveyConfig,
    TOLERANCE,
    WORLD_STRIDE,
    branch_sensations,
    collect_context,
    domain_seed,
    infer_groups,
    make_physical_world,
    partition_matches,
    relabel_transcripts,
    remap_small_alias,
    run_survey,
    score_gates,
    score_overlap,
    shuffled_transcripts,
    truth_partition,
    unreachable_world,
)


def test_fixed_catalog_and_five_slot_denominator_survive_dimension_changes():
    for k in range(1, 6):
        world = make_physical_world(seed=91 + k, k=k)
        assert len(world.actuators) == ACTUATORS
        assert len(world.schema()) == 5
        assert len(world.active_slots) == k
        assert sorted((assignment.target, assignment.size) for assignment in world.actuators) == [
            (slot, size) for slot in range(5) for size in ("large", "small")
        ]
        for slot, axis in enumerate(world.schema().axes):
            expected = EXPANDED_BODY_SCHEMA.axes[slot]
            if slot in world.active_slots:
                assert axis == expected
            else:
                assert axis.resting_rate == axis.moving_rate == 0.0
        assert len(truth_partition(world)) == k


def test_request_and_absorption_clock_is_exact_and_no_death_reading_is_used():
    world = make_physical_world(seed=10, k=5)
    actuator = next(index for index, assignment in enumerate(world.actuators) if assignment.target == 2 and assignment.size == "large")
    seed = 37
    empty = branch_sensations(world, seed=seed, trace=True)
    first = branch_sensations(world, seed=seed, first=actuator, trace=True)
    second = branch_sensations(world, seed=seed, second=actuator, trace=True)
    assert len(empty) == BRANCH_TICKS
    assert first[:6] == empty[:6]
    assert first[6].mean > empty[6].mean
    assert second[:12] == empty[:12]
    assert second[12].mean > empty[12].mean
    assert all(sensation.minimum > 0.0 for trace in (empty, first, second) for sensation in trace)
    config = ExpandedBodyConfig()
    births_rng = Random(domain_seed(seed, "birth"))
    scale_rng = Random(domain_seed(seed, "scale"))
    births = [births_rng.choice(config.birth_levels) for _ in range(5)]
    rates = [axis.resting_rate * scale_rng.uniform(0.4, 1.6) for axis in EXPANDED_BODY_SCHEMA.axes]
    before_first = births[2] - 6 * rates[2]
    expected_final = min(1.0, before_first + 0.6) - 7 * rates[2]
    assert first[-1].mean - empty[-1].mean == pytest.approx((expected_final - (births[2] - 13 * rates[2])) / 5)


def test_domain_separation_removes_the_existing_three_block_cross_domain_alias():
    config = SurveyConfig()
    all_domain_seeds = set()
    all_context_seeds = set()
    for block in range(5):
        for k in range(1, 6):
            base = config.seed_base + block * BLOCK_STRIDE + (k - 1) * WORLD_STRIDE
            for offset in ("development", "held_out", "remap_development", "remap_held_out", "unreachable"):
                for index in range(40):
                    context = base + OFFSETS[offset] + index
                    assert context not in all_context_seeds
                    all_context_seeds.add(context)
                    for domain in ("birth", "scale", "shock"):
                        physical_seed = domain_seed(context, domain)
                        assert physical_seed not in all_domain_seeds
                        all_domain_seeds.add(physical_seed)
    first_context = SURVEY_BASE + OFFSETS["development"]
    aliased_context = first_context + 3 * BLOCK_STRIDE + 2
    assert domain_seed(first_context, "scale") != domain_seed(aliased_context, "birth")
    assert DIAGNOSTIC_BASE > max(all_context_seeds)


def test_real_physics_exposes_cross_occlusion_without_axis_labels():
    world = make_physical_world(seed=35, k=5)
    transcript = collect_context(world, seed=64)
    for a, assignment_a in enumerate(world.actuators):
        for b, assignment_b in enumerate(world.actuators):
            interaction = transcript.interaction(a, b)
            if assignment_a.target != assignment_b.target:
                assert abs(interaction) < TOLERANCE
            elif assignment_a.size == assignment_b.size == "large":
                assert interaction < -TOLERANCE
    assert [field.name for field in fields(ContextTranscript)] == ["empty", "first", "second", "joint"]
    assert [field.name for field in fields(Sensation)] == ["mean", "minimum"]


def _synthetic_transcripts(groups=((1, 4, 7), (0, 8)), *, variations=3):
    """Causal scalar evidence with unequal aliases; no physical truth in fit."""
    transcripts = []
    active = {actuator for group in groups for actuator in group}
    for variation in range(variations):
        empty = Sensation(0.2 + variation * 0.01, 0.1)
        effects = [0.04 + actuator * 0.001 if actuator in active else 0.0 for actuator in range(ACTUATORS)]
        singles = tuple(Sensation(empty.mean + effect, 0.1) for effect in effects)
        joint = tuple(tuple(Sensation(
            empty.mean + effects[a] + effects[b] - (0.025 if any(a in group and b in group for group in groups) else 0),
            0.1,
        ) for b in range(ACTUATORS)) for a in range(ACTUATORS))
        transcripts.append(ContextTranscript(empty, singles, singles, joint))
    return tuple(transcripts)


def test_inference_counts_causal_components_without_two_alias_or_size_assumptions():
    transcripts = _synthetic_transcripts()
    inference = infer_groups(transcripts)
    assert inference.active == (0, 1, 4, 7, 8)
    assert inference.groups == ((0, 8), (1, 4, 7))
    assert inference.representatives == (8, 7)
    assert inference.count == 2
    score = score_overlap(inference, _synthetic_transcripts(variations=2))
    assert score["accuracy"] == score["occluding_recall"] == score["non_occluding_recall"] == 1.0


def test_prediction_uses_any_held_out_collision_not_every_context():
    evidence = _synthetic_transcripts()
    inferred = infer_groups(evidence)
    original = evidence[0]
    no_collision = replace(original, joint=tuple(tuple(Sensation(original.first[a].mean + original.second[b].mean - original.empty.mean, 0.1) for b in range(ACTUATORS)) for a in range(ACTUATORS)))
    assert score_overlap(inferred, (no_collision, original))["passes"]
    assert not score_overlap(inferred, (no_collision,))["passes"]


def test_surface_permutation_transports_groups_and_pairing_shuffle_breaks_them():
    evidence = _synthetic_transcripts(variations=40)
    inferred = infer_groups(evidence)
    permutation = list(reversed(range(ACTUATORS)))
    permuted = infer_groups(relabel_transcripts(evidence, permutation))
    expected = tuple(sorted(tuple(sorted(permutation[actuator] for actuator in group)) for group in inferred.groups))
    assert permuted.groups == expected
    assert permuted.count == inferred.count
    shuffled = shuffled_transcripts(evidence, seed=58)
    for old, new in zip(evidence, shuffled, strict=True):
        assert sorted((s.mean, s.minimum) for s in old.outputs()) == sorted((s.mean, s.minimum) for s in new.outputs())
    damaged = infer_groups(shuffled)
    assert damaged.active != inferred.active or damaged.groups != inferred.groups


def test_remapping_changes_a_small_consequence_and_keeps_the_old_large_alias():
    world = make_physical_world(seed=15, k=3)
    changed, moved = remap_small_alias(world, seed=21)
    original = world.actuators[moved]
    assert original.size == changed.actuators[moved].size == "small"
    assert original.target != changed.actuators[moved].target
    assert world.active_slots == changed.active_slots
    assert all(old == new for index, (old, new) in enumerate(zip(world.actuators, changed.actuators, strict=True)) if index != moved)
    assert any(assignment.target == original.target and assignment.size == "large" for assignment in changed.actuators)
    assert truth_partition(changed) != truth_partition(world)


def test_unreachable_live_slot_has_inert_actuators_and_cannot_be_counted_as_reachable():
    world = make_physical_world(seed=18, k=2)
    challenged = unreachable_world(world, seed=28)
    added = set(challenged.active_slots) - set(world.active_slots)
    assert len(added) == 1
    assert added == set(challenged.blocked_slots)
    assert truth_partition(challenged) == truth_partition(world)
    blocked_ids = [index for index, assignment in enumerate(challenged.actuators) if assignment.target in added]
    assert len(blocked_ids) == 2
    assert all(not challenged.actionable(index) for index in blocked_ids)
    empty = branch_sensations(challenged, seed=52)
    for actuator in blocked_ids:
        assert branch_sensations(challenged, seed=52, first=actuator, second=actuator) == empty
    assert branch_sensations(world, seed=52) != empty


def _passing_records():
    return [{
        "block": block, "k": k, "count_exact": True,
        "inference": {"count": k}, "partition_exact": True,
        "prediction": {"passes": True}, "permutation": {"transport_exact": True},
        "shuffled": {"partition_exact": False},
        "remap": {"stale_invalidated": True, "partition_exact": True} if k >= 2 else None,
    } for block in range(5) for k in range(1, 6)]


def test_locked_gates_use_per_k_four_of_five_and_targeted_remap_error():
    records = _passing_records()
    assert score_gates(records)["all_pass"]
    for record in records[:2]:
        record["prediction"]["passes"] = False
    assert score_gates(records)["gates"]["G3_prediction"]
    records[5]["prediction"]["passes"] = False
    assert not score_gates(records)["gates"]["G3_prediction"]
    records = _passing_records()
    records[1]["remap"]["stale_invalidated"] = False
    records[6]["remap"]["stale_invalidated"] = False
    assert not score_gates(records)["gates"]["G6_moved_mapping"]
    with pytest.raises(ValueError, match="complete"):
        score_gates(records[:-1])


def test_configuration_budget_and_resume_are_source_pinned(tmp_path, monkeypatch):
    from homesocial.organism import saturation_structure as module

    with pytest.raises(ValueError, match="locked"):
        SurveyConfig(tolerance=0.1)
    config = SurveyConfig(diagnostic=True)
    assert config.contexts == 2
    assert config.blocks == 1
    assert BRANCHES_PER_CONTEXT == 121
    output = tmp_path / "progress.json"
    hashes = {"module": "first"}
    monkeypatch.setattr(module, "source_hashes", lambda: hashes)
    calls = []

    def fake_world(*, block, k, config):
        calls.append((block, k))
        if k == 2 and len(calls) == 2:
            raise RuntimeError("interrupted")
        return {"block": block, "k": k, "inference": {"count": k}, "partition_exact": True, "elapsed_seconds": 0.1, "budget": {"actual_ticks": 13}}

    monkeypatch.setattr(module, "run_world", fake_world)
    with pytest.raises(RuntimeError, match="interrupted"):
        run_survey(output=output, config=config)
    saved = json.loads(output.read_text())
    assert len(saved["records"]) == 1
    completed = run_survey(output=output, config=config)
    assert completed["complete"]
    assert not completed["gates_evaluated"]
    assert calls.count((0, 1)) == 1
    assert completed["budget"]["actual_ticks"] == 65
    hashes["module"] = "changed"
    with pytest.raises(ValueError, match="mismatch"):
        run_survey(output=output, config=config)
