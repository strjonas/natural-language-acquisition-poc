from copy import deepcopy
from dataclasses import replace
import json

import pytest

from homesocial.organism import sequential_structure as seq
from homesocial.organism.saturation_structure import make_physical_world, unreachable_world


def synthetic():
    return [{"action": b, "previous": a, "effect": (0.02 if a // 2 == b // 2 else 0.1)}
            for a in range(4) for b in range(4) for _ in range(5)]


def test_inference_recovers_suppression_without_known_count_or_alias_size():
    rows = synthetic()
    model = seq.infer(rows)
    assert model["groups"] == [[0, 1], [2, 3]]
    assert model["total_dimension"] == "unresolved"
    rows = [{"action": b, "previous": a, "effect": 0.02 if a < 3 and b < 3 else 0.1}
            for a in range(4) for b in range(4) for _ in range(5)]
    assert seq.infer(rows)["groups"] == [[0, 1, 2], [3]]


def test_no_unsupported_edges_and_shuffling_breaks_pairing():
    assert seq.infer(synthetic()[:2])["active"] == []
    rows = synthetic()
    shuffled = [{**r, "effect": 0.1} for r in rows]
    assert seq.infer(shuffled)["groups"] == [[0], [1], [2], [3]]


def test_permutation_transports_predictions_and_heldout_cannot_fit():
    rows = synthetic()
    permutation = [7, 2, 9, 3, 0, 1, 4, 5, 6, 8]
    model = seq.infer(rows)
    assert seq.infer(seq.permute(rows, permutation))["groups"] == [[2, 7], [3, 9]]
    before = deepcopy(model)
    seq.prediction_score(model, [{"action": 0, "previous": 1, "effect": 100.0}])
    assert model == before


def test_drift_baseline_is_past_only_and_robust_to_one_shock():
    assert seq.estimate_effect(.1, [-.01, -.01]) is None
    assert seq.estimate_effect(.1, [-.01, -.06, -.01, -.01, -.01]) == pytest.approx(.11)


def test_episode_absorption_and_death_respect_lived_budget(monkeypatch):
    world = make_physical_world(seed=4, k=1)
    monkeypatch.setattr(seq, "CONFIG", replace(seq.CONFIG, life_steps=20, shock_probability=0))
    life = seq.run_episode(world, identity_seed=seq.DIAGNOSTIC_BASE,
                           episode_seed=seq.DIAGNOSTIC_BASE + 1)
    assert [r["tick"] for r in life["records"]] == [7, 13, 19]
    assert life["records"][0]["previous"] is None
    assert life["records"][1]["previous"] == life["records"][0]["action"]
    monkeypatch.setattr(seq, "CONFIG", replace(seq.CONFIG, birth_levels=(0.001,)))
    died = seq.run_episode(world, identity_seed=seq.DIAGNOSTIC_BASE,
                           episode_seed=seq.DIAGNOSTIC_BASE + 1)
    assert not died["survived"] and died["steps"] == 1 and died["records"] == []


def test_unreachable_count_is_not_equated_to_bodily_dimension():
    world = make_physical_world(seed=14, k=2)
    hidden = unreachable_world(world, seed=10)
    assert len(seq.actual_groups(hidden)) == 2 and len(hidden.active_slots) == 3
    model = seq.infer([])
    score = seq.score(model, world)
    assert score["balanced_accuracy"] == 0.0  # missing IDs cannot earn true negatives


def test_seed_domains_and_all_survey_episode_seeds_are_disjoint():
    seeds = [base + block*2_000_000 + k*100_000 + identity*100 + offset
             for base, blocks, count in ((seq.BASE, 5, 20), (seq.DIAGNOSTIC_BASE, 1, 1))
             for block in range(blocks) for k in range(1, 6) for identity in range(count)
             for offset in (*range(8), *range(10, 16), *range(20, 26))]
    assert len(seeds) == len(set(seeds))
    assert len({seq.seed_for(s, d) for s in seeds for d in ("birth", "scale", "shock", "actions", "map")}) == 5*len(seeds)


def test_diagnostic_and_source_pinned_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(seq, "CONFIG", replace(seq.CONFIG, life_steps=20))
    out = tmp_path / "diagnostic.json"
    first = seq.run(diagnostic=True, out=out)
    assert first["complete"] and first["summary"]["1"]["gates"] is None
    assert seq.run(diagnostic=True, out=out) == first
    bad = json.loads(out.read_text())
    bad["manifest"]["base"] += 1
    out.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="source changed"):
        seq.run(diagnostic=True, out=out)


def test_full_gates_are_binding_and_duplicate_worlds_rejected():
    template = {"exact": 1., "count": 1., "count_exact": 1., "balanced_accuracy": 1.,
                "active_recall": 1., "active_precision": 1., "supported_fraction": 1.,
                "actual_ticks": 400, "observations": 60, "survival": 1.,
                "shuffled": {"exact": 0.}, "permutation_exact": True,
                "heldout": {"baseline_sse": 1., "model_sse": .5}}
    rows = [{"block": b, "k": k, "identity": i,
             "budgets": {budget: {**deepcopy(template), "count": k} for budget in ("1", "6")},
             "challenges": {"remap": {"exact": 1., "stale_wrong": True}}}
            for b in range(5) for k in range(1, 6) for i in range(20)]
    assert seq.summarize(rows, blocks=5, identities=20)["1"]["all_pass"]
    bad = deepcopy(rows)
    for r in bad:
        r["budgets"]["1"]["heldout"]["model_sse"] = 1.
    assert not seq.summarize(bad, blocks=5, identities=20)["1"]["gates"]["G4_prediction"]
    with pytest.raises(ValueError, match="duplicate"):
        seq.summarize(rows + rows[:1], blocks=5, identities=20)
