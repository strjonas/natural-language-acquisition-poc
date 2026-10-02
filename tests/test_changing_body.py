from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from homesocial.env import Action
from homesocial.organism import changing_body as cb
from homesocial.organism.cross_life_self import RateMemory, reset_identity_world
from homesocial.organism.train import ACTIONS, OrganismConfig, load_organism_checkpoint

PARENT = Path("runs/organism/probe52_guided_report_lexicon/adult/organism_report_seed1.npz")


def setup_memory(kind=RateMemory):
    organism = OrganismConfig()
    world, packet, report = reset_identity_world(organism, identity_seed=cb.DIAGNOSTIC_IDENTITY,
                                                episode_seed=cb.DIAGNOSTIC_EPISODE)
    memory = kind(organism, report)
    memory.start_episode(packet, None)
    return organism, world, packet, report, memory


def test_cue_clears_rates_without_new_state_or_fabricated_evidence():
    _, _, _, _, memory = setup_memory()
    memory._a[:] = .3
    memory._m[:] = np.log1p(.3)
    memory._covariance *= .1
    memory._burn_species[:] = -2
    memory._burn_corrected[:] = -2.6
    memory.readings, memory.evidence_ticks = 10, 200
    jacobian = memory._jacobian
    before = memory._jacobian.belief.copy()
    cb.clear_rate_evidence(memory)
    assert memory._jacobian is jacobian
    np.testing.assert_array_equal(memory._jacobian.belief, before)
    assert not np.any(memory._a) and not np.any(memory._m)
    assert not np.any(memory._burn_species) and not np.any(memory._burn_corrected)
    np.testing.assert_array_equal(memory._covariance[0], np.eye(memory._a.shape[1]) / memory._ridge)
    assert (memory.readings, memory.evidence_ticks) == (10, 200)


def test_forgetting_applies_to_readout_and_does_not_read_world_truth():
    _, world, packet, _, memory = setup_memory(cb.ForgettingMemory)
    memory._burn_species[:] = -10
    memory._burn_corrected[:] = -12
    world.report = replace(world.report, interoception_probability=1.)
    before = packet
    packet, _, _, _, info = world.step(Action.WAIT)
    class NoTruth:
        def __getattr__(self, name):
            raise AssertionError(f"hidden access {name}")
    # A clipped reading skips fitting. Existing RLS still accumulates the
    # interval's species burn, which must be added after the old sum decays.
    memory.update(before, ACTIONS.index(Action.WAIT), packet, NoTruth(), (1., 1., 1.))
    np.testing.assert_allclose(memory._burn_species, -9. - np.array([.008, .012, .016]))
    np.testing.assert_allclose(memory._burn_corrected, -10.8 - np.array([.008, .012, .016]))
    assert memory.readings == 1 and memory.evidence_ticks == 1
    # Actual eligible public readings also run without a simulator object.
    before = packet
    packet, _, _, _, info = world.step(Action.WAIT)
    memory.update(before, ACTIONS.index(Action.WAIT), packet, NoTruth(), info["interoception"])
    assert memory.readings == 2


def test_schedule_domains_and_world_bands_are_disjoint():
    seeds = [base + b*cb.STRIDE + i*16 + e
             for base, blocks, count in ((cb.EPISODE_BASE, 5, 28), (cb.DIAGNOSTIC_EPISODE, 1, 2))
             for b in range(blocks) for i in range(count) for e in range(6)]
    assert len(seeds) == len(set(seeds))
    schedules = [cb.change_schedule(cb.IDENTITY_BASE+i) for i in range(100)]
    assert {s["episode"] for s in schedules} == {2, 3, 4}
    assert {s["tick"] for s in schedules} == {24, 48, 72}
    assert all(set(s["constants"]) == set(cb.BODY_NEEDS) for s in schedules)
    assert cb.domain_seed(1, "timing") != cb.domain_seed(1, "rates")


def test_real_episode_sham_parity_and_change_after_missed_boundary(monkeypatch):
    model, organism = load_organism_checkpoint(PARENT)
    organism = replace(organism, report=replace(organism.report, life_steps=30))
    world, packet, report = reset_identity_world(organism,
        identity_seed=cb.DIAGNOSTIC_IDENTITY, episode_seed=cb.DIAGNOSTIC_EPISODE)
    warm = RateMemory(organism, report)
    warm.start_episode(packet, None)
    priors = {name: warm.snapshot() for name in cb.MEMORIES}
    kwargs = dict(identity_seed=cb.DIAGNOSTIC_IDENTITY, episode_seed=cb.DIAGNOSTIC_EPISODE+1,
                  episode=1, priors=priors, cue_seen=False)
    first = cb.run_episode(model, organism, cell="retained", cohort="stable", **kwargs)
    second = cb.run_episode(model, organism, cell="oracle_reset", cohort="stable", **kwargs)
    assert first == second
    schedule = {"episode": 2, "tick": 72, "constants": dict.fromkeys(cb.BODY_NEEDS, .7)}
    monkeypatch.setattr(cb, "change_schedule", lambda _: schedule)
    kwargs["episode"] = 2
    missed = cb.run_episode(model, organism, cell="retained", cohort="changed", **kwargs)
    assert missed["reset_tick"] is None and not missed["physical_change_exposed"]
    kwargs.update(episode=3, priors=missed["priors"], cue_seen=missed["cue_seen"])
    birth = cb.run_episode(model, organism, cell="retained", cohort="changed", **kwargs)
    assert birth["reset_tick"] == 0 and birth["physical_change_exposed"]
    schedule["tick"] = 0
    kwargs.update(episode=2, priors=priors, cue_seen=False)
    inside = cb.run_episode(model, organism, cell="retained", cohort="changed", **kwargs)
    assert inside["physical_change_exposed"] and inside["reset_tick"] == 0


def units_fixture():
    units = []
    for block in range(5):
        for identity in range(28):
            for cohort in cb.COHORTS:
                for cell in cb.CELLS:
                    scores = {p: {n: {"regret": (1 if n == "oracle_reset" else 2),
                                       "rate_mae": (1 if n == "oracle_reset" else 2), "word_changes": 0}
                                   for n in cb.CELLS} for p in ("all", "after_change")}
                    rows = [{"episode": e, "survived": int(cohort == "changed" and cell in ("oracle_reset", "true_rate")),
                             "steps": 50, "readings": 1, "reset_tick": None,
                             "physical_change_exposed": False, "counts": {"all": 10, "after_change": 0},
                             "scores": deepcopy(scores)} for e in range(1, 6)]
                    units.append({"block": block, "identity": identity, "cohort": cohort, "cell": cell, "episodes": rows})
    return units


def test_summary_gates_pairing_and_tick_weighting():
    units = units_fixture()
    result = cb.summarize(units, seeds=5, identities=28)
    assert result["all_pass"] and result["stable_exact_parity"]
    assert result["contrasts"]["C3_oracle_reset_regret"]["mean"] == pytest.approx(.1)
    bad = deepcopy(units)
    for u in bad:
        if u["cohort"] == "changed" and u["cell"] == "oracle_reset":
            for r in u["episodes"]:
                r["survived"] = 0
    assert not cb.summarize(bad, seeds=5, identities=28)["gates"]["C2_oracle_reset_survival"]
    with pytest.raises(ValueError, match="duplicate"):
        cb.summarize(units + units[:1], seeds=5, identities=28)
    bad = deepcopy(units)
    next(u for u in bad if u["cohort"] == "stable" and u["cell"] == "oracle_reset")["episodes"][0]["steps"] += 1
    with pytest.raises(ValueError, match="invalid construction"):
        cb.summarize(bad, seeds=5, identities=28)


def test_real_interruption_resumes_without_repeating_completed_units(tmp_path, monkeypatch):
    model, organism = load_organism_checkpoint(PARENT)
    organism = replace(organism, report=replace(organism.report, life_steps=30))
    original = cb.run_unit
    calls = 0
    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return original(*args, **kwargs)
    monkeypatch.setattr(cb, "run_unit", interrupted)
    resumed_path = tmp_path / "resumed.json"
    with pytest.raises(RuntimeError, match="interruption"):
        cb.run(model, organism, checkpoint=PARENT, out=resumed_path, diagnostic=True)
    monkeypatch.setattr(cb, "run_unit", original)
    resumed = cb.run(model, organism, checkpoint=PARENT, out=resumed_path, diagnostic=True)
    reference = cb.run(model, organism, checkpoint=PARENT, out=tmp_path/"fresh.json", diagnostic=True)
    assert resumed["units"] == reference["units"]
    assert resumed["calibration"] == reference["calibration"]
    assert resumed["summary"] == reference["summary"]
    assert resumed["summary"]["gates"] is None
    assert cb.run(model, organism, checkpoint=PARENT, out=resumed_path, diagnostic=True) == resumed
    monkeypatch.setattr(cb, "FORGETTING", .8)
    with pytest.raises(ValueError, match="configuration changed"):
        cb.run(model, organism, checkpoint=PARENT, out=resumed_path, diagnostic=True)
