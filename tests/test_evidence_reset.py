from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from homesocial.env import Action
from homesocial.organism import evidence_reset as er
from homesocial.organism.cross_life_self import RateMemory, reset_identity_world
from homesocial.organism.train import ACTIONS, OrganismConfig, load_organism_checkpoint

PARENT = Path("runs/organism/probe52_guided_report_lexicon/adult/organism_report_seed1.npz")


def setup(mode="adaptive"):
    organism = OrganismConfig()
    world,packet,report = reset_identity_world(organism,identity_seed=er.DIAGNOSTIC_IDENTITY,
                                              episode_seed=er.DIAGNOSTIC_EPISODE)
    memory = er.EvidenceResetMemory(organism,report,mode=mode,random_seed=7)
    memory.start_episode(packet,None)
    return organism,world,packet,report,memory


def test_streak_requires_two_same_sign_errors_and_clears_at_bounds_or_small_errors():
    _,_,_,_,m = setup()
    prediction = [.5,.5,.5]
    assert not m.observe_error([.6,.5,.5],prediction,tick=1)
    assert m.observe_error([.6,.5,.5],prediction,tick=2)
    assert m.reset_ticks == [2]
    assert not m.observe_error([.6,.5,.5],prediction,tick=3)
    assert not m.observe_error([.4,.5,.5],prediction,tick=4)
    assert m.observe_error([.4,.5,.5],prediction,tick=5)
    assert not m.observe_error([.6,.5,.5],prediction,tick=6)
    assert not m.observe_error([1.,.5,.5],prediction,tick=7)
    assert not m.observe_error([.6,.5,.5],prediction,tick=8)
    assert not m.observe_error([.51,.5,.5],prediction,tick=9)
    assert not m.observe_error([.6,.5,.5],prediction,tick=10)


def test_muted_never_resets_and_birth_preserves_rates_but_clears_streak():
    _,world,packet,_,m = setup("muted")
    m._a[:] = .2
    for tick in range(10):
        assert not m.observe_error([.7,.7,.7],[.5,.5,.5],tick=tick)
    assert np.all(m._a == .2)
    saved = m.snapshot()
    m.start_episode(packet,None,saved)
    assert np.all(m.last_sign == 0) and np.all(m._a == .2)
    assert m.reset_ticks == []


def test_real_muted_update_is_bit_identical_to_existing_rls_without_truth_access():
    organism,world,packet,report,m = setup("muted")
    baseline = RateMemory(organism,report)
    baseline.start_episode(packet,None)
    world.report = replace(world.report,interoception_probability=1.)
    class NoTruth:
        def __getattr__(self,name):
            raise AssertionError(name)
    for _ in range(16):
        before = packet
        packet,_,terminated,truncated,info = world.step(Action.WAIT)
        if terminated or truncated:
            break
        m.update(before,ACTIONS.index(Action.WAIT),packet,NoTruth(),info["interoception"])
        baseline.update(before,ACTIONS.index(Action.WAIT),packet,NoTruth(),info["interoception"])
        assert m.snapshot() == baseline.snapshot()
    assert m.readings > 3


def test_error_is_observed_after_one_transition_but_before_fitting(monkeypatch):
    _,world,packet,_,m = setup()
    world.report = replace(world.report,interoception_probability=1.)
    before = packet
    packet,_,_,_,info = world.step(Action.WAIT)
    calls = []
    original_update = m._jacobian.update
    def advance(*args):
        calls.append("advance")
        return original_update(*args)
    monkeypatch.setattr(m._jacobian,"update",advance)
    def observe(actual,prediction,*,tick):
        assert calls == ["advance"]
        assert m.readings == 0 and not np.any(m._a)
        calls.append("observe")
        return False
    monkeypatch.setattr(m,"observe_error",observe)
    m.update(before,ACTIONS.index(Action.WAIT),packet,None,info["interoception"])
    assert calls == ["advance","observe"] and m.readings == 1 and m.evidence_ticks == 1


def test_sign_corruption_changes_only_trigger_not_fitting_target():
    _,world,packet,_,m = setup("scrambled")
    _,_,_,_,muted = setup("muted")
    before = packet
    packet,_,_,_,_ = world.step(Action.WAIT)
    target = (.4,.6,.7)
    for memory in (m,muted):
        memory.update(before,ACTIONS.index(Action.WAIT),packet,None,target)
    assert m.snapshot() == muted.snapshot()
    # Future errors may reset; the actual reading is never replaced by a
    # sign-scrambled target, even at this first scored update.
    np.testing.assert_array_equal(m._jacobian.belief,target)


def fixture_units():
    units=[]
    for b in range(5):
        for i in range(28):
            for cohort in er.COHORTS:
                scores={p:{n:{"rate_mae":(1. if n=="adaptive" and cohort=="changed" else 2.),
                                "regret":(1. if n=="adaptive" and cohort=="changed" else 2.),
                                "word_changes":0} for n in er.READOUTS} for p in ("all","after_change")}
                rows=[{"survived":1,"steps":100,"readings":3,"counts":{"all":10,"after_change":0},
                       "scores":deepcopy(scores),"detector_resets":{n:[] for n in er.DETECTORS}}
                      for _ in range(5)]
                units.append({"block":b,"identity":i,"cohort":cohort,"episodes":rows})
    return units


def test_each_screen_gate_is_binding_and_false_alarm_is_per_identity():
    units=fixture_units()
    assert er.summarize(units,seeds=5,identities=28)["all_pass"]
    bad=deepcopy(units)
    for u in bad:
        if u["cohort"]=="stable" and u["identity"]<3:
            u["episodes"][0]["detector_resets"]["adaptive"]=[1,2,3]
    result=er.summarize(bad,seeds=5,identities=28)
    assert result["stable_false_alarm_share"] == pytest.approx(3/28)
    assert not result["gates"]["G2_stable_preservation"]
    for u in units:
        for r in u["episodes"]:
            r["scores"]["all"]["scrambled"] = deepcopy(r["scores"]["all"]["adaptive"])
    assert not er.summarize(units,seeds=5,identities=28)["gates"]["G3_controls"]
    with pytest.raises(ValueError,match="duplicate"):
        er.summarize(units+units[:1],seeds=5,identities=28)


def test_seed_bands_and_random_sign_domains_do_not_overlap():
    values=[base+b*er.STRIDE+i*16+e for base,bs,ns in
            ((er.EPISODE_BASE,5,28),(er.DIAGNOSTIC_EPISODE,1,2))
            for b in range(bs) for i in range(ns) for e in range(6)]
    assert len(values)==len(set(values))
    assert min(values)>4_294_967_295  # Deliberately supported Python integer seeds.
    assert er.rng_seed(1,"adaptive") != er.rng_seed(1,"scrambled")


def test_short_real_diagnostic_resume_and_muted_parity(tmp_path):
    model,organism=load_organism_checkpoint(PARENT)
    organism=replace(organism,report=replace(organism.report,life_steps=30))
    out=tmp_path/"smoke.json"
    first=er.run(model,organism,checkpoint=PARENT,out=out,diagnostic=True)
    assert first["complete"] and first["summary"]["muted_exact"]
    assert first["summary"]["gates"] is None
    assert er.run(model,organism,checkpoint=PARENT,out=out,diagnostic=True)==first
    wrong=replace(organism,report=replace(organism.report,life_steps=31))
    with pytest.raises(ValueError,match="configuration changed"):
        er.run(model,wrong,checkpoint=PARENT,out=out,diagnostic=True)
