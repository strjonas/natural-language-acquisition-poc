import json
from dataclasses import replace

import numpy as np
import pytest

from homesocial.organism import active_self as probe
from homesocial.organism.saturation_structure import make_physical_world
from homesocial.organism.self_posterior import PosteriorConfig, SelfPosterior


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(probe, "POSTERIOR", PosteriorConfig(particles=32))
    monkeypatch.setattr(probe, "LIVES", 2)


def _world(k=2, seed=7):
    return make_physical_world(seed=probe.domain_seed(seed, "map"), k=k)


def test_seed_bands_are_disjoint_and_identities_unique():
    bases = (probe.SURVEY_BASE, probe.DIAGNOSTIC_BASE, probe.DEVELOPMENT_BASE)
    seen = set()
    for base in bases:
        for block in range(probe.BLOCKS):
            for k in range(1, 6):
                for identity in range(probe.IDENTITIES):
                    seed = probe.identity_seed(base, block, k, identity)
                    assert seed not in seen and seed + 50 not in seen
                    seen.update((seed, seed + 50))
    spans = sorted((b, b + probe.BLOCKS * probe.BLOCK_STRIDE) for b in bases)
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
    assert probe.SURVEY_BASE > 4_700_000_000  # above every earlier probe's band


def test_lived_body_absorbs_next_tick_and_dies_at_zero():
    world = _world(k=1)
    body = probe.LivedBody(world, identity=1, life=0)
    active = world.active_slots[0]
    actuator = next(a for a, x in enumerate(world.actuators) if x.target == active and x.size == "large")
    level = body.body.state.values[active]
    rate = body.true_rates()[active]
    body.tick(actuator)
    after = body.body.state.values[active]
    shocked = after < min(1.0, level + 0.6) - rate - 1e-9
    assert shocked or after == pytest.approx(min(1.0, level + 0.6) - rate)
    total, minimum = body.sense()
    assert total == pytest.approx(sum(body.body.state.values))
    assert minimum == pytest.approx(min(body.body.state.values))
    while body.alive:
        body.tick(None)
    assert min(body.body.state.values) <= 0.0


def test_inactive_slots_stay_full_and_unshocked():
    world = _world(k=1, seed=3)
    body = probe.LivedBody(world, identity=3, life=0)
    for _ in range(300):
        if not body.tick(None):
            break
        for slot in range(5):
            if slot not in world.active_slots:
                assert body.body.state.values[slot] == 1.0


def test_self_models_receive_only_their_own_actuator_and_two_scalars(small, monkeypatch):
    calls = []
    original = SelfPosterior.observe

    def guarded(self, grant, total, minimum):
        assert grant is None or isinstance(grant, int)
        assert isinstance(total, float) and isinstance(minimum, float)
        calls.append(grant)
        return original(self, grant, total, minimum)

    monkeypatch.setattr(SelfPosterior, "observe", guarded)
    world, donor = _world(k=2), _world(k=2, seed=8)
    probe.run_arm("posterior", world, donor, ident=11, donor_ident=61)
    assert calls


def test_yoked_arm_senses_the_donor_but_lives_in_its_own_body(small, monkeypatch):
    world, donor = _world(k=2), _world(k=2, seed=8)
    seen = []
    original = SelfPosterior.observe

    def spy(self, grant, total, minimum):
        seen.append((grant, total, minimum))
        return original(self, grant, total, minimum)

    monkeypatch.setattr(SelfPosterior, "observe", spy)
    result = probe.run_arm("yoked", world, donor, ident=11, donor_ident=61, lives=1)
    replay = probe.LivedBody(donor, identity=61, life=0)
    for grant, total, minimum in seen:
        replay.tick(grant)
        assert (total, minimum) == pytest.approx(replay.sense())
    # After the donor dies the organism keeps living (and acting) without new
    # sensations, so its own life is at least as long as the observed prefix.
    assert result["lives"][0]["ticks"] >= len(seen) or not replay.alive


def test_oracle_structure_posterior_carries_the_true_map(small):
    world = _world(k=3)
    body = probe.LivedBody(world, identity=5, life=0)
    posterior = probe.oracle_structure_posterior(world, body.true_rates(), np.random.default_rng(0))
    posterior.birth(*body.sense(), retain=False)
    truth = probe.true_partition(world)
    assert posterior.structure_report().partition == truth


def test_identity_row_has_every_arm_and_valid_records(small):
    row = probe.run_identity(base=probe.DIAGNOSTIC_BASE, block=0, k=2, identity=0)
    assert set(row["arms"]) == set(probe.ARMS)
    for arm, result in row["arms"].items():
        assert "_history" not in result
        assert len(result["lives"]) == 2
        for record in result["lives"]:
            assert 1 <= record["ticks"] <= 400
            if arm in probe.POSTERIOR_ARMS:
                assert 0.0 <= record["confidence"] <= 1.0
    assert row["permutation"]["transports"] is True


def _fake_rows(blocks=5, identities=1, lives=6):
    """A synthetic complete survey in which the treatment separates cleanly
    from every lesion except reset, which ties it."""

    rows = []
    for b in range(blocks):
        for k in range(1, 6):
            for i in range(identities):
                arms = {}
                for arm in probe.ARMS:
                    alive = arm in ("posterior", "uncritical", "reset", "certainty", "oracle", "oracle_structure") \
                        or (arm == "bandit" and k == 1)
                    recs = []
                    for life in range(lives):
                        rec = {"life": life, "survived": alive and not (arm == "random"), "ticks": 400, "grants": 66}
                        if arm in probe.POSTERIOR_ARMS:
                            exact = arm in ("posterior", "uncritical", "reset")
                            confidence = 1.0 if exact else 0.0
                            if arm == "uncritical":
                                confidence = 0.5 + 0.02 * b  # miscalibrated, by a different amount per block
                            rec.update({"exact": exact, "confidence": confidence, "resource_count": k})
                        recs.append(rec)
                    arms[arm] = {"arm": arm, "lives": recs}
                row = {"block": b, "k": k, "identity": i, "arms": arms}
                if i == 0:
                    row["permutation"] = {"transports": True}
                rows.append(row)
    return rows


def test_summary_refuses_incomplete_surveys_and_gates_bind(monkeypatch):
    monkeypatch.setattr(probe, "LIVES", 6)
    rows = _fake_rows()
    with pytest.raises(ValueError):
        probe.summarize(rows[:-1], blocks=5, identities=1)
    # give random a little per-block variation so paired intervals are defined
    for row in rows:
        if row["k"] in (1, 2) and row["block"] in (0, 1):
            row["arms"]["random"]["lives"][0]["survived"] = True
    summary = probe.summarize(rows, blocks=5, identities=1)
    graded = probe.grade(summary)
    assert set(graded["gates"]) == {
        "G1_discovery", "G2_action_uses_self", "G3_structure_not_action_values", "G4_own_body",
        "G5_uncertainty_drives_discovery", "G6_persistence", "G7_calibrated_self_report",
        "G8_self_criticism_load_bearing", "G9_permutation"}
    assert graded["gates"]["G1_discovery"] is True
    assert graded["gates"]["G7_calibrated_self_report"] is True
    assert graded["gates"]["G8_self_criticism_load_bearing"] is True
    assert graded["gates"]["G6_persistence"] is False  # reset ties the treatment here
    assert graded["all_pass"] is False


def test_resume_refuses_a_changed_configuration(tmp_path, small, monkeypatch):
    out = tmp_path / "block.json"
    out.write_text(json.dumps({"manifest": {"different": True}, "block": 0, "rows": []}))
    with pytest.raises(ValueError):
        probe.run_block(base=probe.DIAGNOSTIC_BASE, block=0, identities=1, blocks=1, out=out)
