"""Causal and reproducibility guards for persistent-identity rate memory."""

from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest

from homesocial.env import Action
from homesocial.island.report import BODY_NEEDS, ReportConfig
from homesocial.organism.cross_life_self import (
    CELLS, RateMemory, grade, identity_constants, memory_rates,
    reset_identity_world, run_identity_cell, run_survey, validate_run,
)
from homesocial.organism.portion_request import SelfModelTier
from homesocial.organism.train import ACTIONS, OrganismConfig, load_organism_checkpoint


def test_identity_persists_while_birth_and_episode_randomness_change():
    organism = OrganismConfig()
    first, packet1, report = reset_identity_world(
        organism, identity_seed=3_200_000_001, episode_seed=3_300_000_001,
    )
    second, packet2, _ = reset_identity_world(
        organism, identity_seed=3_200_000_001, episode_seed=3_300_000_002,
    )
    paired, paired_packet, _ = reset_identity_world(
        organism, identity_seed=3_200_000_001, episode_seed=3_300_000_001,
    )
    assert set(first.metabolic_scale) == set(BODY_NEEDS)
    assert first.metabolic_scale == second.metabolic_scale == paired.metabolic_scale
    assert first.metabolic_scale == identity_constants(3_200_000_001, report)
    assert first.metabolic_scale != identity_constants(3_200_000_002, report)
    assert not np.array_equal(packet1.vector()[:3], packet2.vector()[:3])
    np.testing.assert_array_equal(packet1.vector(), paired_packet.vector())
    assert first._shock_rng.getstate() != second._shock_rng.getstate()
    assert first._intero_rng.getstate() != second._intero_rng.getstate()
    assert first._help_rng.getstate() != second._help_rng.getstate()
    assert first._shock_rng.getstate() == paired._shock_rng.getstate()


def test_episode_reset_retains_statistics_but_resets_current_state():
    organism = OrganismConfig()
    world, birth, report = reset_identity_world(
        organism, identity_seed=3_200_000_003, episode_seed=3_300_000_003,
    )
    memory = RateMemory(organism, report)
    memory.start_episode(birth, world)
    memory._a[:] = 0.25
    memory._m[:] = np.log1p(0.25)
    memory._covariance *= 0.1
    memory._burn_species[:] = -2.0
    memory._burn_corrected[:] = -2.5
    memory.readings = 8
    memory.evidence_ticks = 150
    saved = memory.snapshot()
    original_jacobian = memory._jacobian
    next_world, next_birth, _ = reset_identity_world(
        organism, identity_seed=3_200_000_003, episode_seed=3_300_000_004,
    )
    memory.start_episode(next_birth, next_world, saved)
    assert memory.snapshot() == saved
    assert memory._jacobian is not original_jacobian
    assert np.all(memory._jacobian.matrix() == 0.0)
    np.testing.assert_allclose(memory.point(), next_birth.vector()[:3])
    assert memory.recovered_scale() == dict.fromkeys(("food", "water", "energy"), 1.25)
    saved["a"][0][0] = 1.0
    assert memory._a[0, 0] == 0.25


def test_fresh_memory_matches_existing_recursive_readout_and_never_reads_truth():
    organism = OrganismConfig()
    world, birth, report = reset_identity_world(
        organism, identity_seed=3_200_000_005, episode_seed=3_300_000_005,
    )

    class NoTruth:
        @property
        def metabolic_scale(self):
            raise AssertionError("learner read true identity constants")

        @property
        def uptake_scale(self):
            raise AssertionError("learner read true uptake")

    tier = SelfModelTier("recursive", organism, report)
    tier.reset(birth, NoTruth())
    memory = RateMemory(organism, report)
    memory.start_episode(birth, NoTruth())
    np.testing.assert_allclose(memory.point(), tier.point())
    np.testing.assert_allclose(memory_rates(memory, report, 0.5),
                               [tier.believed_rate(n, 0.5) for n in ("food", "water", "energy")])
    assert set(memory.snapshot()) == {
        "a", "covariance", "burn_species", "burn_corrected",
        "readings", "skipped", "evidence_ticks",
    }


def test_updates_use_only_public_transitions_and_readings():
    organism = OrganismConfig()
    world, packet, report = reset_identity_world(
        organism, identity_seed=3_200_000_005, episode_seed=3_300_000_005,
    )
    # Deliver actual public readings on each transition, exercising the fitting
    # branch as well as prediction. The learner receives no simulator object.
    world.report = replace(world.report, interoception_probability=1.0)

    class NoTruth:
        def __getattr__(self, name):
            raise AssertionError(f"learner read simulator property {name}")

    hidden_world = NoTruth()
    tier = SelfModelTier("recursive", organism, report)
    tier.reset(packet, hidden_world)
    memory = RateMemory(organism, report)
    memory.start_episode(packet, hidden_world)
    prior = memory.snapshot()
    wait = ACTIONS.index(Action.WAIT)
    for _ in range(4):
        before = packet
        packet, _, terminated, truncated, info = world.step(Action.WAIT)
        assert not terminated and not truncated
        reading = info["interoception"]
        assert reading is not None
        tier.update(before, wait, packet, hidden_world, reading)
        memory.update(before, wait, packet, hidden_world, reading)
        np.testing.assert_allclose(memory.point(), tier.point())
        np.testing.assert_allclose(
            memory_rates(memory, report, 0.5),
            [tier.believed_rate(need, 0.5) for need in ("food", "water", "energy")],
        )
    assert memory.readings == 4
    assert memory.evidence_ticks == 4
    assert memory.snapshot()["a"] != prior["a"]


def _cells():
    cells = []
    for cell, survival in zip(CELLS, (0.2, 0.25, 0.3, 0.18)):
        cells.append({
            "cell": cell, "survival": survival,
            "per_seed_survival": [survival] * 5,
            "per_seed_matched_regret_value": [0.003] * 5,
            "per_seed_birth_rate_mae_value": [0.002] * 5,
            "per_seed_birth_identity_mae_value": [0.002] * 5,
        })
    return cells


def test_each_preregistered_gate_is_binding_and_smoke_is_not_replication():
    cells = _cells()
    assert grade(cells)["all_pass"]
    for key, gate in (
        ("per_seed_survival", "C2_retained_evidence_reaches_survival"),
        ("per_seed_matched_regret_value", "C3_matched_regret_bridge"),
        ("per_seed_birth_rate_mae_value", "C4_less_rediscovery_at_birth"),
        ("per_seed_birth_identity_mae_value", "C5_memory_belongs_to_the_individual"),
    ):
        bad = deepcopy(cells)
        bad[1][key] = [0.0] * 5
        assert grade(bad)[gate]["verdict"] == "fail"
    bad = deepcopy(cells)
    bad[2]["per_seed_survival"] = [0.2] * 5
    assert grade(bad)["C1_persistent_identity_physical_ceiling"]["verdict"] == "fail"
    bad[2]["survival"] = 0.14
    assert grade(bad)["C6_viable_construction"]["verdict"] == "fail"
    bad = deepcopy(cells)
    bad[1]["per_seed_survival"] = [0.3, 0.3, 0.3, 0.19, 0.19]
    assert grade(bad)["C2_retained_evidence_reaches_survival"]["verdict"] == "fail"
    bad[1]["per_seed_survival"] = [0.25]
    with pytest.raises(ValueError, match="five"):
        grade(bad)
    with pytest.raises(ValueError, match="distinct"):
        grade(cells + [cells[0]])


def test_seed_boundaries():
    validate_run(seeds=5, identities=28, episodes=5)
    for kwargs in (
        dict(seeds=0, identities=28, episodes=5),
        dict(seeds=5, identities=1, episodes=5),
        dict(seeds=5, identities=28, episodes=16),
        dict(seeds=5, identities=125_001, episodes=5),
        dict(seeds=100, identities=28, episodes=5),
    ):
        with pytest.raises(ValueError):
            validate_run(**kwargs)


def _episode_record():
    return {
        "survived": 1, "steps": 400, "death_need": None, "scored_ticks": 10,
        "scores": {cell: {"regret": 0.03 if cell == "reset_rate" else 0.01,
                           "rate_mae": 0.002} for cell in CELLS},
        "word_differences": {"reset": 2, "swapped": 3},
        "birth_rate_mae": {cell: 0.001 if cell == "persistent_rate" else 0.003
                           for cell in CELLS},
        "readings": 2, "evidence_ticks": 399,
    }


def test_resume_after_interruption_preserves_donor_assignment_and_skips_work(tmp_path):
    calls = []
    fail = True

    def episode_runner(_model, _organism, **kwargs):
        calls.append(("calibration", kwargs["identity_seed"]))
        return {**_episode_record(), "own_memory": {"identity": kwargs["identity_seed"]},
                "donor_memory": {}}

    def cell_runner(_model, _organism, **kwargs):
        nonlocal fail
        if fail and kwargs["cell"] == "persistent_rate":
            fail = False
            raise RuntimeError("interrupted")
        calls.append((kwargs["cell"], kwargs["identity_seed"]))
        assert kwargs["own_prior"]["identity"] == kwargs["identity_seed"]
        assert kwargs["donor_prior"]["identity"] != kwargs["identity_seed"]
        return {"episodes": [_episode_record()] * kwargs["episodes"],
                "final_evidence_ticks": 798, "final_readings": 4}

    out = tmp_path / "survey.json"
    kwargs = dict(parent_fingerprint="parent-A", seeds=1, identities=2, episodes=1,
                  out=out, episode_runner=episode_runner, cell_runner=cell_runner)
    organism = OrganismConfig()
    with pytest.raises(RuntimeError, match="interrupted"):
        run_survey(object(), organism, **kwargs)
    partial = json.loads(out.read_text())
    assert partial["completed_units"] == 3
    assert not partial["complete"]
    result = run_survey(object(), organism, **kwargs)
    assert result["complete"]
    assert result["gates"] is None
    assert len(calls) == 10
    assert result["cells"][0]["matched_regret_value"] == pytest.approx(0.002)
    assert result["cells"][0]["birth_rate_mae_value"] == pytest.approx(0.002)
    assert result["cells"][0]["word_difference_share"] == pytest.approx(0.2)
    assert result["cells"][0]["per_episode_survival"] == [1.0]
    run_survey(object(), organism, **kwargs)
    assert len(calls) == 10
    with pytest.raises(ValueError, match="parent checkpoint changed"):
        run_survey(object(), organism, **{**kwargs, "parent_fingerprint": "parent-B"})
    with pytest.raises(ValueError, match="configuration"):
        run_survey(object(), replace(organism, report=ReportConfig(life_steps=300)), **kwargs)
    saved = json.loads(out.read_text())
    saved["units"].append(saved["units"][0])
    out.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="duplicate"):
        run_survey(object(), organism, **kwargs)


def test_real_survey_resume_matches_uninterrupted_execution(tmp_path):
    parent = "runs/organism/probe52_guided_report_lexicon/adult/organism_report_seed1.npz"
    try:
        model, organism = load_organism_checkpoint(parent)
    except (FileNotFoundError, OSError):  # pragma: no cover - artifact absent
        pytest.skip("The probe52 lexical parent checkpoint is not present.")
    kwargs = dict(parent_fingerprint="real-parent-resume-guard", seeds=1,
                  identities=2, episodes=2)
    expected = run_survey(model, organism, **kwargs)
    completed = []

    def interrupt_after_computing_cell(*args, **cell_kwargs):
        result = run_identity_cell(*args, **cell_kwargs)
        completed.append((cell_kwargs["identity_seed"], cell_kwargs["cell"]))
        if len(completed) == 2:
            # The second real trajectory ran, but its complete identity/cell
            # unit never reached the atomic checkpoint.
            raise RuntimeError("interrupted after real trajectory")
        return result

    out = tmp_path / "real-survey.json"
    with pytest.raises(RuntimeError, match="interrupted after real trajectory"):
        run_survey(model, organism, out=out, cell_runner=interrupt_after_computing_cell,
                   **kwargs)
    partial = json.loads(out.read_text())
    assert len(partial["calibration"]) == 2
    assert len(partial["units"]) == 1
    actual = run_survey(model, organism, out=out,
                        cell_runner=interrupt_after_computing_cell, **kwargs)
    assert actual == expected
    assert completed.count((3_200_000_000, "reset_rate")) == 1
    assert completed.count((3_200_000_000, "persistent_rate")) == 2
    calls = len(completed)
    assert run_survey(model, organism, out=out,
                      cell_runner=interrupt_after_computing_cell, **kwargs) == expected
    assert len(completed) == calls
