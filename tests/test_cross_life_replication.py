"""Seed isolation, source provenance and resume guards for Probe75."""

from copy import deepcopy
import inspect
import json
import sys

import pytest

from homesocial.organism import cross_life_replication as replication
from homesocial.organism import cross_life_self as survey
from homesocial.organism.train import OrganismConfig


def _episode_record(episode_seed: int, cell: str = "reset_rate") -> dict:
    return {
        "episode_seed": episode_seed,
        "survived": int(cell in ("persistent_rate", "true_rate")),
        "steps": 400, "death_need": None, "scored_ticks": 10,
        "scores": {name: {"regret": 0.03 if name == "reset_rate" else 0.01,
                          "rate_mae": 0.002} for name in survey.CELLS},
        "word_differences": {"reset": 2, "swapped": 3},
        "birth_rate_mae": {name: 0.001 if name == "persistent_rate" else 0.003
                           for name in survey.CELLS},
        "readings": 2, "evidence_ticks": 399,
    }


def _runners(calls):
    def episode_runner(_model, _organism, **kwargs):
        calls.append(("calibration", deepcopy(kwargs)))
        return {
            **_episode_record(kwargs["episode_seed"]),
            "own_memory": {"identity": kwargs["identity_seed"]}, "donor_memory": {},
        }

    def cell_runner(_model, _organism, **kwargs):
        calls.append(("cell", deepcopy(kwargs)))
        assert kwargs["own_prior"]["identity"] == kwargs["identity_seed"]
        return {
            "episodes": [_episode_record(kwargs["episode_base"] + episode, kwargs["cell"])
                         for episode in range(1, kwargs["episodes"] + 1)],
            "final_evidence_ticks": 399 * (kwargs["episodes"] + 1),
            "final_readings": 2 * (kwargs["episodes"] + 1),
        }

    return {"episode_runner": episode_runner, "cell_runner": cell_runner}


def test_probe74_defaults_still_use_original_bands():
    for function in (survey.validate_run, survey.run_survey):
        parameters = inspect.signature(function).parameters
        assert parameters["identity_seed_base"].default == 3_200_000_000
        assert parameters["episode_seed_base"].default == 3_300_000_000
    calls = []
    result = survey.run_survey(
        None, OrganismConfig(), parent_fingerprint="parent", seeds=1,
        identities=2, episodes=1, **_runners(calls),
    )
    assert result["configuration"]["identity_seed_base"] == 3_200_000_000
    assert result["configuration"]["episode_seed_base"] == 3_300_000_000
    assert calls[0][1]["identity_seed"] == 3_200_000_000
    assert calls[0][1]["episode_seed"] == 3_300_000_000
    assert calls[2][1]["episode_base"] == 3_300_000_000


def test_confirmation_reuses_the_exact_mechanism_grade_and_locked_sample():
    assert replication.run_episode is survey.run_episode
    assert replication.run_identity_cell is survey.run_identity_cell
    assert replication.run_survey is survey.run_survey
    assert replication.grade is survey.grade
    calls = []
    result = replication.run_replication(
        None, OrganismConfig(), parent_fingerprint="parent", **_runners(calls),
    )
    config = result["configuration"]
    assert (config["seeds"], config["identities"], config["test_episodes"]) == (5, 28, 5)
    assert (config["identity_seed_base"], config["episode_seed_base"]) == (
        3_600_000_000, 3_700_000_000,
    )
    assert len(result["calibration"]) == 140
    assert len(result["units"]) == 560
    assert all(cell["test_episodes"] == 700 for cell in result["cells"])
    assert result["gates"] == survey.grade(result["cells"])
    assert result["gates"]["all_pass"]
    assert result["ungated_true_minus_persistent_survival"]["values"] == [0.0] * 5
    assert config["mechanism_fingerprint"] == survey.mechanism_fingerprint()
    # A complete paired block lands before the next block begins. All donor
    # priors come from that block's cyclic successor, including the wrap.
    for block in range(5):
        group = calls[block * 140:(block + 1) * 140]
        assert [kind for kind, _ in group[:28]] == ["calibration"] * 28
        for identity in range(28):
            expected_identity = 3_600_000_000 + block * 2_000_000 + identity
            expected_episode = 3_700_000_000 + block * 2_000_000 + identity * 16
            calibration = group[identity][1]
            assert calibration["identity_seed"] == expected_identity
            assert calibration["episode_seed"] == expected_episode
            cells = group[28 + identity * 4:28 + (identity + 1) * 4]
            assert [kwargs["cell"] for _, kwargs in cells] == list(survey.CELLS)
            for _, kwargs in cells:
                assert kwargs["identity_seed"] == expected_identity
                assert kwargs["episode_base"] == expected_episode
                assert kwargs["donor_prior"]["identity"] == (
                    3_600_000_000 + block * 2_000_000 + (identity + 1) % 28
                )
    for unit in result["units"]:
        base = 3_700_000_000 + unit["seed_index"] * 2_000_000 + unit["identity"] * 16
        assert [row["episode_seed"] for row in unit["episodes"]] == list(range(base + 1, base + 6))


def test_diagnostic_is_small_disjoint_and_cannot_produce_gates():
    calls = []
    result = replication.run_diagnostic(
        None, OrganismConfig(), parent_fingerprint="parent", **_runners(calls),
    )
    config = result["configuration"]
    assert (config["seeds"], config["identities"], config["test_episodes"]) == (1, 2, 1)
    assert (config["identity_seed_base"], config["episode_seed_base"]) == (
        3_400_000_000, 3_500_000_000,
    )
    assert len(calls) == 10
    assert result["gates"] is None


@pytest.mark.parametrize("kwargs", [
    {"identity_seed_base": -1},
    {"episode_seed_base": -1},
    {"identity_seed_base": True},
    {"episodes": 1.5},
    {"identity_seed_base": 3_700_000_000, "episode_seed_base": 3_700_000_000},
    {"identity_seed_base": 3_700_000_017, "episode_seed_base": 3_700_000_000},
    {"identity_seed_base": 3_700_000_000, "episode_seed_base": 3_700_000_001},
    {"identity_seed_base": 2**32},
    {"episode_seed_base": 2**32},
    {"identities": 125_001},
    {"seeds": 2, "identity_seed_base": 1_000, "episode_seed_base": 100_000},
])
def test_invalid_or_overlapping_bands_are_rejected(kwargs):
    arguments = dict(seeds=1, identities=2, episodes=1,
                     identity_seed_base=3_600_000_000, episode_seed_base=3_700_000_000)
    with pytest.raises(ValueError):
        survey.validate_run(**{**arguments, **kwargs})


def test_seed_boundaries_include_derived_offsets_and_allow_reversed_disjoint_bands():
    survey.validate_run(seeds=5, identities=28, episodes=5,
                        identity_seed_base=3_600_000_000, episode_seed_base=3_700_000_000)
    # The final motor seed can equal UINT32_MAX, but cannot exceed it.
    episode_base = 2**32 - 1 - 59_000_003 - 17
    survey.validate_run(seeds=1, identities=2, episodes=1,
                        identity_seed_base=3_600_000_000, episode_seed_base=episode_base)
    with pytest.raises(ValueError, match="32-bit"):
        survey.validate_run(seeds=1, identities=2, episodes=1,
                            identity_seed_base=3_600_000_000, episode_seed_base=episode_base + 1)
    identity_base = 2**32 - 1 - 9_900_023 - 1
    survey.validate_run(seeds=1, identities=2, episodes=1,
                        identity_seed_base=identity_base, episode_seed_base=1_000_000)
    with pytest.raises(ValueError, match="32-bit"):
        survey.validate_run(seeds=1, identities=2, episodes=1,
                            identity_seed_base=identity_base + 1, episode_seed_base=1_000_000)
    # Closed seed bands touch only when they share an actual seed.
    survey.validate_run(seeds=1, identities=2, episodes=1,
                        identity_seed_base=1_000_000, episode_seed_base=1_000_002)


def test_partial_resume_pins_bands_parent_and_mechanism(tmp_path, monkeypatch):
    calls = []
    runners = _runners(calls)
    intact_runner = runners["cell_runner"]
    fail = True

    def interrupted(_model, _organism, **kwargs):
        nonlocal fail
        if fail and kwargs["cell"] == "persistent_rate":
            fail = False
            raise RuntimeError("interrupted")
        return intact_runner(_model, _organism, **kwargs)

    path = tmp_path / "replication.json"
    kwargs = dict(parent_fingerprint="parent", seeds=1, identities=2, episodes=1,
                  out=path, **{**runners, "cell_runner": interrupted})
    with pytest.raises(RuntimeError, match="interrupted"):
        replication.run_replication(None, OrganismConfig(), **kwargs)
    partial = json.loads(path.read_text())
    assert partial["completed_units"] == 3
    assert partial["configuration"]["schema"] == 2
    assert partial["configuration"]["mechanism_fingerprint"] == survey.mechanism_fingerprint()
    completed_calls = len(calls)
    with monkeypatch.context() as patch:
        patch.setattr(survey, "mechanism_fingerprint", lambda: "changed-source")
        with pytest.raises(ValueError, match="mechanism"):
            replication.run_replication(None, OrganismConfig(), **kwargs)
    with pytest.raises(ValueError, match="parent checkpoint changed"):
        replication.run_replication(None, OrganismConfig(), **{**kwargs, "parent_fingerprint": "other"})
    for key in ("identity_seed_base", "episode_seed_base"):
        with pytest.raises(ValueError, match="configuration"):
            survey.run_survey(
                None, OrganismConfig(), **kwargs,
                **{name: value + int(name == key) for name, value in (
                    ("identity_seed_base", 3_600_000_000), ("episode_seed_base", 3_700_000_000),
                )},
            )
    assert len(calls) == completed_calls
    assert json.loads(path.read_text()) == partial
    result = replication.run_replication(None, OrganismConfig(), **kwargs)
    assert result["complete"] and result["gates"] is None
    assert len(calls) == 10
    replication.run_replication(None, OrganismConfig(), **kwargs)
    assert len(calls) == 10
    assert json.loads(path.read_text())["ungated_true_minus_persistent_survival"] == (
        result["ungated_true_minus_persistent_survival"]
    )


def _legacy_file(path, calls):
    kwargs = dict(parent_fingerprint="parent", seeds=1, identities=2, episodes=1,
                  out=path, **_runners(calls))
    result = survey.run_survey(None, OrganismConfig(), **kwargs)
    result["configuration"]["schema"] = 1
    result["configuration"].pop("mechanism_fingerprint")
    path.write_text(json.dumps(result))
    return kwargs, result


def test_completed_legacy_probe74_is_readable_without_new_episodes_or_relabeling(tmp_path):
    calls = []
    path = tmp_path / "legacy.json"
    kwargs, saved = _legacy_file(path, calls)
    original = path.read_bytes()
    result = survey.run_survey(None, OrganismConfig(), **kwargs)
    assert result == saved
    assert len(calls) == 10
    assert path.read_bytes() == original


@pytest.mark.parametrize("incomplete_flag", [True, False])
def test_legacy_partial_data_cannot_resume_even_if_marked_complete(tmp_path, incomplete_flag):
    calls = []
    path = tmp_path / "legacy.json"
    kwargs, saved = _legacy_file(path, calls)
    if incomplete_flag:
        saved["complete"] = False
    else:
        saved["units"].pop()
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="legacy progress"):
        survey.run_survey(None, OrganismConfig(), **kwargs)
    assert len(calls) == 10


def test_source_fingerprint_covers_source_bytes_and_relative_paths(tmp_path):
    root = tmp_path / "package"
    root.mkdir()
    dependency = root / "dependency.py"
    dependency.write_text("def simulate():\n    return 1\n")
    fingerprint = survey.mechanism_fingerprint(root)
    assert len(fingerprint) == 64
    assert fingerprint == survey.mechanism_fingerprint(root)
    (root / "result.json").write_text("unrelated artifact")
    assert fingerprint == survey.mechanism_fingerprint(root)
    dependency.write_text("def simulate():\n    return 2\n")
    assert fingerprint != survey.mechanism_fingerprint(root)
    dependency.write_text("def simulate():\n    return 1\n")
    dependency.rename(root / "other.py")
    assert fingerprint != survey.mechanism_fingerprint(root)
    with pytest.raises(ValueError, match="source package"):
        survey.mechanism_fingerprint(tmp_path / "missing")


def test_parent_fingerprint_covers_weights_and_configuration(tmp_path):
    path = tmp_path / "parent.npz"
    path.write_bytes(b"weights")
    config = path.with_suffix(".npz.json")
    config.write_text('{"life_steps":400}')
    fingerprint = survey.checkpoint_fingerprint(path)
    config.write_text('{"life_steps":300}')
    assert fingerprint != survey.checkpoint_fingerprint(path)
    config.write_text('{"life_steps":400}')
    path.write_bytes(b"different weights")
    assert fingerprint != survey.checkpoint_fingerprint(path)


def test_probe74_cli_threads_explicit_seed_bases(monkeypatch):
    captured = {}
    monkeypatch.setattr(sys, "argv", [
        "cross_life_self", "--identity-seed-base", "3600000000",
        "--episode-seed-base", "3700000000", "--seeds", "1",
        "--identities", "2", "--episodes", "1",
    ])
    monkeypatch.setattr(survey, "load_organism_checkpoint", lambda path: (None, OrganismConfig()))
    monkeypatch.setattr(survey, "checkpoint_fingerprint", lambda path: "parent")

    def runner(model, organism, **kwargs):
        captured.update(kwargs)
        return {"cells": [], "gates": None}

    monkeypatch.setattr(survey, "run_survey", runner)
    survey.main()
    assert captured["identity_seed_base"] == 3_600_000_000
    assert captured["episode_seed_base"] == 3_700_000_000


def test_confirmation_cli_does_not_allow_seed_band_overrides(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["cross_life_replication", "--identity-seed-base", "3200000000"])
    with pytest.raises(SystemExit) as error:
        replication.main()
    assert error.value.code == 2
