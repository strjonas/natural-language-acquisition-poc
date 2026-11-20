import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "summarize_probe80.py"
spec = importlib.util.spec_from_file_location("summarize_probe80", SCRIPT)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def _write_units(directory: Path, *, flip_one=False):
    manifest = {"blocks": 5, "identities": 1, "lives": 6}
    for b in range(5):
        for k in range(1, 6):
            arms = {}
            for arm in checker.ARMS:
                lives = []
                for life in range(6):
                    good = arm in ("posterior", "uncritical", "reset", "certainty", "oracle", "oracle_structure") \
                        or (arm == "bandit" and k == 1) or (arm == "random" and k == 1 and b < 2)
                    rec = {"life": life, "survived": good, "ticks": 400 if good else 50}
                    if arm in checker.POSTERIOR_ARMS:
                        exact = arm in ("posterior", "uncritical") or (arm == "reset" and life < 3)
                        confidence = 1.0 if (exact and arm != "uncritical") else (0.6 + 0.01 * b if arm == "uncritical" else 0.0)
                        rec.update({"exact": exact, "confidence": confidence, "resource_count": k,
                                    "falsified": False, "rebuilds": 0})
                    lives.append(rec)
                arms[arm] = {"lives": lives}
            if flip_one and b == 0 and k == 1:
                arms["posterior"]["lives"][5]["exact"] = False
            row = {"block": b, "k": k, "identity": 0, "arms": arms, "permutation": {"transports": True}}
            (directory / f"survey_block{b}_k{k}.json").write_text(
                json.dumps({"manifest": manifest, "block": b, "rows": [row]}))


def test_compact_evidence_reconstructs_the_same_gates_as_raw_units(tmp_path):
    _write_units(tmp_path)
    rows, manifest = checker.load(tmp_path, "survey")
    raw = checker.reconstruct(rows, manifest)
    evidence = checker.compact(tmp_path, "survey")
    assert set(evidence["raw_sha256"]) == {f"survey_block{b}_k{k}.json" for b in range(5) for k in range(1, 6)}
    expanded = checker.reconstruct(*checker.expand(json.loads(json.dumps(evidence))))
    assert expanded["gates"] == raw["gates"]
    assert expanded["calibration"] == raw["calibration"]
    assert expanded["table"] == raw["table"]


def test_checker_refuses_incomplete_or_malformed_surveys(tmp_path):
    _write_units(tmp_path)
    (tmp_path / "survey_block4_k5.json").unlink()
    rows, manifest = checker.load(tmp_path, "survey")
    with pytest.raises(SystemExit):
        checker.reconstruct(rows, manifest)


def test_gates_respond_to_the_data(tmp_path):
    _write_units(tmp_path)
    good = checker.reconstruct(*checker.load(tmp_path, "survey"))
    _write_units(tmp_path, flip_one=True)
    worse = checker.reconstruct(*checker.load(tmp_path, "survey"))
    assert good["table"][1]["posterior"]["exact_life6"] == 1.0
    assert worse["table"][1]["posterior"]["exact_life6"] == pytest.approx(0.8)
