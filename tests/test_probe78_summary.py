from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from test_changing_body import units_fixture
from homesocial.organism.changing_body import summarize

SPEC = importlib.util.spec_from_file_location("independent_probe78",Path(__file__).resolve().parents[1]/"scripts/summarize_probe78.py")
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


def raw_fixture():
    units = units_fixture()
    for u in units:
        for e,r in enumerate(u["episodes"],1):
            r["episode_seed"] = 4_100_000_000+u["block"]*2_000_000+u["identity"]*16+e
    return {"complete":True,"manifest":{"seeds":5,"identities":28,"episodes":5,"diagnostic":False,
            "identity_base":4_000_000_000,"episode_base":4_100_000_000},"units":units,
            "calibration":[{"block":b,"identity":i,"episode_seed":4_100_000_000+b*2_000_000+i*16}
                           for b in range(5) for i in range(28)],
            "summary":summarize(units,seeds=5,identities=28)}


def test_raw_and_compact_independently_reconstruct_every_gate():
    raw = raw_fixture()
    result = checker.verify(raw)
    assert result["all_pass"]
    compact = {"complete":True,"block_evidence":result["cohorts"],"summary":result}
    assert checker.verify(compact) == result


@pytest.mark.parametrize("corruption",("duplicate","seed","summary","nonfinite","incomplete"))
def test_malformed_evidence_is_rejected(corruption):
    raw = raw_fixture()
    if corruption == "duplicate":
        raw["units"][-1] = deepcopy(raw["units"][0])
    elif corruption == "seed":
        raw["units"][0]["episodes"][0]["episode_seed"] += 1
    elif corruption == "summary":
        raw["summary"]["gates"]["C2_oracle_reset_survival"] = False
    elif corruption == "nonfinite":
        raw["units"][0]["episodes"][0]["scores"]["all"]["retained"]["regret"] = float("nan")
    else:
        raw["complete"] = False
    with pytest.raises(ValueError):
        checker.verify(raw)


def test_regret_uses_decision_weighting_not_episode_ratio_means():
    raw = raw_fixture()
    for u in raw["units"]:
        if u["cohort"] == "changed" and u["cell"] == "retained":
            u["episodes"][0]["counts"]["all"] = 50
    raw["summary"] = summarize(raw["units"],seeds=5,identities=28)
    result = checker.verify(raw)
    assert result["contrasts"]["C3_oracle_reset_regret"]["mean"] == pytest.approx(5/90)


def test_compact_survivor_totals_are_checked():
    raw = raw_fixture()
    result = checker.verify(raw)
    compact = {"complete":True,"block_evidence":deepcopy(result["cohorts"]),"summary":result}
    compact["block_evidence"]["changed"]["retained"]["survivors"] = 4
    with pytest.raises(ValueError,match="total survivors"):
        checker.verify(compact)
