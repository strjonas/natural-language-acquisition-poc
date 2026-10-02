"""Independently reconstruct Probe78 from episode records or compact block sums.

Standard library only. No organism, MLX, or treatment scorer imports.
"""
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
CELLS = ("retained", "episodic", "forgetting", "oracle_reset", "true_rate")
COHORTS = ("stable", "changed")
DIRECTORY = ROOT / "runs/organism/probe78_changing_body"


def contrast(values):
    if len(values) != 5 or not all(math.isfinite(v) for v in values):
        raise ValueError("five finite independent block differences required")
    average = mean(values)
    half = 2.776 * stdev(values) / math.sqrt(5)
    return {"mean": average, "low": average-half, "high": average+half,
            "n": 5, "values": values, "positive": sum(v > 0 for v in values),
            "negative": sum(v < 0 for v in values),
            "same_sign": sum(v*average > 0 for v in values)}


def agrees(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(agrees(v, b[k]) for k,v in a.items())
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(agrees(x,y) for x,y in zip(a,b))
    if isinstance(a, (int, float)) and not isinstance(a, bool):
        return isinstance(b, (int,float)) and math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12)
    return a == b


def grade(cohorts):
    changed = cohorts["changed"]
    retained = changed["retained"]["blocks"]
    contrasts = {}
    for name, key in (("true_rate", "C1_true_rate_survival"), ("oracle_reset", "C2_oracle_reset_survival")):
        contrasts[key] = contrast([a["survival"]-b["survival"]
                                   for a,b in zip(changed[name]["blocks"], retained, strict=True)])
    for metric in ("rate_mae", "regret"):
        differences = []
        for block in retained:
            panel = block["panels"]["all"]
            if panel["ticks"] <= 0:
                raise ValueError("no scored decisions")
            differences.append((panel["totals"]["retained"][metric]-panel["totals"]["oracle_reset"][metric])/panel["ticks"])
        contrasts[f"C3_oracle_reset_{metric}"] = contrast(differences)
    gates = {k: v["low"] > 0 and v["positive"] >= 4 for k,v in contrasts.items()}
    return {"cohorts": cohorts, "contrasts": contrasts, "stable_exact_parity": True,
            "gates": gates, "all_pass": all(gates.values())}


def from_raw(data):
    manifest = data["manifest"]
    if (manifest["seeds"], manifest["identities"], manifest["episodes"], manifest["diagnostic"]) != (5,28,5,False):
        raise ValueError("not the locked survey")
    if (manifest["identity_base"],manifest["episode_base"]) != (4_000_000_000,4_100_000_000):
        raise ValueError("wrong survey bands")
    units = data["units"]
    expected = {(b,i,c,a) for b in range(5) for i in range(28) for c in COHORTS for a in CELLS}
    keyed = {(u["block"],u["identity"],u["cohort"],u["cell"]):u for u in units}
    if len(units) != len(expected) or set(keyed) != expected:
        raise ValueError("duplicate or incomplete units")
    warm = {(r["block"],r["identity"]):r for r in data["calibration"]}
    if len(data["calibration"]) != 140 or set(warm) != {(b,i) for b in range(5) for i in range(28)}:
        raise ValueError("duplicate or incomplete calibration")
    for (b,i), record in warm.items():
        if record["episode_seed"] != 4_100_000_000+b*2_000_000+i*16:
            raise ValueError("wrong calibration seed")
    for (b,i,c,a), unit in keyed.items():
        if len(unit["episodes"]) != 5:
            raise ValueError("incomplete episode sequence")
        for e, row in enumerate(unit["episodes"],1):
            if row["episode"] != e or row["episode_seed"] != 4_100_000_000+b*2_000_000+i*16+e:
                raise ValueError("incorrect episode or seed")
            if row["survived"] not in (0,1) or row["steps"] <= 0:
                raise ValueError("invalid physical episode")
            for panel in ("all", "after_change"):
                if not 0 <= row["counts"][panel] <= row["steps"]:
                    raise ValueError("invalid decision count")
                for scores in row["scores"][panel].values():
                    if any(not math.isfinite(x) or x < -1e-12 for x in scores.values()):
                        raise ValueError("invalid score")
        if c == "stable" and a == "retained":
            if unit["episodes"] != keyed[b,i,c,"oracle_reset"]["episodes"]:
                raise ValueError("stationary sham parity failed")
        if c == "changed":
            # Before the randomized change episode, the two cohorts have the
            # same body, memories, motor randomness and physical observations.
            schedule = unit.get("schedule")
            if schedule is not None:
                if schedule != keyed[b,i,"stable",a].get("schedule"):
                    raise ValueError("cohorts carry different physical schedules")
                cutoff = schedule["episode"] - 1
                if unit["episodes"][:cutoff] != keyed[b,i,"stable",a]["episodes"][:cutoff]:
                    raise ValueError("cohorts diverged before physical change")
    cohorts = {}
    for cohort in COHORTS:
        cells = {}
        for cell in CELLS:
            blocks = []
            for b in range(5):
                rows = [r for i in range(28) for r in keyed[b,i,cohort,cell]["episodes"]]
                panels = {p: {"ticks": sum(r["counts"][p] for r in rows),
                              "totals": {a: {m: sum(r["scores"][p][a][m] for r in rows)
                                              for m in ("regret","rate_mae","word_changes")}
                                         for a in CELLS}} for p in ("all","after_change")}
                blocks.append({"survived": sum(r["survived"] for r in rows), "episodes":140,
                               "survival":sum(r["survived"] for r in rows)/140,
                               "steps":sum(r["steps"] for r in rows),
                               "readings":sum(r["readings"] for r in rows),
                               "exposed_episodes":sum(r["physical_change_exposed"] for r in rows),
                               "within_life_cues":sum(r["reset_tick"] is not None and r["reset_tick"]>0 for r in rows),
                               "birth_cues":sum(r["reset_tick"]==0 for r in rows), "panels":panels})
            cells[cell] = {"survival":mean(b["survival"] for b in blocks),
                           "survivors":sum(b["survived"] for b in blocks), "blocks":blocks}
        cohorts[cohort] = cells
    return grade(cohorts)


def from_compact(data):
    cohorts = data["block_evidence"]
    if set(cohorts) != set(COHORTS):
        raise ValueError("incorrect cohorts")
    for cohort,cells in cohorts.items():
        if set(cells) != set(CELLS):
            raise ValueError("incorrect cells")
        for cell in cells.values():
            if len(cell["blocks"]) != 5:
                raise ValueError("five independent blocks required")
            for b in cell["blocks"]:
                if b["episodes"] != 140 or not 0 <= b["survived"] <= 140:
                    raise ValueError("invalid episode counts")
                if not agrees(b["survival"],b["survived"]/140):
                    raise ValueError("inconsistent survival")
            if cell["survivors"] != sum(b["survived"] for b in cell["blocks"]):
                raise ValueError("incorrect total survivors")
            if not agrees(cell["survival"],mean(b["survival"] for b in cell["blocks"])):
                raise ValueError("incorrect mean survival")
    if not agrees(cohorts["stable"]["retained"],cohorts["stable"]["oracle_reset"]):
        raise ValueError("stationary sham parity failed")
    return grade(cohorts)


def verify(data):
    if not data.get("complete"):
        raise ValueError("incomplete run")
    result = from_compact(data) if "block_evidence" in data else from_raw(data)
    if not agrees(result,data["summary"]):
        raise ValueError("stored summary or gates disagree")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path)
    parser.add_argument("--out",type=Path)
    args = parser.parse_args()
    source = args.input or (DIRECTORY / ("survey.json" if (DIRECTORY/"survey.json").exists() else "evidence.json"))
    data = json.loads(source.read_text())
    result = verify(data)
    if args.out:
        compact = {"complete":True, "manifest":data["manifest"],
                   "raw_sha256":sha256(source.read_bytes()).hexdigest(),
                   "block_evidence":result["cohorts"], "summary":result,
                   "boundary":"Block arithmetic can be reconstructed here; individual seed checks occurred during raw extraction."}
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(compact,indent=2,sort_keys=True)+"\n")
    for cohort,cells in result["cohorts"].items():
        print(cohort,{c:v["survivors"] for c,v in cells.items()},"out of 700 each")
    for name,value in result["contrasts"].items():
        print(name, value["mean"], [value["low"],value["high"]], result["gates"][name])
    print("all_pass =",result["all_pass"],"; stored summaries and gates verified")


if __name__ == "__main__":
    main()
