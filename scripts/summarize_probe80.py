"""Independent reconstruction of Probe80's locked gates from raw unit records.

Standard library only. Imports nothing from the runner or the mechanism. Reads
the per (block, K) progress files, rebuilds every table, paired interval and
gate from the per-life records, and optionally compares with the runner's
graded output.

    python3 scripts/summarize_probe80.py [--dir runs/organism/probe80_active_self] [--mode survey]
    python3 scripts/summarize_probe80.py --write-compact runs/organism/probe80_active_self/evidence.json
    python3 scripts/summarize_probe80.py --input runs/organism/probe80_active_self/evidence.json

The compact evidence keeps, per identity, arm and life, exactly the fields the
gates use (survival, ticks; for posterior arms exactness, confidence, reported
count, falsified flag and rebuilds), plus each raw unit file's SHA-256. It does
not keep partitions, evidence traces or timings.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

ARMS = ("random", "bandit", "posterior", "uncritical", "reset", "certainty", "yoked", "oracle_structure", "oracle")
POSTERIOR_ARMS = ("posterior", "uncritical", "reset", "certainty", "yoked")
CORE = (1, 2, 3, 4)
T_CRITICAL = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}


def load(directory: Path, mode: str) -> tuple[list[dict], dict]:
    rows, manifests = [], []
    for path in sorted(glob.glob(str(directory / f"{mode}_block*_k*.json"))):
        saved = json.loads(Path(path).read_text())
        manifests.append(saved["manifest"])
        rows.extend(saved["rows"])
    if not rows:
        raise SystemExit("no unit records found")
    if any(m != manifests[0] for m in manifests):
        raise SystemExit("units were run under different manifests")
    return rows, manifests[0]


def interval(values: list[float]) -> dict:
    m = mean(values)
    if len(values) < 2:
        return {"mean": m, "low": m, "high": m, "positive": sum(v > 0 for v in values), "values": values}
    half = T_CRITICAL[len(values) - 1] * stdev(values) / math.sqrt(len(values))
    return {"mean": m, "low": m - half, "high": m + half,
            "positive": sum(v > 0 for v in values), "values": values}


def positive(c: dict) -> bool:
    return c["mean"] > 0 and c["low"] > 0 and c["positive"] >= 4


def block_mean(rows, block, arm, metric, ks, lives):
    values = [metric(rec) for r in rows if r["block"] == block and r["k"] in ks
              for rec in r["arms"][arm]["lives"] if rec["life"] in lives]
    return mean(values)


def reconstruct(rows: list[dict], manifest: dict) -> dict:
    blocks, identities, n_lives = manifest["blocks"], manifest["identities"], manifest["lives"]
    expected = {(b, k, i) for b in range(blocks) for k in range(1, 6) for i in range(identities)}
    seen = [(r["block"], r["k"], r["identity"]) for r in rows]
    if set(seen) != expected or len(seen) != len(expected):
        raise SystemExit(f"incomplete survey: {len(set(seen))}/{len(expected)} identities")
    for r in rows:
        for arm in ARMS:
            lives = r["arms"][arm]["lives"]
            if [rec["life"] for rec in lives] != list(range(n_lives)):
                raise SystemExit("malformed life records")
            for rec in lives:
                if not 1 <= rec["ticks"] <= 400 or rec["survived"] != (rec["ticks"] == 400 and rec["survived"]):
                    raise SystemExit("inconsistent survival record")
                if arm in POSTERIOR_ARMS and not 0.0 <= rec["confidence"] <= 1.0:
                    raise SystemExit("confidence outside [0, 1]")
    all_lives, last = range(n_lives), (n_lives - 1,)
    survived = lambda rec: float(rec["survived"])
    exact = lambda rec: float(rec["exact"])
    brier = lambda rec: (rec["confidence"] - rec["exact"]) ** 2

    def contrast(a, b, metric, ks=CORE, lives=all_lives):
        return interval([block_mean(rows, blk, a, metric, ks, lives) - block_mean(rows, blk, b, metric, ks, lives)
                         for blk in range(blocks)])

    table = {}
    for k in range(1, 6):
        table[k] = {arm: {"survival": mean(block_mean(rows, b, arm, survived, (k,), all_lives) for b in range(blocks))}
                    for arm in ARMS}
        for arm in POSTERIOR_ARMS:
            table[k][arm]["exact_life6"] = mean(block_mean(rows, b, arm, exact, (k,), last) for b in range(blocks))
            table[k][arm]["exact_life1"] = mean(block_mean(rows, b, arm, exact, (k,), (0,)) for b in range(blocks))
            table[k][arm]["count_life6"] = mean(r["arms"][arm]["lives"][-1]["resource_count"] for r in rows if r["k"] == k)
    reports = [(rec["confidence"], rec["exact"]) for r in rows if r["k"] in CORE
               for rec in r["arms"]["posterior"]["lives"]]
    high = [e for c, e in reports if c >= 0.9]
    low = [e for c, e in reports if c <= 0.1]
    calibration = {"brier": mean((c - e) ** 2 for c, e in reports),
                   "high_n": len(high), "high_accuracy": mean(high) if high else None,
                   "low_n": len(low), "low_accuracy": mean(low) if low else None}
    c = {
        "random": {k: contrast("posterior", "random", survived, (k,)) for k in range(1, 6)},
        "bandit": {k: contrast("posterior", "bandit", survived, (k,)) for k in range(1, 6)},
        "yoked": contrast("posterior", "yoked", survived),
        "reset": contrast("posterior", "reset", survived, lives=range(1, n_lives)),
        "certainty_exact": contrast("posterior", "certainty", exact, lives=last),
        "uncritical_brier": contrast("uncritical", "posterior", brier),
    }
    yoked_own = mean(float(r["arms"]["yoked"]["lives"][-1]["exact"]) for r in rows if r["k"] in CORE)
    permutation = [r["permutation"]["transports"] for r in rows if "permutation" in r]
    counts = [table[k]["posterior"]["count_life6"] for k in range(1, 6)]
    gates = {
        "G1_discovery": all(table[k]["posterior"]["exact_life6"] >= 0.80 for k in (1, 2, 3))
        and table[4]["posterior"]["exact_life6"] >= 0.40 and all(a < b for a, b in zip(counts, counts[1:])),
        "G2_action_uses_self": all(positive(c["random"][k]) for k in CORE),
        "G3_structure_not_action_values": all(positive(c["bandit"][k]) for k in (2, 3, 4)),
        "G4_own_body": positive(c["yoked"]) and yoked_own <= 0.10,
        "G5_uncertainty_drives_discovery": positive(c["certainty_exact"]) and c["certainty_exact"]["mean"] >= 0.30,
        "G6_persistence": positive(c["reset"]),
        "G7_calibrated_self_report": calibration["brier"] <= 0.10 and calibration["high_accuracy"] is not None
        and calibration["high_accuracy"] >= 0.90 and (calibration["low_accuracy"] is None or calibration["low_accuracy"] <= 0.10),
        "G8_self_criticism_load_bearing": positive(c["uncritical_brier"]),
        "G9_permutation": len(permutation) > 0 and all(permutation),
    }
    return {"table": table, "contrasts": c, "calibration": calibration, "yoked_own_exact": yoked_own,
            "permutation_controls": len(permutation), "gates": gates, "all_pass": all(gates.values())}


POSTERIOR_FIELDS = ("exact", "confidence", "resource_count", "falsified", "rebuilds")


def compact(directory: Path, mode: str) -> dict:
    rows, manifest = load(directory, mode)
    hashes = {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
              for p in sorted(glob.glob(str(directory / f"{mode}_block*_k*.json")))}
    out = []
    for r in sorted(rows, key=lambda r: (r["block"], r["k"], r["identity"])):
        arms = {}
        for arm in ARMS:
            lives = []
            for rec in r["arms"][arm]["lives"]:
                item = [rec["life"], int(rec["survived"]), rec["ticks"]]
                if arm in POSTERIOR_ARMS:
                    item += [int(rec["exact"]), rec["confidence"], rec["resource_count"],
                             int(rec["falsified"]), rec["rebuilds"]]
                lives.append(item)
            arms[arm] = lives
        entry = {"block": r["block"], "k": r["k"], "identity": r["identity"], "arms": arms}
        if "permutation" in r:
            entry["permutation"] = bool(r["permutation"]["transports"])
        out.append(entry)
    return {"probe": 80, "manifest": manifest, "raw_sha256": hashes, "rows": out}


def expand(evidence: dict) -> tuple[list[dict], dict]:
    rows = []
    for entry in evidence["rows"]:
        arms = {}
        for arm, lives in entry["arms"].items():
            records = []
            for item in lives:
                rec = {"life": item[0], "survived": bool(item[1]), "ticks": item[2]}
                if arm in POSTERIOR_ARMS:
                    rec.update(dict(zip(POSTERIOR_FIELDS, item[3:])))
                    rec["exact"] = bool(rec["exact"])
                    rec["falsified"] = bool(rec["falsified"])
                records.append(rec)
            arms[arm] = {"lives": records}
        row = {"block": entry["block"], "k": entry["k"], "identity": entry["identity"], "arms": arms}
        if "permutation" in entry:
            row["permutation"] = {"transports": entry["permutation"]}
        rows.append(row)
    return rows, evidence["manifest"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("runs/organism/probe80_active_self"))
    parser.add_argument("--mode", default="survey")
    parser.add_argument("--input", type=Path, help="reconstruct from compact evidence instead of raw units")
    parser.add_argument("--write-compact", type=Path, help="write compact evidence from the raw units and exit")
    args = parser.parse_args()
    if args.write_compact:
        payload = compact(args.dir, args.mode)
        args.write_compact.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
        print(f"wrote {args.write_compact} ({args.write_compact.stat().st_size} bytes)")
        return
    if args.input:
        rows, manifest = expand(json.loads(args.input.read_text()))
    else:
        rows, manifest = load(args.dir, args.mode)
    result = reconstruct(rows, manifest)
    print("survival (six-life mean); exact partition after life 6 for posterior arms")
    print("K  " + " ".join(f"{a[:10]:>10s}" for a in ARMS))
    for k in range(1, 6):
        print(f"K{k} " + " ".join(f"{result['table'][k][a]['survival']:10.3f}" for a in ARMS))
    for k in range(1, 6):
        t = result["table"][k]
        print(f"K{k} exact life1/life6: " + "  ".join(
            f"{a}={t[a]['exact_life1']:.2f}/{t[a]['exact_life6']:.2f}" for a in POSTERIOR_ARMS)
              + f"  count={t['posterior']['count_life6']:.2f}")
    for name, value in result["contrasts"].items():
        items = value.items() if isinstance(next(iter(value.values())), dict) else [("", value)]
        for k, ci in items:
            print(f"{name}{'[K'+str(k)+']' if k != '' else ''}: {ci['mean']:+.4f} [{ci['low']:+.4f}, {ci['high']:+.4f}] positive {ci['positive']}/5")
    print("calibration K1-K4:", json.dumps(result["calibration"]))
    print("yoked exact on own body K1-K4:", result["yoked_own_exact"], "| permutation controls:", result["permutation_controls"])
    print(json.dumps(result["gates"], indent=2), "\nALL PASS:", result["all_pass"])
    graded = args.dir / f"{args.mode}_graded.json"
    if graded.exists() and not args.input:
        runner = json.loads(graded.read_text())["grade"]["gates"]
        agree = all(runner[name] == value for name, value in result["gates"].items())
        print("agrees with runner's grade:", agree)
        if not agree:
            raise SystemExit("independent reconstruction disagrees with the runner")


if __name__ == "__main__":
    main()
