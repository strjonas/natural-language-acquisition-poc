"""Recompute Probe77 aggregate gates from saved per-body scores; no MLX.

This checks aggregation, not the effect extraction or grouping algorithm.
Compact evidence retains every per-body score and drops episode transcripts.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean

from summarize_probe78 import agrees

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "runs/organism/probe77_sequential_structure"


def reconstruct(rows):
    expected = {(b,k,i) for b in range(5) for k in range(1,6) for i in range(20)}
    if len(rows) != 500 or {(r["block"],r["k"],r["identity"]) for r in rows} != expected:
        raise ValueError("incomplete or duplicate worlds")
    result = {}
    for budget in ("1","6"):
        by_k = {}
        for k in range(1,6):
            chosen = [r for r in rows if r["k"] == k]
            values = [r["budgets"][budget] for r in chosen]
            keys = ("exact","count","count_exact","balanced_accuracy","active_recall",
                    "active_precision","supported_fraction","actual_ticks","observations","survival")
            by_k[str(k)] = {key:mean(v[key] for v in values) for key in keys}
            by_k[str(k)]["shuffled_exact"] = mean(v["shuffled"]["exact"] for v in values)
            by_k[str(k)]["per_block_exact"] = [mean(r["budgets"][budget]["exact"] for r in chosen if r["block"] == b) for b in range(5)]
        reductions = []
        for b in range(5):
            scores = [r["budgets"][budget]["heldout"] for r in rows if r["block"] == b]
            base = sum(s["baseline_sse"] for s in scores)
            reductions.append((base - sum(s["model_sse"] for s in scores))/base if base else 0.)
        remap = {str(k):mean(r["challenges"]["remap"]["exact"] for r in rows if r["k"] == k) for k in range(2,6)}
        stale = {str(k):mean(r["challenges"]["remap"]["stale_wrong"] for r in rows if r["k"] == k) for k in range(2,6)}
        gates = {
            "G1_partition": all(v["exact"] >= .8 and sum(x >= .7 for x in v["per_block_exact"]) >= 4 for v in by_k.values()),
            "G2_structure": all(by_k[str(k)]["balanced_accuracy"] >= .8 for k in range(2,6)) and all(by_k[str(k)]["count"] < by_k[str(k+1)]["count"] for k in range(1,5)),
            "G3_controls": all(v["exact"]-v["shuffled_exact"] >= .2 for v in by_k.values()) and all(r["budgets"][budget]["permutation_exact"] for r in rows),
            "G4_prediction": sum(x>0 for x in reductions) >= 4 and mean(reductions) >= .1,
            "G5_remapping": all(x>=.8 for x in remap.values()) and all(x>=.8 for x in stale.values()),
        }
        result[budget] = {"by_k":by_k,"prediction_reduction_per_block":reductions,
                          "remap_exact":remap,"stale_wrong":stale,"gates":gates,"all_pass":all(gates.values())}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path)
    parser.add_argument("--out",type=Path)
    args = parser.parse_args()
    source = args.input or (DIRECTORY / ("survey.json" if (DIRECTORY/"survey.json").exists() else "evidence.json"))
    data = json.loads(source.read_text())
    if not data["complete"] or data["manifest"]["diagnostic"]:
        raise ValueError("not a complete survey")
    summary = reconstruct(data["rows"])
    if not agrees(summary,data["summary"]):
        raise ValueError("saved summary or gates disagree")
    if args.out:
        compact = {"complete":True,"manifest":data["manifest"],"summary":summary,
                   "raw_sha256":sha256(source.read_bytes()).hexdigest(),
                   "rows":[{k:v for k,v in r.items() if k != "episodes"} for r in data["rows"]],
                   "boundary":"Reconstructs aggregate gates from per-body scores; does not rerun grouping or physics."}
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(compact,separators=(",",":"))+"\n")
    print("500 bodies; aggregate summaries and gates verified")
    for budget, row in summary.items():
        print("episodes",budget,"exact partitions",{k:v["exact"] for k,v in row["by_k"].items()},"gates",row["gates"])


if __name__ == "__main__":
    main()
