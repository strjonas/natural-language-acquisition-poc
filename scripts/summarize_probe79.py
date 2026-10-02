"""Reconstruct Probe79's matched-history gates without importing the learner."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean

from summarize_probe78 import agrees, contrast

DIRECTORY = Path(__file__).resolve().parents[1] / "runs/organism/probe79_evidence_reset"
READOUTS = ("retained","adaptive","muted","scrambled","oracle_reset","true_rate")
DETECTORS = ("adaptive","muted","scrambled")
COHORTS = ("stable","changed")


def blocks_from_raw(data):
    m=data["manifest"]
    if (m["seeds"],m["identities"],m["episodes"],m["diagnostic"],m["identity_base"],m["episode_base"]) != (5,28,5,False,4_400_000_000,4_500_000_000):
        raise ValueError("not the locked sample and bands")
    units=data["units"]
    expected={(b,i,c) for b in range(5) for i in range(28) for c in COHORTS}
    keyed={(u["block"],u["identity"],u["cohort"]):u for u in units}
    if len(units)!=280 or set(keyed)!=expected:
        raise ValueError("incomplete or duplicate units")
    warm=data["calibration"]
    if len(warm)!=140 or {(r["block"],r["identity"]) for r in warm}!={(b,i) for b in range(5) for i in range(28)}:
        raise ValueError("incomplete calibration")
    for r in warm:
        if r["episode_seed"]!=4_500_000_000+r["block"]*2_000_000+r["identity"]*16:
            raise ValueError("incorrect calibration seed")
    for (b,i,c),u in keyed.items():
        if len(u["episodes"])!=5:
            raise ValueError("incomplete episodes")
        for e,r in enumerate(u["episodes"],1):
            if r["episode"]!=e or r["episode_seed"]!=4_500_000_000+b*2_000_000+i*16+e:
                raise ValueError("incorrect episode seed")
            if r["detector_resets"]["muted"]:
                raise ValueError("muted detector reset")
            if any(r["scores"][p]["muted"]!=r["scores"][p]["retained"] for p in ("all","after_change")):
                raise ValueError("muted episode scores differ from retained")
        if c=="changed":
            cutoff=u["schedule"]["episode"]-1
            if u["episodes"][:cutoff]!=keyed[b,i,"stable"]["episodes"][:cutoff]:
                raise ValueError("cohorts differ before physical change")
    blocks={}
    for c in COHORTS:
        blocks[c]=[]
        for b in range(5):
            group=[keyed[b,i,c] for i in range(28)]
            rows=[r for u in group for r in u["episodes"]]
            panels={p:{"ticks":sum(r["counts"][p] for r in rows),
                       "totals":{n:{metric:sum(r["scores"][p][n][metric] for r in rows)
                                    for metric in ("rate_mae","regret","word_changes")}
                                 for n in READOUTS}} for p in ("all","after_change")}
            blocks[c].append({"identities":28,"episodes":140,
                "survivors":sum(r["survived"] for r in rows),"steps":sum(r["steps"] for r in rows),
                "readings":sum(r["readings"] for r in rows),"panels":panels,
                "reset_identities":{n:sum(any(r["detector_resets"][n] for r in u["episodes"]) for u in group) for n in DETECTORS},
                "resets":{n:sum(len(r["detector_resets"][n]) for r in rows) for n in DETECTORS}})
    return blocks


def reconstruct(blocks):
    if set(blocks)!=set(COHORTS) or any(len(bs)!=5 for bs in blocks.values()):
        raise ValueError("wrong independent blocks")
    for bs in blocks.values():
        for b in bs:
            if (b["identities"],b["episodes"])!=(28,140) or b["panels"]["all"]["ticks"]<=0:
                raise ValueError("wrong sample or missing scores")
            if any(not 0<=v<=28 for v in b["reset_identities"].values()):
                raise ValueError("invalid identity-level resets")
    def values(cohort,name,metric):
        return [b["panels"]["all"]["totals"][name][metric]/b["panels"]["all"]["ticks"] for b in blocks[cohort]]
    contrasts,reductions={},{}
    for metric in ("rate_mae","regret"):
        adaptive=values("changed","adaptive",metric)
        for baseline in ("retained","scrambled"):
            contrasts[f"{baseline}_minus_adaptive_{metric}"]=contrast([x-y for x,y in zip(values("changed",baseline,metric),adaptive,strict=True)])
        base=mean(values("changed","retained",metric))
        reductions[metric]=(base-mean(adaptive))/base if base>0 else 0.
    stable_mae=mean(values("stable","adaptive","rate_mae"))
    stable_base=mean(values("stable","retained","rate_mae"))
    stable_cost=mean(values("stable","adaptive","regret"))-mean(values("stable","retained","regret"))
    alarms=sum(b["reset_identities"]["adaptive"] for b in blocks["stable"])/140
    muted=all(b["panels"][p]["totals"]["muted"]==b["panels"][p]["totals"]["retained"]
              for bs in blocks.values() for b in bs for p in ("all","after_change"))
    positive=lambda c: c["low"]>0 and c["positive"]>=4
    gates={
        "G1_changed_improvement":all(positive(contrasts[f"retained_minus_adaptive_{m}"]) and reductions[m]>=.2 for m in ("rate_mae","regret")),
        "G2_stable_preservation":stable_mae<=1.1*stable_base and stable_cost<=.001 and alarms<=.1,
        "G3_controls":muted and all(positive(contrasts[f"scrambled_minus_adaptive_{m}"]) for m in ("rate_mae","regret")),
    }
    return {"blocks":blocks,"contrasts":contrasts,"changed_reductions":reductions,
            "stable_mae":stable_mae,"stable_baseline_mae":stable_base,"stable_regret_cost":stable_cost,
            "stable_false_alarm_share":alarms,"muted_exact":muted,"gates":gates,"all_pass":all(gates.values())}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path)
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    source=args.input or DIRECTORY/("survey.json" if (DIRECTORY/"survey.json").exists() else "evidence.json")
    data=json.loads(source.read_text())
    if not data["complete"]:
        raise ValueError("incomplete run")
    blocks=data["block_evidence"] if "block_evidence" in data else blocks_from_raw(data)
    summary=reconstruct(blocks)
    if not agrees(summary,data["summary"]):
        raise ValueError("saved summary or gates disagree")
    if args.out:
        compact={"complete":True,"manifest":data["manifest"],"raw_sha256":sha256(source.read_bytes()).hexdigest(),
                 "block_evidence":blocks,"summary":summary,
                 "boundary":"Block reconstruction; episode seed and pre-change parity checks occurred at extraction. Common driver survival is not a detector effect."}
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(compact,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k!="blocks"},indent=2))
    print("Saved summary and gates independently reconstructed")


if __name__=="__main__":
    main()
