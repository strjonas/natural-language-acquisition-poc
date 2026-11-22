"""Why does the shared-convention speaking learner die in held-out bodies?

Diagnostic for docs/decisions/2026-11-21-caregiver-v2-sickness-and-held-out-pair.md
(results 2026-11-22). Held-out bodies differ from training bodies in three
ways at once: channel 0 is social *together with another* social resource
(training: when channel 0 is social it is the only one), channel 0 drains
fast (rate factor 1.3-1.6 against 0.4-0.7), and channel 0 is born at the
family's birth level rather than 0.75/0.85. This script removes those
differences one at a time on the same held-out identities and reports, for
the trained learner and the reference speakers: survival, what the learner
asks for, which resource it dies of, and when.

    PYTHONPATH=src .venv/bin/python scripts/held_out_diagnosis.py runs/organism/speaking_self_dev/v2r_shared_s0
"""

import copy
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

from homesocial.organism import caregiver_v1 as cv
from homesocial.organism import caregiver_v2 as c2
from homesocial.organism import speaking_self as ss
from homesocial.organism.vec_body import BASE_RATES, CHANNELS, LIFE_TICKS, LIVES

PER_CELL = 300
HOLD = c2.HOLD_CHANNEL


def held_out(per_cell: int) -> cv.V1Bodies:
    parts = []
    for k, social in c2.HOLD_CELLS:
        idents = [ss.HOLD_BASE + k * 1_000_000 + social * 100_000 + i * 10 for i in range(3 * per_cell)]
        b = c2.hold_bodies(idents, [k] * len(idents), [social] * len(idents), c2.TEST_FACTOR)
        parts.append(b.take(np.arange(min(per_cell, b.n))))
    return ss._concat(parts)


def variant(base: cv.V1Bodies, *, slow=False, alone=False, train_birth=False) -> cv.V1Bodies:
    b = copy.deepcopy(base)
    if slow:  # map the held-out rate factor 1.3..1.6 linearly onto training's 0.4..0.7
        f = b.rates[:, HOLD] / BASE_RATES[HOLD]
        lo, hi = c2.TEST_FACTOR
        tlo, thi = c2.TRAIN_FACTOR
        b.rates[:, HOLD] = BASE_RATES[HOLD] * (tlo + (f - lo) * (thi - tlo) / (hi - lo))
    if alone:  # channel 0 the only social resource, as in training
        b.blocked[:] = False
        b.blocked[:, HOLD] = True
    if train_birth:
        born = np.random.default_rng(0).choice(np.asarray(c2.TRAIN_BIRTH), (b.n, LIVES))
        b.birth[:, :, HOLD] = born
    b.ration[:] = cv.RATION_FACTOR * cv.species_need(b.blocked)
    return b


def run_learned(model, bodies: cv.V1Bodies, rng: np.random.Generator) -> dict:
    env = ss.make_env(bodies, "v2r")
    n, rows = env.n, np.arange(env.n)
    deaths = {"cause": [], "tick": [], "row": []}
    end = env._end_lives

    def record(index, survived):
        index = np.asarray(index)
        failed = index[~np.broadcast_to(np.asarray(survived), index.shape)]
        if len(failed):
            lv = np.where(bodies.active[failed], env.levels[failed], np.inf)
            deaths["cause"].extend(lv.argmin(1).tolist())
            deaths["tick"].extend(env.tick[failed].tolist())
            deaths["row"].extend(failed.tolist())
        end(index, survived)

    env._end_lives = record
    said = -np.ones(n, dtype=np.int64)
    h = mx.zeros((n, model.hidden))
    req = {"hold": np.zeros(n), "other_social": np.zeros(n), "own": np.zeros(n), "nothing": np.zeros(n),
           "visits": np.zeros(n), "spoken": np.zeros(n)}
    hold_sizes, other_sizes, hold_level_asked, hold_level_visit = np.zeros(3), np.zeros(3), [], []
    while not env.finished.all():
        h, a_logits, _, t_logits, w1_logits = model.act(mx.array(ss.features(env, said)), h, mx.zeros((n, 1)))
        mx.eval(h, a_logits, t_logits, w1_logits)
        action = ss._sample(np.array(a_logits), rng)[0]
        talk = ss._sample(np.array(t_logits), rng)[0]
        w1 = ss._sample(np.array(w1_logits), rng)[0]
        w2 = ss._sample(np.array(model.second(h, mx.array(np.eye(ss.W1, dtype=np.float32)[w1]))), rng)[0]
        present = env.present()
        talk = talk * present
        spoken = talk > 0
        channel = bodies.word_channel[rows, w1]
        social = spoken & (channel >= 0) & bodies.blocked[rows, np.maximum(channel, 0)]
        hold = social & (channel == HOLD)
        other = social & (channel != HOLD)
        req["visits"] += present
        req["spoken"] += spoken
        req["hold"] += hold
        req["other_social"] += other
        req["own"] += spoken & (channel >= 0) & ~social
        req["nothing"] += spoken & (channel < 0)
        meaning = bodies.mod_meaning[rows, w2]
        np.add.at(hold_sizes, meaning[hold], 1)
        np.add.at(other_sizes, meaning[other], 1)
        hold_level_asked.extend(env.levels[hold, HOLD].tolist())
        hold_level_visit.extend(env.levels[present & bodies.blocked[:, HOLD], HOLD].tolist())
        speech = np.where(spoken, w1 * ss.W2 + w2, -1)
        env.step(action, speech)
        said = np.where(env.birth & (env.life == 0), -1, speech)
    spoken = max(req["spoken"].sum(), 1)
    cause = np.asarray(deaths["cause"], dtype=np.int64)
    dead_rows = np.asarray(deaths["row"], dtype=np.int64)
    hold_social = bodies.blocked[dead_rows, HOLD] if len(dead_rows) else np.zeros(0, bool)
    other_social = (bodies.blocked[dead_rows, cause] & (cause != HOLD)) if len(dead_rows) else np.zeros(0, bool)
    return {
        "survival": float(env.survived.mean()),
        "survival_by_life": [float(v) for v in env.survived.mean(0)],
        "speak_rate": float(req["spoken"].sum() / max(req["visits"].sum(), 1)),
        "request_share": {k: float(req[k].sum() / spoken) for k in ("hold", "other_social", "own", "nothing")},
        "hold_size_share": [float(v) for v in hold_sizes / max(hold_sizes.sum(), 1)],
        "other_social_size_share": [float(v) for v in other_sizes / max(other_sizes.sum(), 1)],
        "hold_level_when_asked": float(np.mean(hold_level_asked)) if hold_level_asked else None,
        "hold_level_at_visits": float(np.mean(hold_level_visit)) if hold_level_visit else None,
        "deaths": len(cause),
        "death_cause_share": {
            "channel0": float(np.mean(cause == HOLD)) if len(cause) else 0.0,
            "other_social": float(np.mean(other_social)) if len(cause) else 0.0,
            "own": float(np.mean((cause != HOLD) & ~other_social)) if len(cause) else 0.0,
        },
        "channel0_social_in_dead": float(np.mean(hold_social)) if len(cause) else 0.0,
        "median_death_tick": float(np.median(deaths["tick"])) if deaths["tick"] else None,
        "life_ticks": LIFE_TICKS,
    }


def references(bodies: cv.V1Bodies) -> dict:
    return {arm: float(c2.run_arm(bodies, arm, ration=True)["survived"].mean())
            for arm in ("silent", "single_0.4", "composed", "no_held_out_pair")}


def main() -> None:
    run = Path(sys.argv[1])
    trainer, state = ss.Trainer.load(run / "checkpoint")
    if trainer.c.convention != "shared" or trainer.c.world != "v2r":
        raise SystemExit("needs a v2r shared-convention checkpoint")
    base = held_out(PER_CELL)
    variants = {
        "held_out (fast, ch0 + other social)": {},
        "slow ch0, ch0 + other social": {"slow": True},
        "fast ch0, ch0 the only social": {"alone": True},
        "slow ch0, ch0 the only social": {"slow": True, "alone": True},
        "training-like (slow, alone, training births)": {"slow": True, "alone": True, "train_birth": True},
    }
    out = {"decisions": state["decisions"], "per_cell": PER_CELL, "cells": [list(c) for c in c2.HOLD_CELLS],
           "variants": {}}
    for name, kw in variants.items():
        b = variant(base, **kw)
        cells = {}
        for k, social in c2.HOLD_CELLS:
            sel = np.flatnonzero(base.k == k)
            sub = b.take(sel)
            cells[f"K{k}"] = {"identities": int(sub.n), "reference": references(sub),
                              "learned": run_learned(trainer.model, sub, np.random.default_rng([0, 22, k]))}
        out["variants"][name] = cells
        print(f"\n== {name}")
        for cell, d in cells.items():
            L, R = d["learned"], d["reference"]
            print(f"  {cell} n={d['identities']}  learned {L['survival']:.3f}  composed* {R['composed']:.3f}  "
                  f"no pair* {R['no_held_out_pair']:.3f}  single.4* {R['single_0.4']:.3f}  silent* {R['silent']:.3f}")
            print(f"      speak {L['speak_rate']:.2f}  asks: " + "  ".join(f"{k} {v:.2f}" for k, v in L["request_share"].items())
                  + f"  | ch0 sizes {np.round(L['hold_size_share'], 2).tolist()} other {np.round(L['other_social_size_share'], 2).tolist()}")
            print(f"      ch0 level asked {L['hold_level_when_asked']}  at visits {L['hold_level_at_visits']}  "
                  f"| dies of: " + "  ".join(f"{k} {v:.2f}" for k, v in L["death_cause_share"].items())
                  + f"  median tick {L['median_death_tick']}/{LIFE_TICKS}")
    (run / "held_out_diagnosis.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
