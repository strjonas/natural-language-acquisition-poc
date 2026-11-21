"""Diagnostic for docs/decisions/2026-11-21-caregiver-v1-ceiling-survey.md.

Two questions about a trained ``speaking_self`` checkpoint, at caregiver visits:

1. Does the recurrent state encode how full each social resource is?
   Ridge regression from the state to the true level, fitted on half the
   identities and scored on the other half (R^2, mean absolute error).
2. Does the chosen size word follow the named resource's true deficit?
   Size shares by deficit bin, next to the reference speaker's rule.

    PYTHONPATH=src .venv/bin/python scripts/speaking_probe.py runs/organism/speaking_self_dev/scratch_s0
"""

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

from homesocial.organism import caregiver_v1 as cv
from homesocial.organism import speaking_self as ss

CELLS = ((2, 1), (3, 1), (4, 1), (3, 2), (4, 2))


def collect(model, bodies, rng):
    env = cv.V1Body(bodies, ss.VARIANT)
    n, rows = env.n, np.arange(env.n)
    said = -np.ones(n, dtype=np.int64)
    h = mx.zeros((n, model.hidden))
    out = {"h": [], "level": [], "row": [], "deficit": [], "meaning": [], "social_named": []}
    while not env.finished.all():
        h, a_logits, _, t_logits, w1_logits = model.act(mx.array(ss.features(env, said)), h, mx.zeros((n, 1)))
        mx.eval(h, a_logits, t_logits, w1_logits)
        action = ss._sample(np.array(a_logits), rng)[0]
        talk = ss._sample(np.array(t_logits), rng)[0]
        w1 = ss._sample(np.array(w1_logits), rng)[0]
        w2 = ss._sample(np.array(model.second(h, mx.array(np.eye(ss.W1, dtype=np.float32)[w1]))), rng)[0]
        present = env.present()
        talk = talk * present
        hs = np.array(h)
        for i in np.flatnonzero(present):
            social = np.flatnonzero(bodies.blocked[i])
            out["h"].append(hs[i])
            out["level"].append(np.sort(env.levels[i, social]))  # lowest social level first
            out["row"].append(i)
            c = bodies.word_channel[i, w1[i]]
            named = talk[i] > 0 and c >= 0 and bodies.blocked[i, c]
            out["social_named"].append(named)
            out["deficit"].append(1.0 - env.levels[i, c] if named else np.nan)
            out["meaning"].append(bodies.mod_meaning[i, w2[i]] if named else -1)
        speech = np.where(talk > 0, w1 * ss.W2 + w2, -1)
        env.step(action, speech)
        said = speech
    return {k: np.array(v) for k, v in out.items()}


def ridge_r2(x, y, train, test, lam=1.0):
    mu, sd = x[train].mean(0), x[train].std(0) + 1e-6
    xs = (x - mu) / sd
    xs = np.concatenate([xs, np.ones((len(xs), 1))], 1)
    w = np.linalg.solve(xs[train].T @ xs[train] + lam * np.eye(xs.shape[1]), xs[train].T @ y[train])
    pred = xs[test] @ w
    resid = ((y[test] - pred) ** 2).sum()
    total = ((y[test] - y[train].mean()) ** 2).sum()
    return float(1 - resid / total), float(np.abs(y[test] - pred).mean()), float(np.abs(y[test] - y[train].mean()).mean())


def main():
    run = Path(sys.argv[1])
    trainer, state = ss.Trainer.load(run / "checkpoint")
    report = {"checkpoint": str(run), "decisions": state["decisions"], "cells": {}}
    for k, social in CELLS:
        bodies = cv.v1_bodies(*ss.dev_identities(200, cells=((k, social),)))
        d = collect(trainer.model, bodies, np.random.default_rng(5))
        train = d["row"] % 2 == 0
        test = ~train
        level = d["level"][:, 0]
        r2, mae, base = ridge_r2(d["h"], level, train, test)
        named = d["social_named"]
        bins = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]
        by_bin = {}
        for lo, hi in bins:
            m = named & (d["deficit"] >= lo) & (d["deficit"] < hi)
            shares = np.bincount(d["meaning"][m], minlength=3) / max(m.sum(), 1)
            by_bin[f"{lo:.1f}-{hi:.1f}"] = {"n": int(m.sum()), "shares_0.2_0.4_0.6": [round(float(s), 3) for s in shares]}
        report["cells"][f"K{k}_social{social}"] = {
            "visits": int(len(level)), "lowest_social_level": {"r2": r2, "mae": mae, "mae_of_mean": base},
            "size_by_deficit": by_bin}
        print(f"K{k} social{social}: level R2 {r2:.3f}, MAE {mae:.3f} (mean-only {base:.3f}); size shares by deficit:",
              "  ".join(f"{b}: {v['shares_0.2_0.4_0.6']} (n={v['n']})" for b, v in by_bin.items()), flush=True)
    (run / "probe.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__" and len(sys.argv) == 2:
    main()


def word_meaning_probe(run: Path) -> None:
    """Can the state at late-life visits tell which modifier word means 0.2 / 0.6
    for this identity? Ridge on one-hot targets, argmax accuracy, held-out half."""

    trainer, _ = ss.Trainer.load(run / "checkpoint")
    for k, social in ((2, 1), (3, 2)):
        bodies = cv.v1_bodies(*ss.dev_identities(200, cells=((k, social),)))
        env = cv.V1Body(bodies, ss.VARIANT)
        n = env.n
        rng = np.random.default_rng(6)
        said = -np.ones(n, dtype=np.int64)
        h = mx.zeros((n, trainer.model.hidden))
        xs, ys, rows = [], [], []
        while not env.finished.all():
            h, a, _, t, w1 = trainer.model.act(mx.array(ss.features(env, said)), h, mx.zeros((n, 1)))
            mx.eval(h)
            action, talk, w1s = (ss._sample(np.array(z), rng)[0] for z in (a, t, w1))
            w2 = ss._sample(np.array(trainer.model.second(h, mx.array(np.eye(ss.W1, dtype=np.float32)[w1s]))), rng)[0]
            late = env.present() & (env.life >= 3)
            hs = np.array(h)
            for i in np.flatnonzero(late):
                xs.append(hs[i]); ys.append(bodies.meaning_mod[i]); rows.append(i)
            talk = talk * env.present()
            speech = np.where(talk > 0, w1s * ss.W2 + w2, -1)
            env.step(action, speech)
            said = speech
        x, y, r = np.array(xs), np.array(ys), np.array(rows)
        train, test = r % 2 == 0, r % 2 == 1
        for m, name in ((0, "0.2"), (2, "0.6")):
            target = np.eye(3)[y[:, m]]
            mu, sd = x[train].mean(0), x[train].std(0) + 1e-6
            z = np.concatenate([(x - mu) / sd, np.ones((len(x), 1))], 1)
            w = np.linalg.solve(z[train].T @ z[train] + np.eye(z.shape[1]), z[train].T @ target[train])
            acc = float(((z[test] @ w).argmax(1) == y[test, m]).mean())
            print(f"K{k} social{social}: which word means {name}, decoded at lives 4-6: accuracy {acc:.3f} (chance 0.333)")


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "words":
    word_meaning_probe(Path(sys.argv[1]))
