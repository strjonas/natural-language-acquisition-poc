"""Share of the learned agent's actuator choices that match the oracle rule, by life.

Development diagnostic for docs/decisions/2026-11-20-learned-self-budget-experiment.md.
    PYTHONPATH=src .venv/bin/python scripts/learned_self_actions.py
"""
from pathlib import Path
import numpy as np, mlx.core as mx
from homesocial.organism.learned_self import Trainer, dev_identities, features, sample_actions
from homesocial.organism.vec_body import VecBody, bodies_from_identities, LIVES, ACTUATORS
tr, _ = Trainer.load(Path("runs/organism/learned_self_dev/checkpoint"))
ids, ks = dev_identities(200, ks=(3, 4, 5))
b = bodies_from_identities(ids, ks)
env = VecBody(b); n = env.n; rng = np.random.default_rng(1)
h = mx.zeros((n, tr.model.hidden))
cat = np.zeros((LIVES, 4)); rows = np.arange(n)
large = b.large_actuator()
while not env.finished.all():
    h, logits, _ = tr.model.act(mx.array(features(env)), h, mx.zeros((n, 1)))
    mx.eval(h, logits)
    a = sample_actions(np.array(logits), rng)[0]
    o = env.oracle_action()
    live = np.flatnonzero(~env.finished)
    useful = b.actionable()[rows, a]
    same_target = useful & (b.target[rows, a] == b.target[rows, o])
    is_large = useful & (large[rows, b.target[rows, a]] == a)
    for i in live:
        l = env.life[i]
        cat[l] += [a[i] == o[i], same_target[i] and a[i] != o[i], useful[i] and not same_target[i], not useful[i]]
    env.step(a)
cat /= cat.sum(1, keepdims=True)
print("K3-K5, learned policy, share of decisions per life")
print("life  =oracle  same-resource-other-portion  other-resource  useless-actuator")
for l in range(LIVES): print(l+1, *[f"{v:8.3f}" for v in cat[l]])
print("survival by K:", {k: round(float(env.survived[b.k == k].mean()), 3) for k in (3, 4, 5)})
