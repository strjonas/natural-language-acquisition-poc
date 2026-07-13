# Organism learning probe 1: from-scratch grounded, 200k steps

Date: 2026-07-13
Goal-level metric: survival learning (prerequisite of gate G1).

Setup: `homesocial.organism.harness --train-steps 200000 --language-modes
grounded --eval-episodes 20 --seed 1`. Defaults: entropy 0.02, hidden 256,
segment 64, live-reward 0.02, death penalty 1.0.

Result: NEGATIVE (expected direction, mechanics healthy).

- 3500 lives; mean life length 54.5 (first 100 lives) -> 57.2 (all),
  occasional 90+ step lives late in training; loss decreasing.
- Eval: survival 0%, mean steps 56.5, mean viability 0.673 (vs random
  baseline 56.6 / 0.548 — viability is higher than random, so the policy
  is not degenerate, but no reliable drinking).
- Bottleneck is exploration: outliving the ~54-step thirst clock needs the
  approach -> face -> consume chain plus kind knowledge; entropy 0.02 does
  not find it.

Next (mitigation order from PLAN section 7): probe 2 = entropy 0.06 at 400k
steps; probe 3 = live-reward/curiosity variation; then BC bootstrap with
mandatory ablation. Timebox: after 3 negative probes, question the
substrate (e.g., metabolism curriculum: gentler thirst early in training),
not the hyperparameters.
