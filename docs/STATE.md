# STATE

Last rewritten: 2026-07-13. Rewrite this file, never append.

## Where the frontier is

Probe era closed and tagged `v0-probe-era` (2026-07-13). Established results
are summarized in `results_synthesis.md`; direction in
`DIRECTION_2026-07-12.md`; engineering plan in `PLAN_ORGANISM.md`.

Current phase: **C1/C2 groundwork** — creole language package + utterance
bank (workstream A), island environment with recalibrated difficulty and
token channel (workstream B), organism agent skeleton (workstream C).

## What is next

1. Finish workstream A: closed vocab, situation schema, OpenRouter bank
   generation (`deepseek/deepseek-v4-flash`, reasoning disabled, cached),
   validator tests.
2. Workstream B1: port extended-renewable ecology to `island/world.py`;
   tune until scripted oracle survives >= 95% of 1000-step lives.
3. Workstream C: organism model + online loop + smoke test.

## What is forbidden (closed lines)

- Appending to `grand_architecture_roadmap.md` or `evaluation2045.md`
  (frozen historical logs).
- The recovery/calibration line (risk gates, recovery policies, floor
  calibrators) — closed with rationale in DIRECTION section 2.
- New one-off probe scripts in `src/homesocial/` top level.
- Live LLM calls inside the training loop; LLM-as-agent-mind.
- Rewarding language directly.

## Resources

- OpenROUTER key in `.env` (gitignored, throwaway, limited). Model:
  `deepseek/deepseek-v4-flash` with `reasoning: {"enabled": false}`.
  Verified working 2026-07-13; ~ $2e-6 per short call.
- GPU ask deferred until gates G1+G2 pass (see DIRECTION section 6).
