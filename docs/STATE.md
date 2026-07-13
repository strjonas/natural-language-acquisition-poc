# STATE

Last rewritten: 2026-07-13 (evening). Rewrite this file, never append.

## Where the frontier is

Probe era closed and tagged `v0-probe-era`. Direction:
`DIRECTION_2026-07-12.md`. Plan: `PLAN_ORGANISM.md`. Synthesis of old
results: `results_synthesis.md`.

Today's build (all committed, all tests green — 162 tests):

- **Workstream A DONE.** `homesocial/creole/`: closed ~70-word vocabulary,
  63 enumerable speech-act situations, strict OOV/required-token validation,
  seeded sampling, shuffled control. LLM-generated utterance bank committed
  at `src/homesocial/creole/data/utterance_bank.jsonl` (13.5 variants/key,
  cost $0.005 total, cache in `runs/organism/bank_cache/`).
- **Workstream B1/B2 DONE, gate B1 PASSED.** `homesocial/island/`: surface
  names are percepts, bodily kinds assigned per world (berry may be food or
  poison — only the caregiver's label tells). Renewable ecology, 1000-step
  lives. Calibration: oracle survival 100% (60 episodes), random 0% (dies
  ~step 54). Caregiver speaks bank tokens; grounded/silent/shuffled matched
  shapes; ObsPacket hides kinds (tested).
- **Workstream C skeleton DONE.** `homesocial/organism/`: one MLX model
  (token GRU + recurrent core; policy/value/next-obs/next-needs/reward/
  next-utterance heads), online lifelong A2C+GAE with per-segment updates,
  hidden persists within a life, `harness.py` is the one benchmark command.
  Throughput ~650 env steps/sec with updates.

## What is next

Three learning probes are done and recorded in
`decisions/2026-07-13-organism-probes2-3.md`: entropy 0.02/0.06 and an
infancy metabolism curriculum all fail at consume-interaction discovery
(lives track the metabolic clock; the agent learns resting but never
drinking). This replicates probe-era failure F1 in the new substrate.

1. **Implement the BC-bootstrap childhood** (sanctioned by PLAN section 4
   C4): clone ~50-200 oracle lives into the organism (motor competence
   only; the oracle ignores tokens, and per-world kinds are not in its
   observable state, so the language effect stays uncontaminated), then
   continue online lifelong RL. Mandatory with/without ablation.
   Alternative if BC contaminates: caregiver OFFER-based feeding of
   starving infants (in-loop, developmentally natural).
2. When survival learning works: G1 comparison grounded vs silent vs
   shuffled via `organism.harness` on held-out seeds.
3. Then caregiver refinements (B3/B4) and fast-mapping probes (gate G2).

## What is forbidden (closed lines)

- Appending to `grand_architecture_roadmap.md` / `evaluation2045.md`.
- The recovery/calibration line (see DIRECTION section 2).
- New one-off probe scripts at `src/homesocial/` top level.
- Live LLM calls in the training loop; LLM-as-agent-mind.
- Rewarding language directly.

## Resources

- OpenRouter key in `.env` (gitignored, throwaway). Model
  `deepseek/deepseek-v4-flash`, `reasoning: {"enabled": false}`. Bank
  regeneration is incremental via the cache.
- GPU ask deferred until gates G1+G2 pass.
