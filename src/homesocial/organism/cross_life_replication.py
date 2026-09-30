"""Probe75: fixed, disjoint-band replication of Probe74's rate-memory survey.

The confirmation reuses the same episode mechanism, four cells and six gates.
The diagnostic has separately preregistered seeds and cannot produce gates.
See docs/decisions/2026-09-30-cross-life-replication-preregistration.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
from typing import Callable

from homesocial.organism.cross_life_self import (
    IDENTITIES, SEEDS, TEST_EPISODES, checkpoint_fingerprint, grade,
    run_episode, run_identity_cell, run_survey, validate_run,
)
from homesocial.organism.learned_rate_survival import _contrast, _write_progress
from homesocial.organism.train import load_organism_checkpoint


IDENTITY_SEED_BASE = 3_600_000_000
EPISODE_SEED_BASE = 3_700_000_000
DIAGNOSTIC_IDENTITY_SEED_BASE = 3_400_000_000
DIAGNOSTIC_EPISODE_SEED_BASE = 3_500_000_000
DEFAULT_CHECKPOINT = "runs/organism/probe52_guided_report_lexicon/adult/organism_report_seed1.npz"


def _run(model, organism, *, parent_fingerprint: str, seeds: int,
         identities: int, episodes: int, identity_seed_base: int,
         episode_seed_base: int, out: Path | None,
         episode_runner: Callable, cell_runner: Callable) -> dict[str, object]:
    result = run_survey(
        model, organism, parent_fingerprint=parent_fingerprint,
        seeds=seeds, identities=identities, episodes=episodes,
        identity_seed_base=identity_seed_base, episode_seed_base=episode_seed_base,
        out=out, episode_runner=episode_runner, cell_runner=cell_runner,
    )
    cells = {row["cell"]: row for row in result["cells"]}
    result["ungated_true_minus_persistent_survival"] = _contrast(
        cells["true_rate"]["per_seed_survival"],
        cells["persistent_rate"]["per_seed_survival"],
    )
    if out is not None:
        _write_progress(out, result)
    return result


def run_replication(model, organism, *, parent_fingerprint: str,
                    seeds: int = SEEDS, identities: int = IDENTITIES,
                    episodes: int = TEST_EPISODES, out: Path | None = None,
                    episode_runner: Callable = run_episode,
                    cell_runner: Callable = run_identity_cell) -> dict[str, object]:
    """Confirm on the locked 3.6B/3.7B bands; exact sample gets Probe74 gates."""

    return _run(
        model, organism, parent_fingerprint=parent_fingerprint,
        seeds=seeds, identities=identities, episodes=episodes,
        identity_seed_base=IDENTITY_SEED_BASE, episode_seed_base=EPISODE_SEED_BASE,
        out=out, episode_runner=episode_runner, cell_runner=cell_runner,
    )


def run_diagnostic(model, organism, *, parent_fingerprint: str,
                   out: Path | None = None, episode_runner: Callable = run_episode,
                   cell_runner: Callable = run_identity_cell) -> dict[str, object]:
    """Time the locked two-identity, one-test-episode diagnostic independently."""

    return _run(
        model, organism, parent_fingerprint=parent_fingerprint,
        seeds=1, identities=2, episodes=1,
        identity_seed_base=DIAGNOSTIC_IDENTITY_SEED_BASE,
        episode_seed_base=DIAGNOSTIC_EPISODE_SEED_BASE,
        out=out, episode_runner=episode_runner, cell_runner=cell_runner,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--seeds", type=int, default=SEEDS)
    parser.add_argument("--identities", type=int, default=IDENTITIES)
    parser.add_argument("--episodes", type=int, default=TEST_EPISODES)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.diagnostic:
        if (args.seeds, args.identities, args.episodes) != (SEEDS, IDENTITIES, TEST_EPISODES):
            parser.error("--diagnostic fixes the sample at one block, two identities, one test episode")
    else:
        validate_run(seeds=args.seeds, identities=args.identities, episodes=args.episodes,
                     identity_seed_base=IDENTITY_SEED_BASE, episode_seed_base=EPISODE_SEED_BASE)
    checkpoint = Path(args.checkpoint)
    model, organism = load_organism_checkpoint(checkpoint)
    common = dict(parent_fingerprint=checkpoint_fingerprint(checkpoint),
                  out=Path(args.out or (
                      "runs/organism/probe75_cross_life_replication/diagnostic.json"
                      if args.diagnostic else "runs/organism/probe75_cross_life_replication/replication.json")))
    result = (run_diagnostic(model, organism, **common) if args.diagnostic else
              run_replication(model, organism, seeds=args.seeds, identities=args.identities,
                              episodes=args.episodes, **common))
    for cell in result["cells"]:
        print(f"{cell['cell']}: survival {cell['survival']:.4f}; blocks {cell['per_seed_survival']}")
    print(json.dumps(result["gates"], indent=2))


if __name__ == "__main__":
    main()
