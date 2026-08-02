"""Extend the probe61 sparsity grid downward and apply the locked selection rule.

The original grid was written at the wrong scale: the sensory mean-squared error
at the optimum is about 1.7e-4, so even 1e-4 times eight gates swamps the data
term. See the third amendment in
`docs/decisions/2026-08-02-discovered-self-structure-preregistration.md`.

The rule is unchanged and reads held-out sensory error only: the largest
coefficient whose held-out RMSE is within 5% of the best on the grid.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

from homesocial.organism.discovered_self import (
    DiscoveredSelfConfig,
    collect_development_stream,
    development_seed_base,
    fit_discovered_self,
)
from homesocial.organism.train import load_organism_checkpoint

PARENT = "runs/organism/probe52_guided_report_lexicon/adult/organism_report_seed1.npz"
RUN_DIR = Path("runs/organism/probe61_discovered_self")
EXTENSION = (1e-6, 3e-6, 1e-5, 3e-5)


def main() -> None:
    parent, organism = load_organism_checkpoint(PARENT)
    coarse = json.loads((RUN_DIR / "selection.json").read_text())
    config = DiscoveredSelfConfig(
        learning_rate=coarse["chosen_learning_rate"],
        shooting_schedule=(
            ((25, 3000), (100, 1500), (0, 1500))
            if coarse["chosen_schedule"] == "B"
            else ((25, 1500), (100, 750), (0, 750))
        ),
    )
    stream = collect_development_stream(
        parent,
        organism,
        config,
        world_name="K4",
        seed_base=development_seed_base(0, "K4"),
        log_every_lives=0,
    )
    rows = list(coarse["sparsity"])
    for value in EXTENSION:
        start = time.time()
        _, _, metrics = fit_discovered_self(
            stream, replace(config, sparsity=value), seed=0, log_every=0
        )
        rows.append({"stage": "sparsity", "sparsity": value, "holdout_rmse": metrics["holdout_rmse"]})
        print(
            f"LAM {value:g} rmse={metrics['holdout_rmse']:.6f} "
            f"t={time.time() - start:.0f}s",
            flush=True,
        )
    best = min(row["holdout_rmse"] for row in rows)
    eligible = [row for row in rows if row["holdout_rmse"] <= 1.05 * best]
    chosen = max(row["sparsity"] for row in eligible)
    print(f"CHOSEN sparsity {chosen:g} (best rmse {best:.6f})", flush=True)
    coarse["sparsity"] = sorted(rows, key=lambda row: row["sparsity"])
    coarse["chosen_sparsity"] = chosen
    coarse["sparsity_grid_extended"] = True
    (RUN_DIR / "selection.json").write_text(json.dumps(coarse, indent=2))
    print("EXTENDED SELECTION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
