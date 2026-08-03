"""Merge per-world probe61 runs into one record and evaluate the locked gates.

`run_world_seed` is pure in `(world, seed)` -- the developmental stream comes
from `development_seed_base(seed_index, world_name)` and the fit is seeded by
`seed_index` alone -- so the twenty treatment fits may be run as four processes,
one per world, and merged afterwards. This produces exactly the payload a single
`--worlds K1,K2,K3,K4 --seeds 5` invocation would have written, and evaluates
`evaluate_gates` over the pooled records, which is the only place gate arithmetic
happens.

    PYTHONPATH=src .venv/bin/python scripts/probe61_merge_worlds.py \
        --run-dir runs/organism/probe61_discovered_self/full \
        --worlds K1,K2,K3,K4
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from homesocial.organism.discovered_self import _flatten, evaluate_gates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--worlds", default="K1,K2,K3,K4")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    worlds = tuple(name.strip() for name in args.worlds.split(",") if name.strip())

    results: list[dict[str, object]] = []
    controls: list[dict[str, object]] = []
    configs: dict[str, object] = {}
    missing: list[str] = []
    for world in worlds:
        payload_path = run_dir / world / "discovered_self.json"
        if not payload_path.exists():
            missing.append(world)
            continue
        payload = json.loads(payload_path.read_text())
        results.extend(payload["results"])
        controls.extend(payload.get("controls", []))
        configs[world] = payload["config"]

    if missing:
        raise SystemExit(
            f"Not finished: no discovered_self.json for {', '.join(missing)}. "
            "Merging now would evaluate the locked gates on a partial sweep."
        )

    # The hyperparameters must be identical across worlds or the pooled gates
    # are comparing different experiments.
    shared = ("learning_rate", "sparsity", "latent_size", "development_ticks", "lives")
    for key in shared:
        values = {str(configs[world][key]) for world in worlds}  # type: ignore[index]
        if len(values) != 1:
            raise SystemExit(f"Worlds disagree on {key}: {sorted(values)}.")

    seeds = sorted({int(record["seed_index"]) for record in results})
    per_world = {
        world: sorted(
            int(r["seed_index"]) for r in results if r["world"] == world
        )
        for world in worlds
    }
    if any(per_world[world] != seeds for world in worlds):
        raise SystemExit(f"Ragged sweep, seeds per world: {per_world}.")

    gates = evaluate_gates(results, worlds)
    merged = {
        "config": {
            **{key: configs[worlds[0]][key] for key in shared},  # type: ignore[index]
            "seeds": len(seeds),
            "worlds": list(worlds),
            "merged_from": [str(run_dir / world) for world in worlds],
        },
        "results": results,
        "controls": controls,
        "gates": gates,
    }
    out = Path(args.out) if args.out else run_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "discovered_self.json").write_text(json.dumps(merged, indent=2, default=str))

    rows = [_flatten(record) for record in results + controls]
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with (out / "discovered_self.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"merged {len(results)} fits over {len(worlds)} worlds, seeds {seeds}")
    print(json.dumps(gates, indent=2, default=str))


if __name__ == "__main__":
    main()
