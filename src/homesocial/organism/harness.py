"""The one benchmark command (gate G1 and its controls).

Trains a fresh organism under each language mode, evaluates on held-out
fixed seeds, and prints a table alongside scripted baselines. All headline
numbers come from here.

    PYTHONPATH=src python3 -m homesocial.organism.harness \
        --train-steps 200000 --language-modes grounded silent shuffled
"""

from __future__ import annotations

import argparse
from pathlib import Path

from homesocial.island.calibrate import run_policy
from homesocial.island.world import IslandConfig
from homesocial.organism.train import (
    OrganismConfig,
    evaluate_organism,
    train_organism,
)

EVAL_SEED_BASE = 900_000


def main() -> None:
    args = _parse_args()
    rows: list[dict[str, object]] = []

    for baseline in args.baselines:
        stats = run_policy(
            baseline,
            episodes=args.eval_episodes,
            config=IslandConfig(max_steps=args.eval_max_steps),
            base_seed=EVAL_SEED_BASE,
        )
        rows.append({"condition": f"baseline_{baseline}", **stats})

    for language_mode in args.language_modes:
        config = OrganismConfig(
            language_mode=language_mode,
            total_steps=args.train_steps,
            segment_length=args.segment_length,
            hidden_size=args.hidden_size,
            seed=args.seed,
            max_steps=args.train_max_steps,
            checkpoint=str(
                Path(args.run_dir) / f"organism_{language_mode}_seed{args.seed}.npz"
            ),
            stats_csv=str(
                Path(args.run_dir) / f"lives_{language_mode}_seed{args.seed}.csv"
            ),
        )
        print(f"=== training organism under {language_mode} ===")
        model, _ = train_organism(config)
        stats = evaluate_organism(
            model,
            language_mode=language_mode,
            episodes=args.eval_episodes,
            base_seed=EVAL_SEED_BASE,
            max_steps=args.eval_max_steps,
        )
        rows.append({"condition": f"organism_{language_mode}", **stats})

    _write_and_print(rows, args)


def _write_and_print(rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    keys = ["condition"] + [key for key in rows[0] if key != "condition"]
    out_path = Path(args.run_dir) / f"harness_seed{args.seed}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(keys) + "\n")
        for row in rows:
            handle.write(
                ",".join(
                    f"{row.get(key):.4f}"
                    if isinstance(row.get(key), float)
                    else str(row.get(key, ""))
                    for key in keys
                )
                + "\n"
            )
    print(f"\nresults ({out_path}):")
    for row in rows:
        formatted = ", ".join(
            f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
            for key, value in row.items()
        )
        print("  " + formatted)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-steps", type=int, default=200_000)
    parser.add_argument("--segment-length", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--train-max-steps", type=int, default=1000)
    parser.add_argument("--eval-max-steps", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument(
        "--language-modes",
        nargs="+",
        default=["grounded", "silent", "shuffled"],
        choices=["grounded", "silent", "shuffled"],
    )
    parser.add_argument(
        "--baselines", nargs="+", default=["oracle", "random"],
        choices=["oracle", "random"],
    )
    parser.add_argument("--run-dir", default="runs/organism")
    return parser.parse_args()


if __name__ == "__main__":
    main()
