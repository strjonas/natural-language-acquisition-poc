"""Probe70 ceiling: can a constitution-aware caregiver budget its own basket?

Preregistered in
``docs/decisions/2026-08-20-constitution-budget-ceiling-preregistration.md``.
This is a ceiling instrument, not a language mechanism: the caregiver receives
oracle rates directly and the organism receives no new word.

The comparison keeps probe64's strongest requester fixed and changes only how a
finite caregiver store is allocated. ``first_come`` is the existing world.
``oracle_rates`` partitions the store by this body's true per-need burden;
``species_rates`` removes individual content; ``permuted_rates`` preserves the
same burden values and assigns them to the wrong bodily axes.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
from random import Random
from typing import Iterable

import numpy as np

from homesocial.island.report import (
    REPORT_NEEDS,
    SIZE_TO_WORD,
    ReportConfig,
    ReportWorld,
)
from homesocial.organism.portion_request import (
    GrantWatcher,
    MoveFraction,
    SelfModelTier,
    Speaker,
    _size_word_of,
    species_rate,
)
from homesocial.organism.self_belief import _sample_motor_action
from homesocial.organism.train import (
    OrganismConfig,
    execute_agent_action,
    load_organism_checkpoint,
)

POLICIES = ("first_come", "oracle_rates", "species_rates", "permuted_rates")
STORE_SWEEP = (0.0, 34.0, 30.0, 28.0, 26.0, 24.0, 22.0)
SURVEY_SEED_BASE = 2_300_000_000
SEED_STRIDE = 2_000_000
SEEDS = 5
LIVES = 40
INTEROCEPTION = 0.03
METABOLIC_SPREAD = 0.60
UPTAKE_SPREAD = 0.0
EPSILON = 1e-12


def predicted_burdens(
    report: ReportConfig,
    *,
    metabolic_scale: dict[str, float],
    uptake_scale: dict[str, float],
) -> dict[str, float]:
    """Expected resource mass per tick for each reportable bodily axis.

    Shocks select uniformly among the three reportable needs. Energy's movement
    mixture is the public 0.5 prior already used by ``MoveFraction`` before it
    has observed an action. There is no fitted quantity in this calculation.
    """

    shock_burden = report.shock_probability * report.shock_size / len(REPORT_NEEDS)
    burdens = {}
    for need in REPORT_NEEDS:
        metabolism = species_rate(report, need, 0.5) * metabolic_scale[need]
        uptake = max(EPSILON, uptake_scale[need])
        burdens[need] = (metabolism + shock_burden) / uptake
    return burdens


def policy_burdens(
    policy: str,
    report: ReportConfig,
    *,
    metabolic_scale: dict[str, float],
    uptake_scale: dict[str, float],
) -> dict[str, float]:
    """The burden vector available to one caregiver-side control arm."""

    if policy not in POLICIES:
        raise ValueError(f"Unknown budget policy: {policy}.")
    if policy == "species_rates":
        return predicted_burdens(
            report,
            metabolic_scale={need: 1.0 for need in REPORT_NEEDS},
            uptake_scale={need: 1.0 for need in REPORT_NEEDS},
        )

    oracle = predicted_burdens(
        report,
        metabolic_scale=metabolic_scale,
        uptake_scale=uptake_scale,
    )
    if policy == "permuted_rates":
        values = [oracle[need] for need in REPORT_NEEDS]
        return {
            need: values[(index + 1) % len(values)]
            for index, need in enumerate(REPORT_NEEDS)
        }
    return oracle


class LifetimeShareAllocator:
    """No-free-parameter per-need accounts over one caregiver store."""

    def __init__(self, policy: str) -> None:
        if policy not in POLICIES:
            raise ValueError(f"Unknown budget policy: {policy}.")
        self.policy = policy
        self.store = 0.0
        self.burdens = {need: 0.0 for need in REPORT_NEEDS}
        self.quotas = {need: 0.0 for need in REPORT_NEEDS}
        self.spent = {need: 0.0 for need in REPORT_NEEDS}
        self.refused = {need: 0 for need in REPORT_NEEDS}
        self.refused_capacity = 0.0
        self.refused_capacity_max = 0.0

    def reset(
        self,
        report: ReportConfig,
        *,
        metabolic_scale: dict[str, float],
        uptake_scale: dict[str, float],
    ) -> None:
        self.store = float(report.caregiver_store)
        self.burdens = policy_burdens(
            self.policy,
            report,
            metabolic_scale=metabolic_scale,
            uptake_scale=uptake_scale,
        )
        total = sum(self.burdens.values())
        self.quotas = {
            need: self.store * self.burdens[need] / max(EPSILON, total)
            for need in REPORT_NEEDS
        }
        self.spent = {need: 0.0 for need in REPORT_NEEDS}
        self.refused = {need: 0 for need in REPORT_NEEDS}
        self.refused_capacity = 0.0
        self.refused_capacity_max = 0.0

    def permit(self, need: str, *, cost: float, capacity: float) -> bool:
        """Approve while an account has not yet crossed its lifetime share.

        The grant that crosses a fractional quota is allowed. This avoids making
        a remainder smaller than one discrete portion permanently unusable.
        """

        if self.policy == "first_come" or self.store <= 0.0:
            self.spent[need] += cost
            return True
        if self.spent[need] < self.quotas[need] - EPSILON:
            self.spent[need] += cost
            return True
        self.refused[need] += 1
        self.refused_capacity += capacity
        self.refused_capacity_max = max(self.refused_capacity_max, capacity)
        return False

    def snapshot(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "burdens": dict(self.burdens),
            "quotas": dict(self.quotas),
            "spent_by_need": dict(self.spent),
            "refused_by_need": dict(self.refused),
            "refused_capacity": self.refused_capacity,
            "refused_capacity_max": self.refused_capacity_max,
        }


class ConstitutionBudgetWorld(ReportWorld):
    """ReportWorld with one audit-only finite-store allocation policy."""

    def __init__(self, *args, allocation_policy: str, **kwargs) -> None:
        self.allocation = LifetimeShareAllocator(allocation_policy)
        super().__init__(*args, **kwargs)

    def reset(self, seed: int | None = None):
        packet = super().reset(seed)
        self.allocation.reset(
            self.report,
            metabolic_scale=self.metabolic_scale,
            uptake_scale=self.uptake_scale,
        )
        return packet

    def _caregiver_allows_grant(
        self, need: str, *, large: bool, cost: float
    ) -> bool:
        del large
        # Read only for the diagnostic attached to a refusal. The allocator's
        # decision receives ``need`` and ``cost`` but never the current body.
        level = float(getattr(self.grid.needs, need))
        delivered = cost * self.uptake_scale[need]
        capacity = min(delivered, max(0.0, 1.0 - level))
        return self.allocation.permit(need, cost=cost, capacity=capacity)


def make_budget_world(
    organism: OrganismConfig,
    *,
    seed: int,
    store: float,
    policy: str,
) -> ConstitutionBudgetWorld:
    report = replace(
        organism.report,
        metabolic_spread=METABOLIC_SPREAD,
        uptake_spread=UPTAKE_SPREAD,
        interoception_probability=INTEROCEPTION,
        portion_requests=True,
        caregiver_store=store,
    )
    world = ConstitutionBudgetWorld(
        organism.island_config(semantic_choice_trial=False),
        report=report,
        seed=seed,
        allocation_policy=policy,
    )
    world.reset(seed)
    return world


def _speaker_with_known_size_words(tier: SelfModelTier) -> Speaker:
    speaker = Speaker(tier)
    # Probe64 already established this convention. Pooling means one example of
    # each word is sufficient for every need, and both observations are supplied
    # identically to all arms before the life starts.
    speaker.listener.observe("food", SIZE_TO_WORD["large"], True)
    speaker.listener.observe("food", SIZE_TO_WORD["small"], False)
    return speaker


def run_condition(
    model,
    organism: OrganismConfig,
    *,
    policy: str,
    store: float,
    lives: int,
    seed_base: int,
) -> dict[str, object]:
    """Run one paired survey cell with the oracle requester held fixed."""

    survived = 0
    life_steps = 0
    spent = 0.0
    delivered = 0.0
    overflow = 0.0
    refused = 0
    grants = 0
    large_grants = 0
    death_causes = {need: 0 for need in (*REPORT_NEEDS, "safety", "unknown")}
    grants_by_need = {need: 0 for need in REPORT_NEEDS}
    quotas = {need: 0.0 for need in REPORT_NEEDS}
    allocated = {need: 0.0 for need in REPORT_NEEDS}
    allocation_refusals = {need: 0 for need in REPORT_NEEDS}
    refusal_capacity = 0.0
    refusal_capacity_max = 0.0

    for life in range(lives):
        seed = seed_base + life
        world = make_budget_world(
            organism, seed=seed, store=store, policy=policy
        )
        packet = world.reset(seed)
        tier = SelfModelTier("oracle", organism, world.report)
        tier.reset(packet, world)
        moves = MoveFraction(organism, world.report)
        moves.reset(packet)
        speaker = _speaker_with_known_size_words(tier)
        watcher = GrantWatcher(world.report.help_period)
        hidden = None
        rng = Random(seed + 59_000_003)

        while True:
            need = REPORT_NEEDS[int(np.argmin(tier.point()))]
            fraction = moves.fraction
            tokens = speaker.utterance(need, fraction)
            action, hidden = _sample_motor_action(model, packet, hidden, rng)
            world.hear(tokens)
            before = packet
            packet, _, terminated, truncated, info = execute_agent_action(
                world,
                packet,
                action,
                consume_options=organism.consume_options,
                inspect_options=organism.inspect_options,
            )
            life_steps += int(info["duration"])
            granted = watcher.poll(packet)
            if granted is not None:
                word = _size_word_of(tokens)
                if word is not None:
                    speaker.listener.observe(granted[0], word, granted[1])
            uptake = info.get("uptake")
            if uptake is not None:
                delivered += float(uptake["delivered"])
                overflow += float(uptake["overflow"])
            if terminated or truncated:
                survived += int(not terminated)
                if terminated:
                    cause = info.get("death_need")
                    key = str(cause) if cause in death_causes else "unknown"
                    death_causes[key] += 1
                break
            moves.observe(before, action, packet)
            tier.update(before, action, packet, world, info.get("interoception"))

        counts = world.grant_counts
        spent += world.store_spent
        refused += int(counts["refused"])
        grants += int(counts["total"]) - int(counts["none"]) - int(counts["refused"])
        large_grants += int(counts["granted_large"])
        for need in REPORT_NEEDS:
            grants_by_need[need] += int(counts[need])
        snapshot = world.allocation.snapshot()
        for need in REPORT_NEEDS:
            quotas[need] += float(snapshot["quotas"][need])
            allocated[need] += float(snapshot["spent_by_need"][need])
            allocation_refusals[need] += int(snapshot["refused_by_need"][need])
        refusal_capacity += float(snapshot["refused_capacity"])
        refusal_capacity_max = max(
            refusal_capacity_max, float(snapshot["refused_capacity_max"])
        )

    useful = delivered - overflow
    return {
        "policy": policy,
        "caregiver_store": float(store),
        "lives": int(lives),
        "seed_base": int(seed_base),
        "survival": survived / lives,
        "mean_life_steps": life_steps / lives,
        "store_spent": spent / lives,
        "delivered": delivered / lives,
        "overflow": overflow / lives,
        "uptake_efficiency": useful / max(EPSILON, delivered),
        "useful_per_store": useful / max(EPSILON, spent),
        "refused_grants": refused / lives,
        "grants": grants / lives,
        "large_share_granted": large_grants / max(1, grants),
        "grants_by_need": {
            need: grants_by_need[need] / lives for need in REPORT_NEEDS
        },
        "death_causes": {
            need: death_causes[need] / lives for need in death_causes
        },
        "mean_quota": {need: quotas[need] / lives for need in REPORT_NEEDS},
        "mean_spent_by_need": {
            need: allocated[need] / lives for need in REPORT_NEEDS
        },
        "allocation_refusals": {
            need: allocation_refusals[need] / lives for need in REPORT_NEEDS
        },
        "refused_capacity": refusal_capacity / lives,
        "refused_capacity_max": refusal_capacity_max,
    }


SCALAR_METRICS = (
    "survival",
    "mean_life_steps",
    "store_spent",
    "delivered",
    "overflow",
    "uptake_efficiency",
    "useful_per_store",
    "refused_grants",
    "grants",
    "large_share_granted",
    "refused_capacity",
    "refused_capacity_max",
)


def summarize_cells(cells: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[float, str], list[dict[str, object]]] = defaultdict(list)
    for cell in cells:
        grouped[(float(cell["caregiver_store"]), str(cell["policy"]))].append(cell)

    summary = []
    for (store, policy), rows in grouped.items():
        item: dict[str, object] = {
            "caregiver_store": store,
            "policy": policy,
            "seeds": len(rows),
        }
        for metric in SCALAR_METRICS:
            values = [float(row[metric]) for row in rows]
            item[metric] = float(np.mean(values))
            item[f"{metric}_per_seed"] = values
        summary.append(item)
    order = {policy: index for index, policy in enumerate(POLICIES)}
    return sorted(summary, key=lambda row: (-float(row["caregiver_store"]), order[str(row["policy"])]))


def continuation_result(
    cells: list[dict[str, object]], *, stores: tuple[float, ...], seeds: int
) -> dict[str, object]:
    by_cell: dict[tuple[float, str], list[dict[str, object]]] = defaultdict(list)
    for cell in cells:
        by_cell[(float(cell["caregiver_store"]), str(cell["policy"]))].append(cell)

    complete = all(
        len(by_cell[(store, policy)]) == seeds
        for store in stores
        for policy in POLICIES
    )
    unlimited_exact = False
    if 0.0 in stores and all(len(by_cell[(0.0, p)]) == seeds for p in POLICIES):
        unlimited_exact = True
        reference = by_cell[(0.0, "first_come")]
        invariant = (
            "survival",
            "mean_life_steps",
            "store_spent",
            "delivered",
            "overflow",
            "refused_grants",
            "grants",
            "large_share_granted",
        )
        for policy in POLICIES[1:]:
            rows = by_cell[(0.0, policy)]
            unlimited_exact = unlimited_exact and all(
                all(left[key] == right[key] for key in invariant)
                for left, right in zip(reference, rows)
            )

    finite = [store for store in stores if store > 0.0]
    candidates = []
    if complete:
        for index, store in enumerate(finite):
            rows = {policy: by_cell[(store, policy)] for policy in POLICIES}
            means = {
                policy: float(np.mean([float(row["survival"]) for row in policy_rows]))
                for policy, policy_rows in rows.items()
            }
            paired = [
                float(oracle["survival"]) - float(first["survival"])
                for oracle, first in zip(rows["oracle_rates"], rows["first_come"])
            ]
            neighbor_positive = False
            for neighbor_index in (index - 1, index + 1):
                if 0 <= neighbor_index < len(finite):
                    neighbor = finite[neighbor_index]
                    oracle_neighbor = by_cell[(neighbor, "oracle_rates")]
                    first_neighbor = by_cell[(neighbor, "first_come")]
                    neighbor_positive = neighbor_positive or float(
                        np.mean(
                            [
                                float(o["survival"]) - float(f["survival"])
                                for o, f in zip(oracle_neighbor, first_neighbor)
                            ]
                        )
                    ) > 0.0
            checks = {
                "usable_baseline": 0.05 < means["first_come"] < 0.95,
                "mean_advantage_at_least_0_10": float(np.mean(paired)) >= 0.10,
                "positive_on_four_of_five": sum(value > 0.0 for value in paired) >= 4,
                "beats_species": means["oracle_rates"] > means["species_rates"],
                "beats_permuted": means["oracle_rates"] > means["permuted_rates"],
                "adjacent_store_positive": neighbor_positive,
                "unlimited_exact": unlimited_exact,
            }
            candidates.append(
                {
                    "caregiver_store": store,
                    "survival": means,
                    "oracle_minus_first_per_seed": paired,
                    "oracle_minus_first_mean": float(np.mean(paired)),
                    "checks": checks,
                    "passes": all(checks.values()),
                }
            )
    return {
        "complete": complete,
        "unlimited_exact": unlimited_exact,
        "candidates": candidates,
        "licensed": complete and any(bool(row["passes"]) for row in candidates),
    }


def _write_result(
    out: Path,
    *,
    cells: list[dict[str, object]],
    stores: tuple[float, ...],
    seeds: int,
    lives: int,
) -> dict[str, object]:
    result = {
        "status": "complete" if len(cells) == len(stores) * len(POLICIES) * seeds else "running",
        "policies": list(POLICIES),
        "stores": list(stores),
        "seeds": seeds,
        "lives_per_seed": lives,
        "survey_seed_base": SURVEY_SEED_BASE,
        "seed_stride": SEED_STRIDE,
        "cells": cells,
        "summary": summarize_cells(cells),
        "continuation": continuation_result(cells, stores=stores, seeds=seeds),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def survey(
    model,
    organism: OrganismConfig,
    *,
    stores: tuple[float, ...] = STORE_SWEEP,
    seeds: int = SEEDS,
    lives: int = LIVES,
    out: Path | None = None,
) -> dict[str, object]:
    if seeds <= 0 or lives <= 0:
        raise ValueError("seeds and lives must be positive.")
    if any(store < 0.0 for store in stores):
        raise ValueError("stores must be nonnegative.")
    out = out or Path(
        "runs/organism/probe70_constitution_budget/ceiling_survey.json"
    )
    cells: list[dict[str, object]] = []
    for seed_index in range(seeds):
        seed_base = SURVEY_SEED_BASE + seed_index * SEED_STRIDE
        for store in stores:
            for policy in POLICIES:
                cell = run_condition(
                    model,
                    organism,
                    policy=policy,
                    store=store,
                    lives=lives,
                    seed_base=seed_base,
                )
                cell["seed_index"] = seed_index
                cells.append(cell)
                _write_result(
                    out, cells=cells, stores=stores, seeds=seeds, lives=lives
                )
                print(
                    f"seed {seed_index} store {store:>4.0f} {policy:>14}: "
                    f"survival {cell['survival']:.3f} "
                    f"steps {cell['mean_life_steps']:.1f} "
                    f"refused {cell['refused_grants']:.1f}",
                    flush=True,
                )
    return _write_result(out, cells=cells, stores=stores, seeds=seeds, lives=lives)


def _parse_stores(value: str) -> tuple[float, ...]:
    stores = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not stores:
        raise argparse.ArgumentTypeError("at least one store is required")
    return stores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=(
            "runs/organism/probe52_guided_report_lexicon/adult/"
            "organism_report_seed1.npz"
        ),
    )
    parser.add_argument("--stores", type=_parse_stores, default=STORE_SWEEP)
    parser.add_argument("--seeds", type=int, default=SEEDS)
    parser.add_argument("--lives", type=int, default=LIVES)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "runs/organism/probe70_constitution_budget/ceiling_survey.json"
        ),
    )
    args = parser.parse_args()

    model, organism = load_organism_checkpoint(args.checkpoint)
    result = survey(
        model,
        organism,
        stores=args.stores,
        seeds=args.seeds,
        lives=args.lives,
        out=args.out,
    )
    print("\n== continuation rule ==")
    for row in result["continuation"]["candidates"]:
        print(
            f"store {row['caregiver_store']:.0f}: "
            f"oracle-first {row['oracle_minus_first_mean']:+.3f} "
            f"pass={row['passes']}"
        )
    print(f"licensed = {result['continuation']['licensed']}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
