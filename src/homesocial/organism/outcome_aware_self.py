"""Outcome-aware planning through the learned causal self-model.

The legacy causal planner applies a nonlinear homeostatic utility after
averaging the listener's possible responses. That turns an uncertain word
which might help any of three needs into an imaginary outcome which helps all
three at once. Probe60 keeps response branches separate and computes expected
utility over realized bodily consequences instead.

See ``docs/decisions/2026-07-30-outcome-aware-self-planner-preregistration.md``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from random import Random
from typing import Literal

import mlx.core as mx
import numpy as np

from homesocial.creole.vocab import PAD_TOKEN, TOKEN_TO_ID
from homesocial.island.report import NEED_TO_REPORT_WORD, heard_need
from homesocial.island.world import SURFACES
from homesocial.organism.causal_self import (
    _apply_causal_transition,
    _public_transition_features,
    train_structured_causal_self_model,
)
from homesocial.organism.model import OrganismModel
from homesocial.organism.online_adaptation import (
    ADAPTATION_TICKS,
    ADAPT_SEED_BASE,
    EVAL_SEED_BASE,
    constant_recovery,
    shift_report_config,
)
from homesocial.organism.report_audit import (
    FIDELITY_WARMUP,
    _ObservableBodyFilter,
    make_report_world,
)
from homesocial.organism.self_belief import _sample_motor_action
from homesocial.organism.train import (
    OrganismConfig,
    audit_report_lexical_comprehension,
    execute_agent_action,
    load_organism_checkpoint,
)

PAD_ID = TOKEN_TO_ID[PAD_TOKEN]
PlannerObjective = Literal["outcome_aware", "legacy", "truthful"]
BeliefIntervention = Literal["none", "zero"]


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum(axis=-1, keepdims=True)


def outcome_aware_scores(
    listener_probabilities: np.ndarray,
    drift: np.ndarray,
    uptake: np.ndarray,
    belief: np.ndarray,
    *,
    ticks_to_help: int,
) -> np.ndarray:
    """Compute ``E[min(next body) | token]`` over realized help branches.

    ``listener_probabilities`` has one column per public help surface plus a
    final no-help outcome. ``uptake`` maps each help surface to its learned
    three-axis bodily consequence. The operation intentionally has no access
    to need names or word identities.
    """

    listener = np.asarray(listener_probabilities, dtype=np.float64)
    drift_array = np.asarray(drift, dtype=np.float64)
    uptake_array = np.asarray(uptake, dtype=np.float64)
    current = np.asarray(belief, dtype=np.float64)
    if listener.ndim != 2 or uptake_array.ndim != 2:
        raise ValueError("Listener and uptake arrays must both be rank two.")
    if listener.shape[1] != uptake_array.shape[0] + 1:
        raise ValueError("Listener outcomes must be help surfaces plus no-help.")
    if (
        uptake_array.shape[1] != current.shape[0]
        or drift_array.shape != current.shape
    ):
        raise ValueError("Drift, uptake, and belief dimensions must agree.")
    if ticks_to_help <= 0:
        raise ValueError("ticks_to_help must be positive.")

    before_help = np.clip(
        current + float(ticks_to_help) * drift_array, 0.0, 1.0
    )
    help_branches = np.clip(
        before_help[None, :] + uptake_array, 0.0, 1.0
    )
    branches = np.concatenate((help_branches, before_help[None, :]), axis=0)
    branch_utilities = branches.min(axis=-1)
    return listener @ branch_utilities


def legacy_expected_state_scores(
    listener_probabilities: np.ndarray,
    drift: np.ndarray,
    uptake: np.ndarray,
    belief: np.ndarray,
    *,
    ticks_to_help: int,
) -> np.ndarray:
    """Numpy-equivalent control for the legacy ``min(E[next body])`` score."""

    listener = np.asarray(listener_probabilities, dtype=np.float64)
    drift_array = np.asarray(drift, dtype=np.float64)
    uptake_array = np.asarray(uptake, dtype=np.float64)
    current = np.asarray(belief, dtype=np.float64)
    before_help = np.clip(
        current + float(ticks_to_help) * drift_array, 0.0, 1.0
    )
    expected_uptake = listener[:, : uptake_array.shape[0]] @ uptake_array
    futures = np.clip(before_help[None, :] + expected_uptake, 0.0, 1.0)
    return futures.min(axis=-1)


@dataclass(frozen=True)
class CausalTokenPlanner:
    """Frozen numpy view of learned social and bodily consequences."""

    listener_probabilities: np.ndarray
    drift: np.ndarray
    uptake: np.ndarray
    objective: Literal["outcome_aware", "legacy"]

    @classmethod
    def from_model(
        cls,
        model: OrganismModel,
        *,
        objective: Literal["outcome_aware", "legacy"] = "outcome_aware",
    ) -> "CausalTokenPlanner":
        drift, _, uptake, _ = model.causal_self_parameters()
        mx.eval(model.causal_listener_logits, drift, uptake)
        return cls(
            listener_probabilities=_softmax_rows(
                np.asarray(model.causal_listener_logits, dtype=np.float64)
            ),
            drift=np.asarray(drift, dtype=np.float64),
            uptake=np.asarray(uptake, dtype=np.float64),
            objective=objective,
        )

    def scores(
        self,
        belief: np.ndarray,
        *,
        step_count: int,
        help_period: int,
    ) -> np.ndarray:
        ticks_to_help = help_period - (step_count % help_period)
        scorer = (
            outcome_aware_scores
            if self.objective == "outcome_aware"
            else legacy_expected_state_scores
        )
        return scorer(
            self.listener_probabilities,
            self.drift,
            self.uptake,
            belief,
            ticks_to_help=ticks_to_help,
        )

    def token(
        self,
        belief: np.ndarray,
        *,
        step_count: int,
        help_period: int,
    ) -> int:
        return int(
            np.argmax(
                self.scores(
                    belief,
                    step_count=step_count,
                    help_period=help_period,
                )
            )
        )


def outcome_aware_causal_social_token(
    model: OrganismModel,
    belief: np.ndarray,
    *,
    step_count: int,
    help_period: int,
) -> int:
    """Choose a full-vocabulary token by expected realized bodily utility."""

    return CausalTokenPlanner.from_model(model).token(
        belief, step_count=step_count, help_period=help_period
    )


def tied_deficit_intervention() -> dict[str, float]:
    """The locked synthetic intervention which isolates the Jensen error."""

    uptake = np.eye(3, dtype=np.float64) * 0.90
    listener = np.zeros((2, 4), dtype=np.float64)
    listener[0, 0] = 1.0
    listener[1, :3] = 1.0 / 3.0
    drift = np.zeros((3,), dtype=np.float64)
    clear = np.asarray((0.20, 0.90, 0.95), dtype=np.float64)
    tied = np.asarray((0.30, 0.32, 0.90), dtype=np.float64)

    clear_old = int(
        np.argmax(
            legacy_expected_state_scores(
                listener, drift, uptake, clear, ticks_to_help=1
            )
        )
    )
    clear_new = int(
        np.argmax(
            outcome_aware_scores(
                listener, drift, uptake, clear, ticks_to_help=1
            )
        )
    )
    tied_old = int(
        np.argmax(
            legacy_expected_state_scores(
                listener, drift, uptake, tied, ticks_to_help=1
            )
        )
    )
    tied_new = int(
        np.argmax(
            outcome_aware_scores(
                listener, drift, uptake, tied, ticks_to_help=1
            )
        )
    )
    return {
        "clear_legacy_targeted": float(clear_old == 0),
        "clear_outcome_aware_targeted": float(clear_new == 0),
        "tied_legacy_diffuse": float(tied_old == 1),
        "tied_outcome_aware_targeted": float(tied_new == 0),
        "gate_passed": float(
            clear_old == 0
            and clear_new == 0
            and tied_old == 1
            and tied_new == 0
        ),
    }


def evaluate_planner_condition(
    belief_model: OrganismModel,
    planner_model: OrganismModel,
    world_config: OrganismConfig,
    *,
    lives: int,
    seed_base: int,
    objective: PlannerObjective = "outcome_aware",
    listener_mode: str = "grounded",
    filter_config: OrganismConfig | None = None,
    belief_intervention: BeliefIntervention = "none",
) -> dict[str, float]:
    """Evaluate explicitly separated belief, planning, and listener factors."""

    if lives <= 0:
        raise ValueError("Planner evaluation requires positive lives.")
    if belief_intervention not in {"none", "zero"}:
        raise ValueError(f"Unknown belief intervention: {belief_intervention}.")
    planner = None
    if objective != "truthful":
        planner = CausalTokenPlanner.from_model(
            planner_model,
            objective=(
                "outcome_aware" if objective == "outcome_aware" else "legacy"
            ),
        )

    survived = 0
    said = 0
    truthful = 0
    grants = 0
    truthful_grants = 0
    error_sum = 0.0
    error_samples = 0
    steps = 0
    viability_sum = 0.0
    for life in range(lives):
        seed = seed_base + life
        world = make_report_world(
            world_config, seed=seed, listener_mode=listener_mode
        )
        packet = world.reset(seed)
        analytic = None
        if filter_config is None:
            belief = np.asarray(packet.vector()[:3], dtype=np.float64)
        else:
            analytic = _ObservableBodyFilter(filter_config)
            analytic.reset(packet)
            belief = analytic.belief
        motor_hidden: mx.array | None = None
        motor_rng = Random(seed + 59_000_003)
        life_viability = 0.0
        while True:
            true_need = world.lowest_need()
            if objective == "truthful":
                token = TOKEN_TO_ID[NEED_TO_REPORT_WORD[true_need]]
            else:
                assert planner is not None
                planning_belief = (
                    np.zeros_like(belief)
                    if belief_intervention == "zero"
                    else belief
                )
                token = planner.token(
                    planning_belief,
                    step_count=packet.step_count,
                    help_period=world_config.report.help_period,
                )
            word = heard_need((token, PAD_ID))
            if packet.step_count >= FIDELITY_WARMUP:
                if word is not None:
                    said += 1
                    truthful += int(word == true_need)
                error_sum += float(
                    np.abs(belief - np.asarray(packet.needs[:3])).mean()
                )
                error_samples += 1

            action, motor_hidden = _sample_motor_action(
                planner_model, packet, motor_hidden, motor_rng
            )
            world.hear((token, PAD_ID))
            before = packet
            packet, _, terminated, truncated, info = execute_agent_action(
                world,
                packet,
                action,
                consume_options=world_config.consume_options,
                inspect_options=world_config.inspect_options,
            )
            if analytic is None:
                belief = _apply_causal_transition(
                    belief_model,
                    belief,
                    _public_transition_features(
                        world_config, before, action, packet
                    ),
                )
            else:
                analytic.update(before, action, packet)
                belief = analytic.belief
            duration = int(info["duration"])
            steps += duration
            life_viability += float(info["mean_viability_sum"])
            if info.get("granted_need") is not None:
                grants += 1
                truthful_grants += int(
                    info.get("granted_need") == info.get("lowest_need")
                )
            if terminated or truncated:
                survived += int(not terminated)
                break
        viability_sum += life_viability / world_config.report.life_steps

    return {
        "survival": survived / lives,
        "report_fidelity": truthful / max(1, said),
        "speech_rate": said / max(1, steps),
        "grant_fidelity": truthful_grants / max(1, grants),
        "mean_absolute_need_error": error_sum / max(1, error_samples),
        "mean_life_steps": steps / lives,
        "mean_viability": viability_sum / lives,
    }


def _prefix(condition: str, metrics: dict[str, float]) -> dict[str, float]:
    return {
        f"{condition}_{name}": float(value) for name, value in metrics.items()
    }


def _mean(rows: list[dict[str, object]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _summarize(
    rows: list[dict[str, object]],
    *,
    truthful: dict[str, float],
    mechanism: dict[str, float],
) -> dict[str, object]:
    adapted_survivals = [
        float(row["adapted_outcome_survival"]) for row in rows
    ]
    constant_passes = [
        float(row["constant_error_after"])
        < float(row["constant_error_before"])
        for row in rows
    ]
    f0 = truthful["survival"] >= 0.90
    g1 = bool(mechanism["gate_passed"])
    g2 = bool(
        np.mean(adapted_survivals) >= 0.80
        and all(value >= 0.75 for value in adapted_survivals)
        and _mean(rows, "adapted_outcome_survival")
        - _mean(rows, "adapted_legacy_survival")
        >= 0.15
        and _mean(rows, "adapted_outcome_report_fidelity") >= 0.90
    )
    g3 = bool(
        _mean(rows, "adapted_outcome_survival")
        - _mean(rows, "frozen_outcome_survival")
        >= 0.05
        and _mean(rows, "adapted_outcome_survival")
        - _mean(rows, "stale_belief_survival")
        >= 0.05
        and _mean(rows, "adapted_outcome_mean_absolute_need_error") <= 0.03
        and all(constant_passes)
    )
    g4 = bool(
        _mean(rows, "adapted_outcome_survival")
        - _mean(rows, "scrambled_listener_survival")
        >= 0.30
        and _mean(rows, "adapted_outcome_survival")
        - _mean(rows, "zero_belief_survival")
        >= 0.30
    )
    g5 = bool(
        all(
            float(row["oracle_belief_survival"]) + 0.01
            >= float(row["adapted_outcome_survival"])
            for row in rows
        )
        and all(
            float(row["base_parameters_unchanged"]) == 1.0 for row in rows
        )
        and all(float(row["lexical_gate_passed"]) == 1.0 for row in rows)
    )
    return {
        "seeds": len(rows),
        "truthful_oracle": truthful,
        "mechanism_intervention": mechanism,
        "means": {
            key: _mean(rows, key)
            for key in rows[0]
            if key not in {"seed_index", "adapt_seed_base"}
        },
        "f0_ecology_ceiling": float(f0),
        "g1_defect_intervention": float(g1),
        "g2_behavioural_repair": float(g2),
        "g3_continual_model_load_bearing": float(g3),
        "g4_causal_necessity": float(g4),
        "g5_ceiling_and_persistence": float(g5),
        "all_gates_passed": float(f0 and g1 and g2 and g3 and g4 and g5),
    }


def run_outcome_aware_experiment(
    parent: str,
    run_dir: str,
    *,
    seeds: int = 5,
    lives: int = 100,
    adaptation_ticks: int = ADAPTATION_TICKS,
    lexical_lives: int = 90,
) -> dict[str, object]:
    """Run the locked five-seed Phase A1b treatment and its lesions."""

    if seeds < 5:
        raise ValueError(
            "The preregistered treatment requires at least five seeds."
        )
    target = Path(run_dir)
    target.mkdir(parents=True, exist_ok=True)

    ceiling_model, base_config = load_organism_checkpoint(parent)
    if not ceiling_model.has_structured_causal_self_model:
        raise ValueError("The parent must carry the structured causal self-model.")
    shifted = shift_report_config(base_config)
    truthful = evaluate_planner_condition(
        ceiling_model,
        ceiling_model,
        shifted,
        lives=lives,
        seed_base=EVAL_SEED_BASE,
        objective="truthful",
    )
    mechanism = tied_deficit_intervention()
    (target / "feasibility.json").write_text(
        json.dumps(
            {"truthful_oracle": truthful, "mechanism": mechanism}, indent=2
        )
        + "\n"
    )
    if truthful["survival"] < 0.90:
        stopped = {
            "stopped_at_f0": 1.0,
            "truthful_oracle": truthful,
            "mechanism_intervention": mechanism,
        }
        (target / "probe60_summary.json").write_text(
            json.dumps(stopped, indent=2) + "\n"
        )
        return stopped

    rows: list[dict[str, object]] = []
    detailed: list[dict[str, object]] = []
    for index in range(seeds):
        adapt_seed = ADAPT_SEED_BASE + index * 10_000_000
        frozen_model, _ = load_organism_checkpoint(parent)
        adapted_model, _ = load_organism_checkpoint(parent)
        checkpoint = target / f"adapted_seed{index}.npz"
        stats = target / f"adapted_seed{index}.csv"
        _, development = train_structured_causal_self_model(
            adapted_model,
            shifted,
            steps=adaptation_ticks,
            learning_rate=3e-3,
            seed_base=adapt_seed,
            checkpoint=str(checkpoint),
            stats_csv=str(stats),
            log_every_lives=0,
        )

        conditions = {
            "adapted_outcome": evaluate_planner_condition(
                adapted_model,
                adapted_model,
                shifted,
                lives=lives,
                seed_base=EVAL_SEED_BASE,
            ),
            "adapted_legacy": evaluate_planner_condition(
                adapted_model,
                adapted_model,
                shifted,
                lives=lives,
                seed_base=EVAL_SEED_BASE,
                objective="legacy",
            ),
            "frozen_outcome": evaluate_planner_condition(
                frozen_model,
                frozen_model,
                shifted,
                lives=lives,
                seed_base=EVAL_SEED_BASE,
            ),
            "stale_belief": evaluate_planner_condition(
                adapted_model,
                adapted_model,
                shifted,
                lives=lives,
                seed_base=EVAL_SEED_BASE,
                filter_config=base_config,
            ),
            "oracle_belief": evaluate_planner_condition(
                adapted_model,
                adapted_model,
                shifted,
                lives=lives,
                seed_base=EVAL_SEED_BASE,
                filter_config=shifted,
            ),
            "adapted_belief_frozen_planner": evaluate_planner_condition(
                adapted_model,
                frozen_model,
                shifted,
                lives=lives,
                seed_base=EVAL_SEED_BASE,
            ),
            "scrambled_listener": evaluate_planner_condition(
                adapted_model,
                adapted_model,
                shifted,
                lives=lives,
                seed_base=EVAL_SEED_BASE,
                listener_mode="scrambled",
            ),
            "zero_belief": evaluate_planner_condition(
                adapted_model,
                adapted_model,
                shifted,
                lives=lives,
                seed_base=EVAL_SEED_BASE,
                belief_intervention="zero",
            ),
        }
        before = constant_recovery(frozen_model, shifted)
        after = constant_recovery(adapted_model, shifted)
        lexical = audit_report_lexical_comprehension(
            adapted_model,
            lives=lexical_lives,
            base_seed=2_000_000 + index * 10_000,
        )
        row: dict[str, object] = {
            "seed_index": index,
            "adapt_seed_base": adapt_seed,
            "constant_error_before": before["mean_absolute_constant_error"],
            "constant_error_after": after["mean_absolute_constant_error"],
            "base_parameters_unchanged": development[
                "base_parameters_unchanged"
            ],
            "lexical_gate_passed": lexical["gate_passed"],
            "lexical_intact_correct_rate": lexical["intact_correct_rate"],
            "lexical_cyclic_correct_rate": lexical["cyclic_correct_rate"],
            "lexical_paired_action_change_rate": lexical[
                "paired_action_change_rate"
            ],
        }
        for name, metrics in conditions.items():
            row.update(_prefix(name, metrics))
        rows.append(row)
        detailed.append(
            {
                "seed_index": index,
                "adapt_seed_base": adapt_seed,
                "development": development,
                "constant_recovery_before": before,
                "constant_recovery_after": after,
                "lexical": lexical,
                "conditions": conditions,
            }
        )
        print(f"seed {index}: " + json.dumps(row, sort_keys=True))

    summary = _summarize(rows, truthful=truthful, mechanism=mechanism)
    (target / "probe60_rows.json").write_text(
        json.dumps(detailed, indent=2) + "\n"
    )
    (target / "probe60_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    with (target / "probe60_rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\n=== outcome-aware self-planner summary ===")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--lives", type=int, default=100)
    parser.add_argument("--adaptation-ticks", type=int, default=ADAPTATION_TICKS)
    parser.add_argument("--lexical-lives", type=int, default=90)
    args = parser.parse_args()
    run_outcome_aware_experiment(
        args.parent,
        args.run_dir,
        seeds=args.seeds,
        lives=args.lives,
        adaptation_ticks=args.adaptation_ticks,
        lexical_lives=args.lexical_lives,
    )


if __name__ == "__main__":
    main()
