"""The one benchmark command (gate G1 and its controls).

Trains a fresh organism under each language mode, evaluates on held-out
fixed seeds, and prints a table alongside scripted baselines. All headline
numbers come from here.

    PYTHONPATH=src python3 -m homesocial.organism.harness \
        --train-steps 200000 --language-modes grounded silent shuffled
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from homesocial.island.calibrate import run_policy
from homesocial.island.world import IslandConfig
from homesocial.organism.model import OrganismModel
from homesocial.organism.train import (
    OrganismConfig,
    audit_binding_consequence_geometry,
    audit_bound_event_gradient_alignment,
    audit_label_referent_binding,
    audit_label_to_self_model,
    audit_metabolic_drift_forecast,
    audit_observation_branching_planner,
    audit_protocol_return_origin,
    audit_real_vs_explicit_event_transfer,
    audit_terminal_consume_value_calibration,
    audit_cross_round_label_reuse,
    audit_persistent_choice_environment,
    audit_persistent_mapping_information_rent,
    audit_semantic_choice_information_upper_bound,
    audit_self_model_actions,
    evaluate_organism,
    evaluate_semantic_choice,
    load_organism_checkpoint,
    train_organism,
)

EVAL_SEED_BASE = 900_000


def main() -> None:
    args = _parse_args()
    rows: list[dict[str, object]] = []

    if args.semantic_choice_information_upper_bound_contexts > 0:
        audit = audit_semantic_choice_information_upper_bound(
            episodes=args.semantic_choice_information_upper_bound_contexts,
        )
        rows.append(
            {
                "condition": "semantic_choice_information_upper_bound",
                **audit,
            }
        )
        _write_and_print(rows, args)
        return

    if args.persistent_mapping_rent_lives > 0:
        audit = audit_persistent_mapping_information_rent(
            lives=args.persistent_mapping_rent_lives,
        )
        rows.append(
            {
                "condition": "persistent_mapping_information_rent",
                **audit,
            }
        )
        _write_and_print(rows, args)
        return

    if args.persistent_choice_mechanics_lives > 0:
        audit = audit_persistent_choice_environment(
            lives=args.persistent_choice_mechanics_lives,
        )
        rows.append(
            {
                "condition": "persistent_choice_environment_mechanics",
                **audit,
            }
        )
        _write_and_print(rows, args)
        return

    if args.load_checkpoint is not None:
        model, loaded_config = load_organism_checkpoint(args.load_checkpoint)
        if args.evaluate_loaded_checkpoint:
            args.consume_options = loaded_config.consume_options
            args.inspect_options = loaded_config.inspect_options
            args.episodic_binding_size = loaded_config.episodic_binding_size
            args.semantic_choice_horizon = (
                loaded_config.island.semantic_choice_horizon
            )
            args.semantic_choice_objects = (
                loaded_config.island.semantic_choice_objects
            )
            args.semantic_choice_low_need = (
                loaded_config.island.semantic_choice_low_need
            )
            args.semantic_choice_rounds = (
                loaded_config.island.semantic_choice_rounds
            )
            args.semantic_choice_return_duration = (
                loaded_config.island.semantic_choice_return_duration
            )
            if not args.self_model_planning:
                # Without an explicit request the loaded checkpoint is
                # evaluated exactly as it was trained.
                args.planning_scale = loaded_config.self_model_planning_scale
                args.planning_reward_weight = (
                    loaded_config.self_model_planning_reward_weight
                )
                args.planning_horizon = (
                    loaded_config.self_model_planning_horizon
                )
                args.self_model_planning = (
                    loaded_config.self_model_planning_scale > 0.0
                )
            _append_semantic_choice_rows(
                rows,
                model=model,
                language_mode=loaded_config.language_mode,
                run_label="loaded_checkpoint",
                args=args,
            )
            _write_and_print(rows, args)
            return
        if args.drift_forecast_audit_contexts > 0:
            audit = audit_metabolic_drift_forecast(
                model,
                contexts=args.drift_forecast_audit_contexts,
                semantic_choice_horizon=args.semantic_choice_horizon,
                semantic_choice_low_need=args.semantic_choice_low_need,
                semantic_choice_return_duration=(
                    args.semantic_choice_return_duration
                ),
                semantic_choice_rounds=args.semantic_choice_rounds,
            )
            rows.append(
                {"condition": "metabolic_drift_forecast", **audit}
            )
            _write_and_print(rows, args)
            return
        if args.protocol_return_origin_audit_contexts > 0:
            audit = audit_protocol_return_origin(
                model,
                contexts=args.protocol_return_origin_audit_contexts,
                semantic_choice_horizon=args.semantic_choice_horizon,
                semantic_choice_low_need=args.semantic_choice_low_need,
                semantic_choice_return_duration=(
                    args.semantic_choice_return_duration
                ),
                semantic_choice_rounds=args.semantic_choice_rounds,
            )
            rows.append(
                {"condition": "protocol_return_origin", **audit}
            )
            _write_and_print(rows, args)
            return
        if args.terminal_consume_calibration_contexts > 0:
            audit = audit_terminal_consume_value_calibration(
                model,
                contexts=args.terminal_consume_calibration_contexts,
                semantic_choice_horizon=args.semantic_choice_horizon,
                semantic_choice_low_need=args.semantic_choice_low_need,
                semantic_choice_return_duration=(
                    args.semantic_choice_return_duration
                ),
                semantic_choice_rounds=args.semantic_choice_rounds,
            )
            rows.append(
                {"condition": "terminal_consume_value_calibration", **audit}
            )
            _write_and_print(rows, args)
            return
        if args.real_explicit_event_transfer_contexts > 0:
            audit = audit_real_vs_explicit_event_transfer(
                model,
                contexts=args.real_explicit_event_transfer_contexts,
                semantic_choice_horizon=args.semantic_choice_horizon,
                semantic_choice_low_need=args.semantic_choice_low_need,
                semantic_choice_return_duration=(
                    args.semantic_choice_return_duration
                ),
                semantic_choice_rounds=args.semantic_choice_rounds,
            )
            rows.append(
                {"condition": "real_vs_explicit_event_transfer", **audit}
            )
            _write_and_print(rows, args)
            return
        if args.binding_consequence_geometry_contexts > 0:
            audit = audit_binding_consequence_geometry(
                model,
                contexts=args.binding_consequence_geometry_contexts,
                semantic_choice_horizon=args.semantic_choice_horizon,
                semantic_choice_low_need=args.semantic_choice_low_need,
                semantic_choice_return_duration=(
                    args.semantic_choice_return_duration
                ),
                semantic_choice_rounds=args.semantic_choice_rounds,
            )
            rows.append(
                {"condition": "binding_consequence_geometry", **audit}
            )
            _write_and_print(rows, args)
            return
        if args.bound_event_gradient_alignment_segments > 0:
            audit = audit_bound_event_gradient_alignment(
                model,
                loaded_config,
                segments=args.bound_event_gradient_alignment_segments,
            )
            raw_samples = audit.pop("raw_samples")
            raw_path = (
                Path(args.run_dir)
                / "bound_event_gradient_alignment_samples.json"
            )
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(
                json.dumps(raw_samples, indent=2) + "\n",
                encoding="utf-8",
            )
            rows.append(
                {"condition": "bound_event_gradient_alignment", **audit}
            )
            _write_and_print(rows, args)
            return
        if args.cross_round_reuse_lives > 0:
            for label, mode, writes in (
                ("grounded", "grounded", True),
                ("acute_silent", "silent", True),
                ("acute_no_writes", "grounded", False),
            ):
                writes_before = model.episodic_binding_writes
                model.episodic_binding_writes = writes
                try:
                    audit = audit_cross_round_label_reuse(
                        model,
                        lives=args.cross_round_reuse_lives,
                        semantic_choice_horizon=args.semantic_choice_horizon,
                        semantic_choice_low_need=args.semantic_choice_low_need,
                        semantic_choice_return_duration=(
                            args.semantic_choice_return_duration
                        ),
                        language_mode=mode,
                        urgent_deficit_utility=args.urgent_deficit_utility,
                    )
                finally:
                    model.episodic_binding_writes = writes_before
                rows.append(
                    {"condition": f"cross_round_label_reuse_{label}", **audit}
                )
            _write_and_print(rows, args)
            return
        audit = audit_observation_branching_planner(
            model,
            episodes=args.observation_branching_audit_contexts,
            semantic_choice_horizon=args.semantic_choice_horizon,
            semantic_choice_low_need=args.semantic_choice_low_need,
            semantic_choice_return_duration=(
                args.semantic_choice_return_duration
            ),
            persistent_information_reuses=(
                args.persistent_information_reuses
            ),
            urgent_deficit_utility=args.urgent_deficit_utility,
            protocol_branch=args.protocol_branch_planning,
            protocol_return_from_inspect_state=(
                args.protocol_return_from_inspect_state
            ),
        )
        rows.append(
            {
                "condition": "checkpoint_observation_branching_feasibility",
                **audit,
            }
        )
        _write_and_print(rows, args)
        return

    if args.eval_episodes > 0:
        for baseline in args.baselines:
            stats = run_policy(
                baseline,
                episodes=args.eval_episodes,
                config=IslandConfig(max_steps=args.eval_max_steps),
                base_seed=EVAL_SEED_BASE,
            )
            rows.append({"condition": f"baseline_{baseline}", **stats})

    for language_mode in args.language_modes:
        model_horizon = args.model_horizon or args.planning_horizon
        run_label = language_mode
        if args.bc_warmstart:
            run_label += f"_bc{args.bc_lives}"
        if args.offer_childhood:
            run_label += (
                f"_offer{args.offer_fade_steps}d{args.offer_distance_end}"
            )
        if args.semantic_choice_childhood_steps > 0:
            run_label += (
                f"_choice{args.semantic_choice_childhood_steps}"
                + (
                    f"x{args.semantic_choice_objects}"
                    if args.semantic_choice_objects != 2
                    else ""
                )
                + f"h{args.semantic_choice_horizon}"
                + (
                    f"r{args.semantic_choice_return_duration}"
                    if args.semantic_choice_return_duration > 0
                    else ""
                )
                + (
                    f"n{args.semantic_choice_low_need:g}"
                    if args.semantic_choice_low_need != 0.35
                    else ""
                )
                + (
                    f"q{args.semantic_choice_rounds}"
                    if args.semantic_choice_rounds != 1
                    else ""
                )
            )
        if args.consume_options:
            run_label += "_options"
        if args.inspect_options:
            run_label += "_inspect"
        if args.episodic_binding_size > 0:
            run_label += f"_bind{args.episodic_binding_size}"
            if args.disable_episodic_binding_writes:
                run_label += "nowrite"
        if args.self_model_planning:
            run_label += (
                f"_plan{args.planning_scale:g}h{args.planning_horizon}"
            )
            if args.observation_branching_planning:
                run_label += "branch"
                if args.persistent_information_reuses > 0:
                    run_label += f"q{args.persistent_information_reuses}"
            if args.urgent_deficit_utility:
                run_label += "urgent"
        if args.replay_updates > 0:
            run_label += f"_replay{args.replay_capacity}x{args.replay_updates}"
        if args.bodily_drift_loss_weight > 0.0:
            run_label += f"_drift{args.bodily_drift_loss_weight:g}"
        if args.bodily_event_loss_weight > 0.0:
            run_label += f"_event{args.bodily_event_loss_weight:g}"
        if args.bodily_consumption_event_loss_weight > 0.0:
            run_label += (
                f"_consume_event{args.bodily_consumption_event_loss_weight:g}"
            )
        if args.bodily_bound_consumption_event_loss_weight > 0.0:
            run_label += (
                "_bound_consume_event"
                f"{args.bodily_bound_consumption_event_loss_weight:g}"
            )
        if args.split_drift_head:
            run_label += "split"
        if args.max_grad_norm != 1.0:
            run_label += f"_clip{args.max_grad_norm:g}"
        if model_horizon != args.planning_horizon:
            run_label += f"_modelh{model_horizon}"
        config = OrganismConfig(
            language_mode=language_mode,
            total_steps=args.train_steps,
            segment_length=args.segment_length,
            hidden_size=args.hidden_size,
            bc_warmstart_lives=args.bc_lives if args.bc_warmstart else 0,
            bc_epochs=args.bc_epochs,
            caregiver_offer_threshold_start=(
                args.offer_threshold if args.offer_childhood else 0.0
            ),
            caregiver_offer_curriculum_steps=(
                args.offer_fade_steps if args.offer_childhood else 0
            ),
            caregiver_offer_distance_end=(
                args.offer_distance_end if args.offer_childhood else 0
            ),
            semantic_choice_childhood_steps=(
                args.semantic_choice_childhood_steps
            ),
            consume_options=args.consume_options,
            inspect_options=args.inspect_options,
            episodic_binding_size=args.episodic_binding_size,
            episodic_binding_writes=not args.disable_episodic_binding_writes,
            self_model_planning_scale=(
                args.planning_scale if args.self_model_planning else 0.0
            ),
            self_model_planning_start_steps=args.planning_start_steps,
            self_model_planning_reward_weight=args.planning_reward_weight,
            self_model_planning_horizon=args.planning_horizon,
            observation_branching_planning=(
                args.observation_branching_planning
            ),
            persistent_information_reuses=args.persistent_information_reuses,
            urgent_deficit_utility=args.urgent_deficit_utility,
            protocol_branch_planning=args.protocol_branch_planning,
            multi_step_model_horizon=model_horizon,
            multi_step_model_weight=args.multi_step_model_weight,
            bodily_drift_loss_weight=args.bodily_drift_loss_weight,
            bodily_event_loss_weight=args.bodily_event_loss_weight,
            bodily_consumption_event_loss_weight=(
                args.bodily_consumption_event_loss_weight
            ),
            bodily_bound_consumption_event_loss_weight=(
                args.bodily_bound_consumption_event_loss_weight
            ),
            split_drift_head=args.split_drift_head,
            max_grad_norm=args.max_grad_norm,
            world_model_replay_capacity=args.replay_capacity,
            world_model_replay_updates=args.replay_updates,
            seed=args.seed,
            max_steps=args.train_max_steps,
            island=IslandConfig(
                semantic_choice_horizon=args.semantic_choice_horizon,
                semantic_choice_objects=args.semantic_choice_objects,
                semantic_choice_low_need=args.semantic_choice_low_need,
                semantic_choice_rounds=args.semantic_choice_rounds,
                semantic_choice_return_duration=(
                    args.semantic_choice_return_duration
                ),
            ),
            checkpoint=str(
                Path(args.run_dir) / f"organism_{run_label}_seed{args.seed}.npz"
            ),
            stats_csv=str(
                Path(args.run_dir) / f"lives_{run_label}_seed{args.seed}.csv"
            ),
            bodily_event_audit_json=(
                str(
                    Path(args.run_dir)
                    / f"bodily_event_training_path_{run_label}_seed{args.seed}.json"
                )
                if args.audit_bodily_event_training_path
                else None
            ),
        )
        print(f"=== training organism under {language_mode} ===")
        model, _ = train_organism(config)
        if args.eval_episodes <= 0:
            _append_semantic_choice_rows(
                rows,
                model=model,
                language_mode=language_mode,
                run_label=run_label,
                args=args,
            )
            continue
        stats = evaluate_organism(
            model,
            language_mode=language_mode,
            episodes=args.eval_episodes,
            base_seed=EVAL_SEED_BASE,
            max_steps=args.eval_max_steps,
            greedy=args.eval_greedy,
            consume_options=args.consume_options,
            inspect_options=args.inspect_options,
            self_model_planning_scale=(
                args.planning_scale if args.self_model_planning else 0.0
            ),
            self_model_planning_reward_weight=args.planning_reward_weight,
            self_model_planning_horizon=args.planning_horizon,
        )
        if args.self_model_audit_decisions > 0:
            audit = audit_self_model_actions(
                model,
                language_mode=language_mode,
                episodes=args.eval_episodes,
                base_seed=EVAL_SEED_BASE,
                max_decisions=args.self_model_audit_decisions,
                max_steps=args.eval_max_steps,
                consume_options=args.consume_options,
                inspect_options=args.inspect_options,
                self_model_planning_scale=(
                    args.planning_scale if args.self_model_planning else 0.0
                ),
                reward_weight=args.planning_reward_weight,
                self_model_planning_horizon=args.planning_horizon,
            )
            stats.update({f"self_model_{key}": value for key, value in audit.items()})
        if args.label_self_model_audit_inspections > 0:
            label_audit = audit_label_to_self_model(
                model,
                episodes=args.eval_episodes,
                base_seed=EVAL_SEED_BASE,
                max_inspections=args.label_self_model_audit_inspections,
                max_steps=args.eval_max_steps,
                self_model_planning_scale=(
                    args.planning_scale if args.self_model_planning else 0.0
                ),
                self_model_planning_horizon=args.planning_horizon,
            )
            stats.update(
                {f"label_audit_{key}": value for key, value in label_audit.items()}
            )
        rows.append({"condition": f"organism_{run_label}", **stats})
        if args.self_model_interventions and args.self_model_planning:
            for intervention, scale, score_sign in (
                ("planning_removed", 0.0, 1.0),
                ("planning_reversed", args.planning_scale, -1.0),
            ):
                intervention_stats = evaluate_organism(
                    model,
                    language_mode=language_mode,
                    episodes=args.eval_episodes,
                    base_seed=EVAL_SEED_BASE,
                    max_steps=args.eval_max_steps,
                    greedy=args.eval_greedy,
                    consume_options=args.consume_options,
                    inspect_options=args.inspect_options,
                    self_model_planning_scale=scale,
                    self_model_planning_reward_weight=args.planning_reward_weight,
                    self_model_planning_score_sign=score_sign,
                    self_model_planning_horizon=args.planning_horizon,
                )
                rows.append(
                    {
                        "condition": f"organism_{run_label}_{intervention}",
                        **intervention_stats,
                    }
                )
        if args.body_only_self_model_audit and args.self_model_planning:
            body_stats = evaluate_organism(
                model,
                language_mode=language_mode,
                episodes=args.eval_episodes,
                base_seed=EVAL_SEED_BASE,
                max_steps=args.eval_max_steps,
                greedy=args.eval_greedy,
                consume_options=args.consume_options,
                inspect_options=args.inspect_options,
                self_model_planning_scale=args.planning_scale,
                self_model_planning_reward_weight=0.0,
                self_model_planning_horizon=args.planning_horizon,
            )
            if args.self_model_audit_decisions > 0:
                body_audit = audit_self_model_actions(
                    model,
                    language_mode=language_mode,
                    episodes=args.eval_episodes,
                    base_seed=EVAL_SEED_BASE,
                    max_decisions=args.self_model_audit_decisions,
                    max_steps=args.eval_max_steps,
                    consume_options=args.consume_options,
                    inspect_options=args.inspect_options,
                    self_model_planning_scale=args.planning_scale,
                    reward_weight=0.0,
                    self_model_planning_horizon=args.planning_horizon,
                )
                body_stats.update(
                    {f"self_model_{key}": value for key, value in body_audit.items()}
                )
            rows.append(
                {
                    "condition": f"organism_{run_label}_body_only",
                    **body_stats,
                }
            )
            if args.self_model_interventions:
                body_reversed = evaluate_organism(
                    model,
                    language_mode=language_mode,
                    episodes=args.eval_episodes,
                    base_seed=EVAL_SEED_BASE,
                    max_steps=args.eval_max_steps,
                    greedy=args.eval_greedy,
                    consume_options=args.consume_options,
                    inspect_options=args.inspect_options,
                    self_model_planning_scale=args.planning_scale,
                    self_model_planning_reward_weight=0.0,
                    self_model_planning_score_sign=-1.0,
                    self_model_planning_horizon=args.planning_horizon,
                )
                rows.append(
                    {
                        "condition": f"organism_{run_label}_body_only_reversed",
                        **body_reversed,
                    }
                )

        _append_semantic_choice_rows(
            rows,
            model=model,
            language_mode=language_mode,
            run_label=run_label,
            args=args,
        )

    _write_and_print(rows, args)


def _append_semantic_choice_rows(
    rows: list[dict[str, object]],
    *,
    model: OrganismModel,
    language_mode: str,
    run_label: str,
    args: argparse.Namespace,
) -> None:
    """Append fixed-seed semantic-choice evaluations and causal audits."""

    if args.semantic_choice_eval_episodes <= 0:
        return

    planning_scale = args.planning_scale if args.self_model_planning else 0.0
    branching_kwargs = {
        "observation_branching_planning": args.observation_branching_planning,
        "persistent_information_reuses": args.persistent_information_reuses,
        "urgent_deficit_utility": args.urgent_deficit_utility,
        "protocol_branch_planning": args.protocol_branch_planning,
    }
    for greedy, policy_name in ((False, "stochastic"), (True, "greedy")):
        stats = evaluate_semantic_choice(
            model,
            language_mode=language_mode,
            episodes=args.semantic_choice_eval_episodes,
            base_seed=EVAL_SEED_BASE,
            semantic_choice_horizon=args.semantic_choice_horizon,
            semantic_choice_objects=args.semantic_choice_objects,
            semantic_choice_low_need=args.semantic_choice_low_need,
            semantic_choice_rounds=args.semantic_choice_rounds,
            semantic_choice_return_duration=(
                args.semantic_choice_return_duration
            ),
            sample_seed=0,
            greedy=greedy,
            consume_options=args.consume_options,
            inspect_options=args.inspect_options,
            self_model_planning_scale=planning_scale,
            self_model_planning_reward_weight=args.planning_reward_weight,
            self_model_planning_horizon=args.planning_horizon,
            **branching_kwargs,
        )

        # Audits are policy-sampled diagnostics, so one stochastic fixed-seed
        # row is the canonical home for them. The greedy row remains a compact
        # behavioral diagnostic over the exact same environment seeds.
        if not greedy and args.self_model_audit_decisions > 0:
            audit = audit_self_model_actions(
                model,
                language_mode=language_mode,
                episodes=args.semantic_choice_eval_episodes,
                base_seed=EVAL_SEED_BASE,
                max_decisions=args.self_model_audit_decisions,
                max_steps=args.semantic_choice_horizon,
                consume_options=args.consume_options,
                inspect_options=args.inspect_options,
                self_model_planning_scale=planning_scale,
                reward_weight=args.planning_reward_weight,
                self_model_planning_horizon=args.planning_horizon,
                semantic_choice_trial=True,
                semantic_choice_horizon=args.semantic_choice_horizon,
                semantic_choice_objects=args.semantic_choice_objects,
                semantic_choice_low_need=args.semantic_choice_low_need,
                semantic_choice_return_duration=(
                    args.semantic_choice_return_duration
                ),
            )
            stats.update(
                {
                    f"prelabel_self_model_{key}": value
                    for key, value in audit.items()
                }
            )

        if not greedy and args.label_self_model_audit_inspections > 0:
            label_audit = audit_label_to_self_model(
                model,
                episodes=args.semantic_choice_eval_episodes,
                base_seed=EVAL_SEED_BASE,
                max_inspections=args.label_self_model_audit_inspections,
                max_steps=args.semantic_choice_horizon,
                self_model_planning_scale=planning_scale,
                self_model_planning_horizon=args.planning_horizon,
                semantic_choice_trial=True,
                semantic_choice_horizon=args.semantic_choice_horizon,
                semantic_choice_objects=args.semantic_choice_objects,
                semantic_choice_low_need=args.semantic_choice_low_need,
                semantic_choice_return_duration=(
                    args.semantic_choice_return_duration
                ),
            )
            stats.update(
                {f"label_audit_{key}": value for key, value in label_audit.items()}
            )

        if not greedy and args.label_referent_audit_contexts > 0:
            referent_audit = audit_label_referent_binding(
                model,
                episodes=args.semantic_choice_eval_episodes,
                base_seed=EVAL_SEED_BASE,
                max_contexts=args.label_referent_audit_contexts,
                semantic_choice_horizon=args.semantic_choice_horizon,
                semantic_choice_objects=args.semantic_choice_objects,
                semantic_choice_low_need=args.semantic_choice_low_need,
                semantic_choice_return_duration=(
                    args.semantic_choice_return_duration
                ),
            )
            stats.update(
                {
                    f"referent_audit_{key}": value
                    for key, value in referent_audit.items()
                }
            )

        if (
            not greedy
            and args.body_only_self_model_audit
            and args.self_model_planning
            and args.self_model_audit_decisions > 0
        ):
            body_audit = audit_self_model_actions(
                model,
                language_mode=language_mode,
                episodes=args.semantic_choice_eval_episodes,
                base_seed=EVAL_SEED_BASE,
                max_decisions=args.self_model_audit_decisions,
                max_steps=args.semantic_choice_horizon,
                consume_options=args.consume_options,
                inspect_options=args.inspect_options,
                self_model_planning_scale=args.planning_scale,
                reward_weight=0.0,
                self_model_planning_horizon=args.planning_horizon,
                semantic_choice_trial=True,
                semantic_choice_horizon=args.semantic_choice_horizon,
                semantic_choice_objects=args.semantic_choice_objects,
                semantic_choice_low_need=args.semantic_choice_low_need,
                semantic_choice_return_duration=(
                    args.semantic_choice_return_duration
                ),
            )
            stats.update(
                {
                    f"prelabel_body_only_self_model_{key}": value
                    for key, value in body_audit.items()
                }
            )

        rows.append(
            {
                "condition": (
                    f"organism_{run_label}_semantic_choice_{policy_name}"
                ),
                **stats,
            }
        )

    for acute_mode in args.semantic_choice_acute_modes:
        acute_stats = evaluate_semantic_choice(
            model,
            language_mode=acute_mode,
            episodes=args.semantic_choice_eval_episodes,
            base_seed=EVAL_SEED_BASE,
            semantic_choice_horizon=args.semantic_choice_horizon,
            semantic_choice_objects=args.semantic_choice_objects,
            semantic_choice_low_need=args.semantic_choice_low_need,
            semantic_choice_rounds=args.semantic_choice_rounds,
            semantic_choice_return_duration=(
                args.semantic_choice_return_duration
            ),
            sample_seed=0,
            greedy=False,
            consume_options=args.consume_options,
            inspect_options=args.inspect_options,
            self_model_planning_scale=planning_scale,
            self_model_planning_reward_weight=args.planning_reward_weight,
            self_model_planning_horizon=args.planning_horizon,
            **branching_kwargs,
        )
        rows.append(
            {
                "condition": (
                    f"organism_{run_label}_semantic_choice_acute_{acute_mode}"
                ),
                **acute_stats,
            }
        )

    if args.semantic_choice_acute_disable_binding_writes:
        if not model.has_episodic_bindings:
            raise ValueError(
                "Acute binding-write ablation requires episodic bindings."
            )
        writes_before = model.episodic_binding_writes
        model.episodic_binding_writes = False
        try:
            acute_stats = evaluate_semantic_choice(
                model,
                language_mode=language_mode,
                episodes=args.semantic_choice_eval_episodes,
                base_seed=EVAL_SEED_BASE,
                semantic_choice_horizon=args.semantic_choice_horizon,
                semantic_choice_objects=args.semantic_choice_objects,
                semantic_choice_low_need=args.semantic_choice_low_need,
                semantic_choice_rounds=args.semantic_choice_rounds,
                semantic_choice_return_duration=(
                    args.semantic_choice_return_duration
                ),
                sample_seed=0,
                greedy=False,
                consume_options=args.consume_options,
                inspect_options=args.inspect_options,
                self_model_planning_scale=planning_scale,
                self_model_planning_reward_weight=args.planning_reward_weight,
                self_model_planning_horizon=args.planning_horizon,
                **branching_kwargs,
            )
        finally:
            model.episodic_binding_writes = writes_before
        rows.append(
            {
                "condition": (
                    f"organism_{run_label}_semantic_choice_acute_no_writes"
                ),
                **acute_stats,
            }
        )


def _write_and_print(rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    keys = ["condition"]
    for row in rows:
        keys.extend(key for key in row if key not in keys)
    suffix = f"_bc{args.bc_lives}" if args.bc_warmstart else ""
    if args.offer_childhood:
        suffix += f"_offer{args.offer_fade_steps}d{args.offer_distance_end}"
    if args.semantic_choice_childhood_steps > 0:
        suffix += (
            f"_choice{args.semantic_choice_childhood_steps}"
            + (
                f"x{args.semantic_choice_objects}"
                if args.semantic_choice_objects != 2
                else ""
            )
            + f"h{args.semantic_choice_horizon}"
            + (
                f"r{args.semantic_choice_return_duration}"
                if args.semantic_choice_return_duration > 0
                else ""
            )
            + (
                f"n{args.semantic_choice_low_need:g}"
                if args.semantic_choice_low_need != 0.35
                else ""
            )
            + (
                f"q{args.semantic_choice_rounds}"
                if args.semantic_choice_rounds != 1
                else ""
            )
        )
    if args.consume_options:
        suffix += "_options"
    if args.inspect_options:
        suffix += "_inspect"
    if args.episodic_binding_size > 0:
        suffix += f"_bind{args.episodic_binding_size}"
        if args.disable_episodic_binding_writes:
            suffix += "nowrite"
    if args.self_model_planning:
        suffix += f"_plan{args.planning_scale:g}h{args.planning_horizon}"
    if args.replay_updates > 0:
        suffix += f"_replay{args.replay_capacity}x{args.replay_updates}"
    if args.bodily_drift_loss_weight > 0.0:
        suffix += f"_drift{args.bodily_drift_loss_weight:g}"
    if args.bodily_event_loss_weight > 0.0:
        suffix += f"_event{args.bodily_event_loss_weight:g}"
    if args.bodily_consumption_event_loss_weight > 0.0:
        suffix += (
            f"_consume_event{args.bodily_consumption_event_loss_weight:g}"
        )
    if args.bodily_bound_consumption_event_loss_weight > 0.0:
        suffix += (
            "_bound_consume_event"
            f"{args.bodily_bound_consumption_event_loss_weight:g}"
        )
    if args.split_drift_head:
        suffix += "split"
    if args.max_grad_norm != 1.0:
        suffix += f"_clip{args.max_grad_norm:g}"
    model_horizon = args.model_horizon or args.planning_horizon
    if model_horizon != args.planning_horizon:
        suffix += f"_modelh{model_horizon}"
    languages = "-".join(args.language_modes)
    out_path = (
        Path(args.run_dir)
        / f"harness_seed{args.seed}_{languages}{suffix}.csv"
    )
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
    parser.add_argument(
        "--load-checkpoint",
        default=None,
        help=(
            "Load an existing unified-organism checkpoint for a read-only "
            "audit instead of training."
        ),
    )
    parser.add_argument(
        "--evaluate-loaded-checkpoint",
        action="store_true",
        help=(
            "Run semantic-choice evaluations on --load-checkpoint without "
            "training; task mechanics are read from checkpoint metadata."
        ),
    )
    parser.add_argument(
        "--observation-branching-audit-contexts",
        type=int,
        default=0,
        help=(
            "Run the preregistered delayed-choice belief-planner feasibility "
            "audit on this many fixed contexts."
        ),
    )
    parser.add_argument(
        "--semantic-choice-information-upper-bound-contexts",
        type=int,
        default=0,
        help=(
            "Run the audit-only exact-environment information-rent ceiling "
            "on this many fixed delayed-choice contexts instead of training."
        ),
    )
    parser.add_argument(
        "--persistent-mapping-rent-lives",
        type=int,
        default=0,
        help=(
            "Run the preregistered exact-dynamics persistent-mapping rent "
            "audit on this many fixed lives instead of training."
        ),
    )
    parser.add_argument(
        "--persistent-choice-mechanics-lives",
        type=int,
        default=0,
        help=(
            "Audit the implemented eight-round persistent-choice environment "
            "with public-label and blind policies."
        ),
    )
    parser.add_argument("--train-steps", type=int, default=200_000)
    parser.add_argument("--segment-length", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument(
        "--bc-warmstart",
        action="store_true",
        help="Pretrain on token-masked oracle lives before online learning.",
    )
    parser.add_argument("--bc-lives", type=int, default=100)
    parser.add_argument("--bc-epochs", type=int, default=2)
    parser.add_argument(
        "--offer-childhood",
        action="store_true",
        help="Fade in-loop caregiver food/water offers to zero during training.",
    )
    parser.add_argument("--offer-threshold", type=float, default=0.75)
    parser.add_argument("--offer-fade-steps", type=int, default=100_000)
    parser.add_argument("--offer-distance-end", type=int, default=0)
    parser.add_argument(
        "--semantic-choice-childhood-steps",
        type=int,
        default=0,
        help=(
            "Train in remapped semantic-choice trials for this many primitive "
            "ticks before switching to the open island."
        ),
    )
    parser.add_argument(
        "--semantic-choice-horizon",
        type=int,
        default=20,
        help="Maximum primitive ticks in each semantic-choice trial.",
    )
    parser.add_argument(
        "--semantic-choice-objects",
        type=int,
        choices=[2, 3],
        default=2,
        help=(
            "Use the original resource/poison pair or a crossed "
            "food/water/poison choice."
        ),
    )
    parser.add_argument(
        "--semantic-choice-return-duration",
        type=int,
        default=0,
        help=(
            "After each label in the delayed probe, force a fixed-duration "
            "padding-only return to center/NORTH; zero disables."
        ),
    )
    parser.add_argument(
        "--semantic-choice-rounds",
        type=int,
        default=1,
        help=(
            "Keep one hidden surface-kind mapping across this many recurring "
            "body-choice rounds per childhood life."
        ),
    )
    parser.add_argument(
        "--semantic-choice-low-need",
        type=float,
        default=0.35,
        help="Initial value of the independently selected low bodily need.",
    )
    parser.add_argument(
        "--semantic-choice-eval-episodes",
        type=int,
        default=0,
        help=(
            "Evaluate both stochastic and greedy policies on this many "
            "fixed-seed held-out semantic-choice trials."
        ),
    )
    parser.add_argument(
        "--semantic-choice-acute-modes",
        nargs="*",
        default=[],
        choices=["silent", "shuffled"],
        help=(
            "Also evaluate the trained model under these acute token-channel "
            "interventions without parameter updates."
        ),
    )
    parser.add_argument(
        "--semantic-choice-acute-disable-binding-writes",
        action="store_true",
        help=(
            "Evaluate the trained checkpoint with every episodic write "
            "suppressed, without parameter updates."
        ),
    )
    parser.add_argument(
        "--consume-options",
        action="store_true",
        help="Add kind-blind visible-slot navigate-and-consume actions.",
    )
    parser.add_argument(
        "--inspect-options",
        action="store_true",
        help="Add kind-blind visible-slot navigate-face-and-ask actions.",
    )
    parser.add_argument(
        "--episodic-binding-size",
        type=int,
        default=0,
        help=(
            "Width of each per-life visual-key/lexical-value binding; zero "
            "uses the recurrent-only control."
        ),
    )
    parser.add_argument(
        "--disable-episodic-binding-writes",
        action="store_true",
        help=(
            "Keep the binding architecture and parameters but suppress every "
            "episodic write as a matched causal ablation."
        ),
    )
    parser.add_argument(
        "--self-model-planning",
        action="store_true",
        help="Bias action logits with detached predicted future-body value.",
    )
    parser.add_argument("--planning-start-steps", type=int, default=100_000)
    parser.add_argument("--planning-scale", type=float, default=6.0)
    parser.add_argument("--planning-reward-weight", type=float, default=0.5)
    parser.add_argument("--planning-horizon", type=int, choices=[1, 2], default=1)
    parser.add_argument(
        "--observation-branching-planning",
        action="store_true",
        help=(
            "Value inspection by branching over learner-possible caregiver "
            "labels, writing each hypothetically, and scoring the resulting "
            "embodied choice. Requires --planning-horizon 2 and "
            "--semantic-choice-return-duration."
        ),
    )
    parser.add_argument(
        "--cross-round-reuse-lives",
        type=int,
        default=0,
        help=(
            "Read-only audit: acquire one food and one water label in the "
            "first rounds, then measure whether the organism's own terminal "
            "scores still select the needed object in every later round."
        ),
    )
    parser.add_argument(
        "--drift-forecast-audit-contexts",
        type=int,
        default=0,
        help=(
            "Read-only audit: predicted against realized bodily drift over "
            "the planner's own inspect and return steps, with the oracle "
            "substitution that bounds how well the chain could rank needs."
        ),
    )
    parser.add_argument(
        "--protocol-return-origin-audit-contexts",
        type=int,
        default=0,
        help=(
            "Read-only paired audit: compare the sealed pre-inspection return "
            "query with a return query chained from the learned inspect latent."
        ),
    )
    parser.add_argument(
        "--terminal-consume-calibration-contexts",
        type=int,
        default=0,
        help=(
            "Read-only audit: compare learned and simulator-scored terminal "
            "consume values at the protocol branch decision."
        ),
    )
    parser.add_argument(
        "--real-explicit-event-transfer-contexts",
        type=int,
        default=0,
        help=(
            "Read-only paired audit: compare terminal value after real label "
            "acquisition and an explicit matched lexical write."
        ),
    )
    parser.add_argument(
        "--binding-consequence-geometry-contexts",
        type=int,
        default=0,
        help=(
            "Read-only audit: separate controlled lexical-value geometry from "
            "its fixed-core, fixed-action bodily consequence map."
        ),
    )
    parser.add_argument(
        "--bound-event-gradient-alignment-segments",
        type=int,
        default=0,
        help=(
            "Read-only audit: collect fresh fixed-policy bound-event segments "
            "and compare base versus calibration gradient direction."
        ),
    )
    parser.add_argument(
        "--protocol-branch-planning",
        action="store_true",
        help=(
            "Value inspection by writing each candidate word to the lexical "
            "bank and travelling the public padding-only protocol, instead of "
            "reconstructing a post-inspect observation with the decoder."
        ),
    )
    parser.add_argument(
        "--protocol-return-from-inspect-state",
        action="store_true",
        help=(
            "Repair the protocol branch's return query by composing the "
            "learned inspect latent into WAIT. The recurrent memory-settling "
            "path remains observation-driven and unchanged."
        ),
    )
    parser.add_argument(
        "--urgent-deficit-utility",
        action="store_true",
        help=(
            "Score an imagined action by the predicted level of the need the "
            "organism currently observes as lowest, instead of the predicted "
            "minimum over all needs. Only own interoception is read."
        ),
    )
    parser.add_argument(
        "--persistent-information-reuses",
        type=int,
        default=0,
        help=(
            "Back a hypothetical lexical write up through this many recurring "
            "future bodily contexts, matching the public multi-round horizon."
        ),
    )
    parser.add_argument("--model-horizon", type=int, choices=[1, 2], default=None)
    parser.add_argument("--multi-step-model-weight", type=float, default=0.0)
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help=(
            "Global gradient-norm trust region. Auxiliary losses with a large "
            "gradient scale otherwise steal step size from every other "
            "objective through this shared budget."
        ),
    )
    parser.add_argument(
        "--split-drift-head",
        action="store_true",
        help=(
            "Give slow metabolism its own range-limited output path so it "
            "cannot overwrite the binding-conditioned consumption jump."
        ),
    )
    parser.add_argument(
        "--bodily-drift-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Supervise the slow-metabolism regime the change boost starves. "
            "Zero is the sealed default and reproduces prior artifacts."
        ),
    )
    parser.add_argument(
        "--bodily-event-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Calibrate rare lived bodily events per need at their measured "
            "physical scale. Zero preserves prior loss behavior exactly."
        ),
    )
    parser.add_argument(
        "--bodily-consumption-event-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Calibrate above-threshold bodily changes only at public consume "
            "actions; horizon-two calibration requires consume as the final "
            "action. Zero preserves prior behavior."
        ),
    )
    parser.add_argument(
        "--bodily-bound-consumption-event-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Calibrate only consume events whose selected surface already has "
            "a valid lexical binding, with selected errors balanced per bodily "
            "need. Zero preserves prior behavior."
        ),
    )
    parser.add_argument(
        "--audit-bodily-event-training-path",
        action="store_true",
        help=(
            "Record target coverage, selected-surface binding validity, "
            "pre-update error, replay composition, and sampled event gradients. "
            "The audit does not change learning."
        ),
    )
    parser.add_argument("--replay-capacity", type=int, default=0)
    parser.add_argument("--replay-updates", type=int, default=0)
    parser.add_argument(
        "--self-model-audit-decisions",
        type=int,
        default=0,
        help=(
            "For this many held-out states, branch simulator copies to score "
            "all available actions against the model's bodily predictions."
        ),
    )
    parser.add_argument(
        "--label-self-model-audit-inspections",
        type=int,
        default=0,
        help=(
            "Audit this many held-out inspect events under true, silent, and "
            "counterfactual kind labels. Requires consume and inspect options."
        ),
    )
    parser.add_argument(
        "--label-referent-audit-contexts",
        type=int,
        default=0,
        help=(
            "Audit delayed target versus nonreferent consequence binding in "
            "this many fixed held-out choice contexts."
        ),
    )
    parser.add_argument(
        "--self-model-interventions",
        action="store_true",
        help="Also evaluate the trained checkpoint with planning removed/reversed.",
    )
    parser.add_argument(
        "--body-only-self-model-audit",
        action="store_true",
        help=(
            "Also evaluate/audit planning with predicted external reward weight "
            "zero, isolating the bodily consequence channel."
        ),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--train-max-steps", type=int, default=1000)
    parser.add_argument("--eval-max-steps", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument(
        "--eval-greedy",
        action="store_true",
        help="Use argmax actions for a deterministic policy diagnostic.",
    )
    parser.add_argument(
        "--language-modes",
        nargs="+",
        default=["grounded", "silent", "shuffled"],
        choices=["grounded", "silent", "shuffled"],
    )
    parser.add_argument(
        "--baselines", nargs="*", default=["oracle", "random"],
        choices=["oracle", "random"],
    )
    parser.add_argument("--run-dir", default="runs/organism")
    args = parser.parse_args()
    if args.disable_episodic_binding_writes and args.episodic_binding_size <= 0:
        parser.error(
            "--disable-episodic-binding-writes requires --episodic-binding-size."
        )
    if (
        args.semantic_choice_acute_disable_binding_writes
        and args.episodic_binding_size <= 0
    ):
        parser.error(
            "--semantic-choice-acute-disable-binding-writes requires "
            "--episodic-binding-size."
        )
    if args.protocol_branch_planning and not args.observation_branching_planning:
        parser.error(
            "--protocol-branch-planning requires "
            "--observation-branching-planning."
        )
    if (
        args.protocol_return_from_inspect_state
        and not args.protocol_branch_planning
    ):
        parser.error(
            "--protocol-return-from-inspect-state requires "
            "--protocol-branch-planning."
        )
    if args.observation_branching_planning:
        if args.planning_horizon != 2:
            parser.error(
                "--observation-branching-planning requires "
                "--planning-horizon 2."
            )
        if args.semantic_choice_return_duration <= 0:
            parser.error(
                "--observation-branching-planning requires a positive "
                "--semantic-choice-return-duration."
            )
        if args.episodic_binding_size <= 0:
            parser.error(
                "--observation-branching-planning requires "
                "--episodic-binding-size."
            )
    if args.persistent_information_reuses < 0:
        parser.error("--persistent-information-reuses must be nonnegative.")
    if args.bodily_drift_loss_weight < 0.0:
        parser.error("--bodily-drift-loss-weight must be nonnegative.")
    if args.bodily_event_loss_weight < 0.0:
        parser.error("--bodily-event-loss-weight must be nonnegative.")
    if args.bodily_consumption_event_loss_weight < 0.0:
        parser.error(
            "--bodily-consumption-event-loss-weight must be nonnegative."
        )
    if args.bodily_bound_consumption_event_loss_weight < 0.0:
        parser.error(
            "--bodily-bound-consumption-event-loss-weight must be nonnegative."
        )
    if args.load_checkpoint is not None:
        if (
            args.semantic_choice_information_upper_bound_contexts > 0
            or args.persistent_mapping_rent_lives > 0
            or args.persistent_choice_mechanics_lives > 0
        ):
            parser.error(
                "--load-checkpoint and simulator-only audits are mutually "
                "exclusive."
            )
        if (
            args.observation_branching_audit_contexts > 0
            and args.evaluate_loaded_checkpoint
        ):
            parser.error(
                "Choose either --evaluate-loaded-checkpoint or the "
                "observation-branching audit."
            )
        if (
            args.observation_branching_audit_contexts <= 0
            and args.cross_round_reuse_lives <= 0
            and args.drift_forecast_audit_contexts <= 0
            and args.protocol_return_origin_audit_contexts <= 0
            and args.terminal_consume_calibration_contexts <= 0
            and args.real_explicit_event_transfer_contexts <= 0
            and args.binding_consequence_geometry_contexts <= 0
            and args.bound_event_gradient_alignment_segments <= 0
            and not args.evaluate_loaded_checkpoint
        ):
            parser.error(
                "--load-checkpoint requires --evaluate-loaded-checkpoint, a "
                "positive --observation-branching-audit-contexts, a positive "
                "--cross-round-reuse-lives, or a positive "
                "--drift-forecast-audit-contexts or "
                "--protocol-return-origin-audit-contexts or "
                "--terminal-consume-calibration-contexts or "
                "--real-explicit-event-transfer-contexts or "
                "--binding-consequence-geometry-contexts or "
                "--bound-event-gradient-alignment-segments."
            )
        if (
            args.evaluate_loaded_checkpoint
            and args.semantic_choice_eval_episodes <= 0
        ):
            parser.error(
                "--evaluate-loaded-checkpoint requires a positive "
                "--semantic-choice-eval-episodes."
            )
    elif args.cross_round_reuse_lives > 0:
        parser.error("--cross-round-reuse-lives requires --load-checkpoint.")
    elif args.drift_forecast_audit_contexts > 0:
        parser.error(
            "--drift-forecast-audit-contexts requires --load-checkpoint."
        )
    elif args.protocol_return_origin_audit_contexts > 0:
        parser.error(
            "--protocol-return-origin-audit-contexts requires "
            "--load-checkpoint."
        )
    elif args.terminal_consume_calibration_contexts > 0:
        parser.error(
            "--terminal-consume-calibration-contexts requires "
            "--load-checkpoint."
        )
    elif args.real_explicit_event_transfer_contexts > 0:
        parser.error(
            "--real-explicit-event-transfer-contexts requires "
            "--load-checkpoint."
        )
    elif args.observation_branching_audit_contexts > 0:
        parser.error(
            "--observation-branching-audit-contexts requires "
            "--load-checkpoint."
        )
    elif args.evaluate_loaded_checkpoint:
        parser.error("--evaluate-loaded-checkpoint requires --load-checkpoint.")
    if (
        sum(
            int(value > 0)
            for value in (
                args.semantic_choice_information_upper_bound_contexts,
                args.persistent_mapping_rent_lives,
                args.persistent_choice_mechanics_lives,
            )
        )
        > 1
    ):
        parser.error(
            "Choose only one simulator-only information-rent audit."
        )
    return args


if __name__ == "__main__":
    main()
