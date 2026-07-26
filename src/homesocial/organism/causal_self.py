"""Structured causal self-state system identification and promotion gates."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from random import Random

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np

from homesocial.creole.vocab import PAD_TOKEN, TOKEN_TO_ID, VOCAB
from homesocial.env import Action, DELTAS, DIRECTION_ORDER
from homesocial.island.report import (
    HELP_SURFACES,
    NEED_TO_REPORT_WORD,
    REPORT_NEEDS,
    heard_need,
)
from homesocial.island.world import ObsPacket, SURFACES
from homesocial.organism.model import OrganismModel
from homesocial.organism.report_audit import (
    FIDELITY_WARMUP,
    _ridge_decoder_metrics,
    make_report_world,
)
from homesocial.organism.self_belief import _balanced_accuracy, _sample_motor_action
from homesocial.organism.train import (
    ACTIONS,
    OrganismConfig,
    decode_object_option,
    execute_agent_action,
    save_checkpoint,
)

PAD_ID = TOKEN_TO_ID[PAD_TOKEN]


def _surface_at(packet: ObsPacket, dx: int, dy: int) -> int | None:
    for visible_dx, visible_dy, surface_index in packet.visible:
        if visible_dx == dx and visible_dy == dy:
            return surface_index
    return None


def _public_transition_features(
    config: OrganismConfig,
    before: ObsPacket,
    action_index: int,
    after: ObsPacket,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Features derivable from observation, own action, and public time only."""

    duration = after.step_count - before.step_count
    if duration <= 0:
        raise ValueError("A causal transition must advance primitive time.")
    surface_count = len(SURFACES)
    uptake = np.zeros((surface_count,), dtype=np.float32)
    shock = np.zeros((surface_count,), dtype=np.float32)
    terminal_surface: int | None = None
    routed_moves = 0
    terminal_kind: str | None = None
    if action_index < len(ACTIONS):
        action = ACTIONS[action_index]
        routed_moves = int(action == Action.MOVE_FORWARD)
        if action in {Action.CONSUME, Action.REST}:
            terminal_kind = action.value
            terminal_surface = _surface_at(before, 0, 0)
            if terminal_surface is None:
                direction = DIRECTION_ORDER[before.direction_index]
                dx, dy = DELTAS[direction]
                terminal_surface = _surface_at(before, dx, dy)
    else:
        decoded = decode_object_option(
            action_index,
            consume_options=config.consume_options,
            inspect_options=config.inspect_options,
            visible_slots=config.island.max_visible_slots,
        )
        if decoded is not None:
            terminal_kind, slot = decoded
            if slot < len(before.visible):
                dx, dy, terminal_surface = before.visible[slot]
                distance = abs(dx) + abs(dy)
                routed_moves = 1 if distance == 0 else max(0, distance - 1)

    spoiled = any(
        (before.step_count + offset) % config.report.help_period == 0
        for offset in range(1, duration)
    )
    if (
        terminal_surface is not None
        and terminal_kind in {"consume", Action.REST.value}
        and not spoiled
    ):
        uptake[terminal_surface] = 1.0

    before_surfaces = {surface for _, _, surface in before.visible}
    for _, _, surface in after.visible:
        if surface not in before_surfaces:
            shock[surface] = 1.0
    return float(duration), float(routed_moves), uptake, shock


def _listener_target(
    config: OrganismConfig,
    before: ObsPacket,
    after: ObsPacket,
) -> int | None:
    """Visible surface/no-help label at an unambiguous one-tick grant boundary."""

    if after.step_count - before.step_count != 1:
        return None
    if after.step_count % config.report.help_period != 0:
        return None
    surface = _surface_at(after, 0, 0)
    return len(SURFACES) if surface is None else surface


def _causal_loss(
    model: OrganismModel,
    current: mx.array,
    targets: mx.array,
    durations: mx.array,
    move_counts: mx.array,
    uptake: mx.array,
    shock: mx.array,
    listener_tokens: mx.array,
    listener_targets: mx.array,
) -> mx.array:
    predicted = model.causal_self_transition(
        current, durations, move_counts, uptake, shock
    )
    dynamics = ((predicted - targets) ** 2).mean()
    if listener_tokens.shape[0] == 0:
        return dynamics
    logits = model.causal_listener_logits[listener_tokens]
    log_probabilities = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    chosen = mx.take_along_axis(
        log_probabilities, listener_targets[:, None], axis=-1
    )[:, 0]
    return dynamics - chosen.mean()


def _snapshot_base(model: OrganismModel) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(value).copy()
        for name, value in tree_flatten(model.parameters())
        if not name.startswith("causal_")
    }


def _base_unchanged(model: OrganismModel, before: dict[str, np.ndarray]) -> bool:
    after = dict(tree_flatten(model.parameters()))
    return all(
        name in after and np.array_equal(value, np.asarray(after[name]))
        for name, value in before.items()
    )


def train_structured_causal_self_model(
    model: OrganismModel,
    config: OrganismConfig,
    *,
    steps: int = 80_000,
    learning_rate: float = 3e-3,
    seed_base: int = 6_100_000,
    checkpoint: str | None = None,
    stats_csv: str | None = None,
    log_every_lives: int = 200,
) -> tuple[OrganismConfig, dict[str, float]]:
    """Identify causal dynamics and listener effects from public experience."""

    if steps <= 0:
        raise ValueError("Structured causal development steps must be positive.")
    if not model.has_structured_causal_self_model:
        raise ValueError("Enable the structured causal self-model first.")
    base_before = _snapshot_base(model)
    optimizer = optim.Adam(learning_rate=learning_rate)
    loss_and_grad = nn.value_and_grad(model, _causal_loss)
    token_rng = Random(seed_base + 37_000_003)
    ticks = 0
    lives = 0
    updates = 0
    listener_samples = 0
    losses: list[float] = []
    rows: list[tuple[int, int, int, int, float]] = []

    while ticks < steps:
        seed = seed_base + lives
        world = make_report_world(config, seed=seed, listener_mode="grounded")
        packet = world.reset(seed)
        motor_hidden: mx.array | None = None
        motor_rng = Random(seed + 39_000_003)
        current_values: list[tuple[float, float, float]] = []
        next_values: list[tuple[float, float, float]] = []
        durations: list[float] = []
        move_counts: list[float] = []
        uptakes: list[np.ndarray] = []
        shocks: list[np.ndarray] = []
        listener_tokens: list[int] = []
        listener_targets: list[int] = []
        life_ticks = 0
        while ticks < steps:
            action, motor_hidden = _sample_motor_action(
                model, packet, motor_hidden, motor_rng
            )
            token = token_rng.randrange(len(VOCAB))
            world.hear((token, PAD_ID))
            remaining = steps - ticks
            world.grid.max_steps = min(
                world.grid.max_steps, world.grid.step_count + remaining
            )
            before = packet
            after, _, terminated, truncated, info = execute_agent_action(
                world,
                packet,
                action,
                consume_options=config.consume_options,
                inspect_options=config.inspect_options,
            )
            duration, moves, uptake, shock = _public_transition_features(
                config, before, action, after
            )
            current_values.append(tuple(float(v) for v in before.needs[:3]))
            next_values.append(tuple(float(v) for v in after.needs[:3]))
            durations.append(duration)
            move_counts.append(moves)
            uptakes.append(uptake)
            shocks.append(shock)
            listener_target = _listener_target(config, before, after)
            if listener_target is not None:
                listener_tokens.append(token)
                listener_targets.append(listener_target)
            ticks += int(info["duration"])
            life_ticks += int(info["duration"])
            packet = after
            if terminated or truncated or ticks >= steps:
                break

        loss, grads = loss_and_grad(
            model,
            mx.array(np.asarray(current_values, dtype=np.float32)),
            mx.array(np.asarray(next_values, dtype=np.float32)),
            mx.array(np.asarray(durations, dtype=np.float32)),
            mx.array(np.asarray(move_counts, dtype=np.float32)),
            mx.array(np.stack(uptakes)),
            mx.array(np.stack(shocks)),
            mx.array(np.asarray(listener_tokens, dtype=np.int32)),
            mx.array(np.asarray(listener_targets, dtype=np.int32)),
        )
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        loss_value = float(loss)
        losses.append(loss_value)
        updates += 1
        lives += 1
        listener_samples += len(listener_tokens)
        rows.append((updates, ticks, life_ticks, len(listener_tokens), loss_value))
        if log_every_lives > 0 and lives % log_every_lives == 0:
            print(
                f"causal-self ticks {ticks}: lives {lives}, listener samples "
                f"{listener_samples}, recent loss "
                f"{np.mean(losses[-log_every_lives:]):.6f}"
            )

    unchanged = _base_unchanged(model, base_before)
    if not unchanged:
        raise RuntimeError("Causal development changed a pre-existing parameter.")
    output_config = replace(
        config,
        structured_causal_self_model=True,
        structured_causal_development_steps=steps,
        checkpoint=checkpoint,
        stats_csv=stats_csv,
    )
    if checkpoint is not None:
        save_checkpoint(model, output_config, checkpoint)
    if stats_csv is not None:
        target = Path(stats_csv)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "update,ticks,life_ticks,listener_samples,loss\n"
            + "".join(
                f"{u},{t},{lt},{ls},{loss:.9f}\n"
                for u, t, lt, ls, loss in rows
            )
        )
    window = max(1, len(losses) // 4)
    return output_config, {
        "development_ticks": float(ticks),
        "development_lives": float(lives),
        "updates": float(updates),
        "listener_samples": float(listener_samples),
        "first_quarter_loss": float(np.mean(losses[:window])),
        "last_quarter_loss": float(np.mean(losses[-window:])),
        "base_parameters_unchanged": float(unchanged),
    }


def _apply_causal_transition(
    model: OrganismModel,
    belief: np.ndarray,
    features: tuple[float, float, np.ndarray, np.ndarray],
) -> np.ndarray:
    duration, moves, uptake, shock = features
    predicted = model.causal_self_transition(
        mx.array(belief[None, :], dtype=mx.float32),
        mx.array([duration], dtype=mx.float32),
        mx.array([moves], dtype=mx.float32),
        mx.array(uptake[None, :]),
        mx.array(shock[None, :]),
    )
    mx.eval(predicted)
    return np.asarray(predicted[0], dtype=np.float64)


def _listener_metrics(model: OrganismModel) -> dict[str, float]:
    logits = np.asarray(model.causal_listener_logits, dtype=np.float64)
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    no_help = len(SURFACES)
    need_surface_indices = {
        need: [
            SURFACES.index(surface)
            for (mapped_need, _), surface in HELP_SURFACES.items()
            if mapped_need == need
        ]
        for need in REPORT_NEEDS
    }
    effective_correct = []
    for need in REPORT_NEEDS:
        token = TOKEN_TO_ID[NEED_TO_REPORT_WORD[need]]
        scores = [
            probabilities[token, need_surface_indices[candidate]].sum()
            for candidate in REPORT_NEEDS
        ]
        effective_correct.append(int(np.argmax(scores) == REPORT_NEEDS.index(need)))
    effective_ids = {
        TOKEN_TO_ID[NEED_TO_REPORT_WORD[need]] for need in REPORT_NEEDS
    }
    other = [token for token in range(len(VOCAB)) if token not in effective_ids]
    no_help_correct = np.argmax(probabilities[other], axis=1) == no_help
    return {
        "listener_effective_token_accuracy": float(np.mean(effective_correct)),
        "listener_food_correct": float(effective_correct[0]),
        "listener_water_correct": float(effective_correct[1]),
        "listener_energy_correct": float(effective_correct[2]),
        "listener_no_help_accuracy": float(np.mean(no_help_correct)),
        "listener_no_help_mean_probability": float(
            probabilities[other, no_help].mean()
        ),
    }


def audit_structured_causal_portion_forks(
    model: OrganismModel,
    config: OrganismConfig,
    *,
    lives: int = 200,
    seed_base: int = 6_900_000,
) -> dict[str, float]:
    divergent = 0
    followed = 0
    changed = 0
    report_followed = 0
    report_changed = 0
    for life in range(lives):
        seed = seed_base + life
        worlds = []
        packets = []
        beliefs = []
        for large in (False, True):
            world = make_report_world(
                config,
                seed=seed,
                listener_mode="grounded",
                shock_probability=0.0,
                unified_uptake=True,
            )
            packet = world.reset(seed)
            world.force_next_help_portion(large=large)
            worlds.append(world)
            packets.append(packet)
            beliefs.append(np.asarray(packet.vector()[:3], dtype=np.float64))
        requested = worlds[0].lowest_need()
        word = (TOKEN_TO_ID[NEED_TO_REPORT_WORD[requested]], PAD_ID)
        actions = [ACTIONS.index(Action.WAIT)] * config.report.help_period + [
            ACTIONS.index(Action.CONSUME)
        ]
        for action in actions:
            for branch in range(2):
                before = packets[branch]
                worlds[branch].hear(word)
                after, _, terminated, truncated, _ = execute_agent_action(
                    worlds[branch],
                    before,
                    action,
                    consume_options=config.consume_options,
                    inspect_options=config.inspect_options,
                )
                assert not terminated and not truncated
                features = _public_transition_features(
                    config, before, action, after
                )
                beliefs[branch] = _apply_causal_transition(
                    model, beliefs[branch], features
                )
                packets[branch] = after
        true_lowest = [world.lowest_need() for world in worlds]
        if true_lowest[0] == true_lowest[1]:
            continue
        divergent += 1
        predicted = [REPORT_NEEDS[int(np.argmin(belief))] for belief in beliefs]
        changed += int(predicted[0] != predicted[1])
        followed += int(predicted == true_lowest)
        tokens = [
            causal_social_token(
                model,
                beliefs[branch],
                step_count=packets[branch].step_count,
                help_period=config.report.help_period,
            )
            for branch in range(2)
        ]
        reports = [heard_need((token, PAD_ID)) for token in tokens]
        report_changed += int(tokens[0] != tokens[1])
        report_followed += int(reports == true_lowest)
    return {
        "forked_lives": float(lives),
        "divergent_lives": float(divergent),
        "fork_prediction_changed": changed / max(1, divergent),
        "fork_following": followed / max(1, divergent),
        "report_changed_rate": report_changed / max(1, divergent),
        "report_followed_body_rate": report_followed / max(1, divergent),
    }


def audit_structured_causal_self_model(
    model: OrganismModel,
    config: OrganismConfig,
    *,
    lives: int = 200,
    seed_base: int = 6_500_000,
) -> dict[str, float]:
    """Frozen promotion gate before the learned social planner may run."""

    predictions: list[np.ndarray] = []
    truths: list[int] = []
    absolute_error = 0.0
    samples = 0
    survived = 0
    steps = 0
    for life in range(lives):
        seed = seed_base + life
        world = make_report_world(config, seed=seed, listener_mode="grounded")
        packet = world.reset(seed)
        belief = np.asarray(packet.vector()[:3], dtype=np.float64)
        motor_hidden: mx.array | None = None
        motor_rng = Random(seed + 49_000_003)
        while True:
            if packet.step_count >= FIDELITY_WARMUP:
                predictions.append(belief.copy())
                truths.append(REPORT_NEEDS.index(world.lowest_need()))
                true_values = np.asarray(packet.needs[:3], dtype=np.float64)
                absolute_error += float(np.abs(belief - true_values).mean())
                samples += 1
            action, motor_hidden = _sample_motor_action(
                model, packet, motor_hidden, motor_rng
            )
            # Privileged speech regulates the audit world only; it never enters
            # the causal self-model or its measurements.
            true_need = world.lowest_need()
            world.hear((TOKEN_TO_ID[NEED_TO_REPORT_WORD[true_need]], PAD_ID))
            before = packet
            packet, _, terminated, truncated, info = execute_agent_action(
                world,
                packet,
                action,
                consume_options=config.consume_options,
                inspect_options=config.inspect_options,
            )
            belief = _apply_causal_transition(
                model,
                belief,
                _public_transition_features(config, before, action, packet),
            )
            steps += int(info["duration"])
            if terminated or truncated:
                survived += int(not terminated)
                break
    values = np.stack(predictions)
    predicted = np.argmin(values, axis=1)
    truth = np.asarray(truths, dtype=np.int64)
    balanced = _balanced_accuracy(predicted, truth)
    zero_balanced = _balanced_accuracy(np.zeros_like(truth), truth)
    shuffled_values = values.copy()
    np.random.default_rng(seed_base + 53_000_003).shuffle(
        shuffled_values, axis=0
    )
    shuffle_balanced = _balanced_accuracy(
        np.argmin(shuffled_values, axis=1), truth
    )
    fork = audit_structured_causal_portion_forks(
        model, config, lives=lives, seed_base=seed_base + 200_000
    )
    listener = _listener_metrics(model)
    result = {
        "lives": float(lives),
        "samples": float(samples),
        "survival": survived / lives,
        "mean_life_steps": steps / lives,
        "accuracy": float(np.mean(predicted == truth)),
        "balanced_accuracy": balanced,
        "mean_absolute_need_error": absolute_error / max(1, samples),
        "zero_balanced_accuracy": zero_balanced,
        "shuffle_balanced_accuracy": shuffle_balanced,
        "zero_accuracy_drop": balanced - zero_balanced,
        "shuffle_accuracy_drop": balanced - shuffle_balanced,
        **listener,
        **fork,
    }
    result["gate_passed"] = float(
        balanced >= 0.90
        and result["mean_absolute_need_error"] <= 0.03
        and result["zero_accuracy_drop"] >= 0.30
        and result["shuffle_accuracy_drop"] >= 0.30
        and fork["fork_following"] >= 0.80
        and listener["listener_food_correct"] == 1.0
        and listener["listener_water_correct"] == 1.0
        and listener["listener_energy_correct"] == 1.0
        and listener["listener_no_help_accuracy"] >= 0.95
    )
    return result


def causal_social_token(
    model: OrganismModel,
    belief: np.ndarray,
    *,
    step_count: int,
    help_period: int,
) -> int:
    """Choose among the full vocabulary by learned future bodily consequence."""

    logits = model.causal_listener_logits
    listener = mx.softmax(logits, axis=-1)
    drift, _, uptake, _ = model.causal_self_parameters()
    ticks_to_help = help_period - (step_count % help_period)
    future_before_help = mx.clip(
        mx.array(belief, dtype=mx.float32)
        + float(ticks_to_help) * drift,
        0.0,
        1.0,
    )
    expected_uptake = listener[:, : len(SURFACES)] @ uptake
    futures = mx.clip(future_before_help[None, :] + expected_uptake, 0.0, 1.0)
    scores = mx.min(futures, axis=-1)
    mx.eval(scores)
    return int(np.argmax(np.asarray(scores)))


def evaluate_causal_social_planner(
    model: OrganismModel,
    config: OrganismConfig,
    *,
    lives: int,
    seed_base: int,
    listener_mode: str = "grounded",
    belief_intervention: str = "none",
    fixed_word: str | None = None,
    mute_organism: bool = False,
    report_overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    """Deploy the learned self-belief -> listener consequence planner."""

    if belief_intervention not in {"none", "zero", "shuffle", "freeze"}:
        raise ValueError(f"Unknown causal belief intervention: {belief_intervention}.")
    survived = 0
    viability_sum = 0.0
    steps_sum = 0
    said_need = 0
    truthful = 0
    grants = 0
    truthful_grants = 0
    word_counts = {need: 0 for need in REPORT_NEEDS}
    bucket_said = [0, 0, 0, 0]
    bucket_truthful = [0, 0, 0, 0]
    for life in range(lives):
        seed = seed_base + life
        overrides = dict(report_overrides or {})
        overrides["listener_mode"] = listener_mode
        world = make_report_world(config, seed=seed, **overrides)
        packet = world.reset(seed)
        belief = np.asarray(packet.vector()[:3], dtype=np.float64)
        frozen_belief = belief.copy()
        belief_history: list[np.ndarray] = []
        motor_hidden: mx.array | None = None
        motor_rng = Random(seed + 59_000_003)
        intervention_rng = Random(seed + 61_000_003)
        life_viability = 0.0
        while True:
            mouth_belief = belief
            if belief_intervention == "zero":
                mouth_belief = np.zeros_like(belief)
            elif belief_intervention == "freeze":
                mouth_belief = frozen_belief
            elif belief_intervention == "shuffle" and belief_history:
                mouth_belief = belief_history[
                    intervention_rng.randrange(len(belief_history))
                ]
            if fixed_word is not None:
                token = TOKEN_TO_ID[NEED_TO_REPORT_WORD[fixed_word]]
            else:
                token = causal_social_token(
                    model,
                    mouth_belief,
                    step_count=packet.step_count,
                    help_period=config.report.help_period,
                )
            said = (token, PAD_ID)
            true_before = world.lowest_need()
            tick = packet.step_count
            word = heard_need(said)
            if word is not None and tick >= FIDELITY_WARMUP:
                said_need += 1
                word_counts[word] += 1
                correct = int(word == true_before)
                truthful += correct
                bucket = min(3, tick // 100)
                bucket_said[bucket] += 1
                bucket_truthful[bucket] += correct
            action, motor_hidden = _sample_motor_action(
                model, packet, motor_hidden, motor_rng
            )
            world.hear(None if mute_organism else said)
            before = packet
            packet, _, terminated, truncated, info = execute_agent_action(
                world,
                packet,
                action,
                consume_options=config.consume_options,
                inspect_options=config.inspect_options,
            )
            belief_history.append(belief.copy())
            belief = _apply_causal_transition(
                model,
                belief,
                _public_transition_features(config, before, action, packet),
            )
            duration = int(info["duration"])
            steps_sum += duration
            life_viability += float(info["mean_viability_sum"])
            if info.get("granted_need") is not None:
                grants += 1
                truthful_grants += int(
                    info.get("granted_need") == info.get("lowest_need")
                )
            if terminated or truncated:
                survived += int(not terminated)
                break
        viability_sum += life_viability / config.report.life_steps
    return {
        "lives": lives,
        "listener_mode": listener_mode,
        "belief_intervention": belief_intervention,
        "survival": survived / lives,
        "mean_viability": viability_sum / lives,
        "mean_life_steps": steps_sum / lives,
        "need_word_ticks": said_need,
        "report_fidelity": truthful / max(1, said_need),
        "speech_rate": said_need / max(1, steps_sum),
        "grants": grants,
        "grant_fidelity": truthful_grants / max(1, grants),
        "word_distribution": {
            need: word_counts[need] / max(1, sum(word_counts.values()))
            for need in REPORT_NEEDS
        },
        "fidelity_by_bucket": [
            bucket_truthful[index] / max(1, bucket_said[index])
            for index in range(4)
        ],
        "bucket_counts": bucket_said,
    }


def audit_causal_planner_observation_decoder(
    model: OrganismModel,
    config: OrganismConfig,
    *,
    lives: int,
    seed_base: int,
) -> dict[str, float]:
    split = max(1, int(0.7 * lives))
    train_x: list[np.ndarray] = []
    train_y: list[int] = []
    test_x: list[np.ndarray] = []
    test_y: list[int] = []
    for life in range(lives):
        seed = seed_base + life
        world = make_report_world(config, seed=seed, listener_mode="grounded")
        packet = world.reset(seed)
        belief = np.asarray(packet.vector()[:3], dtype=np.float64)
        motor_hidden: mx.array | None = None
        motor_rng = Random(seed + 67_000_003)
        while True:
            if packet.step_count >= FIDELITY_WARMUP:
                target_x = train_x if life < split else test_x
                target_y = train_y if life < split else test_y
                target_x.append(np.asarray(packet.vector(), dtype=np.float32))
                target_y.append(REPORT_NEEDS.index(world.lowest_need()))
            token = causal_social_token(
                model,
                belief,
                step_count=packet.step_count,
                help_period=config.report.help_period,
            )
            action, motor_hidden = _sample_motor_action(
                model, packet, motor_hidden, motor_rng
            )
            world.hear((token, PAD_ID))
            before = packet
            packet, _, terminated, truncated, _ = execute_agent_action(
                world,
                packet,
                action,
                consume_options=config.consume_options,
                inspect_options=config.inspect_options,
            )
            belief = _apply_causal_transition(
                model,
                belief,
                _public_transition_features(config, before, action, packet),
            )
            if terminated or truncated:
                break
    train_y_array = np.asarray(train_y, dtype=np.int64)
    test_y_array = np.asarray(test_y, dtype=np.int64)
    metrics = _ridge_decoder_metrics(
        np.stack(train_x), train_y_array, np.stack(test_x), test_y_array
    )
    counts = np.bincount(test_y_array, minlength=len(REPORT_NEEDS))
    return {
        "samples": float(len(test_y)),
        "heldout_accuracy": metrics["accuracy"],
        "heldout_balanced_accuracy": metrics["balanced_accuracy"],
        "majority_baseline": float(counts.max() / max(1, counts.sum())),
        "chance": 1.0 / len(REPORT_NEEDS),
    }


def causal_social_report_battery(
    model: OrganismModel,
    config: OrganismConfig,
    *,
    lives: int = 200,
    seed_base: int = 7_100_000,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rows.append(
        {
            "condition": "causal_planner_grounded",
            **evaluate_causal_social_planner(
                model, config, lives=lives, seed_base=seed_base
            ),
        }
    )
    for condition, listener in (
        ("causal_planner_listener_scrambled", "scrambled"),
        ("causal_planner_listener_mute", "mute"),
    ):
        rows.append(
            {
                "condition": condition,
                **evaluate_causal_social_planner(
                    model,
                    config,
                    lives=lives,
                    seed_base=seed_base,
                    listener_mode=listener,
                ),
            }
        )
    rows.append(
        {
            "condition": "causal_planner_organism_mute",
            **evaluate_causal_social_planner(
                model,
                config,
                lives=lives,
                seed_base=seed_base,
                mute_organism=True,
            ),
        }
    )
    for need in REPORT_NEEDS:
        rows.append(
            {
                "condition": f"causal_planner_fixed_word_{need}",
                **evaluate_causal_social_planner(
                    model,
                    config,
                    lives=lives,
                    seed_base=seed_base,
                    fixed_word=need,
                ),
            }
        )
    for intervention in ("zero", "shuffle", "freeze"):
        rows.append(
            {
                "condition": f"causal_planner_intervention_{intervention}",
                **evaluate_causal_social_planner(
                    model,
                    config,
                    lives=lives,
                    seed_base=seed_base,
                    belief_intervention=intervention,
                ),
            }
        )
    rows.append(
        {
            "condition": "causal_planner_heldout_birth_levels",
            **evaluate_causal_social_planner(
                model,
                config,
                lives=lives,
                seed_base=seed_base + 100_000,
                report_overrides={
                    "birth_levels": (0.30, 0.40, 0.50, 0.60, 0.70, 0.90)
                },
            ),
        }
    )
    rows.append(
        {
            "condition": "causal_planner_heldout_portions",
            **evaluate_causal_social_planner(
                model,
                config,
                lives=lives,
                seed_base=seed_base + 200_000,
                report_overrides={"portion_small": 0.25, "portion_large": 0.55},
            ),
        }
    )
    rows.append(
        {
            "condition": "causal_planner_observation_decoder",
            **audit_causal_planner_observation_decoder(
                model,
                config,
                lives=lives,
                seed_base=seed_base + 300_000,
            ),
        }
    )
    fork = audit_structured_causal_portion_forks(
        model, config, lives=lives, seed_base=seed_base + 400_000
    )
    rows.append({"condition": "causal_planner_counterfactual_body", **fork})
    grounded = rows[0]
    scrambled = rows[1]
    gate_passed = (
        float(grounded["report_fidelity"]) >= 0.60
        and float(grounded["survival"]) >= 0.80
        and float(grounded["survival"]) - float(scrambled["survival"]) >= 0.15
        and float(fork["report_followed_body_rate"]) >= 0.60
    )
    rows.append(
        {
            "condition": "causal_social_report_gate",
            "gate_passed": float(gate_passed),
            "grounded_survival": grounded["survival"],
            "grounded_report_fidelity": grounded["report_fidelity"],
            "scrambled_survival": scrambled["survival"],
            "grounded_minus_scrambled_survival": float(grounded["survival"])
            - float(scrambled["survival"]),
            "counterfactual_following": fork["report_followed_body_rate"],
        }
    )
    return rows
