"""Development and falsification gates for an explicit hidden-body belief.

The learned module belongs to ``OrganismModel`` but is staged separately from
speech. Its developmental loss sees bodily teaching targets while its inputs
are forcibly restricted to the one birth reading and subsequent public
action/perception history. Existing organism parameters are held invariant.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from random import Random

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np

from homesocial.creole.vocab import PAD_TOKEN, TOKEN_TO_ID
from homesocial.env import Action
from homesocial.island.report import NEED_TO_REPORT_WORD, REPORT_NEEDS
from homesocial.organism.model import OrganismModel
from homesocial.organism.report_audit import FIDELITY_WARMUP, make_report_world
from homesocial.organism.train import (
    ACTIONS,
    OrganismConfig,
    available_action_mask,
    execute_agent_action,
    save_checkpoint,
)

PAD_ID = TOKEN_TO_ID[PAD_TOKEN]


def _sample_motor_action(
    model: OrganismModel,
    packet,
    hidden: mx.array | None,
    rng: Random,
) -> tuple[int, mx.array]:
    vector = mx.array(packet.vector()[None, None, :])
    tokens = mx.array(np.asarray(packet.tokens, dtype=np.int32)[None, None, :])
    states, next_hidden = model.core_states(vector, tokens, hidden)
    logits = model.policy_logits(states, vector)
    mask = available_action_mask(
        packet,
        model.action_size,
        visible_slots=model.visible_slots or None,
    )
    logits = mx.where(mx.array(mask)[None, None, :], logits, -1e9)
    mx.eval(logits, next_hidden)
    probabilities = np.asarray(mx.softmax(logits[0, 0], axis=-1), dtype=np.float64)
    probabilities /= probabilities.sum()
    action = int(rng.choices(range(model.action_size), weights=probabilities)[0])
    return action, next_hidden


def _belief_loss(
    model: OrganismModel,
    vectors: mx.array,
    durations: mx.array,
    targets: mx.array,
    rank_weight: mx.array,
    urgency_rank_weight: mx.array,
) -> mx.array:
    beliefs, urgency, _ = model.self_belief_outputs(vectors, durations, None)
    mse = ((beliefs - targets) ** 2).mean()
    rank_terms: list[mx.array] = []
    for left, right in ((0, 1), (0, 2), (1, 2)):
        true_difference = targets[..., left] - targets[..., right]
        predicted_difference = beliefs[..., left] - beliefs[..., right]
        direction = mx.sign(true_difference)
        valid = (mx.abs(true_difference) > 1e-6).astype(beliefs.dtype)
        pair = mx.logaddexp(
            mx.array(0.0), -direction * predicted_difference
        )
        rank_terms.append((pair * valid).sum() / mx.maximum(valid.sum(), 1.0))
    rank_loss = mx.stack(rank_terms).mean()
    urgency_rank_loss = mx.array(0.0)
    if urgency is not None:
        urgency_terms: list[mx.array] = []
        for left, right in ((0, 1), (0, 2), (1, 2)):
            true_difference = targets[..., left] - targets[..., right]
            urgency_difference = urgency[..., left] - urgency[..., right]
            direction = mx.sign(true_difference)
            valid = (mx.abs(true_difference) > 1e-6).astype(urgency.dtype)
            pair = mx.logaddexp(
                mx.array(0.0), -direction * urgency_difference
            )
            urgency_terms.append(
                (pair * valid).sum() / mx.maximum(valid.sum(), 1.0)
            )
        urgency_rank_loss = mx.stack(urgency_terms).mean()
    return (
        mse
        + rank_weight * rank_loss
        + urgency_rank_weight * urgency_rank_loss
    )


def _snapshot_base_parameters(model: OrganismModel) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(value).copy()
        for name, value in tree_flatten(model.parameters())
        if not name.startswith("self_belief_")
    }


def _base_parameters_unchanged(
    model: OrganismModel, before: dict[str, np.ndarray]
) -> bool:
    after = dict(tree_flatten(model.parameters()))
    return all(
        name in after and np.array_equal(value, np.asarray(after[name]))
        for name, value in before.items()
    )


def train_explicit_self_belief(
    model: OrganismModel,
    config: OrganismConfig,
    *,
    steps: int = 80_000,
    learning_rate: float = 3e-4,
    rank_weight: float = 0.0,
    urgency_rank_weight: float = 0.0,
    seed_base: int = 5_100_000,
    checkpoint: str | None = None,
    stats_csv: str | None = None,
    log_every_lives: int = 100,
) -> tuple[OrganismConfig, dict[str, float]]:
    """Train only the appended belief module on need-independent experience."""

    if steps <= 0:
        raise ValueError("Self-belief development steps must be positive.")
    if rank_weight < 0.0:
        raise ValueError("Self-belief rank weight must be nonnegative.")
    if urgency_rank_weight < 0.0:
        raise ValueError("Self-belief urgency rank weight must be nonnegative.")
    if urgency_rank_weight > 0.0 and not (
        model.has_self_belief_relational_urgency
    ):
        raise ValueError("Urgency rank loss requires a relational urgency head.")
    if not model.has_explicit_self_belief:
        raise ValueError("Enable explicit self-belief before development.")
    base_before = _snapshot_base_parameters(model)
    optimizer = optim.Adam(learning_rate=learning_rate)
    loss_and_grad = nn.value_and_grad(model, _belief_loss)
    provision_rng = Random(seed_base + 17_000_003)

    ticks = 0
    lives = 0
    updates = 0
    losses: list[float] = []
    rows: list[tuple[int, int, int, float]] = []
    while ticks < steps:
        seed = seed_base + lives
        world = make_report_world(config, seed=seed, listener_mode="grounded")
        packet = world.reset(seed)
        motor_hidden: mx.array | None = None
        motor_rng = Random(seed + 19_000_003)
        duration_since_previous = 0
        vectors: list[np.ndarray] = []
        durations: list[float] = []
        targets: list[tuple[float, float, float]] = []
        life_ticks = 0
        while ticks < steps:
            # Target access is confined to this developmental teaching buffer;
            # ``self_belief_states`` forcibly masks it from its own input.
            vectors.append(packet.vector())
            durations.append(float(duration_since_previous))
            targets.append(tuple(float(value) for value in packet.needs[:3]))
            action, motor_hidden = _sample_motor_action(
                model, packet, motor_hidden, motor_rng
            )
            random_need = REPORT_NEEDS[provision_rng.randrange(len(REPORT_NEEDS))]
            world.hear(
                (TOKEN_TO_ID[NEED_TO_REPORT_WORD[random_need]], PAD_ID)
            )
            remaining = steps - ticks
            world.grid.max_steps = min(
                world.grid.max_steps, world.grid.step_count + remaining
            )
            next_packet, _, terminated, truncated, info = execute_agent_action(
                world,
                packet,
                action,
                consume_options=config.consume_options,
                inspect_options=config.inspect_options,
            )
            duration_since_previous = int(info["duration"])
            ticks += duration_since_previous
            life_ticks += duration_since_previous
            packet = next_packet
            if terminated or truncated or ticks >= steps:
                break

        loss, grads = loss_and_grad(
            model,
            mx.array(np.stack(vectors)[None, ...]),
            mx.array(np.asarray(durations, dtype=np.float32)[None, ...]),
            mx.array(np.asarray(targets, dtype=np.float32)[None, ...]),
            mx.array(rank_weight),
            mx.array(urgency_rank_weight),
        )
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        loss_value = float(loss)
        losses.append(loss_value)
        updates += 1
        lives += 1
        rows.append((updates, ticks, life_ticks, loss_value))
        if log_every_lives > 0 and lives % log_every_lives == 0:
            print(
                f"self-belief ticks {ticks}: lives {lives}, "
                f"recent loss {np.mean(losses[-log_every_lives:]):.6f}"
            )

    base_unchanged = _base_parameters_unchanged(model, base_before)
    if not base_unchanged:
        raise RuntimeError("Self-belief development changed a pre-existing parameter.")
    output_config = replace(
        config,
        explicit_self_belief=True,
        self_belief_hidden_size=model.self_belief_hidden_size,
        self_belief_development_steps=steps,
        self_belief_rank_weight=rank_weight,
        self_belief_relational_urgency=(
            model.has_self_belief_relational_urgency
        ),
        checkpoint=checkpoint,
        stats_csv=stats_csv,
    )
    if checkpoint is not None:
        save_checkpoint(model, output_config, checkpoint)
    if stats_csv is not None:
        target = Path(stats_csv)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "update,ticks,life_ticks,loss\n"
            + "".join(
                f"{update},{tick},{life_ticks},{loss:.9f}\n"
                for update, tick, life_ticks, loss in rows
            )
        )
    window = max(1, len(losses) // 4)
    return output_config, {
        "development_ticks": float(ticks),
        "development_lives": float(lives),
        "updates": float(updates),
        "first_quarter_loss": float(np.mean(losses[:window])),
        "last_quarter_loss": float(np.mean(losses[-window:])),
        "base_parameters_unchanged": float(base_unchanged),
        "rank_weight": float(rank_weight),
        "urgency_rank_weight": float(urgency_rank_weight),
    }


def _balanced_accuracy(predicted: np.ndarray, truth: np.ndarray) -> float:
    recalls = [
        float(np.mean(predicted[truth == label] == label))
        for label in range(len(REPORT_NEEDS))
        if np.any(truth == label)
    ]
    return float(np.mean(recalls))


def _diagnostic_ridge_predictions(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
) -> np.ndarray:
    """Disposable frozen-feature classifier; never an organism component."""

    mean = train_x.mean(axis=0, keepdims=True)
    scale = train_x.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    train = (train_x - mean) / scale
    test = (test_x - mean) / scale
    train = np.concatenate(
        [train, np.ones((len(train), 1), dtype=np.float32)], axis=1
    )
    test = np.concatenate(
        [test, np.ones((len(test), 1), dtype=np.float32)], axis=1
    )
    targets = np.eye(len(REPORT_NEEDS), dtype=np.float32)[train_y]
    penalty = np.eye(train.shape[1], dtype=np.float32) * 1e-3
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(train.T @ train + penalty, train.T @ targets)
    return np.argmax(test @ weights, axis=1)


def _pairwise_change_ordering(
    predictions: np.ndarray,
    truth: np.ndarray,
    ticks: np.ndarray,
) -> tuple[int, int]:
    agreements = 0
    comparisons = 0
    tick_to_index = {int(tick): index for index, tick in enumerate(ticks)}
    for start, tick in enumerate(ticks):
        stop = tick_to_index.get(int(tick) + 10)
        if stop is None:
            continue
        predicted_change = predictions[stop] - predictions[start]
        true_change = truth[stop] - truth[start]
        for left, right in ((0, 1), (0, 2), (1, 2)):
            predicted_sign = np.sign(
                predicted_change[left] - predicted_change[right]
            )
            true_sign = np.sign(true_change[left] - true_change[right])
            agreements += int(predicted_sign == true_sign)
            comparisons += 1
    return agreements, comparisons


def audit_learned_self_belief_portion_forks(
    model: OrganismModel,
    config: OrganismConfig,
    *,
    lives: int = 200,
    seed_base: int = 5_900_000,
) -> dict[str, float]:
    """Does the learned belief follow a visible portion's lived consequence?"""

    divergent = 0
    followed = 0
    prediction_changed = 0
    for life in range(lives):
        seed = seed_base + life
        worlds = []
        packets = []
        belief_hidden: list[mx.array | None] = [None, None]
        predictions: list[np.ndarray | None] = [None, None]
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
        requested = worlds[0].lowest_need()
        word = (TOKEN_TO_ID[NEED_TO_REPORT_WORD[requested]], PAD_ID)
        wait_index = ACTIONS.index(Action.WAIT)
        for decision in range(config.report.help_period + 1):
            action = (
                ACTIONS.index(Action.CONSUME)
                if decision == config.report.help_period
                else wait_index
            )
            for branch in range(2):
                vector = mx.array(packets[branch].vector()[None, None, :])
                duration = mx.array(
                    [[0.0 if decision == 0 else 1.0]], dtype=mx.float32
                )
                belief, urgency, belief_hidden[branch] = model.self_belief_outputs(
                    vector, duration, belief_hidden[branch]
                )
                if urgency is not None:
                    mx.eval(belief, urgency, belief_hidden[branch])
                    predictions[branch] = np.asarray(urgency[0, 0])
                else:
                    mx.eval(belief, belief_hidden[branch])
                    predictions[branch] = np.asarray(belief[0, 0])
                worlds[branch].hear(word)
                packet, _, terminated, truncated, _ = execute_agent_action(
                    worlds[branch],
                    packets[branch],
                    action,
                    consume_options=config.consume_options,
                    inspect_options=config.inspect_options,
                )
                if terminated or truncated:
                    break
                packets[branch] = packet
        # Process the post-uptake observation carrying the organism's own last
        # action; this is when a causal filter is entitled to change belief.
        for branch in range(2):
            vector = mx.array(packets[branch].vector()[None, None, :])
            belief, urgency, belief_hidden[branch] = model.self_belief_outputs(
                vector, mx.array([[1.0]], dtype=mx.float32), belief_hidden[branch]
            )
            if urgency is not None:
                mx.eval(belief, urgency)
                predictions[branch] = np.asarray(urgency[0, 0])
            else:
                mx.eval(belief)
                predictions[branch] = np.asarray(belief[0, 0])
        true_lowest = [world.lowest_need() for world in worlds]
        if true_lowest[0] == true_lowest[1]:
            continue
        divergent += 1
        predicted_lowest = [
            REPORT_NEEDS[int(np.argmin(prediction))]
            for prediction in predictions
            if prediction is not None
        ]
        prediction_changed += int(predicted_lowest[0] != predicted_lowest[1])
        followed += int(predicted_lowest == true_lowest)
    return {
        "forked_lives": float(lives),
        "divergent_lives": float(divergent),
        "fork_prediction_changed": prediction_changed / max(1, divergent),
        "fork_following": followed / max(1, divergent),
    }


def audit_explicit_self_belief(
    model: OrganismModel,
    config: OrganismConfig,
    *,
    lives: int = 200,
    seed_base: int = 5_500_000,
) -> dict[str, float]:
    """Frozen held-out gate before any self-belief may control a word planner."""

    if not model.has_explicit_self_belief:
        raise ValueError("The checkpoint has no explicit self-belief module.")
    all_predictions: list[np.ndarray] = []
    all_identity_values: list[np.ndarray] = []
    all_truth: list[int] = []
    all_observation: list[int] = []
    all_hidden: list[np.ndarray] = []
    all_true_values: list[np.ndarray] = []
    all_life_indices: list[int] = []
    per_life_predictions: list[np.ndarray] = []
    change_agreements = 0
    change_comparisons = 0
    survived = 0
    steps = 0
    absolute_error = 0.0
    absolute_error_by_need = np.zeros((len(REPORT_NEEDS),), dtype=np.float64)
    samples = 0
    all_ticks: list[int] = []
    for life in range(lives):
        seed = seed_base + life
        world = make_report_world(config, seed=seed, listener_mode="grounded")
        packet = world.reset(seed)
        motor_hidden: mx.array | None = None
        belief_hidden: mx.array | None = None
        motor_rng = Random(seed + 29_000_003)
        duration_since_previous = 0
        life_predictions: list[np.ndarray] = []
        life_truth_values: list[np.ndarray] = []
        life_ticks: list[int] = []
        while True:
            vector_np = packet.vector()
            vector = mx.array(vector_np[None, None, :])
            belief, urgency, belief_hidden = model.self_belief_outputs(
                vector,
                mx.array([[float(duration_since_previous)]], dtype=mx.float32),
                belief_hidden,
            )
            if urgency is not None:
                mx.eval(belief, urgency, belief_hidden)
            else:
                mx.eval(belief, belief_hidden)
            predicted_values = np.asarray(belief[0, 0], dtype=np.float64)
            identity_values = (
                np.asarray(urgency[0, 0], dtype=np.float64)
                if urgency is not None
                else predicted_values
            )
            predicted_need_index = int(np.argmin(identity_values))
            true_values = np.asarray(packet.needs[:3], dtype=np.float64)
            true_need_index = int(np.argmin(true_values))
            predicted_need = REPORT_NEEDS[predicted_need_index]
            if packet.step_count >= FIDELITY_WARMUP:
                all_predictions.append(predicted_values)
                all_identity_values.append(identity_values)
                all_truth.append(true_need_index)
                all_observation.append(int(np.argmin(vector_np[:3])))
                all_ticks.append(packet.step_count)
                assert belief_hidden is not None
                all_hidden.append(np.asarray(belief_hidden[0], dtype=np.float32))
                all_true_values.append(true_values.astype(np.float32))
                all_life_indices.append(life)
                life_predictions.append(predicted_values)
                life_truth_values.append(true_values)
                life_ticks.append(packet.step_count)
                absolute_error += float(np.abs(predicted_values - true_values).mean())
                absolute_error_by_need += np.abs(predicted_values - true_values)
                samples += 1
            action, motor_hidden = _sample_motor_action(
                model, packet, motor_hidden, motor_rng
            )
            world.hear(
                (TOKEN_TO_ID[NEED_TO_REPORT_WORD[predicted_need]], PAD_ID)
            )
            packet, _, terminated, truncated, info = execute_agent_action(
                world,
                packet,
                action,
                consume_options=config.consume_options,
                inspect_options=config.inspect_options,
            )
            duration_since_previous = int(info["duration"])
            steps += duration_since_previous
            if terminated or truncated:
                survived += int(not terminated)
                break
        if life_predictions:
            predictions_array = np.stack(life_predictions)
            truth_values_array = np.stack(life_truth_values)
            ticks_array = np.asarray(life_ticks, dtype=np.int64)
            agreements, comparisons = _pairwise_change_ordering(
                predictions_array, truth_values_array, ticks_array
            )
            change_agreements += agreements
            change_comparisons += comparisons
            per_life_predictions.append(predictions_array)

    predicted_values = np.stack(all_predictions)
    identity_values = np.stack(all_identity_values)
    predicted = np.argmin(identity_values, axis=1)
    truth = np.asarray(all_truth, dtype=np.int64)
    observation = np.asarray(all_observation, dtype=np.int64)
    ticks = np.asarray(all_ticks, dtype=np.int64)
    hidden_states = np.stack(all_hidden)
    true_values = np.stack(all_true_values)
    life_indices = np.asarray(all_life_indices, dtype=np.int64)
    zero = np.zeros_like(truth)
    shuffled_values = identity_values.copy()
    rng = np.random.default_rng(seed_base + 43_000_003)
    rng.shuffle(shuffled_values, axis=0)
    shuffled = np.argmin(shuffled_values, axis=1)
    balanced = _balanced_accuracy(predicted, truth)
    zero_balanced = _balanced_accuracy(zero, truth)
    shuffle_balanced = _balanced_accuracy(shuffled, truth)
    observation_balanced = _balanced_accuracy(observation, truth)
    split_life = max(1, int(0.7 * lives))
    diagnostic_train = life_indices < split_life
    diagnostic_test = ~diagnostic_train
    hidden_ridge = _diagnostic_ridge_predictions(
        hidden_states[diagnostic_train],
        truth[diagnostic_train],
        hidden_states[diagnostic_test],
    )
    affine_train = np.concatenate(
        [
            predicted_values[diagnostic_train],
            np.ones((int(diagnostic_train.sum()), 1), dtype=np.float32),
        ],
        axis=1,
    )
    affine_test = np.concatenate(
        [
            predicted_values[diagnostic_test],
            np.ones((int(diagnostic_test.sum()), 1), dtype=np.float32),
        ],
        axis=1,
    )
    affine_penalty = np.eye(affine_train.shape[1], dtype=np.float32) * 1e-3
    affine_penalty[-1, -1] = 0.0
    affine_weights = np.linalg.solve(
        affine_train.T @ affine_train + affine_penalty,
        affine_train.T @ true_values[diagnostic_train],
    )
    affine = np.argmin(affine_test @ affine_weights, axis=1)
    fork = audit_learned_self_belief_portion_forks(
        model, config, lives=lives, seed_base=seed_base + 200_000
    )
    result = {
        "lives": float(lives),
        "samples": float(samples),
        "survival": survived / lives,
        "mean_life_steps": steps / lives,
        "accuracy": float(np.mean(predicted == truth)),
        "balanced_accuracy": balanced,
        "mean_absolute_need_error": absolute_error / max(1, samples),
        "observation_balanced_accuracy": observation_balanced,
        "diagnostic_affine_belief_balanced_accuracy": _balanced_accuracy(
            affine, truth[diagnostic_test]
        ),
        "diagnostic_hidden_ridge_balanced_accuracy": _balanced_accuracy(
            hidden_ridge, truth[diagnostic_test]
        ),
        "diagnostic_train_lives": float(split_life),
        "diagnostic_test_lives": float(lives - split_life),
        "zero_balanced_accuracy": zero_balanced,
        "shuffle_balanced_accuracy": shuffle_balanced,
        "zero_accuracy_drop": balanced - zero_balanced,
        "shuffle_accuracy_drop": balanced - shuffle_balanced,
        "ten_tick_change_ordering": (
            change_agreements / max(1, change_comparisons)
        ),
        "ten_tick_change_comparisons": float(change_comparisons),
        **fork,
    }
    for index, need in enumerate(REPORT_NEEDS):
        need_mask = truth == index
        result[f"{need}_recall"] = float(
            np.mean(predicted[need_mask] == index)
        )
        result[f"{need}_mean_absolute_error"] = float(
            absolute_error_by_need[index] / max(1, samples)
        )
    for bucket, (start, stop) in enumerate(
        ((25, 100), (100, 200), (200, 300), (300, 400))
    ):
        mask = (ticks >= start) & (ticks < stop)
        result[f"bucket_{bucket}_samples"] = float(mask.sum())
        result[f"bucket_{bucket}_balanced_accuracy"] = (
            _balanced_accuracy(predicted[mask], truth[mask])
            if np.any(mask)
            else 0.0
        )
    result["gate_passed"] = float(
        balanced >= 0.70
        and observation_balanced <= 0.45
        and result["ten_tick_change_ordering"] >= 0.80
        and result["zero_accuracy_drop"] >= 0.20
        and result["shuffle_accuracy_drop"] >= 0.20
        and fork["fork_following"] >= 0.60
    )
    return result
