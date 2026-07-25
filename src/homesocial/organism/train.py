"""Online lifelong training loop for the organism.

The agent lives on the island: it acts, hears, and learns from every segment
of experience as it happens. One optimizer updates one model; actor-critic
and world-model prediction losses share the recurrent core. Hidden state
persists within a life and resets at death.

Language is never rewarded. Utterance prediction is a perceptual loss (the
caregiver is part of the world worth predicting); any survival benefit of
language must come through behavior.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from random import Random

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from homesocial.creole.vocab import (
    PAD_TOKEN,
    TOKEN_TO_ID,
    VOCAB,
    encode_utterance,
)
from homesocial.creole.situations import Situation
from homesocial.env import Action, DELTAS, DIRECTION_ORDER, Direction
from homesocial.island.oracle import OraclePolicy
from homesocial.island.world import (
    CONSUMABLE_SURFACES,
    SURFACES,
    SURFACE_INDEX,
    IslandConfig,
    IslandWorld,
    ObsPacket,
)
from homesocial.organism.model import OrganismModel

ACTIONS = list(Action)
OBJECT_OPTION_ORDER = ("consume", "inspect")


def enabled_object_options(
    *, consume_options: bool, inspect_options: bool
) -> tuple[str, ...]:
    return tuple(
        option
        for option, enabled in (
            ("consume", consume_options),
            ("inspect", inspect_options),
        )
        if enabled
    )


def agent_action_size(
    *, consume_options: bool, inspect_options: bool = False, visible_slots: int
) -> int:
    option_types = enabled_object_options(
        consume_options=consume_options,
        inspect_options=inspect_options,
    )
    return len(ACTIONS) + visible_slots * len(option_types)


def decode_object_option(
    action_index: int,
    *,
    consume_options: bool,
    inspect_options: bool,
    visible_slots: int,
) -> tuple[str, int] | None:
    """Return the learner-visible option kind and slot for an action index."""

    offset = action_index - len(ACTIONS)
    if offset < 0 or visible_slots <= 0:
        return None
    option_types = enabled_object_options(
        consume_options=consume_options,
        inspect_options=inspect_options,
    )
    option_type_index, slot = divmod(offset, visible_slots)
    if option_type_index >= len(option_types):
        return None
    return option_types[option_type_index], slot


def object_option_action_index(
    option_kind: str,
    slot: int,
    *,
    consume_options: bool,
    inspect_options: bool,
    visible_slots: int,
) -> int:
    """Encode an object-option kind and learner-visible slot as an action."""

    option_types = enabled_object_options(
        consume_options=consume_options,
        inspect_options=inspect_options,
    )
    if option_kind not in option_types:
        raise ValueError(f"Object option {option_kind!r} is not enabled.")
    if not 0 <= slot < visible_slots:
        raise ValueError(f"Object slot {slot} is outside [0, {visible_slots}).")
    return len(ACTIONS) + option_types.index(option_kind) * visible_slots + slot


def available_action_mask(
    packet: ObsPacket,
    action_size: int,
    *,
    visible_slots: int | None = None,
    semantic_choice_delayed: bool = False,
    return_pending: bool = False,
    round_pending: bool = False,
) -> np.ndarray:
    """Primitive actions plus perceived slots in every object-option block."""

    mask = np.ones(action_size, dtype=np.bool_)
    extra_actions = action_size - len(ACTIONS)
    if extra_actions > 0:
        slots = packet.max_visible_slots if visible_slots is None else visible_slots
        if slots <= 0 or extra_actions % slots:
            raise ValueError("Object-option actions must form visible-slot blocks.")
        mask[len(ACTIONS) :] = False
        for block in range(extra_actions // slots):
            start = len(ACTIONS) + block * slots
            mask[start : start + len(packet.visible)] = True
    if semantic_choice_delayed:
        # The delayed probe exposes only abstract object options at the
        # canonical choice pose.  A label packet admits one non-agent-chosen
        # WAIT-coded return macro; the collector excludes it from actor and
        # entropy credit while still learning its physical consequences.
        wait_index = ACTIONS.index(Action.WAIT)
        mask[: len(ACTIONS)] = False
        mask[wait_index] = True
        if return_pending or round_pending:
            mask[len(ACTIONS) :] = False
    return mask


def _turn_toward(current: Direction, desired: Direction) -> Action:
    delta = (
        DIRECTION_ORDER.index(desired) - DIRECTION_ORDER.index(current)
    ) % len(DIRECTION_ORDER)
    return Action.TURN_RIGHT if delta in (1, 2) else Action.TURN_LEFT


def _desired_direction(
    current: tuple[int, int], target: tuple[int, int]
) -> Direction:
    dx = target[0] - current[0]
    dy = target[1] - current[1]
    if abs(dx) >= abs(dy) and dx != 0:
        return Direction.EAST if dx > 0 else Direction.WEST
    return Direction.SOUTH if dy > 0 else Direction.NORTH


def _motor_action_toward(
    world: IslandWorld,
    target: tuple[int, int],
    terminal_action: Action,
) -> Action:
    """Kind-blind routed action toward and then upon a perceived object."""

    grid = world.grid
    distance = abs(target[0] - grid.agent_pos[0]) + abs(
        target[1] - grid.agent_pos[1]
    )
    if distance == 1:
        desired = _desired_direction(grid.agent_pos, target)
        if grid.direction == desired:
            return terminal_action
        return _turn_toward(grid.direction, desired)

    # Consumables are non-blocking, so ordinary movement can leave the agent
    # standing on a selected object (distance zero).  Route to a free adjacent
    # cell rather than incorrectly applying ASK/CONSUME to whatever is ahead.
    # The servo reads only positions and the public physical ``blocks`` flag;
    # it never reads a target's hidden bodily kind.
    goals = {
        (target[0] + dx, target[1] + dy)
        for dx, dy in DELTAS.values()
        if 0 <= target[0] + dx < grid.width
        and 0 <= target[1] + dy < grid.height
        and not (
            (obj := grid.object_at((target[0] + dx, target[1] + dy)))
            is not None
            and obj.blocks
        )
    }
    queue = deque([grid.agent_pos])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {
        grid.agent_pos: None
    }
    goal: tuple[int, int] | None = None
    while queue:
        current = queue.popleft()
        if current in goals and current != target:
            goal = current
            break
        for direction in DIRECTION_ORDER:
            dx, dy = DELTAS[direction]
            neighbor = (current[0] + dx, current[1] + dy)
            if neighbor in parent or neighbor == target:
                continue
            if not (0 <= neighbor[0] < grid.width and 0 <= neighbor[1] < grid.height):
                continue
            obj = grid.object_at(neighbor)
            if obj is not None and obj.blocks:
                continue
            parent[neighbor] = current
            queue.append(neighbor)
    if goal is None:
        return Action.WAIT
    waypoint = goal
    while parent[waypoint] != grid.agent_pos:
        previous = parent[waypoint]
        if previous is None:
            return Action.WAIT
        waypoint = previous
    desired = _desired_direction(grid.agent_pos, waypoint)
    if grid.direction != desired:
        return _turn_toward(grid.direction, desired)
    return Action.MOVE_FORWARD


def _semantic_choice_return_action(world: IslandWorld) -> Action:
    """Next public-geometry motor act toward the canonical choice pose."""

    grid = world.grid
    center = world.semantic_choice_center
    if grid.agent_pos != center:
        desired = _desired_direction(grid.agent_pos, center)
        if grid.direction != desired:
            return _turn_toward(grid.direction, desired)
        return Action.MOVE_FORWARD
    if grid.direction != Direction.NORTH:
        return _turn_toward(grid.direction, Direction.NORTH)
    return Action.WAIT


def _execute_semantic_choice_return(
    world: IslandWorld,
) -> tuple[ObsPacket, float, bool, bool, dict[str, object]]:
    """Execute the fixed-duration, padding-only post-label return phase."""

    duration = world.config.semantic_choice_return_duration
    if duration <= 0 or not world.semantic_choice_return_pending:
        raise ValueError("No semantic-choice return is pending.")
    reward_sum = 0.0
    viability_sum = 0.0
    min_viability = 1.0
    events: list[object] = []
    current_packet: ObsPacket | None = None
    final_info: dict[str, object] = {}
    terminated = False
    truncated = False
    for _ in range(duration):
        primitive = _semantic_choice_return_action(world)
        current_packet, reward, terminated, truncated, info = world.step(primitive)
        reward_sum += float(reward)
        viability_sum += float(info["mean_viability"])
        min_viability = min(min_viability, float(info["viability"]))
        events.append(info.get("event"))
        final_info = dict(info)
        if terminated or truncated:
            break
    world.complete_semantic_choice_return()
    assert current_packet is not None
    if not terminated and not truncated:
        if world.grid.agent_pos != world.semantic_choice_center:
            raise RuntimeError("Fixed return failed to reach semantic-choice center.")
        if world.grid.direction != Direction.NORTH:
            raise RuntimeError("Fixed return failed to canonicalize heading.")
        if current_packet.last_action_index != ACTIONS.index(Action.WAIT):
            raise RuntimeError("Fixed return must end in a padding WAIT packet.")
    final_info["duration"] = len(events)
    final_info["mean_viability_sum"] = viability_sum
    final_info["min_viability"] = min_viability
    final_info["events"] = tuple(events)
    final_info["option_kind"] = "return"
    final_info["forced_return"] = True
    return current_packet, reward_sum, terminated, truncated, final_info


def _execute_semantic_choice_round_transition(
    world: IslandWorld,
) -> tuple[ObsPacket, float, bool, bool, dict[str, object]]:
    """Expose consumption, then force the next recurring bodily demand."""

    packet, reward, terminated, truncated, info = (
        world.start_next_semantic_choice_round()
    )
    aggregate = dict(info)
    aggregate["duration"] = 1
    aggregate["mean_viability_sum"] = float(info["mean_viability"])
    aggregate["min_viability"] = float(info["viability"])
    aggregate["events"] = (info.get("event"),)
    aggregate["option_kind"] = "round_transition"
    aggregate["forced_round_transition"] = True
    return packet, reward, terminated, truncated, aggregate


def execute_agent_action(
    world: IslandWorld,
    packet: ObsPacket,
    action_index: int,
    *,
    consume_options: bool,
    inspect_options: bool = False,
) -> tuple[ObsPacket, float, bool, bool, dict[str, object]]:
    """Execute one primitive decision or a visible-slot object option.

    The option can access only the learner-perceived relative target location
    and proprioceptive pose. It never reads the target's hidden bodily kind.
    """

    if world.semantic_choice_round_pending:
        return _execute_semantic_choice_round_transition(world)
    if world.semantic_choice_return_pending:
        return _execute_semantic_choice_return(world)

    if action_index < len(ACTIONS):
        next_packet, reward, terminated, truncated, info = world.step(
            ACTIONS[action_index]
        )
        aggregate = dict(info)
        aggregate["duration"] = 1
        aggregate["mean_viability_sum"] = float(info["mean_viability"])
        aggregate["min_viability"] = float(info["viability"])
        aggregate["events"] = (info.get("event"),)
        return next_packet, reward, terminated, truncated, aggregate

    decoded = decode_object_option(
        action_index,
        consume_options=consume_options,
        inspect_options=inspect_options,
        visible_slots=world.config.max_visible_slots,
    )
    if decoded is None:
        return execute_agent_action(
            world,
            packet,
            ACTIONS.index(Action.WAIT),
            consume_options=False,
            inspect_options=False,
        )
    option_kind, slot = decoded
    if slot >= len(packet.visible):
        return execute_agent_action(
            world,
            packet,
            ACTIONS.index(Action.WAIT),
            consume_options=False,
            inspect_options=False,
        )
    dx, dy, _ = packet.visible[slot]
    target = (packet.position[0] + dx, packet.position[1] + dy)
    terminal_action = (
        Action.CONSUME if option_kind == "consume" else Action.ASK
    )
    reward_sum = 0.0
    viability_sum = 0.0
    min_viability = 1.0
    events: list[object] = []
    current_packet = packet
    terminated = False
    truncated = False
    final_info: dict[str, object] = {}
    # The delayed three-way probe always begins options at its canonical center
    # with cardinal targets at distance two. Pad every outgoing inspect and
    # consume macro to four ticks, applying ASK/CONSUME only on the final tick,
    # so target direction cannot leak through metabolism or duration. Generic
    # island options keep their routed variable-duration behavior.
    fixed_choice_duration = (
        4
        if world.config.semantic_choice_trial
        and world.config.semantic_choice_return_duration > 0
        else None
    )
    option_budget = fixed_choice_duration or 12
    for option_step in range(option_budget):
        primitive = _motor_action_toward(world, target, terminal_action)
        if (
            fixed_choice_duration is not None
            and option_step < fixed_choice_duration - 1
            and primitive == terminal_action
        ):
            primitive = Action.WAIT
        if (
            fixed_choice_duration is not None
            and option_step == fixed_choice_duration - 1
            and primitive != terminal_action
        ):
            raise RuntimeError(
                "Fixed semantic-choice option did not reach its target in time."
            )
        current_packet, reward, terminated, truncated, info = world.step(primitive)
        reward_sum += float(reward)
        viability_sum += float(info["mean_viability"])
        min_viability = min(min_viability, float(info["viability"]))
        events.append(info.get("event"))
        final_info = dict(info)
        if (
            fixed_choice_duration is None
            and primitive in {terminal_action, Action.WAIT}
        ) or terminated or truncated:
            break
    final_info["duration"] = len(events)
    final_info["mean_viability_sum"] = viability_sum
    final_info["min_viability"] = min_viability
    final_info["events"] = tuple(events)
    final_info["option_kind"] = option_kind
    final_info["option_slot"] = slot
    return current_packet, reward_sum, terminated, truncated, final_info


def planned_policy_logits(
    model: OrganismModel,
    states: mx.array,
    current_vectors: mx.array,
    base_logits: mx.array,
    planning_scales: mx.array,
    *,
    reward_weight: float,
    score_sign: float = 1.0,
    horizon: int = 1,
    force_return_after_inspect: bool = False,
    observation_branching: bool = False,
    persistent_information_reuses: int = 0,
    persistent_low_need: float = 0.55,
    urgent_deficit_utility: bool = False,
    protocol_branch: bool = False,
) -> mx.array:
    """Bias policy logits with detached predicted future-body values.

    ``score_sign=-1`` is an evaluation intervention: it makes the organism
    prefer the consequences its own self-model judges worse while preserving
    the policy, recurrent state, consequence predictions, and action mask.
    """

    if horizon == 1:
        predicted_needs, predicted_rewards = predict_all_action_consequences(
            model, states, current_vectors
        )
        scores = (
            _bodily_terminal_score(
                predicted_needs,
                current_vectors[:, :, None, :4],
                urgent_deficit_utility=urgent_deficit_utility,
            )
            + reward_weight * predicted_rewards
        )
    elif horizon == 2:
        if observation_branching and force_return_after_inspect:
            scores = observation_branching_action_scores(
                model,
                states,
                current_vectors,
                persistent_information_reuses=(
                    persistent_information_reuses
                ),
                persistent_low_need=persistent_low_need,
                urgent_deficit_utility=urgent_deficit_utility,
                protocol_branch=protocol_branch,
            )
        else:
            scores = two_step_action_scores(
                model,
                states,
                current_vectors,
                reward_weight=reward_weight,
                force_return_after_inspect=force_return_after_inspect,
                urgent_deficit_utility=urgent_deficit_utility,
            )
    else:
        raise ValueError("Self-model planning horizon must be 1 or 2.")
    scores = mx.stop_gradient(scores - scores.mean(axis=-1, keepdims=True))
    return base_logits + score_sign * planning_scales[:, :, None] * scores


def predict_all_action_consequences(
    model: OrganismModel,
    states: mx.array,
    current_vectors: mx.array,
) -> tuple[mx.array, mx.array]:
    """Predict bodily needs and reward after every candidate action."""

    batch, steps, hidden = states.shape
    action_size = model.action_size
    expanded_states = mx.broadcast_to(
        states[:, :, None, :], (batch, steps, action_size, hidden)
    ).reshape(batch, steps * action_size, hidden)
    candidate_actions = mx.broadcast_to(
        mx.arange(action_size)[None, None, :],
        (batch, steps, action_size),
    ).reshape(batch, steps * action_size)
    expanded_vectors = mx.broadcast_to(
        current_vectors[:, :, None, :],
        (batch, steps, action_size, current_vectors.shape[-1]),
    ).reshape(batch, steps * action_size, current_vectors.shape[-1])
    _, predicted_need_deltas, predicted_rewards, _ = model.predict_consequences(
        expanded_states, candidate_actions, expanded_vectors
    )
    predicted_need_deltas = predicted_need_deltas.reshape(
        batch, steps, action_size, 4
    )
    return (
        mx.clip(
            current_vectors[:, :, None, :4] + predicted_need_deltas,
            0.0,
            1.0,
        ),
        predicted_rewards.reshape(batch, steps, action_size),
    )


def two_step_action_scores(
    model: OrganismModel,
    states: mx.array,
    current_vectors: mx.array,
    *,
    reward_weight: float,
    force_return_after_inspect: bool = False,
    urgent_deficit_utility: bool = False,
) -> mx.array:
    """Best predicted two-decision bodily outcome for each first action."""

    batch, steps, hidden = states.shape
    action_size = model.action_size
    vector_size = current_vectors.shape[-1]
    first_states = mx.broadcast_to(
        states[:, :, None, :], (batch, steps, action_size, hidden)
    )
    first_vectors = mx.broadcast_to(
        current_vectors[:, :, None, :],
        (batch, steps, action_size, vector_size),
    )
    first_actions = mx.broadcast_to(
        mx.arange(action_size)[None, None, :],
        (batch, steps, action_size),
    )
    imagined_state = model.transition_state(
        first_states, first_actions, first_vectors
    )
    predicted_vector, need_delta, first_reward, _ = model.decode_transition(
        imagined_state
    )
    first_needs = mx.clip(
        current_vectors[:, :, None, :4] + need_delta, 0.0, 1.0
    )
    predicted_vector = mx.concatenate(
        [first_needs, predicted_vector[..., 4:]], axis=-1
    )

    second_states = mx.broadcast_to(
        imagined_state[:, :, :, None, :],
        (batch, steps, action_size, action_size, hidden),
    )
    second_vectors = mx.broadcast_to(
        predicted_vector[:, :, :, None, :],
        (batch, steps, action_size, action_size, vector_size),
    )
    second_actions = mx.broadcast_to(
        mx.arange(action_size)[None, None, None, :],
        (batch, steps, action_size, action_size),
    )
    second_imagined_state = model.transition_state(
        second_states, second_actions, second_vectors
    )
    _, second_need_delta, second_reward, _ = model.decode_transition(
        second_imagined_state
    )
    second_needs = mx.clip(
        first_needs[:, :, :, None, :] + second_need_delta,
        0.0,
        1.0,
    )
    second_scores = (
        _bodily_terminal_score(
            second_needs,
            first_needs[:, :, :, None, :],
            urgent_deficit_utility=urgent_deficit_utility,
        )
        + reward_weight * (first_reward[:, :, :, None] + second_reward)
    )
    if model.has_object_options:
        start = model.object_feature_offset
        stop = start + model.visible_slots * model.object_feature_size
        slot_presence = predicted_vector[..., start:stop].reshape(
            batch,
            steps,
            action_size,
            model.visible_slots,
            model.object_feature_size,
        )[..., 0]
        second_mask = mx.concatenate(
            [
                mx.ones(
                    (batch, steps, action_size, model.primitive_action_size),
                    dtype=mx.bool_,
                ),
                *[
                    slot_presence > 0.5
                    for _ in range(model.object_option_types)
                ],
            ],
            axis=-1,
        )
        if force_return_after_inspect and model.object_option_types >= 2:
            inspect_start = model.primitive_action_size + model.visible_slots
            inspect_stop = inspect_start + model.visible_slots
            first_ids = mx.arange(action_size)
            first_is_inspect = (
                (first_ids >= inspect_start) & (first_ids < inspect_stop)
            )[None, None, :, None]
            return_only = mx.broadcast_to(
                (
                    mx.arange(action_size) == ACTIONS.index(Action.WAIT)
                )[None, None, None, :],
                second_mask.shape,
            )
            second_mask = mx.where(first_is_inspect, return_only, second_mask)
        second_scores = mx.where(second_mask, second_scores, -1e9)
    best_scores = mx.max(second_scores, axis=-1)
    if force_return_after_inspect and model.object_option_types >= 1:
        # Every consume option terminates the delayed semantic-choice life.
        # Its value is therefore the first predicted consequence, never a
        # fictitious second transition from an out-of-distribution terminal
        # latent. Generic open-island planning retains its two-step rollout.
        consume_start = model.primitive_action_size
        consume_stop = consume_start + model.visible_slots
        first_ids = mx.arange(action_size)
        first_is_consume = (
            (first_ids >= consume_start) & (first_ids < consume_stop)
        )[None, None, :]
        first_scores = (
            _bodily_terminal_score(
                first_needs,
                current_vectors[:, :, None, :4],
                urgent_deficit_utility=urgent_deficit_utility,
            )
            + reward_weight * first_reward
        )
        best_scores = mx.where(first_is_consume, first_scores, best_scores)
    return best_scores


SEMANTIC_CHOICE_LABEL_KINDS = ("food", "water", "danger")


def semantic_choice_label_candidates(*, collapsed: bool = False) -> mx.array:
    """Candidate label packets for the controlled three-way belief backup."""

    pad_id = TOKEN_TO_ID[PAD_TOKEN]
    if collapsed:
        packets = [
            (pad_id,) * len(encode_utterance(("this", kind)))
            for kind in SEMANTIC_CHOICE_LABEL_KINDS
        ]
    else:
        packets = [
            encode_utterance(("this", kind))
            for kind in SEMANTIC_CHOICE_LABEL_KINDS
        ]
    return mx.array(np.asarray(packets, dtype=np.int32))


def _sequence_branch_probabilities(
    token_logits: mx.array,
    candidate_tokens: mx.array,
) -> mx.array:
    """Normalize complete candidate-sequence likelihoods into branches."""

    batch, steps, length, vocab_size = token_logits.shape
    candidates = candidate_tokens.shape[0]
    log_probabilities = token_logits - mx.logsumexp(
        token_logits, axis=-1, keepdims=True
    )
    expanded_log_probabilities = mx.broadcast_to(
        log_probabilities[:, :, None, :, :],
        (batch, steps, candidates, length, vocab_size),
    )
    expanded_tokens = mx.broadcast_to(
        candidate_tokens[None, None, :, :, None],
        (batch, steps, candidates, length, 1),
    )
    sequence_log_probabilities = mx.take_along_axis(
        expanded_log_probabilities,
        expanded_tokens,
        axis=-1,
    )[..., 0].sum(axis=-1)
    return mx.softmax(sequence_log_probabilities, axis=-1)


def _semantic_choice_return_vector(
    current_vectors: mx.array,
    predicted_needs: mx.array,
) -> mx.array:
    """Exact public center/NORTH/WAIT observation after the fixed return."""

    leading = current_vectors.shape[:-1]
    north = mx.broadcast_to(
        mx.eye(len(DIRECTION_ORDER), dtype=current_vectors.dtype)[
            DIRECTION_ORDER.index(Direction.NORTH)
        ],
        (*leading, len(DIRECTION_ORDER)),
    )
    wait = mx.broadcast_to(
        mx.eye(len(ACTIONS), dtype=current_vectors.dtype)[
            ACTIONS.index(Action.WAIT)
        ],
        (*leading, len(ACTIONS)),
    )
    position_start = 4 + len(DIRECTION_ORDER)
    action_start = position_start + 2
    tail_start = action_start + len(ACTIONS)
    return mx.concatenate(
        [
            predicted_needs,
            north,
            current_vectors[..., position_start:action_start],
            wait,
            current_vectors[..., tail_start:],
        ],
        axis=-1,
    )


def _bodily_terminal_score(
    terminal_needs: mx.array,
    reference_needs: mx.array,
    *,
    urgent_deficit_utility: bool,
) -> mx.array:
    """Reduce predicted post-action needs to one bodily value.

    The legacy rule is the predicted minimum need, the ``n -> infinity`` limit
    of a homeostatic drive. After a multi-tick detour that minimum is fixed by
    a need no consumption choice affects, and it rewards an uncertain smeared
    prediction over a correct concentrated one. The corrected rule scores the
    predicted level of whichever need the organism *currently* observes as its
    lowest, which is ordinary drive reduction on the most urgent deficit and
    reads only the organism's own interoception.
    """

    if not urgent_deficit_utility:
        return mx.min(terminal_needs, axis=-1)
    reference_needs = mx.broadcast_to(reference_needs, terminal_needs.shape)
    urgent = mx.argmin(reference_needs, axis=-1, keepdims=True)
    return mx.take_along_axis(terminal_needs, urgent, axis=-1).squeeze(-1)


def _terminal_consume_scores(
    model: OrganismModel,
    states: mx.array,
    vectors: mx.array,
    *,
    urgent_deficit_utility: bool = False,
) -> mx.array:
    """Pure bodily value for each visible-slot terminal consume option."""

    batch, steps, state_size = states.shape
    vector_size = vectors.shape[-1]
    consume_actions = mx.broadcast_to(
        (
            model.primitive_action_size
            + mx.arange(model.visible_slots)
        )[None, None, :],
        (batch, steps, model.visible_slots),
    )
    expanded_states = mx.broadcast_to(
        states[:, :, None, :],
        (batch, steps, model.visible_slots, state_size),
    )
    expanded_vectors = mx.broadcast_to(
        vectors[:, :, None, :],
        (batch, steps, model.visible_slots, vector_size),
    )
    consume_states = model.transition_state(
        expanded_states,
        consume_actions,
        expanded_vectors,
    )
    _, need_deltas, _, _ = model.decode_transition(consume_states)
    terminal_needs = mx.clip(
        vectors[:, :, None, :4] + need_deltas,
        0.0,
        1.0,
    )
    scores = _bodily_terminal_score(
        terminal_needs,
        vectors[:, :, None, :4],
        urgent_deficit_utility=urgent_deficit_utility,
    )
    slot_start = model.object_feature_offset
    slot_stop = (
        slot_start + model.visible_slots * model.object_feature_size
    )
    presence = vectors[..., slot_start:slot_stop].reshape(
        batch,
        steps,
        model.visible_slots,
        model.object_feature_size,
    )[..., 0]
    return mx.where(presence > 0.5, scores, -1e9)


# The public post-label protocol places exactly three padding-only
# observations between hearing a word and being able to act on it in the next
# recurring demand: the fixed return, the forced round transition, and the next
# choice pose. A counterfactual write has to travel the same distance, because
# an external bank row reaches the recurrent core only through observation
# steps.
PROTOCOL_SETTLING_OBSERVATIONS = 3


def _settled_context_state(
    model: OrganismModel,
    states: mx.array,
    context_vector: mx.array,
    *,
    steps: int,
) -> mx.array:
    """Apply the public padding-only protocol observations to a state."""

    padding_tokens = mx.broadcast_to(
        mx.array(
            [TOKEN_TO_ID[PAD_TOKEN]] * model.tokens_per_utterance,
            dtype=mx.int32,
        )[None, None, :],
        (*context_vector.shape[:2], model.tokens_per_utterance),
    )
    state = states
    for _ in range(steps):
        state = model.observe_from_state(state, context_vector, padding_tokens)
    return state


def _persistent_self_context_value(
    model: OrganismModel,
    states: mx.array,
    current_vectors: mx.array,
    *,
    low_need: float,
    urgent_deficit_utility: bool = False,
    settling_steps: int = 1,
) -> mx.array:
    """Expected best bodily outcome across recurring hungry/thirsty selves.

    These are counterfactual queries to the learned consequence model, not
    simulator branches. Object surfaces and geometry remain exactly those in
    the learner's current observation; only its public bodily context changes.
    """

    if not 0.0 < low_need < 0.75:
        raise ValueError("Persistent self-query low need must be in (0, 0.75).")
    batch, steps = current_vectors.shape[:2]
    context_values: list[mx.array] = []
    for need_index in (0, 1):
        needs = [0.75, 0.75, 0.75, 0.75]
        needs[need_index] = low_need
        context_needs = mx.broadcast_to(
            mx.array(needs, dtype=current_vectors.dtype)[None, None, :],
            (batch, steps, 4),
        )
        context_vector = _semantic_choice_return_vector(
            current_vectors,
            context_needs,
        )
        context_state = _settled_context_state(
            model,
            states,
            context_vector,
            steps=settling_steps,
        )
        scores = _terminal_consume_scores(
            model,
            context_state,
            context_vector,
            urgent_deficit_utility=urgent_deficit_utility,
        )
        context_values.append(mx.max(scores, axis=-1))
    return mx.stack(context_values, axis=-1).mean(axis=-1)


def observation_branching_inspect_values(
    model: OrganismModel,
    states: mx.array,
    current_vectors: mx.array,
    *,
    collapsed_labels: bool = False,
    persistent_information_reuses: int = 0,
    persistent_low_need: float = 0.55,
    urgent_deficit_utility: bool = False,
    protocol_branch: bool = False,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Expected bodily value after inspect -> observation -> return -> consume.

    Returns inspect values, observation probabilities, best branch values, and
    best terminal consume-slot indices. The last three tensors have shape
    ``(batch, steps, visible_slots, three_label_branches)``.
    """

    if not model.has_episodic_bindings:
        raise ValueError("Observation branching requires episodic bindings.")
    if model.object_option_types != 2:
        raise ValueError(
            "Observation branching requires consume and inspect option blocks."
        )
    if persistent_information_reuses < 0:
        raise ValueError("Persistent information reuses must be nonnegative.")

    batch, steps, state_size = states.shape
    vector_size = current_vectors.shape[-1]
    candidate_tokens = semantic_choice_label_candidates(
        collapsed=collapsed_labels
    )
    padding_tokens = mx.broadcast_to(
        mx.array(
            [TOKEN_TO_ID[PAD_TOKEN]] * model.tokens_per_utterance,
            dtype=mx.int32,
        )[None, None, :],
        (batch, steps, model.tokens_per_utterance),
    )
    wait_actions = mx.broadcast_to(
        mx.array(ACTIONS.index(Action.WAIT), dtype=mx.int32),
        (batch, steps),
    )
    slots = current_vectors[
        ...,
        model.object_feature_offset : (
            model.object_feature_offset
            + model.visible_slots * model.object_feature_size
        ),
    ].reshape(
        batch,
        steps,
        model.visible_slots,
        model.object_feature_size,
    )

    inspect_values: list[mx.array] = []
    all_probabilities: list[mx.array] = []
    all_branch_values: list[mx.array] = []
    all_branch_choices: list[mx.array] = []
    for slot in range(model.visible_slots):
        inspect_action = (
            model.primitive_action_size + model.visible_slots + slot
        )
        inspect_actions = mx.broadcast_to(
            mx.array(inspect_action, dtype=mx.int32),
            (batch, steps),
        )
        inspect_state = model.transition_state(
            states,
            inspect_actions,
            current_vectors,
        )
        predicted_vector, inspect_delta, _, token_logits = (
            model.decode_transition(inspect_state)
        )
        inspect_needs = mx.clip(
            current_vectors[..., :4] + inspect_delta,
            0.0,
            1.0,
        )
        label_vector = mx.concatenate(
            [inspect_needs, predicted_vector[..., 4:]],
            axis=-1,
        )
        probabilities = _sequence_branch_probabilities(
            token_logits,
            candidate_tokens,
        )
        target_surface = slots[..., slot, 3:]

        branch_values: list[mx.array] = []
        branch_choices: list[mx.array] = []
        for branch in range(candidate_tokens.shape[0]):
            branch_tokens = mx.broadcast_to(
                candidate_tokens[branch][None, None, :],
                (batch, steps, model.tokens_per_utterance),
            )
            if protocol_branch:
                # No sensory scene is fabricated. The hypothetical word is
                # written to the external bank, and the state then travels the
                # public padding-only protocol, which is also what lets the
                # written row reach the recurrent core.
                post_label_state = model.write_binding_into_state(
                    states,
                    target_surface,
                    branch_tokens,
                )
                return_carrier = _semantic_choice_return_vector(
                    current_vectors,
                    inspect_needs,
                )
            else:
                post_label_state = model.observe_from_state(
                    states,
                    label_vector,
                    branch_tokens,
                    binding_surfaces=target_surface,
                )
                return_carrier = label_vector
            return_state = model.transition_state(
                post_label_state,
                wait_actions,
                return_carrier,
            )
            _, return_delta, _, _ = model.decode_transition(return_state)
            return_needs = mx.clip(
                inspect_needs + return_delta,
                0.0,
                1.0,
            )
            return_vector = _semantic_choice_return_vector(
                current_vectors,
                return_needs,
            )
            if protocol_branch:
                post_return_state = _settled_context_state(
                    model,
                    post_label_state,
                    return_vector,
                    steps=PROTOCOL_SETTLING_OBSERVATIONS,
                )
            else:
                post_return_state = model.observe_from_state(
                    post_label_state,
                    return_vector,
                    padding_tokens,
                )
            consume_scores = _terminal_consume_scores(
                model,
                post_return_state,
                return_vector,
                urgent_deficit_utility=urgent_deficit_utility,
            )
            branch_value = mx.max(consume_scores, axis=-1)
            if persistent_information_reuses > 0:
                branch_value = branch_value + (
                    persistent_information_reuses
                    * _persistent_self_context_value(
                        model,
                        post_label_state if protocol_branch
                        else post_return_state,
                        return_vector,
                        low_need=persistent_low_need,
                        urgent_deficit_utility=urgent_deficit_utility,
                        settling_steps=(
                            PROTOCOL_SETTLING_OBSERVATIONS
                            if protocol_branch
                            else 1
                        ),
                    )
                )
            branch_values.append(branch_value)
            branch_choices.append(mx.argmax(consume_scores, axis=-1))

        stacked_values = mx.stack(branch_values, axis=-1)
        stacked_choices = mx.stack(branch_choices, axis=-1)
        inspect_values.append(
            mx.sum(probabilities * stacked_values, axis=-1)
        )
        all_probabilities.append(probabilities)
        all_branch_values.append(stacked_values)
        all_branch_choices.append(stacked_choices)

    return (
        mx.stack(inspect_values, axis=-1),
        mx.stack(all_probabilities, axis=-2),
        mx.stack(all_branch_values, axis=-2),
        mx.stack(all_branch_choices, axis=-2),
    )


def observation_branching_action_scores(
    model: OrganismModel,
    states: mx.array,
    current_vectors: mx.array,
    *,
    collapsed_labels: bool = False,
    persistent_information_reuses: int = 0,
    persistent_low_need: float = 0.55,
    urgent_deficit_utility: bool = False,
    protocol_branch: bool = False,
) -> mx.array:
    """Delayed-choice scores with exact observation branches for inspection."""

    base_scores = two_step_action_scores(
        model,
        states,
        current_vectors,
        reward_weight=0.0,
        force_return_after_inspect=True,
        urgent_deficit_utility=urgent_deficit_utility,
    )
    consume_scores = _terminal_consume_scores(
        model,
        states,
        current_vectors,
        urgent_deficit_utility=urgent_deficit_utility,
    )
    inspect_values, _, _, _ = observation_branching_inspect_values(
        model,
        states,
        current_vectors,
        collapsed_labels=collapsed_labels,
        persistent_information_reuses=persistent_information_reuses,
        persistent_low_need=persistent_low_need,
        urgent_deficit_utility=urgent_deficit_utility,
        protocol_branch=protocol_branch,
    )
    if persistent_information_reuses > 0:
        continuation = _persistent_self_context_value(
            model,
            states,
            current_vectors,
            low_need=persistent_low_need,
            urgent_deficit_utility=urgent_deficit_utility,
            settling_steps=(
                PROTOCOL_SETTLING_OBSERVATIONS if protocol_branch else 1
            ),
        )
        consume_scores = consume_scores + (
            persistent_information_reuses * continuation[..., None]
        )

        # Under the stationary caregiver used by this experiment, a valid
        # surface-keyed row means that reinspection cannot add a new lexical
        # fact. Its score therefore reverts to the ordinary costly rollout.
        # This gate reads only the organism's own memory-valid state.
        _, _, valid = model._state_memory_parts(states)
        slots = current_vectors[
            ...,
            model.object_feature_offset : (
                model.object_feature_offset
                + model.visible_slots * model.object_feature_size
            ),
        ].reshape(
            *current_vectors.shape[:-1],
            model.visible_slots,
            model.object_feature_size,
        )
        selected_valid = mx.einsum(
            "...vs,...s->...v",
            slots[..., 3:],
            valid,
        )
        inspect_start = model.primitive_action_size + model.visible_slots
        ordinary_inspect = base_scores[
            ..., inspect_start : inspect_start + model.visible_slots
        ]
        inspect_values = mx.where(
            selected_valid > 0.5,
            ordinary_inspect,
            inspect_values,
        )
    return mx.concatenate(
        [
            base_scores[..., : model.primitive_action_size],
            consume_scores,
            inspect_values,
        ],
        axis=-1,
    )


# Metabolic drift over the longest option in the implemented task tops out
# near 0.14, while consumption events begin near 0.225; the delta histogram is
# empty in between. The cut is a gap in the data, not a tuned threshold.
DRIFT_REGIME_THRESHOLD = 0.175
# The per-tick metabolic scale of the world: food 0.010, water 0.014, energy
# 0.015 while walking. Dividing by its square measures the drift regime as a
# relative error instead of an absolute one.
DRIFT_ERROR_SCALE = 0.02


def bodily_delta_prediction_loss(
    predicted_deltas: mx.array,
    target_deltas: mx.array,
    *,
    change_boost: float,
    drift_weight: float = 0.0,
    valid: mx.array | None = None,
) -> mx.array:
    """MSE that preserves rare, action-specific bodily consequences.

    The change-boosted term is the original one and is unchanged. Alone it
    gives the slow-metabolism regime about one percent of the head's gradient,
    because drift entries are both smaller and less heavily weighted than
    consumption events; what the head fits instead is the need-anticorrelated
    line through the events, which extrapolates to a large spurious decay
    whenever a need is high and nothing is consumed.

    ``drift_weight`` adds a second term over the entries that carry no event,
    stratified per need rather than per transition so that the water drift on a
    food-consumption transition still counts as drift, and normalized by the
    metabolic scale so a residual of 0.01 is not invisible beside one of 0.5.
    The term is self-limiting: it falls below the event term once the drift
    residual approaches that scale, so it cannot trade away the consumption
    fit. Zero reproduces the original loss exactly.
    """

    per_entry = (predicted_deltas - target_deltas) ** 2
    per_transition = per_entry.mean(axis=-1)
    weights = 1.0 + change_boost * mx.max(mx.abs(target_deltas), axis=-1)
    if valid is not None:
        weights = weights * valid
    event_term = (per_transition * weights).sum() / mx.maximum(
        weights.sum(), mx.array(1e-8)
    )
    if drift_weight <= 0.0:
        return event_term
    drift_mask = (
        mx.abs(target_deltas) <= DRIFT_REGIME_THRESHOLD
    ).astype(per_entry.dtype)
    if valid is not None:
        drift_mask = drift_mask * valid[..., None]
    drift_term = (per_entry * drift_mask).sum() / mx.maximum(
        drift_mask.sum(), mx.array(1e-8)
    )
    return event_term + drift_weight * drift_term / (DRIFT_ERROR_SCALE**2)


@dataclass(frozen=True)
class OrganismConfig:
    language_mode: str = "grounded"
    total_steps: int = 200_000
    segment_length: int = 64
    hidden_size: int = 256
    token_embed_size: int = 32
    learning_rate: float = 3e-4
    discount: float = 0.99
    gae_lambda: float = 0.95
    entropy_weight: float = 0.02
    value_weight: float = 0.5
    next_vector_weight: float = 0.1
    next_needs_weight: float = 1.0
    bodily_change_loss_boost: float = 20.0
    # Supervises the slow-metabolism regime the change boost starves. Zero is
    # the sealed default and reproduces every prior artifact exactly.
    bodily_drift_loss_weight: float = 0.0
    reward_prediction_weight: float = 0.2
    token_prediction_weight: float = 0.5
    nonpad_token_weight: float = 5.0
    multi_step_model_horizon: int = 1
    multi_step_model_weight: float = 0.0
    world_model_replay_capacity: int = 0
    world_model_replay_updates: int = 0
    live_reward_weight: float = 0.02
    death_penalty: float = 1.0
    max_grad_norm: float = 1.0
    # Infancy curriculum: scale food/water metabolism from `start` at global
    # step 0 to 1.0 at `steps` (0 disables). Development, not reward hacking:
    # early lives are metabolically subsidized so consume events get sampled.
    metabolism_curriculum_start: float = 1.0
    metabolism_curriculum_steps: int = 0
    # Childhood bootstrap. Demonstrations are collected with caregiver tokens
    # masked to padding, so imitation can teach embodied action sequences but
    # cannot pretrain a word-to-action mapping. Zero is the mandatory ablation.
    bc_warmstart_lives: int = 0
    bc_epochs: int = 2
    bc_max_action_weight: float = 8.0
    # In-loop infancy: the caregiver repeatedly offers an embodied food/water
    # resource below this threshold, fading linearly to no help. The agent must
    # still act to consume it; there is no language or shaping reward.
    caregiver_offer_threshold_start: float = 0.0
    caregiver_offer_curriculum_steps: int = 0
    caregiver_offer_distance_end: int = 0
    # A concentrated developmental phase: every life presents a remapped
    # semantic choice (a resource/poison pair or crossed three-object set).
    # Zero preserves the ordinary island (unless ``island`` explicitly enables
    # the diagnostic trial).  Positive values switch to the ordinary island at
    # the first life boundary after this many primitive ticks.
    semantic_choice_childhood_steps: int = 0
    consume_options: bool = False
    inspect_options: bool = False
    # Per-life visual-key/lexical-value memory. Zero is the recurrent-only
    # architectural control; positive values allocate one learned lexical
    # value of this width per learner-visible surface identity.
    episodic_binding_size: int = 0
    episodic_binding_writes: bool = True
    self_model_planning_scale: float = 0.0
    self_model_planning_start_steps: int = 0
    self_model_planning_reward_weight: float = 0.5
    self_model_planning_horizon: int = 1
    observation_branching_planning: bool = False
    persistent_information_reuses: int = 0
    urgent_deficit_utility: bool = False
    protocol_branch_planning: bool = False
    seed: int = 1
    max_steps: int = 1000
    log_every_lives: int = 10
    checkpoint: str | None = None
    stats_csv: str | None = None
    island: IslandConfig = field(default_factory=IslandConfig)

    def __post_init__(self) -> None:
        if self.self_model_planning_horizon not in {1, 2}:
            raise ValueError("self_model_planning_horizon must be 1 or 2.")
        if self.multi_step_model_horizon not in {1, 2}:
            raise ValueError("multi_step_model_horizon must be 1 or 2.")
        if self.persistent_information_reuses < 0:
            raise ValueError(
                "persistent_information_reuses must be nonnegative."
            )
        if self.observation_branching_planning and (
            self.self_model_planning_horizon != 2
            or self.episodic_binding_size <= 0
        ):
            raise ValueError(
                "Observation-branching planning requires horizon two and "
                "episodic bindings."
            )
        if self.world_model_replay_capacity < 0:
            raise ValueError("world_model_replay_capacity must be nonnegative.")
        if self.world_model_replay_updates < 0:
            raise ValueError("world_model_replay_updates must be nonnegative.")
        if self.semantic_choice_childhood_steps < 0:
            raise ValueError("semantic_choice_childhood_steps must be nonnegative.")
        if self.episodic_binding_size < 0:
            raise ValueError("episodic_binding_size must be nonnegative.")
        if self.episodic_binding_size > 0 and not (
            self.consume_options and self.inspect_options
        ):
            raise ValueError(
                "Episodic bindings require consume and inspect object options."
            )

    def island_config(
        self, *, semantic_choice_trial: bool | None = None
    ) -> IslandConfig:
        choice_trial = (
            self.island.semantic_choice_trial
            if semantic_choice_trial is None
            else semantic_choice_trial
        )
        return IslandConfig(
            width=self.island.width,
            height=self.island.height,
            max_steps=self.max_steps,
            semantic_choice_trial=choice_trial,
            semantic_choice_horizon=self.island.semantic_choice_horizon,
            semantic_choice_objects=self.island.semantic_choice_objects,
            semantic_choice_low_need=self.island.semantic_choice_low_need,
            semantic_choice_rounds=(
                self.island.semantic_choice_rounds if choice_trial else 1
            ),
            semantic_choice_return_duration=(
                self.island.semantic_choice_return_duration if choice_trial else 0
            ),
            language_mode=self.language_mode,
            ask_state_period=self.island.ask_state_period,
            visible_radius=self.island.visible_radius,
            max_visible_slots=self.island.max_visible_slots,
            low_need_praise_threshold=self.island.low_need_praise_threshold,
            low_need_ask_threshold=self.island.low_need_ask_threshold,
            caregiver_offer_threshold=self.island.caregiver_offer_threshold,
            caregiver_offer_distance=self.island.caregiver_offer_distance,
        )


@dataclass
class LifeStats:
    life_index: int
    seed: int
    steps: int
    survived: bool
    mean_viability: float
    min_viability: float
    utterances_heard: int
    consume_attempts: int
    resource_consumes: int
    food_consumes: int
    water_consumes: int
    offered_consumes: int
    option_decisions: int
    inspect_option_decisions: int
    labels_received: int
    harm_events: int
    semantic_choice_trial: bool
    choice_need: str
    chosen_kind: str
    chosen_surface: str
    chosen_surface_inspected: bool
    choice_correct: bool
    choice_poison: bool
    choice_wrong_resource: bool
    choice_timeout: bool
    choice_rounds_completed: int
    choice_correct_rounds: int
    choice_poison_rounds: int
    choice_wrong_resource_rounds: int


@dataclass(frozen=True)
class BehaviorCloningStats:
    lives: int
    transitions: int
    epochs: int
    first_epoch_loss: float
    final_epoch_loss: float
    final_accuracy: float
    final_consume_recall: float


class _Demonstration:
    """One oracle life as learner-visible sequential transitions."""

    def __init__(self) -> None:
        self.vectors: list[np.ndarray] = []
        self.tokens: list[tuple[int, ...]] = []
        self.actions: list[int] = []
        self.next_vectors: list[np.ndarray] = []
        self.next_needs: list[tuple[float, ...]] = []
        self.env_rewards: list[float] = []

    def __len__(self) -> int:
        return len(self.actions)


def build_model(config: OrganismConfig, world: IslandWorld) -> OrganismModel:
    object_feature_offset = 4 + len(DIRECTION_ORDER) + 2 + len(ACTIONS)
    return OrganismModel(
        vector_size=ObsPacket.vector_size(
            max_visible_slots=world.config.max_visible_slots
        ),
        vocab_size=len(VOCAB),
        tokens_per_utterance=world.tokens_per_utterance,
        action_size=agent_action_size(
            consume_options=config.consume_options,
            inspect_options=config.inspect_options,
            visible_slots=world.config.max_visible_slots,
        ),
        hidden_size=config.hidden_size,
        token_embed_size=config.token_embed_size,
        primitive_action_size=len(ACTIONS),
        visible_slots=world.config.max_visible_slots,
        object_feature_offset=object_feature_offset,
        object_feature_size=3 + len(SURFACES),
        object_option_types=len(
            enabled_object_options(
                consume_options=config.consume_options,
                inspect_options=config.inspect_options,
            )
        ),
        episodic_binding_size=config.episodic_binding_size,
        episodic_binding_writes=config.episodic_binding_writes,
        visible_radius=world.config.visible_radius,
        pad_token_id=TOKEN_TO_ID[PAD_TOKEN],
        referential_action_indices=(
            ACTIONS.index(Action.POINT),
            ACTIONS.index(Action.ASK),
        ),
    )


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    bootstrap: float,
    dones: np.ndarray,
    durations: np.ndarray | None = None,
    *,
    discount: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    steps = len(rewards)
    advantages = np.zeros(steps, dtype=np.float32)
    last_advantage = 0.0
    next_value = bootstrap
    if durations is None:
        durations = np.ones(steps, dtype=np.float32)
    for t in reversed(range(steps)):
        not_done = 1.0 - float(dones[t])
        step_discount = discount ** float(durations[t])
        delta = rewards[t] + step_discount * next_value * not_done - values[t]
        last_advantage = (
            delta + step_discount * gae_lambda * not_done * last_advantage
        )
        advantages[t] = last_advantage
        next_value = values[t]
    returns = advantages + values
    return advantages, returns


class _Segment:
    def __init__(self) -> None:
        self.semantic_choice_trial = False
        self.semantic_choice_delayed = False
        self.vectors: list[np.ndarray] = []
        self.tokens: list[tuple[int, ...]] = []
        self.actions: list[int] = []
        self.shaped_rewards: list[float] = []
        self.env_rewards: list[float] = []
        self.values: list[float] = []
        self.dones: list[bool] = []
        self.durations: list[int] = []
        self.planning_scales: list[float] = []
        self.action_masks: list[np.ndarray] = []
        self.decision_weights: list[float] = []
        self.next_vectors: list[np.ndarray] = []
        self.next_needs: list[tuple[float, ...]] = []
        self.next_tokens: list[tuple[int, ...]] = []

    def __len__(self) -> int:
        return len(self.actions)


class OrganismTrainer:
    def __init__(self, config: OrganismConfig) -> None:
        self.config = config
        mx.random.seed(config.seed)
        initial_choice_trial = (
            config.semantic_choice_childhood_steps > 0
            or config.island.semantic_choice_trial
        )
        self.world = IslandWorld(
            config.island_config(semantic_choice_trial=initial_choice_trial),
            seed=config.seed,
        )
        self.model = build_model(config, self.world)
        self.optimizer = optim.Adam(learning_rate=config.learning_rate)
        self.rng = Random(config.seed)
        self.replay_rng = Random(config.seed + 1_000_003)
        self.world_model_replay: list[
            tuple[_Segment, np.ndarray | None]
        ] = []
        self._replay_segments_seen = 0
        self.life_stats: list[LifeStats] = []
        self.loss_log: list[dict[str, float]] = []
        self.bc_stats: BehaviorCloningStats | None = None
        self._loss_and_grad = nn.value_and_grad(self.model, self._loss)
        self._bc_loss_and_grad = nn.value_and_grad(self.model, self._bc_loss)
        self._world_model_loss_and_grad = nn.value_and_grad(
            self.model, self._world_model_loss
        )

        self.life_index = 0
        self.life_seed = config.seed
        self.global_steps = 0
        self.packet = self.world.reset(self.life_seed)
        self._apply_development_curricula()
        self.hidden: mx.array | None = None
        self._reset_life_counters()

    def _reset_life_counters(self) -> None:
        self._life_steps = 0
        self._life_viability_sum = 0.0
        self._life_min_viability = 1.0
        self._life_utterances = 0
        self._life_consume_attempts = 0
        self._life_resource_consumes = 0
        self._life_food_consumes = 0
        self._life_water_consumes = 0
        self._life_offered_consumes = 0
        self._life_option_decisions = 0
        self._life_inspect_option_decisions = 0
        self._life_labels_received = 0
        self._life_harm_events = 0
        self._life_semantic_choice_trial = self.world.config.semantic_choice_trial
        self._life_choice_need = self.world.grid.choice_need or ""
        self._life_chosen_kind = ""
        self._life_chosen_surface = ""
        self._life_chosen_surface_inspected = False
        self._life_choice_correct = False
        self._life_choice_poison = False
        self._life_choice_wrong_resource = False
        self._life_choice_timeout = False
        self._life_choice_rounds_completed = 0
        self._life_choice_correct_rounds = 0
        self._life_choice_poison_rounds = 0
        self._life_choice_wrong_resource_rounds = 0

    def _semantic_choice_active(self) -> bool:
        if self.config.semantic_choice_childhood_steps > 0:
            return self.global_steps < self.config.semantic_choice_childhood_steps
        return self.config.island.semantic_choice_trial

    def _start_next_life(self) -> None:
        choice_trial = self._semantic_choice_active()
        if self.world.config.semantic_choice_trial != choice_trial:
            self.world = IslandWorld(
                self.config.island_config(semantic_choice_trial=choice_trial),
                seed=self.life_seed,
            )
        self.packet = self.world.reset(self.life_seed)
        self._apply_development_curricula()
        self.hidden = None
        self._reset_life_counters()

    # --------------------------------------------------- childhood bootstrap

    def collect_oracle_demonstrations(self) -> list[_Demonstration]:
        """Collect learner-visible oracle lives with the language channel masked.

        The oracle can read true kinds to navigate, so these demonstrations are
        not evidence for language learning. Masking every heard token prevents
        the stronger contamination in which behavior cloning directly teaches
        a word-to-action mapping. The zero-BC control remains mandatory.
        """

        demonstrations: list[_Demonstration] = []
        pad_id = TOKEN_TO_ID[PAD_TOKEN]
        padding = (pad_id,) * self.world.tokens_per_utterance
        config = IslandConfig(
            width=self.config.island.width,
            height=self.config.island.height,
            max_steps=self.config.max_steps,
            language_mode="silent",
            ask_state_period=self.config.island.ask_state_period,
            visible_radius=self.config.island.visible_radius,
            max_visible_slots=self.config.island.max_visible_slots,
            low_need_praise_threshold=self.config.island.low_need_praise_threshold,
            low_need_ask_threshold=self.config.island.low_need_ask_threshold,
        )
        oracle = OraclePolicy()
        seed_base = self.config.seed + 100_000
        for life_index in range(self.config.bc_warmstart_lives):
            seed = seed_base + life_index
            world = IslandWorld(config, seed=seed)
            packet = world.reset(seed)
            demonstration = _Demonstration()
            while True:
                action = oracle.act(world.grid)
                next_packet, reward, terminated, truncated, _ = world.step(action)
                demonstration.vectors.append(packet.vector())
                demonstration.tokens.append(padding)
                demonstration.actions.append(ACTIONS.index(action))
                demonstration.next_vectors.append(next_packet.vector())
                demonstration.next_needs.append(next_packet.needs)
                demonstration.env_rewards.append(float(reward))
                if terminated or truncated:
                    break
                packet = next_packet
            demonstrations.append(demonstration)
        return demonstrations

    @staticmethod
    def _balanced_action_weights(
        demonstrations: list[_Demonstration], max_weight: float, action_size: int
    ) -> np.ndarray:
        counts = np.zeros(action_size, dtype=np.float64)
        for demonstration in demonstrations:
            counts += np.bincount(
                np.asarray(demonstration.actions, dtype=np.int32),
                minlength=action_size,
            )
        total = max(1.0, counts.sum())
        weights = np.sqrt(total / (action_size * np.maximum(counts, 1.0)))
        weights = np.minimum(weights, max_weight)
        sample_mean = float((weights * counts).sum() / total)
        return (weights / max(sample_mean, 1e-8)).astype(np.float32)

    def _bc_loss(
        self,
        vectors: mx.array,
        tokens: mx.array,
        hidden: mx.array | None,
        actions: mx.array,
        action_weights: mx.array,
        next_vectors: mx.array,
        next_needs: mx.array,
        env_rewards: mx.array,
    ) -> mx.array:
        states, _ = self.model.core_states(vectors, tokens, hidden)
        logits = self.model.policy_logits(states, vectors)[0]
        log_probabilities = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        chosen = mx.take_along_axis(
            log_probabilities, actions[:, None], axis=-1
        ).squeeze(-1)
        sample_weights = action_weights[actions]
        policy_loss = -(chosen * sample_weights).sum() / sample_weights.sum()

        predicted_vectors, predicted_need_deltas, predicted_rewards, _ = (
            self.model.predict_consequences(states, actions[None, :], vectors)
        )
        vector_loss = ((predicted_vectors[0] - next_vectors) ** 2).mean()
        target_need_deltas = next_needs - vectors[0, :, :4]
        config = self.config
        needs_loss = bodily_delta_prediction_loss(
            predicted_need_deltas[0],
            target_need_deltas,
            change_boost=config.bodily_change_loss_boost,
            drift_weight=config.bodily_drift_loss_weight,
        )
        reward_loss = ((predicted_rewards[0] - env_rewards) ** 2).mean()
        return (
            policy_loss
            + config.next_vector_weight * vector_loss
            + config.next_needs_weight * needs_loss
            + config.reward_prediction_weight * reward_loss
        )

    def _bc_update(
        self,
        demonstration: _Demonstration,
        start: int,
        stop: int,
        hidden: mx.array | None,
        action_weights: mx.array,
    ) -> tuple[float, int, int, mx.array]:
        vectors = mx.array(np.stack(demonstration.vectors[start:stop])[None, ...])
        tokens = mx.array(
            np.asarray(demonstration.tokens[start:stop], dtype=np.int32)[None, ...]
        )
        actions = mx.array(
            np.asarray(demonstration.actions[start:stop], dtype=np.int32)
        )
        next_vectors = mx.array(np.stack(demonstration.next_vectors[start:stop]))
        next_needs = mx.array(
            np.asarray(demonstration.next_needs[start:stop], dtype=np.float32)
        )
        env_rewards = mx.array(
            np.asarray(demonstration.env_rewards[start:stop], dtype=np.float32)
        )
        loss, grads = self._bc_loss_and_grad(
            vectors,
            tokens,
            hidden,
            actions,
            action_weights,
            next_vectors,
            next_needs,
            env_rewards,
        )
        grads, _ = optim.clip_grad_norm(grads, self.config.max_grad_norm)
        self.optimizer.update(self.model, grads)
        mx.eval(self.model.parameters(), self.optimizer.state, loss)

        logits, _, carry = self.model.policy_value(vectors, tokens, hidden)
        predictions = mx.argmax(logits[0], axis=-1)
        mx.eval(predictions, carry)
        predicted = np.asarray(predictions)
        expected = np.asarray(demonstration.actions[start:stop], dtype=np.int32)
        correct = int((predicted == expected).sum())
        consume_index = ACTIONS.index(Action.CONSUME)
        consume_mask = expected == consume_index
        consume_correct = int((predicted[consume_mask] == consume_index).sum())
        return float(loss), correct, consume_correct, mx.stop_gradient(carry)

    def behavior_clone(self) -> BehaviorCloningStats | None:
        config = self.config
        if self.bc_stats is not None:
            return self.bc_stats
        if config.bc_warmstart_lives <= 0 or config.bc_epochs <= 0:
            return None
        demonstrations = self.collect_oracle_demonstrations()
        action_weights = mx.array(
            self._balanced_action_weights(
                demonstrations,
                config.bc_max_action_weight,
                self.model.action_size,
            )
        )
        transitions = sum(len(demonstration) for demonstration in demonstrations)
        consume_index = ACTIONS.index(Action.CONSUME)
        consume_labels = sum(
            demonstration.actions.count(consume_index)
            for demonstration in demonstrations
        )
        epoch_losses: list[float] = []
        final_correct = 0
        final_consume_correct = 0
        bc_rng = Random(config.seed + 200_000)
        for epoch in range(config.bc_epochs):
            order = list(demonstrations)
            bc_rng.shuffle(order)
            loss_sum = 0.0
            seen = 0
            correct = 0
            consume_correct = 0
            for demonstration in order:
                hidden: mx.array | None = None
                for start in range(0, len(demonstration), config.segment_length):
                    stop = min(start + config.segment_length, len(demonstration))
                    loss, batch_correct, batch_consume_correct, hidden = (
                        self._bc_update(
                            demonstration,
                            start,
                            stop,
                            hidden,
                            action_weights,
                        )
                    )
                    batch_size = stop - start
                    loss_sum += loss * batch_size
                    seen += batch_size
                    correct += batch_correct
                    consume_correct += batch_consume_correct
            epoch_losses.append(loss_sum / max(1, seen))
            final_correct = correct
            final_consume_correct = consume_correct
            print(
                f"BC epoch {epoch + 1}/{config.bc_epochs}: "
                f"loss {epoch_losses[-1]:.4f}, action accuracy "
                f"{correct / max(1, seen):.3f}, consume recall "
                f"{consume_correct / max(1, consume_labels):.3f}"
            )
        self.bc_stats = BehaviorCloningStats(
            lives=len(demonstrations),
            transitions=transitions,
            epochs=config.bc_epochs,
            first_epoch_loss=epoch_losses[0],
            final_epoch_loss=epoch_losses[-1],
            final_accuracy=final_correct / max(1, transitions),
            final_consume_recall=final_consume_correct / max(1, consume_labels),
        )
        return self.bc_stats

    def metabolism_factor(self) -> float:
        config = self.config
        if config.metabolism_curriculum_steps <= 0:
            return 1.0
        progress = min(1.0, self.global_steps / config.metabolism_curriculum_steps)
        return config.metabolism_curriculum_start + progress * (
            1.0 - config.metabolism_curriculum_start
        )

    def _apply_metabolism_curriculum(self) -> None:
        factor = self.metabolism_factor()
        self.world.grid.food_metabolism *= factor
        self.world.grid.water_metabolism *= factor

    def caregiver_offer_threshold(self) -> float:
        config = self.config
        if config.caregiver_offer_curriculum_steps <= 0:
            return config.island.caregiver_offer_threshold
        progress = min(
            1.0, self.global_steps / config.caregiver_offer_curriculum_steps
        )
        return config.caregiver_offer_threshold_start * (1.0 - progress)

    def _apply_development_curricula(self) -> None:
        self._apply_metabolism_curriculum()
        self.world.caregiver_offer_threshold = self.caregiver_offer_threshold()
        config = self.config
        if config.caregiver_offer_curriculum_steps > 0:
            progress = min(
                1.0,
                self.global_steps / config.caregiver_offer_curriculum_steps,
            )
            self.world.caregiver_offer_distance = round(
                config.caregiver_offer_distance_end * progress
            )
        else:
            self.world.caregiver_offer_distance = (
                config.island.caregiver_offer_distance
            )

    # ------------------------------------------------------------- acting

    def _act(self) -> tuple[int, float, float, np.ndarray, float]:
        vector = mx.array(self.packet.vector()[None, None, :])
        tokens = mx.array(
            np.asarray(self.packet.tokens, dtype=np.int32)[None, None, :]
        )
        states, self.hidden = self.model.core_states(
            vector, tokens, self.hidden
        )
        logits = self.model.policy_logits(states, vector)
        values = self.model.value(states).squeeze(-1)
        planning_scale = (
            self.config.self_model_planning_scale
            if self.global_steps >= self.config.self_model_planning_start_steps
            else 0.0
        )
        return_pending = self.world.semantic_choice_return_pending
        round_pending = self.world.semantic_choice_round_pending
        delayed_choice = (
            self.world.config.semantic_choice_trial
            and self.world.config.semantic_choice_return_duration > 0
        )
        if return_pending or round_pending:
            planning_scale = 0.0
        if planning_scale > 0.0:
            logits = planned_policy_logits(
                self.model,
                states,
                vector,
                logits,
                mx.array([[[planning_scale]]]).reshape(1, 1),
                reward_weight=self.config.self_model_planning_reward_weight,
                horizon=self.config.self_model_planning_horizon,
                force_return_after_inspect=delayed_choice,
                observation_branching=(
                    self.config.observation_branching_planning
                ),
                persistent_information_reuses=(
                    self.config.persistent_information_reuses
                ),
                urgent_deficit_utility=(
                    self.config.urgent_deficit_utility
                ),
                protocol_branch=self.config.protocol_branch_planning,
                persistent_low_need=(
                    self.config.island.semantic_choice_low_need
                ),
            )
        action_mask = available_action_mask(
            self.packet,
            self.model.action_size,
            visible_slots=self.model.visible_slots or None,
            semantic_choice_delayed=delayed_choice,
            return_pending=return_pending,
            round_pending=round_pending,
        )
        logits = mx.where(
            mx.array(action_mask)[None, None, :], logits, -1e9
        )
        mx.eval(logits, values, self.hidden)
        if return_pending or round_pending:
            action_index = ACTIONS.index(Action.WAIT)
            decision_weight = 0.0
        else:
            probabilities = np.asarray(
                mx.softmax(logits[0, 0], axis=-1), dtype=np.float64
            )
            probabilities = probabilities / probabilities.sum()
            action_index = int(
                self.rng.choices(
                    range(self.model.action_size), weights=probabilities
                )[0]
            )
            decision_weight = 1.0
        return (
            action_index,
            float(values[0, 0]),
            planning_scale,
            action_mask,
            decision_weight,
        )

    def collect_segment(self) -> tuple[_Segment, mx.array | None, float]:
        """Collect experience until segment length or end of life."""

        segment = _Segment()
        segment.semantic_choice_trial = self.world.config.semantic_choice_trial
        segment.semantic_choice_delayed = (
            self.world.config.semantic_choice_trial
            and self.world.config.semantic_choice_return_duration > 0
        )
        initial_hidden = self.hidden
        pad_id = TOKEN_TO_ID[PAD_TOKEN]
        while len(segment) < self.config.segment_length:
            vector_before = self.packet.vector()
            tokens_before = self.packet.tokens
            (
                action_index,
                value,
                planning_scale,
                action_mask,
                decision_weight,
            ) = self._act()
            if action_index >= len(ACTIONS):
                self._life_option_decisions += 1
                decoded = decode_object_option(
                    action_index,
                    consume_options=self.config.consume_options,
                    inspect_options=self.config.inspect_options,
                    visible_slots=self.world.config.max_visible_slots,
                )
                if decoded is not None and decoded[0] == "inspect":
                    self._life_inspect_option_decisions += 1
            next_packet, env_reward, terminated, truncated, info = execute_agent_action(
                self.world,
                self.packet,
                action_index,
                consume_options=self.config.consume_options,
                inspect_options=self.config.inspect_options,
            )
            duration = int(info["duration"])
            mean_viability_sum = float(info["mean_viability_sum"])
            mean_viability = mean_viability_sum / duration
            # A fixed short choice trial would otherwise pay the open-island
            # per-tick survival bonus for delaying until timeout.  Its real
            # homeostatic delta already rewards repair, penalizes poison and
            # telescopes over waiting, so keep that unshaped signal here.
            live_reward_weight = (
                0.0
                if self.world.config.semantic_choice_trial
                else self.config.live_reward_weight
            )
            shaped = (
                env_reward
                + live_reward_weight * mean_viability_sum
                - (self.config.death_penalty if terminated else 0.0)
            )
            segment.vectors.append(vector_before)
            segment.tokens.append(tokens_before)
            segment.actions.append(action_index)
            segment.shaped_rewards.append(shaped)
            segment.env_rewards.append(float(env_reward))
            segment.values.append(value)
            segment.dones.append(terminated)
            segment.durations.append(duration)
            segment.planning_scales.append(planning_scale)
            segment.action_masks.append(action_mask)
            segment.decision_weights.append(decision_weight)
            segment.next_vectors.append(next_packet.vector())
            segment.next_needs.append(next_packet.needs)
            segment.next_tokens.append(next_packet.tokens)

            self._life_steps += duration
            self.global_steps += duration
            self._life_viability_sum += mean_viability_sum
            self._life_min_viability = min(
                self._life_min_viability, float(info["min_viability"])
            )
            if any(token != pad_id for token in next_packet.tokens):
                self._life_utterances += 1
            events = [str(event) for event in info.get("events", ())]
            self._life_consume_attempts += sum(
                event.startswith("consumed_") for event in events
            )
            self._life_resource_consumes += sum(
                event in {"consumed_food", "consumed_water"} for event in events
            )
            self._life_food_consumes += events.count("consumed_food")
            self._life_water_consumes += events.count("consumed_water")
            if bool(info.get("offered_consumed")):
                self._life_offered_consumes += 1
            if str(info.get("situation", "")).startswith("label|"):
                self._life_labels_received += 1
            self._life_harm_events += sum(
                event in {"hit_danger", "consumed_poison"} for event in events
            )
            if bool(info.get("semantic_choice_trial")):
                self._life_choice_need = str(info.get("choice_need") or "")
                chosen_kind = info.get("chosen_kind")
                if chosen_kind is not None:
                    self._life_chosen_kind = str(chosen_kind)
                    self._life_chosen_surface = str(
                        info.get("chosen_surface") or ""
                    )
                    self._life_chosen_surface_inspected = bool(
                        info.get("chosen_surface_inspected")
                    )
                    self._life_choice_correct = bool(info.get("correct"))
                    self._life_choice_poison = bool(info.get("poison"))
                    self._life_choice_wrong_resource = bool(
                        info.get("wrong_resource")
                    )
                if bool(info.get("semantic_choice_round_complete")):
                    self._life_choice_rounds_completed += 1
                    self._life_choice_correct_rounds += int(
                        bool(info.get("correct"))
                    )
                    self._life_choice_poison_rounds += int(
                        bool(info.get("poison"))
                    )
                    self._life_choice_wrong_resource_rounds += int(
                        bool(info.get("wrong_resource"))
                    )
                self._life_choice_timeout = bool(info.get("timeout"))

            if terminated or truncated:
                self._finish_life(survived=not terminated)
                bootstrap = 0.0
                return segment, initial_hidden, bootstrap
            self.packet = next_packet
        bootstrap = self._bootstrap_value()
        return segment, initial_hidden, bootstrap

    def _bootstrap_value(self) -> float:
        vector = mx.array(self.packet.vector()[None, None, :])
        tokens = mx.array(
            np.asarray(self.packet.tokens, dtype=np.int32)[None, None, :]
        )
        _, values, _ = self.model.policy_value(vector, tokens, self.hidden)
        mx.eval(values)
        return float(values[0, 0])

    def _finish_life(self, *, survived: bool) -> None:
        self.life_stats.append(
            LifeStats(
                life_index=self.life_index,
                seed=self.life_seed,
                steps=self._life_steps,
                survived=survived,
                mean_viability=self._life_viability_sum / max(1, self._life_steps),
                min_viability=self._life_min_viability,
                utterances_heard=self._life_utterances,
                consume_attempts=self._life_consume_attempts,
                resource_consumes=self._life_resource_consumes,
                food_consumes=self._life_food_consumes,
                water_consumes=self._life_water_consumes,
                offered_consumes=self._life_offered_consumes,
                option_decisions=self._life_option_decisions,
                inspect_option_decisions=self._life_inspect_option_decisions,
                labels_received=self._life_labels_received,
                harm_events=self._life_harm_events,
                semantic_choice_trial=self._life_semantic_choice_trial,
                choice_need=self._life_choice_need,
                chosen_kind=self._life_chosen_kind,
                chosen_surface=self._life_chosen_surface,
                chosen_surface_inspected=self._life_chosen_surface_inspected,
                choice_correct=self._life_choice_correct,
                choice_poison=self._life_choice_poison,
                choice_wrong_resource=self._life_choice_wrong_resource,
                choice_timeout=self._life_choice_timeout,
                choice_rounds_completed=self._life_choice_rounds_completed,
                choice_correct_rounds=self._life_choice_correct_rounds,
                choice_poison_rounds=self._life_choice_poison_rounds,
                choice_wrong_resource_rounds=(
                    self._life_choice_wrong_resource_rounds
                ),
            )
        )
        self.life_index += 1
        self.life_seed = self.config.seed + self.life_index
        self._start_next_life()

    # ------------------------------------------------------------ learning

    def _multi_step_prediction_loss(
        self,
        states: mx.array,
        vectors: mx.array,
        actions: mx.array,
        next_vectors: mx.array,
        next_needs: mx.array,
        env_rewards: mx.array,
        next_tokens: mx.array,
    ) -> mx.array:
        """Open-loop latent rollout loss over consecutive lived actions."""

        config = self.config
        horizon = config.multi_step_model_horizon
        decision_count = actions.shape[0]
        if horizon <= 1 or decision_count < horizon:
            return mx.array(0.0)
        usable = decision_count - horizon + 1
        imagined_state = states[:, :usable, :]
        imagined_vector = vectors[:, :usable, :]
        start_needs = imagined_vector[0, :, :4]
        predicted_needs = start_needs
        cumulative_reward = mx.zeros((1, usable))
        for offset in range(horizon):
            step_actions = actions[offset : offset + usable][None, :]
            imagined_state = self.model.transition_state(
                imagined_state, step_actions, imagined_vector
            )
            raw_vector, need_delta, reward, _ = self.model.decode_transition(
                imagined_state
            )
            predicted_needs = mx.clip(
                imagined_vector[:, :, :4] + need_delta, 0.0, 1.0
            )[0]
            imagined_vector = mx.concatenate(
                [predicted_needs[None, :, :], raw_vector[:, :, 4:]],
                axis=-1,
            )
            cumulative_reward = cumulative_reward + reward

        final_offset = horizon - 1
        target_vectors = next_vectors[final_offset : final_offset + usable]
        target_needs = next_needs[final_offset : final_offset + usable]
        target_reward = mx.zeros((usable,))
        for offset in range(horizon):
            target_reward = target_reward + env_rewards[offset : offset + usable]
        valid = mx.ones((usable,), dtype=mx.bool_)
        pad_id = TOKEN_TO_ID[PAD_TOKEN]
        for offset in range(horizon - 1):
            valid = valid & mx.all(
                next_tokens[offset : offset + usable] == pad_id,
                axis=-1,
            )
        valid_float = valid.astype(mx.float32)
        valid_count = mx.maximum(valid_float.sum(), mx.array(1.0))
        vector_error = ((imagined_vector[0] - target_vectors) ** 2).mean(
            axis=-1
        )
        vector_loss = (vector_error * valid_float).sum() / valid_count
        needs_loss = bodily_delta_prediction_loss(
            predicted_needs - start_needs,
            target_needs - start_needs,
            change_boost=config.bodily_change_loss_boost,
            drift_weight=config.bodily_drift_loss_weight,
            valid=valid_float,
        )
        reward_error = (cumulative_reward[0] - target_reward) ** 2
        reward_loss = (reward_error * valid_float).sum() / valid_count
        return (
            config.next_vector_weight * vector_loss
            + config.next_needs_weight * needs_loss
            + config.reward_prediction_weight * reward_loss
        )

    def _world_model_loss(
        self,
        vectors: mx.array,
        tokens: mx.array,
        hidden: mx.array | None,
        actions: mx.array,
        next_vectors: mx.array,
        next_needs: mx.array,
        env_rewards: mx.array,
        next_tokens: mx.array,
    ) -> mx.array:
        """Auxiliary-only loss, safe for off-policy episodic replay."""

        config = self.config
        states, _ = self.model.core_states(vectors, tokens, hidden)
        predicted_vectors, predicted_need_deltas, predicted_rewards, token_logits = (
            self.model.predict_consequences(states, actions[None, :], vectors)
        )
        vector_loss = ((predicted_vectors[0] - next_vectors) ** 2).mean()
        target_need_deltas = next_needs - vectors[0, :, :4]
        needs_loss = bodily_delta_prediction_loss(
            predicted_need_deltas[0],
            target_need_deltas,
            change_boost=config.bodily_change_loss_boost,
            drift_weight=config.bodily_drift_loss_weight,
        )
        reward_loss = ((predicted_rewards[0] - env_rewards) ** 2).mean()
        token_log_probabilities = token_logits[0] - mx.logsumexp(
            token_logits[0], axis=-1, keepdims=True
        )
        token_nll = -mx.take_along_axis(
            token_log_probabilities, next_tokens[..., None], axis=-1
        ).squeeze(-1)
        pad_id = TOKEN_TO_ID[PAD_TOKEN]
        token_weights = mx.where(
            next_tokens == pad_id, 1.0, config.nonpad_token_weight
        )
        token_loss = (token_nll * token_weights).sum() / token_weights.sum()
        multi_step_loss = self._multi_step_prediction_loss(
            states,
            vectors,
            actions,
            next_vectors,
            next_needs,
            env_rewards,
            next_tokens,
        )
        return (
            config.next_vector_weight * vector_loss
            + config.next_needs_weight * needs_loss
            + config.reward_prediction_weight * reward_loss
            + config.token_prediction_weight * token_loss
            + config.multi_step_model_weight * multi_step_loss
        )

    def _loss(
        self,
        vectors: mx.array,
        tokens: mx.array,
        hidden: mx.array | None,
        actions: mx.array,
        advantages: mx.array,
        returns: mx.array,
        planning_scales: mx.array,
        action_masks: mx.array,
        decision_weights: mx.array,
        semantic_choice_delayed: bool,
        next_vectors: mx.array,
        next_needs: mx.array,
        env_rewards: mx.array,
        next_tokens: mx.array,
    ) -> mx.array:
        config = self.config
        states, _ = self.model.core_states(vectors, tokens, hidden)
        logits = self.model.policy_logits(states, vectors)
        if config.self_model_planning_scale > 0.0:
            logits = planned_policy_logits(
                self.model,
                states,
                vectors,
                logits,
                planning_scales[None, :],
                reward_weight=config.self_model_planning_reward_weight,
                horizon=config.self_model_planning_horizon,
                force_return_after_inspect=semantic_choice_delayed,
                observation_branching=config.observation_branching_planning,
                persistent_information_reuses=(
                    config.persistent_information_reuses
                ),
                urgent_deficit_utility=config.urgent_deficit_utility,
                protocol_branch=config.protocol_branch_planning,
                persistent_low_need=config.island.semantic_choice_low_need,
            )
        logits = mx.where(action_masks[None, :, :], logits, -1e9)
        logits = logits[0]
        values = self.model.value(states).squeeze(-1)[0]
        log_probabilities = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        chosen = mx.take_along_axis(
            log_probabilities, actions[:, None], axis=-1
        ).squeeze(-1)
        decision_denominator = mx.maximum(
            decision_weights.sum(), mx.array(1.0)
        )
        policy_loss = -(
            advantages * chosen * decision_weights
        ).sum() / decision_denominator
        value_loss = ((values - returns) ** 2).mean()
        entropy_per_step = -(
            mx.softmax(logits, axis=-1) * log_probabilities
        ).sum(-1)
        entropy = (
            entropy_per_step * decision_weights
        ).sum() / decision_denominator

        predicted_vectors, predicted_need_deltas, predicted_rewards, token_logits = (
            self.model.predict_consequences(states, actions[None, :], vectors)
        )
        vector_loss = ((predicted_vectors[0] - next_vectors) ** 2).mean()
        target_need_deltas = next_needs - vectors[0, :, :4]
        needs_loss = bodily_delta_prediction_loss(
            predicted_need_deltas[0],
            target_need_deltas,
            change_boost=config.bodily_change_loss_boost,
            drift_weight=config.bodily_drift_loss_weight,
        )
        reward_loss = ((predicted_rewards[0] - env_rewards) ** 2).mean()
        token_log_probabilities = token_logits[0] - mx.logsumexp(
            token_logits[0], axis=-1, keepdims=True
        )
        token_nll = -mx.take_along_axis(
            token_log_probabilities, next_tokens[..., None], axis=-1
        ).squeeze(-1)
        pad_id = TOKEN_TO_ID[PAD_TOKEN]
        token_weights = mx.where(
            next_tokens == pad_id, 1.0, config.nonpad_token_weight
        )
        token_loss = (token_nll * token_weights).sum() / token_weights.sum()
        multi_step_loss = self._multi_step_prediction_loss(
            states,
            vectors,
            actions,
            next_vectors,
            next_needs,
            env_rewards,
            next_tokens,
        )

        return (
            policy_loss
            + config.value_weight * value_loss
            - config.entropy_weight * entropy
            + config.next_vector_weight * vector_loss
            + config.next_needs_weight * needs_loss
            + config.reward_prediction_weight * reward_loss
            + config.token_prediction_weight * token_loss
            + config.multi_step_model_weight * multi_step_loss
        )

    def _policy_targets(
        self, segment: _Segment, bootstrap: float
    ) -> tuple[np.ndarray, np.ndarray]:
        config = self.config
        values = np.asarray(segment.values, dtype=np.float32)
        rewards = np.asarray(segment.shaped_rewards, dtype=np.float32)
        dones = np.asarray(segment.dones, dtype=np.float32)
        durations = np.asarray(segment.durations, dtype=np.float32)
        advantages, returns = compute_gae(
            rewards,
            values,
            bootstrap,
            dones,
            durations,
            discount=config.discount,
            gae_lambda=config.gae_lambda,
        )
        # Centering within one short choice life makes every early information
        # action negative whenever the later consume has the larger return,
        # even when both are successful. Preserve raw cross-time return scale
        # in paired childhood; ordinary longer island segments keep the
        # variance-reducing normalization.
        if len(advantages) > 1 and not segment.semantic_choice_trial:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
        return advantages, returns

    def update(
        self, segment: _Segment, hidden: mx.array | None, bootstrap: float
    ) -> float:
        advantages, returns = self._policy_targets(segment, bootstrap)

        loss, grads = self._loss_and_grad(
            mx.array(np.stack(segment.vectors)[None, ...]),
            mx.array(np.asarray(segment.tokens, dtype=np.int32)[None, ...]),
            hidden,
            mx.array(np.asarray(segment.actions, dtype=np.int32)),
            mx.array(advantages),
            mx.array(returns),
            mx.array(np.asarray(segment.planning_scales, dtype=np.float32)),
            mx.array(np.stack(segment.action_masks)),
            mx.array(np.asarray(segment.decision_weights, dtype=np.float32)),
            segment.semantic_choice_delayed,
            mx.array(np.stack(segment.next_vectors)),
            mx.array(np.asarray(segment.next_needs, dtype=np.float32)),
            mx.array(np.asarray(segment.env_rewards, dtype=np.float32)),
            mx.array(np.asarray(segment.next_tokens, dtype=np.int32)),
        )
        grads, _ = optim.clip_grad_norm(grads, self.config.max_grad_norm)
        self.optimizer.update(self.model, grads)
        mx.eval(self.model.parameters(), self.optimizer.state, loss)
        return float(loss)

    def _remember_for_world_model_replay(
        self, segment: _Segment, hidden: mx.array | None
    ) -> None:
        capacity = self.config.world_model_replay_capacity
        if capacity <= 0:
            return
        hidden_copy: np.ndarray | None = None
        if hidden is not None:
            mx.eval(hidden)
            hidden_copy = np.asarray(hidden, dtype=np.float32).copy()
        item = (segment, hidden_copy)
        self._replay_segments_seen += 1
        if len(self.world_model_replay) < capacity:
            self.world_model_replay.append(item)
            return
        replacement = self.replay_rng.randrange(self._replay_segments_seen)
        if replacement < capacity:
            self.world_model_replay[replacement] = item

    def _world_model_replay_update(
        self, segment: _Segment, hidden: np.ndarray | None
    ) -> float:
        loss, grads = self._world_model_loss_and_grad(
            mx.array(np.stack(segment.vectors)[None, ...]),
            mx.array(np.asarray(segment.tokens, dtype=np.int32)[None, ...]),
            None if hidden is None else mx.array(hidden),
            mx.array(np.asarray(segment.actions, dtype=np.int32)),
            mx.array(np.stack(segment.next_vectors)),
            mx.array(np.asarray(segment.next_needs, dtype=np.float32)),
            mx.array(np.asarray(segment.env_rewards, dtype=np.float32)),
            mx.array(np.asarray(segment.next_tokens, dtype=np.int32)),
        )
        grads, _ = optim.clip_grad_norm(grads, self.config.max_grad_norm)
        self.optimizer.update(self.model, grads)
        mx.eval(self.model.parameters(), self.optimizer.state, loss)
        return float(loss)

    def replay_world_model(self) -> list[float]:
        """Update prediction/dynamics only from reservoir-sampled experience."""

        losses: list[float] = []
        for _ in range(self.config.world_model_replay_updates):
            if not self.world_model_replay:
                break
            segment, hidden = self.replay_rng.choice(self.world_model_replay)
            losses.append(self._world_model_replay_update(segment, hidden))
        return losses

    # -------------------------------------------------------------- driver

    def train(self) -> list[LifeStats]:
        config = self.config
        self.behavior_clone()
        steps_done = 0
        lives_logged = 0
        while steps_done < config.total_steps:
            remaining_steps = config.total_steps - steps_done
            if self.world.config.semantic_choice_trial:
                # Choice collection normally ends only at a trial boundary.
                # Shorten the final held-out training life so the declared
                # primitive-tick budget is exact even when it is not divisible
                # by the trial horizon or an option duration.
                choice_steps_left = remaining_steps
                if config.semantic_choice_childhood_steps > 0:
                    choice_steps_left = min(
                        choice_steps_left,
                        config.semantic_choice_childhood_steps
                        - self.global_steps,
                    )
                exact_limit = self.world.grid.step_count + choice_steps_left
                self.world.grid.max_steps = min(
                    self.world.grid.max_steps, exact_limit
                )
            segment, hidden, bootstrap = self.collect_segment()
            if len(segment) == 0:
                continue
            loss = self.update(segment, hidden, bootstrap)
            self._remember_for_world_model_replay(segment, hidden)
            replay_losses = self.replay_world_model()
            steps_done += sum(segment.durations)
            self.loss_log.append(
                {
                    "steps": float(steps_done),
                    "loss": loss,
                    "replay_loss": (
                        float(np.mean(replay_losses)) if replay_losses else 0.0
                    ),
                }
            )
            if (
                len(self.life_stats) >= lives_logged + config.log_every_lives
            ):
                recent = self.life_stats[lives_logged:]
                lives_logged = len(self.life_stats)
                survival = sum(life.survived for life in recent) / len(recent)
                mean_steps = sum(life.steps for life in recent) / len(recent)
                mean_viability = sum(life.mean_viability for life in recent) / len(
                    recent
                )
                print(
                    f"steps {steps_done}: lives {lives_logged}, "
                    f"recent survival {survival:.2f}, "
                    f"life steps {mean_steps:.1f}, viability {mean_viability:.3f}, "
                    f"loss {loss:.4f}"
                )
        if config.checkpoint:
            save_checkpoint(self.model, config, config.checkpoint)
        if config.stats_csv:
            write_life_stats(self.life_stats, config.stats_csv)
        return self.life_stats


def train_organism(config: OrganismConfig) -> tuple[OrganismModel, list[LifeStats]]:
    trainer = OrganismTrainer(config)
    stats = trainer.train()
    return trainer.model, stats


def evaluate_organism(
    model: OrganismModel,
    *,
    language_mode: str,
    episodes: int,
    base_seed: int,
    max_steps: int = 1000,
    sample_seed: int = 0,
    greedy: bool = False,
    consume_options: bool = False,
    inspect_options: bool = False,
    self_model_planning_scale: float = 0.0,
    self_model_planning_reward_weight: float = 0.5,
    self_model_planning_score_sign: float = 1.0,
    self_model_planning_horizon: int = 1,
) -> dict[str, float]:
    rng = Random(sample_seed)
    survived = 0
    viability_sum = 0.0
    min_viability_sum = 0.0
    steps_sum = 0
    harm_sum = 0
    consume_attempt_sum = 0
    resource_consume_sum = 0
    food_consume_sum = 0
    water_consume_sum = 0
    option_decision_sum = 0
    inspect_option_decision_sum = 0
    labels_received_sum = 0
    terminal_needs_sum = np.zeros(4, dtype=np.float64)
    death_causes = np.zeros(4, dtype=np.float64)
    for episode in range(episodes):
        seed = base_seed + episode
        world = IslandWorld(
            IslandConfig(
                language_mode=language_mode,
                max_steps=max_steps,
                max_visible_slots=model.visible_slots or 8,
            ),
            seed=seed,
        )
        packet = world.reset(seed)
        hidden: mx.array | None = None
        viability_running = 0.0
        min_viability = 1.0
        steps = 0
        while True:
            vector = mx.array(packet.vector()[None, None, :])
            tokens = mx.array(np.asarray(packet.tokens, dtype=np.int32)[None, None, :])
            states, hidden = model.core_states(vector, tokens, hidden)
            logits = model.policy_logits(states, vector)
            if self_model_planning_scale > 0.0:
                logits = planned_policy_logits(
                    model,
                    states,
                    vector,
                    logits,
                    mx.array([[self_model_planning_scale]]),
                    reward_weight=self_model_planning_reward_weight,
                    score_sign=self_model_planning_score_sign,
                    horizon=self_model_planning_horizon,
                )
            action_mask = available_action_mask(
                packet,
                model.action_size,
                visible_slots=model.visible_slots or None,
            )
            logits = mx.where(
                mx.array(action_mask)[None, None, :], logits, -1e9
            )
            mx.eval(logits, hidden)
            if greedy:
                action_index = int(mx.argmax(logits[0, 0]).item())
            else:
                probabilities = np.asarray(
                    mx.softmax(logits[0, 0], axis=-1), dtype=np.float64
                )
                probabilities = probabilities / probabilities.sum()
                action_index = int(
                    rng.choices(range(model.action_size), weights=probabilities)[0]
                )
            packet, _, terminated, truncated, info = execute_agent_action(
                world,
                packet,
                action_index,
                consume_options=consume_options,
                inspect_options=inspect_options,
            )
            option_decision_sum += int(action_index >= len(ACTIONS))
            decoded = decode_object_option(
                action_index,
                consume_options=consume_options,
                inspect_options=inspect_options,
                visible_slots=world.config.max_visible_slots,
            )
            inspect_option_decision_sum += int(
                decoded is not None and decoded[0] == "inspect"
            )
            labels_received_sum += int(
                str(info.get("situation", "")).startswith("label|")
            )
            duration = int(info["duration"])
            steps += duration
            viability_running += float(info["mean_viability_sum"])
            min_viability = min(min_viability, float(info["min_viability"]))
            events = [str(event) for event in info.get("events", ())]
            harm_sum += sum(
                event in {"hit_danger", "consumed_poison"} for event in events
            )
            consume_attempt_sum += sum(
                event.startswith("consumed_") for event in events
            )
            resource_consume_sum += sum(
                event in {"consumed_food", "consumed_water"} for event in events
            )
            food_consume_sum += events.count("consumed_food")
            water_consume_sum += events.count("consumed_water")
            if terminated or truncated:
                survived += int(truncated and not terminated)
                terminal = np.asarray(packet.needs, dtype=np.float64)
                terminal_needs_sum += terminal
                if terminated:
                    death_causes[int(np.argmin(terminal))] += 1
                break
        viability_sum += viability_running / max(1, steps)
        min_viability_sum += min_viability
        steps_sum += steps
    return {
        "survival_rate": survived / episodes,
        "mean_viability": viability_sum / episodes,
        "mean_min_viability": min_viability_sum / episodes,
        "mean_steps": steps_sum / episodes,
        "harm_events_per_episode": harm_sum / episodes,
        "consume_attempts_per_episode": consume_attempt_sum / episodes,
        "resource_consumes_per_episode": resource_consume_sum / episodes,
        "food_consumes_per_episode": food_consume_sum / episodes,
        "water_consumes_per_episode": water_consume_sum / episodes,
        "option_decisions_per_episode": option_decision_sum / episodes,
        "inspect_option_decisions_per_episode": (
            inspect_option_decision_sum / episodes
        ),
        "labels_received_per_episode": labels_received_sum / episodes,
        "terminal_food": terminal_needs_sum[0] / episodes,
        "terminal_water": terminal_needs_sum[1] / episodes,
        "terminal_energy": terminal_needs_sum[2] / episodes,
        "terminal_safety": terminal_needs_sum[3] / episodes,
        "food_death_rate": death_causes[0] / episodes,
        "water_death_rate": death_causes[1] / episodes,
        "energy_death_rate": death_causes[2] / episodes,
        "safety_death_rate": death_causes[3] / episodes,
    }


def _is_voluntary_inspection_event(info: dict[str, object]) -> bool:
    """Whether an embodied action actually elicited an object label."""

    return str(info.get("situation", "")).startswith("label|")


def evaluate_semantic_choice(
    model: OrganismModel,
    *,
    language_mode: str,
    episodes: int,
    base_seed: int,
    semantic_choice_horizon: int = 20,
    semantic_choice_objects: int = 2,
    semantic_choice_low_need: float = 0.35,
    semantic_choice_rounds: int = 1,
    semantic_choice_return_duration: int = 0,
    sample_seed: int = 0,
    greedy: bool = False,
    consume_options: bool = True,
    inspect_options: bool = True,
    self_model_planning_scale: float = 0.0,
    self_model_planning_reward_weight: float = 0.5,
    self_model_planning_score_sign: float = 1.0,
    self_model_planning_horizon: int = 1,
    observation_branching_planning: bool = False,
    persistent_information_reuses: int = 0,
    urgent_deficit_utility: bool = False,
    protocol_branch_planning: bool = False,
) -> dict[str, float]:
    """Evaluate the held-out inspect--remember--choose developmental task.

    Every episode independently remaps visible surfaces and presents one
    need-matched resource against poison.  The returned rates expose both
    coverage and conditional accuracy so timing out cannot masquerade as a
    competent selective policy.
    """

    if episodes <= 0:
        raise ValueError("episodes must be positive.")
    rng = Random(sample_seed)
    pad_id = TOKEN_TO_ID[PAD_TOKEN]
    choices = 0
    correct = 0
    poison = 0
    wrong_resource = 0
    timeouts = 0
    inspected_trials = 0
    inspected_choices = 0
    inspected_correct = 0
    inspect_decisions = 0
    label_opportunities = 0
    utterances = 0
    steps_sum = 0
    round_choices = [0] * semantic_choice_rounds
    round_correct = [0] * semantic_choice_rounds
    round_inspected = [0] * semantic_choice_rounds
    round_inspect_decisions = [0] * semantic_choice_rounds

    for episode in range(episodes):
        seed = base_seed + episode
        world = IslandWorld(
            IslandConfig(
                language_mode=language_mode,
                semantic_choice_trial=True,
                semantic_choice_horizon=semantic_choice_horizon,
                semantic_choice_objects=semantic_choice_objects,
                semantic_choice_low_need=semantic_choice_low_need,
                semantic_choice_rounds=semantic_choice_rounds,
                semantic_choice_return_duration=(
                    semantic_choice_return_duration
                ),
                max_visible_slots=model.visible_slots or 8,
            ),
            seed=seed,
        )
        packet = world.reset(seed)
        hidden: mx.array | None = None
        trial_inspected = False
        steps = 0
        while True:
            vector = mx.array(packet.vector()[None, None, :])
            tokens = mx.array(
                np.asarray(packet.tokens, dtype=np.int32)[None, None, :]
            )
            states, hidden = model.core_states(vector, tokens, hidden)
            logits = model.policy_logits(states, vector)
            return_pending = world.semantic_choice_return_pending
            round_pending = world.semantic_choice_round_pending
            delayed_choice = semantic_choice_return_duration > 0
            if (
                self_model_planning_scale > 0.0
                and not return_pending
                and not round_pending
            ):
                logits = planned_policy_logits(
                    model,
                    states,
                    vector,
                    logits,
                    mx.array([[self_model_planning_scale]]),
                    reward_weight=self_model_planning_reward_weight,
                    score_sign=self_model_planning_score_sign,
                    horizon=self_model_planning_horizon,
                    force_return_after_inspect=delayed_choice,
                    observation_branching=observation_branching_planning,
                    persistent_information_reuses=(
                        persistent_information_reuses
                    ),
                    urgent_deficit_utility=urgent_deficit_utility,
                    protocol_branch=protocol_branch_planning,
                    persistent_low_need=semantic_choice_low_need,
                )
            action_mask = available_action_mask(
                packet,
                model.action_size,
                visible_slots=model.visible_slots or None,
                semantic_choice_delayed=delayed_choice,
                return_pending=return_pending,
                round_pending=round_pending,
            )
            logits = mx.where(
                mx.array(action_mask)[None, None, :], logits, -1e9
            )
            mx.eval(logits, hidden)
            if return_pending or round_pending:
                action_index = ACTIONS.index(Action.WAIT)
            elif greedy:
                action_index = int(mx.argmax(logits[0, 0]).item())
            else:
                probabilities = np.asarray(
                    mx.softmax(logits[0, 0], axis=-1), dtype=np.float64
                )
                probabilities /= probabilities.sum()
                action_index = int(
                    rng.choices(
                        range(model.action_size), weights=probabilities
                    )[0]
                )
            decoded = decode_object_option(
                action_index,
                consume_options=consume_options,
                inspect_options=inspect_options,
                visible_slots=world.config.max_visible_slots,
            )
            if decoded is not None and decoded[0] == "inspect":
                inspect_decisions += 1
                round_inspect_decisions[
                    min(
                        world.grid.choice_round_index,
                        semantic_choice_rounds - 1,
                    )
                ] += 1
            packet, _, terminated, truncated, info = execute_agent_action(
                world,
                packet,
                action_index,
                consume_options=consume_options,
                inspect_options=inspect_options,
            )
            steps += int(info["duration"])
            inspection_event = _is_voluntary_inspection_event(info)
            trial_inspected = trial_inspected or inspection_event
            label_opportunities += int(inspection_event)
            utterances += int(any(token != pad_id for token in packet.tokens))
            if bool(info.get("semantic_choice_round_complete")):
                round_index = int(info.get("semantic_choice_round_index", 0))
                inspected_trials += int(trial_inspected)
                chosen = info.get("chosen_kind") is not None
                choices += int(chosen)
                correct += int(bool(info.get("correct")))
                poison += int(bool(info.get("poison")))
                wrong_resource += int(bool(info.get("wrong_resource")))
                round_choices[round_index] += int(chosen)
                round_correct[round_index] += int(bool(info.get("correct")))
                round_inspected[round_index] += int(trial_inspected)
                if chosen and bool(info.get("chosen_surface_inspected")):
                    inspected_choices += 1
                    inspected_correct += int(bool(info.get("correct")))
                trial_inspected = False
            if terminated or truncated:
                if not bool(info.get("semantic_choice_round_complete")):
                    inspected_trials += int(trial_inspected)
                    timeouts += int(bool(info.get("timeout")))
                break
        steps_sum += steps

    trials = episodes * semantic_choice_rounds
    result = {
        "trials": float(trials),
        "lives": float(episodes),
        "choices_made": float(choices),
        "choice_rate": choices / trials,
        "correct_choices": float(correct),
        "correct_rate_all_trials": correct / trials,
        "choice_accuracy": correct / max(1, choices),
        "poison_choices": float(poison),
        "poison_rate_all_trials": poison / trials,
        "wrong_resource_choices": float(wrong_resource),
        "wrong_resource_rate_all_trials": wrong_resource / trials,
        "timeout_rate": timeouts / trials,
        "trials_with_inspection": float(inspected_trials),
        "inspection_trial_rate": inspected_trials / trials,
        "inspected_choices": float(inspected_choices),
        "inspected_choice_rate": inspected_choices / trials,
        "inspected_choice_accuracy": inspected_correct
        / max(1, inspected_choices),
        "inspect_decisions_per_trial": inspect_decisions / trials,
        "inspect_option_decisions_per_trial": inspect_decisions / trials,
        "label_opportunities_per_trial": label_opportunities / trials,
        "utterances_heard_per_trial": utterances / trials,
        "mean_steps": steps_sum / episodes,
    }
    for round_index in range(semantic_choice_rounds):
        prefix = f"round_{round_index + 1}"
        result.update(
            {
                f"{prefix}_choice_rate": round_choices[round_index]
                / episodes,
                f"{prefix}_correct_rate": round_correct[round_index]
                / episodes,
                f"{prefix}_inspection_rate": round_inspected[round_index]
                / episodes,
                f"{prefix}_inspect_decisions_per_life": (
                    round_inspect_decisions[round_index] / episodes
                ),
            }
        )
    return result


def audit_self_model_actions(
    model: OrganismModel,
    *,
    language_mode: str,
    episodes: int,
    base_seed: int,
    max_decisions: int,
    max_steps: int = 1000,
    sample_seed: int = 0,
    consume_options: bool = False,
    inspect_options: bool = False,
    self_model_planning_scale: float = 0.0,
    reward_weight: float = 0.5,
    self_model_planning_horizon: int = 1,
    semantic_choice_trial: bool = False,
    semantic_choice_horizon: int = 20,
    semantic_choice_objects: int = 2,
    semantic_choice_low_need: float = 0.35,
    semantic_choice_return_duration: int = 0,
) -> dict[str, float]:
    """Compare predicted and simulator-realized outcomes for every action.

    The simulator copies are diagnostic counterfactuals only: they never enter
    training or acting. Centered score correlation and best-action regret test
    whether the head knows which available action will leave *this body* in a
    better state, rather than merely predicting the average next need.
    """

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if max_decisions <= 0:
        raise ValueError("max_decisions must be positive")

    rng = Random(sample_seed)
    audited_decisions = 0
    candidate_outcomes = 0
    needs_abs_error = 0.0
    reward_abs_error = 0.0
    chosen_needs_abs_error = 0.0
    chosen_reward_abs_error = 0.0
    chosen_score_abs_error = 0.0
    centered_dot = 0.0
    predicted_square = 0.0
    actual_square = 0.0
    best_action_hits = 0
    best_action_regret = 0.0
    best_over_worst_advantage = 0.0
    audited_episodes: set[int] = set()

    for episode in range(episodes):
        if audited_decisions >= max_decisions:
            break
        seed = base_seed + episode
        world = IslandWorld(
            IslandConfig(
                language_mode=language_mode,
                max_steps=max_steps,
                semantic_choice_trial=semantic_choice_trial,
                semantic_choice_horizon=semantic_choice_horizon,
                semantic_choice_objects=semantic_choice_objects,
                semantic_choice_low_need=semantic_choice_low_need,
                semantic_choice_return_duration=(
                    semantic_choice_return_duration
                ),
                max_visible_slots=model.visible_slots or 8,
            ),
            seed=seed,
        )
        packet = world.reset(seed)
        hidden: mx.array | None = None
        while audited_decisions < max_decisions:
            vector = mx.array(packet.vector()[None, None, :])
            tokens = mx.array(
                np.asarray(packet.tokens, dtype=np.int32)[None, None, :]
            )
            states, hidden = model.core_states(vector, tokens, hidden)
            predicted_needs_mx, predicted_rewards_mx = (
                predict_all_action_consequences(model, states, vector)
            )
            base_logits = model.policy_logits(states, vector)
            logits = base_logits
            if self_model_planning_scale > 0.0:
                logits = planned_policy_logits(
                    model,
                    states,
                    vector,
                    base_logits,
                    mx.array([[self_model_planning_scale]]),
                    reward_weight=reward_weight,
                    horizon=self_model_planning_horizon,
                    force_return_after_inspect=(
                        semantic_choice_return_duration > 0
                    ),
                )
            action_mask = available_action_mask(
                packet,
                model.action_size,
                visible_slots=model.visible_slots or None,
                semantic_choice_delayed=(
                    semantic_choice_return_duration > 0
                ),
                return_pending=world.semantic_choice_return_pending,
            )
            logits = mx.where(
                mx.array(action_mask)[None, None, :], logits, -1e9
            )
            mx.eval(predicted_needs_mx, predicted_rewards_mx, logits, hidden)
            predicted_needs = np.asarray(predicted_needs_mx[0, 0])
            predicted_rewards = np.asarray(predicted_rewards_mx[0, 0])
            available = np.flatnonzero(action_mask)
            probabilities = np.asarray(
                mx.softmax(logits[0, 0], axis=-1), dtype=np.float64
            )
            probabilities = probabilities / probabilities.sum()
            action_index = int(
                rng.choices(range(model.action_size), weights=probabilities)[0]
            )
            actual_needs: list[np.ndarray] = []
            actual_rewards: list[float] = []
            for candidate_action_index in available:
                branch = deepcopy(world)
                next_packet, reward, _, _, _ = execute_agent_action(
                    branch,
                    packet,
                    int(candidate_action_index),
                    consume_options=consume_options,
                    inspect_options=inspect_options,
                )
                actual_needs.append(
                    np.asarray(next_packet.needs, dtype=np.float64)
                )
                actual_rewards.append(float(reward))
            actual_needs_array = np.stack(actual_needs)
            actual_rewards_array = np.asarray(actual_rewards, dtype=np.float64)
            predicted_needs_available = predicted_needs[available]
            predicted_rewards_available = predicted_rewards[available]
            count = len(available)
            candidate_outcomes += count
            needs_abs_error += float(
                np.abs(predicted_needs_available - actual_needs_array).sum()
            )
            reward_abs_error += float(
                np.abs(predicted_rewards_available - actual_rewards_array).sum()
            )

            predicted_scores = (
                predicted_needs_available.min(axis=-1)
                + reward_weight * predicted_rewards_available
            )
            actual_scores = (
                actual_needs_array.min(axis=-1)
                + reward_weight * actual_rewards_array
            )
            chosen_position = int(np.flatnonzero(available == action_index)[0])
            chosen_needs_abs_error += float(
                np.abs(
                    predicted_needs_available[chosen_position]
                    - actual_needs_array[chosen_position]
                ).sum()
            )
            chosen_reward_abs_error += float(
                abs(
                    predicted_rewards_available[chosen_position]
                    - actual_rewards_array[chosen_position]
                )
            )
            chosen_score_abs_error += float(
                abs(
                    predicted_scores[chosen_position]
                    - actual_scores[chosen_position]
                )
            )
            predicted_centered = predicted_scores - predicted_scores.mean()
            actual_centered = actual_scores - actual_scores.mean()
            centered_dot += float(np.dot(predicted_centered, actual_centered))
            predicted_square += float(np.dot(predicted_centered, predicted_centered))
            actual_square += float(np.dot(actual_centered, actual_centered))
            predicted_best = int(np.argmax(predicted_scores))
            predicted_worst = int(np.argmin(predicted_scores))
            actual_best_score = float(actual_scores.max())
            best_action_hits += int(
                np.isclose(actual_scores[predicted_best], actual_best_score)
            )
            best_action_regret += (
                actual_best_score - float(actual_scores[predicted_best])
            )
            best_over_worst_advantage += float(
                actual_scores[predicted_best] - actual_scores[predicted_worst]
            )
            audited_decisions += 1
            audited_episodes.add(seed)

            if semantic_choice_trial:
                break
            packet, _, terminated, truncated, _ = execute_agent_action(
                world,
                packet,
                action_index,
                consume_options=consume_options,
                inspect_options=inspect_options,
            )
            if terminated or truncated:
                break

    correlation_denominator = (predicted_square * actual_square) ** 0.5
    return {
        "audited_decisions": float(audited_decisions),
        "unique_episodes": float(len(audited_episodes)),
        "candidate_outcomes": float(candidate_outcomes),
        "counterfactual_needs_mae": needs_abs_error
        / max(1, 4 * candidate_outcomes),
        "counterfactual_reward_mae": reward_abs_error
        / max(1, candidate_outcomes),
        "chosen_action_needs_mae": chosen_needs_abs_error
        / max(1, 4 * audited_decisions),
        "chosen_action_reward_mae": chosen_reward_abs_error
        / max(1, audited_decisions),
        "chosen_action_score_mae": chosen_score_abs_error
        / max(1, audited_decisions),
        "within_state_score_correlation": centered_dot
        / max(1e-12, correlation_denominator),
        "best_action_accuracy": best_action_hits / max(1, audited_decisions),
        "mean_best_action_regret": best_action_regret
        / max(1, audited_decisions),
        "predicted_best_over_worst_actual_advantage": best_over_worst_advantage
        / max(1, audited_decisions),
    }


def _directed_resource_swap_probability_shift(
    *,
    true_kind: str,
    choice_need: str,
    true_probability: float,
    swapped_probability: float,
) -> float:
    """Orient a food/water label swap toward the currently needed resource.

    If the inspected object's true resource is currently needed, its true
    label should increase consumption relative to the swapped label.  In a
    crossed body context the same semantic sensitivity has the opposite raw
    sign: replacing the wrong-resource label with the needed-resource label
    should increase consumption.
    """

    if true_kind not in {"food", "water"}:
        raise ValueError("true_kind must be food or water for a resource swap.")
    if choice_need not in {"food", "water"}:
        raise ValueError("choice_need must be food or water.")
    sign = 1.0 if true_kind == choice_need else -1.0
    return sign * (true_probability - swapped_probability)


def audit_label_to_self_model(
    model: OrganismModel,
    *,
    episodes: int,
    base_seed: int,
    max_inspections: int,
    max_steps: int = 400,
    sample_seed: int = 0,
    self_model_planning_scale: float = 0.0,
    self_model_planning_horizon: int = 2,
    semantic_choice_trial: bool = False,
    semantic_choice_horizon: int = 20,
    semantic_choice_objects: int = 2,
    semantic_choice_low_need: float = 0.35,
    semantic_choice_return_duration: int = 0,
) -> dict[str, float]:
    """Causally test whether an inspected label changes bodily prediction.

    Each audit branch executes the same kind-blind inspect option.  From the
    exact same pre-token recurrent state and resulting visual/body observation,
    the core then receives either a controlled in-distribution true-kind label,
    padding, or the same controlled label with its one functional-kind token
    changed counterfactually.  Simulator copies supply audit targets only and
    never enter learning or acting.
    """

    if episodes <= 0:
        raise ValueError("episodes must be positive.")
    if max_inspections <= 0:
        raise ValueError("max_inspections must be positive.")
    if not model.has_object_options or model.object_option_types < 2:
        raise ValueError("Label audit requires consume and inspect object options.")

    kind_words = ("food", "water", "danger")
    pad_tokens = (TOKEN_TO_ID[PAD_TOKEN],) * model.tokens_per_utterance
    rng = Random(sample_seed)

    audited = 0
    true_correct = 0
    silent_correct = 0
    counterfactual_correct = 0
    true_mae = 0.0
    silent_mae = 0.0
    counterfactual_mae = 0.0
    true_margin_over_silent = 0.0
    substituted_kind_shift = 0.0
    true_kind_suppression = 0.0
    prediction_l1_shift = 0.0
    consume_probability_direction = 0.0
    consume_probability_direction_hits = 0
    direct_policy_probability_direction = 0.0
    direct_policy_probability_direction_hits = 0
    planning_mediated_probability_direction = 0.0
    planning_mediated_probability_direction_hits = 0
    action_flip_count = 0
    resource_cases = 0
    danger_cases = 0
    resource_swap_cases = 0
    resource_swap_correct = 0
    resource_swap_substituted_shift = 0.0
    resource_swap_true_suppression = 0.0
    resource_swap_prediction_l1_shift = 0.0
    resource_swap_probability_direction = 0.0
    resource_swap_probability_direction_hits = 0
    resource_swap_direct_probability_direction = 0.0
    resource_swap_planner_increment = 0.0
    audited_episodes: set[int] = set()
    audited_target_keys: set[tuple[int, tuple[int, int], int]] = set()
    threeway_combo_counts = {
        (need, kind): 0
        for need in ("food", "water")
        for kind in kind_words
    }
    threeway_combo_correct = dict(threeway_combo_counts)
    resource_swap_combo_hits = {
        (need, kind): 0
        for need in ("food", "water")
        for kind in ("food", "water")
    }
    resource_swap_combo_shifts = dict(resource_swap_combo_hits)

    def semantic_scores(delta: np.ndarray) -> np.ndarray:
        return np.asarray([delta[0], delta[1], -delta[3]], dtype=np.float64)

    def counter_kind_for(true_kind: str, needs: tuple[float, ...]) -> str:
        if true_kind == "danger":
            return "food" if needs[0] <= needs[1] else "water"
        return "danger"

    def controlled_kind_tokens(kind: str) -> tuple[int, ...]:
        # ``water`` is both a surface word and a functional-kind word.  Using
        # the bank's sampled sentence and replacing every matching token would
        # sometimes change the named surface as well as the kind.  The
        # in-distribution "this <kind>" variants isolate one functional token
        # while the post-inspect visual packet keeps surface identity fixed.
        return encode_utterance(("this", kind))

    def measure_variant(
        label_packet: ObsPacket,
        hidden: mx.array,
        tokens: tuple[int, ...],
        query_packet: ObsPacket,
        consume_action: int,
    ) -> tuple[np.ndarray, float, float, int, int]:
        label_vector = mx.array(label_packet.vector()[None, None, :])
        token_array = mx.array(np.asarray(tokens, dtype=np.int32)[None, None, :])
        states, carry = model.core_states(label_vector, token_array, hidden)
        vector = label_vector
        if query_packet is not label_packet:
            vector = mx.array(query_packet.vector()[None, None, :])
            query_tokens = mx.array(
                np.asarray(query_packet.tokens, dtype=np.int32)[None, None, :]
            )
            states, carry = model.core_states(vector, query_tokens, carry)
        _, predicted_delta, _, _ = model.predict_consequences(
            states,
            mx.array(np.asarray([[consume_action]], dtype=np.int32)),
            vector,
        )
        base_logits = model.policy_logits(states, vector)
        planned_logits = base_logits
        if self_model_planning_scale > 0.0:
            planned_logits = planned_policy_logits(
                model,
                states,
                vector,
                base_logits,
                mx.array([[self_model_planning_scale]]),
                reward_weight=0.0,
                horizon=self_model_planning_horizon,
                force_return_after_inspect=(
                    semantic_choice_return_duration > 0
                ),
            )
        action_mask = available_action_mask(
            query_packet,
            model.action_size,
            visible_slots=model.visible_slots,
            semantic_choice_delayed=(
                semantic_choice_return_duration > 0
            ),
            return_pending=False,
        )
        mask = mx.array(action_mask)[None, None, :]
        base_logits = mx.where(mask, base_logits, -1e9)
        planned_logits = mx.where(mask, planned_logits, -1e9)
        base_probabilities = mx.softmax(base_logits[0, 0], axis=-1)
        planned_probabilities = mx.softmax(planned_logits[0, 0], axis=-1)
        mx.eval(predicted_delta, base_probabilities, planned_probabilities)
        return (
            np.asarray(predicted_delta[0, 0], dtype=np.float64),
            float(base_probabilities[consume_action].item()),
            float(planned_probabilities[consume_action].item()),
            int(mx.argmax(base_probabilities).item()),
            int(mx.argmax(planned_probabilities).item()),
        )

    for episode in range(episodes):
        if audited >= max_inspections:
            break
        seed = base_seed + episode
        world = IslandWorld(
            IslandConfig(
                language_mode="grounded",
                max_steps=max_steps,
                semantic_choice_trial=semantic_choice_trial,
                semantic_choice_horizon=semantic_choice_horizon,
                semantic_choice_objects=semantic_choice_objects,
                semantic_choice_low_need=semantic_choice_low_need,
                semantic_choice_return_duration=(
                    semantic_choice_return_duration
                ),
                max_visible_slots=model.visible_slots or 8,
            ),
            seed=seed,
        )
        packet = world.reset(seed)
        hidden: mx.array | None = None
        while audited < max_inspections:
            vector = mx.array(packet.vector()[None, None, :])
            token_array = mx.array(
                np.asarray(packet.tokens, dtype=np.int32)[None, None, :]
            )
            states, hidden = model.core_states(vector, token_array, hidden)
            mx.eval(states, hidden)

            candidate_slots = [
                slot
                for slot, (_, _, surface_index) in enumerate(packet.visible)
                if SURFACES[surface_index] in CONSUMABLE_SURFACES
            ]
            if semantic_choice_trial:
                # One independent target per held-out trial. The two-object
                # task stays exactly resource/danger balanced; the crossed
                # task cycles food/water/danger so every functional word is
                # tested under body contexts where it is and is not needed.
                if semantic_choice_objects == 2:
                    want_danger = episode % 2 == 1
                    target_slot = next(
                        (
                            slot
                            for slot in candidate_slots
                            if (
                                (
                                    world.grid.object_at(
                                        (
                                            packet.position[0]
                                            + packet.visible[slot][0],
                                            packet.position[1]
                                            + packet.visible[slot][1],
                                        )
                                    ).kind
                                    == "poison"
                                )
                                == want_danger
                            )
                        ),
                        None,
                    )
                else:
                    body_need = str(world.grid.choice_need)
                    per_combo_limit = (
                        max_inspections // 6
                        if max_inspections % 6 == 0
                        else None
                    )
                    candidate_kinds = [
                        kind
                        for kind in kind_words
                        if per_combo_limit is None
                        or threeway_combo_counts[(body_need, kind)]
                        < per_combo_limit
                    ]
                    if candidate_kinds:
                        desired_word = min(
                            candidate_kinds,
                            key=lambda kind: (
                                threeway_combo_counts[(body_need, kind)],
                                kind_words.index(kind),
                            ),
                        )
                        desired_kind = (
                            "poison" if desired_word == "danger" else desired_word
                        )
                        target_slot = next(
                            (
                                slot
                                for slot in candidate_slots
                                if world.grid.object_at(
                                    (
                                        packet.position[0]
                                        + packet.visible[slot][0],
                                        packet.position[1]
                                        + packet.visible[slot][1],
                                    )
                                ).kind
                                == desired_kind
                            ),
                            None,
                        )
                    else:
                        target_slot = None
            else:
                target_slot = next(
                    (
                        slot
                        for slot in candidate_slots
                        if (
                            seed,
                            (
                                packet.position[0] + packet.visible[slot][0],
                                packet.position[1] + packet.visible[slot][1],
                            ),
                            packet.visible[slot][2],
                        )
                        not in audited_target_keys
                    ),
                    None,
                )
            if target_slot is not None:
                dx, dy, surface_index = packet.visible[target_slot]
                target_pos = (packet.position[0] + dx, packet.position[1] + dy)
                inspect_action = object_option_action_index(
                    "inspect",
                    target_slot,
                    consume_options=True,
                    inspect_options=True,
                    visible_slots=model.visible_slots,
                )
                inspected_world = deepcopy(world)
                inspected_packet, _, terminated, truncated, inspect_info = (
                    execute_agent_action(
                        inspected_world,
                        packet,
                        inspect_action,
                        consume_options=True,
                        inspect_options=True,
                    )
                )
                post_slot = next(
                    (
                        slot
                        for slot, (post_dx, post_dy, post_surface) in enumerate(
                            inspected_packet.visible
                        )
                        if (
                            inspected_packet.position[0] + post_dx,
                            inspected_packet.position[1] + post_dy,
                        )
                        == target_pos
                        and post_surface == surface_index
                    ),
                    None,
                )
                situation = str(inspect_info.get("situation", ""))
                if (
                    not terminated
                    and not truncated
                    and post_slot is not None
                    and situation.startswith("label|")
                ):
                    target = inspected_world.grid.object_at(target_pos)
                    if target is not None:
                        true_kind = "danger" if target.kind == "poison" else target.kind
                        label_situation = Situation.from_key(situation)
                        if (
                            true_kind in kind_words
                            and label_situation.slot("surface") == target.name
                            and label_situation.slot("kind") == true_kind
                        ):
                            query_world = inspected_world
                            query_packet = inspected_packet
                            query_slot = post_slot
                            if semantic_choice_return_duration > 0:
                                query_world = deepcopy(inspected_world)
                                (
                                    query_packet,
                                    _,
                                    return_terminated,
                                    return_truncated,
                                    _,
                                ) = execute_agent_action(
                                    query_world,
                                    inspected_packet,
                                    ACTIONS.index(Action.WAIT),
                                    consume_options=True,
                                    inspect_options=True,
                                )
                                if return_terminated or return_truncated:
                                    continue
                                query_slot = next(
                                    (
                                        slot
                                        for slot, (
                                            query_dx,
                                            query_dy,
                                            query_surface,
                                        ) in enumerate(query_packet.visible)
                                        if (
                                            query_packet.position[0] + query_dx,
                                            query_packet.position[1] + query_dy,
                                        )
                                        == target_pos
                                        and query_surface == surface_index
                                    ),
                                    None,
                                )
                                if query_slot is None:
                                    continue
                            consume_action = object_option_action_index(
                                "consume",
                                query_slot,
                                consume_options=True,
                                inspect_options=True,
                                visible_slots=model.visible_slots,
                            )
                            counter_kind = counter_kind_for(
                                true_kind, inspected_packet.needs
                            )
                            true_tokens = controlled_kind_tokens(true_kind)
                            counter_tokens = controlled_kind_tokens(counter_kind)
                            (
                                true_delta,
                                true_base_probability,
                                true_probability,
                                _,
                                true_action,
                            ) = measure_variant(
                                inspected_packet,
                                hidden,
                                true_tokens,
                                query_packet,
                                consume_action,
                            )
                            (
                                silent_delta,
                                _,
                                _,
                                _,
                                _,
                            ) = measure_variant(
                                inspected_packet,
                                hidden,
                                pad_tokens,
                                query_packet,
                                consume_action,
                            )
                            (
                                counter_delta,
                                counter_base_probability,
                                counter_probability,
                                _,
                                counter_action,
                            ) = measure_variant(
                                inspected_packet,
                                hidden,
                                counter_tokens,
                                query_packet,
                                consume_action,
                            )
                            resource_swap_measure = None
                            if true_kind in {"food", "water"}:
                                resource_swap_kind = (
                                    "water" if true_kind == "food" else "food"
                                )
                                resource_swap_measure = (
                                    resource_swap_kind,
                                    *measure_variant(
                                        inspected_packet,
                                        hidden,
                                        controlled_kind_tokens(
                                            resource_swap_kind
                                        ),
                                        query_packet,
                                        consume_action,
                                    ),
                                )

                            outcome_world = deepcopy(query_world)
                            actual_packet, _, _, _, _ = execute_agent_action(
                                outcome_world,
                                query_packet,
                                consume_action,
                                consume_options=True,
                                inspect_options=True,
                            )
                            actual_delta = np.asarray(
                                actual_packet.needs, dtype=np.float64
                            ) - np.asarray(query_packet.needs, dtype=np.float64)
                            true_scores = semantic_scores(true_delta)
                            silent_scores = semantic_scores(silent_delta)
                            counter_scores = semantic_scores(counter_delta)
                            true_index = kind_words.index(true_kind)
                            counter_index = kind_words.index(counter_kind)

                            audited += 1
                            audited_episodes.add(seed)
                            audited_target_keys.add(
                                (seed, target_pos, surface_index)
                            )
                            resource_cases += int(true_kind != "danger")
                            danger_cases += int(true_kind == "danger")
                            true_is_correct = int(np.argmax(true_scores)) == true_index
                            true_correct += int(true_is_correct)
                            if semantic_choice_objects == 3:
                                combo = (str(world.grid.choice_need), true_kind)
                                threeway_combo_counts[combo] += 1
                                threeway_combo_correct[combo] += int(true_is_correct)
                            silent_correct += int(
                                int(np.argmax(silent_scores)) == true_index
                            )
                            counterfactual_correct += int(
                                int(np.argmax(counter_scores)) == counter_index
                            )
                            true_mae += float(np.abs(true_delta - actual_delta).mean())
                            silent_mae += float(
                                np.abs(silent_delta - actual_delta).mean()
                            )
                            counterfactual_mae += float(
                                np.abs(counter_delta - actual_delta).mean()
                            )
                            true_margin_over_silent += float(
                                true_scores[true_index] - silent_scores[true_index]
                            )
                            substituted_kind_shift += float(
                                counter_scores[counter_index]
                                - true_scores[counter_index]
                            )
                            true_kind_suppression += float(
                                true_scores[true_index] - counter_scores[true_index]
                            )
                            prediction_l1_shift += float(
                                np.abs(counter_delta - true_delta).mean()
                            )
                            direction = 1.0 if counter_kind != "danger" else -1.0
                            directed_probability_shift = direction * (
                                counter_probability - true_probability
                            )
                            direct_probability_shift = direction * (
                                counter_base_probability
                                - true_base_probability
                            )
                            mediated_probability_shift = (
                                directed_probability_shift
                                - direct_probability_shift
                            )
                            consume_probability_direction += directed_probability_shift
                            consume_probability_direction_hits += int(
                                directed_probability_shift > 1e-6
                            )
                            direct_policy_probability_direction += (
                                direct_probability_shift
                            )
                            direct_policy_probability_direction_hits += int(
                                direct_probability_shift > 1e-6
                            )
                            planning_mediated_probability_direction += (
                                mediated_probability_shift
                            )
                            planning_mediated_probability_direction_hits += int(
                                mediated_probability_shift > 1e-6
                            )
                            action_flip_count += int(counter_action != true_action)
                            if resource_swap_measure is not None:
                                (
                                    resource_swap_kind,
                                    resource_swap_delta,
                                    resource_swap_base_probability,
                                    resource_swap_probability,
                                    _,
                                    _,
                                ) = resource_swap_measure
                                resource_swap_scores = semantic_scores(
                                    resource_swap_delta
                                )
                                resource_swap_index = kind_words.index(
                                    resource_swap_kind
                                )
                                resource_swap_cases += 1
                                resource_swap_correct += int(
                                    int(np.argmax(resource_swap_scores))
                                    == resource_swap_index
                                )
                                resource_swap_substituted_shift += float(
                                    resource_swap_scores[resource_swap_index]
                                    - true_scores[resource_swap_index]
                                )
                                resource_swap_true_suppression += float(
                                    true_scores[true_index]
                                    - resource_swap_scores[true_index]
                                )
                                resource_swap_prediction_l1_shift += float(
                                    np.abs(
                                        resource_swap_delta - true_delta
                                    ).mean()
                                )
                                choice_need = str(world.grid.choice_need)
                                resource_probability_shift = (
                                    _directed_resource_swap_probability_shift(
                                        true_kind=true_kind,
                                        choice_need=choice_need,
                                        true_probability=true_probability,
                                        swapped_probability=(
                                            resource_swap_probability
                                        ),
                                    )
                                )
                                resource_direct_shift = (
                                    _directed_resource_swap_probability_shift(
                                        true_kind=true_kind,
                                        choice_need=choice_need,
                                        true_probability=(
                                            true_base_probability
                                        ),
                                        swapped_probability=(
                                            resource_swap_base_probability
                                        ),
                                    )
                                )
                                resource_swap_probability_direction += (
                                    resource_probability_shift
                                )
                                resource_swap_probability_direction_hits += int(
                                    resource_probability_shift > 1e-6
                                )
                                if semantic_choice_objects == 3:
                                    resource_combo = (choice_need, true_kind)
                                    resource_swap_combo_hits[resource_combo] += int(
                                        resource_probability_shift > 1e-6
                                    )
                                    resource_swap_combo_shifts[resource_combo] += (
                                        resource_probability_shift
                                    )
                                resource_swap_direct_probability_direction += (
                                    resource_direct_shift
                                )
                                resource_swap_planner_increment += (
                                    resource_probability_shift
                                    - resource_direct_shift
                                )

            if semantic_choice_trial:
                break
            logits = model.policy_logits(states, vector)
            if self_model_planning_scale > 0.0:
                logits = planned_policy_logits(
                    model,
                    states,
                    vector,
                    logits,
                    mx.array([[self_model_planning_scale]]),
                    reward_weight=0.0,
                    horizon=self_model_planning_horizon,
                )
            action_mask = available_action_mask(
                packet,
                model.action_size,
                visible_slots=model.visible_slots,
            )
            logits = mx.where(mx.array(action_mask)[None, None, :], logits, -1e9)
            mx.eval(logits, hidden)
            probabilities = np.asarray(
                mx.softmax(logits[0, 0], axis=-1), dtype=np.float64
            )
            probabilities /= probabilities.sum()
            action_index = int(
                rng.choices(range(model.action_size), weights=probabilities)[0]
            )
            packet, _, terminated, truncated, _ = execute_agent_action(
                world,
                packet,
                action_index,
                consume_options=True,
                inspect_options=True,
            )
            if terminated or truncated:
                break

    denominator = max(1, audited)
    result = {
        "audited_inspections": float(audited),
        "unique_episodes": float(len(audited_episodes)),
        "unique_target_contexts": float(len(audited_target_keys)),
        "resource_cases": float(resource_cases),
        "danger_cases": float(danger_cases),
        "resource_swap_cases": float(resource_swap_cases),
        "true_label_kind_accuracy": true_correct / denominator,
        "silent_kind_accuracy": silent_correct / denominator,
        "counterfactual_label_kind_accuracy": counterfactual_correct / denominator,
        "true_label_consequence_mae": true_mae / denominator,
        "silent_consequence_mae": silent_mae / denominator,
        "counterfactual_consequence_mae": counterfactual_mae / denominator,
        "true_kind_score_gain_over_silence": true_margin_over_silent / denominator,
        "substituted_kind_score_shift": substituted_kind_shift / denominator,
        "true_kind_score_suppression": true_kind_suppression / denominator,
        "prediction_l1_shift_under_kind_substitution": (
            prediction_l1_shift / denominator
        ),
        "directed_consume_probability_shift": (
            consume_probability_direction / denominator
        ),
        "directed_consume_probability_hit_rate": (
            consume_probability_direction_hits / denominator
        ),
        "direct_policy_consume_probability_shift": (
            direct_policy_probability_direction / denominator
        ),
        "direct_policy_consume_probability_hit_rate": (
            direct_policy_probability_direction_hits / denominator
        ),
        "incremental_planner_enabled_consume_probability_shift": (
            planning_mediated_probability_direction / denominator
        ),
        "incremental_planner_enabled_consume_probability_hit_rate": (
            planning_mediated_probability_direction_hits / denominator
        ),
        "resource_swap_counterfactual_kind_accuracy": (
            resource_swap_correct / max(1, resource_swap_cases)
        ),
        "resource_swap_substituted_kind_score_shift": (
            resource_swap_substituted_shift / max(1, resource_swap_cases)
        ),
        "resource_swap_true_kind_score_suppression": (
            resource_swap_true_suppression / max(1, resource_swap_cases)
        ),
        "resource_swap_prediction_l1_shift": (
            resource_swap_prediction_l1_shift / max(1, resource_swap_cases)
        ),
        "resource_swap_directed_consume_probability_shift": (
            resource_swap_probability_direction / max(1, resource_swap_cases)
        ),
        "resource_swap_directed_consume_probability_hit_rate": (
            resource_swap_probability_direction_hits
            / max(1, resource_swap_cases)
        ),
        "resource_swap_direct_policy_probability_shift": (
            resource_swap_direct_probability_direction
            / max(1, resource_swap_cases)
        ),
        "resource_swap_incremental_planner_enabled_probability_shift": (
            resource_swap_planner_increment / max(1, resource_swap_cases)
        ),
        "argmax_action_flip_rate": action_flip_count / denominator,
    }
    if semantic_choice_objects == 3:
        for need in ("food", "water"):
            for kind in kind_words:
                count = threeway_combo_counts[(need, kind)]
                prefix = f"true_label_{kind}_in_{need}_body"
                result[f"{prefix}_cases"] = float(count)
                result[f"{prefix}_accuracy"] = (
                    threeway_combo_correct[(need, kind)] / max(1, count)
                )
                if kind in {"food", "water"}:
                    resource_prefix = (
                        f"resource_swap_{kind}_in_{need}_body"
                    )
                    result[f"{resource_prefix}_cases"] = float(count)
                    result[f"{resource_prefix}_directed_probability_shift"] = (
                        resource_swap_combo_shifts[(need, kind)] / max(1, count)
                    )
                    result[f"{resource_prefix}_directed_hit_rate"] = (
                        resource_swap_combo_hits[(need, kind)] / max(1, count)
                    )
    return result


def audit_label_referent_binding(
    model: OrganismModel,
    *,
    episodes: int,
    base_seed: int,
    max_contexts: int,
    semantic_choice_horizon: int = 24,
    semantic_choice_objects: int = 2,
    semantic_choice_low_need: float = 0.35,
    semantic_choice_return_duration: int = 0,
) -> dict[str, float]:
    """Test whether a delayed label effect stays attached to its referent.

    One object is inspected through the ordinary kind-blind option, then the
    agent is moved back to the center through real padding-token observations.
    With every choice object visible again, the learned consequence model is
    queried separately for the labeled target and a balanced unlabeled object.
    For memory models, the primary audit-only erasure and reassignment act on
    the post-label carry and then replay the identical padding return packet(s),
    testing the total delayed external-memory pathway. Secondary final-state
    edits hold the already memory-influenced recurrent core fixed and isolate
    the selected binding's controlled direct effect on the transition head.
    Simulator kinds select balanced targets and score predictions only; they
    never enter acting or learning.
    """

    if episodes <= 0:
        raise ValueError("episodes must be positive.")
    if max_contexts <= 0:
        raise ValueError("max_contexts must be positive.")
    if semantic_choice_objects not in {2, 3}:
        raise ValueError("semantic_choice_objects must be 2 or 3.")
    if not model.has_object_options or model.object_option_types < 2:
        raise ValueError("Referent audit requires consume and inspect options.")

    kind_words = ("food", "water", "danger")
    audited = 0
    target_correct = 0
    nonreferent_correct = 0
    joint_correct = 0
    same_kind = 0
    prediction_separation = 0.0
    erased_target_correct = 0
    erased_target_score_drop = 0.0
    key_reassignment_hits = 0
    key_reassignment_l1 = 0.0
    direct_erased_target_correct = 0
    direct_erased_target_score_drop = 0.0
    direct_key_reassignment_hits = 0
    direct_key_reassignment_l1 = 0.0

    def class_kind(kind: str) -> str:
        return "danger" if kind == "poison" else kind

    def semantic_scores(delta: np.ndarray) -> np.ndarray:
        return np.asarray([delta[0], delta[1], -delta[3]], dtype=np.float64)

    def predict_pair(
        states: mx.array,
        vector: mx.array,
        target_action: int,
        other_action: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        expanded_states = mx.broadcast_to(
            states, (1, 2, states.shape[-1])
        )
        expanded_vectors = mx.broadcast_to(
            vector, (1, 2, vector.shape[-1])
        )
        _, predicted_delta, _, _ = model.predict_consequences(
            expanded_states,
            mx.array([[target_action, other_action]], dtype=mx.int32),
            expanded_vectors,
        )
        mx.eval(predicted_delta)
        return (
            np.asarray(predicted_delta[0, 0], dtype=np.float64),
            np.asarray(predicted_delta[0, 1], dtype=np.float64),
        )

    def replay_packets(
        carry: mx.array,
        packets: list[ObsPacket],
    ) -> tuple[mx.array, mx.array, mx.array]:
        states: mx.array | None = None
        vector: mx.array | None = None
        for replay_packet in packets:
            vector = mx.array(replay_packet.vector()[None, None, :])
            tokens = mx.array(
                np.asarray(replay_packet.tokens, dtype=np.int32)[None, None, :]
            )
            states, carry = model.core_states(vector, tokens, carry)
        if states is None or vector is None:
            raise ValueError("Referent audit requires at least one return packet.")
        return states, carry, vector

    def edited_memory_carry(
        carry: mx.array,
        *,
        target_surface: int,
        other_surface: int,
        erase: bool,
    ) -> mx.array:
        carry_np = np.asarray(carry, dtype=np.float32).copy()
        memory_start = model.hidden_size
        memory_stop = memory_start + model.binding_bank_size
        memory = carry_np[..., memory_start:memory_stop].reshape(
            1, model.surface_count, model.episodic_binding_size
        )
        valid = carry_np[
            ..., memory_stop : memory_stop + model.binding_valid_size
        ]
        if erase:
            memory[...] = 0.0
            valid[...] = 0.0
        else:
            target_value = memory[..., target_surface, :].copy()
            target_valid = valid[..., target_surface].copy()
            other_value = memory[..., other_surface, :].copy()
            other_valid = valid[..., other_surface].copy()
            memory[..., target_surface, :] = other_value
            valid[..., target_surface] = other_valid
            memory[..., other_surface, :] = target_value
            valid[..., other_surface] = target_valid
        return mx.array(carry_np)

    for episode in range(episodes):
        if audited >= max_contexts:
            break
        seed = base_seed + episode
        world = IslandWorld(
            IslandConfig(
                language_mode="grounded",
                semantic_choice_trial=True,
                semantic_choice_horizon=semantic_choice_horizon,
                semantic_choice_objects=semantic_choice_objects,
                semantic_choice_low_need=semantic_choice_low_need,
                semantic_choice_return_duration=(
                    semantic_choice_return_duration
                ),
                max_visible_slots=model.visible_slots or 8,
            ),
            seed=seed,
        )
        packet = world.reset(seed)
        initial_vector = mx.array(packet.vector()[None, None, :])
        initial_tokens = mx.array(
            np.asarray(packet.tokens, dtype=np.int32)[None, None, :]
        )
        _, hidden = model.core_states(initial_vector, initial_tokens, None)
        mx.eval(hidden)

        if semantic_choice_objects == 2:
            want_danger = episode % 2 == 1
            target = next(
                obj for obj in world.grid.objects if (obj.kind == "poison") == want_danger
            )
            other = next(obj for obj in world.grid.objects if obj.name != target.name)
        else:
            target_kind = ("food", "water", "poison")[episode % 3]
            other_kind = {"food": "water", "water": "poison", "poison": "food"}[
                target_kind
            ]
            target = next(obj for obj in world.grid.objects if obj.kind == target_kind)
            other = next(obj for obj in world.grid.objects if obj.kind == other_kind)

        target_slot = next(
            (
                slot
                for slot, (_, _, surface) in enumerate(packet.visible)
                if surface == SURFACE_INDEX[target.name]
            ),
            None,
        )
        if target_slot is None:
            continue
        inspect_action = object_option_action_index(
            "inspect",
            target_slot,
            consume_options=True,
            inspect_options=True,
            visible_slots=model.visible_slots,
        )
        inspected_packet, _, terminated, truncated, inspect_info = execute_agent_action(
            world,
            packet,
            inspect_action,
            consume_options=True,
            inspect_options=True,
        )
        if terminated or truncated or not str(inspect_info.get("situation", "")).startswith(
            "label|"
        ):
            continue
        target_word = class_kind(target.kind)
        inspected_packet = replace(
            inspected_packet,
            tokens=encode_utterance(("this", target_word)),
        )
        vector = mx.array(inspected_packet.vector()[None, None, :])
        tokens = mx.array(
            np.asarray(inspected_packet.tokens, dtype=np.int32)[None, None, :]
        )
        _, post_label_carry = model.core_states(vector, tokens, hidden)
        mx.eval(post_label_carry)

        center = world.semantic_choice_center
        return_packets: list[ObsPacket] = []
        if semantic_choice_return_duration > 0:
            (
                inspected_packet,
                _,
                terminated,
                truncated,
                _,
            ) = execute_agent_action(
                world,
                inspected_packet,
                ACTIONS.index(Action.WAIT),
                consume_options=True,
                inspect_options=True,
            )
            return_packets.append(inspected_packet)
        else:
            while world.grid.agent_pos != center and not (terminated or truncated):
                desired = _desired_direction(world.grid.agent_pos, center)
                action = (
                    Action.MOVE_FORWARD
                    if world.grid.direction == desired
                    else _turn_toward(world.grid.direction, desired)
                )
                inspected_packet, _, terminated, truncated, _ = world.step(action)
                return_packets.append(inspected_packet)
        if terminated or truncated or world.grid.agent_pos != center:
            continue
        states, hidden, vector = replay_packets(post_label_carry, return_packets)
        mx.eval(states, hidden)

        target_slot = next(
            (
                slot
                for slot, (_, _, surface) in enumerate(inspected_packet.visible)
                if surface == SURFACE_INDEX[target.name]
            ),
            None,
        )
        other_slot = next(
            (
                slot
                for slot, (_, _, surface) in enumerate(inspected_packet.visible)
                if surface == SURFACE_INDEX[other.name]
            ),
            None,
        )
        if target_slot is None or other_slot is None:
            continue
        target_action = object_option_action_index(
            "consume",
            target_slot,
            consume_options=True,
            inspect_options=True,
            visible_slots=model.visible_slots,
        )
        other_action = object_option_action_index(
            "consume",
            other_slot,
            consume_options=True,
            inspect_options=True,
            visible_slots=model.visible_slots,
        )
        target_delta, other_delta = predict_pair(
            states, vector, target_action, other_action
        )
        target_scores = semantic_scores(target_delta)
        other_scores = semantic_scores(other_delta)
        target_index = kind_words.index(target_word)
        other_index = kind_words.index(class_kind(other.kind))
        target_prediction = int(np.argmax(target_scores))
        other_prediction = int(np.argmax(other_scores))

        audited += 1
        target_is_correct = target_prediction == target_index
        other_is_correct = other_prediction == other_index
        target_correct += int(target_is_correct)
        nonreferent_correct += int(other_is_correct)
        joint_correct += int(target_is_correct and other_is_correct)
        same_kind += int(target_prediction == other_prediction)
        prediction_separation += float(np.abs(target_delta - other_delta).mean())

        if model.has_episodic_bindings:
            target_surface = SURFACE_INDEX[target.name]
            other_surface = SURFACE_INDEX[other.name]

            # Primary total-path intervention: edit immediately after the
            # label write, then replay the exact same padding observations.
            erased_carry = edited_memory_carry(
                post_label_carry,
                target_surface=target_surface,
                other_surface=other_surface,
                erase=True,
            )
            erased_states, _, _ = replay_packets(erased_carry, return_packets)
            erased_target, _ = predict_pair(
                erased_states, vector, target_action, other_action
            )
            erased_scores = semantic_scores(erased_target)
            erased_target_correct += int(int(np.argmax(erased_scores)) == target_index)
            erased_target_score_drop += float(
                target_scores[target_index] - erased_scores[target_index]
            )

            swapped_carry = edited_memory_carry(
                post_label_carry,
                target_surface=target_surface,
                other_surface=other_surface,
                erase=False,
            )
            swapped_states, _, _ = replay_packets(swapped_carry, return_packets)
            swapped_target, swapped_other = predict_pair(
                swapped_states, vector, target_action, other_action
            )
            swapped_target_scores = semantic_scores(swapped_target)
            swapped_other_scores = semantic_scores(swapped_other)
            target_drop = target_scores[target_index] - swapped_target_scores[target_index]
            other_gain = swapped_other_scores[target_index] - other_scores[target_index]
            key_reassignment_hits += int(target_drop > 1e-6 and other_gain > 1e-6)
            key_reassignment_l1 += 0.5 * float(
                np.abs(swapped_target - target_delta).mean()
                + np.abs(swapped_other - other_delta).mean()
            )

            # Secondary controlled direct effect: edit only the final bank
            # while holding the recurrent core (and all prior reads) fixed.
            state_np = np.asarray(states, dtype=np.float32).copy()
            memory_start = model.hidden_size
            memory_stop = memory_start + model.binding_bank_size
            memory = state_np[..., memory_start:memory_stop].reshape(
                1, 1, model.surface_count, model.episodic_binding_size
            )
            valid = state_np[
                ...,
                memory_stop : memory_stop + model.binding_valid_size,
            ]
            erased_np = state_np.copy()
            erased_np[..., memory_start:memory_stop] = 0.0
            erased_np[
                ...,
                memory_stop : memory_stop + model.binding_valid_size,
            ] = 0.0
            erased_target, _ = predict_pair(
                mx.array(erased_np), vector, target_action, other_action
            )
            erased_scores = semantic_scores(erased_target)
            direct_erased_target_correct += int(
                int(np.argmax(erased_scores)) == target_index
            )
            direct_erased_target_score_drop += float(
                target_scores[target_index] - erased_scores[target_index]
            )

            swapped_np = state_np.copy()
            swapped_memory = swapped_np[..., memory_start:memory_stop].reshape(
                1, 1, model.surface_count, model.episodic_binding_size
            )
            swapped_valid = swapped_np[
                ...,
                memory_stop : memory_stop + model.binding_valid_size,
            ]
            target_value = memory[..., target_surface, :].copy()
            target_valid = valid[..., target_surface].copy()
            other_value = memory[..., other_surface, :].copy()
            other_valid = valid[..., other_surface].copy()
            swapped_memory[..., target_surface, :] = other_value
            swapped_valid[..., target_surface] = other_valid
            swapped_memory[..., other_surface, :] = target_value
            swapped_valid[..., other_surface] = target_valid
            swapped_target, swapped_other = predict_pair(
                mx.array(swapped_np), vector, target_action, other_action
            )
            swapped_target_scores = semantic_scores(swapped_target)
            swapped_other_scores = semantic_scores(swapped_other)
            target_drop = target_scores[target_index] - swapped_target_scores[target_index]
            other_gain = swapped_other_scores[target_index] - other_scores[target_index]
            direct_key_reassignment_hits += int(
                target_drop > 1e-6 and other_gain > 1e-6
            )
            direct_key_reassignment_l1 += 0.5 * float(
                np.abs(swapped_target - target_delta).mean()
                + np.abs(swapped_other - other_delta).mean()
            )

    denominator = max(1, audited)
    return {
        "audited_contexts": float(audited),
        "labeled_target_kind_accuracy": target_correct / denominator,
        "unlabeled_object_kind_accuracy": nonreferent_correct / denominator,
        "joint_two_object_kind_accuracy": joint_correct / denominator,
        "same_kind_broadcast_rate": same_kind / denominator,
        "target_nonreferent_prediction_l1": prediction_separation / denominator,
        "episodic_memory_available": float(model.has_episodic_bindings),
        "episodic_writes_enabled": float(
            model.has_episodic_bindings and model.episodic_binding_writes
        ),
        "memory_erased_target_kind_accuracy": (
            erased_target_correct / denominator
            if model.has_episodic_bindings
            else 0.0
        ),
        "target_kind_score_drop_under_memory_erasure": (
            erased_target_score_drop / denominator
            if model.has_episodic_bindings
            else 0.0
        ),
        "key_reassignment_directional_hit_rate": (
            key_reassignment_hits / denominator
            if model.has_episodic_bindings
            else 0.0
        ),
        "key_reassignment_prediction_l1_shift": (
            key_reassignment_l1 / denominator
            if model.has_episodic_bindings
            else 0.0
        ),
        "final_state_controlled_direct_erased_target_kind_accuracy": (
            direct_erased_target_correct / denominator
            if model.has_episodic_bindings
            else 0.0
        ),
        "final_state_controlled_direct_target_kind_score_drop": (
            direct_erased_target_score_drop / denominator
            if model.has_episodic_bindings
            else 0.0
        ),
        "final_state_controlled_direct_key_reassignment_hit_rate": (
            direct_key_reassignment_hits / denominator
            if model.has_episodic_bindings
            else 0.0
        ),
        "final_state_controlled_direct_key_reassignment_l1_shift": (
            direct_key_reassignment_l1 / denominator
            if model.has_episodic_bindings
            else 0.0
        ),
    }


def audit_observation_branching_planner(
    model: OrganismModel,
    *,
    episodes: int = 300,
    base_seed: int = 1_700_000,
    semantic_choice_horizon: int = 40,
    semantic_choice_low_need: float = 0.55,
    semantic_choice_return_duration: int = 6,
    persistent_information_reuses: int = 0,
    urgent_deficit_utility: bool = False,
    protocol_branch: bool = False,
) -> dict[str, float]:
    """Read-only feasibility audit for the delayed observation backup."""

    if episodes <= 0:
        raise ValueError("episodes must be positive.")
    if not model.has_episodic_bindings or model.object_option_types != 2:
        raise ValueError(
            "The observation-branching audit requires episodic consume/inspect "
            "options."
        )

    condition_totals = {
        name: {
            "inspect_options": 0.0,
            "contingent": 0.0,
            "matching_selected": 0.0,
            "danger_avoided": 0.0,
            "advantage_sum": 0.0,
            "positive_contexts": 0.0,
        }
        for name in ("intact", "no_write", "collapsed")
    }
    entropy_sum = 0.0
    true_probability_sum = 0.0
    true_branch_correct = 0.0
    true_branch_cases = 0.0
    audited_contexts = 0
    writes_before = model.episodic_binding_writes

    def planner_outputs(
        states: mx.array,
        vector: mx.array,
        *,
        writes_enabled: bool,
        collapsed_labels: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        model.episodic_binding_writes = writes_enabled
        values, probabilities, _, choices = (
            observation_branching_inspect_values(
                model,
                states,
                vector,
                collapsed_labels=collapsed_labels,
                persistent_information_reuses=(
                    persistent_information_reuses
                ),
                persistent_low_need=semantic_choice_low_need,
                urgent_deficit_utility=urgent_deficit_utility,
                protocol_branch=protocol_branch,
            )
        )
        mx.eval(values, probabilities, choices)
        return (
            np.asarray(values[0, 0], dtype=np.float64),
            np.asarray(probabilities[0, 0], dtype=np.float64),
            np.asarray(choices[0, 0], dtype=np.int64),
        )

    try:
        for episode in range(episodes):
            seed = base_seed + episode
            world = IslandWorld(
                IslandConfig(
                    semantic_choice_trial=True,
                    semantic_choice_horizon=semantic_choice_horizon,
                    semantic_choice_objects=3,
                    semantic_choice_low_need=semantic_choice_low_need,
                    semantic_choice_return_duration=(
                        semantic_choice_return_duration
                    ),
                    language_mode="grounded",
                ),
                seed=seed,
            )
            packet = world.reset(seed)
            if len(packet.visible) != 3:
                raise RuntimeError(
                    "Three-way feasibility audit did not expose three objects."
                )
            vector = mx.array(packet.vector()[None, None, :])
            tokens = mx.array(
                np.asarray(packet.tokens, dtype=np.int32)[None, None, :]
            )
            states, _ = model.core_states(vector, tokens)
            immediate_scores = _terminal_consume_scores(
                model,
                states,
                vector,
                urgent_deficit_utility=urgent_deficit_utility,
            )
            if persistent_information_reuses > 0:
                continuation = _persistent_self_context_value(
                    model,
                    states,
                    vector,
                    low_need=semantic_choice_low_need,
                    urgent_deficit_utility=urgent_deficit_utility,
                    settling_steps=(
                        PROTOCOL_SETTLING_OBSERVATIONS
                        if protocol_branch
                        else 1
                    ),
                )
                immediate_scores = immediate_scores + (
                    persistent_information_reuses
                    * continuation[..., None]
                )
            mx.eval(immediate_scores)
            visible_count = len(packet.visible)
            best_immediate = float(
                np.max(
                    np.asarray(
                        immediate_scores[0, 0, :visible_count],
                        dtype=np.float64,
                    )
                )
            )

            outputs = {
                "intact": planner_outputs(
                    states,
                    vector,
                    writes_enabled=True,
                    collapsed_labels=False,
                ),
                "no_write": planner_outputs(
                    states,
                    vector,
                    writes_enabled=False,
                    collapsed_labels=False,
                ),
                "collapsed": planner_outputs(
                    states,
                    vector,
                    writes_enabled=True,
                    collapsed_labels=True,
                ),
            }

            choice_need = str(world.grid.choice_need)
            matching_branch = SEMANTIC_CHOICE_LABEL_KINDS.index(choice_need)
            danger_branch = SEMANTIC_CHOICE_LABEL_KINDS.index("danger")
            correct_slot = next(
                slot
                for slot, (_, _, surface_index) in enumerate(packet.visible)
                if world.grid.kind_by_surface[SURFACES[surface_index]]
                == choice_need
            )

            for condition, (values, probabilities, choices) in outputs.items():
                values = values[:visible_count]
                probabilities = probabilities[:visible_count]
                choices = choices[:visible_count]
                totals = condition_totals[condition]
                totals["inspect_options"] += visible_count
                totals["contingent"] += sum(
                    len(set(row.tolist())) >= 2 for row in choices
                )
                totals["matching_selected"] += sum(
                    int(choices[slot, matching_branch] == slot)
                    for slot in range(visible_count)
                )
                totals["danger_avoided"] += sum(
                    int(choices[slot, danger_branch] != slot)
                    for slot in range(visible_count)
                )
                advantage = float(np.max(values) - best_immediate)
                totals["advantage_sum"] += advantage
                totals["positive_contexts"] += int(advantage > 0.0)

            intact_probabilities = outputs["intact"][1][:visible_count]
            entropy_sum += float(
                (
                    -intact_probabilities
                    * np.log(np.clip(intact_probabilities, 1e-12, 1.0))
                ).sum()
                / np.log(len(SEMANTIC_CHOICE_LABEL_KINDS))
            )
            intact_choices = outputs["intact"][2][:visible_count]
            for slot, (_, _, surface_index) in enumerate(packet.visible):
                true_kind = world.grid.kind_by_surface[
                    SURFACES[surface_index]
                ]
                true_word = "danger" if true_kind == "poison" else true_kind
                true_branch = SEMANTIC_CHOICE_LABEL_KINDS.index(true_word)
                true_probability_sum += intact_probabilities[slot, true_branch]
                true_branch_correct += int(
                    intact_choices[slot, true_branch] == correct_slot
                )
                true_branch_cases += 1.0
            audited_contexts += 1
    finally:
        model.episodic_binding_writes = writes_before

    result: dict[str, float] = {
        "audited_contexts": float(audited_contexts),
        "urgent_deficit_utility": float(urgent_deficit_utility),
        "protocol_branch": float(protocol_branch),
        "persistent_information_reuses": float(
            persistent_information_reuses
        ),
        "candidate_prior_normalized_entropy": (
            entropy_sum / max(1.0, true_branch_cases)
        ),
        "mean_true_label_probability": (
            true_probability_sum / max(1.0, true_branch_cases)
        ),
        "true_label_branch_correct_consume_rate": (
            true_branch_correct / max(1.0, true_branch_cases)
        ),
    }
    for condition, totals in condition_totals.items():
        inspect_denominator = max(1.0, totals["inspect_options"])
        context_denominator = max(1.0, float(audited_contexts))
        result.update(
            {
                f"{condition}_branch_contingency_rate": (
                    totals["contingent"] / inspect_denominator
                ),
                f"{condition}_matching_label_selects_target_rate": (
                    totals["matching_selected"] / inspect_denominator
                ),
                f"{condition}_danger_label_avoids_target_rate": (
                    totals["danger_avoided"] / inspect_denominator
                ),
                f"{condition}_mean_best_inspect_advantage": (
                    totals["advantage_sum"] / context_denominator
                ),
                f"{condition}_positive_advantage_context_rate": (
                    totals["positive_contexts"] / context_denominator
                ),
            }
        )
    result.update(
        {
            "write_causal_advantage_drop": (
                result["intact_mean_best_inspect_advantage"]
                - result["no_write_mean_best_inspect_advantage"]
            ),
            "write_causal_contingency_drop": (
                result["intact_branch_contingency_rate"]
                - result["no_write_branch_contingency_rate"]
            ),
            "collapsed_label_advantage_drop": (
                result["intact_mean_best_inspect_advantage"]
                - result["collapsed_mean_best_inspect_advantage"]
            ),
            "collapsed_label_contingency_drop": (
                result["intact_branch_contingency_rate"]
                - result["collapsed_branch_contingency_rate"]
            ),
        }
    )
    return result


def audit_semantic_choice_information_upper_bound(
    *,
    episodes: int = 10_000,
    base_seed: int = 1_800_000,
    semantic_choice_horizon: int = 40,
    semantic_choice_low_need: float = 0.55,
    semantic_choice_return_duration: int = 6,
) -> dict[str, float]:
    """Exact-environment information-rent ceiling for the delayed task."""

    if episodes <= 0:
        raise ValueError("episodes must be positive.")

    def action_for_surface(
        packet: ObsPacket,
        surface_index: int,
        option_kind: str,
    ) -> int:
        slot = next(
            slot
            for slot, (_, _, visible_surface) in enumerate(packet.visible)
            if visible_surface == surface_index
        )
        return object_option_action_index(
            option_kind,
            slot,
            consume_options=True,
            inspect_options=True,
            visible_slots=packet.max_visible_slots,
        )

    def inspect_and_return(
        world: IslandWorld,
        packet: ObsPacket,
        surface_index: int,
    ) -> tuple[ObsPacket, str]:
        packet, _, terminated, truncated, _ = execute_agent_action(
            world,
            packet,
            action_for_surface(packet, surface_index, "inspect"),
            consume_options=True,
            inspect_options=True,
        )
        if terminated or truncated:
            raise RuntimeError("Information upper-bound inspection ended early.")
        observed_kind = VOCAB[packet.tokens[1]]
        if observed_kind not in SEMANTIC_CHOICE_LABEL_KINDS:
            raise RuntimeError(
                f"Unexpected semantic-choice label: {observed_kind!r}."
            )
        packet, _, terminated, truncated, _ = execute_agent_action(
            world,
            packet,
            ACTIONS.index(Action.WAIT),
            consume_options=True,
            inspect_options=True,
        )
        if terminated or truncated:
            raise RuntimeError("Information upper-bound return ended early.")
        return packet, observed_kind

    def consume_surface(
        world: IslandWorld,
        packet: ObsPacket,
        surface_index: int,
    ) -> tuple[ObsPacket, dict[str, object]]:
        packet, _, terminated, truncated, info = execute_agent_action(
            world,
            packet,
            action_for_surface(packet, surface_index, "consume"),
            consume_options=True,
            inspect_options=True,
        )
        if not (terminated or truncated):
            raise RuntimeError(
                "Delayed semantic-choice consumption was not terminal."
            )
        return packet, info

    policy_names = (
        "blind_immediate",
        "one_inspection",
        "two_inspections",
        "clairvoyant_immediate",
    )
    utilities = {name: [] for name in policy_names}
    ticks = {name: [] for name in policy_names}
    outcomes = {
        name: {"correct": 0, "wrong_resource": 0, "poison": 0}
        for name in policy_names
    }

    for episode in range(episodes):
        seed = base_seed + episode
        for policy_name in policy_names:
            world = IslandWorld(
                IslandConfig(
                    semantic_choice_trial=True,
                    semantic_choice_horizon=semantic_choice_horizon,
                    semantic_choice_objects=3,
                    semantic_choice_low_need=semantic_choice_low_need,
                    semantic_choice_return_duration=(
                        semantic_choice_return_duration
                    ),
                    language_mode="grounded",
                ),
                seed=seed,
            )
            packet = world.reset(seed)
            surfaces = [surface for _, _, surface in packet.visible]
            if len(surfaces) != 3:
                raise RuntimeError(
                    "Information upper bound requires three visible surfaces."
                )
            low_need = "food" if packet.needs[0] < packet.needs[1] else "water"

            if policy_name == "blind_immediate":
                chosen_surface = surfaces[0]
            elif policy_name == "one_inspection":
                packet, first_label = inspect_and_return(
                    world,
                    packet,
                    surfaces[0],
                )
                chosen_surface = (
                    surfaces[0] if first_label == low_need else surfaces[1]
                )
            elif policy_name == "two_inspections":
                packet, first_label = inspect_and_return(
                    world,
                    packet,
                    surfaces[0],
                )
                if first_label == low_need:
                    chosen_surface = surfaces[0]
                else:
                    packet, second_label = inspect_and_return(
                        world,
                        packet,
                        surfaces[1],
                    )
                    chosen_surface = (
                        surfaces[1]
                        if second_label == low_need
                        else surfaces[2]
                    )
            else:
                chosen_surface = next(
                    surface
                    for surface in surfaces
                    if world.grid.kind_by_surface[SURFACES[surface]]
                    == low_need
                )

            packet, info = consume_surface(
                world,
                packet,
                chosen_surface,
            )
            utilities[policy_name].append(min(packet.needs))
            ticks[policy_name].append(world.grid.step_count)
            outcomes[policy_name]["correct"] += int(bool(info["correct"]))
            outcomes[policy_name]["wrong_resource"] += int(
                bool(info["wrong_resource"])
            )
            outcomes[policy_name]["poison"] += int(bool(info["poison"]))

    result: dict[str, float] = {"audited_contexts": float(episodes)}
    blind = np.asarray(utilities["blind_immediate"], dtype=np.float64)
    for policy_name in policy_names:
        policy_utilities = np.asarray(
            utilities[policy_name],
            dtype=np.float64,
        )
        result.update(
            {
                f"{policy_name}_mean_final_min_need": float(
                    policy_utilities.mean()
                ),
                f"{policy_name}_mean_ticks": float(
                    np.mean(ticks[policy_name])
                ),
                f"{policy_name}_correct_rate": (
                    outcomes[policy_name]["correct"] / episodes
                ),
                f"{policy_name}_wrong_resource_rate": (
                    outcomes[policy_name]["wrong_resource"] / episodes
                ),
                f"{policy_name}_poison_rate": (
                    outcomes[policy_name]["poison"] / episodes
                ),
                f"{policy_name}_paired_utility_gain_over_blind": float(
                    (policy_utilities - blind).mean()
                ),
            }
        )
    return result


def audit_persistent_mapping_information_rent(
    *,
    lives: int = 2_500,
    base_seed: int = 1_900_000,
    round_counts: tuple[int, ...] = (1, 2, 4, 8),
    semantic_choice_horizon: int = 40,
    semantic_choice_low_need: float = 0.55,
    semantic_choice_return_duration: int = 6,
) -> dict[str, float]:
    """Exact-dynamics rent audit with one mapping reused across rounds."""

    if lives <= 0:
        raise ValueError("lives must be positive.")
    if not round_counts or any(rounds <= 0 for rounds in round_counts):
        raise ValueError("round_counts must contain positive values.")

    def action_for_surface(
        packet: ObsPacket,
        surface_index: int,
        option_kind: str,
    ) -> int:
        slot = next(
            slot
            for slot, (_, _, visible_surface) in enumerate(packet.visible)
            if visible_surface == surface_index
        )
        return object_option_action_index(
            option_kind,
            slot,
            consume_options=True,
            inspect_options=True,
            visible_slots=packet.max_visible_slots,
        )

    def inspect_and_return(
        world: IslandWorld,
        packet: ObsPacket,
        surface_index: int,
    ) -> tuple[ObsPacket, str]:
        packet, _, terminated, truncated, _ = execute_agent_action(
            world,
            packet,
            action_for_surface(packet, surface_index, "inspect"),
            consume_options=True,
            inspect_options=True,
        )
        if terminated or truncated:
            raise RuntimeError("Persistent-rent inspection ended early.")
        observed_kind = VOCAB[packet.tokens[1]]
        packet, _, terminated, truncated, _ = execute_agent_action(
            world,
            packet,
            ACTIONS.index(Action.WAIT),
            consume_options=True,
            inspect_options=True,
        )
        if terminated or truncated:
            raise RuntimeError("Persistent-rent return ended early.")
        return packet, observed_kind

    def prepare_round(
        world: IslandWorld,
        fixed_objects: list[object],
        fixed_surfaces: list[int],
        low_need: str,
    ) -> ObsPacket:
        world.grid.objects = deepcopy(fixed_objects)
        world.grid.agent_pos = world.semantic_choice_center
        world.grid.direction = Direction.NORTH
        world.grid.step_count = 0
        world.grid.needs = replace(
            world.grid.needs,
            food=(
                semantic_choice_low_need if low_need == "food" else 0.75
            ),
            water=(
                semantic_choice_low_need if low_need == "water" else 0.75
            ),
            energy=0.75,
            safety=0.75,
        )
        world.grid.choice_need = low_need
        world.grid.choice_surfaces = tuple(
            SURFACES[surface] for surface in fixed_surfaces
        )
        world.grid.offered_kind = None
        world.grid.offered_surface = None
        world.grid.offered_pos = None
        world._last_action_index = -1
        world._semantic_choice_return_pending = False
        world._inspected_surfaces.clear()
        return world._packet(world.grid._observe(None, None), None)

    policy_names = ("blind", "persistent", "clairvoyant")
    result: dict[str, float] = {"audited_lives": float(lives)}
    for rounds in round_counts:
        life_utilities = {name: [] for name in policy_names}
        total_ticks = {name: 0.0 for name in policy_names}
        total_inspections = {name: 0.0 for name in policy_names}
        outcomes = {
            name: {"correct": 0, "wrong_resource": 0, "poison": 0}
            for name in policy_names
        }

        for life in range(lives):
            seed = base_seed + life
            starting_need = Random(seed ^ 0x5E1F).choice(("food", "water"))
            need_sequence = [
                (
                    starting_need
                    if round_index % 2 == 0
                    else ("water" if starting_need == "food" else "food")
                )
                for round_index in range(rounds)
            ]
            for policy_name in policy_names:
                world = IslandWorld(
                    IslandConfig(
                        semantic_choice_trial=True,
                        semantic_choice_horizon=semantic_choice_horizon,
                        semantic_choice_objects=3,
                        semantic_choice_low_need=semantic_choice_low_need,
                        semantic_choice_return_duration=(
                            semantic_choice_return_duration
                        ),
                        language_mode="grounded",
                    ),
                    seed=seed,
                )
                initial_packet = world.reset(seed)
                fixed_surfaces = [
                    surface for _, _, surface in initial_packet.visible
                ]
                fixed_objects = deepcopy(world.grid.objects)
                known_labels: dict[int, str] = {}
                life_utility = 0.0

                for low_need in need_sequence:
                    packet = prepare_round(
                        world,
                        fixed_objects,
                        fixed_surfaces,
                        low_need,
                    )
                    if policy_name == "blind":
                        chosen_surface = fixed_surfaces[0]
                    elif policy_name == "clairvoyant":
                        chosen_surface = next(
                            surface
                            for surface in fixed_surfaces
                            if world.grid.kind_by_surface[SURFACES[surface]]
                            == low_need
                        )
                    else:
                        while not any(
                            label == low_need
                            for label in known_labels.values()
                        ):
                            unknown = next(
                                surface
                                for surface in fixed_surfaces
                                if surface not in known_labels
                            )
                            packet, label = inspect_and_return(
                                world,
                                packet,
                                unknown,
                            )
                            known_labels[unknown] = label
                            total_inspections[policy_name] += 1.0
                            if len(known_labels) == 2:
                                remaining_surface = next(
                                    surface
                                    for surface in fixed_surfaces
                                    if surface not in known_labels
                                )
                                remaining_label = next(
                                    label_name
                                    for label_name in (
                                        SEMANTIC_CHOICE_LABEL_KINDS
                                    )
                                    if label_name
                                    not in set(known_labels.values())
                                )
                                known_labels[remaining_surface] = (
                                    remaining_label
                                )
                        chosen_surface = next(
                            surface
                            for surface, label in known_labels.items()
                            if label == low_need
                        )

                    packet, _, terminated, truncated, info = execute_agent_action(
                        world,
                        packet,
                        action_for_surface(
                            packet,
                            chosen_surface,
                            "consume",
                        ),
                        consume_options=True,
                        inspect_options=True,
                    )
                    if not (terminated or truncated):
                        raise RuntimeError(
                            "Persistent-rent consumption was not terminal."
                        )
                    round_utility = min(packet.needs)
                    life_utility += round_utility
                    total_ticks[policy_name] += world.grid.step_count
                    outcomes[policy_name]["correct"] += int(
                        bool(info["correct"])
                    )
                    outcomes[policy_name]["wrong_resource"] += int(
                        bool(info["wrong_resource"])
                    )
                    outcomes[policy_name]["poison"] += int(
                        bool(info["poison"])
                    )
                life_utilities[policy_name].append(life_utility)

        blind = np.asarray(life_utilities["blind"], dtype=np.float64)
        total_rounds = lives * rounds
        prefix = f"rounds_{rounds}"
        for policy_name in policy_names:
            policy_life_utilities = np.asarray(
                life_utilities[policy_name],
                dtype=np.float64,
            )
            paired_gain = policy_life_utilities - blind
            result.update(
                {
                    f"{prefix}_{policy_name}_mean_final_min_need_per_round": (
                        float(policy_life_utilities.mean() / rounds)
                    ),
                    f"{prefix}_{policy_name}_paired_cumulative_gain": (
                        float(paired_gain.mean())
                    ),
                    f"{prefix}_{policy_name}_paired_gain_per_round": (
                        float(paired_gain.mean() / rounds)
                    ),
                    f"{prefix}_{policy_name}_correct_rate": (
                        outcomes[policy_name]["correct"] / total_rounds
                    ),
                    f"{prefix}_{policy_name}_wrong_resource_rate": (
                        outcomes[policy_name]["wrong_resource"] / total_rounds
                    ),
                    f"{prefix}_{policy_name}_poison_rate": (
                        outcomes[policy_name]["poison"] / total_rounds
                    ),
                    f"{prefix}_{policy_name}_mean_ticks_per_round": (
                        total_ticks[policy_name] / total_rounds
                    ),
                    f"{prefix}_{policy_name}_mean_inspections_per_life": (
                        total_inspections[policy_name] / lives
                    ),
                }
            )
    return result


def audit_cross_round_label_reuse(
    model: OrganismModel,
    *,
    lives: int = 300,
    base_seed: int = 1_700_000,
    rounds: int = 8,
    acquisition_rounds: int = 2,
    semantic_choice_horizon: int = 40,
    semantic_choice_low_need: float = 0.55,
    semantic_choice_return_duration: int = 6,
    language_mode: str = "grounded",
    urgent_deficit_utility: bool = True,
) -> dict[str, float]:
    """Does a word acquired early still steer choice in later rounds?

    A scripted read-only driver acquires one food label and one water label in
    the first rounds and never inspects again. In every later round the
    organism's own terminal consume score is read at the choice pose and
    compared with the object its body actually needs. The simulator's kinds
    route the driver and score the table; they never enter a model input, and
    no parameter is updated.
    """

    if lives <= 0:
        raise ValueError("lives must be positive.")
    if rounds <= acquisition_rounds:
        raise ValueError("rounds must exceed acquisition_rounds.")
    if not model.has_episodic_bindings:
        raise ValueError("Cross-round reuse requires episodic bindings.")

    config = IslandConfig(
        semantic_choice_trial=True,
        semantic_choice_horizon=semantic_choice_horizon,
        semantic_choice_objects=3,
        semantic_choice_low_need=semantic_choice_low_need,
        semantic_choice_rounds=rounds,
        semantic_choice_return_duration=semantic_choice_return_duration,
        language_mode=language_mode,
        max_visible_slots=model.visible_slots or 8,
    )
    wait_index = ACTIONS.index(Action.WAIT)
    round_hits = [0 for _ in range(rounds)]
    round_cases = [0 for _ in range(rounds)]
    valid_rows = 0.0
    valid_cases = 0
    completed = 0

    def observe(packet, carry):
        vector = mx.array(packet.vector()[None, None, :])
        tokens = mx.array(
            np.asarray(packet.tokens, dtype=np.int32)[None, None, :]
        )
        states, carry = model.core_states(vector, tokens, carry)
        return states, carry, vector

    for life in range(lives):
        seed = base_seed + life
        world = IslandWorld(config, seed=seed)
        packet = world.reset(seed)
        carry = None
        for round_index in range(rounds):
            kinds = [
                world.grid.kind_by_surface[SURFACES[surface_index]]
                for _, _, surface_index in packet.visible
            ]
            if len(kinds) != 3:
                break
            choice_need = str(world.grid.choice_need)
            correct_slot = kinds.index(choice_need)
            states, carry, vector = observe(packet, carry)

            if round_index >= acquisition_rounds:
                scores = _terminal_consume_scores(
                    model,
                    states,
                    vector,
                    urgent_deficit_utility=urgent_deficit_utility,
                )
                _, _, valid = model._state_memory_parts(states)
                mx.eval(scores, valid)
                visible = np.asarray(scores[0, 0, :3], dtype=np.float64)
                round_hits[round_index] += int(
                    int(np.argmax(visible)) == correct_slot
                )
                round_cases[round_index] += 1
                valid_rows += float(np.asarray(valid[0, 0]).sum())
                valid_cases += 1

            if round_index < acquisition_rounds:
                target = "food" if round_index % 2 == 0 else "water"
                action = object_option_action_index(
                    "inspect",
                    kinds.index(target),
                    consume_options=True,
                    inspect_options=True,
                    visible_slots=world.config.max_visible_slots,
                )
                packet, _, terminated, truncated, _ = execute_agent_action(
                    world,
                    packet,
                    action,
                    consume_options=True,
                    inspect_options=True,
                )
                if terminated or truncated:
                    break
                _, carry, _ = observe(packet, carry)
                packet, _, terminated, truncated, _ = execute_agent_action(
                    world,
                    packet,
                    wait_index,
                    consume_options=True,
                    inspect_options=True,
                )
                if terminated or truncated:
                    break
                _, carry, _ = observe(packet, carry)
                kinds = [
                    world.grid.kind_by_surface[SURFACES[surface_index]]
                    for _, _, surface_index in packet.visible
                ]
                correct_slot = kinds.index(choice_need)

            packet, _, terminated, truncated, _ = execute_agent_action(
                world,
                packet,
                object_option_action_index(
                    "consume",
                    correct_slot,
                    consume_options=True,
                    inspect_options=True,
                    visible_slots=world.config.max_visible_slots,
                ),
                consume_options=True,
                inspect_options=True,
            )
            if round_index == rounds - 1:
                # The final consumption ends the life, so completion is
                # recorded here rather than after a next-round transition.
                completed += 1
                break
            if terminated or truncated:
                break
            _, carry, _ = observe(packet, carry)
            if world.semantic_choice_round_pending:
                packet, _, terminated, truncated, _ = execute_agent_action(
                    world,
                    packet,
                    wait_index,
                    consume_options=True,
                    inspect_options=True,
                )
                if terminated or truncated:
                    break
                _, carry, _ = observe(packet, carry)

    measured = sum(round_cases)
    result: dict[str, float] = {
        "audited_lives": float(lives),
        "completed_lives": float(completed),
        "acquisition_rounds": float(acquisition_rounds),
        "measured_rounds": float(measured),
        "aggregate_reuse_accuracy": (
            sum(round_hits) / max(1, measured)
        ),
        "mean_valid_memory_rows": valid_rows / max(1, valid_cases),
    }
    for round_index in range(acquisition_rounds, rounds):
        result[f"round_{round_index + 1}_reuse_accuracy"] = (
            round_hits[round_index] / max(1, round_cases[round_index])
        )
    return result


NEED_NAMES = ("food", "water", "energy", "health")


def audit_metabolic_drift_forecast(
    model: OrganismModel,
    *,
    contexts: int = 300,
    base_seed: int = 1_700_000,
    semantic_choice_horizon: int = 40,
    semantic_choice_low_need: float = 0.55,
    semantic_choice_return_duration: int = 6,
    semantic_choice_rounds: int = 8,
) -> dict[str, float]:
    """Is the ten-tick bodily forecast accurate enough to rank its own needs?

    The corrected homeostatic utility scores whichever need the organism
    predicts will be most urgent when it returns, so the planner can only be as
    good as that forecast. This audit measures the forecast directly, at the
    two steps the planner actually takes -- the four-tick inspect option and
    the six-tick return -- against the drift the simulator really applies.

    It also reports the oracle substitution: the same chain with the true drift
    in place of the predicted drift. That row is the upper bound the forecast
    is being asked to reach, and it isolates forecast error from every other
    part of the chain. Nothing is trained and no model input sees a hidden
    kind; the simulator's demanded resource only scores the table.
    """

    if contexts <= 0:
        raise ValueError("contexts must be positive.")
    if not model.has_episodic_bindings or model.object_option_types != 2:
        raise ValueError(
            "The drift-forecast audit requires consume/inspect object options."
        )

    config = IslandConfig(
        semantic_choice_trial=True,
        semantic_choice_horizon=semantic_choice_horizon,
        semantic_choice_objects=3,
        semantic_choice_low_need=semantic_choice_low_need,
        semantic_choice_rounds=semantic_choice_rounds,
        semantic_choice_return_duration=semantic_choice_return_duration,
        language_mode="grounded",
        max_visible_slots=model.visible_slots or 8,
    )
    wait_index = ACTIONS.index(Action.WAIT)
    steps = ("inspect", "return")
    predicted = {step: [] for step in steps}
    realized = {step: [] for step in steps}
    survival = dict.fromkeys(
        (
            "real_observation",
            "predicted_post_inspect",
            "predicted_post_return",
            "oracle_post_return",
        ),
        0,
    )
    margins: list[float] = []
    audited = 0

    def urgent_names_demand(needs: np.ndarray, demanded: str) -> int:
        return int(NEED_NAMES[int(np.argmin(needs))] == demanded)

    for context in range(contexts):
        seed = base_seed + context
        world = IslandWorld(config, seed=seed)
        packet = world.reset(seed)
        if len(packet.visible) != 3:
            continue
        demanded = str(world.grid.choice_need)
        vector = mx.array(packet.vector()[None, None, :])
        tokens = mx.array(
            np.asarray(packet.tokens, dtype=np.int32)[None, None, :]
        )
        states, _ = model.core_states(vector, tokens)
        current = np.asarray(vector[0, 0, :4], dtype=np.float64)

        # The drift the world really applies over the same two steps.
        probe = IslandWorld(config, seed=seed)
        probe_packet = probe.reset(seed)
        inspect_action = object_option_action_index(
            "inspect",
            0,
            consume_options=True,
            inspect_options=True,
            visible_slots=probe.config.max_visible_slots,
        )
        probe_packet, _, _, _, _ = execute_agent_action(
            probe,
            probe_packet,
            inspect_action,
            consume_options=True,
            inspect_options=True,
        )
        post_inspect = np.asarray(probe_packet.vector()[:4], dtype=np.float64)
        true_inspect = post_inspect - current
        probe_packet, _, _, _, _ = execute_agent_action(
            probe,
            probe_packet,
            wait_index,
            consume_options=True,
            inspect_options=True,
        )
        true_return = (
            np.asarray(probe_packet.vector()[:4], dtype=np.float64)
            - post_inspect
        )

        # The drift the organism predicts at exactly the planner's two queries.
        inspect_state = model.transition_state(
            states, mx.array([[inspect_action]]), vector
        )
        _, inspect_delta, _, _ = model.decode_transition(inspect_state)
        return_state = model.transition_state(
            states, mx.array([[wait_index]]), vector
        )
        _, return_delta, _, _ = model.decode_transition(return_state)
        mx.eval(inspect_delta, return_delta)
        model_inspect = np.asarray(inspect_delta[0, 0], dtype=np.float64)
        model_return = np.asarray(return_delta[0, 0], dtype=np.float64)

        predicted["inspect"].append(model_inspect)
        realized["inspect"].append(true_inspect)
        predicted["return"].append(model_return)
        realized["return"].append(true_return)

        predicted_inspect_needs = np.clip(current + model_inspect, 0.0, 1.0)
        oracle_inspect_needs = np.clip(current + true_inspect, 0.0, 1.0)
        survival["real_observation"] += urgent_names_demand(current, demanded)
        survival["predicted_post_inspect"] += urgent_names_demand(
            predicted_inspect_needs, demanded
        )
        survival["predicted_post_return"] += urgent_names_demand(
            np.clip(predicted_inspect_needs + model_return, 0.0, 1.0), demanded
        )
        survival["oracle_post_return"] += urgent_names_demand(
            np.clip(oracle_inspect_needs + true_return, 0.0, 1.0), demanded
        )
        ordered = np.sort(current)
        margins.append(float(ordered[1] - ordered[0]))
        audited += 1

    if audited == 0:
        raise RuntimeError("The drift-forecast audit found no usable context.")

    result: dict[str, float] = {"audited_contexts": float(audited)}
    worst_resource_bias = 0.0
    worst_absolute_error = 0.0
    for step in steps:
        pred = np.stack(predicted[step])
        real = np.stack(realized[step])
        for index, need in enumerate(NEED_NAMES):
            bias = float((pred[:, index] - real[:, index]).mean())
            absolute = float(np.abs(pred[:, index] - real[:, index]).mean())
            result[f"{step}_{need}_predicted"] = float(pred[:, index].mean())
            result[f"{step}_{need}_realized"] = float(real[:, index].mean())
            result[f"{step}_{need}_bias"] = bias
            result[f"{step}_{need}_absolute_error"] = absolute
            worst_absolute_error = max(worst_absolute_error, absolute)
            if need in ("food", "water"):
                worst_resource_bias = max(worst_resource_bias, abs(bias))
    result["worst_resource_drift_bias"] = worst_resource_bias
    result["worst_need_absolute_error"] = worst_absolute_error
    for name, hits in survival.items():
        result[f"urgent_index_survives_{name}"] = hits / audited
    result["mean_urgency_margin"] = float(np.mean(margins))
    result["min_urgency_margin"] = float(np.min(margins))
    return result


def audit_persistent_choice_environment(
    *,
    lives: int = 2_500,
    base_seed: int = 2_000_000,
    rounds: int = 8,
    semantic_choice_horizon: int = 40,
    semantic_choice_low_need: float = 0.55,
    semantic_choice_return_duration: int = 6,
) -> dict[str, float]:
    """Verify persistent semantic rent through the implemented environment."""

    if lives <= 0 or rounds <= 0:
        raise ValueError("lives and rounds must be positive.")

    def action_for_surface(
        packet: ObsPacket,
        surface_index: int,
        option_kind: str,
    ) -> int:
        slot = next(
            slot
            for slot, (_, _, visible_surface) in enumerate(packet.visible)
            if visible_surface == surface_index
        )
        return object_option_action_index(
            option_kind,
            slot,
            consume_options=True,
            inspect_options=True,
            visible_slots=packet.max_visible_slots,
        )

    policy_names = ("blind", "persistent", "clairvoyant")
    life_utilities = {name: [] for name in policy_names}
    total_ticks = {name: 0.0 for name in policy_names}
    total_inspections = {name: 0.0 for name in policy_names}
    total_rounds = {name: 0 for name in policy_names}
    outcomes = {
        name: {"correct": 0, "wrong_resource": 0, "poison": 0}
        for name in policy_names
    }

    for life in range(lives):
        seed = base_seed + life
        for policy_name in policy_names:
            world = IslandWorld(
                IslandConfig(
                    semantic_choice_trial=True,
                    semantic_choice_horizon=semantic_choice_horizon,
                    semantic_choice_objects=3,
                    semantic_choice_low_need=semantic_choice_low_need,
                    semantic_choice_rounds=rounds,
                    semantic_choice_return_duration=(
                        semantic_choice_return_duration
                    ),
                    language_mode="grounded",
                ),
                seed=seed,
            )
            packet = world.reset(seed)
            surfaces = [surface for _, _, surface in packet.visible]
            known_labels: dict[int, str] = {}
            life_utility = 0.0
            life_ticks = 0
            completed = 0

            while completed < rounds:
                low_need = (
                    "food" if packet.needs[0] < packet.needs[1] else "water"
                )
                if policy_name == "blind":
                    chosen_surface = surfaces[0]
                elif policy_name == "clairvoyant":
                    chosen_surface = next(
                        surface
                        for surface in surfaces
                        if world.grid.kind_by_surface[SURFACES[surface]]
                        == low_need
                    )
                else:
                    while not any(
                        label == low_need for label in known_labels.values()
                    ):
                        unknown = next(
                            surface
                            for surface in surfaces
                            if surface not in known_labels
                        )
                        packet, _, terminated, truncated, inspect_info = (
                            execute_agent_action(
                                world,
                                packet,
                                action_for_surface(
                                    packet,
                                    unknown,
                                    "inspect",
                                ),
                                consume_options=True,
                                inspect_options=True,
                            )
                        )
                        if terminated or truncated:
                            raise RuntimeError(
                                "Persistent environment inspection ended early."
                            )
                        life_ticks += int(inspect_info["duration"])
                        label = VOCAB[packet.tokens[1]]
                        known_labels[unknown] = label
                        total_inspections[policy_name] += 1.0
                        packet, _, terminated, truncated, return_info = (
                            execute_agent_action(
                                world,
                                packet,
                                ACTIONS.index(Action.WAIT),
                                consume_options=True,
                                inspect_options=True,
                            )
                        )
                        if terminated or truncated:
                            raise RuntimeError(
                                "Persistent environment return ended early."
                            )
                        life_ticks += int(return_info["duration"])
                        if len(known_labels) == 2:
                            remaining_surface = next(
                                surface
                                for surface in surfaces
                                if surface not in known_labels
                            )
                            remaining_label = next(
                                label_name
                                for label_name in (
                                    SEMANTIC_CHOICE_LABEL_KINDS
                                )
                                if label_name
                                not in set(known_labels.values())
                            )
                            known_labels[remaining_surface] = remaining_label
                    chosen_surface = next(
                        surface
                        for surface, label in known_labels.items()
                        if label == low_need
                    )

                packet, _, terminated, truncated, info = execute_agent_action(
                    world,
                    packet,
                    action_for_surface(
                        packet,
                        chosen_surface,
                        "consume",
                    ),
                    consume_options=True,
                    inspect_options=True,
                )
                life_ticks += int(info["duration"])
                if not bool(info.get("semantic_choice_round_complete")):
                    raise RuntimeError(
                        "Persistent environment did not complete a round."
                    )
                completed += 1
                life_utility += min(packet.needs)
                outcomes[policy_name]["correct"] += int(
                    bool(info["correct"])
                )
                outcomes[policy_name]["wrong_resource"] += int(
                    bool(info["wrong_resource"])
                )
                outcomes[policy_name]["poison"] += int(bool(info["poison"]))

                if completed < rounds:
                    if terminated or truncated:
                        raise RuntimeError(
                            "Persistent environment life ended before final round."
                        )
                    packet, _, terminated, truncated, transition_info = (
                        execute_agent_action(
                            world,
                            packet,
                            ACTIONS.index(Action.WAIT),
                            consume_options=True,
                            inspect_options=True,
                        )
                    )
                    if terminated or truncated or not bool(
                        transition_info.get("forced_round_transition")
                    ):
                        raise RuntimeError(
                            "Persistent environment round transition failed."
                        )
                    life_ticks += int(transition_info["duration"])
                elif not (terminated or truncated):
                    raise RuntimeError(
                        "Persistent environment final round did not terminate."
                    )

            total_rounds[policy_name] += completed
            total_ticks[policy_name] += life_ticks
            life_utilities[policy_name].append(life_utility)

    result: dict[str, float] = {
        "audited_lives": float(lives),
        "rounds_per_life": float(rounds),
    }
    blind = np.asarray(life_utilities["blind"], dtype=np.float64)
    for policy_name in policy_names:
        utilities = np.asarray(
            life_utilities[policy_name],
            dtype=np.float64,
        )
        denominator = max(1, total_rounds[policy_name])
        result.update(
            {
                f"{policy_name}_completed_rounds_per_life": (
                    total_rounds[policy_name] / lives
                ),
                f"{policy_name}_mean_final_min_need_per_round": (
                    float(utilities.mean() / rounds)
                ),
                f"{policy_name}_paired_gain_per_round": (
                    float((utilities - blind).mean() / rounds)
                ),
                f"{policy_name}_correct_rate": (
                    outcomes[policy_name]["correct"] / denominator
                ),
                f"{policy_name}_wrong_resource_rate": (
                    outcomes[policy_name]["wrong_resource"] / denominator
                ),
                f"{policy_name}_poison_rate": (
                    outcomes[policy_name]["poison"] / denominator
                ),
                f"{policy_name}_mean_ticks_per_round": (
                    total_ticks[policy_name] / denominator
                ),
                f"{policy_name}_mean_inspections_per_life": (
                    total_inspections[policy_name] / lives
                ),
            }
        )
    return result


def load_organism_checkpoint(
    path: str,
) -> tuple[OrganismModel, OrganismConfig]:
    """Reconstruct the unified organism and load a saved checkpoint."""

    weights_path = Path(path)
    metadata_path = weights_path.with_suffix(weights_path.suffix + ".json")
    config_data = json.loads(metadata_path.read_text())
    island_data = config_data.pop("island")
    config = OrganismConfig(
        **config_data,
        island=IslandConfig(**island_data),
    )
    trainer = OrganismTrainer(config)
    trainer.model.load_weights(str(weights_path))
    mx.eval(trainer.model.parameters())
    return trainer.model, config


def save_checkpoint(model: OrganismModel, config: OrganismConfig, path: str) -> None:
    weights_path = Path(path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(weights_path))
    metadata = asdict(config)
    weights_path.with_suffix(weights_path.suffix + ".json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n"
    )


def write_life_stats(stats: list[LifeStats], path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "life_index,seed,steps,survived,mean_viability,min_viability,"
        "utterances_heard,consume_attempts,resource_consumes,"
        "food_consumes,water_consumes,offered_consumes,"
        "option_decisions,inspect_option_decisions,labels_received,harm_events,"
        "semantic_choice_trial,choice_need,chosen_kind,chosen_surface,"
        "chosen_surface_inspected,choice_correct,choice_poison,"
        "choice_wrong_resource,choice_timeout,choice_rounds_completed,"
        "choice_correct_rounds,choice_poison_rounds,"
        "choice_wrong_resource_rounds\n"
    )
    with target.open("w", encoding="utf-8") as handle:
        handle.write(header)
        for life in stats:
            handle.write(
                f"{life.life_index},{life.seed},{life.steps},{int(life.survived)},"
                f"{life.mean_viability:.6f},{life.min_viability:.6f},"
                f"{life.utterances_heard},{life.consume_attempts},"
                f"{life.resource_consumes},{life.food_consumes},"
                f"{life.water_consumes},{life.offered_consumes},"
                f"{life.option_decisions},{life.inspect_option_decisions},"
                f"{life.labels_received},{life.harm_events},"
                f"{int(life.semantic_choice_trial)},{life.choice_need},"
                f"{life.chosen_kind},{life.chosen_surface},"
                f"{int(life.chosen_surface_inspected)},"
                f"{int(life.choice_correct)},{int(life.choice_poison)},"
                f"{int(life.choice_wrong_resource)},"
                f"{int(life.choice_timeout)},"
                f"{life.choice_rounds_completed},"
                f"{life.choice_correct_rounds},"
                f"{life.choice_poison_rounds},"
                f"{life.choice_wrong_resource_rounds}\n"
            )
