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

from copy import deepcopy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from random import Random

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from homesocial.creole.vocab import PAD_TOKEN, TOKEN_TO_ID, VOCAB
from homesocial.env import Action, DELTAS, DIRECTION_ORDER, Direction
from homesocial.island.oracle import OraclePolicy
from homesocial.island.world import SURFACES, IslandConfig, IslandWorld, ObsPacket
from homesocial.organism.model import OrganismModel

ACTIONS = list(Action)


def agent_action_size(*, consume_options: bool, visible_slots: int) -> int:
    return len(ACTIONS) + (visible_slots if consume_options else 0)


def available_action_mask(packet: ObsPacket, action_size: int) -> np.ndarray:
    """Primitive actions plus only the object slots actually perceived."""

    mask = np.ones(action_size, dtype=np.bool_)
    if action_size > len(ACTIONS):
        mask[len(ACTIONS) :] = False
        mask[len(ACTIONS) : len(ACTIONS) + len(packet.visible)] = True
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
    world: IslandWorld, target: tuple[int, int]
) -> Action:
    """Kind-blind low-level action toward a perceived object location."""

    grid = world.grid
    distance = abs(target[0] - grid.agent_pos[0]) + abs(
        target[1] - grid.agent_pos[1]
    )
    if distance == 0:
        return Action.CONSUME
    desired = _desired_direction(grid.agent_pos, target)
    if distance == 1:
        if grid.direction == desired:
            return Action.CONSUME
        return _turn_toward(grid.direction, desired)
    if grid.direction != desired:
        return _turn_toward(grid.direction, desired)
    return Action.MOVE_FORWARD


def execute_agent_action(
    world: IslandWorld,
    packet: ObsPacket,
    action_index: int,
    *,
    consume_options: bool,
) -> tuple[ObsPacket, float, bool, bool, dict[str, object]]:
    """Execute one primitive decision or a visible-slot consume option.

    The option can access only the learner-perceived relative target location
    and proprioceptive pose. It never reads the target's hidden bodily kind.
    """

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

    slot = action_index - len(ACTIONS)
    if not consume_options or slot >= len(packet.visible):
        return execute_agent_action(
            world,
            packet,
            ACTIONS.index(Action.WAIT),
            consume_options=False,
        )
    dx, dy, _ = packet.visible[slot]
    target = (packet.position[0] + dx, packet.position[1] + dy)
    reward_sum = 0.0
    viability_sum = 0.0
    min_viability = 1.0
    events: list[object] = []
    current_packet = packet
    terminated = False
    truncated = False
    final_info: dict[str, object] = {}
    # Radius-two targets require at most two moves, two turns, and consume.
    # The small margin lets the servo recover from one blocked direct route.
    for _ in range(7):
        primitive = _motor_action_toward(world, target)
        current_packet, reward, terminated, truncated, info = world.step(primitive)
        reward_sum += float(reward)
        viability_sum += float(info["mean_viability"])
        min_viability = min(min_viability, float(info["viability"]))
        events.append(info.get("event"))
        final_info = dict(info)
        if primitive == Action.CONSUME or terminated or truncated:
            break
    final_info["duration"] = len(events)
    final_info["mean_viability_sum"] = viability_sum
    final_info["min_viability"] = min_viability
    final_info["events"] = tuple(events)
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
) -> mx.array:
    """Bias policy logits with detached predicted future-body values.

    ``score_sign=-1`` is an evaluation intervention: it makes the organism
    prefer the consequences its own self-model judges worse while preserving
    the policy, recurrent state, consequence predictions, and action mask.
    """

    predicted_needs, predicted_rewards = predict_all_action_consequences(
        model, states, current_vectors
    )
    scores = mx.min(predicted_needs, axis=-1) + reward_weight * predicted_rewards
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


def bodily_delta_prediction_loss(
    predicted_deltas: mx.array,
    target_deltas: mx.array,
    *,
    change_boost: float,
) -> mx.array:
    """MSE that preserves rare, action-specific bodily consequences."""

    per_transition = ((predicted_deltas - target_deltas) ** 2).mean(axis=-1)
    weights = 1.0 + change_boost * mx.max(mx.abs(target_deltas), axis=-1)
    return (per_transition * weights).sum() / weights.sum()


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
    reward_prediction_weight: float = 0.2
    token_prediction_weight: float = 0.5
    nonpad_token_weight: float = 5.0
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
    consume_options: bool = False
    self_model_planning_scale: float = 0.0
    self_model_planning_start_steps: int = 0
    self_model_planning_reward_weight: float = 0.5
    seed: int = 1
    max_steps: int = 1000
    log_every_lives: int = 10
    checkpoint: str | None = None
    stats_csv: str | None = None
    island: IslandConfig = field(default_factory=IslandConfig)

    def island_config(self) -> IslandConfig:
        return IslandConfig(
            width=self.island.width,
            height=self.island.height,
            max_steps=self.max_steps,
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
    harm_events: int


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
            visible_slots=world.config.max_visible_slots,
        ),
        hidden_size=config.hidden_size,
        token_embed_size=config.token_embed_size,
        primitive_action_size=len(ACTIONS),
        visible_slots=world.config.max_visible_slots,
        object_feature_offset=object_feature_offset,
        object_feature_size=3 + len(SURFACES),
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
        self.next_vectors: list[np.ndarray] = []
        self.next_needs: list[tuple[float, ...]] = []
        self.next_tokens: list[tuple[int, ...]] = []

    def __len__(self) -> int:
        return len(self.actions)


class OrganismTrainer:
    def __init__(self, config: OrganismConfig) -> None:
        self.config = config
        mx.random.seed(config.seed)
        self.world = IslandWorld(config.island_config(), seed=config.seed)
        self.model = build_model(config, self.world)
        self.optimizer = optim.Adam(learning_rate=config.learning_rate)
        self.rng = Random(config.seed)
        self.life_stats: list[LifeStats] = []
        self.loss_log: list[dict[str, float]] = []
        self.bc_stats: BehaviorCloningStats | None = None
        self._loss_and_grad = nn.value_and_grad(self.model, self._loss)
        self._bc_loss_and_grad = nn.value_and_grad(self.model, self._bc_loss)

        self.life_index = 0
        self.life_seed = config.seed
        self.global_steps = 0
        self.packet = self.world.reset(self.life_seed)
        self._apply_development_curricula()
        self.hidden: mx.array | None = None
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
        self._life_harm_events = 0

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
        logits = self.model.policy(states)[0]
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

    def _act(self) -> tuple[int, float, float, np.ndarray]:
        vector = mx.array(self.packet.vector()[None, None, :])
        tokens = mx.array(
            np.asarray(self.packet.tokens, dtype=np.int32)[None, None, :]
        )
        states, self.hidden = self.model.core_states(
            vector, tokens, self.hidden
        )
        logits = self.model.policy(states)
        values = self.model.value(states).squeeze(-1)
        planning_scale = (
            self.config.self_model_planning_scale
            if self.global_steps >= self.config.self_model_planning_start_steps
            else 0.0
        )
        if planning_scale > 0.0:
            logits = planned_policy_logits(
                self.model,
                states,
                vector,
                logits,
                mx.array([[[planning_scale]]]).reshape(1, 1),
                reward_weight=self.config.self_model_planning_reward_weight,
            )
        action_mask = available_action_mask(self.packet, self.model.action_size)
        logits = mx.where(
            mx.array(action_mask)[None, None, :], logits, -1e9
        )
        mx.eval(logits, values, self.hidden)
        probabilities = np.asarray(
            mx.softmax(logits[0, 0], axis=-1), dtype=np.float64
        )
        probabilities = probabilities / probabilities.sum()
        action_index = int(
            self.rng.choices(range(self.model.action_size), weights=probabilities)[0]
        )
        return action_index, float(values[0, 0]), planning_scale, action_mask

    def collect_segment(self) -> tuple[_Segment, mx.array | None, float]:
        """Collect experience until segment length or end of life."""

        segment = _Segment()
        initial_hidden = self.hidden
        pad_id = TOKEN_TO_ID[PAD_TOKEN]
        while len(segment) < self.config.segment_length:
            vector_before = self.packet.vector()
            tokens_before = self.packet.tokens
            action_index, value, planning_scale, action_mask = self._act()
            if action_index >= len(ACTIONS):
                self._life_option_decisions += 1
            next_packet, env_reward, terminated, truncated, info = execute_agent_action(
                self.world,
                self.packet,
                action_index,
                consume_options=self.config.consume_options,
            )
            duration = int(info["duration"])
            mean_viability_sum = float(info["mean_viability_sum"])
            mean_viability = mean_viability_sum / duration
            shaped = (
                env_reward
                + self.config.live_reward_weight * mean_viability_sum
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
            self._life_harm_events += sum(
                event in {"hit_danger", "consumed_poison"} for event in events
            )

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
                harm_events=self._life_harm_events,
            )
        )
        self.life_index += 1
        self.life_seed = self.config.seed + self.life_index
        self.packet = self.world.reset(self.life_seed)
        self._apply_development_curricula()
        self.hidden = None
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
        self._life_harm_events = 0

    # ------------------------------------------------------------ learning

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
        next_vectors: mx.array,
        next_needs: mx.array,
        env_rewards: mx.array,
        next_tokens: mx.array,
    ) -> mx.array:
        config = self.config
        states, _ = self.model.core_states(vectors, tokens, hidden)
        logits = self.model.policy(states)
        if config.self_model_planning_scale > 0.0:
            logits = planned_policy_logits(
                self.model,
                states,
                vectors,
                logits,
                planning_scales[None, :],
                reward_weight=config.self_model_planning_reward_weight,
            )
        logits = mx.where(action_masks[None, :, :], logits, -1e9)
        logits = logits[0]
        values = self.model.value(states).squeeze(-1)[0]
        log_probabilities = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        chosen = mx.take_along_axis(
            log_probabilities, actions[:, None], axis=-1
        ).squeeze(-1)
        policy_loss = -(advantages * chosen).mean()
        value_loss = ((values - returns) ** 2).mean()
        entropy = -(mx.softmax(logits, axis=-1) * log_probabilities).sum(-1).mean()

        predicted_vectors, predicted_need_deltas, predicted_rewards, token_logits = (
            self.model.predict_consequences(states, actions[None, :], vectors)
        )
        vector_loss = ((predicted_vectors[0] - next_vectors) ** 2).mean()
        target_need_deltas = next_needs - vectors[0, :, :4]
        needs_loss = bodily_delta_prediction_loss(
            predicted_need_deltas[0],
            target_need_deltas,
            change_boost=config.bodily_change_loss_boost,
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

        return (
            policy_loss
            + config.value_weight * value_loss
            - config.entropy_weight * entropy
            + config.next_vector_weight * vector_loss
            + config.next_needs_weight * needs_loss
            + config.reward_prediction_weight * reward_loss
            + config.token_prediction_weight * token_loss
        )

    def update(
        self, segment: _Segment, hidden: mx.array | None, bootstrap: float
    ) -> float:
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
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        loss, grads = self._loss_and_grad(
            mx.array(np.stack(segment.vectors)[None, ...]),
            mx.array(np.asarray(segment.tokens, dtype=np.int32)[None, ...]),
            hidden,
            mx.array(np.asarray(segment.actions, dtype=np.int32)),
            mx.array(advantages),
            mx.array(returns),
            mx.array(np.asarray(segment.planning_scales, dtype=np.float32)),
            mx.array(np.stack(segment.action_masks)),
            mx.array(np.stack(segment.next_vectors)),
            mx.array(np.asarray(segment.next_needs, dtype=np.float32)),
            mx.array(np.asarray(segment.env_rewards, dtype=np.float32)),
            mx.array(np.asarray(segment.next_tokens, dtype=np.int32)),
        )
        grads, _ = optim.clip_grad_norm(grads, config.max_grad_norm)
        self.optimizer.update(self.model, grads)
        mx.eval(self.model.parameters(), self.optimizer.state, loss)
        return float(loss)

    # -------------------------------------------------------------- driver

    def train(self) -> list[LifeStats]:
        config = self.config
        self.behavior_clone()
        steps_done = 0
        lives_logged = 0
        while steps_done < config.total_steps:
            segment, hidden, bootstrap = self.collect_segment()
            if len(segment) == 0:
                continue
            loss = self.update(segment, hidden, bootstrap)
            steps_done += sum(segment.durations)
            self.loss_log.append({"steps": float(steps_done), "loss": loss})
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
    self_model_planning_scale: float = 0.0,
    self_model_planning_reward_weight: float = 0.5,
    self_model_planning_score_sign: float = 1.0,
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
    terminal_needs_sum = np.zeros(4, dtype=np.float64)
    death_causes = np.zeros(4, dtype=np.float64)
    for episode in range(episodes):
        seed = base_seed + episode
        world = IslandWorld(
            IslandConfig(language_mode=language_mode, max_steps=max_steps), seed=seed
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
            logits = model.policy(states)
            if self_model_planning_scale > 0.0:
                logits = planned_policy_logits(
                    model,
                    states,
                    vector,
                    logits,
                    mx.array([[self_model_planning_scale]]),
                    reward_weight=self_model_planning_reward_weight,
                    score_sign=self_model_planning_score_sign,
                )
            action_mask = available_action_mask(packet, model.action_size)
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
            )
            option_decision_sum += int(action_index >= len(ACTIONS))
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
        "terminal_food": terminal_needs_sum[0] / episodes,
        "terminal_water": terminal_needs_sum[1] / episodes,
        "terminal_energy": terminal_needs_sum[2] / episodes,
        "terminal_safety": terminal_needs_sum[3] / episodes,
        "food_death_rate": death_causes[0] / episodes,
        "water_death_rate": death_causes[1] / episodes,
        "energy_death_rate": death_causes[2] / episodes,
        "safety_death_rate": death_causes[3] / episodes,
    }


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
    self_model_planning_scale: float = 0.0,
    reward_weight: float = 0.5,
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

    for episode in range(episodes):
        if audited_decisions >= max_decisions:
            break
        seed = base_seed + episode
        world = IslandWorld(
            IslandConfig(language_mode=language_mode, max_steps=max_steps),
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
            base_logits = model.policy(states)
            logits = base_logits
            if self_model_planning_scale > 0.0:
                logits = planned_policy_logits(
                    model,
                    states,
                    vector,
                    base_logits,
                    mx.array([[self_model_planning_scale]]),
                    reward_weight=reward_weight,
                )
            action_mask = available_action_mask(packet, model.action_size)
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

            packet, _, terminated, truncated, _ = execute_agent_action(
                world,
                packet,
                action_index,
                consume_options=consume_options,
            )
            if terminated or truncated:
                break

    correlation_denominator = (predicted_square * actual_square) ** 0.5
    return {
        "audited_decisions": float(audited_decisions),
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
        "option_decisions,harm_events\n"
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
                f"{life.option_decisions},{life.harm_events}\n"
            )
