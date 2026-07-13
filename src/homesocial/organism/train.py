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

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from random import Random

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from homesocial.creole.vocab import PAD_TOKEN, TOKEN_TO_ID, VOCAB
from homesocial.env import Action
from homesocial.island.world import IslandConfig, IslandWorld, ObsPacket
from homesocial.organism.model import OrganismModel

ACTIONS = list(Action)


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
    reward_prediction_weight: float = 0.2
    token_prediction_weight: float = 0.5
    nonpad_token_weight: float = 5.0
    live_reward_weight: float = 0.02
    death_penalty: float = 1.0
    max_grad_norm: float = 1.0
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


def build_model(config: OrganismConfig, world: IslandWorld) -> OrganismModel:
    return OrganismModel(
        vector_size=ObsPacket.vector_size(
            max_visible_slots=world.config.max_visible_slots
        ),
        vocab_size=len(VOCAB),
        tokens_per_utterance=world.tokens_per_utterance,
        action_size=len(ACTIONS),
        hidden_size=config.hidden_size,
        token_embed_size=config.token_embed_size,
    )


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    bootstrap: float,
    dones: np.ndarray,
    *,
    discount: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    steps = len(rewards)
    advantages = np.zeros(steps, dtype=np.float32)
    last_advantage = 0.0
    next_value = bootstrap
    for t in reversed(range(steps)):
        not_done = 1.0 - float(dones[t])
        delta = rewards[t] + discount * next_value * not_done - values[t]
        last_advantage = delta + discount * gae_lambda * not_done * last_advantage
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
        self.next_vectors: list[np.ndarray] = []
        self.next_needs: list[tuple[float, ...]] = []
        self.next_tokens: list[tuple[int, ...]] = []

    def __len__(self) -> int:
        return len(self.actions)


class OrganismTrainer:
    def __init__(self, config: OrganismConfig) -> None:
        self.config = config
        self.world = IslandWorld(config.island_config(), seed=config.seed)
        self.model = build_model(config, self.world)
        self.optimizer = optim.Adam(learning_rate=config.learning_rate)
        self.rng = Random(config.seed)
        self.life_stats: list[LifeStats] = []
        self.loss_log: list[dict[str, float]] = []
        self._loss_and_grad = nn.value_and_grad(self.model, self._loss)

        self.life_index = 0
        self.life_seed = config.seed
        self.packet = self.world.reset(self.life_seed)
        self.hidden: mx.array | None = None
        self._life_steps = 0
        self._life_viability_sum = 0.0
        self._life_min_viability = 1.0
        self._life_utterances = 0

    # ------------------------------------------------------------- acting

    def _act(self) -> tuple[int, float]:
        vector = mx.array(self.packet.vector()[None, None, :])
        tokens = mx.array(
            np.asarray(self.packet.tokens, dtype=np.int32)[None, None, :]
        )
        logits, values, self.hidden = self.model.policy_value(
            vector, tokens, self.hidden
        )
        mx.eval(logits, values, self.hidden)
        probabilities = np.asarray(
            mx.softmax(logits[0, 0], axis=-1), dtype=np.float64
        )
        probabilities = probabilities / probabilities.sum()
        action_index = int(
            self.rng.choices(range(len(ACTIONS)), weights=probabilities)[0]
        )
        return action_index, float(values[0, 0])

    def collect_segment(self) -> tuple[_Segment, mx.array | None, float]:
        """Collect experience until segment length or end of life."""

        segment = _Segment()
        initial_hidden = self.hidden
        pad_id = TOKEN_TO_ID[PAD_TOKEN]
        while len(segment) < self.config.segment_length:
            vector_before = self.packet.vector()
            tokens_before = self.packet.tokens
            action_index, value = self._act()
            next_packet, env_reward, terminated, truncated, info = self.world.step(
                ACTIONS[action_index]
            )
            mean_viability = float(info["mean_viability"])
            shaped = (
                env_reward
                + self.config.live_reward_weight * mean_viability
                - (self.config.death_penalty if terminated else 0.0)
            )
            segment.vectors.append(vector_before)
            segment.tokens.append(tokens_before)
            segment.actions.append(action_index)
            segment.shaped_rewards.append(shaped)
            segment.env_rewards.append(float(env_reward))
            segment.values.append(value)
            segment.dones.append(terminated)
            segment.next_vectors.append(next_packet.vector())
            segment.next_needs.append(next_packet.needs)
            segment.next_tokens.append(next_packet.tokens)

            self._life_steps += 1
            self._life_viability_sum += mean_viability
            self._life_min_viability = min(
                self._life_min_viability, float(info["viability"])
            )
            if any(token != pad_id for token in next_packet.tokens):
                self._life_utterances += 1

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
            )
        )
        self.life_index += 1
        self.life_seed = self.config.seed + self.life_index
        self.packet = self.world.reset(self.life_seed)
        self.hidden = None
        self._life_steps = 0
        self._life_viability_sum = 0.0
        self._life_min_viability = 1.0
        self._life_utterances = 0

    # ------------------------------------------------------------ learning

    def _loss(
        self,
        vectors: mx.array,
        tokens: mx.array,
        hidden: mx.array | None,
        actions: mx.array,
        advantages: mx.array,
        returns: mx.array,
        next_vectors: mx.array,
        next_needs: mx.array,
        env_rewards: mx.array,
        next_tokens: mx.array,
    ) -> mx.array:
        config = self.config
        states, _ = self.model.core_states(vectors, tokens, hidden)
        logits = self.model.policy(states)[0]
        values = self.model.value(states).squeeze(-1)[0]
        log_probabilities = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        chosen = mx.take_along_axis(
            log_probabilities, actions[:, None], axis=-1
        ).squeeze(-1)
        policy_loss = -(advantages * chosen).mean()
        value_loss = ((values - returns) ** 2).mean()
        entropy = -(mx.softmax(logits, axis=-1) * log_probabilities).sum(-1).mean()

        predicted_vectors, predicted_needs, predicted_rewards, token_logits = (
            self.model.predict_consequences(states, actions[None, :])
        )
        vector_loss = ((predicted_vectors[0] - next_vectors) ** 2).mean()
        needs_loss = ((predicted_needs[0] - next_needs) ** 2).mean()
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
        advantages, returns = compute_gae(
            rewards,
            values,
            bootstrap,
            dones,
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
        steps_done = 0
        lives_logged = 0
        while steps_done < config.total_steps:
            segment, hidden, bootstrap = self.collect_segment()
            if len(segment) == 0:
                continue
            loss = self.update(segment, hidden, bootstrap)
            steps_done += len(segment)
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
) -> dict[str, float]:
    rng = Random(sample_seed)
    survived = 0
    viability_sum = 0.0
    min_viability_sum = 0.0
    steps_sum = 0
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
            logits, _, hidden = model.policy_value(vector, tokens, hidden)
            mx.eval(logits, hidden)
            probabilities = np.asarray(
                mx.softmax(logits[0, 0], axis=-1), dtype=np.float64
            )
            probabilities = probabilities / probabilities.sum()
            action_index = int(
                rng.choices(range(len(ACTIONS)), weights=probabilities)[0]
            )
            packet, _, terminated, truncated, info = world.step(ACTIONS[action_index])
            steps += 1
            viability_running += float(info["mean_viability"])
            min_viability = min(min_viability, float(info["viability"]))
            if terminated or truncated:
                survived += int(truncated and not terminated)
                break
        viability_sum += viability_running / max(1, steps)
        min_viability_sum += min_viability
        steps_sum += steps
    return {
        "survival_rate": survived / episodes,
        "mean_viability": viability_sum / episodes,
        "mean_min_viability": min_viability_sum / episodes,
        "mean_steps": steps_sum / episodes,
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
    header = "life_index,seed,steps,survived,mean_viability,min_viability,utterances_heard\n"
    with target.open("w", encoding="utf-8") as handle:
        handle.write(header)
        for life in stats:
            handle.write(
                f"{life.life_index},{life.seed},{life.steps},{int(life.survived)},"
                f"{life.mean_viability:.6f},{life.min_viability:.6f},"
                f"{life.utterances_heard}\n"
            )
