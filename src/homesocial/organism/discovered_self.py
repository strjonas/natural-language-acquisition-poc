"""Discovering the variables of one's own body from how good and how bad it feels.

Probe57's self-model is 658 scalars fitted into a template a human wrote: three
axes, in a fixed order, supervised directly by the true body vector, with
movement hardcoded to cost the third one. Nothing in it was discovered.

This module removes that template. The learner is given an overcomplete latent
of eight dimensions and only two scalar sensations per transition -- the mean
and the minimum of its own bodily variables, which are exactly the world's own
reward and death signals -- and must work out for itself how many variables it
has, what each one's depletion rate is, which of the world's resources restores
which, and which one movement costs. It is not told the number three, an axis
order, an axis name, or a per-life birth reading.

The number of bodily variables is varied in the world, by freezing axes, so that
the recovered dimension can be checked against a ground truth that moves. A
sparsity coefficient that sets the answer cannot track a moving answer.

See ``docs/decisions/2026-08-02-discovered-self-structure-preregistration.md``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import json
from pathlib import Path
from random import Random

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from homesocial.creole.vocab import PAD_TOKEN, TOKEN_TO_ID, VOCAB
from homesocial.env import Action
from homesocial.island.report import (
    BODY_NEEDS,
    HELP_SURFACES,
    NEED_TO_REPORT_WORD,
    REPORT_NEEDS,
    SHOCK_SURFACES,
    heard_need,
)
from homesocial.island.world import SURFACES
from homesocial.organism.causal_self import (
    _listener_target,
    _public_transition_features,
)
from homesocial.organism.model import OrganismModel
from homesocial.organism.report_audit import FIDELITY_WARMUP, make_report_world
from homesocial.organism.self_belief import _sample_motor_action
from homesocial.organism.train import (
    ACTIONS,
    OrganismConfig,
    execute_agent_action,
    load_organism_checkpoint,
)

PAD_ID = TOKEN_TO_ID[PAD_TOKEN]
SURFACE_COUNT = len(SURFACES)
ACTION_WAIT = ACTIONS.index(Action.WAIT)
ACTION_CONSUME = ACTIONS.index(Action.CONSUME)

# The ground-truth sweep. Freezing an axis makes it born full, metabolically
# inert and never shocked, so nothing the organism can feel carries information
# about it. ``K4`` is the unmodified frozen report ecology, whose true bodily
# dimension is four because ``safety`` really does deplete.
DIMENSION_SWEEP: dict[str, tuple[str, ...]] = {
    "K1": ("food", "water", "safety"),
    "K2": ("food", "safety"),
    "K3": ("safety",),
    "K4": (),
}

# Seed bands, kept clear of every evaluation band this repository already uses
# (probe57 gate 6.5M-7.6M, probe58 development 6.1M+10M*i, probe59/60 82.1M+).
DEVELOPMENT_SEED_BASE = 210_000_000
DEVELOPMENT_SEED_STRIDE = 10_000_000
WORLD_SEED_OFFSET = 1_000_000
DEVELOPMENT_SEED_HEADROOM = 900_000
MAPPING_SEED_BASE = 300_000_000
DEPLOYMENT_SEED_BASE = 320_000_000
EVALUATION_SEED_BANDS = (
    (6_100_000, 7_600_000),
    (82_100_000, 122_200_000),
    (MAPPING_SEED_BASE, MAPPING_SEED_BASE + 1_000_000),
    (DEPLOYMENT_SEED_BASE, DEPLOYMENT_SEED_BASE + 1_000_000),
)


def development_seed_base(seed_index: int, world: str) -> int:
    """Developmental world stream for one seed and one ground-truth world."""

    if world not in DIMENSION_SWEEP:
        raise ValueError(f"Unknown ground-truth world: {world}.")
    offset = list(DIMENSION_SWEEP).index(world)
    base = (
        DEVELOPMENT_SEED_BASE
        + seed_index * DEVELOPMENT_SEED_STRIDE
        + offset * WORLD_SEED_OFFSET
    )
    _check_seed_isolation(base)
    return base


def _check_seed_isolation(seed_base: int) -> None:
    low = seed_base
    high = seed_base + DEVELOPMENT_SEED_HEADROOM
    for band_low, band_high in EVALUATION_SEED_BANDS:
        if low <= band_high and band_low <= high:
            raise ValueError(
                f"Developmental seed_base {seed_base} overlaps the evaluation "
                f"band [{band_low}, {band_high}]."
            )


@dataclass(frozen=True)
class DiscoveredSelfConfig:
    """Everything the learner is allowed to be told, in one place."""

    latent_size: int = 8
    development_ticks: int = 80_000
    # Body-independent token distribution. Fixed before any body is seen; its
    # only purpose is to lengthen developmental lives so that slow bodily
    # variables are observable at all.
    need_word_probability: float = 0.10
    learning_rate: float = 3e-3
    sparsity: float = 1e-3
    holdout_fraction: float = 0.20
    # Bock multiple shooting: (segment length, optimizer steps) per stage, the
    # last stage being the honest single-shot trajectory.
    shooting_schedule: tuple[tuple[int, int], ...] = (
        (25, 1500),
        (100, 750),
        (0, 750),
    )
    continuity_weight: float = 10.0
    # Held-out initial conditions are a nuisance variable, not model structure.
    birth_learning_rate: float = 3e-2
    birth_inference_steps: int = 300
    ablation_inference_steps: int = 60
    # Locked in the preregistration: the greedy-elimination stopping fraction
    # of the explainable RMSE range.
    dimension_tolerance: float = 0.10

    def __post_init__(self) -> None:
        if self.latent_size < len(BODY_NEEDS):
            raise ValueError("The latent must be overcomplete against the body.")
        if not 0.0 <= 3 * self.need_word_probability <= 1.0:
            raise ValueError("Need-word probabilities must leave room for the rest.")
        if not 0.0 < self.holdout_fraction < 1.0:
            raise ValueError("holdout_fraction must lie in (0, 1).")


# ---------------------------------------------------------------------------
# The learner
# ---------------------------------------------------------------------------


class DiscoveredSelfModel(nn.Module):
    """An overcomplete bounded-resource body whose structure is not given.

    Every dimension is interchangeable at initialization: the readouts are
    symmetric under permutation, so nothing here can hand over an axis identity,
    an axis count, or an axis order. Symmetry is broken only by seed-dependent
    noise and then by the world.
    """

    def __init__(
        self,
        *,
        latent_size: int,
        surface_count: int = SURFACE_COUNT,
        vocab_size: int = len(VOCAB),
        seed: int = 0,
    ) -> None:
        super().__init__()
        key = mx.random.key(seed)
        keys = mx.random.split(key, 5)
        self.latent_size = latent_size
        self.surface_count = surface_count
        # Generic small-effect initialization with seed-dependent noise.
        self.drift_raw = -4.0 + 0.5 * mx.random.normal((latent_size,), key=keys[0])
        self.move_raw = -4.0 + 0.5 * mx.random.normal((latent_size,), key=keys[1])
        self.uptake_raw = -3.0 + 0.5 * mx.random.normal(
            (surface_count, latent_size), key=keys[2]
        )
        self.shock_raw = -3.0 + 0.5 * mx.random.normal(
            (surface_count, latent_size), key=keys[3]
        )
        # Start believing it has eight variables; sparsity must take them away.
        self.gate_raw = mx.full((latent_size,), 1.0)
        self.birth_raw = 0.5 * mx.random.normal((latent_size,), key=keys[4])
        self.mean_scale_raw = mx.array(-1.0)
        self.mean_offset = mx.array(0.0)
        self.listener_logits = mx.zeros((vocab_size, surface_count + 1))

    @staticmethod
    def _positive(raw: mx.array) -> mx.array:
        return mx.logaddexp(mx.array(0.0), raw)

    def body_parameters(self) -> dict[str, mx.array]:
        return {
            "drift": -self._positive(self.drift_raw),
            "move": -self._positive(self.move_raw),
            "uptake": self._positive(self.uptake_raw),
            "shock": -self._positive(self.shock_raw),
            "gate": mx.sigmoid(self.gate_raw),
            "birth": mx.sigmoid(self.birth_raw),
            "mean_scale": self._positive(self.mean_scale_raw),
            "mean_offset": self.mean_offset,
        }

    def rollout(
        self,
        durations: mx.array,
        moves: mx.array,
        uptake_surfaces: mx.array,
        shock_surfaces: mx.array,
        births: mx.array | None = None,
    ) -> mx.array:
        """Integrate the latent from a birth state through a whole life.

        There is no teacher forcing anywhere in this function: the latent is
        never reset to a truth the learner is not given. ``births`` carries the
        per-life initial condition, itself inferred from the sensations alone;
        without it the learned population prior is used, which is what the
        deployed organism has.
        """

        params = self.body_parameters()
        batch = durations.shape[0]
        steps = durations.shape[1]
        if births is None:
            state = mx.broadcast_to(
                params["birth"][None, :], (batch, self.latent_size)
            )
        else:
            state = births
        drift = params["drift"]
        move = params["move"]
        states = []
        for index in range(steps):
            delta = (
                durations[:, index, None] * drift
                + moves[:, index, None] * move
                + uptake_surfaces[:, index] @ params["uptake"]
                + shock_surfaces[:, index] @ params["shock"]
            )
            state = mx.clip(state + delta, 0.0, 1.0)
            states.append(state)
        return mx.stack(states, axis=1)

    def feelings(
        self, states: mx.array, keep: mx.array | None = None
    ) -> tuple[mx.array, mx.array]:
        """The two sensations the learner is asked to account for.

        An unparticipating dimension is displaced out of reach of the minimum
        and contributes nothing to the mean, so deleting a dimension is exactly
        zeroing its gate. A fully participating dimension enters the minimum
        undistorted, which is what lets the sensation pin the latent's scale.
        """

        params = self.body_parameters()
        gate = params["gate"] if keep is None else params["gate"] * keep
        mean_hat = params["mean_offset"] + params["mean_scale"] * mx.sum(
            gate * states, axis=-1
        )
        min_hat = mx.min(states + (1.0 - gate), axis=-1)
        return mean_hat, min_hat


class _IdentificationProblem(nn.Module):
    """The body being identified, plus the states the trajectory is shot from.

    A life's initial condition is a nuisance variable of system identification:
    it is not part of the body's structure and it is inferred from the same two
    sensations as everything else. Carrying it explicitly is what keeps random
    birth levels from being charged to the dynamics.

    Fitting a whole life open-loop in one shot is the badly conditioned form of
    this problem -- an error in a depletion rate is multiplied by every tick
    that follows it. ``window`` turns it into Bock's multiple-shooting form:
    each short segment is shot from its own free state and the segments are
    stitched by a continuity penalty. The segment states are an optimizer
    device and are discarded; only the body's parameters survive, and held-out
    evaluation always uses single shooting, so the reported error is the honest
    open-loop one.
    """

    def __init__(
        self, body: DiscoveredSelfModel, lives: int, steps: int, window: int = 0
    ) -> None:
        super().__init__()
        self.body = body
        self.window = window
        self.segments = 1 if window <= 0 else (steps + window - 1) // window
        self.shoot_raw = mx.zeros((lives, self.segments, body.latent_size))

    def births(self) -> mx.array:
        return mx.sigmoid(self.shoot_raw[:, 0])


def _shoot_rollout(
    problem: _IdentificationProblem,
    durations: mx.array,
    moves: mx.array,
    uptake_surfaces: mx.array,
    shock_surfaces: mx.array,
    mask: mx.array,
) -> tuple[mx.array, mx.array]:
    body = problem.body
    params = body.body_parameters()
    shoot = mx.sigmoid(problem.shoot_raw)
    window = problem.window
    steps = durations.shape[1]
    state = shoot[:, 0]
    states = []
    continuity = mx.array(0.0)
    for index in range(steps):
        if window > 0 and index > 0 and index % window == 0:
            target = shoot[:, index // window]
            continuity = continuity + mx.sum(
                mask[:, index, None] * (state - target) ** 2
            )
            state = target
        delta = (
            durations[:, index, None] * params["drift"]
            + moves[:, index, None] * params["move"]
            + uptake_surfaces[:, index] @ params["uptake"]
            + shock_surfaces[:, index] @ params["shock"]
        )
        state = mx.clip(state + delta, 0.0, 1.0)
        states.append(state)
    return mx.stack(states, axis=1), continuity


def _identification_loss(
    problem: _IdentificationProblem,
    durations: mx.array,
    moves: mx.array,
    uptake: mx.array,
    shock: mx.array,
    mask: mx.array,
    mean_feel: mx.array,
    min_feel: mx.array,
    sparsity: mx.array,
    channel_weights: mx.array,
    keep: mx.array,
    continuity_weight: mx.array,
) -> mx.array:
    body = problem.body
    states, continuity = _shoot_rollout(
        problem, durations, moves, uptake, shock, mask
    )
    mean_hat, min_hat = body.feelings(states, keep=keep)
    weight = mx.maximum(mask.sum(), 1.0)
    sensory = (
        channel_weights[0] * mx.sum(mask * (mean_hat - mean_feel) ** 2)
        + channel_weights[1] * mx.sum(mask * (min_hat - min_feel) ** 2)
    ) / weight
    gate = mx.sigmoid(body.gate_raw)
    return (
        sensory
        + continuity_weight * continuity / weight
        + sparsity * mx.sum(gate)
    )


def fit_listener_model(stream: DevelopmentStream, *, smoothing: float = 0.5) -> mx.array:
    """The maximum-likelihood word-to-consequence model, in closed form.

    Probe57 learned this table by gradient descent alongside the body. It is a
    plain categorical likelihood with no coupling to the latent, so its exact
    estimate is available directly -- and keeping it out of the body's gradient
    stops it from consuming the whole clipped gradient budget.
    """

    counts = np.full((len(VOCAB), SURFACE_COUNT + 1), smoothing, dtype=np.float64)
    for token, target in zip(stream.tokens, stream.listener_targets):
        counts[int(token), int(target)] += 1.0
    totals = np.maximum(counts.sum(axis=1, keepdims=True), 1e-12)
    probabilities = np.maximum(counts / totals, 1e-12)
    return mx.array(np.log(probabilities).astype(np.float32))


# ---------------------------------------------------------------------------
# One fixed developmental stream
# ---------------------------------------------------------------------------


@dataclass
class DevelopmentStream:
    """One lived 80,000-tick developmental experience, tensorized once."""

    durations: np.ndarray  # (lives, steps)
    moves: np.ndarray
    uptake: np.ndarray  # (lives, steps, surfaces)
    shock: np.ndarray
    mask: np.ndarray
    mean_feel: np.ndarray
    min_feel: np.ndarray
    tokens: np.ndarray  # (samples,)
    listener_targets: np.ndarray
    true_needs: np.ndarray  # (lives, steps, 4) -- audit only, never fitted
    true_birth: np.ndarray  # (lives, 4) -- audit only, never fitted
    ticks: int
    world: str

    def split(self, holdout_fraction: float) -> tuple[DevelopmentStream, DevelopmentStream]:
        lives = self.durations.shape[0]
        cut = max(1, int(round(lives * (1.0 - holdout_fraction))))
        listener_cut = int(round(len(self.tokens) * (1.0 - holdout_fraction)))
        return (
            self._slice(slice(0, cut), slice(0, listener_cut)),
            self._slice(slice(cut, lives), slice(listener_cut, len(self.tokens))),
        )

    def _slice(self, lives: slice, samples: slice) -> DevelopmentStream:
        return DevelopmentStream(
            durations=self.durations[lives],
            moves=self.moves[lives],
            uptake=self.uptake[lives],
            shock=self.shock[lives],
            mask=self.mask[lives],
            mean_feel=self.mean_feel[lives],
            min_feel=self.min_feel[lives],
            tokens=self.tokens[samples],
            listener_targets=self.listener_targets[samples],
            true_needs=self.true_needs[lives],
            true_birth=self.true_birth[lives],
            ticks=int(self.mask[lives].sum()),
            world=self.world,
        )

    def arrays(self) -> tuple[mx.array, ...]:
        return (
            mx.array(self.durations),
            mx.array(self.moves),
            mx.array(self.uptake),
            mx.array(self.shock),
            mx.array(self.mask),
            mx.array(self.mean_feel),
            mx.array(self.min_feel),
        )

    def shuffled_feelings(self, seed: int) -> DevelopmentStream:
        """Control: the sensations no longer belong to the ticks that caused them."""

        rng = np.random.default_rng(seed)
        mean_feel = self.mean_feel.copy()
        min_feel = self.min_feel.copy()
        for life in range(mean_feel.shape[0]):
            length = int(self.mask[life].sum())
            order = rng.permutation(length)
            mean_feel[life, :length] = mean_feel[life, order]
            min_feel[life, :length] = min_feel[life, order]
        return replace(self, mean_feel=mean_feel, min_feel=min_feel)


def _token_sampler(config: DiscoveredSelfConfig) -> tuple[np.ndarray, np.ndarray]:
    need_ids = {TOKEN_TO_ID[NEED_TO_REPORT_WORD[need]] for need in REPORT_NEEDS}
    weights = np.full((len(VOCAB),), 0.0, dtype=np.float64)
    rest = 1.0 - 3 * config.need_word_probability
    others = len(VOCAB) - len(need_ids)
    for token in range(len(VOCAB)):
        weights[token] = (
            config.need_word_probability if token in need_ids else rest / others
        )
    return np.arange(len(VOCAB)), weights


def collect_development_stream(
    model: OrganismModel,
    organism: OrganismConfig,
    config: DiscoveredSelfConfig,
    *,
    world_name: str,
    seed_base: int,
    log_every_lives: int = 200,
) -> DevelopmentStream:
    """Live one fixed developmental stream; fitting then reads only this."""

    frozen = DIMENSION_SWEEP[world_name]
    tokens, weights = _token_sampler(config)
    token_rng = np.random.default_rng(seed_base + 37_000_003)
    lives: list[dict[str, np.ndarray]] = []
    listener_tokens: list[int] = []
    listener_targets: list[int] = []
    ticks = 0
    life_index = 0
    while ticks < config.development_ticks:
        seed = seed_base + life_index
        world = make_report_world(
            organism, seed=seed, listener_mode="grounded", frozen_needs=frozen
        )
        packet = world.reset(seed)
        motor_hidden: mx.array | None = None
        motor_rng = Random(seed + 39_000_003)
        rows: list[tuple[float, float, np.ndarray, np.ndarray, float, float]] = []
        truths: list[tuple[float, ...]] = []
        birth_truth = tuple(
            getattr(world.grid.needs, name) for name in BODY_NEEDS
        )
        while ticks < config.development_ticks:
            action, motor_hidden = _sample_motor_action(
                model, packet, motor_hidden, motor_rng
            )
            token = int(token_rng.choice(tokens, p=weights))
            world.hear((token, PAD_ID))
            remaining = config.development_ticks - ticks
            world.grid.max_steps = min(
                world.grid.max_steps, world.grid.step_count + remaining
            )
            before = packet
            after, _, terminated, truncated, info = execute_agent_action(
                world,
                packet,
                action,
                consume_options=organism.consume_options,
                inspect_options=organism.inspect_options,
            )
            duration, moves, uptake, shock = _public_transition_features(
                organism, before, action, after
            )
            # The only two things the learner is told about its body. Both are
            # the world's own quantities: the mean is its reward signal and the
            # minimum is what kills it. Neither names an axis or a count.
            needs = world.grid.needs
            rows.append(
                (
                    duration,
                    moves,
                    uptake,
                    shock,
                    float(needs.mean_viability()),
                    float(needs.viability()),
                )
            )
            truths.append(tuple(getattr(needs, name) for name in BODY_NEEDS))
            target = _listener_target(organism, before, after)
            if target is not None:
                listener_tokens.append(token)
                listener_targets.append(target)
            ticks += int(info["duration"])
            packet = after
            if terminated or truncated or ticks >= config.development_ticks:
                break
        if rows:
            lives.append(
                {
                    "durations": np.asarray([row[0] for row in rows], dtype=np.float32),
                    "moves": np.asarray([row[1] for row in rows], dtype=np.float32),
                    "uptake": np.stack([row[2] for row in rows]),
                    "shock": np.stack([row[3] for row in rows]),
                    "mean_feel": np.asarray([row[4] for row in rows], dtype=np.float32),
                    "min_feel": np.asarray([row[5] for row in rows], dtype=np.float32),
                    "true_needs": np.asarray(truths, dtype=np.float32),
                    "true_birth": np.asarray(birth_truth, dtype=np.float32),
                }
            )
        life_index += 1
        if log_every_lives > 0 and life_index % log_every_lives == 0:
            print(
                f"  {world_name} stream: {ticks}/{config.development_ticks} ticks, "
                f"{life_index} lives"
            )
    steps = max(len(life["durations"]) for life in lives)
    count = len(lives)

    def pad(name: str, width: int | None = None) -> np.ndarray:
        shape = (count, steps) if width is None else (count, steps, width)
        out = np.zeros(shape, dtype=np.float32)
        for index, life in enumerate(lives):
            length = len(life["durations"])
            out[index, :length] = life[name]
        return out

    mask = np.zeros((count, steps), dtype=np.float32)
    for index, life in enumerate(lives):
        mask[index, : len(life["durations"])] = 1.0
    return DevelopmentStream(
        durations=pad("durations"),
        moves=pad("moves"),
        uptake=pad("uptake", SURFACE_COUNT),
        shock=pad("shock", SURFACE_COUNT),
        mask=mask,
        mean_feel=pad("mean_feel"),
        min_feel=pad("min_feel"),
        tokens=np.asarray(listener_tokens, dtype=np.int32),
        listener_targets=np.asarray(listener_targets, dtype=np.int32),
        true_needs=pad("true_needs", len(BODY_NEEDS)),
        true_birth=np.stack([life["true_birth"] for life in lives]),
        ticks=ticks,
        world=world_name,
    )


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def _channel_weights(channels: str) -> mx.array:
    if channels not in {"both", "mean", "min"}:
        raise ValueError(f"Unknown sensory channel set: {channels}.")
    # Withholding a sensation drops its term; it never zeroes its target.
    return mx.array(
        {"both": (1.0, 1.0), "mean": (1.0, 0.0), "min": (0.0, 1.0)}[channels],
        dtype=mx.float32,
    )


def _run_identification(
    problem: _IdentificationProblem,
    stream: DevelopmentStream,
    *,
    steps: int,
    learning_rate: float,
    sparsity: float,
    channels: str,
    keep: np.ndarray | None,
    continuity_weight: float = 1.0,
    log_every: int = 0,
    label: str = "fit",
) -> list[float]:
    optimizer = optim.Adam(learning_rate=learning_rate)
    loss_and_grad = nn.value_and_grad(problem, _identification_loss)
    arrays = stream.arrays()
    keep_array = mx.ones((problem.body.latent_size,)) if keep is None else mx.array(
        keep.astype(np.float32)
    )
    extras = (
        mx.array(sparsity),
        _channel_weights(channels),
        keep_array,
        mx.array(continuity_weight),
    )
    losses: list[float] = []
    for step in range(steps):
        loss, grads = loss_and_grad(problem, *arrays, *extras)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        optimizer.update(problem, grads)
        mx.eval(problem.parameters(), optimizer.state, loss)
        losses.append(float(loss))
        if log_every > 0 and (step + 1) % log_every == 0:
            print(f"  {label} step {step + 1}/{steps}: loss {losses[-1]:.6f}")
    return losses


def _respread_shooting_states(
    shoot: mx.array, segments: int, steps: int
) -> mx.array:
    """Carry one stage's segment states into a stage with fewer, longer segments."""

    previous = shoot.shape[1]
    if previous == segments:
        return shoot
    window = max(1, steps // max(1, segments))
    previous_window = max(1, steps // max(1, previous))
    picks = [
        min(previous - 1, (segment * window) // previous_window)
        for segment in range(segments)
    ]
    return mx.stack([shoot[:, pick] for pick in picks], axis=1)


def infer_births(
    body: DiscoveredSelfModel,
    stream: DevelopmentStream,
    config: DiscoveredSelfConfig,
    *,
    channels: str = "both",
    keep: np.ndarray | None = None,
    warm_start: mx.array | None = None,
    steps: int | None = None,
) -> mx.array:
    """Infer held-out initial conditions with the body's structure frozen.

    Evaluating a state-space model on unseen sequences requires their initial
    condition. It is estimated here from the same two sensations and nothing
    else; no parameter of the body moves.
    """

    problem = _IdentificationProblem(
        body, stream.durations.shape[0], stream.durations.shape[1], window=0
    )
    if warm_start is not None:
        problem.shoot_raw = mx.array(warm_start)[:, None, :]
    problem.body.freeze()
    _run_identification(
        problem,
        stream,
        steps=config.birth_inference_steps if steps is None else steps,
        learning_rate=config.birth_learning_rate,
        sparsity=0.0,
        channels=channels,
        keep=keep,
    )
    problem.body.unfreeze()
    return problem.shoot_raw[:, 0]


def fit_discovered_self(
    stream: DevelopmentStream,
    config: DiscoveredSelfConfig,
    *,
    seed: int,
    channels: str = "both",
    log_every: int = 200,
) -> tuple[DiscoveredSelfModel, DevelopmentStream, dict[str, float]]:
    """Fit the latent body to one fixed stream by repeated passes over it."""

    train, holdout = stream.split(config.holdout_fraction)
    body = DiscoveredSelfModel(latent_size=config.latent_size, seed=seed)
    body.listener_logits = fit_listener_model(stream)
    lives, steps = train.durations.shape
    losses: list[float] = []
    shoot: mx.array | None = None
    # Multiple shooting first, on short segments, then progressively longer ones
    # until the last stage is the honest single-shot trajectory.
    for window, stage_steps in config.shooting_schedule:
        problem = _IdentificationProblem(body, lives, steps, window=window)
        if shoot is not None:
            problem.shoot_raw = _respread_shooting_states(
                shoot, problem.segments, steps
            )
        losses += _run_identification(
            problem,
            train,
            steps=stage_steps,
            learning_rate=config.learning_rate,
            sparsity=config.sparsity,
            channels=channels,
            keep=None,
            continuity_weight=config.continuity_weight,
            log_every=log_every,
            label=f"fit window={window or 'full'}",
        )
        shoot = problem.shoot_raw
    # The deployed organism has no per-life reading, so its birth belief is the
    # population of inferred births, which is all the identification leaves.
    mean_birth = mx.clip(mx.mean(mx.sigmoid(shoot[:, 0]), axis=0), 1e-4, 1 - 1e-4)
    body.birth_raw = mx.log(mean_birth / (1.0 - mean_birth))
    mx.eval(body.parameters())
    metrics = {
        "fit_lives": float(train.durations.shape[0]),
        "holdout_lives": float(holdout.durations.shape[0]),
        "development_ticks": float(stream.ticks),
        "listener_samples": float(len(stream.tokens)),
        "first_loss": losses[0],
        "final_loss": losses[-1],
        **holdout_metrics(body, holdout, config, channels=channels),
    }
    return body, holdout, metrics


def _masked_channel_errors(
    body: DiscoveredSelfModel,
    stream: DevelopmentStream,
    births: mx.array,
    keep: np.ndarray | None = None,
) -> tuple[float, float]:
    durations, moves, uptake, shock, mask, mean_feel, min_feel = stream.arrays()
    states = body.rollout(durations, moves, uptake, shock, births=mx.sigmoid(births))
    keep_array = None if keep is None else mx.array(keep.astype(np.float32))
    mean_hat, min_hat = body.feelings(states, keep=keep_array)
    weight = mx.maximum(mask.sum(), 1.0)
    mean_mse = mx.sum(mask * (mean_hat - mean_feel) ** 2) / weight
    min_mse = mx.sum(mask * (min_hat - min_feel) ** 2) / weight
    mx.eval(mean_mse, min_mse)
    return float(mean_mse), float(min_mse)


def _joint_rmse(mean_mse: float, min_mse: float, channels: str) -> float:
    if channels == "mean":
        return float(np.sqrt(mean_mse))
    if channels == "min":
        return float(np.sqrt(min_mse))
    return float(np.sqrt(0.5 * (mean_mse + min_mse)))


def holdout_metrics(
    body: DiscoveredSelfModel,
    holdout: DevelopmentStream,
    config: DiscoveredSelfConfig,
    *,
    channels: str = "both",
    births: mx.array | None = None,
) -> dict[str, float]:
    if births is None:
        births = infer_births(body, holdout, config, channels=channels)
    mean_mse, min_mse = _masked_channel_errors(body, holdout, births)
    mask = holdout.mask
    weight = max(1.0, float(mask.sum()))
    null_mean = float((mask * holdout.mean_feel).sum() / weight)
    null_min = float((mask * holdout.min_feel).sum() / weight)
    null_mean_mse = float((mask * (holdout.mean_feel - null_mean) ** 2).sum() / weight)
    null_min_mse = float((mask * (holdout.min_feel - null_min) ** 2).sum() / weight)
    return {
        "holdout_mean_rmse": float(np.sqrt(mean_mse)),
        "holdout_min_rmse": float(np.sqrt(min_mse)),
        "holdout_rmse": _joint_rmse(mean_mse, min_mse, channels),
        "holdout_null_rmse": _joint_rmse(null_mean_mse, null_min_mse, channels),
    }


# ---------------------------------------------------------------------------
# Recovered effective dimension
# ---------------------------------------------------------------------------


def recovered_dimension(
    body: DiscoveredSelfModel,
    holdout: DevelopmentStream,
    config: DiscoveredSelfConfig,
    *,
    channels: str = "both",
) -> dict[str, object]:
    """Greedy backward elimination on held-out lives, as preregistered.

    Every candidate deletion gets its held-out initial conditions re-inferred,
    warm-started from the retained model's. Otherwise a dimension would be
    charged for initial conditions estimated on the assumption it was there.
    """

    latent = config.latent_size
    births = infer_births(body, holdout, config, channels=channels)
    rmse_full = _joint_rmse(
        *_masked_channel_errors(body, holdout, births), channels
    )
    reference = holdout_metrics(
        body, holdout, config, channels=channels, births=births
    )
    rmse_null = reference["holdout_null_rmse"]
    tolerance = rmse_full + config.dimension_tolerance * (rmse_null - rmse_full)
    keep = np.ones((latent,), dtype=np.float32)
    curve: list[tuple[int, float, int]] = [(latent, rmse_full, -1)]
    order: list[int] = []
    for _ in range(latent - 1):
        best_dim = -1
        best_rmse = float("inf")
        best_births: mx.array | None = None
        for dim in range(latent):
            if keep[dim] == 0.0:
                continue
            trial = keep.copy()
            trial[dim] = 0.0
            trial_births = infer_births(
                body,
                holdout,
                config,
                channels=channels,
                keep=trial,
                warm_start=births,
                steps=config.ablation_inference_steps,
            )
            rmse = _joint_rmse(
                *_masked_channel_errors(body, holdout, trial_births, keep=trial),
                channels,
            )
            if rmse < best_rmse:
                best_rmse = rmse
                best_dim = dim
                best_births = trial_births
        keep[best_dim] = 0.0
        births = best_births if best_births is not None else births
        order.append(best_dim)
        curve.append((int(keep.sum()), best_rmse, best_dim))
    retained = [size for size, rmse, _ in curve if rmse <= tolerance]
    effective = min(retained) if retained else latent
    surviving = [dim for dim in range(latent) if dim not in order[: latent - effective]]
    return {
        "effective_dimension": float(effective),
        "rmse_full": rmse_full,
        "rmse_null": rmse_null,
        "rmse_tolerance": tolerance,
        "elimination_curve": [
            {"retained": size, "rmse": rmse, "removed": removed}
            for size, rmse, removed in curve
        ],
        "elimination_order": order,
        "surviving_dimensions": surviving,
    }


# ---------------------------------------------------------------------------
# Which discovered dimension is which bodily variable
# ---------------------------------------------------------------------------


def _concentration(
    vector: np.ndarray, effective: np.ndarray | None = None
) -> tuple[int, float]:
    """Where an effect goes, as a share of where it could have gone.

    Restricted to the effective dimensions. A dimension the sensations cannot
    reach has no identified parameters at all -- its uptake, shock and movement
    entries sit wherever initialization left them -- so including it would
    measure initialization noise rather than discovery.
    """

    magnitude = np.abs(vector)
    if effective is not None:
        magnitude = magnitude * effective
    total = float(magnitude.sum())
    if total <= 0.0:
        return -1, 0.0
    index = int(np.argmax(magnitude))
    return index, float(magnitude[index] / total)


def parameter_mapping(
    model: DiscoveredSelfModel, survivors: list[int] | None = None
) -> dict[str, object]:
    """Read the learned causal parameters for a need-to-dimension map."""

    params = {name: np.asarray(value, dtype=np.float64) for name, value in model.body_parameters().items()}
    effective = np.zeros((model.latent_size,))
    if survivors is None:
        effective[:] = 1.0
    else:
        effective[list(survivors)] = 1.0
    uptake = params["uptake"]
    shock = params["shock"]
    uptake_map: dict[str, int] = {}
    uptake_share: dict[str, float] = {}
    shock_map: dict[str, int] = {}
    shock_share: dict[str, float] = {}
    uptake_share_all: dict[str, float] = {}
    shock_share_all: dict[str, float] = {}
    for need in REPORT_NEEDS:
        rows = [
            SURFACES.index(HELP_SURFACES[(need, large)]) for large in (False, True)
        ]
        combined = uptake[rows].sum(axis=0)
        dim, share = _concentration(combined, effective)
        uptake_map[need] = dim
        uptake_share[need] = share
        uptake_share_all[need] = _concentration(combined)[1]
        shock_row = shock[SURFACES.index(SHOCK_SURFACES[need])]
        dim, share = _concentration(shock_row, effective)
        shock_map[need] = dim
        shock_share[need] = share
        shock_share_all[need] = _concentration(shock_row)[1]
    move = params["move"]
    move_dim, move_share = _concentration(move, effective)
    return {
        "uptake_map": uptake_map,
        "uptake_concentration": uptake_share,
        "shock_map": shock_map,
        "shock_concentration": shock_share,
        "uptake_concentration_all_dimensions": uptake_share_all,
        "shock_concentration_all_dimensions": shock_share_all,
        "move_dimension": move_dim,
        "move_concentration": move_share,
        "move_concentration_all_dimensions": _concentration(move)[1],
        "effective_dimensions": [int(dim) for dim in np.flatnonzero(effective)],
        "drift": params["drift"].tolist(),
        "move_cost": move.tolist(),
        "gate": params["gate"].tolist(),
        "birth_prior": params["birth"].tolist(),
        "uptake_by_surface": {
            surface: uptake[index].tolist() for index, surface in enumerate(SURFACES)
        },
        "shock_by_surface": {
            surface: shock[index].tolist() for index, surface in enumerate(SURFACES)
        },
    }


# ---------------------------------------------------------------------------
# Deployment: the belief, the planner, the lesions
# ---------------------------------------------------------------------------


@dataclass
class DeployedSelf:
    """The learned body, as numpy, for tick-by-tick deployment."""

    drift: np.ndarray
    move: np.ndarray
    uptake: np.ndarray
    shock: np.ndarray
    gate: np.ndarray
    birth: np.ndarray
    listener: np.ndarray

    @classmethod
    def from_model(cls, model: DiscoveredSelfModel) -> DeployedSelf:
        params = model.body_parameters()
        logits = np.asarray(model.listener_logits, dtype=np.float64)
        shifted = logits - logits.max(axis=-1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        return cls(
            drift=np.asarray(params["drift"], dtype=np.float64),
            move=np.asarray(params["move"], dtype=np.float64),
            uptake=np.asarray(params["uptake"], dtype=np.float64),
            shock=np.asarray(params["shock"], dtype=np.float64),
            gate=np.asarray(params["gate"], dtype=np.float64),
            birth=np.asarray(params["birth"], dtype=np.float64),
            listener=probabilities,
        )

    def advance(
        self,
        belief: np.ndarray,
        features: tuple[float, float, np.ndarray, np.ndarray],
    ) -> np.ndarray:
        duration, moves, uptake, shock = features
        delta = (
            duration * self.drift
            + moves * self.move
            + uptake @ self.uptake
            + shock @ self.shock
        )
        return np.clip(belief + delta, 0.0, 1.0)

    def sensed(self, belief: np.ndarray) -> np.ndarray:
        return self.gate * belief + (1.0 - self.gate)

    def token(self, belief: np.ndarray, *, step_count: int, help_period: int) -> int:
        """Probe60's promoted outcome-aware objective, in discovered latent space.

        Reads only the belief, the public clock, and its own learned parameters.
        No need label, no help-surface table, no true body.
        """

        ticks_to_help = help_period - (step_count % help_period)
        pre = np.clip(belief + ticks_to_help * self.drift, 0.0, 1.0)
        branches = np.concatenate(
            [np.clip(pre[None, :] + self.uptake, 0.0, 1.0), pre[None, :]], axis=0
        )
        branch_scores = np.min(self.gate * branches + (1.0 - self.gate), axis=-1)
        return int(np.argmax(self.listener @ branch_scores))


def _life_seeds(seed_base: int, lives: int) -> list[int]:
    return [seed_base + life for life in range(lives)]


def evaluate_discovered_planner(
    model: OrganismModel,
    organism: OrganismConfig,
    deployed: DeployedSelf,
    *,
    world_name: str,
    lives: int,
    seed_base: int,
    listener_mode: str = "grounded",
    belief_intervention: str = "none",
    frozen_dimension: int | None = None,
    fixed_word: str | None = None,
    mute_organism: bool = False,
) -> dict[str, object]:
    """Deploy the discovered self-model and measure what it says and survives."""

    frozen = DIMENSION_SWEEP[world_name]
    survived = 0
    steps_sum = 0
    said_need = 0
    truthful = 0
    per_need_true = {need: 0 for need in REPORT_NEEDS}
    per_need_hit = {need: 0 for need in REPORT_NEEDS}
    word_counts = {need: 0 for need in REPORT_NEEDS}
    deaths = {need: 0 for need in BODY_NEEDS}
    total_ticks = 0
    for seed in _life_seeds(seed_base, lives):
        world = make_report_world(
            organism, seed=seed, listener_mode=listener_mode, frozen_needs=frozen
        )
        packet = world.reset(seed)
        belief = deployed.birth.copy()
        birth_belief = belief.copy()
        motor_hidden: mx.array | None = None
        motor_rng = Random(seed + 59_000_003)
        while True:
            planning_belief = belief
            if belief_intervention == "zero":
                planning_belief = np.zeros_like(belief)
            elif frozen_dimension is not None:
                planning_belief = belief.copy()
                planning_belief[frozen_dimension] = birth_belief[frozen_dimension]
            if fixed_word is not None:
                token = TOKEN_TO_ID[NEED_TO_REPORT_WORD[fixed_word]]
            else:
                token = deployed.token(
                    planning_belief,
                    step_count=packet.step_count,
                    help_period=organism.report.help_period,
                )
            said = (token, PAD_ID)
            tick = packet.step_count
            true_lowest = world.lowest_need()
            word = heard_need(said)
            if tick >= FIDELITY_WARMUP:
                total_ticks += 1
                per_need_true[true_lowest] += 1
                if word is not None:
                    said_need += 1
                    word_counts[word] += 1
                    correct = int(word == true_lowest)
                    truthful += correct
                    per_need_hit[true_lowest] += correct
            action, motor_hidden = _sample_motor_action(
                model, packet, motor_hidden, motor_rng
            )
            world.hear(None if mute_organism else said)
            before = packet
            packet, _, terminated, truncated, info = execute_agent_action(
                world,
                packet,
                action,
                consume_options=organism.consume_options,
                inspect_options=organism.inspect_options,
            )
            belief = deployed.advance(
                belief,
                _public_transition_features(organism, before, action, packet),
            )
            steps_sum += int(info["duration"])
            if terminated or truncated:
                survived += int(not terminated)
                cause = info.get("death_need")
                if terminated and cause is not None:
                    deaths[str(cause)] += 1
                break
    return {
        "lives": lives,
        "world": world_name,
        "listener_mode": listener_mode,
        "belief_intervention": belief_intervention,
        "frozen_dimension": -1 if frozen_dimension is None else frozen_dimension,
        "survival": survived / lives,
        "mean_life_steps": steps_sum / lives,
        "need_word_ticks": said_need,
        "report_fidelity": truthful / max(1, said_need),
        "recall": {
            need: per_need_hit[need] / max(1, per_need_true[need])
            for need in REPORT_NEEDS
        },
        "true_need_ticks": dict(per_need_true),
        "word_distribution": {
            need: word_counts[need] / max(1, sum(word_counts.values()))
            for need in REPORT_NEEDS
        },
        "death_causes": dict(deaths),
        "scored_ticks": total_ticks,
    }


def audit_body_tracking(
    model: OrganismModel,
    organism: OrganismConfig,
    deployed: DeployedSelf,
    need_map: dict[str, int],
    *,
    world_name: str,
    lives: int,
    seed_base: int,
) -> dict[str, float]:
    """How well the discovered latent tracks the body it was never shown."""

    frozen = DIMENSION_SWEEP[world_name]
    errors = {need: 0.0 for need in REPORT_NEEDS}
    samples = 0
    argmin_hits = 0
    for seed in _life_seeds(seed_base, lives):
        world = make_report_world(
            organism, seed=seed, listener_mode="grounded", frozen_needs=frozen
        )
        packet = world.reset(seed)
        belief = deployed.birth.copy()
        motor_hidden: mx.array | None = None
        motor_rng = Random(seed + 71_000_003)
        while True:
            if packet.step_count >= FIDELITY_WARMUP:
                truth = {need: getattr(world.grid.needs, need) for need in REPORT_NEEDS}
                for need in REPORT_NEEDS:
                    errors[need] += abs(belief[need_map[need]] - truth[need])
                predicted = min(
                    REPORT_NEEDS, key=lambda need: belief[need_map[need]]
                )
                argmin_hits += int(predicted == world.lowest_need())
                samples += 1
            token = deployed.token(
                belief,
                step_count=packet.step_count,
                help_period=organism.report.help_period,
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
                consume_options=organism.consume_options,
                inspect_options=organism.inspect_options,
            )
            belief = deployed.advance(
                belief,
                _public_transition_features(organism, before, action, packet),
            )
            if terminated or truncated:
                break
    total = max(1, samples)
    per_need = {need: errors[need] / total for need in REPORT_NEEDS}
    return {
        "samples": float(samples),
        "mean_absolute_need_error": float(np.mean(list(per_need.values()))),
        **{f"error_{need}": value for need, value in per_need.items()},
        "lowest_need_accuracy": argmin_hits / total,
    }


def audit_forced_grant_mapping(
    model: OrganismModel,
    organism: OrganismConfig,
    deployed: DeployedSelf,
    *,
    world_name: str,
    lives: int,
    seed_base: int,
    survivors: list[int] | None = None,
) -> dict[str, object]:
    """Live paired intervention: force which resource arrives, watch the latent."""

    frozen = DIMENSION_SWEEP[world_name]
    latent = deployed.birth.shape[0]
    effective = np.zeros((latent,))
    if survivors is None:
        effective[:] = 1.0
    else:
        effective[list(survivors)] = 1.0
    responses = np.zeros((len(REPORT_NEEDS), latent), dtype=np.float64)
    counted = 0
    period = organism.report.help_period
    for seed in _life_seeds(seed_base, lives):
        branch_beliefs = []
        for need in REPORT_NEEDS:
            world = make_report_world(
                organism,
                seed=seed,
                listener_mode="grounded",
                frozen_needs=frozen,
                shock_probability=0.0,
            )
            packet = world.reset(seed)
            belief = deployed.birth.copy()
            world.force_next_help_need(need)
            world.force_next_help_portion(large=True)
            plan = [ACTION_WAIT] * period + [ACTION_CONSUME]
            for action in plan:
                world.hear((PAD_ID, PAD_ID))
                before = packet
                packet, _, terminated, truncated, _ = execute_agent_action(
                    world,
                    packet,
                    action,
                    consume_options=organism.consume_options,
                    inspect_options=organism.inspect_options,
                )
                belief = deployed.advance(
                    belief,
                    _public_transition_features(organism, before, action, packet),
                )
                if terminated or truncated:
                    break
            branch_beliefs.append(belief)
        stacked = np.stack(branch_beliefs)
        responses += stacked - stacked.mean(axis=0, keepdims=True)
        counted += 1
    responses /= max(1, counted)
    forced_map: dict[str, int] = {}
    forced_share: dict[str, float] = {}
    forced_share_all: dict[str, float] = {}
    for index, need in enumerate(REPORT_NEEDS):
        positive = np.clip(responses[index], 0.0, None)
        dim, share = _concentration(positive, effective)
        forced_map[need] = dim
        forced_share[need] = share
        forced_share_all[need] = _concentration(positive)[1]
    return {
        "forced_map": forced_map,
        "forced_concentration": forced_share,
        "forced_concentration_all_dimensions": forced_share_all,
        "response_matrix": responses.tolist(),
        "paired_lives": float(counted),
    }


# ---------------------------------------------------------------------------
# The rates the body actually has
# ---------------------------------------------------------------------------

# The world's own constants, read from the frozen ecology for scoring only.
# Nothing in the learner ever sees these.
TRUE_METABOLISM = {
    "food": "food_metabolism",
    "water": "water_metabolism",
    "energy": "energy_metabolism",
    "safety": "safety_metabolism",
}


def audit_rate_recovery(
    model: DiscoveredSelfModel,
    organism: OrganismConfig,
    mapping: dict[str, object],
    survivors: list[int],
    *,
    world_name: str,
) -> dict[str, object]:
    """Compare each discovered rate with the rate the world actually uses."""

    params = {
        name: np.asarray(value, dtype=np.float64)
        for name, value in model.body_parameters().items()
    }
    drift = -params["drift"]
    move = -params["move"]
    uptake = params["uptake"]
    shock = -params["shock"]
    report = organism.report
    frozen = set(DIMENSION_SWEEP[world_name])
    need_map: dict[str, int] = mapping["uptake_map"]  # type: ignore[assignment]
    recovered: dict[str, float] = {}
    truth: dict[str, float] = {}
    ratio: dict[str, float] = {}
    for need in REPORT_NEEDS:
        if need in frozen:
            continue
        dim = need_map[need]
        true_rate = float(getattr(report, TRUE_METABOLISM[need]))
        recovered[need] = float(drift[dim])
        truth[need] = true_rate
        ratio[need] = float(drift[dim] / true_rate) if true_rate else float("nan")
    mapped = {need_map[need] for need in REPORT_NEEDS if need not in frozen}
    extras = [dim for dim in survivors if dim not in mapped]
    mapped_uptake = [float(uptake[:, dim].sum()) for dim in mapped] or [0.0]
    extra_rows = [
        {
            "dimension": dim,
            "drift": float(drift[dim]),
            "uptake_total": float(uptake[:, dim].sum()),
            "shock_total": float(shock[:, dim].sum()),
            "uptake_ratio_to_smallest_mapped": float(
                uptake[:, dim].sum() / max(1e-9, min(mapped_uptake))
            ),
        }
        for dim in extras
    ]
    true_safety = 0.0 if "safety" in frozen else float(report.safety_metabolism)
    # The unannounced variable, if it was found: drift-only, no resource
    # restores it, no word requests it, and probe57 has no slot for it.
    silent = [
        row
        for row in extra_rows
        if row["uptake_ratio_to_smallest_mapped"] <= 0.10
        and 0.001 <= row["drift"] <= 0.004
    ]
    move_dim = int(mapping["move_dimension"])
    energy_dim = need_map.get("energy", -1)
    return {
        "recovered_metabolism": recovered,
        "true_metabolism": truth,
        "metabolism_ratio": ratio,
        "metabolism_within_25pct": {
            need: bool(0.75 <= value <= 1.25) for need, value in ratio.items()
        },
        "move_dimension": move_dim,
        "move_concentration": mapping["move_concentration"],
        "move_on_energy_dimension": bool(move_dim == energy_dim),
        "extra_effective_dimensions": extra_rows,
        "silent_variable_found": bool(len(silent) == 1),
        "silent_variable_drift": silent[0]["drift"] if len(silent) == 1 else float("nan"),
        "true_silent_metabolism": true_safety,
    }


# ---------------------------------------------------------------------------
# The privilege ceiling: probe57's supervision on the same stream
# ---------------------------------------------------------------------------


def fit_labeled_reference(
    stream: DevelopmentStream, config: DiscoveredSelfConfig, *, seed: int
) -> dict[str, float]:
    """Probe57's three-axis, body-labelled, teacher-forced fit, for comparison.

    This is the privilege the discovered model gives up: the true per-axis body
    vector at every transition, a latent that is exactly three-dimensional, and
    a one-step target that never has to survive a rollout.
    """

    train, holdout = stream.split(config.holdout_fraction)
    body = DiscoveredSelfModel(latent_size=3, seed=seed)
    optimizer = optim.Adam(learning_rate=1e-2)

    def loss_fn(model: DiscoveredSelfModel, *arrays: mx.array) -> mx.array:
        durations, moves, uptake, shock, mask, current, target = arrays
        params = model.body_parameters()
        delta = (
            durations[..., None] * params["drift"]
            + moves[..., None] * params["move"]
            + uptake @ params["uptake"]
            + shock @ params["shock"]
        )
        predicted = mx.clip(current + delta, 0.0, 1.0)
        return mx.sum(mask[..., None] * (predicted - target) ** 2) / mx.maximum(
            mask.sum(), 1.0
        )

    def tensors(split: DevelopmentStream) -> tuple[mx.array, ...]:
        truth = split.true_needs[:, :, :3]
        birth = split.true_birth[:, None, :3]
        current = np.concatenate([birth, truth[:, :-1]], axis=1)
        return (
            mx.array(split.durations),
            mx.array(split.moves),
            mx.array(split.uptake),
            mx.array(split.shock),
            mx.array(split.mask),
            mx.array(current),
            mx.array(truth),
        )

    loss_and_grad = nn.value_and_grad(body, loss_fn)
    arrays = tensors(train)
    for _ in range(2000):
        loss, grads = loss_and_grad(body, *arrays)
        optimizer.update(body, grads)
        mx.eval(body.parameters(), optimizer.state, loss)
    # Open-loop trajectory error on held out lives, from the true birth body.
    truth = holdout.true_needs[:, :, :3]
    states = body.rollout(
        mx.array(holdout.durations),
        mx.array(holdout.moves),
        mx.array(holdout.uptake),
        mx.array(holdout.shock),
        births=mx.array(holdout.true_birth[:, :3]),
    )
    mx.eval(states)
    error = np.abs(np.asarray(states) - truth).mean(axis=-1)
    weight = max(1.0, float(holdout.mask.sum()))
    params = body.body_parameters()
    return {
        "labeled_teacher_forced_loss": float(loss),
        "labeled_open_loop_body_error": float((holdout.mask * error).sum() / weight),
        "labeled_drift": (-np.asarray(params["drift"])).tolist(),
    }


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def save_discovered_self(model: DiscoveredSelfModel, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        name: np.asarray(value)
        for name, value in model.parameters().items()
        if isinstance(value, mx.array)
    }
    np.savez(target, **arrays)
    target.with_suffix(".npz.json").write_text(
        json.dumps(
            {
                "latent_size": model.latent_size,
                "surface_count": model.surface_count,
            },
            indent=2,
        )
    )


# ---------------------------------------------------------------------------
# The experiment
# ---------------------------------------------------------------------------


def deployment_battery(
    parent: OrganismModel,
    organism: OrganismConfig,
    deployed: DeployedSelf,
    need_map: dict[str, int],
    survivors: list[int],
    *,
    world_name: str,
    lives: int,
    seed_base: int,
) -> list[dict[str, object]]:
    """Deploy the discovered self, then damage it one discovered variable at a time."""

    def run(label: str, **kwargs: object) -> dict[str, object]:
        row = evaluate_discovered_planner(
            parent,
            organism,
            deployed,
            world_name=world_name,
            lives=lives,
            seed_base=seed_base,
            **kwargs,  # type: ignore[arg-type]
        )
        return {"condition": label, **row}

    rows = [run("discovered_planner_grounded")]
    intact = rows[0]
    rows.append(run("listener_scrambled", listener_mode="scrambled"))
    rows.append(run("listener_mute", listener_mode="mute"))
    rows.append(run("organism_mute", mute_organism=True))
    rows.append(run("belief_zero", belief_intervention="zero"))
    for need in REPORT_NEEDS:
        rows.append(run(f"fixed_word_{need}", fixed_word=need))
    frozen = set(DIMENSION_SWEEP[world_name])
    live_needs = [need for need in REPORT_NEEDS if need not in frozen]
    intact_recall: dict[str, float] = intact["recall"]  # type: ignore[assignment]
    for need in live_needs:
        dim = need_map[need]
        lesion = run(f"freeze_dimension_{dim}", frozen_dimension=dim)
        lesion["lesioned_need"] = need
        recall: dict[str, float] = lesion["recall"]  # type: ignore[assignment]
        drops = {other: intact_recall[other] - recall[other] for other in live_needs}
        lesion["recall_drop"] = drops
        lesion["largest_drop_need"] = max(drops, key=lambda key: drops[key])
        lesion["specific"] = bool(
            lesion["largest_drop_need"] == need
            and drops[need] >= 0.30
            and all(drops[other] <= 0.10 for other in live_needs if other != need)
        )
        rows.append(lesion)
    unused = [dim for dim in range(deployed.birth.shape[0]) if dim not in survivors]
    if unused:
        dim = unused[0]
        null = run(f"freeze_unused_dimension_{dim}", frozen_dimension=dim)
        recall = null["recall"]  # type: ignore[assignment]
        drops = {
            other: abs(intact_recall[other] - recall[other]) for other in live_needs
        }
        null["recall_drop"] = drops
        null["inert"] = bool(all(value <= 0.05 for value in drops.values()))
        rows.append(null)
    return rows


def run_world_seed(
    parent: OrganismModel,
    organism: OrganismConfig,
    config: DiscoveredSelfConfig,
    *,
    world_name: str,
    seed_index: int,
    lives: int,
    channels: str = "both",
    shuffle_feelings: bool = False,
    deploy: bool = True,
    reference: bool = False,
    log: bool = True,
) -> tuple[dict[str, object], DiscoveredSelfModel]:
    """One ground-truth world, one seed: discover, then interrogate."""

    seed_base = development_seed_base(seed_index, world_name)
    stream = collect_development_stream(
        parent,
        organism,
        config,
        world_name=world_name,
        seed_base=seed_base,
        log_every_lives=0,
    )
    fitted = stream.shuffled_feelings(seed_base + 91_000_003) if shuffle_feelings else stream
    body, holdout, fit_metrics = fit_discovered_self(
        fitted, config, seed=seed_index, channels=channels, log_every=0
    )
    holdout = fitted.split(config.holdout_fraction)[1]
    dimension = recovered_dimension(body, holdout, config, channels=channels)
    survivors: list[int] = dimension["surviving_dimensions"]  # type: ignore[assignment]
    mapping = parameter_mapping(body, survivors)
    rates = audit_rate_recovery(
        body, organism, mapping, survivors, world_name=world_name
    )
    frozen = set(DIMENSION_SWEEP[world_name])
    live_needs = [need for need in REPORT_NEEDS if need not in frozen]
    need_map: dict[str, int] = mapping["uptake_map"]  # type: ignore[assignment]
    shock_map: dict[str, int] = mapping["shock_map"]  # type: ignore[assignment]
    result: dict[str, object] = {
        "world": world_name,
        "seed_index": seed_index,
        "true_dimension": len(
            replace(
                organism.report, frozen_needs=DIMENSION_SWEEP[world_name]
            ).live_needs()
        ),
        "channels": channels,
        "shuffled_feelings": shuffle_feelings,
        "sparsity": config.sparsity,
        **fit_metrics,
        **{key: value for key, value in dimension.items()},
        "mapping": mapping,
        "rates": rates,
    }
    deployed = DeployedSelf.from_model(body)
    if deploy:
        forced = audit_forced_grant_mapping(
            parent,
            organism,
            deployed,
            world_name=world_name,
            lives=min(lives, 60),
            seed_base=MAPPING_SEED_BASE + 10_000 * seed_index,
            survivors=survivors,
        )
        result["forced"] = forced
        forced_map: dict[str, int] = forced["forced_map"]  # type: ignore[assignment]
        uptake_share: dict[str, float] = mapping["uptake_concentration"]  # type: ignore[assignment]
        shock_share: dict[str, float] = mapping["shock_concentration"]  # type: ignore[assignment]
        forced_share: dict[str, float] = forced["forced_concentration"]  # type: ignore[assignment]
        mapped_dims = [need_map[need] for need in live_needs]
        result["mapping_one_to_one"] = bool(
            len(set(mapped_dims)) == len(mapped_dims)
        )
        result["mapping_agrees"] = bool(
            all(
                need_map[need] == shock_map[need] == forced_map[need]
                for need in live_needs
            )
        )
        result["mapping_concentrated"] = bool(
            all(
                min(uptake_share[need], shock_share[need], forced_share[need]) >= 0.80
                for need in live_needs
            )
        )
        result["mapping_passes"] = bool(
            result["mapping_one_to_one"]
            and result["mapping_agrees"]
            and result["mapping_concentrated"]
        )
        result["tracking"] = audit_body_tracking(
            parent,
            organism,
            deployed,
            need_map,
            world_name=world_name,
            lives=min(lives, 60),
            seed_base=DEPLOYMENT_SEED_BASE + 500_000 + 10_000 * seed_index,
        )
        result["deployment"] = deployment_battery(
            parent,
            organism,
            deployed,
            need_map,
            survivors,
            world_name=world_name,
            lives=lives,
            seed_base=DEPLOYMENT_SEED_BASE + 10_000 * seed_index,
        )
    if reference:
        result["reference"] = fit_labeled_reference(stream, config, seed=seed_index)
    if log:
        print(
            f"  {world_name} seed {seed_index}: d_eff="
            f"{dimension['effective_dimension']:.0f} (true "
            f"{result['true_dimension']}), holdout rmse "
            f"{fit_metrics['holdout_rmse']:.4f}, map={need_map}",
            flush=True,
        )
    return result, body


def select_sparsity(
    parent: OrganismModel,
    organism: OrganismConfig,
    config: DiscoveredSelfConfig,
    *,
    grid: tuple[float, ...],
    world_name: str = "K4",
    seed_index: int = 0,
) -> tuple[float, list[dict[str, float]]]:
    """The preregistered rule: largest coefficient within 5% of the best fit.

    Selection reads held-out sensory error only. No dimension, mapping, rate or
    lesion quantity is computed here.
    """

    seed_base = development_seed_base(seed_index, world_name)
    stream = collect_development_stream(
        parent, organism, config, world_name=world_name,
        seed_base=seed_base, log_every_lives=0,
    )
    rows: list[dict[str, float]] = []
    for value in grid:
        _, _, metrics = fit_discovered_self(
            stream, replace(config, sparsity=value), seed=seed_index, log_every=0
        )
        rows.append({"sparsity": value, "holdout_rmse": metrics["holdout_rmse"]})
        print(
            f"  sparsity {value:g}: holdout rmse {metrics['holdout_rmse']:.5f}",
            flush=True,
        )
    best = min(row["holdout_rmse"] for row in rows)
    eligible = [row for row in rows if row["holdout_rmse"] <= 1.05 * best]
    chosen = max(row["sparsity"] for row in eligible)
    return chosen, rows


def evaluate_gates(
    results: list[dict[str, object]], worlds: tuple[str, ...]
) -> dict[str, object]:
    """The preregistered gates, computed from the per-seed records."""

    def by_world(name: str) -> list[dict[str, object]]:
        return [row for row in results if row["world"] == name]

    dimension_rows = []
    g1_pass = True
    previous_mean = -1.0
    for name in worlds:
        rows = by_world(name)
        if not rows:
            continue
        truth = int(rows[0]["true_dimension"])
        found = [int(row["effective_dimension"]) for row in rows]
        hits = sum(1 for value in found if value == truth)
        mean = float(np.mean(found))
        dimension_rows.append(
            {
                "world": name,
                "true_dimension": truth,
                "recovered": found,
                "mean_recovered": mean,
                "seeds_matching": hits,
                "seeds": len(found),
            }
        )
        if hits < max(1, len(found) - 1):
            g1_pass = False
        if mean <= previous_mean:
            g1_pass = False
        previous_mean = mean

    def gate_seed_fraction(name: str, key: str) -> tuple[int, int]:
        rows = by_world(name)
        return sum(1 for row in rows if row.get(key)), len(rows)

    tracking = [
        row["tracking"]["mean_absolute_need_error"]  # type: ignore[index]
        for row in by_world("K3")
        if "tracking" in row
    ]
    f0_hits = sum(1 for value in tracking if value <= 0.05)
    mapping = {name: gate_seed_fraction(name, "mapping_passes") for name in ("K3", "K4")}
    rate_hits = 0
    rate_seeds = 0
    silent_hits = 0
    for row in by_world("K4"):
        rates = row["rates"]  # type: ignore[index]
        rate_seeds += 1
        within = all(rates["metabolism_within_25pct"].values())  # type: ignore[index]
        if within and rates["move_on_energy_dimension"] and float(
            rates["move_concentration"]
        ) >= 0.80:
            rate_hits += 1
        silent_hits += int(bool(rates["silent_variable_found"]))
    lesion_hits = 0
    lesion_seeds = 0
    deployment_rows = []
    for row in results:
        if "deployment" not in row:
            continue
        battery: list[dict[str, object]] = row["deployment"]  # type: ignore[assignment]
        intact = battery[0]
        lookup = {item["condition"]: item for item in battery}
        specific = [item for item in battery if "specific" in item]
        inert = [item for item in battery if "inert" in item]
        if row["world"] == "K3" and specific:
            lesion_seeds += 1
            lesion_hits += int(
                all(bool(item["specific"]) for item in specific)
                and all(bool(item["inert"]) for item in inert)
            )
        deployment_rows.append(
            {
                "world": row["world"],
                "seed_index": row["seed_index"],
                "survival": intact["survival"],
                "report_fidelity": intact["report_fidelity"],
                "scrambled_survival": lookup["listener_scrambled"]["survival"],
                "zero_belief_survival": lookup["belief_zero"]["survival"],
                "mute_survival": lookup["organism_mute"]["survival"],
                "grounded_minus_scrambled": float(intact["survival"])
                - float(lookup["listener_scrambled"]["survival"]),
            }
        )
    g5_hits = sum(
        1
        for row in deployment_rows
        if row["survival"] >= 0.75
        and row["report_fidelity"] >= 0.80
        and row["grounded_minus_scrambled"] >= 0.15
        and float(row["survival"]) - float(row["zero_belief_survival"]) >= 0.30
        and float(row["survival"]) - float(row["mute_survival"]) >= 0.30
    )
    return {
        "F0_body_tracking": {
            "errors": tracking,
            "seeds_passing": f0_hits,
            "seeds": len(tracking),
            "passed": bool(tracking and f0_hits == len(tracking)),
        },
        "G1_dimension": {
            "worlds": dimension_rows,
            "passed": bool(g1_pass and dimension_rows),
        },
        "G2_mapping": {
            "by_world": {
                name: {"seeds_passing": hits, "seeds": total}
                for name, (hits, total) in mapping.items()
            },
            "passed": bool(
                all(
                    total > 0 and hits >= max(1, total - 1)
                    for hits, total in mapping.values()
                )
            ),
        },
        "G3_rates": {
            "seeds_passing": rate_hits,
            "seeds": rate_seeds,
            "silent_variable_seeds": silent_hits,
            "passed": bool(
                rate_seeds > 0
                and rate_hits >= max(1, rate_seeds - 1)
                and silent_hits >= max(1, rate_seeds - 1)
            ),
        },
        "G4_lesion_specificity": {
            "seeds_passing": lesion_hits,
            "seeds": lesion_seeds,
            "passed": bool(
                lesion_seeds > 0 and lesion_hits >= max(1, lesion_seeds - 1)
            ),
        },
        "G5_deployment": {
            "rows": deployment_rows,
            "seeds_passing": g5_hits,
            "seeds": len(deployment_rows),
            "passed": bool(
                deployment_rows
                and g5_hits >= max(1, len(deployment_rows) - 1)
            ),
        },
    }


def _flatten(record: dict[str, object]) -> dict[str, object]:
    """One CSV row per seed and world; the nested detail stays in the JSON."""

    rates = record.get("rates", {})
    tracking = record.get("tracking", {})
    mapping = record.get("mapping", {})
    deployment = record.get("deployment", [])
    intact = deployment[0] if deployment else {}
    lookup = {item["condition"]: item for item in deployment}
    row: dict[str, object] = {
        "world": record["world"],
        "seed_index": record["seed_index"],
        "channels": record["channels"],
        "shuffled_feelings": record["shuffled_feelings"],
        "sparsity": record["sparsity"],
        "true_dimension": record["true_dimension"],
        "effective_dimension": record["effective_dimension"],
        "holdout_rmse": record["holdout_rmse"],
        "holdout_null_rmse": record["holdout_null_rmse"],
        "surviving_dimensions": record["surviving_dimensions"],
        "uptake_map": mapping.get("uptake_map"),
        "shock_map": mapping.get("shock_map"),
        "forced_map": record.get("forced", {}).get("forced_map"),
        "mapping_passes": record.get("mapping_passes"),
        "recovered_metabolism": rates.get("recovered_metabolism"),
        "metabolism_ratio": rates.get("metabolism_ratio"),
        "move_on_energy_dimension": rates.get("move_on_energy_dimension"),
        "silent_variable_found": rates.get("silent_variable_found"),
        "silent_variable_drift": rates.get("silent_variable_drift"),
        "body_error": tracking.get("mean_absolute_need_error"),
        "lowest_need_accuracy": tracking.get("lowest_need_accuracy"),
        "survival": intact.get("survival"),
        "report_fidelity": intact.get("report_fidelity"),
        "scrambled_survival": lookup.get("listener_scrambled", {}).get("survival"),
        "zero_belief_survival": lookup.get("belief_zero", {}).get("survival"),
        "mute_survival": lookup.get("organism_mute", {}).get("survival"),
    }
    if "reference" in record:
        row["labeled_open_loop_body_error"] = record["reference"][
            "labeled_open_loop_body_error"
        ]
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--lives", type=int, default=100)
    parser.add_argument("--worlds", default="K1,K2,K3,K4")
    parser.add_argument("--development-ticks", type=int, default=80_000)
    parser.add_argument("--latent-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--sparsity", type=float, default=1e-3)
    parser.add_argument("--long-schedule", action="store_true")
    parser.add_argument("--controls", action="store_true")
    args = parser.parse_args()

    parent, organism = load_organism_checkpoint(args.parent)
    schedule = (
        ((25, 3000), (100, 1500), (0, 1500))
        if args.long_schedule
        else ((25, 1500), (100, 750), (0, 750))
    )
    config = DiscoveredSelfConfig(
        latent_size=args.latent_size,
        development_ticks=args.development_ticks,
        learning_rate=args.learning_rate,
        sparsity=args.sparsity,
        shooting_schedule=schedule,
    )
    worlds = tuple(name.strip() for name in args.worlds.split(",") if name.strip())
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    for world_name in worlds:
        for seed_index in range(args.seeds):
            record, body = run_world_seed(
                parent,
                organism,
                config,
                world_name=world_name,
                seed_index=seed_index,
                lives=args.lives,
                deploy=world_name in {"K3", "K4"},
                reference=(world_name == "K4" and seed_index == 0),
            )
            results.append(record)
            save_discovered_self(
                body, run_dir / f"discovered_self_{world_name}_seed{seed_index}.npz"
            )

    controls: list[dict[str, object]] = []
    if args.controls:
        for label, kwargs in (
            ("shuffled_feelings", {"shuffle_feelings": True}),
            ("mean_channel_only", {"channels": "mean"}),
            ("min_channel_only", {"channels": "min"}),
        ):
            for seed_index in range(args.seeds):
                record, _ = run_world_seed(
                    parent,
                    organism,
                    config,
                    world_name="K4",
                    seed_index=seed_index,
                    lives=args.lives,
                    deploy=False,
                    **kwargs,  # type: ignore[arg-type]
                )
                record["control"] = label
                controls.append(record)
            for world_name in ("K1", "K2", "K3"):
                record, _ = run_world_seed(
                    parent,
                    organism,
                    config,
                    world_name=world_name,
                    seed_index=0,
                    lives=args.lives,
                    deploy=False,
                    **kwargs,  # type: ignore[arg-type]
                )
                record["control"] = label
                controls.append(record)

    gates = evaluate_gates(results, worlds)
    payload = {
        "config": {
            "latent_size": config.latent_size,
            "development_ticks": config.development_ticks,
            "learning_rate": config.learning_rate,
            "sparsity": config.sparsity,
            "shooting_schedule": config.shooting_schedule,
            "need_word_probability": config.need_word_probability,
            "dimension_tolerance": config.dimension_tolerance,
            "seeds": args.seeds,
            "lives": args.lives,
            "parent": args.parent,
        },
        "results": results,
        "controls": controls,
        "gates": gates,
    }
    (run_dir / "discovered_self.json").write_text(json.dumps(payload, indent=2, default=str))
    rows = [_flatten(record) for record in results + controls]
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with (run_dir / "discovered_self.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(json.dumps(gates, indent=2, default=str))


if __name__ == "__main__":
    main()
