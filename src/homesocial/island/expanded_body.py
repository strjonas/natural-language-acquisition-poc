"""A default-off five-axis body substrate for Probe71.

The report ecology still owns the legacy three reportable needs.  This module
does not widen those globals or old observation packets: it defines an explicit
schema and a small body transition that can be qualified before any checkpoint
or recurrent model is retrained.

The first three axes reproduce the constants and public consequence identities
of :mod:`homesocial.island.report`.  The expanded schema makes the existing
``safety`` variable reportable and adds one genuinely new variable, ``health``.
All transition functions are axis-generic; no branch silently maps "the fourth
thing" to a privileged meaning.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Iterable, Mapping


@dataclass(frozen=True)
class AxisSpec:
    """One consequence-bearing bodily variable and its public identities."""

    name: str
    report_word: str
    resting_rate: float
    moving_rate: float
    small_surface: str
    large_surface: str
    shock_surface: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("An axis name must be non-empty.")
        if not self.report_word:
            raise ValueError("An axis report word must be non-empty.")
        if self.resting_rate < 0.0 or self.moving_rate < 0.0:
            raise ValueError("Axis depletion rates must be nonnegative.")
        if not self.resting_rate <= self.moving_rate:
            raise ValueError("An axis's moving rate cannot be below its resting rate.")
        if len({self.small_surface, self.large_surface, self.shock_surface}) != 3:
            raise ValueError("Help sizes and shock must have distinct surfaces.")

    def rate(self, *, moved: bool) -> float:
        return self.moving_rate if moved else self.resting_rate


@dataclass(frozen=True)
class BodySchema:
    """An ordered collection of exchangeable body-axis definitions."""

    axes: tuple[AxisSpec, ...]

    def __post_init__(self) -> None:
        if len(self.axes) < 2:
            raise ValueError("A homeostatic body needs at least two axes.")
        names = [axis.name for axis in self.axes]
        words = [axis.report_word for axis in self.axes]
        surfaces = [
            surface
            for axis in self.axes
            for surface in (
                axis.small_surface,
                axis.large_surface,
                axis.shock_surface,
            )
        ]
        if len(set(names)) != len(names):
            raise ValueError("Body-axis names must be unique.")
        if len(set(words)) != len(words):
            raise ValueError("Report words must identify one axis each.")
        if len(set(surfaces)) != len(surfaces):
            raise ValueError("Every expanded-body consequence needs a unique surface.")

    def __len__(self) -> int:
        return len(self.axes)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(axis.name for axis in self.axes)

    @property
    def report_words(self) -> tuple[str, ...]:
        return tuple(axis.report_word for axis in self.axes)

    @property
    def surfaces(self) -> tuple[str, ...]:
        return tuple(
            surface
            for axis in self.axes
            for surface in (
                axis.small_surface,
                axis.large_surface,
                axis.shock_surface,
            )
        )

    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError as error:
            raise KeyError(f"Unknown body axis: {name}.") from error

    def axis(self, name: str) -> AxisSpec:
        return self.axes[self.index(name)]

    def need_for_word(self, word: str) -> str | None:
        for axis in self.axes:
            if axis.report_word == word:
                return axis.name
        return None


# The first three rows are intentionally the report ecology's exact constants
# and public surface identities.  The fourth row promotes its existing safety
# state into the request protocol.  Only the fifth row is a new state variable.
LEGACY_REPORT_AXES: tuple[AxisSpec, ...] = (
    AxisSpec("food", "hungry", 0.008, 0.008, "berry", "roots", "thorn"),
    AxisSpec("water", "thirsty", 0.012, 0.012, "water", "spring", "tree"),
    AxisSpec("energy", "tired", 0.016, 0.035, "hut", "mushroom", "rock"),
)
EXPANDED_AXES: tuple[AxisSpec, ...] = (
    *LEGACY_REPORT_AXES,
    AxisSpec("safety", "safe", 0.002, 0.002, "shield", "refuge", "alarm"),
    AxisSpec("health", "hurt", 0.010, 0.010, "herb", "salve", "wound"),
)

LEGACY_BODY_SCHEMA = BodySchema(LEGACY_REPORT_AXES)
EXPANDED_BODY_SCHEMA = BodySchema(EXPANDED_AXES)


@dataclass(frozen=True)
class ExpandedBodyConfig:
    """Fixed ecology constants used by the construction ceiling."""

    birth_levels: tuple[float, ...] = (0.35, 0.45, 0.55, 0.65, 0.75, 0.85)
    metabolic_spread: float = 0.60
    shock_probability: float = 0.025
    shock_size: float = 0.25
    portion_small: float = 0.20
    portion_large: float = 0.60
    help_period: int = 6
    life_steps: int = 400

    def __post_init__(self) -> None:
        if not self.birth_levels:
            raise ValueError("birth_levels must be non-empty.")
        if any(not 0.0 < value <= 1.0 for value in self.birth_levels):
            raise ValueError("birth levels must lie in (0, 1].")
        if not 0.0 <= self.metabolic_spread < 1.0:
            raise ValueError("metabolic_spread must lie in [0, 1).")
        if not 0.0 <= self.shock_probability <= 1.0:
            raise ValueError("shock_probability must lie in [0, 1].")
        if self.shock_size < 0.0:
            raise ValueError("shock_size must be nonnegative.")
        if not 0.0 < self.portion_small <= self.portion_large:
            raise ValueError("portions must satisfy 0 < small <= large.")
        if self.help_period <= 0 or self.life_steps <= 0:
            raise ValueError("help_period and life_steps must be positive.")


@dataclass(frozen=True)
class BodyState:
    """One immutable state vector interpreted by a declared schema."""

    schema: BodySchema
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.values) != len(self.schema):
            raise ValueError(
                f"State has {len(self.values)} values for {len(self.schema)} axes."
            )
        if any(not 0.0 <= value <= 1.0 for value in self.values):
            raise ValueError("Body-state values must lie in [0, 1].")

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.schema.names, self.values, strict=True))

    def value(self, need: str) -> float:
        return self.values[self.schema.index(need)]

    def viability(self) -> float:
        return min(self.values)

    def mean_viability(self) -> float:
        return sum(self.values) / len(self.values)

    def lowest_need(self) -> str:
        index = min(range(len(self.values)), key=self.values.__getitem__)
        return self.schema.axes[index].name

    def axis_gap(self) -> float:
        """Gap between the two lowest axes, Probe68's saturated margin term."""

        ordered = sorted(self.values)
        return ordered[1] - ordered[0]

    def replace(self, values: Mapping[str, float]) -> BodyState:
        unknown = set(values).difference(self.schema.names)
        if unknown:
            raise KeyError(f"Unknown body axes: {sorted(unknown)}.")
        updated = list(self.values)
        for need, value in values.items():
            updated[self.schema.index(need)] = _clip01(float(value))
        return BodyState(self.schema, tuple(updated))

    def add(self, need: str, amount: float) -> BodyState:
        return self.replace({need: self.value(need) + float(amount)})


@dataclass(frozen=True)
class BodyStep:
    """Audit record for one metabolism/shock transition."""

    before: BodyState
    after: BodyState
    rates: dict[str, float]
    shock_need: str | None
    terminated: bool


@dataclass(frozen=True)
class OracleRequestEvent:
    """One natural help opportunity in the construction ceiling."""

    tick: int
    need: str
    size: str
    axis_gap: float
    margin: float
    useful_word_difference: float
    consequential_first_use: bool

    @property
    def combination(self) -> tuple[str, str]:
        return self.need, self.size


@dataclass(frozen=True)
class OracleLife:
    """Read-only result of one perfect-requester construction life."""

    seed: int
    schema_size: int
    survived: bool
    steps: int
    death_need: str | None
    events: tuple[OracleRequestEvent, ...]
    shock_counts: dict[str, int]
    grant_counts: dict[str, int]


class ExpandedBody:
    """Seeded, axis-generic body dynamics with separated random streams.

    This is the body half of the construction survey, not a speaking organism.
    Birth, individual scale and shock streams are independent.  Consequently,
    drawing the two extra axes never changes the first three birth or scale
    draws, which is the paired-stream guarantee the preregistration requires.
    """

    _BIRTH_SALT = 31_000_019
    _SCALE_SALT = 37_000_021
    _SHOCK_SALT = 41_000_027

    def __init__(
        self,
        schema: BodySchema,
        *,
        config: ExpandedBodyConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.schema = schema
        self.config = config or ExpandedBodyConfig()
        self.seed = int(seed)
        self.state = BodyState(schema, (1.0,) * len(schema))
        self.metabolic_scale = {name: 1.0 for name in schema.names}
        self.step_count = 0
        self._shock_rng = Random(0)

    def reset(self, seed: int | None = None) -> BodyState:
        if seed is not None:
            self.seed = int(seed)
        birth_rng = Random(self.seed + self._BIRTH_SALT)
        scale_rng = Random(self.seed + self._SCALE_SALT)
        self._shock_rng = Random(self.seed + self._SHOCK_SALT)
        self.state = BodyState(
            self.schema,
            tuple(birth_rng.choice(self.config.birth_levels) for _ in self.schema.axes),
        )
        spread = self.config.metabolic_spread
        self.metabolic_scale = {
            axis.name: scale_rng.uniform(1.0 - spread, 1.0 + spread)
            for axis in self.schema.axes
        }
        self.step_count = 0
        return self.state

    def rates(self, *, moved: bool = False) -> dict[str, float]:
        return {
            axis.name: axis.rate(moved=moved) * self.metabolic_scale[axis.name]
            for axis in self.schema.axes
        }

    def grant(self, need: str, size: str, *, uptake: float = 1.0) -> BodyState:
        if size not in ("small", "large"):
            raise ValueError(f"Unknown portion size: {size}.")
        if uptake < 0.0:
            raise ValueError("uptake must be nonnegative.")
        portion = (
            self.config.portion_large
            if size == "large"
            else self.config.portion_small
        )
        self.state = self.state.add(need, portion * uptake)
        return self.state

    def step(self, *, moved: bool = False) -> BodyStep:
        before = self.state
        rates = self.rates(moved=moved)
        after = before.replace(
            {need: before.value(need) - rate for need, rate in rates.items()}
        )
        shock_need = None
        # Consume the Bernoulli draw even after a body is already at its floor;
        # stream position is a property of time, not of survival.
        shock_occurs = self._shock_rng.random() < self.config.shock_probability
        if after.viability() > 0.0 and shock_occurs:
            shock_need = self._shock_rng.choice(self.schema.names)
            after = after.add(shock_need, -self.config.shock_size)
        self.state = after
        self.step_count += 1
        return BodyStep(
            before=before,
            after=after,
            rates=rates,
            shock_need=shock_need,
            terminated=after.viability() <= 0.0,
        )


def projected_state(
    state: BodyState,
    rates: Mapping[str, float],
    *,
    ticks: int,
) -> BodyState:
    """Project a body forward without shocks or grants, for oracle diagnostics."""

    if ticks < 0:
        raise ValueError("Projection ticks must be nonnegative.")
    missing = set(state.schema.names).difference(rates)
    if missing:
        raise KeyError(f"Missing rates for axes: {sorted(missing)}.")
    return state.replace(
        {
            need: state.value(need) - float(rates[need]) * ticks
            for need in state.schema.names
        }
    )


def portion_for_burden(
    state: BodyState,
    need: str,
    *,
    rate: float,
    interval: int,
    config: ExpandedBodyConfig | None = None,
    uptake: float = 1.0,
) -> str:
    """The existing Probe64 portion rule, expressed for an arbitrary axis."""

    body = config or ExpandedBodyConfig()
    if rate < 0.0 or interval <= 0 or uptake <= 0.0:
        raise ValueError("rate, interval and uptake must be positive (rate may be 0).")
    required = rate * interval
    headroom = max(0.0, 1.0 - state.value(need))
    delivered = {
        "small": min(body.portion_small * uptake, headroom),
        "large": min(body.portion_large * uptake, headroom),
    }
    for size in ("small", "large"):
        if delivered[size] >= required:
            return size
    best = max(delivered.values())
    for size in ("small", "large"):
        if delivered[size] >= best - 1e-12:
            return size
    raise AssertionError("The two declared portions did not produce a choice.")


def run_oracle_life(
    schema: BodySchema,
    *,
    seed: int,
    config: ExpandedBodyConfig | None = None,
) -> OracleLife:
    """Run one ideal requester/motor life on the expanded body substrate.

    The request is chosen from the true current state and individual resting
    rates. A help object granted at the end of one tick is absorbed at the start
    of the next, matching the order of ``ReportWorld`` with a perfect consume
    policy. The tabular cold-start fallback is Probe64's fixed first word:
    ``more``. It agrees with an intended large request and disagrees with an
    intended small one.
    """

    body_config = config or ExpandedBodyConfig()
    body = ExpandedBody(schema, config=body_config, seed=seed)
    body.reset()
    pending: tuple[str, str] | None = None
    events: list[OracleRequestEvent] = []
    shocks = {need: 0 for need in schema.names}
    grants = {need: 0 for need in schema.names}
    death_need = None

    for _ in range(body_config.life_steps):
        # The organism speaks from the body state in its current packet, before
        # acting on a help object that is already visible at its feet.
        need = body.state.lowest_need()
        rates = body.rates(moved=False)
        size = portion_for_burden(
            body.state,
            need,
            rate=rates[need],
            interval=body_config.help_period * len(schema),
            config=body_config,
        )
        if pending is not None:
            body.grant(*pending)
            pending = None

        transition = body.step(moved=False)
        if transition.shock_need is not None:
            shocks[transition.shock_need] += 1
        if transition.terminated:
            death_need = body.state.lowest_need()
            break

        if body.step_count % body_config.help_period == 0:
            state = body.state
            headroom = max(0.0, 1.0 - state.value(need))
            useful_small = min(body_config.portion_small, headroom)
            useful_large = min(body_config.portion_large, headroom)
            # The unseen tabular cell tries `more`, the convention's large word.
            # It therefore differs from correct factorization only when the
            # intended size is small.
            word_difference = (
                max(0.0, useful_large - useful_small)
                if size == "small"
                else 0.0
            )
            grant = (
                body_config.portion_large
                if size == "large"
                else body_config.portion_small
            )
            gap = state.axis_gap()
            margin = min(grant, gap)
            consequential = word_difference > margin + 1e-12
            events.append(
                OracleRequestEvent(
                    tick=body.step_count,
                    need=need,
                    size=size,
                    axis_gap=gap,
                    margin=margin,
                    useful_word_difference=word_difference,
                    consequential_first_use=consequential,
                )
            )
            grants[need] += 1
            pending = (need, size)

    survived = body.step_count >= body_config.life_steps and body.state.viability() > 0.0
    return OracleLife(
        seed=seed,
        schema_size=len(schema),
        survived=survived,
        steps=body.step_count,
        death_need=death_need,
        events=tuple(events),
        shock_counts=shocks,
        grant_counts=grants,
    )


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def schema_from_axes(axes: Iterable[AxisSpec]) -> BodySchema:
    """Small public constructor used by lesions and tests."""

    return BodySchema(tuple(axes))
