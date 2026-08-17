"""Probe63: an organism that finds out what its own body is.

Preregistered in `docs/decisions/2026-08-03-individual-self-calibration-preregistration.md`.
Gates are locked there and are not adjusted here.

The claim this attacks is the oldest one open in `docs/STATE.md`: *nothing yet
shows the learned model doing what the analytic filter cannot*. Probe62 found the
reason, and it was not any of the mechanisms tried. Every organism in this
ecology burned fuel at exactly the species rate, so the "self-model" was a model
of bodies in general -- a physics the designer knows in closed form, which is
precisely why the hand-written probe53 filter was exact and unbeatable. There was
nothing individual to learn.

Probe63's ecology gives each organism its own body and an intermittent channel
through which it might find that out. What this module adds is the finding out.

The mechanism has one idea. The organism keeps a copy of its own body filter for
every constant it could be wrong about, each with exactly that one constant
perturbed, and reads off

    J[need, parameter] = d(predicted body) / d(log parameter)

That is not a fact about where its body is. It is a fact about how its own
*model* would respond if a particular belief about itself were wrong -- which is
the question that has to be answered to find out what about oneself is
miscalibrated rather than merely that something is. When a reading arrives, the
residual is attributed across its own parameters in proportion to how much each
one could have caused it, by normalized LMS. Nothing tells it which parameter is
at fault.

The correction is exact rather than linearized: the predicted body depends
linearly on every one of these constants, so scaling a constant by ``exp(m)``
moves the prediction by ``(exp(m) - 1) * J`` and no approximation is needed.

The contrast that matters is against ``snap``, an organism given exactly the same
readings which corrects its belief to each one and carries nothing between them.
Phase B1 warns that its first gate is "satisfiable by clipping"; ``snap`` is that
warning made into a control. Everything ``learned`` has that ``snap`` lacks is a
model of itself.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from random import Random

import numpy as np

from homesocial.creole.vocab import PAD_TOKEN, TOKEN_TO_ID
from homesocial.island.report import NEED_TO_REPORT_WORD, REPORT_NEEDS
from homesocial.organism.individual_self import (
    BOUND_EPSILON,
    CALIBRATION_PARAMETERS,
    METABOLIC_PARAMETERS,
    UPTAKE_PARAMETERS,
    SelfJacobian,
    _true_body,
    make_tier,
)
from homesocial.organism.model import OrganismModel
from homesocial.organism.report_audit import FIDELITY_WARMUP, make_report_world
from homesocial.organism.self_belief import _sample_motor_action
from homesocial.organism.train import (
    OrganismConfig,
    execute_agent_action,
    load_organism_checkpoint,
)

PAD_ID = TOKEN_TO_ID[PAD_TOKEN]
TIERS = ("population", "snap", "learned", "recursive", "individual")

# Fixed a priori by the preregistration and never swept. NLMS is stable for any
# step in (0, 2), which is why it was chosen: it needs no per-ecology tuning.
LEARNING_RATE = 0.5
NLMS_EPSILON = 1e-8
# The recursive arm's constants, fixed for the same reason. The ridge matches
# the one the localization ceiling was measured with, so the online estimator
# and the batch ceiling it is scored against differ only in being online. The
# body is constant within a life, so nothing is forgotten.
RLS_RIDGE = 1e-6
RLS_FORGETTING = 1.0
# A sanity bound on self-belief: no organism concludes that its own metabolism
# is a thousand times its species' rate. Legitimate corrections here are around
# |log 1.6| = 0.47, so this never binds on an organism reading its own body. It
# binds only under the shuffled-readings control, where NLMS is driven by
# another organism's body and diverges -- which is the control succeeding. It is
# clamped rather than left to overflow so that the divergence is reported as a
# number instead of as a NaN.
MAX_LOG_CORRECTION = 6.9

SEED_BASE = 930_000_000
SEED_STRIDE = 2_000_000
SEEDS = 5

# The four worlds. The ground truth has to move, or a mechanism that always
# answers "metabolism" would score well without discriminating anything.
WORLDS = {
    "metabolic": {"metabolic_spread": 0.60, "uptake_spread": 0.00},
    "absorption": {"metabolic_spread": 0.00, "uptake_spread": 0.60},
    "both": {"metabolic_spread": 0.60, "uptake_spread": 0.60},
    "null": {"metabolic_spread": 0.00, "uptake_spread": 0.00},
}
TREATMENT_WORLD = "metabolic"
INTEROCEPTION = 0.03

_METABOLIC_INDEX = [CALIBRATION_PARAMETERS.index(p) for p in METABOLIC_PARAMETERS]
_UPTAKE_INDEX = [CALIBRATION_PARAMETERS.index(p) for p in UPTAKE_PARAMETERS]


class SelfCalibration:
    """A body filter that also carries a model of its own constants.

    ``m`` is the log-correction to each of this organism's own parameters, held
    separately per need so that "how well I absorb food" and "how well I absorb
    water" can be different facts about this body. It starts at exactly zero,
    which is the species body -- the organism begins by assuming it is typical
    and has to find out otherwise.
    """

    def __init__(
        self,
        organism: OrganismConfig,
        report,
        *,
        learning_rate: float = LEARNING_RATE,
    ) -> None:
        self._organism = organism
        self._report = report
        self._rate = learning_rate
        self._jacobian = SelfJacobian(organism, report)
        self._m = np.zeros((len(REPORT_NEEDS), len(CALIBRATION_PARAMETERS)))
        self._burn_corrected = np.zeros(len(REPORT_NEEDS))
        self._burn_species = np.zeros(len(REPORT_NEEDS))
        self.readings = 0
        self.skipped = 0

    # -- belief --------------------------------------------------------------

    def reset(self, packet, world) -> None:
        self._jacobian = SelfJacobian(self._organism, self._report)
        self._jacobian.reset(packet)
        self._m = np.zeros((len(REPORT_NEEDS), len(CALIBRATION_PARAMETERS)))
        self._burn_corrected = np.zeros(len(REPORT_NEEDS))
        self._burn_species = np.zeros(len(REPORT_NEEDS))
        self.readings = 0
        self.skipped = 0

    def _gain(self) -> np.ndarray:
        """``d(predicted body) / dm``, exact because the model is linear in theta."""

        return np.exp(self._m) * self._jacobian.matrix()

    def point(self) -> np.ndarray:
        """The species prediction plus the effect of its own corrections.

        Exact rather than first-order: the predicted body is linear in each
        constant, so scaling one by ``exp(m)`` moves the prediction by exactly
        ``(exp(m) - 1) * J``.
        """

        base = self._jacobian.belief
        correction = ((np.exp(self._m) - 1.0) * self._jacobian.matrix()).sum(axis=1)
        return np.clip(base + correction, 0.0, 1.0)

    # -- learning ------------------------------------------------------------

    def update(self, before, action_index: int, after, world, reading) -> None:
        self._jacobian.update(before, action_index, after)
        if reading is None:
            return

        truth = np.asarray(reading, dtype=np.float64)
        predicted = self.point()
        gain = self._gain()
        self.readings += 1

        for index in range(len(REPORT_NEEDS)):
            if not BOUND_EPSILON < truth[index] < 1.0 - BOUND_EPSILON:
                # Clipping has erased this need's history, so the residual says
                # nothing about the constants that produced it.
                self.skipped += 1
                continue
            row = gain[index]
            energy = float(row @ row)
            if energy <= 0.0:
                self.skipped += 1
                continue
            residual = float(truth[index] - predicted[index])
            self._m[index] += self._rate * residual * row / (energy + NLMS_EPSILON)
        np.clip(self._m, -MAX_LOG_CORRECTION, MAX_LOG_CORRECTION, out=self._m)

        # How much of this interval's burn each model accounts for, so the
        # recovered rate can be reported as the effective multiplier it is
        # rather than as one arbitrarily chosen coefficient.
        matrix = self._jacobian.matrix()
        for index in range(len(REPORT_NEEDS)):
            species = matrix[index, _METABOLIC_INDEX].sum()
            corrected = (
                np.exp(self._m[index, _METABOLIC_INDEX])
                * matrix[index, _METABOLIC_INDEX]
            ).sum()
            self._burn_species[index] += species
            self._burn_corrected[index] += corrected

        self._jacobian.snap(truth)

    # -- what it came to believe about itself --------------------------------

    def correction(self) -> np.ndarray:
        return self._m.copy()

    def recovered_scale(self) -> dict[str, float]:
        """Its own answer to "how fast do I burn", as a multiplier on the species."""

        scale = {}
        for index, need in enumerate(REPORT_NEEDS):
            species = self._burn_species[index]
            scale[need] = (
                float(self._burn_corrected[index] / species)
                if abs(species) > 1e-12
                else 1.0
            )
        return scale

    def recovered_uptake(self) -> dict[str, float]:
        """Its own answer to "how much good does a portion do me".

        The two portion classes are separate parameters of the model, so a body
        that absorbs badly shows up as a correction to both. Their mean is the
        multiplier a planner needs, and at zero correction it is exactly one --
        the species body the organism starts out assuming it has.
        """

        uptake = {}
        for index, need in enumerate(REPORT_NEEDS):
            uptake[need] = float(np.exp(self._m[index, _UPTAKE_INDEX]).mean())
        return uptake


class RecursiveSelfCalibration(SelfCalibration):
    """The same self-model, attributing by accumulated evidence rather than greedily.

    NLMS moves along the current sensitivity direction, which spreads a residual
    across parameters in proportion to how sensitive each one is *right now*. It
    therefore cannot separate two constants whose effects arrive together, and in
    this ecology help arrives on a clock, so "I absorb less from each portion"
    and "I burn faster between them" both look like a deficit proportional to
    elapsed time. Recursive least squares keeps the accumulated second-order
    statistics that tell them apart, and is the online form of the pooled fit
    whose localization ceiling was measured before any gate was locked.

    Estimated in the amplitude ``a = exp(m) - 1``, in which the predicted body is
    exactly linear, so this is ordinary recursive linear regression with no
    approximation.
    """

    def __init__(
        self,
        organism: OrganismConfig,
        report,
        *,
        ridge: float = RLS_RIDGE,
        forgetting: float = RLS_FORGETTING,
    ) -> None:
        super().__init__(organism, report)
        self._ridge = ridge
        self._forgetting = forgetting
        self._a = np.zeros((len(REPORT_NEEDS), len(CALIBRATION_PARAMETERS)))
        self._covariance = np.stack(
            [
                np.eye(len(CALIBRATION_PARAMETERS)) / ridge
                for _ in range(len(REPORT_NEEDS))
            ]
        )

    def reset(self, packet, world) -> None:
        super().reset(packet, world)
        self._a = np.zeros((len(REPORT_NEEDS), len(CALIBRATION_PARAMETERS)))
        self._covariance = np.stack(
            [
                np.eye(len(CALIBRATION_PARAMETERS)) / self._ridge
                for _ in range(len(REPORT_NEEDS))
            ]
        )

    def update(self, before, action_index: int, after, world, reading) -> None:
        self._jacobian.update(before, action_index, after)
        if reading is None:
            return

        truth = np.asarray(reading, dtype=np.float64)
        base = self._jacobian.belief
        matrix = self._jacobian.matrix()
        self.readings += 1

        for index in range(len(REPORT_NEEDS)):
            if not BOUND_EPSILON < truth[index] < 1.0 - BOUND_EPSILON:
                self.skipped += 1
                continue
            regressor = matrix[index]
            target = float(truth[index] - base[index])
            covariance = self._covariance[index]
            projected = covariance @ regressor
            denominator = self._forgetting + float(regressor @ projected)
            if denominator <= 0.0:
                self.skipped += 1
                continue
            gain = projected / denominator
            self._a[index] += gain * (target - float(regressor @ self._a[index]))
            self._covariance[index] = (
                covariance - np.outer(gain, projected)
            ) / self._forgetting

        # ``a`` is the canonical estimate; ``m`` is its log form, kept so that
        # both arms report their self-model on one scale.
        np.clip(
            self._a, -0.99, np.expm1(MAX_LOG_CORRECTION), out=self._a
        )
        self._m = np.log1p(self._a)

        for index in range(len(REPORT_NEEDS)):
            species = matrix[index, _METABOLIC_INDEX].sum()
            corrected = (
                (1.0 + self._a[index, _METABOLIC_INDEX])
                * matrix[index, _METABOLIC_INDEX]
            ).sum()
            self._burn_species[index] += species
            self._burn_corrected[index] += corrected

        self._jacobian.snap(truth)


class OracleCalibration(SelfCalibration):
    """Born knowing its own body exactly. The ceiling, in every world.

    ``ReportConfig`` cannot express a per-need absorption -- it carries one
    ``portion_small`` for the whole species -- so an "individual" tier built by
    substituting constants into the filter is a ceiling in a metabolism world
    and silently *not* one in an absorption world, where it scores identically
    to the species filter. Expressing the ceiling as a frozen correction in the
    same per-need parameterization the learner uses avoids that, and makes the
    two differ in exactly one thing: whether the correction was given or found.

    It receives no readings and never snaps. It is what knowing yourself is
    worth with no evidence at all.
    """

    def reset(self, packet, world) -> None:
        super().reset(packet, world)
        metabolic = world.metabolic_scale
        uptake = world.uptake_scale
        for index, need in enumerate(REPORT_NEEDS):
            for position in _METABOLIC_INDEX:
                self._m[index, position] = np.log(metabolic[need])
            for position in _UPTAKE_INDEX:
                self._m[index, position] = np.log(uptake[need])

    def update(self, before, action_index: int, after, world, reading) -> None:
        self._jacobian.update(before, action_index, after)


# -- one condition -------------------------------------------------------------


def _donor_bodies(organism: OrganismConfig, model, *, seed: int, world_kwargs):
    """One other life's body trajectory, for the shuffled-readings control."""

    world = make_report_world(organism, seed=seed, **world_kwargs)
    packet = world.reset(seed)
    hidden = None
    rng = Random(seed + 59_000_003)
    bodies = []
    while True:
        action, hidden = _sample_motor_action(model, packet, hidden, rng)
        world.hear((TOKEN_TO_ID[NEED_TO_REPORT_WORD["food"]], PAD_ID))
        packet, _, terminated, truncated, _ = execute_agent_action(
            world,
            packet,
            action,
            consume_options=organism.consume_options,
            inspect_options=organism.inspect_options,
        )
        bodies.append(_true_body(world))
        if terminated or truncated:
            break
    return bodies


def run_world(
    model: OrganismModel,
    organism: OrganismConfig,
    *,
    world_name: str,
    lives: int,
    seed_base: int,
    interoception: float = INTEROCEPTION,
    shuffle_readings: bool = False,
) -> dict[str, object]:
    """Every tier on one shared history, driven by the species filter.

    The driver is `population` in every condition, so each tier is scored on the
    same transitions and a difference between tiers is a difference between
    self-models rather than between trajectories that diverged.
    """

    spreads = WORLDS[world_name]
    report = replace(
        organism.report, **spreads, interoception_probability=interoception
    )
    world_kwargs = dict(spreads, interoception_probability=interoception)

    error = {tier: 0.0 for tier in TIERS}
    hits = {tier: 0 for tier in TIERS}
    ticks = 0
    readings = 0
    corrections: dict[str, list[np.ndarray]] = {"learned": [], "recursive": []}
    scale_error = {"learned": 0.0, "recursive": 0.0}
    scale_lives = 0

    for life in range(lives):
        seed = seed_base + life
        donor = (
            _donor_bodies(
                organism,
                model,
                seed=seed + 777_777,
                world_kwargs=world_kwargs,
            )
            if shuffle_readings
            else None
        )
        world = make_report_world(organism, seed=seed, **world_kwargs)
        packet = world.reset(seed)
        tiers = {
            name: make_tier(name, organism, report)
            for name in ("population", "snap")
        }
        learner = SelfCalibration(organism, report)
        recursive = RecursiveSelfCalibration(organism, report)
        tiers["learned"] = learner  # type: ignore[assignment]
        tiers["recursive"] = recursive  # type: ignore[assignment]
        tiers["individual"] = OracleCalibration(organism, report)  # type: ignore[assignment]
        for tier in tiers.values():
            tier.reset(packet, world)
        hidden = None
        rng = Random(seed + 59_000_003)
        reading_index = 0

        while True:
            truth = _true_body(world)
            if packet.step_count >= FIDELITY_WARMUP:
                ticks += 1
                for name, tier in tiers.items():
                    believed = tier.point()
                    error[name] += float(np.abs(believed - truth).mean())
                    hits[name] += int(
                        int(np.argmin(believed)) == int(np.argmin(truth))
                    )
            need = REPORT_NEEDS[int(np.argmin(tiers["population"].point()))]
            action, hidden = _sample_motor_action(model, packet, hidden, rng)
            world.hear((TOKEN_TO_ID[NEED_TO_REPORT_WORD[need]], PAD_ID))
            before = packet
            packet, _, terminated, truncated, info = execute_agent_action(
                world,
                packet,
                action,
                consume_options=organism.consume_options,
                inspect_options=organism.inspect_options,
            )
            if terminated or truncated:
                break
            reading = info.get("interoception")
            if reading is not None:
                readings += 1
                if donor is not None:
                    # Matched in count and in timing; only the body is another
                    # organism's. A self-model that improves on this was never
                    # reading the evidence.
                    reading = tuple(donor[reading_index % len(donor)])
                reading_index += 1
            for tier in tiers.values():
                tier.update(before, action, packet, world, reading)

        for name, arm in (("learned", learner), ("recursive", recursive)):
            corrections[name].append(arm.correction())
            recovered = arm.recovered_scale()
            actual = world.metabolic_scale
            scale_error[name] += float(
                np.mean([abs(recovered[n] - actual[n]) for n in REPORT_NEEDS])
            )
        scale_lives += 1

    denominator = max(1, ticks)
    attribution = {}
    for name in ("learned", "recursive"):
        magnitude = np.abs(np.asarray(corrections[name])).mean(axis=0).sum(axis=0)
        metabolic_mass = float(magnitude[_METABOLIC_INDEX].sum())
        uptake_mass = float(magnitude[_UPTAKE_INDEX].sum())
        total = float(magnitude.sum())
        identified = total > 1e-9
        attribution[name] = {
            "total_correction": total,
            "metabolic_share": metabolic_mass / total if identified else None,
            "uptake_share": uptake_mass / total if identified else None,
            "rate_error": scale_error[name] / max(1, scale_lives),
        }
    return {
        "world": world_name,
        "shuffled_readings": shuffle_readings,
        "interoception_probability": interoception,
        "lives": lives,
        "seed_base": seed_base,
        "scored_ticks": ticks,
        "readings_per_life": readings / lives,
        "body_error": {tier: error[tier] / denominator for tier in TIERS},
        "argmin_accuracy": {tier: hits[tier] / denominator for tier in TIERS},
        "attribution": attribution,
        "prior_rate_error": spreads["metabolic_spread"] / 2.0,
    }


def run_closed_loop(
    model: OrganismModel,
    organism: OrganismConfig,
    *,
    tier_name: str,
    world_name: str,
    lives: int,
    seed_base: int,
    interoception: float = INTEROCEPTION,
) -> dict[str, object]:
    """One tier speaking for itself, so it lives its own life. Context only.

    Never gated. The survey established that this help loop separates the
    species filter from everything else and does *not* separate evidence from a
    self-model, so survival cannot carry the contrast this probe is about. It is
    reported because a belief-side gain that moves nothing at all would be worth
    knowing about, and this one does move something.
    """

    spreads = WORLDS[world_name]
    report = replace(
        organism.report, **spreads, interoception_probability=interoception
    )
    world_kwargs = dict(spreads, interoception_probability=interoception)
    survived = 0
    truthful = 0
    said = 0
    for life in range(lives):
        seed = seed_base + life
        world = make_report_world(organism, seed=seed, **world_kwargs)
        packet = world.reset(seed)
        if tier_name == "learned":
            tier = SelfCalibration(organism, report)
        elif tier_name == "recursive":
            tier = RecursiveSelfCalibration(organism, report)
        elif tier_name == "individual":
            tier = OracleCalibration(organism, report)
        else:
            tier = make_tier(tier_name, organism, report)
        tier.reset(packet, world)
        hidden = None
        rng = Random(seed + 59_000_003)
        while True:
            need = REPORT_NEEDS[int(np.argmin(tier.point()))]
            if packet.step_count >= FIDELITY_WARMUP:
                said += 1
                truthful += int(need == world.lowest_need())
            action, hidden = _sample_motor_action(model, packet, hidden, rng)
            world.hear((TOKEN_TO_ID[NEED_TO_REPORT_WORD[need]], PAD_ID))
            before = packet
            packet, _, terminated, truncated, info = execute_agent_action(
                world,
                packet,
                action,
                consume_options=organism.consume_options,
                inspect_options=organism.inspect_options,
            )
            if terminated or truncated:
                survived += int(not terminated)
                break
            tier.update(before, action, packet, world, info.get("interoception"))
    return {
        "tier": tier_name,
        "world": world_name,
        "interoception_probability": interoception,
        "lives": lives,
        "survival": survived / lives,
        "report_fidelity": truthful / max(1, said),
    }


# -- the locked gates ----------------------------------------------------------


def _verdict(flags: list[bool], needed: int = 4) -> dict[str, object]:
    return {
        "per_seed": flags,
        "passing": sum(flags),
        "required": needed,
        "pass": sum(flags) >= needed,
    }


def score_arm(rows: dict[str, list[dict[str, object]]], arm: str) -> dict[str, object]:
    """Exactly the gates in the preregistration, scored per seed, for one arm.

    Both arms face the same locked thresholds. Nothing here is arm-specific
    except which tier's numbers are read.
    """

    def error(world: str, tier: str) -> list[float]:
        return [float(row["body_error"][tier]) for row in rows[world]]  # type: ignore[index]

    def accuracy(world: str, tier: str) -> list[float]:
        return [float(row["argmin_accuracy"][tier]) for row in rows[world]]  # type: ignore[index]

    def attribution(world: str, key: str) -> list[object]:
        return [row["attribution"][arm][key] for row in rows[world]]  # type: ignore[index]

    treatment = rows[TREATMENT_WORLD]
    g1 = [
        e <= 0.75 * s
        for e, s in zip(error(TREATMENT_WORLD, arm), error(TREATMENT_WORLD, "snap"))
    ]
    g2 = [
        (l - s) >= 0.04
        for l, s in zip(
            accuracy(TREATMENT_WORLD, arm), accuracy(TREATMENT_WORLD, "snap")
        )
    ]
    g3_metabolic = [
        (share or 0.0) >= 0.70 for share in attribution("metabolic", "metabolic_share")
    ]
    g3_absorption = [
        (share or 0.0) >= 0.45 for share in attribution("absorption", "uptake_share")
    ]
    treatment_total = float(
        np.mean([float(row["attribution"][arm]["total_correction"]) for row in treatment])  # type: ignore[index]
    )
    g4 = [
        float(row["attribution"][arm]["total_correction"]) <= 0.10 * treatment_total  # type: ignore[index]
        and abs(
            float(row["body_error"][arm]) - float(row["body_error"]["population"])  # type: ignore[index]
        )
        <= 0.005
        for row in rows["null"]
    ]
    g5 = [
        float(row["body_error"][arm]) >= float(row["body_error"]["population"])  # type: ignore[index]
        for row in rows["shuffled"]
    ]
    g6 = [
        float(row["attribution"][arm]["rate_error"]) < float(row["prior_rate_error"])  # type: ignore[index]
        for row in treatment
    ]

    gates = {
        "G1_corrigibility": _verdict(g1),
        "G2_report": _verdict(g2),
        "G3_localization_metabolic": _verdict(g3_metabolic),
        "G3_localization_absorption": _verdict(g3_absorption),
        "G4_no_false_discovery": _verdict(g4),
        "G5_shuffled_readings": _verdict(g5),
        "G6_rate_recovery": _verdict(g6),
    }
    gates["all_pass"] = all(g["pass"] for g in gates.values())  # type: ignore[index]
    return gates


def score_gates(rows: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    return {arm: score_arm(rows, arm) for arm in ("learned", "recursive")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=(
            "runs/organism/probe52_guided_report_lexicon/adult/"
            "organism_report_seed1.npz"
        ),
    )
    parser.add_argument("--lives", type=int, default=40)
    parser.add_argument("--seeds", type=int, default=SEEDS)
    parser.add_argument(
        "--out", default="runs/organism/probe63_individual_self/treatment.json"
    )
    args = parser.parse_args()

    model, organism = load_organism_checkpoint(args.checkpoint)
    rows: dict[str, list[dict[str, object]]] = {name: [] for name in WORLDS}
    rows["shuffled"] = []

    for index in range(args.seeds):
        seed_base = SEED_BASE + SEED_STRIDE * index
        for name in WORLDS:
            row = run_world(
                model,
                organism,
                world_name=name,
                lives=args.lives,
                seed_base=seed_base,
            )
            rows[name].append(row)
            body = row["body_error"]
            accuracy = row["argmin_accuracy"]
            print(
                f"seed {index} {name:>10}: err pop {body['population']:.4f} "
                f"snap {body['snap']:.4f} nlms {body['learned']:.4f} "
                f"rls {body['recursive']:.4f} ind {body['individual']:.4f} | "
                f"acc snap {accuracy['snap']:.3f} nlms {accuracy['learned']:.3f} "
                f"rls {accuracy['recursive']:.3f}"
            )
        shuffled = run_world(
            model,
            organism,
            world_name=TREATMENT_WORLD,
            lives=args.lives,
            seed_base=seed_base,
            shuffle_readings=True,
        )
        rows["shuffled"].append(shuffled)
        print(
            f"seed {index} {'shuffled':>10}: err pop "
            f"{shuffled['body_error']['population']:.4f} "
            f"nlms {shuffled['body_error']['learned']:.4f} "
            f"rls {shuffled['body_error']['recursive']:.4f}"
        )

    gates = score_gates(rows)
    result = {
        "rows": rows,
        "gates": gates,
        "learning_rate": LEARNING_RATE,
        "rls_ridge": RLS_RIDGE,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str))

    for arm in ("learned", "recursive"):
        print(f"\n== locked gates: {arm} ==")
        for name, verdict in gates[arm].items():
            if name == "all_pass":
                continue
            mark = "PASS" if verdict["pass"] else "FAIL"
            print(
                f"{mark}  {name}: {verdict['passing']}/{len(verdict['per_seed'])} "
                f"{verdict['per_seed']}"
            )
        print(f"all_pass = {gates[arm]['all_pass']}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
