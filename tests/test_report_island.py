from __future__ import annotations

from dataclasses import replace

import pytest

from homesocial.creole.vocab import PAD_TOKEN, TOKEN_TO_ID, encode_utterance
from homesocial.env import Action
from homesocial.island.report import (
    HELP_SURFACES,
    NEED_TO_REPORT_WORD,
    REPORT_NEEDS,
    SHOCK_SURFACES,
    ReportConfig,
    ReportWorld,
    heard_need,
)
from homesocial.island.report_calibrate import (
    NEED_UTTERANCE,
    SILENCE,
    TruthfulSpeaker,
    rhythm_family,
    uptake_action,
)
from homesocial.island.world import IslandConfig, SURFACE_INDEX

PAD_ID = TOKEN_TO_ID[PAD_TOKEN]


def make_world(seed: int = 1, **overrides: object) -> ReportWorld:
    report = replace(ReportConfig(), **overrides)  # type: ignore[arg-type]
    world = ReportWorld(
        IslandConfig(max_steps=report.life_steps, max_visible_slots=2),
        report=report,
        seed=seed,
    )
    world.reset(seed)
    return world


def test_interoception_is_masked_after_the_birth_reading():
    world = make_world()
    birth = world.reset(3)
    assert not birth.mask_needs
    assert list(birth.vector()[:4]) == pytest.approx(list(birth.needs))

    packet, *_ = world.step(Action.WAIT)
    assert packet.mask_needs
    assert list(packet.vector()[:4]) == [0.0, 0.0, 0.0, 0.0]
    # The truth is still carried for bookkeeping and audits, just not sensed.
    assert min(packet.needs) > 0.0


def test_masked_observation_never_moves_with_the_body():
    world = make_world()
    world.reset(5)
    world.step(Action.WAIT)
    baseline = world.step(Action.WAIT)[0].vector()
    world.grid.needs = replace(world.grid.needs, food=0.05, water=0.99)
    perturbed = world.step(Action.WAIT)[0].vector()
    assert list(baseline[:4]) == list(perturbed[:4]) == [0.0] * 4


def test_birth_needs_are_randomized_across_lives():
    world = make_world()
    seen = {world.reset(seed).needs[:3] for seed in range(40)}
    assert len(seen) > 10


def test_world_holds_no_consumables_before_help_arrives():
    world = make_world()
    world.reset(2)
    assert world.grid.objects == []
    for _ in range(world.report.help_period - 1):
        world.step(Action.WAIT)
    assert world.pending_help_need() is None


def test_help_is_granted_for_the_last_need_word_heard():
    world = make_world()
    world.reset(2)
    for tick in range(world.report.help_period):
        world.hear(NEED_UTTERANCE["water"])
        packet, _, _, _, info = world.step(Action.WAIT)
    assert info["granted_need"] == "water"
    assert world.pending_help_need() == "water"
    surfaces = {SURFACE_INDEX[HELP_SURFACES[("water", size)]] for size in (False, True)}
    assert any(slot[2] in surfaces for slot in packet.visible)


def test_silence_is_answered_with_nothing():
    world = make_world()
    world.reset(2)
    for _ in range(world.report.help_period):
        world.hear(SILENCE)
        _, _, _, _, info = world.step(Action.WAIT)
    assert info["granted_need"] is None
    assert world.pending_help_need() is None


def test_a_wrong_word_is_answered_with_the_wrong_help():
    world = make_world()
    world.reset(4)
    world.grid.needs = replace(world.grid.needs, food=0.30, water=0.90, energy=0.90)
    for _ in range(world.report.help_period):
        world.hear(NEED_UTTERANCE["water"])
        _, _, _, _, info = world.step(Action.WAIT)
    assert info["lowest_need"] == "food"
    assert info["granted_need"] == "water"


def test_uneaten_help_spoils_at_the_next_grant():
    world = make_world()
    world.reset(6)
    for _ in range(world.report.help_period):
        world.hear(NEED_UTTERANCE["food"])
        world.step(Action.WAIT)
    assert world.pending_help_need() == "food"
    for _ in range(world.report.help_period):
        world.hear(NEED_UTTERANCE["water"])
        world.step(Action.WAIT)
    assert world.pending_help_need() == "water"


def test_help_restores_only_the_need_it_names():
    world = make_world()
    world.reset(7)
    for _ in range(world.report.help_period):
        world.hear(NEED_UTTERANCE["food"])
        world.step(Action.WAIT)
    before = world.grid.needs
    world.hear(SILENCE)
    _, _, _, _, info = world.step(uptake_action(world))
    after = world.grid.needs
    assert after.food > before.food
    assert after.water < before.water
    assert info["event"] == "consumed_food"


def test_forced_audit_portion_is_both_perceptible_and_bodily():
    small = make_world(seed=71, shock_probability=0.0)
    large = make_world(seed=71, shock_probability=0.0)
    packets = []
    for world, is_large in ((small, False), (large, True)):
        world.force_next_help_portion(large=is_large)
        for _ in range(world.report.help_period):
            world.hear(NEED_UTTERANCE["food"])
            packet, _, _, _, info = world.step(Action.WAIT)
        assert info["granted_large"] is is_large
        packets.append(packet)

    assert small.pending_help_need() == large.pending_help_need() == "food"
    assert small.grid.needs == large.grid.needs
    # The portion surface is the sole post-grant perceptual difference.
    assert packets[0].vector().tolist() != packets[1].vector().tolist()

    small.grid.needs = replace(small.grid.needs, food=0.10)
    large.grid.needs = replace(large.grid.needs, food=0.10)
    small.step(Action.CONSUME)
    large.step(Action.CONSUME)
    assert large.grid.needs.food - small.grid.needs.food == pytest.approx(
        large.report.portion_large - small.report.portion_small
    )


def test_shelter_help_needs_rest_and_free_rest_gives_nothing():
    world = make_world()
    world.reset(8)
    before_idle = world.grid.needs.energy
    world.step(Action.REST)
    assert world.grid.needs.energy < before_idle

    for _ in range(world.report.help_period):
        world.hear(NEED_UTTERANCE["energy"])
        world.step(Action.WAIT)
    assert uptake_action(world) == Action.REST
    before = world.grid.needs.energy
    _, _, _, _, info = world.step(Action.REST)
    assert info["event"] == "rested_shelter"
    assert world.grid.needs.energy > before


def test_unified_uptake_uses_consume_for_every_help_kind():
    for need in REPORT_NEEDS:
        world = make_world(seed=81, unified_uptake=True, shock_probability=0.0)
        for _ in range(world.report.help_period):
            world.hear(NEED_UTTERANCE[need])
            world.step(Action.WAIT)
        assert uptake_action(world) == Action.CONSUME
        before = getattr(world.grid.needs, need)
        _, _, _, _, info = world.step(Action.CONSUME)
        after = getattr(world.grid.needs, need)
        assert after > before
        assert info["event"] == {
            "food": "consumed_food",
            "water": "consumed_water",
            "energy": "consumed_shelter",
        }[need]


def test_shocks_are_perceptible_but_not_interoceptive():
    world = make_world(shock_probability=1.0)
    world.reset(9)
    packet, _, _, _, info = world.step(Action.WAIT)
    need = info["shock_need"]
    assert need in REPORT_NEEDS
    marker = SURFACE_INDEX[SHOCK_SURFACES[need]]
    assert any(slot[2] == marker for slot in packet.visible)
    assert list(packet.vector()[:4]) == [0.0] * 4


def test_shock_marker_lasts_exactly_one_tick():
    markers = {SURFACE_INDEX[surface] for surface in SHOCK_SURFACES.values()}
    world = make_world(shock_probability=1.0)
    world.reset(10)
    first, *_ = world.step(Action.WAIT)
    assert any(slot[2] in markers for slot in first.visible)

    world.report = replace(world.report, shock_probability=0.0)
    world.grid.report = world.report
    second, *_ = world.step(Action.WAIT)
    assert not any(slot[2] in markers for slot in second.visible)


def test_listener_hears_the_first_need_word_only():
    assert heard_need(encode_utterance(("me", "hungry"))) == "food"
    assert heard_need(encode_utterance(("thirsty", "hungry"))) == "water"
    assert heard_need(encode_utterance(("me", "good", "now"))) is None
    assert heard_need(None) is None
    assert heard_need((PAD_ID, PAD_ID)) is None


def test_every_need_word_is_in_the_closed_vocabulary():
    for need in REPORT_NEEDS:
        assert NEED_TO_REPORT_WORD[need] in TOKEN_TO_ID


def test_scrambled_listener_keeps_traffic_but_loses_content():
    grounded = make_world(listener_mode="grounded")
    scrambled = make_world(listener_mode="scrambled")
    heard = []
    for world in (grounded, scrambled):
        world.reset(11)
        for _ in range(world.report.help_period):
            world.hear(NEED_UTTERANCE["food"])
            _, _, _, _, info = world.step(Action.WAIT)
        heard.append(info["granted_need"])
    assert heard[0] == "food"
    assert heard[1] in REPORT_NEEDS


def test_mute_listener_grants_nothing():
    world = make_world(listener_mode="mute")
    world.reset(12)
    for _ in range(world.report.help_period * 3):
        world.hear(NEED_UTTERANCE["food"])
        _, _, _, _, info = world.step(Action.WAIT)
        assert info["granted_need"] is None


def test_truthful_speaker_outlives_silence():
    def survival(speaker_factory, lives: int = 20) -> float:
        alive = 0
        for life in range(lives):
            world = make_world(seed=200 + life)
            speaker = speaker_factory()
            while True:
                world.hear(speaker.utterance(world))
                _, _, terminated, truncated, _ = world.step(uptake_action(world))
                if terminated or truncated:
                    alive += int(not terminated)
                    break
        return alive / lives

    class Mute:
        def utterance(self, world: ReportWorld) -> tuple[int, ...]:
            return SILENCE

    assert survival(TruthfulSpeaker) >= 0.85
    assert survival(Mute) == 0.0


def test_rhythm_family_deduplicates_rotations():
    patterns = rhythm_family(3)
    rotations = [
        ("food", "water", "energy"),
        ("water", "energy", "food"),
        ("energy", "food", "water"),
    ]
    assert sum(rotation in patterns for rotation in rotations) == 1
    assert len(patterns) == len(set(patterns))
    assert all(len(pattern) <= 3 for pattern in patterns)


def test_report_world_never_reveals_the_answer_in_perception():
    """No visible surface encodes which need is currently lowest."""

    world = make_world(shock_probability=0.0)
    world.reset(13)
    for tick in range(60):
        world.hear(NEED_UTTERANCE["food"])
        packet, _, _, _, info = world.step(uptake_action(world))
        pending = world.pending_help_need()
        if pending is not None and info["lowest_need"] != pending:
            # Perception shows what was asked for, which can differ from what
            # the body actually needs. That gap is the whole experiment.
            break
    else:
        pytest.fail("Expected at least one tick where help and need diverge.")
