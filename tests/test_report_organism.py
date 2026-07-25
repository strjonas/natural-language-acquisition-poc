from __future__ import annotations

from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest

from homesocial.creole.vocab import PAD_TOKEN, TOKEN_TO_ID, VOCAB
from homesocial.island.report import (
    NEED_TO_REPORT_WORD,
    REPORT_NEEDS,
    ReportConfig,
    ReportWorld,
)
from homesocial.island.world import IslandConfig
from homesocial.organism.report_audit import (
    audit_counterfactual_body,
    audit_observation_decoder,
    evaluate_report,
    most_frequent_utterances,
    report_battery,
)
from homesocial.organism.train import (
    OrganismConfig,
    OrganismTrainer,
    load_organism_checkpoint,
    save_checkpoint,
)

PAD_ID = TOKEN_TO_ID[PAD_TOKEN]


def report_config(**overrides: object) -> OrganismConfig:
    base = dict(
        report_task=True,
        report_slots=2,
        total_steps=600,
        segment_length=32,
        hidden_size=32,
        token_prediction_weight=0.0,
        next_needs_weight=0.0,
        max_steps=ReportConfig().life_steps,
        report=ReportConfig(life_steps=60),
        island=IslandConfig(max_steps=60, max_visible_slots=2),
        seed=3,
    )
    base.update(overrides)
    return OrganismConfig(**base)  # type: ignore[arg-type]


def test_report_head_requires_the_report_task():
    with pytest.raises(ValueError):
        OrganismConfig(report_slots=2)


def test_trainer_builds_a_report_world_and_a_mouth():
    trainer = OrganismTrainer(report_config())
    assert isinstance(trainer.world, ReportWorld)
    assert trainer.model.can_speak
    assert trainer.model.report_slots == 2


def test_report_logits_cover_the_whole_closed_vocabulary():
    trainer = OrganismTrainer(report_config())
    vector = mx.array(trainer.packet.vector()[None, None, :])
    tokens = mx.array(
        np.asarray(trainer.packet.tokens, dtype=np.int32)[None, None, :]
    )
    states, _ = trainer.model.core_states(vector, tokens, None)
    logits = trainer.model.report_logits(states)
    assert logits.shape == (1, 1, 2, len(VOCAB))


def test_a_mute_organism_is_unchanged_and_still_trains():
    trainer = OrganismTrainer(report_config(report_slots=0))
    assert not trainer.model.can_speak
    stats = trainer.train()
    assert stats
    assert all(life.report_need_words == 0 for life in stats)


def test_training_runs_and_records_what_was_said():
    trainer = OrganismTrainer(report_config())
    stats = trainer.train()
    assert stats
    assert sum(life.report_need_words for life in stats) > 0
    assert all(
        life.report_truthful_words <= life.report_need_words for life in stats
    )


def test_nothing_in_the_loss_ever_sees_the_true_body_when_unsupervised():
    """The report head's only gradient is the advantage of the life it led to.

    Zeroing the bodily-prediction weight removes the one loss term that is
    given true need values, so no term in the objective can state the hidden
    body. This test pins that wiring: with that weight at zero, replacing the
    stored true needs with garbage must not change the loss at all.
    """

    trainer = OrganismTrainer(report_config(next_needs_weight=0.0))
    segment, hidden, bootstrap = trainer.collect_segment()
    advantages, returns = trainer._policy_targets(segment, bootstrap)

    def loss_with(needs: list[tuple[float, ...]], next_needs: list[tuple[float, ...]]):
        return float(
            trainer._loss(
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
                mx.array(np.asarray(next_needs, dtype=np.float32)),
                mx.array(np.asarray(segment.env_rewards, dtype=np.float32)),
                mx.array(np.asarray(segment.next_tokens, dtype=np.int32)),
                mx.array(np.asarray(needs, dtype=np.float32)),
                mx.array(np.asarray(segment.report_tokens, dtype=np.int32)),
                mx.array(np.asarray(segment.report_weights, dtype=np.float32)),
            )
        )

    honest = loss_with(segment.needs, segment.next_needs)
    garbage = [(0.0, 0.0, 0.0, 0.0)] * len(segment.needs)
    lied_to = loss_with(garbage, garbage)
    assert honest == pytest.approx(lied_to, abs=1e-6)


def test_the_report_head_does_receive_gradient_from_consequences():
    trainer = OrganismTrainer(report_config())
    segment, hidden, bootstrap = trainer.collect_segment()
    before = np.asarray(trainer.model.report_head.weight)
    trainer.update(segment, hidden, bootstrap)
    after = np.asarray(trainer.model.report_head.weight)
    assert not np.allclose(before, after)


def test_utterance_is_sampled_from_the_state_that_acts():
    trainer = OrganismTrainer(report_config())
    vector = mx.array(trainer.packet.vector()[None, None, :])
    tokens = mx.array(
        np.asarray(trainer.packet.tokens, dtype=np.int32)[None, None, :]
    )
    states, _ = trainer.model.core_states(vector, tokens, None)
    said = trainer._sample_report(states)
    assert len(said) == 2
    assert all(0 <= token < len(VOCAB) for token in said)


def test_checkpoint_round_trip_preserves_the_report_configuration():
    config = report_config()
    trainer = OrganismTrainer(config)
    path = "runs/organism/_test_report_checkpoint.npz"
    save_checkpoint(trainer.model, config, path)
    model, restored = load_organism_checkpoint(path)
    assert model.can_speak
    assert restored.report_task
    assert restored.report == config.report
    from pathlib import Path

    Path(path).unlink()
    Path(path + ".json").unlink()


def test_audits_run_read_only_on_an_untrained_organism():
    config = report_config()
    trainer = OrganismTrainer(config)
    before = np.asarray(trainer.model.report_head.weight).copy()
    result = evaluate_report(trainer.model, config, lives=2, seed_base=7)
    assert 0.0 <= result["report_fidelity"] <= 1.0
    assert result["lives"] == 2
    for intervention in ("zero", "shuffle", "freeze"):
        lesioned = evaluate_report(
            trainer.model,
            config,
            lives=1,
            seed_base=7,
            intervention=intervention,
        )
        assert 0.0 <= lesioned["report_fidelity"] <= 1.0
    after = np.asarray(trainer.model.report_head.weight)
    assert np.array_equal(before, after)


def test_observation_decoder_cannot_read_a_masked_body():
    """The senses alone must not reveal which need is lowest."""

    config = report_config()
    result = audit_observation_decoder(config, lives=12, seed_base=31)
    assert result["samples"] > 50
    assert result["heldout_accuracy"] < 0.55


def test_counterfactual_and_census_audits_produce_rows():
    config = report_config()
    trainer = OrganismTrainer(config)
    counterfactual = audit_counterfactual_body(
        trainer.model, config, lives=3, seed_base=41
    )
    assert "report_followed_body_rate" in counterfactual
    census = most_frequent_utterances(
        trainer.model, config, lives=2, seed_base=41, top=3
    )
    assert len(census) <= 3


def test_battery_covers_every_preregistered_condition():
    config = report_config()
    trainer = OrganismTrainer(config)
    rows = report_battery(trainer.model, config, lives=2, seed_base=53)
    conditions = {row["condition"] for row in rows}
    assert {
        "grounded",
        "listener_scrambled",
        "listener_mute",
        "organism_mute",
        "intervention_zero",
        "intervention_shuffle",
        "intervention_freeze",
        "heldout_birth_levels",
        "heldout_portions",
        "observation_decoder",
        "counterfactual_body",
    } <= conditions
    assert all(f"fixed_word_{need}" in conditions for need in REPORT_NEEDS)


def test_need_words_are_ordinary_vocabulary_members():
    """The mouth is not given a special three-way choice; it has a language."""

    trainer = OrganismTrainer(report_config())
    logits = trainer.model.report_logits(
        mx.zeros((1, 1, trainer.model.state_size))
    )
    assert logits.shape[-1] == len(VOCAB)
    need_ids = {TOKEN_TO_ID[NEED_TO_REPORT_WORD[need]] for need in REPORT_NEEDS}
    assert len(need_ids) == 3
    assert need_ids < set(range(len(VOCAB)))


def test_only_the_utterance_the_listener_acted_on_gets_credit():
    """An utterance said into the air must not be credited for what followed."""

    trainer = OrganismTrainer(report_config())
    segment, _, _ = trainer.collect_segment()
    weights = segment.report_weights
    assert len(weights) == len(segment)
    credited = sum(weights)
    assert credited > 0
    # At most one utterance per help window can have been acted on.
    period = trainer.config.report.help_period
    assert credited <= len(segment) / period + 1
    assert all(weight in (0.0, 1.0) for weight in weights)


def test_credit_lands_on_the_tick_the_listener_used():
    from homesocial.island.report import heard_need

    trainer = OrganismTrainer(report_config())
    world = trainer.world
    packet = world.reset(5)
    said_ticks: dict[int, str] = {}
    for _ in range(world.report.help_period * 2):
        tick = world.grid.step_count
        tokens = (TOKEN_TO_ID[NEED_TO_REPORT_WORD["water"]], PAD_ID)
        if tick % 2 == 0:
            tokens = (TOKEN_TO_ID["good"], PAD_ID)
        if heard_need(tokens) is not None:
            said_ticks[tick] = "need"
        world.hear(tokens)
        packet, _, terminated, truncated, info = world.step("wait")
        source = info.get("grant_source_tick")
        if source is not None:
            assert info["granted_need"] == "water"
            assert source in said_ticks
            assert source == max(t for t in said_ticks if t < world.grid.step_count)
        if terminated or truncated:
            break
