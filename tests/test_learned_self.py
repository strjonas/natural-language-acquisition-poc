"""Guards for the learned-self budget experiment: information boundary, seed
isolation, the equivalence of acting and training passes, and resumability."""

from dataclasses import replace

import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np
import pytest

from homesocial.organism import active_self as probe
from homesocial.organism import learned_self as ls
from homesocial.organism.saturation_structure import make_physical_world
from homesocial.organism.vec_body import VecBody, bodies_from_identities, sample_bodies

TINY = ls.TrainConfig(envs=8, horizon=12, hidden=16, minibatches=2, epochs=1)


def _flat(model):
    return {k: np.array(v) for k, v in tree_flatten(model.parameters())}


def test_development_identities_are_isolated_from_every_probe80_band():
    ls._check_seed_isolation()
    idents, ks = ls.dev_identities(ls.DEV_MAX_PER_K)
    assert len(set(idents)) == len(idents) == 5 * ls.DEV_MAX_PER_K
    probe80 = {probe.identity_seed(base, block, k, i)
               for base in (probe.SURVEY_BASE, probe.DIAGNOSTIC_BASE, probe.DEVELOPMENT_BASE)
               for block in range(probe.BLOCKS) for k in range(1, 6) for i in range(probe.IDENTITIES)}
    donors = {ident + 50 for ident in probe80}
    assert not (set(idents) & (probe80 | donors))
    assert min(idents) > max(probe80 | donors)
    with pytest.raises(ValueError):
        ls.dev_identities(ls.DEV_MAX_PER_K + 1)


def test_features_are_built_from_sensations_only():
    """Two bodies with different hidden structure but identical public
    buffers look identical to the learner."""

    rng = np.random.default_rng(0)
    env = VecBody(sample_bodies(rng, 6))
    env.step(rng.integers(0, ls.ACTUATORS, env.n))
    before = ls.features(env)
    assert before.shape == (6, ls.D_OBS) and before.dtype == np.float32 and np.isfinite(before).all()
    hidden = sample_bodies(rng, 6)
    env.b.assign(np.arange(6), hidden)
    env.levels[:] = rng.random(env.levels.shape)
    np.testing.assert_array_equal(ls.features(env), before)


def test_all_probe_facts_right_is_exactly_probe80s_partition():
    idents, ks = ls.dev_identities(4)
    bodies = bodies_from_identities(idents, ks)
    targets = ls.probe_targets(bodies)
    for i, (ident, k) in enumerate(zip(idents, ks)):
        world = make_physical_world(seed=probe.domain_seed(ident, "map"), k=k)
        truth = probe.true_partition(world)
        usable = targets[i, :ls.ACTUATORS] > 0.5
        groups: dict[int, set[int]] = {}
        for a in np.flatnonzero(usable):
            groups.setdefault(int(a), {int(a)})
        for (a, b), same in zip(ls.PAIRS, targets[i, ls.ACTUATORS:] > 0.5):
            if same:
                merged = groups[int(a)] | groups[int(b)]
                for member in merged:
                    groups[member] = merged
        decoded = tuple(sorted({tuple(sorted(g)) for g in groups.values()}))
        assert decoded == truth


def test_gae_stops_at_the_end_of_an_identity():
    reward = np.array([[1.0, 1.0, 1.0]])
    value = np.zeros((1, 3))
    done = np.array([[0.0, 1.0, 0.0]])
    advantage, target = ls.gae(reward, value, done, np.array([10.0]), gamma=0.5, lam=1.0)
    np.testing.assert_allclose(advantage[0], [1.5, 1.0, 6.0])
    np.testing.assert_allclose(target, advantage)


def test_initial_weights_vary_with_the_seed():
    a, b = _flat(ls.Trainer(TINY).model), _flat(ls.Trainer(replace(TINY, seed=1)).model)
    assert any(not np.array_equal(a[k], b[k]) for k in a)
    again = _flat(ls.Trainer(TINY).model)
    assert all(np.array_equal(a[k], again[k]) for k in a)


def test_training_pass_reproduces_the_acting_pass():
    """PPO's ratio is only meaningful if unrolling the stored segment gives
    the log-probabilities the organism acted with, memory wipes included."""

    trainer = ls.Trainer(TINY)
    for _ in range(40):  # run until some identities have ended, so wipes occur mid-segment
        buf, h0, _ = trainer.rollout()
        if buf["wipe"][:, 1:].any():
            break
    assert buf["wipe"][:, 1:].any()
    h = trainer.model.unroll(mx.array(buf["x"]), h0, mx.array(buf["wipe"]))
    logits, value = trainer.model.heads(h)
    logp = np.array(mx.take_along_axis(ls.nn.log_softmax(logits, axis=-1), mx.array(buf["action"])[..., None], axis=-1))[..., 0]
    np.testing.assert_allclose(logp, buf["logp"], atol=2e-5)
    np.testing.assert_allclose(np.array(value), buf["value"], atol=2e-5)


def test_one_update_learns_something_and_stays_finite():
    trainer = ls.Trainer(TINY)
    before = _flat(trainer.model)
    record = trainer.train_step()
    after = _flat(trainer.model)
    assert all(np.isfinite(v) for k, v in record.items() if isinstance(v, float))
    assert trainer.decisions == TINY.envs * TINY.horizon and trainer.updates == 1
    changed = {k for k in before if not np.array_equal(before[k], after[k])}
    assert {"pi.weight", "gate_h.weight", "pred2.weight", "probe2.weight"} <= changed


def test_probe_never_shapes_the_recurrent_state():
    trainer = ls.Trainer(TINY)
    buf, h0, _ = trainer.rollout()

    def probe_only(model):
        guess = model.probe(model.unroll(mx.array(buf["x"]), h0, mx.array(buf["wipe"])))
        return ls.nn.losses.binary_cross_entropy(guess, mx.array(buf["probe"]), with_logits=True)

    _, grads = ls.nn.value_and_grad(trainer.model, probe_only)(trainer.model)
    norms = {k: float(np.abs(np.array(v)).sum()) for k, v in tree_flatten(grads)}
    assert norms["probe2.weight"] > 0
    assert all(v == 0.0 for k, v in norms.items() if not k.startswith("probe"))


def test_checkpoint_round_trip(tmp_path):
    trainer = ls.Trainer(TINY)
    trainer.train_step()
    trainer.save(tmp_path, {"next_eval": 5, "evaluations": 1})
    loaded, state = ls.Trainer.load(tmp_path)
    assert (state["next_eval"], state["evaluations"]) == (5, 1)
    assert (loaded.decisions, loaded.updates, loaded.c) == (trainer.decisions, trainer.updates, trainer.c)
    a, b = _flat(trainer.model), _flat(loaded.model)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert int(loaded.opt.state["step"]) == int(trainer.opt.state["step"])
    saved = np.random.default_rng(0)
    saved.bit_generator.state = state["rng"]  # the stream continues where it stopped
    np.testing.assert_array_equal(sample_bodies(saved, TINY.envs, TINY.ks).rates, loaded.env.b.rates)
    assert np.isfinite(loaded.train_step()["policy_loss"])


def test_evaluation_is_deterministic_and_covers_every_life():
    model = ls.Trainer(TINY).model
    bodies = bodies_from_identities(*ls.dev_identities(3, ks=(1, 3)))
    first = ls.evaluate_all(model, bodies, seed=0)
    assert first == ls.evaluate_all(model, bodies, seed=0)
    assert set(first) == {"none", "reset"} and set(first["none"]) == {"1", "3"}
    row = first["none"]["3"]
    assert row["identities"] == 3 and len(row["survival_by_life"]) == ls.LIVES
    assert 0.0 <= row["probe_facts_life6"] <= 1.0
    with pytest.raises(ValueError):
        ls.evaluate(model, bodies, np.random.default_rng(0), lesion="yoked")
