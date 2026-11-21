"""A learned self that also speaks: the learned_self GRU in caregiver-v1-size.

Development build. See docs/decisions/2026-11-21-caregiver-v1-ceiling-survey.md.

The learner of ``learned_self`` (one GRU state that acts, predicts its own
sensations and carries body knowledge across lives) gets a voice. At every
decision it chooses an actuator and, when the caregiver is present, whether to
speak and which two words: a resource word (6) and a modifier word (3). The
words' meanings are private to each identity and are never given: the only
learning signals are bodily reward and the auxiliary next-sensation
prediction. No signal rewards a word for being correct.

Architectural prior, disclosed: the utterance is produced by two heads, the
second conditioned on the first word. That makes two-slot structure available;
it does not say what any word means.

Inputs per decision: everything ``learned_self`` receives, plus whether the
caregiver is present and what the organism itself said at the last decision.

    PYTHONPATH=src .venv/bin/python -m homesocial.organism.speaking_self train \\
        --out-dir runs/organism/speaking_self_dev --init runs/organism/learned_self_dev/checkpoint
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np

from homesocial.organism import caregiver_v1 as cv
from homesocial.organism import learned_self as ls
from homesocial.organism.vec_body import ACTUATORS, CHANNELS, DECISIONS, LIVES, WINDOW, sample_bodies

VARIANT = "size"
W1, W2 = cv.WORDS, len(cv.SIZES)
D_SAID = W1 + W2 + 1
D_OBS = ls.D_OBS + 1 + D_SAID
D_PRED = ls.D_PRED
SOCIAL_MAX = 2

# Development evaluation identities: their own band, disjoint from the v1
# survey (6.3B), v0 (6.2B) and learned_self (6.1B).
DEV_EVAL_BASE = 6_400_000_000
EVAL_CELLS = ((1, 0), (2, 1), (3, 1), (3, 2), (4, 1), (4, 2), (5, 2))


def _check_seed_isolation() -> None:
    top = DEV_EVAL_BASE + 6 * 1_000_000 + 10 * 100_000 + 5_000 * 10
    if not cv.SURVEY_BASE + cv.BLOCKS * cv.BLOCK_STRIDE < DEV_EVAL_BASE < top < DEV_EVAL_BASE + 100_000_000:
        raise AssertionError("speaking_self development identities overlap another band")


_check_seed_isolation()


def dev_identities(per_cell: int, cells=EVAL_CELLS):
    if not 0 < per_cell <= 5_000:
        raise ValueError("per_cell must lie in 1..5000")
    idents, ks, socials = [], [], []
    for k, social in cells:
        for i in range(per_cell):
            idents.append(DEV_EVAL_BASE + k * 1_000_000 + social * 100_000 + i * 10)
            ks.append(k)
            socials.append(social)
    return idents, ks, socials


def sample_v1_bodies(rng: np.random.Generator, n: int, ks=(1, 2, 3, 4, 5)) -> cv.V1Bodies:
    """Training identities: Probe80's family from a NumPy stream, with 0..2
    social resources (always at least one own resource) and a private
    convention, drawn as ``caregiver_v1.v1_bodies`` draws them."""

    base = sample_bodies(rng, n, ks)
    social = np.minimum(rng.integers(0, SOCIAL_MAX + 1, n), base.k - 1)
    order = np.where(base.active, rng.random((n, CHANNELS)), 2.0).argsort(1)
    rank = order.argsort(1)
    base.blocked = base.active & (rank < social[:, None])
    words = rng.random((n, W1)).argsort(1)[:, :CHANNELS]  # channel c is named by word words[c]
    word_channel = -np.ones((n, W1), dtype=np.int64)
    np.put_along_axis(word_channel, words, np.arange(CHANNELS)[None].repeat(n, 0), 1)
    mod_meaning = rng.random((n, W2)).argsort(1)
    fields = {name: getattr(base, name) for name in cv.Bodies.__dataclass_fields__}
    return cv.V1Bodies(**fields, word_channel=word_channel, channel_word=words, mod_meaning=mod_meaning,
                       meaning_mod=np.argsort(mod_meaning, 1),
                       ration=cv.RATION_FACTOR * cv.species_need(base.blocked))


def features(env: cv.V1Body, said: np.ndarray) -> np.ndarray:
    """(N, D_OBS): learned_self's senses, the caregiver's presence, and the
    organism's own last utterance (``said``: -1 silent, else w1 * W2 + w2)."""

    n = env.n
    own = np.zeros((n, D_SAID), np.float32)
    spoke = np.flatnonzero(said >= 0)
    own[spoke, said[spoke] // W2] = 1.0
    own[spoke, W1 + said[spoke] % W2] = 1.0
    own[spoke, -1] = 1.0
    return np.concatenate([ls.features(env), env.present()[:, None].astype(np.float32), own], 1)


def _sample(logits: np.ndarray, rng: np.random.Generator):
    return ls.sample_actions(logits, rng)


# ----------------------------------------------------------------- model
class Agent(nn.Module):
    def __init__(self, hidden: int = 256) -> None:
        super().__init__()
        self.hidden = hidden
        self.enc1 = nn.Linear(D_OBS, hidden)
        self.enc2 = nn.Linear(hidden, hidden)
        self.gate_x = nn.Linear(hidden, 3 * hidden)
        self.gate_h = nn.Linear(hidden, 3 * hidden)
        self.pi = nn.Linear(hidden, ACTUATORS)
        self.v = nn.Linear(hidden, 1)
        self.talk = nn.Linear(hidden, 2)
        self.word1 = nn.Linear(hidden, W1)
        self.word2 = nn.Linear(hidden + W1, W2)
        self.pred1 = nn.Linear(hidden + ACTUATORS + D_SAID, hidden)
        self.pred2 = nn.Linear(hidden, D_PRED)
        self.probe1 = nn.Linear(hidden, hidden)
        self.probe2 = nn.Linear(hidden, ls.N_PROBE)
        for head in (self.pi, self.talk, self.word1, self.word2):
            head.weight = head.weight * 0.01

    cell = ls.Agent.cell
    probe = ls.Agent.probe

    def act(self, x, h, wipe):
        h = self.cell(x, h * (1.0 - wipe))
        return h, self.pi(h), self.v(h)[..., 0], self.talk(h), self.word1(h)

    def second(self, h, w1_onehot):
        return self.word2(mx.concatenate([h, w1_onehot], -1))

    def unroll(self, x, h0, wipe):
        return ls.Agent.unroll(self, x, h0, wipe)

    def predict(self, h, act_onehot, said):
        return self.pred2(nn.relu(self.pred1(mx.concatenate([h, act_onehot, said], -1))))


def warm_start(model: Agent, checkpoint: Path) -> None:
    """Copy a ``learned_self`` checkpoint into the speaking agent. New input
    columns and the new heads start as in a fresh agent (inputs at zero)."""

    old = dict(mx.load(str(checkpoint / "weights.npz")).items())
    new = dict(tree_flatten(model.parameters()))
    for name, value in old.items():
        if name not in new:
            continue
        if name == "enc1.weight":
            w = mx.zeros_like(new[name])
            w[:, :ls.D_OBS] = value
            new[name] = w
        elif name == "pred1.weight":
            w = new[name] * 0.0
            w[:, :value.shape[1]] = value
            new[name] = w
        elif new[name].shape == value.shape:
            new[name] = value
    model.update(tree_unflatten(list(new.items())))
    mx.eval(model.parameters())


# -------------------------------------------------------------- training
@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    envs: int = 1024
    horizon: int = 128
    hidden: int = 256
    ks: tuple[int, ...] = (1, 2, 3, 4, 5)
    lr: float = 3e-4
    gamma: float = 0.997
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 3
    minibatches: int = 4
    entropy: float = 0.01
    value_coef: float = 0.5
    predict_coef: float = 1.0
    max_grad_norm: float = 1.0
    alive_reward: float = 0.5 / DECISIONS
    survive_bonus: float = 0.5
    init: str = ""


AUX = ("policy_loss", "value_loss", "entropy", "predict_loss", "probe_loss", "probe_exact", "kl")


class Trainer:
    def __init__(self, config: TrainConfig) -> None:
        self.c = c = config
        self.rng = np.random.default_rng([c.seed, 20261121])
        mx.random.seed(c.seed)
        self.model = Agent(c.hidden)
        mx.eval(self.model.parameters())
        if c.init:
            warm_start(self.model, Path(c.init))
        self.opt = optim.Adam(learning_rate=c.lr)
        self.opt.init(self.model.trainable_parameters())
        self._loss = nn.value_and_grad(self.model, self._loss_fn)
        self.decisions, self.updates, self.seconds = 0, 0, 0.0
        self._begin_world()

    def _begin_world(self) -> None:
        c = self.c
        self.env = cv.V1Body(sample_v1_bodies(self.rng, c.envs, c.ks), VARIANT,
                             alive_reward=c.alive_reward, survive_bonus=c.survive_bonus)
        self.targets = ls.probe_targets(self.env.b)
        self.said = -np.ones(c.envs, dtype=np.int64)
        self.x = features(self.env, self.said)
        self.wipe = np.ones(c.envs, dtype=np.float32)
        self.h = mx.zeros((c.envs, c.hidden))

    def rollout(self):
        c, env = self.c, self.env
        N, T = c.envs, c.horizon
        f32 = np.float32
        buf = {"x": np.zeros((N, T, D_OBS), f32), "wipe": np.zeros((N, T), f32),
               "action": np.zeros((N, T), np.int32), "talk": np.zeros((N, T), np.int32),
               "w1": np.zeros((N, T), np.int32), "w2": np.zeros((N, T), np.int32),
               "present": np.zeros((N, T), f32), "logp": np.zeros((N, T), f32),
               "value": np.zeros((N, T), f32), "reward": np.zeros((N, T), f32),
               "done": np.zeros((N, T), f32), "pred": np.zeros((N, T, D_PRED), f32),
               "pred_mask": np.zeros((N, T), f32), "probe": np.zeros((N, T, ls.N_PROBE), f32)}
        lives, survived = np.zeros(5), np.zeros(5)
        spoke = visits = 0
        h0 = h = self.h
        for t in range(T):
            h, a_logits, value, t_logits, w1_logits = self.model.act(mx.array(self.x), h, mx.array(self.wipe)[:, None])
            mx.eval(h, a_logits, value, t_logits, w1_logits)
            action, lp_a = _sample(np.array(a_logits), self.rng)
            talk, lp_t = _sample(np.array(t_logits), self.rng)
            w1, lp_1 = _sample(np.array(w1_logits), self.rng)
            w2_logits = self.model.second(h, mx.array(np.eye(W1, dtype=f32)[w1]))
            w2, lp_2 = _sample(np.array(w2_logits), self.rng)
            present = env.present()
            talk = talk * present
            speech = np.where(talk > 0, w1 * W2 + w2, -1)
            logp = lp_a + present * (lp_t + talk * (lp_1 + lp_2))
            spoke += int(talk.sum())
            visits += int(present.sum())
            k_now, life_now = env.b.k.copy(), env.life.copy()
            buf["x"][:, t], buf["wipe"][:, t], buf["probe"][:, t] = self.x, self.wipe, self.targets
            reward, life_ended = env.step(action, speech)
            ended = np.flatnonzero(life_ended)
            np.add.at(lives, k_now[ended] - 1, 1)
            np.add.at(survived, k_now[ended] - 1, env.survived[ended, life_now[ended]])
            done = env.finished.copy()
            index = np.flatnonzero(done)
            while len(index):
                env.replace(index, sample_v1_bodies(self.rng, len(index), c.ks))
                index = index[env.finished[index]]
            if done.any():
                self.targets[done] = ls.probe_targets(env.b.take(np.flatnonzero(done)))
            self.said = np.where(done, -1, speech)
            self.x, self.wipe = features(env, self.said), done.astype(f32)
            for key, val in (("action", action), ("talk", talk), ("w1", w1), ("w2", w2), ("present", present),
                             ("logp", logp), ("value", np.array(value)), ("reward", reward), ("done", done)):
                buf[key][:, t] = val
            buf["pred"][:, t] = self.x[:, 2 * WINDOW:4 * WINDOW]
            buf["pred_mask"][:, t] = ~life_ended
        self.h = h
        _, _, last, _, _ = self.model.act(mx.array(self.x), h, mx.array(self.wipe)[:, None])
        buf["advantage"], buf["return"] = ls.gae(buf["reward"], buf["value"], buf["done"], np.array(last),
                                                 c.gamma, c.lam)
        return buf, h0, {"lives": lives, "survived": survived, "reward": float(buf["reward"].mean()),
                         "speak_rate": spoke / max(visits, 1)}

    def _loss_fn(self, x, h0, wipe, action, talk, w1, w2, present, old_logp, advantage, target, pred, pred_mask,
                 probe):
        c = self.c
        h = self.model.unroll(x, h0, wipe)
        a_logits, value = self.model.pi(h), self.model.v(h)[..., 0]
        w1_hot = (w1[..., None] == mx.arange(W1)).astype(mx.float32)
        parts = [(a_logits, action, 1.0), (self.model.talk(h), talk, present),
                 (self.model.word1(h), w1, present * talk), (self.model.second(h, w1_hot), w2, present * talk)]
        logp = 0.0
        entropy = 0.0
        for logits, chosen, weight in parts:
            lp_all = nn.log_softmax(logits, axis=-1)
            logp = logp + weight * mx.take_along_axis(lp_all, chosen[..., None], axis=-1)[..., 0]
            entropy = entropy + (weight * -(mx.exp(lp_all) * lp_all).sum(-1)).mean()
        ratio = mx.exp(logp - old_logp)
        policy = -mx.minimum(ratio * advantage, mx.clip(ratio, 1.0 - c.clip, 1.0 + c.clip) * advantage).mean()
        value_loss = ((value - target) ** 2).mean()
        act_hot = (action[..., None] == mx.arange(ACTUATORS)).astype(mx.float32)
        w2_hot = (w2[..., None] == mx.arange(W2)).astype(mx.float32)
        spoken = (present * talk).astype(mx.float32)[..., None]
        said = mx.concatenate([w1_hot * spoken, w2_hot * spoken, spoken], -1)
        error = ((self.model.predict(h, act_hot, said) - pred) ** 2).mean(-1)
        predict = (error * pred_mask).sum() / mx.maximum(pred_mask.sum(), 1.0)
        guess = self.model.probe(h)
        probe_loss = nn.losses.binary_cross_entropy(guess, probe, with_logits=True)
        exact = ((guess > 0) == (probe > 0.5)).all(axis=-1).astype(mx.float32).mean()
        loss = policy + c.value_coef * value_loss - c.entropy * entropy + c.predict_coef * predict + probe_loss
        return loss, mx.stack([policy, value_loss, entropy, predict, probe_loss, exact, (old_logp - logp).mean()])

    def update(self, buf, h0):
        c = self.c
        stats = []
        keys = ("x", "wipe", "action", "talk", "w1", "w2", "present", "logp")
        for _ in range(c.epochs):
            for index in np.array_split(self.rng.permutation(c.envs), c.minibatches):
                adv = buf["advantage"][index]
                adv = ((adv - adv.mean()) / (adv.std() + 1e-8)).astype(np.float32)
                args = [mx.array(buf[k][index]) for k in keys]
                (_, aux), grads = self._loss(
                    args[0], h0[mx.array(index)], *args[1:], mx.array(adv),
                    mx.array(buf["return"][index].astype(np.float32)), mx.array(buf["pred"][index]),
                    mx.array(buf["pred_mask"][index]), mx.array(buf["probe"][index]))
                grads, _ = optim.clip_grad_norm(grads, c.max_grad_norm)
                self.opt.update(self.model, grads)
                mx.eval(self.model.parameters(), self.opt.state, aux)
                stats.append(np.array(aux))
        return dict(zip(AUX, (float(v) for v in np.mean(stats, 0))))

    def train_step(self):
        start = perf_counter()
        buf, h0, lived = self.rollout()
        record = self.update(buf, h0)
        self.decisions += self.c.envs * self.c.horizon
        self.updates += 1
        self.seconds += perf_counter() - start
        record.update(lived)
        return record

    save = ls.Trainer.save

    @classmethod
    def load(cls, directory: Path):
        state = json.loads((directory / "state.json").read_text())
        config = state["config"]
        config["ks"] = tuple(config["ks"])
        config["init"] = ""  # weights come from the checkpoint itself
        trainer = cls(TrainConfig(**config))
        trainer.model.load_weights(str(directory / "weights.npz"))
        trainer.opt.state = tree_unflatten(list(mx.load(str(directory / "optimizer.npz")).items()))
        mx.eval(trainer.model.parameters(), trainer.opt.state)
        trainer.decisions, trainer.updates, trainer.seconds = state["decisions"], state["updates"], state["seconds"]
        trainer.rng.bit_generator.state = state["rng"]
        trainer._begin_world()
        return trainer, state


# ------------------------------------------------------------ evaluation
LESIONS = ("none", "mute", "reset", "scramble_w2")


def evaluate(model: Agent, bodies: cv.V1Bodies, rng: np.random.Generator, lesion: str = "none") -> dict:
    """Six lives of every identity.

    - ``mute``: nothing the organism says is heard (its voice is cut);
    - ``reset``: recurrent state emptied at every birth;
    - ``scramble_w2``: the modifier word it chose is replaced by a random one
      before the caregiver hears it (tests whether the second word matters).

    Speech statistics, per heard request: did it name a social resource; how
    much of what arrived was absorbed; and *novel combinations*: a word pair
    this identity has never said before whose two words it has each said in
    other pairs (a new combination of known words)."""

    if lesion not in LESIONS:
        raise ValueError(f"unknown lesion: {lesion}")
    env = cv.V1Body(bodies, VARIANT)
    n = env.n
    rows = np.arange(n)
    said = -np.ones(n, dtype=np.int64)
    h = mx.zeros((n, model.hidden))
    seen = np.zeros((n, W1, W2), dtype=bool)
    stats = {key: np.zeros(n) for key in ("visits", "requests", "social_named", "novel", "novel_social",
                                          "novel_meanings")}
    meaning_count = np.zeros((n, W2))
    while not env.finished.all():
        wipe = env.birth.astype(np.float32) if lesion == "reset" else np.zeros(n, np.float32)
        h, a_logits, _, t_logits, w1_logits = model.act(mx.array(features(env, said)), h, mx.array(wipe)[:, None])
        mx.eval(h, a_logits, t_logits, w1_logits)
        action = _sample(np.array(a_logits), rng)[0]
        talk = _sample(np.array(t_logits), rng)[0]
        w1 = _sample(np.array(w1_logits), rng)[0]
        w2 = _sample(np.array(model.second(h, mx.array(np.eye(W1, dtype=np.float32)[w1]))), rng)[0]
        present = env.present()
        talk = talk * present
        live = ~env.finished
        stats["visits"] += present
        heard_w2 = rng.integers(0, W2, n) if lesion == "scramble_w2" else w2
        speech = np.where(talk > 0, w1 * W2 + w2, -1)
        spoken = talk > 0
        channel = bodies.word_channel[rows, w1]
        social = spoken & (channel >= 0) & bodies.blocked[rows, np.maximum(channel, 0)]
        knows1 = seen[rows, w1].any(1)
        knows2 = seen[rows, :, w2].any(1)
        novel = spoken & ~seen[rows, w1, w2] & knows1 & knows2
        stats["requests"] += spoken & live
        stats["social_named"] += social
        stats["novel"] += novel
        stats["novel_social"] += novel & social
        np.add.at(meaning_count, (rows[social], bodies.mod_meaning[rows[social], w2[social]]), 1)
        seen[rows[spoken], w1[spoken], w2[spoken]] = True
        heard = np.where(lesion == "mute", -1, np.where(talk > 0, w1 * W2 + heard_w2, -1))
        env.step(action, heard)
        said = np.where(env.birth & (env.life == 0), -1, speech)
    out = {}
    for k, social in sorted({(int(k), int(s)) for k, s in zip(bodies.k, bodies.blocked.sum(1))}):
        cell = (bodies.k == k) & (bodies.blocked.sum(1) == social)
        req = max(stats["requests"][cell].sum(), 1)
        out[f"K{k}_social{social}"] = {
            "identities": int(cell.sum()), "survival": float(env.survived[cell].mean()),
            "survival_by_life": [float(v) for v in env.survived[cell].mean(0)],
            "speak_rate": float(stats["requests"][cell].sum() / max(stats["visits"][cell].sum(), 1)),
            "social_named": float(stats["social_named"][cell].sum() / req),
            "novel_per_identity": float(stats["novel"][cell].mean()),
            "novel_social_share": float(stats["novel_social"][cell].sum() / max(stats["novel"][cell].sum(), 1)),
            "size_meaning_share": [float(v) for v in meaning_count[cell].sum(0) / max(meaning_count[cell].sum(), 1)],
            "wasted_per_life": float(env.wasted[cell].mean() / LIVES),
        }
    return out


def evaluate_all(model: Agent, bodies: cv.V1Bodies, seed: int) -> dict:
    return {lesion: evaluate(model, bodies, np.random.default_rng([seed, 9, i]), lesion)
            for i, lesion in enumerate(LESIONS)}


def references(bodies: cv.V1Bodies) -> dict:
    out = {}
    for arm in ("silent", "composed", "single_0.4", "single_0.6", "species"):
        survived = cv.run_arm(bodies, VARIANT, arm)["survived"]
        out[arm] = {f"K{k}_social{s}": float(survived[(bodies.k == k) & (bodies.blocked.sum(1) == s)].mean())
                    for k, s in sorted({(int(k), int(s)) for k, s in zip(bodies.k, bodies.blocked.sum(1))})}
    return out


def table(result: dict, refs: dict) -> str:
    lines = ["six-life mean survival on development identities (caregiver-v1-size)",
             f"{'cell':<12}{'silent*':>8}{'single.4*':>10}{'composed*':>10}{'learned':>9}{'mute':>7}{'reset':>7}"
             f"{'scr.w2':>8}{'speak':>7}{'right':>7}{'novel':>7}{'novel ok':>9}   size shares",
             "  (* reference speakers with true levels)"]
    for cell in result["none"]:
        r = result["none"][cell]
        shares = " ".join(f"{v:.2f}" for v in r["size_meaning_share"])
        lines.append(f"{cell:<12}{refs['silent'][cell]:>8.3f}{refs['single_0.4'][cell]:>10.3f}"
                     f"{refs['composed'][cell]:>10.3f}{r['survival']:>9.3f}{result['mute'][cell]['survival']:>7.3f}"
                     f"{result['reset'][cell]['survival']:>7.3f}{result['scramble_w2'][cell]['survival']:>8.3f}"
                     f"{r['speak_rate']:>7.2f}{r['social_named']:>7.2f}{r['novel_per_identity']:>7.1f}"
                     f"{r['novel_social_share']:>9.2f}   {shares}")
    return "\n".join(lines)


# ------------------------------------------------------------- execution
def manifest(config: TrainConfig, per_cell: int) -> dict:
    here = Path(__file__)
    files = [here, here.with_name("caregiver_v1.py"), here.with_name("learned_self.py"), here.with_name("vec_body.py")]
    out = {"config": asdict(config), "dev_eval_base": DEV_EVAL_BASE, "dev_eval_per_cell": per_cell,
           "eval_cells": EVAL_CELLS, "features": D_OBS, "variant": VARIANT, "mlx": mx.__version__,
           "numpy": np.__version__, "pins": {p.name: sha256(p.read_bytes()).hexdigest() for p in files}}
    if config.init:
        out["init_weights_sha256"] = sha256((Path(config.init) / "weights.npz").read_bytes()).hexdigest()
    return out


def train(args: argparse.Namespace) -> None:
    out: Path = args.out_dir
    checkpoint = out / "checkpoint"
    if (checkpoint / "state.json").exists():
        if not args.resume:
            raise SystemExit(f"{checkpoint} exists; pass --resume to continue it or choose another --out-dir")
        trainer, state = Trainer.load(checkpoint)
        next_eval, evaluations = state["next_eval"], state["evaluations"]
        print(f"resumed at {trainer.decisions:,} decisions", flush=True)
    else:
        config = TrainConfig(seed=args.seed, envs=args.envs, horizon=args.horizon, hidden=args.hidden,
                             lr=args.lr, entropy=args.entropy, epochs=args.epochs,
                             init=str(args.init) if args.init else "")
        trainer = Trainer(config)
        next_eval, evaluations = 0, 0
        out.mkdir(parents=True, exist_ok=True)
        (out / "manifest.json").write_text(json.dumps(manifest(config, args.eval_per_cell), indent=1))
    bodies = cv.v1_bodies(*dev_identities(args.eval_per_cell))
    refs = references(bodies)
    started = perf_counter()
    mark_decisions, mark_seconds = trainer.decisions, trainer.seconds
    window: list[dict] = []
    while True:
        capped = trainer.decisions >= args.max_decisions or (perf_counter() - started) / 3600.0 >= args.max_hours
        if trainer.decisions >= next_eval or capped:
            result = evaluate_all(trainer.model, bodies, trainer.c.seed)
            evaluations += 1
            next_eval = trainer.decisions + args.eval_every
            ls._append(out / "curve.jsonl", {"decisions": trainer.decisions, "updates": trainer.updates,
                                              "train_seconds": trainer.seconds, "eval": result, "reference": refs})
            trainer.save(checkpoint, {"next_eval": next_eval, "evaluations": evaluations})
            print(f"\n== {trainer.decisions:,} decisions, {trainer.seconds / 3600:.2f} h of training ==\n"
                  + table(result, refs), flush=True)
        if capped:
            break
        window.append(trainer.train_step())
        if trainer.updates % args.log_every == 0:
            lives = np.sum([r["lives"] for r in window], 0)
            survived = np.sum([r["survived"] for r in window], 0)
            record = {"decisions": trainer.decisions, "updates": trainer.updates,
                      "decisions_per_second": (trainer.decisions - mark_decisions)
                      / max(trainer.seconds - mark_seconds, 1e-9),
                      "train_survival_by_k": [float(s / max(l, 1)) for s, l in zip(survived, lives)],
                      **{key: float(np.mean([r[key] for r in window])) for key in (*AUX, "reward", "speak_rate")}}
            mark_decisions, mark_seconds = trainer.decisions, trainer.seconds
            ls._append(out / "train.jsonl", record)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else
                                  [round(x, 3) for x in v] if isinstance(v, list) else v)
                              for k, v in record.items()}), flush=True)
            window = []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    t = sub.add_parser("train")
    t.add_argument("--out-dir", type=Path, required=True)
    t.add_argument("--init", type=Path, default=None)
    t.add_argument("--resume", action="store_true")
    t.add_argument("--max-hours", type=float, default=9.0)
    t.add_argument("--max-decisions", type=float, default=3e9)
    t.add_argument("--eval-every", type=float, default=2e7)
    t.add_argument("--eval-per-cell", type=int, default=100)
    t.add_argument("--log-every", type=int, default=10)
    for name, kind in (("seed", int), ("envs", int), ("horizon", int), ("hidden", int),
                       ("lr", float), ("entropy", float), ("epochs", int)):
        t.add_argument(f"--{name}", type=kind, default=getattr(TrainConfig, name))
    e = sub.add_parser("eval")
    e.add_argument("--out-dir", type=Path, required=True)
    e.add_argument("--per-cell", type=int, default=300)
    args = parser.parse_args()
    if args.mode == "train":
        train(args)
    else:
        trainer, state = Trainer.load(args.out_dir / "checkpoint")
        bodies = cv.v1_bodies(*dev_identities(args.per_cell))
        result, refs = evaluate_all(trainer.model, bodies, trainer.c.seed), references(bodies)
        (args.out_dir / "eval.json").write_text(json.dumps(
            {"decisions": state["decisions"], "per_cell": args.per_cell, "eval": result, "reference": refs}, indent=1))
        print(f"{state['decisions']:,} decisions\n" + table(result, refs))


if __name__ == "__main__":
    main()
