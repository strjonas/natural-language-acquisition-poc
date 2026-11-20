"""A learned self on Probe80's body family: development budget experiment.

See docs/decisions/2026-11-20-learned-self-budget-experiment.md.

Question. Probe80's self-model is a hand-derived posterior for physics its
designer wrote. Can one recurrent learner, trained across many bodies and given
exactly the same lived information, come to do the same job -- find out which
body it has and keep it alive -- and at what training budget?

The learner (``Agent``) is a GRU with four heads on one recurrent state:

- a policy over the ten opaque actuators (acting);
- a value estimate (training only);
- a predictor of its own next six ticks of sensation given the actuator it is
  about to use (prediction; an auxiliary objective, not a reward);
- a diagnostic structure probe, trained on the true actuator partition behind
  a stop-gradient. The probe is an instrument with privileged labels. It never
  shapes the recurrent state and is not the organism's own report.

Inputs per decision: the two scalar sensations after each of the last six
ticks (and their tick-to-tick changes), the actuator it chose, and birth/clock
flags. That is the information ``SelfPosterior`` receives, nothing more.

Learning sources: reward is bodily (a small amount per decision survived plus
a bonus for completing a life); recurrent PPO; next-sensation prediction.
Training bodies come from a NumPy stream; evaluation identities come from a
dedicated seed band and are built by the exact ``LivedBody`` generator.

    PYTHONPATH=src .venv/bin/python -m homesocial.organism.learned_self train \
        --out-dir runs/organism/learned_self_dev --max-hours 9
    PYTHONPATH=src .venv/bin/python -m homesocial.organism.learned_self eval \
        --out-dir runs/organism/learned_self_dev
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

from homesocial.organism import active_self
from homesocial.organism.vec_body import (
    ACTUATORS, CHANNELS, DECISIONS, LIVES, PROBE80_EXACT_LIFE6, PROBE80_SURVIVAL, WINDOW,
    Bodies, VecBody, bodies_from_identities, reference_survival, sample_bodies,
)

DELTA_SCALE = 10.0
D_OBS = 4 * WINDOW + ACTUATORS + 3
D_PRED = 2 * WINDOW
PAIRS = np.array([(a, b) for a in range(ACTUATORS) for b in range(a + 1, ACTUATORS)])
N_PROBE = ACTUATORS + len(PAIRS)

# Evaluation identities. Disjoint from every Probe80 band (5.0B-5.2B, used and
# to stay untouched) and from the earlier probes below them. Development only:
# a confirmatory comparison needs its own band and a preregistration.
DEV_EVAL_BASE = 6_100_000_000
DEV_K_STRIDE, DEV_IDENTITY_STRIDE, DEV_MAX_PER_K = 1_000_000, 100, 5_000


def _check_seed_isolation() -> None:
    probe80_top = max(active_self.SURVEY_BASE, active_self.DIAGNOSTIC_BASE, active_self.DEVELOPMENT_BASE) + 100_000_000
    low = DEV_EVAL_BASE + DEV_K_STRIDE
    high = DEV_EVAL_BASE + 5 * DEV_K_STRIDE + DEV_MAX_PER_K * DEV_IDENTITY_STRIDE + 50
    if not probe80_top < low < high < DEV_EVAL_BASE + 100_000_000:
        raise AssertionError("development evaluation identities overlap another seed band")
    if DEV_MAX_PER_K * DEV_IDENTITY_STRIDE > DEV_K_STRIDE:
        raise AssertionError("development identities of different K overlap")


_check_seed_isolation()


def dev_identities(per_k: int, ks=(1, 2, 3, 4, 5)) -> tuple[list[int], list[int]]:
    if not 0 < per_k <= DEV_MAX_PER_K:
        raise ValueError(f"per_k must lie in 1..{DEV_MAX_PER_K}")
    idents = [DEV_EVAL_BASE + k * DEV_K_STRIDE + i * DEV_IDENTITY_STRIDE for k in ks for i in range(per_k)]
    return idents, [k for k in ks for _ in range(per_k)]


# ------------------------------------------------------------- interface
def features(env: VecBody) -> np.ndarray:
    """(N, D_OBS) what the organism senses at a decision. No physics."""

    n = env.n
    before_total = np.concatenate([env.prev_total[:, None], env.obs_total[:, :-1]], 1)
    before_min = np.concatenate([env.prev_min[:, None], env.obs_min[:, :-1]], 1)
    chosen = np.zeros((n, ACTUATORS))
    acted = np.flatnonzero(env.prev_action >= 0)
    chosen[acted, env.prev_action[acted]] = 1.0
    clock = np.stack([env.birth.astype(float), np.minimum(env.life, LIVES - 1) / (LIVES - 1),
                      env.decision / (DECISIONS - 1)], 1)
    return np.concatenate([
        env.obs_total / CHANNELS, env.obs_min,
        (env.obs_total - before_total) * DELTA_SCALE, (env.obs_min - before_min) * DELTA_SCALE,
        chosen, clock,
    ], 1).astype(np.float32)


def probe_targets(bodies: Bodies) -> np.ndarray:
    """(N, N_PROBE) the true structure as binary facts: which actuators do
    anything, and which pairs restore the same resource. All 55 right is
    exactly Probe80's "exact partition". Privileged; used only by the probe."""

    usable = bodies.actionable()
    a, b = PAIRS[:, 0], PAIRS[:, 1]
    same = usable[:, a] & usable[:, b] & (bodies.target[:, a] == bodies.target[:, b])
    return np.concatenate([usable, same], 1).astype(np.float32)


def sample_actions(logits: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    p = np.exp(logits - logits.max(1, keepdims=True)).astype(np.float64)
    p /= p.sum(1, keepdims=True)
    action = np.minimum((p.cumsum(1) < rng.random(len(p))[:, None]).sum(1), p.shape[1] - 1)
    return action, np.log(p[np.arange(len(p)), action]).astype(np.float32)


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
        self.pred1 = nn.Linear(hidden + ACTUATORS, hidden)
        self.pred2 = nn.Linear(hidden, D_PRED)
        self.probe1 = nn.Linear(hidden, hidden)
        self.probe2 = nn.Linear(hidden, N_PROBE)
        self.pi.weight = self.pi.weight * 0.01

    def cell(self, x: mx.array, h: mx.array) -> mx.array:
        e = nn.relu(self.enc2(nn.relu(self.enc1(x))))
        gx, gh = self.gate_x(e), self.gate_h(h)
        H = self.hidden
        r = mx.sigmoid(gx[..., :H] + gh[..., :H])
        z = mx.sigmoid(gx[..., H:2 * H] + gh[..., H:2 * H])
        n = mx.tanh(gx[..., 2 * H:] + r * gh[..., 2 * H:])
        return (1.0 - z) * n + z * h

    def heads(self, h: mx.array) -> tuple[mx.array, mx.array]:
        return self.pi(h), self.v(h)[..., 0]

    def act(self, x: mx.array, h: mx.array, wipe: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """One decision. ``wipe`` (N, 1) is 1 where memory must start empty."""

        h = self.cell(x, h * (1.0 - wipe))
        return (h, *self.heads(h))

    def unroll(self, x: mx.array, h0: mx.array, wipe: mx.array) -> mx.array:
        h, states = h0, []
        for t in range(x.shape[1]):
            h = self.cell(x[:, t], h * (1.0 - wipe[:, t:t + 1]))
            states.append(h)
        return mx.stack(states, 1)

    def predict(self, h: mx.array, action_onehot: mx.array) -> mx.array:
        return self.pred2(nn.relu(self.pred1(mx.concatenate([h, action_onehot], -1))))

    def probe(self, h: mx.array) -> mx.array:
        return self.probe2(nn.relu(self.probe1(mx.stop_gradient(h))))


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


def gae(reward, value, done, last_value, gamma, lam):
    """Generalized advantage over (N, T). ``done`` marks the last decision of
    an identity; lives within an identity are one continuing problem."""

    advantage = np.zeros_like(reward)
    running, following = np.zeros(len(reward)), last_value
    for t in reversed(range(reward.shape[1])):
        live = 1.0 - done[:, t]
        delta = reward[:, t] + gamma * following * live - value[:, t]
        running = delta + gamma * lam * live * running
        advantage[:, t] = running
        following = value[:, t]
    return advantage, advantage + value


AUX = ("policy_loss", "value_loss", "entropy", "predict_loss", "probe_loss", "probe_exact", "kl")


class Trainer:
    def __init__(self, config: TrainConfig) -> None:
        self.c = c = config
        self.rng = np.random.default_rng([c.seed, 20261120])
        mx.random.seed(c.seed)
        self.model = Agent(c.hidden)
        mx.eval(self.model.parameters())
        self.opt = optim.Adam(learning_rate=c.lr)
        self.opt.init(self.model.trainable_parameters())
        self._loss = nn.value_and_grad(self.model, self._loss_fn)
        self.decisions, self.updates, self.seconds = 0, 0, 0.0
        self._begin_world()

    def _begin_world(self) -> None:
        c = self.c
        self.env = VecBody(sample_bodies(self.rng, c.envs, c.ks),
                           alive_reward=c.alive_reward, survive_bonus=c.survive_bonus)
        self.targets = probe_targets(self.env.b)
        self.x = features(self.env)
        self.wipe = np.ones(c.envs, dtype=np.float32)
        self.h = mx.zeros((c.envs, c.hidden))

    # one segment of lived experience
    def rollout(self) -> tuple[dict, mx.array, dict]:
        c, env = self.c, self.env
        N, T = c.envs, c.horizon
        f32 = np.float32
        buf = {"x": np.zeros((N, T, D_OBS), f32), "wipe": np.zeros((N, T), f32),
               "action": np.zeros((N, T), np.int32), "logp": np.zeros((N, T), f32),
               "value": np.zeros((N, T), f32), "reward": np.zeros((N, T), f32),
               "done": np.zeros((N, T), f32), "pred": np.zeros((N, T, D_PRED), f32),
               "pred_mask": np.zeros((N, T), f32), "probe": np.zeros((N, T, N_PROBE), f32)}
        lives, survived = np.zeros(5), np.zeros(5)
        h0 = h = self.h
        for t in range(T):
            h, logits, value = self.model.act(mx.array(self.x), h, mx.array(self.wipe)[:, None])
            mx.eval(h, logits, value)
            action, logp = sample_actions(np.array(logits), self.rng)
            k_now, life_now = env.b.k.copy(), env.life.copy()
            buf["x"][:, t], buf["wipe"][:, t], buf["probe"][:, t] = self.x, self.wipe, self.targets
            reward, life_ended = env.step(action)
            ended = np.flatnonzero(life_ended)
            np.add.at(lives, k_now[ended] - 1, 1)
            np.add.at(survived, k_now[ended] - 1, env.survived[ended, life_now[ended]])
            done = env.finished.copy()
            index = np.flatnonzero(done)
            while len(index):
                env.replace(index, sample_bodies(self.rng, len(index), c.ks))
                index = index[env.finished[index]]
            if done.any():
                self.targets[done] = probe_targets(env.b.take(np.flatnonzero(done)))
            self.x, self.wipe = features(env), done.astype(f32)
            buf["action"][:, t], buf["logp"][:, t], buf["value"][:, t] = action, logp, np.array(value)
            buf["reward"][:, t], buf["done"][:, t] = reward, done
            buf["pred"][:, t] = self.x[:, 2 * WINDOW:4 * WINDOW]
            buf["pred_mask"][:, t] = ~life_ended
        self.h = h
        _, _, last = self.model.act(mx.array(self.x), h, mx.array(self.wipe)[:, None])
        buf["advantage"], buf["return"] = gae(buf["reward"], buf["value"], buf["done"], np.array(last), c.gamma, c.lam)
        return buf, h0, {"lives": lives, "survived": survived, "reward": float(buf["reward"].mean())}

    def _loss_fn(self, x, h0, wipe, action, old_logp, advantage, target, pred, pred_mask, probe):
        c = self.c
        h = self.model.unroll(x, h0, wipe)
        logits, value = self.model.heads(h)
        logp_all = nn.log_softmax(logits, axis=-1)
        logp = mx.take_along_axis(logp_all, action[..., None], axis=-1)[..., 0]
        ratio = mx.exp(logp - old_logp)
        policy = -mx.minimum(ratio * advantage, mx.clip(ratio, 1.0 - c.clip, 1.0 + c.clip) * advantage).mean()
        value_loss = ((value - target) ** 2).mean()
        entropy = -(mx.exp(logp_all) * logp_all).sum(-1).mean()
        onehot = (action[..., None] == mx.arange(ACTUATORS)).astype(mx.float32)
        error = ((self.model.predict(h, onehot) - pred) ** 2).mean(-1)
        predict = (error * pred_mask).sum() / mx.maximum(pred_mask.sum(), 1.0)
        guess = self.model.probe(h)
        probe_loss = nn.losses.binary_cross_entropy(guess, probe, with_logits=True)
        exact = ((guess > 0) == (probe > 0.5)).all(axis=-1).astype(mx.float32).mean()
        loss = policy + c.value_coef * value_loss - c.entropy * entropy + c.predict_coef * predict + probe_loss
        return loss, mx.stack([policy, value_loss, entropy, predict, probe_loss, exact, (old_logp - logp).mean()])

    def update(self, buf: dict, h0: mx.array) -> dict:
        c = self.c
        stats = []
        for _ in range(c.epochs):
            for index in np.array_split(self.rng.permutation(c.envs), c.minibatches):
                advantage = buf["advantage"][index]
                advantage = ((advantage - advantage.mean()) / (advantage.std() + 1e-8)).astype(np.float32)
                (_, aux), grads = self._loss(
                    mx.array(buf["x"][index]), h0[mx.array(index)], mx.array(buf["wipe"][index]),
                    mx.array(buf["action"][index]), mx.array(buf["logp"][index]), mx.array(advantage),
                    mx.array(buf["return"][index].astype(np.float32)), mx.array(buf["pred"][index]),
                    mx.array(buf["pred_mask"][index]), mx.array(buf["probe"][index]))
                grads, _ = optim.clip_grad_norm(grads, c.max_grad_norm)
                self.opt.update(self.model, grads)
                mx.eval(self.model.parameters(), self.opt.state, aux)
                stats.append(np.array(aux))
        return dict(zip(AUX, (float(v) for v in np.mean(stats, 0))))

    def train_step(self) -> dict:
        start = perf_counter()
        buf, h0, lived = self.rollout()
        record = self.update(buf, h0)
        self.decisions += self.c.envs * self.c.horizon
        self.updates += 1
        self.seconds += perf_counter() - start
        record.update(lived)
        return record

    # ---------------------------------------------------------- persistence
    def save(self, directory: Path, extra: dict | None = None) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(str(directory / "weights.tmp.npz"))
        mx.savez(str(directory / "optimizer.tmp.npz"), **dict(tree_flatten(self.opt.state)))
        state = {"config": asdict(self.c), "decisions": self.decisions, "updates": self.updates,
                 "seconds": self.seconds, "rng": self.rng.bit_generator.state, **(extra or {})}
        (directory / "state.tmp.json").write_text(json.dumps(state))
        for name in ("weights", "optimizer"):
            (directory / f"{name}.tmp.npz").replace(directory / f"{name}.npz")
        (directory / "state.tmp.json").replace(directory / "state.json")

    @classmethod
    def load(cls, directory: Path) -> tuple["Trainer", dict]:
        """Resume from a checkpoint. Weights, optimizer, counters and the
        training stream are restored; the bodies alive at the save are not, so
        training continues on freshly drawn identities."""

        state = json.loads((directory / "state.json").read_text())
        config = state["config"]
        config["ks"] = tuple(config["ks"])
        trainer = cls(TrainConfig(**config))
        trainer.model.load_weights(str(directory / "weights.npz"))
        trainer.opt.state = tree_unflatten(list(mx.load(str(directory / "optimizer.npz")).items()))
        mx.eval(trainer.model.parameters(), trainer.opt.state)
        trainer.decisions, trainer.updates, trainer.seconds = state["decisions"], state["updates"], state["seconds"]
        trainer.rng.bit_generator.state = state["rng"]
        trainer._begin_world()
        return trainer, state


# ------------------------------------------------------------ evaluation
def evaluate(model: Agent, bodies: Bodies, rng: np.random.Generator, *, lesion: str = "none",
             greedy: bool = False) -> dict:
    """Six lives of every identity under the learned policy.

    ``lesion="reset"`` empties the recurrent state at every birth: the same
    learner without anything retained from this body's earlier lives."""

    if lesion not in ("none", "reset"):
        raise ValueError(f"unknown lesion: {lesion}")
    env = VecBody(bodies)
    n = env.n
    targets = probe_targets(bodies)
    exact = np.zeros((n, LIVES), dtype=bool)
    facts = np.zeros((n, LIVES))
    h = mx.zeros((n, model.hidden))
    while not env.finished.all():
        wipe = env.birth.astype(np.float32) if lesion == "reset" else np.zeros(n, np.float32)
        h, logits, _ = model.act(mx.array(features(env)), h, mx.array(wipe)[:, None])
        guess = model.probe(h)
        mx.eval(h, logits, guess)
        live = np.flatnonzero(~env.finished)
        right = (np.array(guess)[live] > 0) == (targets[live] > 0.5)
        exact[live, env.life[live]] = right.all(1)
        facts[live, env.life[live]] = right.mean(1)
        logits = np.array(logits)
        env.step(logits.argmax(1) if greedy else sample_actions(logits, rng)[0])
    out = {}
    for k in sorted(set(int(v) for v in bodies.k)):
        rows = bodies.k == k
        out[str(k)] = {"identities": int(rows.sum()),
                       "survival": float(env.survived[rows].mean()),
                       "survival_by_life": [float(v) for v in env.survived[rows].mean(0)],
                       "probe_exact_life1": float(exact[rows, 0].mean()),
                       "probe_exact_life6": float(exact[rows, -1].mean()),
                       "probe_facts_life6": float(facts[rows, -1].mean())}
    return out


def evaluate_all(model: Agent, bodies: Bodies, seed: int) -> dict:
    return {lesion: evaluate(model, bodies, np.random.default_rng([seed, 7, i]), lesion=lesion)
            for i, lesion in enumerate(("none", "reset"))}


def reference(bodies: Bodies, seed: int = 0) -> dict:
    """Random and oracle survival on the same identities, by K."""

    out = {}
    for i, policy in enumerate(("random", "oracle")):
        survived = reference_survival(bodies, policy, np.random.default_rng([seed, 11, i]))
        out[policy] = {str(k): float(survived[bodies.k == k].mean()) for k in sorted(set(int(v) for v in bodies.k))}
    return out


def table(result: dict, refs: dict) -> str:
    lines = ["six-life mean survival on development identities; Probe80 columns are its published "
             "survey (other bodies, same family)",
             f"{'K':<3}{'random':>8}{'oracle':>8}{'p80 rand':>10}{'p80 post':>10}{'p80 orac':>10}"
             f"{'learned':>9}{'reset':>8}{'probe6':>8}{'p80 exact6':>12}"]
    for k in sorted(result["none"], key=int):
        j = int(k) - 1
        lines.append(f"{k:<3}{refs['random'][k]:>8.3f}{refs['oracle'][k]:>8.3f}"
                     f"{PROBE80_SURVIVAL['random'][j]:>10.3f}{PROBE80_SURVIVAL['posterior'][j]:>10.3f}"
                     f"{PROBE80_SURVIVAL['oracle'][j]:>10.3f}{result['none'][k]['survival']:>9.3f}"
                     f"{result['reset'][k]['survival']:>8.3f}{result['none'][k]['probe_exact_life6']:>8.3f}"
                     f"{PROBE80_EXACT_LIFE6[j]:>12.2f}")
    return "\n".join(lines)


# ------------------------------------------------------------- execution
def manifest(config: TrainConfig, per_k: int) -> dict:
    here = Path(__file__)
    files = [here, here.with_name("vec_body.py"), here.with_name("active_self.py"),
             here.with_name("saturation_structure.py"), here.parent.parent / "island/expanded_body.py"]
    return {"config": asdict(config), "dev_eval_base": DEV_EVAL_BASE, "dev_eval_per_k": per_k,
            "features": D_OBS, "mlx": mx.__version__, "numpy": np.__version__,
            "pins": {p.name: sha256(p.read_bytes()).hexdigest() for p in files}}


def _append(path: Path, record: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")


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
                             lr=args.lr, entropy=args.entropy, epochs=args.epochs)
        trainer = Trainer(config)
        next_eval, evaluations = 0, 0
        out.mkdir(parents=True, exist_ok=True)
        (out / "manifest.json").write_text(json.dumps(manifest(config, args.eval_per_k), indent=1))
    bodies = bodies_from_identities(*dev_identities(args.eval_per_k))
    refs = reference(bodies)
    started = perf_counter()
    mark_decisions, mark_seconds = trainer.decisions, trainer.seconds
    window: list[dict] = []
    while True:
        capped = (trainer.decisions >= args.max_decisions
                  or (perf_counter() - started) / 3600.0 >= args.max_hours)
        if trainer.decisions >= next_eval or capped:
            result = evaluate_all(trainer.model, bodies, trainer.c.seed)
            evaluations += 1
            next_eval = trainer.decisions + args.eval_every
            _append(out / "curve.jsonl", {"decisions": trainer.decisions, "updates": trainer.updates,
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
                      **{key: float(np.mean([r[key] for r in window])) for key in (*AUX, "reward")}}
            mark_decisions, mark_seconds = trainer.decisions, trainer.seconds
            _append(out / "train.jsonl", record)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else
                                  [round(x, 3) for x in v] if isinstance(v, list) else v)
                              for k, v in record.items()}), flush=True)
            window = []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    t = sub.add_parser("train")
    t.add_argument("--out-dir", type=Path, required=True)
    t.add_argument("--resume", action="store_true")
    t.add_argument("--max-hours", type=float, default=9.0)
    t.add_argument("--max-decisions", type=float, default=2e9)
    t.add_argument("--eval-every", type=float, default=2e7)
    t.add_argument("--eval-per-k", type=int, default=200)
    t.add_argument("--log-every", type=int, default=10)
    for name, kind in (("seed", int), ("envs", int), ("horizon", int), ("hidden", int),
                       ("lr", float), ("entropy", float), ("epochs", int)):
        t.add_argument(f"--{name}", type=kind, default=getattr(TrainConfig, name))
    e = sub.add_parser("eval")
    e.add_argument("--out-dir", type=Path, required=True)
    e.add_argument("--per-k", type=int, default=500)
    r = sub.add_parser("reference")
    r.add_argument("--per-k", type=int, default=300)
    args = parser.parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        trainer, state = Trainer.load(args.out_dir / "checkpoint")
        bodies = bodies_from_identities(*dev_identities(args.per_k))
        result = evaluate_all(trainer.model, bodies, trainer.c.seed)
        (args.out_dir / "eval.json").write_text(json.dumps(
            {"decisions": state["decisions"], "per_k": args.per_k, "eval": result, "reference": reference(bodies)}, indent=1))
        print(f"{state['decisions']:,} decisions\n" + table(result, reference(bodies)))
    else:
        bodies = bodies_from_identities(*dev_identities(args.per_k))
        refs = reference(bodies)
        for k in sorted(refs["random"], key=int):
            j = int(k) - 1
            print(f"K{k}: random {refs['random'][k]:.3f} (Probe80 {PROBE80_SURVIVAL['random'][j]:.3f})   "
                  f"oracle {refs['oracle'][k]:.3f} (Probe80 {PROBE80_SURVIVAL['oracle'][j]:.3f})")


if __name__ == "__main__":
    main()
