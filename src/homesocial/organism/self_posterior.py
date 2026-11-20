"""An online self-posterior over a body whose structure is not given.

The organism is told how many sensor channels its two bodily sensations
summarize (``CHANNELS``) and that bodily resources deplete, can be restored
up to a bound, and are occasionally shocked. It is *not* told how many of the
channels are resources, which opaque actuator restores which resource, how
much an actuator restores, how fast anything depletes, or where any resource
is now. Every one of those is inferred from two scalars per tick -- the sum
and the minimum of the channel levels -- and from its own actuator choices.

Representation (a Rao-Blackwellized particle posterior):

- each particle samples a *discrete* self-structure: which channels are active
  resources, and which channel each actuator drives (assigned lazily, at the
  actuator's first use, from the locally optimal proposal);
- given that structure, the *continuous* self-parameters -- levels, depletion
  rates, portion sizes and shock size -- are carried as one exact Gaussian
  per particle and updated by scalar Kalman steps under sampled modes
  (saturating or not, which channel was shocked, which channel is the minimum);
- the inequalities the sensations imply -- every other channel lies above the
  observed minimum, a saturating grant filled its channel, levels never exceed
  one -- enter by moment-matched truncation (assumed-density filtering), and
  the probability of each implied inequality enters the particle's evidence;
- an optional structural hazard lets any use of a known actuator re-propose
  which channel it drives, at prior rate ``hazard``. This is a changepoint
  prior over the self, not a detector with a threshold.

Nothing in this module reads a simulator. Its only inputs are the actuator the
organism itself chose and the two scalar sensations that followed.

The same posterior is used three ways, which is the point:

- to act (``sample_action``: imagine a self consistent with the evidence and
  act as that self would; or ``certainty_action``: the uncertainty lesion);
- to predict (``evidence``: the running log predictive probability of its own
  sensations);
- to report on itself (``structure_report``: the partition of its actuators
  into resources it believes, with a confidence that can be scored).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CHANNELS = 5
ACTUATORS = 10
_X = np.arange(0, CHANNELS)
_R = np.arange(CHANNELS, 2 * CHANNELS)
_P = np.arange(2 * CHANNELS, 2 * CHANNELS + ACTUATORS)
_S = 2 * CHANNELS + ACTUATORS
DIM = _S + 1
_LOG2PI = float(np.log(2.0 * np.pi))


@dataclass(frozen=True)
class PosteriorConfig:
    """Fixed hyperparameters. Priors are species-level expectations, not facts
    about any individual body."""

    particles: int = 400
    hazard: float = 0.0
    sensor_sigma: float = 0.003
    shock_hazard: float = 0.025
    ess_fraction: float = 0.5
    prior_level: tuple[float, float] = (0.6, 0.2)
    prior_rate: tuple[float, float] = (0.012, 0.01)
    prior_portion: tuple[float, float] = (0.45, 0.3)
    prior_shock: tuple[float, float] = (0.3, 0.15)
    # 'mixture': a shock's channel is not sampled; the moment-matched
    # mixture over channels (weighted by how well each explains the next
    # minimum) is carried instead. 'sample': draw one channel per particle.
    shock_attribution: str = "mixture"
    # Posterior predictive self-criticism. A tick whose sensations have log
    # predictive probability below -falsification_nats under the whole
    # posterior (about 1e-13 at 30) falsifies the current self-model: it then
    # rebuilds itself by replaying its remembered experience from the prior,
    # keeping the best-evidenced of ``rebuild_attempts`` rebuilds. 0 disables.
    falsification_nats: float = 0.0
    rebuild_attempts: int = 3

    def __post_init__(self) -> None:
        if self.particles < 2:
            raise ValueError("a particle posterior needs at least two particles")
        if not 0.0 <= self.hazard < 1.0:
            raise ValueError("hazard must lie in [0, 1)")
        if self.sensor_sigma <= 0.0:
            raise ValueError("sensor_sigma must be positive")
        if not 0.0 < self.shock_hazard < 1.0:
            raise ValueError("shock_hazard must lie in (0, 1)")
        if not 0.0 < self.ess_fraction <= 1.0:
            raise ValueError("ess_fraction must lie in (0, 1]")
        if self.falsification_nats < 0.0 or self.rebuild_attempts < 1:
            raise ValueError("falsification_nats must be >= 0 and rebuild_attempts >= 1")
        if self.shock_attribution not in ("mixture", "sample"):
            raise ValueError("shock_attribution must be 'mixture' or 'sample'")


def log_norm_cdf(z: np.ndarray | float) -> np.ndarray:
    """Stable vectorized log Phi(z) (complementary error function in log
    space; relative error below 1.2e-7)."""

    z = np.asarray(z, dtype=np.float64)
    x = -z / np.sqrt(2.0)
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.5 * ax)
    poly = -1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (
        -0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (
            -0.82215223 + t * 0.17087277))))))))
    log_erfc = np.log(t) - ax * ax + poly
    with np.errstate(over="ignore"):
        upper = np.log1p(-0.5 * np.exp(log_erfc))
    return np.where(x >= 0.0, np.log(0.5) + log_erfc, upper)


def _norm_logpdf(x, mean, var):
    return -0.5 * (x - mean) ** 2 / var - 0.5 * np.log(var) - 0.5 * _LOG2PI


def _logsumexp(values: np.ndarray) -> float:
    top = float(np.max(values))
    if not np.isfinite(top):
        return top
    return top + float(np.log(np.sum(np.exp(values - top))))


@dataclass(frozen=True)
class StructureReport:
    """What the organism believes about its own structure, and how sure it is.

    ``partition`` groups actuator IDs by the resource they restore; actuators
    believed inert are omitted. ``confidence`` is the posterior probability of
    exactly that partition, including the organism's uncertainty about
    actuators it has never tried."""

    partition: tuple[tuple[int, ...], ...]
    confidence: float
    resource_count: int
    count_posterior: tuple[float, ...]
    tried: tuple[int, ...]
    falsified: bool = False


class SelfPosterior:
    """Online posterior over this body's structure, parameters and state."""

    def __init__(self, config: PosteriorConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng
        self.n = config.particles
        self.ready = False
        self.evidence = 0.0
        self.collapses = 0
        self.tried: set[int] = set()
        self.memory: list[tuple] = []
        self.falsified = False
        self.rebuilds = 0
        self._life_rebuilds = 0
        self._replaying = False
        self._var_m = config.sensor_sigma ** 2
        self._var_n = config.sensor_sigma ** 2

    # ------------------------------------------------------------------ birth
    def _fresh_structure(self, count: int):
        rng, cfg = self.rng, self.config
        k = rng.integers(1, CHANNELS + 1, size=count)
        active = np.zeros((count, CHANNELS), bool)
        for i in range(count):
            active[i, rng.choice(CHANNELS, int(k[i]), replace=False)] = True
        drives = -np.ones((count, ACTUATORS), dtype=np.int64)
        mean = np.zeros((count, DIM))
        cov = np.zeros((count, DIM, DIM))
        mean[:, _R] = np.where(active, cfg.prior_rate[0], 0.0)
        cov[:, _R, _R] = np.where(active, cfg.prior_rate[1] ** 2, 0.0)
        mean[:, _P] = cfg.prior_portion[0]
        cov[:, _P, _P] = cfg.prior_portion[1] ** 2
        mean[:, _S] = cfg.prior_shock[0]
        cov[:, _S, _S] = cfg.prior_shock[1] ** 2
        return active, drives, mean, cov

    def birth(self, total: float, minimum: float, *, retain: bool) -> None:
        """A new life begins. ``retain`` keeps what was learned about this
        body's structure and constants; levels are always new."""

        cfg, N = self.config, self.n
        if not self._replaying:
            if not (retain and self.ready):
                self.memory = []
                self.falsified = False
            self.memory.append(("birth", bool(retain and self.ready), float(total), float(minimum)))
            self._life_rebuilds = 0
            if retain and self.ready and self.falsified:
                # Between lives, a self-model known to be broken re-thinks its past.
                self._rebuild()
                return
        if retain and self.ready:
            self._take(self._resample_index())
            self.cov[:, _X, :] = 0.0
            self.cov[:, :, _X] = 0.0
        else:
            self.active, self.drives, self.mean, self.cov = self._fresh_structure(N)
            self.tried = set()
            self.evidence = 0.0
        self.mean[:, _X] = np.where(self.active, cfg.prior_level[0], 1.0)
        index = np.arange(N)
        for c in range(CHANNELS):
            self.cov[index, c, c] = np.where(self.active[:, c], cfg.prior_level[1] ** 2, 0.0)
        self.logw = np.full(N, -np.log(N))
        h = np.zeros((N, DIM))
        h[:, _X] = 1.0
        self._kalman(h, np.zeros(N), total, self._var_m, weight=True)
        self._project_physical()
        self._observe_minimum(minimum)
        self.ready = True
        self._normalize()
        self._maybe_resample(force=True)

    # ------------------------------------------------------- Gaussian updates
    def _kalman(self, h, b, y, var, weight=False):
        Sh = np.einsum("nde,ne->nd", self.cov, h)
        s = np.einsum("nd,nd->n", h, Sh) + var
        pred = np.einsum("nd,nd->n", h, self.mean) + b
        if weight:
            self.logw = self.logw + _norm_logpdf(y, pred, s)
        gain = Sh / s[:, None]
        self.mean = self.mean + gain * (y - pred)[:, None]
        cov = self.cov - gain[:, :, None] * Sh[:, None, :]
        self.cov = 0.5 * (cov + np.transpose(cov, (0, 2, 1)))

    def _kalman_coordinate(self, coord, y, var):
        """Kalman update observing one state coordinate per particle."""

        rows = np.arange(self.n)
        Sh = self.cov[rows, coord, :].copy()
        s = Sh[rows, coord] + var
        gain = Sh / s[:, None]
        self.mean += gain * (y - self.mean[rows, coord])[:, None]
        self.cov -= gain[:, :, None] * Sh[:, None, :]

    def _truncate(self, coords, signs, c, mask):
        """Moment-matched update for ``sum(signs * x[coords]) >= c`` on masked
        particles (``coords``/``signs`` are per-particle index/sign arrays of
        shape (N, q)). Returns each particle's log probability of the
        constraint before the update. Constraints that are not binding
        (more than eight standard deviations inside) are skipped: their update
        is below 1e-14 and their log probability below 1e-15."""

        N = self.n
        out = np.zeros(N)
        if not mask.any():
            return out
        rows = np.arange(N)
        m = (signs * self.mean[rows[:, None], coords]).sum(1)
        block = self.cov[rows[:, None, None], coords[:, :, None], coords[:, None, :]]
        s2 = np.einsum("nq,nqr,nr->n", signs, block, signs)
        deterministic = s2 < 1e-14
        s = np.sqrt(np.maximum(s2, 1e-14))
        z = (m - c) / s
        out = np.where(mask & deterministic, np.where(m >= c - 1e-6, 0.0, -1e3), 0.0)
        live = mask & ~deterministic & (z < 8.0)
        if not live.any():
            return out
        r = rows[live]
        zl = z[live]
        log_phi = log_norm_cdf(zl)
        with np.errstate(all="ignore"):
            lam = np.exp(-0.5 * zl * zl - 0.5 * _LOG2PI - log_phi)
        s2l = np.maximum(s2[live], 1e-14)
        shrink = np.minimum(lam * (lam + zl) / s2l, 1.0 / s2l)
        Sh = np.einsum("nq,nqd->nd", signs[live], self.cov[r[:, None], coords[live], :])
        self.mean[r] += Sh * (lam / np.sqrt(s2l))[:, None]
        self.cov[r] -= (Sh[:, :, None] * Sh[:, None, :]) * shrink[:, None, None]
        out[r] = log_phi
        return out

    def _project_physical(self):
        """Levels never exceed one and rates are never negative. A projection,
        not evidence: it carries no weight."""

        N = self.n
        minus, plus = -np.ones((N, 1)), np.ones((N, 1))
        for j in range(CHANNELS):
            self._truncate(np.full((N, 1), _X[j]), minus, -1.0, self.active[:, j])
            self._truncate(np.full((N, 1), _R[j]), plus, 0.0, self.active[:, j])

    def _observe_minimum(self, minimum: float) -> None:
        N = self.n
        index = np.arange(N)
        mx = self.mean[:, _X]
        vx = np.maximum(np.diagonal(self.cov, axis1=1, axis2=2)[:, _X], 0.0) + self._var_n
        density = _norm_logpdf(minimum, mx, vx)
        above = log_norm_cdf((mx - minimum) / np.sqrt(vx))
        proposal = density + above.sum(1, keepdims=True) - above
        top = proposal.max(1, keepdims=True)
        top = np.where(np.isfinite(top), top, 0.0)
        p = np.exp(proposal - top)
        self.logw = self.logw + np.log(np.maximum(p.sum(1), 1e-300)) + top[:, 0]
        cdf = np.cumsum(p, 1)
        u = self.rng.uniform(0.0, 1.0, N) * cdf[:, -1]
        m = np.minimum((cdf < u[:, None]).sum(1), CHANNELS - 1)
        approximate = proposal[index, m] - density[index, m]
        self._kalman_coordinate(m, minimum, self._var_n)
        refined = np.zeros(N)
        slack = np.sqrt(self._var_n)
        plus = np.ones((N, 1))
        for j in range(CHANNELS):
            refined += self._truncate(np.full((N, 1), _X[j]), plus, minimum - slack, self.active[:, j] & (m != j))
        self.logw = self.logw + refined - approximate

    def _mix_shock(self, rows: np.ndarray, minimum: float) -> None:
        """Moment-matched shock of unknown channel J on the given particles.
        J's weights are the prior over active channels times how well a shock
        there explains the minimum about to be observed. x <- x - pi*s, then
        the attribution uncertainty (Sigma_ss + mu_s^2)(diag pi - pi pi^T) is
        added to the level block."""

        mean, cov = self.mean[rows], self.cov[rows]
        act = self.active[rows]
        mx = mean[:, _X]
        vx = np.maximum(np.diagonal(cov, axis1=1, axis2=2)[:, _X], 0.0)
        ms, vs = mean[:, _S], np.maximum(cov[:, _S, _S], 0.0)
        cxs = cov[:, _X, _S]
        # candidate j: channel j lowered by s
        eye = np.eye(CHANNELS)[None, :, :]
        m_j = mx[:, None, :] - eye * ms[:, None, None]
        v_j = vx[:, None, :] + eye * (vs[:, None, None] - 2.0 * cxs[:, None, :]) + self._var_n
        v_j = np.maximum(v_j, 1e-12)
        dens = _norm_logpdf(minimum, m_j, v_j)
        above = log_norm_cdf((m_j - minimum) / np.sqrt(v_j))
        per_min = dens + above.sum(2, keepdims=True) - above
        top = per_min.max(2, keepdims=True)
        top = np.where(np.isfinite(top), top, 0.0)
        log_like = np.log(np.maximum(np.exp(per_min - top).sum(2), 1e-300)) + top[:, :, 0]
        log_like = np.where(act, log_like, -np.inf)
        best = log_like.max(1, keepdims=True)
        pi = np.where(act, np.exp(log_like - best), 0.0)
        pi = pi / pi.sum(1, keepdims=True)
        # linear part: x <- x - pi s
        mean[:, _X] -= pi * ms[:, None]
        cov[:, _X, :] -= pi[:, :, None] * cov[:, _S, :][:, None, :]
        cov[:, :, _X] -= cov[:, :, _S][:, :, None] * pi[:, None, :]
        # attribution uncertainty
        second = (vs + ms * ms)[:, None, None]
        q = second * (np.einsum("nc,cd->ncd", pi, np.eye(CHANNELS)) - pi[:, :, None] * pi[:, None, :])
        cov[:, _X[:, None], _X[None, :]] += q
        self.mean[rows] = mean
        self.cov[rows] = 0.5 * (cov + np.transpose(cov, (0, 2, 1)))

    # -------------------------------------------------------------- one tick
    def observe(self, grant: int | None, total: float, minimum: float) -> None:
        """One physical tick: absorb ``grant`` (the organism's own actuator
        choice, or None), deplete, perhaps be shocked; then sense ``total``
        (sum of channel levels) and ``minimum``."""

        if not self.ready:
            raise RuntimeError("observe() before birth()")
        cfg, N, rng = self.config, self.n, self.rng
        index = np.arange(N)
        start = self.logw.copy()
        mean, cov, active = self.mean, self.cov, self.active
        if grant is None:
            modes_c = np.zeros((N, 1), dtype=np.int64)
            modes_k = np.full((N, 1), -1)
            log_mode = np.zeros((N, 1))
            log_channel = np.zeros((N, CHANNELS))
        else:
            if not 0 <= grant < ACTUATORS:
                raise ValueError("unknown actuator")
            self.tried.add(int(grant))
            cs = np.repeat(np.arange(CHANNELS), 3)
            ks = np.tile(np.arange(3), CHANNELS)
            modes_c = np.broadcast_to(cs, (N, 3 * CHANNELS)).copy()
            modes_k = np.broadcast_to(ks, (N, 3 * CHANNELS)).copy()
            lazy = self.drives[:, grant] < 0
            eps = cfg.hazard
            keep = np.log(1.0 - eps + eps / CHANNELS) if eps > 0.0 else 0.0
            move = np.log(eps / CHANNELS) if eps > 0.0 else -np.inf
            current = np.arange(CHANNELS)[None, :] == self.drives[:, grant][:, None]
            log_channel = np.where(lazy[:, None], -np.log(CHANNELS), np.where(current, keep, move))
            pa = _P[grant]
            msum = mean[:, _X] + mean[:, [pa]]
            vsum = np.maximum(
                np.diagonal(cov, axis1=1, axis2=2)[:, _X] + cov[:, pa, pa][:, None] + 2.0 * cov[:, _X, pa],
                1e-12,
            )
            z = (1.0 - msum) / np.sqrt(vsum)
            kind = np.stack(
                [
                    np.where(active, log_norm_cdf(z), -np.inf),
                    np.where(active, log_norm_cdf(-z), -np.inf),
                    np.where(active, -np.inf, 0.0),
                ],
                axis=2,
            )
            log_mode = (log_channel[:, :, None] + kind).reshape(N, 3 * CHANNELS)
        G = modes_c.shape[1]
        mixture = cfg.shock_attribution == "mixture"
        if mixture:
            # effective shock: a uniformly chosen channel that is active
            k_active = active.sum(1).astype(float)
            p_shock = cfg.shock_hazard * k_active / CHANNELS
            log_shock_n = np.stack([np.log1p(-p_shock), np.log(np.maximum(p_shock, 1e-300))], axis=1)
            shock_on = np.concatenate([np.zeros((N, 1)), np.ones((N, 1))], axis=1)
        else:
            log_shock_n = np.broadcast_to(
                np.concatenate([[np.log(1.0 - cfg.shock_hazard)], np.full(CHANNELS, np.log(cfg.shock_hazard / CHANNELS))]),
                (N, CHANNELS + 1))
            shock_on = np.concatenate([np.zeros((N, 1)), active.astype(float)], axis=1)
        # Observation h = h0 + delta(mode): h0 sums levels and subtracts rates;
        # delta adds the granted portion (+p_a, non-saturating), removes the
        # pre-grant level of a saturated channel (-x_c, plus constant 1), and
        # subtracts the shock size on a shocked active channel (-s).
        S0 = cov[:, :, _X].sum(-1) - cov[:, :, _R].sum(-1)
        base_var = S0[:, _X].sum(-1) - S0[:, _R].sum(-1)
        base_pred = mean[:, _X].sum(-1) - mean[:, _R].sum(-1)
        # grant part: coefficient vectors as (index, sign) with optional constant
        if grant is not None:
            pa = _P[grant]
            g_idx = np.where(modes_k == 0, pa, np.where(modes_k == 1, modes_c, 0))
            g_sgn = np.where(modes_k == 0, 1.0, np.where(modes_k == 1, -1.0, 0.0))
            g_const = np.where(modes_k == 1, 1.0, 0.0)
        else:
            g_idx = np.zeros((N, 1), dtype=np.int64)
            g_sgn = np.zeros((N, 1))
            g_const = np.zeros((N, 1))
        rows = index[:, None]
        g_S0 = g_sgn * S0[rows, g_idx]
        g_var = g_sgn * g_sgn * cov[rows, g_idx, g_idx]
        g_mu = g_sgn * mean[rows, g_idx]
        g_covS = g_sgn * cov[rows, g_idx, _S]
        sh_S0 = -shock_on * S0[:, [_S]]
        sh_var = shock_on * cov[:, _S, _S][:, None]
        sh_mu = -shock_on * mean[:, [_S]]
        # total for (g, j): h0.S0 + 2 delta.S0 + delta.Sigma.delta
        s = (base_var[:, None, None] + 2.0 * (g_S0[:, :, None] + sh_S0[:, None, :])
             + g_var[:, :, None] + sh_var[:, None, :]
             + 2.0 * g_covS[:, :, None] * (-shock_on)[:, None, :]
             + self._var_m)
        pred = base_pred[:, None, None] + g_mu[:, :, None] + sh_mu[:, None, :] + g_const[:, :, None]
        joint = log_mode[:, :, None] + log_shock_n[:, None, :] + _norm_logpdf(total, pred, s)
        flat = joint.reshape(N, -1)
        top = flat.max(1, keepdims=True)
        top = np.where(np.isfinite(top), top, 0.0)
        p = np.exp(flat - top)
        self.logw = self.logw + np.log(np.maximum(p.sum(1), 1e-300)) + top[:, 0]
        cdf = np.cumsum(p, 1)
        u = rng.uniform(0.0, 1.0, N) * cdf[:, -1]
        chosen = np.minimum((cdf < u[:, None]).sum(1), flat.shape[1] - 1)
        n_shock = shock_on.shape[1]
        g_sel = chosen // n_shock
        j_sel = chosen % n_shock
        # one Kalman update with the selected h
        gi, gs = g_idx[index, g_sel], g_sgn[index, g_sel]
        sj = shock_on[index, j_sel]
        Sh = S0 + gs[:, None] * cov[index, :, gi] - sj[:, None] * cov[:, :, _S]
        s_sel = s[index, g_sel, j_sel]
        innovation = total - pred[index, g_sel, j_sel]
        gain = Sh / s_sel[:, None]
        self.mean = mean + gain * innovation[:, None]
        self.cov = cov - gain[:, :, None] * Sh[:, None, :]
        if grant is not None:
            c_sel = modes_c[index, g_sel]
            k_sel = modes_k[index, g_sel]
            pa = _P[grant]
            coords = np.stack([c_sel, np.full(N, pa)], axis=1)
            ones = np.ones((N, 2))
            before = log_mode[index, g_sel] - log_channel[index, c_sel]
            lp_non = self._truncate(coords, -ones, -1.0, k_sel == 0)
            lp_sat = self._truncate(coords, ones, 1.0, k_sel == 1)
            after = np.where(k_sel == 0, lp_non, np.where(k_sel == 1, lp_sat, 0.0))
            self.logw = self.logw + np.where(k_sel == 2, 0.0, after - before)
            self.drives[:, grant] = c_sel
            mean, cov = self.mean, self.cov
            nonsat = k_sel == 0
            if nonsat.any():
                rows, cc = index[nonsat], c_sel[nonsat]
                mean[rows, cc] += mean[rows, pa]
                cov[rows, cc, :] += cov[rows, pa, :]
                cov[rows, :, cc] += cov[rows, :, pa]
            sat = k_sel == 1
            if sat.any():
                rows, cc = index[sat], c_sel[sat]
                mean[rows, cc] = 1.0
                cov[rows, cc, :] = 0.0
                cov[rows, :, cc] = 0.0
        mean, cov = self.mean, self.cov
        mean[:, _X] -= mean[:, _R]
        cov[:, _X, :] -= cov[:, _R, :]
        cov[:, :, _X] -= cov[:, :, _R]
        shocked = j_sel > 0
        if shocked.any() and not mixture:
            rows, jj = index[shocked], j_sel[shocked] - 1
            on = active[rows, jj]
            rows, jj = rows[on], jj[on]
            mean[rows, jj] -= mean[rows, _S]
            cov[rows, jj, :] -= cov[rows, _S, :]
            cov[rows, :, jj] -= cov[rows, :, _S]
        self.mean, self.cov = mean, 0.5 * (cov + np.transpose(cov, (0, 2, 1)))
        if shocked.any() and mixture:
            self._mix_shock(index[shocked], minimum)
        self._project_physical()
        self._observe_minimum(minimum)
        increment = _logsumexp(self.logw) - _logsumexp(start)
        self.evidence += increment
        self._normalize()
        self._maybe_resample()
        if self._replaying:
            self._replay_falsified |= bool(cfg.falsification_nats and increment < -cfg.falsification_nats)
            return
        self.memory.append(("tick", None if grant is None else int(grant), float(total), float(minimum)))
        if cfg.falsification_nats and increment < -cfg.falsification_nats:
            if self._life_rebuilds < 2:
                self._life_rebuilds += 1
                self._rebuild()
            else:
                self.falsified = True

    def _rebuild(self) -> None:
        """Re-interpret remembered experience from the prior. Keep the attempt
        with the highest evidence among those its own replay did not falsify,
        else the highest evidence overall and mark the self-model falsified."""

        self.rebuilds += 1
        best = None
        for _ in range(self.config.rebuild_attempts):
            trial = SelfPosterior(self.config, np.random.default_rng(int(self.rng.integers(2 ** 63))))
            trial._replaying = True
            trial._replay_falsified = False
            for kind, value, total, minimum in self.memory:
                if kind == "birth":
                    trial.birth(total, minimum, retain=value)
                else:
                    trial.observe(value, total, minimum)
            key = (not trial._replay_falsified, trial.evidence)
            if best is None or key > best[0]:
                best = (key, trial)
        trial = best[1]
        for name in ("active", "drives", "mean", "cov", "logw", "evidence", "collapses"):
            setattr(self, name, getattr(trial, name))
        self.falsified = not best[0][0]

    # ----------------------------------------------------- particle upkeep
    def _normalize(self) -> None:
        total = _logsumexp(self.logw)
        if np.isfinite(total):
            self.logw = self.logw - total

    def weights(self) -> np.ndarray:
        lw = self.logw
        finite = np.isfinite(lw)
        if not finite.any():
            return np.full(self.n, 1.0 / self.n)
        w = np.where(finite, np.exp(lw - lw[finite].max()), 0.0)
        return w / w.sum()

    def _resample_index(self) -> np.ndarray:
        u = (self.rng.uniform() + np.arange(self.n)) / self.n
        return np.minimum(np.searchsorted(np.cumsum(self.weights()), u), self.n - 1)

    def _take(self, index: np.ndarray) -> None:
        self.active = self.active[index].copy()
        self.drives = self.drives[index].copy()
        self.mean = self.mean[index].copy()
        self.cov = self.cov[index].copy()

    def _maybe_resample(self, force: bool = False) -> None:
        if not np.isfinite(self.logw).any():
            self.collapses += 1
            self.logw = np.full(self.n, -np.log(self.n))
            return
        w = self.weights()
        if force or 1.0 / float((w ** 2).sum()) < self.n * self.config.ess_fraction:
            self._take(self._resample_index())
            self.logw = np.full(self.n, -np.log(self.n))

    # ---------------------------------------------------------- readouts
    def _imagined_drives(self, i: int) -> np.ndarray:
        drives = self.drives[i].copy()
        lazy = drives < 0
        drives[lazy] = self.rng.integers(0, CHANNELS, int(lazy.sum()))
        return drives

    def structure_report(self, draws: int = 4) -> StructureReport:
        """Posterior mode of the full actuator partition and its probability.
        Untried actuators are imagined from the prior, so confidence includes
        what the organism has not yet tested."""

        w = self.weights()
        mass: dict[tuple, float] = {}
        for i in range(self.n):
            for _ in range(draws):
                drives = self._imagined_drives(i)
                groups: dict[int, list[int]] = {}
                for a in range(ACTUATORS):
                    if self.active[i, drives[a]]:
                        groups.setdefault(int(drives[a]), []).append(a)
                key = tuple(sorted(tuple(g) for g in groups.values()))
                mass[key] = mass.get(key, 0.0) + w[i] / draws
        best = max(mass, key=mass.get)
        counts = np.bincount(self.active.sum(1), weights=w, minlength=CHANNELS + 1)[1:]
        confidence = 0.0 if self.falsified else min(1.0, max(0.0, float(mass[best])))
        return StructureReport(
            partition=best,
            confidence=confidence,
            resource_count=int(np.argmax(counts)) + 1,
            count_posterior=tuple(float(v) for v in counts),
            tried=tuple(sorted(self.tried)),
            falsified=self.falsified,
        )

    def same_resource_probability(self, a: int, b: int) -> float:
        """Posterior probability that two tried actuators restore one resource."""

        w = self.weights()
        da, db = self.drives[:, a], self.drives[:, b]
        index = np.arange(self.n)
        same = (da >= 0) & (da == db) & self.active[index, np.maximum(da, 0)]
        return float((w * same).sum())

    # ------------------------------------------------------------ actions
    def _act_on(self, active, drives, x, r, p) -> int:
        time_left = np.where(active, np.maximum(x, 0.0) / np.maximum(r, 1e-6), np.inf)
        for c in np.argsort(time_left):
            if not active[c]:
                break
            candidates = np.flatnonzero(drives == c)
            if len(candidates):
                return int(candidates[np.argmax(p[candidates])])
        return int(self.rng.integers(0, ACTUATORS))

    def sample_action(self) -> int:
        """Posterior sampling: draw one self consistent with the evidence --
        structure, parameters, current levels, and imagined effects of untried
        actuators -- and restore whichever resource that self would lose first,
        with the largest portion it believes available."""

        i = int(self.rng.choice(self.n, p=self.weights()))
        cov = self.cov[i] + 1e-12 * np.eye(DIM)
        values, vectors = np.linalg.eigh(cov)
        theta = self.mean[i] + vectors @ (np.sqrt(np.maximum(values, 0.0)) * self.rng.standard_normal(DIM))
        return self._act_on(self.active[i], self._imagined_drives(i), theta[_X], theta[_R], theta[_P])

    def certainty_action(self) -> int:
        """Uncertainty lesion: act on the single most probable self (highest
        weight particle, mean parameters). Untried actuators are not imagined;
        one is chosen uniformly only when no known actuator restores the most
        urgent believed resource."""

        i = int(np.argmax(self.weights()))
        active, drives = self.active[i], self.drives[i]
        x, r, p = self.mean[i, _X], self.mean[i, _R], self.mean[i, _P]
        untried = np.flatnonzero(drives < 0)
        time_left = np.where(active, np.maximum(x, 0.0) / np.maximum(r, 1e-6), np.inf)
        for c in np.argsort(time_left):
            if not active[c]:
                break
            candidates = np.flatnonzero(drives == c)
            if len(candidates):
                return int(candidates[np.argmax(p[candidates])])
            if len(untried):
                return int(self.rng.choice(untried))
        if len(untried):
            return int(self.rng.choice(untried))
        return int(self.rng.integers(0, ACTUATORS))
