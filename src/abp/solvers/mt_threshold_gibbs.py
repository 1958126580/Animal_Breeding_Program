"""Gibbs sampler for a multi-trait model with one ordered categorical trait and one
or more continuous traits (method id ``threshold.gibbs_multitrait``).

Model (records ``i = 1..n``, traits ``j = 1..t``, trait ``c`` categorical):

    l_i = (I_t (x) x_i') beta + (I_t (x) z_i') u + e_i,    e_i ~ N(0, R0),
    u ~ N(0, K (x) G0),     y_ic = k  <=>  tau_{k-1} < l_ic <= tau_k,

where ``l_ij = y_ij`` for a continuous trait and ``l_ic`` is the liability of the
categorical trait; ``R0[c, c] = 1`` (the liability scale), ``tau_1 = 0`` (the model has
an intercept per trait).  ``x_i`` and ``z_i`` are shared by the traits (one fixed
design ``X`` and one animal incidence ``Z``); every trait has its own coefficients.
Unknowns are ordered animal-major: ``beta[k t + j]``, ``u[a t + j]``.

Priors: flat on ``beta`` and the free thresholds; for ``G0`` an inverse Wishart
``IW(nu, nu G_prior)`` (for ``t = 1`` the scaled inverse chi-square prior of the
single-trait sampler), by default ``nu = -(t + 1)``, ``G_prior = 0``: **flat** over the
positive definite matrices.  With one categorical record per animal the flat prior
gives an improper posterior for the categorical trait's genetic variance (the
likelihood tends to a positive constant as that variance grows: the probability
that the relatives' liabilities fall in the observed orthants), so the chains
drift and fail the convergence checks; a proper prior (``nu > t - 1``) is then
needed.
for ``R0`` the parameterisation of Korsgaard, Lund, Sorensen, Gianola, Madsen & Jensen
(2003, Genet Sel Evol 35:159) with the categorical trait first,

    e_ic ~ N(0, 1),   e_i,o | e_ic ~ N(b e_ic, S),   R0 = [[1, b'], [b, S + b b']],

flat on ``b`` and ``S`` (``o`` = the continuous traits).

One iteration (each step leaves the joint posterior invariant):

1. missing continuous values drawn from their conditional normal given the record's
   other values (data augmentation: the equations always see complete records);
2. thresholds and categorical liabilities jointly (Cowles 1996) given everything
   else: the ordinal likelihood uses the conditional distribution of ``l_ic`` given
   the record's other traits, ``N(m_i, s^2)``; then ``l_ic`` from ``N(m_i, s^2)``
   truncated to its category (unrestricted when ``y_ic`` is missing);
3. ``theta = (beta, u)`` as one block from ``N(C^{-1} W'(I (x) R0^{-1}) l, C^{-1})``,
   ``C = W'(I (x) R0^{-1}) W + blockdiag(0, K^{-1} (x) G0^{-1})``, by perturbation
   ``theta = C^{-1}(W'(I (x) R0^{-1})(l + e*) + [0; (F (x) L) z])`` with
   ``e*_i ~ N(0, R0)``, ``F F' = K^{-1}``, ``L L' = G0^{-1}``;
3b. parameter expansion (Liu & Sabatti 2000) for the categorical trait: the move
   ``u_.c -> g u_.c``, ``G0 -> D G0 D`` (``D = diag(1, .., g, .., 1)``) with
   ``g`` drawn from ``exp(Q(g)) g^(t+1) p(D G0 D)`` on the log scale (Jacobian
   ``g^(q + t + 1)``, ``|D G0 D|^(-q/2)`` gives ``g^(-q)``, Haar measure ``dg / g``;
   ``Q`` the Gaussian log-likelihood of the liabilities, quadratic in ``g``; the prior
   gives ``g^(-(nu + t + 1)) exp(-tr(nu G_prior D^-1 G0^-1 D^-1) / 2)``) by slice sampling;
4. ``G0 | u ~ IW(U' K^{-1} U + nu G_prior, q + nu)`` (``U`` = ``u`` as ``q x t``);
5. ``S | e ~ IW(E_o'E_o - E_o'e_c e_c'E_o / e_c'e_c, n - t - 1)`` and
   ``b | S, e ~ N(E_o'e_c / e_c'e_c, S / e_c'e_c)`` (``b`` integrated out for ``S``,
   so the pair is an exact draw from its conditional).

Chains, diagnostics and the stopping rule are those of :mod:`threshold_gibbs`
(R-hat, bulk/tail ESS for every scalar and every breeding value of every trait);
unconverged results are withheld by the workflow (``ABP-E405``).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr
from scipy.special import ndtr, ndtri
from scipy.stats import invwishart

from ..errors import ABPError
from . import mcmc_diagnostics as dg
from .threshold import _log_p
from .threshold_gibbs import EBV_STORE_LIMIT, _align, _slice_sample, precision_root, rtruncnorm


@dataclass
class MTThresholdGibbsConfig:
    chains: int = 4
    iterations: int = 6000
    burn_in: int = 1000
    thin: int = 5
    seed: int = 20260925
    rhat_max: float = 1.01
    ess_min: float = 400.0
    max_iterations: int = 30000
    start_G0: np.ndarray | None = None      #: default: 0.2 (liability) / 0.3 var(y) on the diagonal
    start_R0: np.ndarray | None = None      #: default: 1 (liability) / 0.7 var(y) on the diagonal
    fix_covariances: bool = False           #: keep G0, R0 at the start values (known)
    scale_move: bool = True                 #: parameter expansion for the categorical trait
    prior_nu: float | None = None           #: IW degrees of freedom (default -(t + 1): flat)
    prior_G0: np.ndarray | None = None      #: prior guess of G0 (IW scale nu G_prior)


@dataclass
class MTThresholdGibbsResult:
    ebv: np.ndarray                 # q x t posterior means
    pev: np.ndarray                 # q x t posterior variances
    fixed_mean: np.ndarray          # p x t
    thresholds_mean: np.ndarray     # tau_1 .. tau_{K-1}
    categories: np.ndarray
    G0: dict                        # "mean", "sd", "q025", "q975" (t x t lists)
    R0: dict
    derived: dict                   # h2_<j>, rG_<j>_<k> summaries
    summaries: dict
    ebv_diagnostics: dict
    converged: bool
    iterations: int
    draws_per_chain: int
    seeds: list
    acceptance: float | None
    wall_seconds: float
    n_equations: int
    traces: dict = field(default_factory=dict)


def _check_pd(S: np.ndarray, what: str) -> np.ndarray:
    S = np.asarray(S, dtype=np.float64)
    if S.shape[0] != S.shape[1] or not np.allclose(S, S.T) or np.linalg.eigvalsh(S)[0] <= 0:
        raise ABPError("COVARIANCE_NOT_PD", f"{what} must be symmetric positive definite")
    return S


class MTProblem:
    """Data and fixed structure shared by the chains."""

    def __init__(self, Y: np.ndarray, cat: int, X, Z, k_inv):
        Y = np.asarray(Y, dtype=np.float64)
        self.n, self.t = Y.shape
        if self.t < 2:
            raise ABPError("SPEC_INVALID", "the multi-trait threshold model needs >= 2 traits")
        if not 0 <= cat < self.t:
            raise ABPError("SPEC_INVALID", "categorical trait index out of range")
        self.c = cat
        self.o = np.array([j for j in range(self.t) if j != cat])
        obs = ~np.isnan(Y)
        if np.any(~obs.any(axis=1)):
            raise ABPError("SCHEMA_TYPE", "every record needs at least one observed trait")
        yc = Y[obs[:, cat], cat]
        if not np.all(yc == np.round(yc)):
            raise ABPError("SCHEMA_TYPE", "the categorical trait needs integer category codes")
        self.cats = np.unique(yc)
        self.K = self.cats.size
        if self.K < 2:
            raise ABPError("MODEL_NOT_IDENTIFIABLE", "all records are in one category")
        self.Y = Y
        self.obs = obs
        self.cat_obs = obs[:, cat]
        self.yk = np.zeros(self.n, dtype=np.int64)
        self.yk[self.cat_obs] = np.searchsorted(self.cats, Y[self.cat_obs, cat])
        self.X = sp.csr_matrix(X)
        self.Z = sp.csr_matrix(Z)
        self.p = self.X.shape[1]
        self.q = self.Z.shape[1]
        if self.q <= 2 * self.t + 1:
            raise ABPError("MODEL_NOT_IDENTIFIABLE", "too few animals for the flat G0 prior")
        self.k_inv = sp.csr_matrix(k_inv) if sp.issparse(k_inv) else np.asarray(k_inv)
        self.root = precision_root(k_inv)
        B = sp.hstack([self.X, self.Z], format="csr")          # n x (p + q), one trait
        self.B = B
        self.Bt = B.T.tocsr()
        self.BtB = (self.Bt @ B).tocsr()
        Kpad = sp.block_diag([sp.csr_matrix((self.p, self.p)), sp.csr_matrix(self.k_inv)],
                             format="csr")
        self.Kpad = Kpad
        self.n_eq = (self.p + self.q) * self.t
        # fixed pattern kron(pattern(B'B + K), ones(t, t)): the numeric refactorization
        # needs the same pattern even when entries of R0^{-1} or G0^{-1} are exactly 0
        pat1 = (abs(self.BtB) + abs(Kpad)).tocsr()
        pat1.data[:] = 1.0
        pat = sp.kron(pat1, np.ones((self.t, self.t)), format="csr")
        pat.sum_duplicates()
        pat.sort_indices()
        self.pat = pat
        # missing patterns of the continuous traits (records with >= 1 missing)
        miss_o = ~obs[:, self.o]
        self.patterns = {}
        for i in np.flatnonzero(miss_o.any(axis=1)):
            key = tuple(np.flatnonzero(miss_o[i]))
            self.patterns.setdefault(key, []).append(i)
        self.patterns = {k: np.array(v) for k, v in self.patterns.items()}
        self.free = np.arange(2, self.K)                    # indices into tau (0..K); tau_1 = 0

    def coefficient(self, R0inv: np.ndarray, G0inv: np.ndarray) -> sp.csr_matrix:
        """``C`` in animal-major order (index ``level * t + trait``)."""
        data = (_align(sp.kron(self.BtB, R0inv, format="csr"), self.pat)
                + _align(sp.kron(self.Kpad, G0inv, format="csr"), self.pat))
        return sp.csr_matrix((data, self.pat.indices, self.pat.indptr), shape=self.pat.shape)

    def eta(self, theta: np.ndarray) -> np.ndarray:
        """Location part of the liabilities, ``n x t``."""
        return self.B @ theta.reshape(-1, self.t)

    def rhs(self, L: np.ndarray, R0inv: np.ndarray) -> np.ndarray:
        """``W'(I (x) R0^{-1}) vec(L)`` for ``L`` of shape ``n x t``."""
        return (self.Bt @ (L @ R0inv)).ravel()


def draw_location(P: MTProblem, fac, L: np.ndarray, R0: np.ndarray, G0: np.ndarray,
                  rng) -> np.ndarray:
    """One exact draw of ``theta ~ N(C^{-1} W'(I (x) R0^{-1}) vec L, C^{-1})`` (``fac``
    factorises ``C`` at ``R0``, ``G0``)."""
    t = P.t
    R0inv = np.linalg.inv(R0)
    E = rng.standard_normal((P.n, t)) @ np.linalg.cholesky(R0).T
    rhs = P.rhs(L + E, R0inv)
    Lg = np.linalg.cholesky(np.linalg.inv(G0))                # L L' = G0^{-1}
    Zq = rng.standard_normal((P.q, t)) @ Lg.T                  # rows: L z_a
    pri = np.column_stack([P.root(Zq[:, j]) for j in range(t)])   # (F (x) L) z
    rhs[P.p * t:] += pri.ravel()
    return fac.solve(rhs)


def cond_normal(R0: np.ndarray, j: int):
    """Regression of trait ``j`` on the others: ``(coefficients, residual variance)``."""
    o = [k for k in range(R0.shape[0]) if k != j]
    w = np.linalg.solve(R0[np.ix_(o, o)], R0[o, j])
    return o, w, float(R0[j, j] - R0[j, o] @ w)


def draw_R0(E: np.ndarray, c: int, rng) -> np.ndarray:
    """Exact draw of ``R0 | residuals`` under the flat prior on ``(b, S)`` with
    ``R0[c, c] = 1`` (module notes, step 5); ``E`` is ``n x t``."""
    n, t = E.shape
    o = [k for k in range(t) if k != c]
    ec = E[:, c]
    Eo = E[:, o]
    cc = float(ec @ ec)
    bhat = Eo.T @ ec / cc
    Sres = Eo.T @ Eo - np.outer(bhat, bhat) * cc
    m = t - 1
    S = np.atleast_2d(invwishart.rvs(df=n - m - 2, scale=Sres, random_state=rng))
    b = bhat + np.linalg.cholesky(S / cc) @ rng.standard_normal(m)
    R0 = np.empty((t, t))
    R0[c, c] = 1.0
    R0[c, o] = R0[o, c] = b
    R0[np.ix_(o, o)] = S + np.outer(b, b)
    return R0


def draw_G0(U: np.ndarray, k_inv, rng, nu: float | None = None,
            G_prior: np.ndarray | None = None) -> np.ndarray:
    """``G0 | u ~ IW(U'K^{-1}U + nu G_prior, q + nu)``; the default ``nu = -(t + 1)``,
    ``G_prior = 0`` is the flat prior on the PD matrices."""
    q, t = U.shape
    nu = -(t + 1.0) if nu is None else float(nu)
    S = U.T @ (k_inv @ U)
    if G_prior is not None:
        S = S + nu * np.asarray(G_prior, dtype=np.float64)
    return np.atleast_2d(invwishart.rvs(df=q + nu, scale=0.5 * (S + S.T), random_state=rng))


class _Chain:
    def __init__(self, P: MTProblem, cfg: MTThresholdGibbsConfig, rng, G0, R0, beta0):
        from .cholesky import SparseLDL
        self.P, self.cfg, self.rng = P, cfg, rng
        self.G0, self.R0 = G0.copy(), R0.copy()
        t = P.t
        self.theta = np.zeros(P.n_eq)
        self.theta[:P.p * t] = beta0.ravel()
        self.tau = np.concatenate([[-np.inf], P.tau0, [np.inf]])
        # factor built on the full pattern (a surrogate without exact zeros), then the
        # numeric factor of the actual C (which has zeros while R0, G0 are diagonal)
        Ri, Gi = np.linalg.inv(R0), np.linalg.inv(G0)
        eps_r, eps_g = 1e-3 * np.trace(Ri) / t, 1e-3 * np.trace(Gi) / t
        self.fac = SparseLDL(P.coefficient(Ri + eps_r * (np.eye(t) + np.ones((t, t))),
                                           Gi + eps_g * (np.eye(t) + np.ones((t, t)))))
        self.fac.refactor(P.coefficient(Ri, Gi))
        self.sd = 0.1
        self.n_prop = self.n_acc = 0
        eta = P.eta(self.theta)
        L = np.where(P.obs, P.Y, eta)
        L[:, P.c] = eta[:, P.c]
        self.L = L
        self._draw_categorical(eta, tuning=False, it=1, mh=False)

    # -- step 1: missing continuous values -------------------------------------
    def _impute(self, eta):
        P, R0 = self.P, self.R0
        res = self.L - eta
        for key, rows in P.patterns.items():
            mj = P.o[list(key)]                                  # missing trait indices
            kj = np.array([k for k in range(P.t) if k not in set(mj)])
            A = R0[np.ix_(mj, kj)] @ np.linalg.inv(R0[np.ix_(kj, kj)])
            V = R0[np.ix_(mj, mj)] - A @ R0[np.ix_(kj, mj)]
            mean = res[np.ix_(rows, kj)] @ A.T
            draw = mean + self.rng.standard_normal((rows.size, mj.size)) @ np.linalg.cholesky(V).T
            self.L[np.ix_(rows, mj)] = eta[np.ix_(rows, mj)] + draw

    # -- step 2: thresholds (Cowles) and categorical liabilities -----------------
    def _draw_categorical(self, eta, tuning: bool, it: int, mh: bool = True):
        P, c = self.P, self.P.c
        o, w, v = cond_normal(self.R0, c)
        s = math.sqrt(v)
        m = eta[:, c] + (self.L[:, o] - eta[:, o]) @ w
        ob = P.cat_obs
        if mh and P.free.size:
            old = self.tau
            new = old.copy()
            sd = self.sd
            for j in P.free:
                lo, hi = new[j - 1], old[j + 1]
                a, b = (lo - old[j]) / sd, (hi - old[j]) / sd
                pa, pb = ndtr(a), ndtr(b)
                new[j] = old[j] + sd * ndtri(pa + self.rng.random() * (pb - pa))
            log_q = 0.0
            for j in P.free:
                log_q += math.log(max(ndtr((old[j + 1] - old[j]) / sd)
                                      - ndtr((new[j - 1] - old[j]) / sd), 1e-300))
                log_q -= math.log(max(ndtr((new[j + 1] - new[j]) / sd)
                                      - ndtr((old[j - 1] - new[j]) / sd), 1e-300))
            mo, yk = m[ob], P.yk[ob]
            lp_new, _ = _log_p((new[yk] - mo) / s, (new[yk + 1] - mo) / s)
            lp_old, _ = _log_p((old[yk] - mo) / s, (old[yk + 1] - mo) / s)
            log_r = float(lp_new.sum() - lp_old.sum()) + log_q
            self.n_prop += 1
            if math.log(max(self.rng.random(), 1e-300)) < log_r:
                self.tau = new
                self.n_acc += 1
            if tuning and it % 50 == 0 and self.n_prop:
                rate = self.n_acc / self.n_prop
                self.sd *= 0.7 if rate < 0.2 else (1.4 if rate > 0.5 else 1.0)
                self.n_prop = self.n_acc = 0
        lc = self.L[:, c]
        z = rtruncnorm(self.rng, np.zeros(int(ob.sum())), (self.tau[P.yk[ob]] - m[ob]) / s,
                       (self.tau[P.yk[ob] + 1] - m[ob]) / s)
        lc[ob] = m[ob] + s * z
        lc[~ob] = m[~ob] + s * self.rng.standard_normal(int((~ob).sum()))

    # -- step 3b: parameter expansion for the categorical trait ------------------
    def _scale_move(self):
        P, c, t = self.P, self.P.c, self.P.t
        U = self.theta[P.p * t:].reshape(P.q, t)
        v = P.Z @ U[:, c]
        vv = float(v @ v)
        if vv <= 0.0:
            return
        Ri = np.linalg.inv(self.R0)
        eta = P.eta(self.theta)
        r = self.L - eta
        r[:, c] += v                                           # residual without g u_.c
        a2 = vv * Ri[c, c]
        a1 = float(v @ (r @ Ri[:, c]))
        nu = self.P.nu
        Psi = self.P.psi
        if Psi is not None:
            Gi = np.linalg.inv(self.G0)
            # tr(Psi D^-1 Gi D^-1) = p0 + p1 / g + p2 / g^2
            p2 = Psi[c, c] * Gi[c, c]
            p1 = 2.0 * float(Psi[c, :] @ Gi[:, c] - Psi[c, c] * Gi[c, c])

        def logf(x):
            g = math.exp(x)
            out = -0.5 * (g * g * a2 - 2.0 * g * a1) + (t + 1) * x - (nu + t + 1) * x
            if Psi is not None:
                out -= 0.5 * (p1 / g + p2 / (g * g))
            return out
        g = math.exp(_slice_sample(logf, 0.0, 0.5, self.rng))
        th = self.theta[P.p * t:].reshape(P.q, t)
        th[:, c] *= g
        D = np.ones(t)
        D[c] = g
        self.G0 = self.G0 * np.outer(D, D)

    def iterate(self, tuning: bool, it: int):
        P, t = self.P, self.P.t
        eta = P.eta(self.theta)
        self._impute(eta)
        self._draw_categorical(eta, tuning, it)
        self.theta = draw_location(P, self.fac, self.L, self.R0, self.G0, self.rng)
        if self.cfg.fix_covariances:
            return
        if self.cfg.scale_move:
            self._scale_move()
        U = self.theta[P.p * t:].reshape(P.q, t)
        self.G0 = draw_G0(U, P.k_inv, self.rng, P.nu, P.g_prior)
        self.R0 = draw_R0(self.L - P.eta(self.theta), P.c, self.rng)
        self.fac.refactor(P.coefficient(np.linalg.inv(self.R0), np.linalg.inv(self.G0)))


def _summ(A: np.ndarray) -> dict:
    return {"mean": A.mean(axis=0).tolist(), "sd": A.std(axis=0, ddof=1).tolist(),
            "q025": np.quantile(A, 0.025, axis=0).tolist(),
            "q975": np.quantile(A, 0.975, axis=0).tolist()}


def mt_threshold_gibbs(Y, cat: int, X, Z, k_inv, cfg: MTThresholdGibbsConfig,
                       k_diag: np.ndarray | None = None) -> MTThresholdGibbsResult:
    """Run the sampler.  ``Y``: ``n x t`` (NaN = missing; column ``cat`` holds integer
    categories), ``X``: ``n x p`` fixed design shared by the traits (must contain an
    intercept), ``Z``: ``n x q`` animal incidence, ``k_inv``: ``K^{-1}`` (``q x q``)."""
    t0 = time.perf_counter()
    if cfg.chains < 2:
        raise ABPError("SPEC_INVALID", "at least 2 chains are required for R-hat (4 recommended)")
    if not (0 <= cfg.burn_in < cfg.iterations) or cfg.thin < 1:
        raise ABPError("SPEC_INVALID", "need 0 <= burn_in < iterations and thin >= 1")
    P = MTProblem(Y, cat, X, Z, k_inv)
    t, c = P.t, P.c
    P.nu = -(t + 1.0) if cfg.prior_nu is None else float(cfg.prior_nu)
    P.g_prior = None
    P.psi = None
    if cfg.prior_G0 is not None:
        if P.nu <= t - 1:
            raise ABPError("SPEC_INVALID", f"a proper inverse Wishart prior needs nu > {t - 1}")
        P.g_prior = _check_pd(cfg.prior_G0, "prior G0")
        if P.g_prior.shape != (t, t):
            raise ABPError("SPEC_INVALID", f"prior G0 must be {t} x {t}")
        P.psi = P.nu * P.g_prior
    elif cfg.prior_nu is not None:
        raise ABPError("SPEC_INVALID", "prior_nu needs prior_G0")
    Yo = P.Y
    vy = np.array([np.nanvar(Yo[:, j]) if j != c else 1.0 for j in range(t)])
    G0s = _check_pd(cfg.start_G0, "start G0") if cfg.start_G0 is not None else \
        np.diag(np.where(np.arange(t) == c, 0.2, 0.3 * vy))
    R0s = _check_pd(cfg.start_R0, "start R0") if cfg.start_R0 is not None else \
        np.diag(np.where(np.arange(t) == c, 1.0, 0.7 * vy))
    if abs(R0s[c, c] - 1.0) > 1e-12:
        raise ABPError("SPEC_INVALID", "the residual variance of the categorical trait must be 1")
    # starting thresholds from the category proportions, tau_1 = 0 (intercept absorbs it)
    cum = np.cumsum(np.bincount(P.yk[P.cat_obs], minlength=P.K))[:-1] / P.cat_obs.sum()
    raw = ndtri(np.clip(cum, 1e-6, 1 - 1e-6))
    P.tau0 = raw - raw[0]
    # starting fixed effects: least squares per trait on the observed records
    beta0 = np.zeros((P.p, t))
    Xd = P.X
    for j in range(t):
        rows = P.obs[:, j]
        yj = Yo[rows, j] if j != c else np.full(int(rows.sum()), -raw[0])
        sol = lsqr(Xd[rows], yj, atol=1e-10, btol=1e-10)[0]
        beta0[:, j] = sol
    ss = np.random.SeedSequence(cfg.seed)
    child = ss.spawn(cfg.chains)
    chains = []
    for ch in child:
        rng = np.random.default_rng(ch)
        if cfg.fix_covariances:
            G0, R0 = G0s, R0s
        else:
            f = np.exp(rng.uniform(-0.5, 0.5, t))
            G0 = G0s * np.outer(f, f)
            R0 = R0s.copy()
        chains.append(_Chain(P, cfg, rng, G0, R0, beta0))
    tri = [(i, j) for i in range(t) for j in range(i, t)]
    scal = [f"G0_{i}_{j}" for i, j in tri] + \
        [f"R0_{i}_{j}" for i, j in tri if not (i == c and j == c)] + \
        [f"rG_{i}_{j}" for i, j in tri if i < j] + [f"h2_{j}" for j in range(t)] + \
        [f"tau_{j}" for j in P.free]
    store = {k: [[] for _ in chains] for k in scal}
    Gs, Rs = [], []
    keep_ebv = P.q * t * (cfg.max_iterations // cfg.thin) * cfg.chains <= EBV_STORE_LIMIT
    ebv_draws = [[] for _ in chains]
    s_theta = np.zeros(P.n_eq)
    q_theta = np.zeros(P.n_eq)
    s_tau = np.zeros(P.K - 1)
    n_saved = 0
    it = 0
    target = cfg.iterations
    while True:
        while it < target:
            it += 1
            tuning = it <= cfg.burn_in
            for k, chn in enumerate(chains):
                if it == cfg.burn_in + 1:
                    chn.n_prop = chn.n_acc = 0
                chn.iterate(tuning, it)
                if it > cfg.burn_in and (it - cfg.burn_in) % cfg.thin == 0:
                    G, R = chn.G0, chn.R0
                    vals = {f"G0_{i}_{j}": G[i, j] for i, j in tri}
                    vals.update({f"R0_{i}_{j}": R[i, j] for i, j in tri})
                    vals.update({f"rG_{i}_{j}": G[i, j] / math.sqrt(G[i, i] * G[j, j])
                                 for i, j in tri if i < j})
                    vals.update({f"h2_{j}": G[j, j] / (G[j, j] + R[j, j]) for j in range(t)})
                    vals.update({f"tau_{j}": chn.tau[j] for j in P.free})
                    for key in scal:
                        store[key][k].append(float(vals[key]))
                    Gs.append(G.copy())
                    Rs.append(R.copy())
                    if keep_ebv:
                        ebv_draws[k].append(chn.theta[P.p * t:].astype(np.float32))
                    s_theta += chn.theta
                    q_theta += chn.theta * chn.theta
                    s_tau += chn.tau[1:P.K]
                    if k == 0:
                        n_saved += 1
        summaries = {key: dg.summarize(np.array(store[key])) for key in scal
                     if np.ptp(np.array(store[key])) > 0}
        ediag = {"not_computed": f"per-EBV diagnostics skipped (> {EBV_STORE_LIMIT:.0e} "
                                 "stored values)", "n_animals": int(P.q)} if not keep_ebv else {}
        if keep_ebv and n_saved >= 4:
            E = np.array(ebv_draws, dtype=np.float64)            # chains x draws x (q t)
            rh = np.array([dg.rhat(E[:, :, i]) for i in range(E.shape[2])])
            eb = np.array([dg.bulk_ess(E[:, :, i]) for i in range(E.shape[2])])
            ediag = {"max_rhat": float(np.nanmax(rh)), "min_ess_bulk": float(np.nanmin(eb)),
                     "n_animals": int(P.q), "n_traits": int(t)}
        ok = all(dg.passes(s, cfg.rhat_max, cfg.ess_min) for s in summaries.values()) and (
            "max_rhat" not in ediag or (ediag["max_rhat"] < cfg.rhat_max
                                        and ediag["min_ess_bulk"] >= cfg.ess_min))
        if ok or target >= cfg.max_iterations:
            break
        target = min(cfg.max_iterations, target * 2)
    total = n_saved * len(chains)
    mean = s_theta / total
    var = np.maximum(q_theta / total - mean ** 2, 0.0) * total / max(total - 1, 1)
    Ga, Ra = np.array(Gs), np.array(Rs)
    derived = {}
    for key in scal:
        if key.startswith(("rG_", "h2_")):
            v = np.array(store[key]).ravel()
            derived[key] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                            "q025": float(np.quantile(v, 0.025)),
                            "q975": float(np.quantile(v, 0.975))}
    acc_prop = sum(ch.n_prop for ch in chains)
    acc_n = sum(ch.n_acc for ch in chains)
    return MTThresholdGibbsResult(
        mean[P.p * t:].reshape(P.q, t), var[P.p * t:].reshape(P.q, t),
        mean[:P.p * t].reshape(P.p, t), s_tau / total, P.cats, _summ(Ga), _summ(Ra), derived,
        summaries, ediag, bool(ok), it, n_saved, [int(s.generate_state(1)[0]) for s in child],
        (acc_n / acc_prop) if acc_prop else None, time.perf_counter() - t0, P.n_eq,
        traces={key: np.array(store[key]) for key in scal})
