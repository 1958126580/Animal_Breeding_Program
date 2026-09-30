"""Gibbs sampler for a multi-trait model with one ordered categorical trait and one
or more continuous traits (method id ``threshold.gibbs_multitrait``).

Model (records ``r``, traits ``j = 1..t``, trait ``c`` categorical): the multi-trait
animal model of :mod:`abp.solvers.multitrait` on the liability scale,

    l_rj = x_rj' beta_j + u_{a(r) j} + e_rj,   e_r ~ N(0, R0),   u ~ N(0, K (x) G0),
    y_rc = k  <=>  tau_{k-1} < l_rc <= tau_k,

with ``l_rj = y_rj`` for a continuous trait, **its own fixed design per trait**
(``X_j`` on the records where trait ``j`` is observed), missing traits, ``R0[c, c] = 1``
(liability scale) and ``tau_1 = 0`` (every trait has an intercept).  Unknowns:
``theta = (beta_1 .. beta_t, u)`` with ``u`` animal-major (``a t + j``), as in
:mod:`abp.solvers.multitrait`.

Priors: flat on ``beta`` and the free thresholds; ``G0 ~ IW(nu, nu G_prior)`` (for
``t = 1`` the scaled inverse chi-square prior of the single-trait sampler), by default
``nu = -(t + 1)``, ``G_prior = 0``: **flat** over the positive definite matrices.  With
one categorical record per animal the flat prior gives an improper posterior for the
categorical trait's genetic variance (the likelihood tends to a positive constant as
that variance grows: the probability that the relatives' liabilities fall in the
observed orthants), so chains drift and fail the convergence checks; a proper prior is
then needed.  ``R0``: the parameterisation of Korsgaard, Lund, Sorensen, Gianola,
Madsen & Jensen (2003, Genet Sel Evol 35:159) with the categorical trait first,
``e_c ~ N(0, 1)``, ``e_o | e_c ~ N(b e_c, S)``, ``R0 = [[1, b'], [b, S + b b']]``, flat
on ``b`` and ``S``.

State: ``theta``, the categorical liabilities of the records where it is observed,
``tau``, the residuals of the missing traits ``e_m``, ``G0``, ``R0``.  One iteration
(each step leaves the joint posterior invariant; steps 1-2 condition on the observed
data only, with ``e_m`` integrated out, and ``e_m`` is drawn from its exact
conditional right after them - a partially collapsed Gibbs sampler):

1. thresholds and liabilities jointly (Cowles 1996): the ordinal likelihood uses the
   conditional distribution of ``l_rc`` given the record's **observed** continuous
   traits, ``N(m_r, s_r^2)``; then ``l_rc`` from it truncated to the category;
2. ``theta`` as one block from ``N(C^{-1} W'R^{-1} y, C^{-1})`` of the observed-data
   mixed-model equations (``R^{-1}`` block-diagonal: ``R0[o_r, o_r]^{-1}`` per record,
   ``C = W'R^{-1}W + blockdiag(0, K^{-1} (x) G0^{-1})``, ``y`` with the liabilities) by
   perturbation ``C^{-1}(W'R^{-1}(y + e*) + [0; (F (x) L) z])``, ``e*_r ~ N(0, R0[o_r, o_r])``,
   ``F F' = K^{-1}``, ``L L' = G0^{-1}``;
2b. parameter expansion (Liu & Sabatti 2000): ``u_.c -> g u_.c``, ``G0 -> D G0 D``
   (``D = diag(1, .., g, .., 1)``), ``g`` drawn from ``exp(Q(g)) g^(t+1) p(D G0 D)`` on
   the log scale by slice sampling (Jacobian ``g^(q + t + 1)``, ``|D G0 D|^(-q/2)``
   gives ``g^(-q)``, Haar measure ``dg / g``; ``Q`` the observed-data Gaussian
   log-likelihood, quadratic in ``g``; the prior gives
   ``g^(-(nu + t + 1)) exp(-tr(nu G_prior D^-1 G0^-1 D^-1) / 2)``);
2c. shear moves (the same generalised Gibbs construction on the additive group), for
   every ordered pair of traits ``i != j``: ``u_.i -> u_.i + h u_.j``,
   ``G0 -> S G0 S'`` with ``S = I + h e_i e_j'``.  ``det S = 1``, so the Jacobian is 1
   and ``|G0|`` and ``u'(K^{-1} (x) G0^{-1})u`` are unchanged; the observed-data
   log-likelihood and the prior term ``tr(Psi S^-T G0^-1 S^-1)`` are quadratic in
   ``h``, so ``h`` is drawn exactly from a normal distribution with precision
   ``w'R^{-1}w + (G0^{-1})_ii Psi_jj`` and linear term ``w'R^{-1}r + (Psi G0^{-1})_ji``
   (``w`` = ``u_.j`` at the observations of trait ``i``, ``r`` the residuals).  These
   moves let the genetic covariances mix: without them ``u`` and ``G0`` are strongly
   dependent and the correlation between a categorical and a continuous trait moves
   slowly;
3. ``e_m | e_o ~ N(R0_mo R0_oo^{-1} e_o, R0_mm - R0_mo R0_oo^{-1} R0_om)`` per record;
4. ``G0 | u ~ IW(U'K^{-1}U + nu G_prior, q + nu)``;
5. ``S | e ~ IW(E_o'E_o - E_o'e_c e_c'E_o / e_c'e_c, n - t - 1)``,
   ``b | S, e ~ N(E_o'e_c / e_c'e_c, S / e_c'e_c)`` from the complete residuals.

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
from .threshold_gibbs import EBV_STORE_LIMIT, _slice_sample, precision_root, rtruncnorm


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
    shear_moves: bool = True                #: parameter expansion for the genetic covariances
    prior_nu: float | None = None           #: IW degrees of freedom (default -(t + 1): flat)
    prior_G0: np.ndarray | None = None      #: prior guess of G0 (IW scale nu G_prior)


@dataclass
class MTThresholdGibbsResult:
    ebv: np.ndarray                 # q x t posterior means
    pev: np.ndarray                 # q x t posterior variances
    pev_blocks: np.ndarray          # q x t x t posterior covariances between traits
    fixed_mean: list                # per trait: posterior means of beta_j
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


def split_design(X, Y: np.ndarray) -> list:
    """A fixed design shared by the traits (rows = records) as per-trait designs
    (rows = records where the trait is observed)."""
    X = sp.csr_matrix(X)
    return [X[~np.isnan(Y[:, j])] for j in range(Y.shape[1])]


class MTProblem:
    """Data and fixed structure shared by the chains."""

    def __init__(self, Y: np.ndarray, cat: int, X_per_trait: list, animal_col: np.ndarray,
                 k_inv):
        Y = np.asarray(Y, dtype=np.float64)
        self.n, self.t = Y.shape
        t = self.t
        if t < 2:
            raise ABPError("SPEC_INVALID", "the multi-trait threshold model needs >= 2 traits")
        if not 0 <= cat < t:
            raise ABPError("SPEC_INVALID", "categorical trait index out of range")
        self.c = cat
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
        self.Y, self.obs = Y, obs
        self.cat_obs = obs[:, cat]
        self.yk = np.zeros(self.n, dtype=np.int64)
        self.yk[self.cat_obs] = np.searchsorted(self.cats, Y[self.cat_obs, cat])
        self.Xj = [sp.csr_matrix(X) for X in X_per_trait]
        if len(self.Xj) != t:
            raise ABPError("SPEC_INVALID", "one fixed design per trait is needed")
        for j, X in enumerate(self.Xj):
            if X.shape[0] != int(obs[:, j].sum()):
                raise ValueError(f"X for trait {j} has {X.shape[0]} rows, expected "
                                 f"{int(obs[:, j].sum())}")
        self.p_off = np.cumsum([0] + [X.shape[1] for X in self.Xj])
        self.P = int(self.p_off[-1])
        self.k_inv = sp.csr_matrix(k_inv) if sp.issparse(k_inv) else np.asarray(k_inv)
        self.q = self.k_inv.shape[0]
        if self.q <= 2 * t + 1:
            raise ABPError("MODEL_NOT_IDENTIFIABLE", "too few animals for the covariance priors")
        self.animal_col = np.asarray(animal_col, dtype=np.int64)
        self.root = precision_root(k_inv)
        self._check_fixed_effects()
        # observations, record-major: position of (record, trait) in the observation vector
        rec_idx, trait_idx = np.nonzero(obs)
        self.rec_idx, self.trait_idx = rec_idx, trait_idx
        self.n_obs = rec_idx.size
        self.pos = np.full((self.n, t), -1, dtype=np.int64)
        self.pos[rec_idx, trait_idx] = np.arange(self.n_obs)
        rows, cols, vals = [], [], []
        for j, X in enumerate(self.Xj):
            Xc = X.tocoo()
            recs_j = np.flatnonzero(obs[:, j])
            rows.append(self.pos[recs_j[Xc.row], j])
            cols.append(Xc.col + self.p_off[j])
            vals.append(Xc.data)
        Xs = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                           shape=(self.n_obs, self.P))
        Zs = sp.csr_matrix((np.ones(self.n_obs), (np.arange(self.n_obs),
                                                  self.animal_col[rec_idx] * t + trait_idx)),
                           shape=(self.n_obs, self.q * t))
        self.W = sp.hstack([Xs, Zs], format="csr")
        self.Wt = self.W.T.tocsr()
        self.n_eq = self.P + self.q * t
        # residual blocks: records grouped by the set of observed traits
        pats: dict = {}
        for r in range(self.n):
            pats.setdefault(tuple(int(k) for k in np.flatnonzero(obs[r])), []).append(r)
        self.patterns = {k: np.array(v) for k, v in pats.items()}
        rr, rc = [], []
        for key, recs in self.patterns.items():
            P_ = self.pos[np.ix_(recs, list(key))]                  # records x |key|
            rr.append(np.repeat(P_, len(key), axis=1).ravel())
            rc.append(np.tile(P_, (1, len(key))).ravel())
        rr, rc = np.concatenate(rr), np.concatenate(rc)
        probe = sp.csr_matrix((np.arange(1, rr.size + 1, dtype=np.float64), (rr, rc)),
                              shape=(self.n_obs, self.n_obs))
        self._rinv_order = probe.data.astype(np.int64) - 1          # csr slot -> our order
        self._rinv_ind = (probe.indices.copy(), probe.indptr.copy())
        self.free = np.arange(2, self.K)                    # indices into tau (0..K); tau_1 = 0

    def _check_fixed_effects(self):
        """Every trait's design must have full column rank on its records, and no 0/1
        column of the categorical trait's design may have all its records in one
        extreme category (the probit coefficient is then infinite under a flat prior:
        'extreme category problem')."""
        for j, X in enumerate(self.Xj):
            Xd = X.toarray()
            if np.linalg.matrix_rank(Xd) < Xd.shape[1]:
                raise ABPError("MODEL_NOT_IDENTIFIABLE", f"trait {j + 1}: the fixed-effect "
                               "design is not of full column rank on the records of this trait")
        Xd = self.Xj[self.c].toarray()
        yk = self.yk[self.cat_obs]
        for k in range(Xd.shape[1]):
            col = Xd[:, k]
            if not np.all((col == 0) | (col == 1)) or np.all(col == 1):
                continue
            sel = col == 1
            if sel.any() and (np.all(yk[sel] == 0) or np.all(yk[sel] == self.K - 1)):
                raise ABPError("MODEL_NOT_IDENTIFIABLE",
                               f"categorical trait, fixed-effect column {k}: all "
                               f"{int(sel.sum())} records are in one extreme category (its "
                               "effect is not finite under a flat prior); merge the level")

    def rinv(self, R0: np.ndarray) -> sp.csr_matrix:
        """Block-diagonal ``R^{-1}``: ``R0[o_r, o_r]^{-1}`` for every record."""
        data = []
        for key, recs in self.patterns.items():
            blk = np.linalg.inv(R0[np.ix_(key, key)])
            data.append(np.broadcast_to(blk.ravel(), (recs.size, blk.size)).ravel())
        d = np.concatenate(data)[self._rinv_order]
        return sp.csr_matrix((d, *self._rinv_ind), shape=(self.n_obs, self.n_obs))

    def coefficient(self, Rinv: sp.csr_matrix, G0inv: np.ndarray) -> sp.csr_matrix:
        prior = sp.block_diag([sp.csr_matrix((self.P, self.P)),
                               sp.kron(sp.csr_matrix(self.k_inv), G0inv)], format="csr")
        return (self.Wt @ Rinv @ self.W + prior).tocsr()

    def u(self, theta: np.ndarray) -> np.ndarray:
        return theta[self.P:].reshape(self.q, self.t)


def draw_location(P: MTProblem, fac, y: np.ndarray, R0: np.ndarray, G0: np.ndarray,
                  rng, Rinv: sp.csr_matrix | None = None) -> np.ndarray:
    """One exact draw of ``theta ~ N(C^{-1} W'R^{-1} y, C^{-1})`` for the observation
    vector ``y`` (record-major, liabilities for the categorical trait); ``fac``
    factorises ``C`` at ``R0``, ``G0``."""
    t = P.t
    Rinv = P.rinv(R0) if Rinv is None else Rinv
    e = np.empty(P.n_obs)
    for key, recs in P.patterns.items():
        L = np.linalg.cholesky(R0[np.ix_(key, key)])
        e[P.pos[np.ix_(recs, list(key))]] = rng.standard_normal((recs.size, len(key))) @ L.T
    rhs = P.Wt @ (Rinv @ (y + e))
    Lg = np.linalg.cholesky(np.linalg.inv(G0))                # L L' = G0^{-1}
    Zq = rng.standard_normal((P.q, t)) @ Lg.T
    rhs[P.P:] += np.column_stack([P.root(Zq[:, j]) for j in range(t)]).ravel()
    return fac.solve(rhs)


def draw_R0(E: np.ndarray, c: int, rng) -> np.ndarray:
    """Exact draw of ``R0 | residuals`` under the flat prior on ``(b, S)`` with
    ``R0[c, c] = 1`` (module notes, step 5); ``E`` is ``n x t`` (complete)."""
    n, t = E.shape
    o = [k for k in range(t) if k != c]
    ec = E[:, c]
    Eo = E[:, o]
    cc = float(ec @ ec)
    bhat = Eo.T @ ec / cc
    Sres = Eo.T @ Eo - np.outer(bhat, bhat) * cc
    m = t - 1
    S = np.atleast_2d(invwishart.rvs(df=n - m - 2, scale=0.5 * (Sres + Sres.T),
                                     random_state=rng))
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
        self.theta[:P.P] = beta0
        self.tau = np.concatenate([[-np.inf], P.tau0, [np.inf]])
        # the factor is built on the full pattern (a surrogate without exact zeros);
        # the actual C (zeros while R0, G0 are diagonal) is then refactorised on it
        ones = np.ones((t, t))
        Rs = R0 + 1e-3 * np.trace(R0) / t * ones
        Gi = np.linalg.inv(G0)
        self.fac = SparseLDL(P.coefficient(P.rinv(Rs), Gi + 1e-3 * np.trace(Gi) / t * ones))
        self._refactor()
        self.sd = 0.1
        self.n_prop = self.n_acc = 0
        yo = P.Y[P.rec_idx, P.trait_idx]
        self.y = np.where(P.trait_idx == P.c, 0.0, yo)
        self.e_full = np.zeros((P.n, t))
        self._draw_categorical(tuning=False, it=1, mh=False)

    def _refactor(self):
        self.Rinv = self.P.rinv(self.R0)
        self.fac.refactor(self.P.coefficient(self.Rinv, np.linalg.inv(self.G0)))

    # -- step 1: thresholds (Cowles) and categorical liabilities -----------------
    def _cond_cat(self, eta):
        """Mean and SD of ``l_rc`` given the record's observed continuous traits
        (records where the categorical trait is observed; other entries unused)."""
        P, c, R0 = self.P, self.P.c, self.R0
        m = np.zeros(P.n)
        s = np.ones(P.n)
        for key, recs in P.patterns.items():
            if c not in key:
                continue
            o = [k for k in key if k != c]
            m[recs] = eta[P.pos[recs, c]]
            if o:
                w = np.linalg.solve(R0[np.ix_(o, o)], R0[o, c])
                po = P.pos[np.ix_(recs, o)]
                m[recs] += (self.y[po] - eta[po]) @ w
                s[recs] = math.sqrt(R0[c, c] - R0[c, o] @ w)
        return m, s

    def _draw_categorical(self, tuning: bool, it: int, mh: bool = True):
        P = self.P
        eta = P.W @ self.theta
        m, s = self._cond_cat(eta)
        ob = P.cat_obs
        mo, so, yk = m[ob], s[ob], P.yk[ob]
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
            lp_new, _ = _log_p((new[yk] - mo) / so, (new[yk + 1] - mo) / so)
            lp_old, _ = _log_p((old[yk] - mo) / so, (old[yk + 1] - mo) / so)
            log_r = float(lp_new.sum() - lp_old.sum()) + log_q
            self.n_prop += 1
            if math.log(max(self.rng.random(), 1e-300)) < log_r:
                self.tau = new
                self.n_acc += 1
            if tuning and it % 50 == 0 and self.n_prop:
                rate = self.n_acc / self.n_prop
                self.sd *= 0.7 if rate < 0.2 else (1.4 if rate > 0.5 else 1.0)
                self.n_prop = self.n_acc = 0
        z = rtruncnorm(self.rng, np.zeros(mo.size), (self.tau[yk] - mo) / so,
                       (self.tau[yk + 1] - mo) / so)
        self.y[P.pos[ob, P.c]] = mo + so * z

    # -- step 2b: parameter expansion for the categorical trait ------------------
    def _scale_move(self):
        P, c, t = self.P, self.P.c, self.P.t
        U = P.u(self.theta)
        w = np.zeros(P.n_obs)
        sel = P.trait_idx == c
        w[sel] = U[P.animal_col[P.rec_idx[sel]], c]
        a2 = float(w @ (self.Rinv @ w))
        if a2 <= 0.0:
            return
        r = self.y - P.W @ self.theta + w
        a1 = float(w @ (self.Rinv @ r))
        nu, Psi = P.nu, P.psi
        if Psi is not None:
            Gi = np.linalg.inv(self.G0)
            p2 = Psi[c, c] * Gi[c, c]
            p1 = 2.0 * float(Psi[c, :] @ Gi[:, c] - Psi[c, c] * Gi[c, c])

        def logf(x):
            g = math.exp(x)
            out = -0.5 * (g * g * a2 - 2.0 * g * a1) + (t + 1) * x - (nu + t + 1) * x
            if Psi is not None:
                out -= 0.5 * (p1 / g + p2 / (g * g))
            return out
        g = math.exp(_slice_sample(logf, 0.0, 0.5, self.rng))
        U[:, c] *= g                                  # view into theta
        D = np.ones(t)
        D[c] = g
        self.G0 = self.G0 * np.outer(D, D)

    # -- step 2c: shear moves for the genetic covariances --------------------------
    def _shear_moves(self):
        P, t = self.P, self.P.t
        Psi = P.psi
        for i in range(t):
            sel = P.trait_idx == i
            ani = P.animal_col[P.rec_idx[sel]]
            for j in range(t):
                if i == j:
                    continue
                U = P.u(self.theta)
                w = np.zeros(P.n_obs)
                w[sel] = U[ani, j]
                Rw = self.Rinv @ w
                prec = float(w @ Rw)
                lin = float((self.y - P.W @ self.theta) @ Rw)
                if Psi is not None:
                    A = np.linalg.inv(self.G0)
                    prec += A[i, i] * Psi[j, j]
                    lin += float((Psi @ A)[j, i])
                if prec <= 0.0:
                    continue
                h = lin / prec + self.rng.standard_normal() / math.sqrt(prec)
                U[:, i] += h * U[:, j]                    # view into theta
                S = np.eye(t)
                S[i, j] = h
                self.G0 = S @ self.G0 @ S.T

    # -- step 3: residuals of the missing traits -------------------------------
    def _draw_missing_residuals(self):
        P, R0 = self.P, self.R0
        e_obs = self.y - P.W @ self.theta
        E = self.e_full
        for key, recs in P.patterns.items():
            ko = list(key)
            E[np.ix_(recs, ko)] = e_obs[P.pos[np.ix_(recs, ko)]]
            km = [k for k in range(P.t) if k not in key]
            if not km:
                continue
            A = R0[np.ix_(km, ko)] @ np.linalg.inv(R0[np.ix_(ko, ko)])
            V = R0[np.ix_(km, km)] - A @ R0[np.ix_(ko, km)]
            E[np.ix_(recs, km)] = (E[np.ix_(recs, ko)] @ A.T
                                   + self.rng.standard_normal((recs.size, len(km)))
                                   @ np.linalg.cholesky(V).T)

    def iterate(self, tuning: bool, it: int):
        P = self.P
        self._draw_categorical(tuning, it)
        self.theta = draw_location(P, self.fac, self.y, self.R0, self.G0, self.rng, self.Rinv)
        if self.cfg.fix_covariances:
            return
        if self.cfg.scale_move:
            self._scale_move()
        if self.cfg.shear_moves:
            self._shear_moves()
        self._draw_missing_residuals()
        self.G0 = draw_G0(P.u(self.theta), P.k_inv, self.rng, P.nu, P.g_prior)
        self.R0 = draw_R0(self.e_full, P.c, self.rng)
        self._refactor()


def _summ(A: np.ndarray) -> dict:
    return {"mean": A.mean(axis=0).tolist(), "sd": A.std(axis=0, ddof=1).tolist(),
            "q025": np.quantile(A, 0.025, axis=0).tolist(),
            "q975": np.quantile(A, 0.975, axis=0).tolist()}


def mt_threshold_gibbs(Y, cat: int, X, animal_col, k_inv, cfg: MTThresholdGibbsConfig
                       ) -> MTThresholdGibbsResult:
    """Run the sampler.  ``Y``: ``n x t`` (NaN = missing; column ``cat`` holds integer
    categories); ``X``: a list of per-trait designs (rows = records where the trait is
    observed, each with an intercept) or one design shared by the traits (rows =
    records); ``animal_col``: column of each record's animal in ``K``; ``k_inv``:
    ``K^{-1}``."""
    t0 = time.perf_counter()
    if cfg.chains < 2:
        raise ABPError("SPEC_INVALID", "at least 2 chains are required for R-hat (4 recommended)")
    if not (0 <= cfg.burn_in < cfg.iterations) or cfg.thin < 1:
        raise ABPError("SPEC_INVALID", "need 0 <= burn_in < iterations and thin >= 1")
    Y = np.asarray(Y, dtype=np.float64)
    Xs = list(X) if isinstance(X, (list, tuple)) else split_design(X, Y)
    P = MTProblem(Y, cat, Xs, animal_col, k_inv)
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
    vy = np.array([np.nanvar(Y[:, j]) if j != c else 1.0 for j in range(t)])
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
    beta0 = np.zeros(P.P)
    for j in range(t):
        rows = P.obs[:, j]
        yj = Y[rows, j] if j != c else np.full(int(rows.sum()), -raw[0])
        beta0[P.p_off[j]:P.p_off[j + 1]] = lsqr(P.Xj[j], yj, atol=1e-10, btol=1e-10)[0]
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
    s_uu = np.zeros((P.q, t, t))
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
                    Ud = P.u(chn.theta)
                    if keep_ebv:
                        ebv_draws[k].append(Ud.ravel().astype(np.float32))
                    s_theta += chn.theta
                    q_theta += chn.theta * chn.theta
                    s_uu += Ud[:, :, None] * Ud[:, None, :]
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
    Um = P.u(mean)
    pev_blocks = (s_uu / total - Um[:, :, None] * Um[:, None, :]) * total / max(total - 1, 1)
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
        Um.copy(), P.u(var).copy(), pev_blocks,
        [mean[P.p_off[j]:P.p_off[j + 1]].copy() for j in range(t)], s_tau / total, P.cats,
        _summ(Ga), _summ(Ra), derived, summaries, ediag, bool(ok), it, n_saved,
        [int(s.generate_state(1)[0]) for s in child],
        (acc_n / acc_prop) if acc_prop else None, time.perf_counter() - t0, P.n_eq,
        traces={key: np.array(store[key]) for key in scal})
