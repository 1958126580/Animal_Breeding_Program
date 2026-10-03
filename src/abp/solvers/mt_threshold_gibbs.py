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
    scale_move: bool = True                 #: scale moves for every trait (u/G0 and pe/P0)
    shear_moves: bool = True                #: parameter expansion for the genetic covariances
    prior_nu: float | None = None           #: IW degrees of freedom (default -(t + 1): flat)
    prior_G0: np.ndarray | None = None      #: prior guess of G0 (IW scale nu G_prior)
    residual_groups: list | None = None     #: partition of trait indices; R0 = 0 between groups
    start_P0: np.ndarray | None = None      #: permanent environment start (default 0.1 var(y) / 0.1)
    prior_nu_pe: float | None = None        #: IW degrees of freedom for P0 (default flat)
    prior_P0: np.ndarray | None = None      #: prior guess of P0 (IW scale nu_pe P_prior)
    prior_nu_r: float | None = None         #: IW degrees of freedom for R0 (continuous blocks)
    prior_R0: np.ndarray | None = None      #: prior guess of R0 (IW scale nu_r R_prior per block)


@dataclass
class MTThresholdGibbsResult:
    ebv: np.ndarray                 # q x t posterior means
    pev: np.ndarray                 # q x t posterior variances
    pev_blocks: np.ndarray          # q x t x t posterior covariances between traits
    fixed_mean: list                # per trait: posterior means of beta_j
    thresholds_mean: np.ndarray     # tau_1 .. tau_{K-1}
    categories: np.ndarray
    G0: dict                        # "mean", "sd", "median", "q025", "q975" (t x t lists)
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
    P0: dict | None = None          # permanent-environment covariance summaries (pe term only)
    pe_mean: np.ndarray | None = None   # m x t posterior means of the permanent environment
    maternal_ebv: np.ndarray | None = None   # q x t maternal genetic effects (dam_col only)
    maternal_pev: np.ndarray | None = None   # q x t their posterior variances
    genetic_blocks: np.ndarray | None = None  # q x 2t x 2t direct+maternal (dam_col only)
    thresholds_by_trait: dict = field(default_factory=dict)   # trait -> tau_1..tau_{K-1}
    categories_by_trait: dict = field(default_factory=dict)   # trait -> category codes


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

    def __init__(self, Y: np.ndarray, cat, X_per_trait: list,
                 animal_col: np.ndarray, k_inv, pe_col: np.ndarray | None = None,
                 dam_col: np.ndarray | None = None):
        Y = np.asarray(Y, dtype=np.float64)
        self.n, self.t = Y.shape
        t = self.t
        # genetic effects per animal: r = t (direct) or 2t (direct, then maternal)
        self.maternal = dam_col is not None
        self.r = 2 * t if self.maternal else t
        if self.r < 2:
            raise ABPError("SPEC_INVALID", "the multi-trait sampler needs >= 2 traits "
                                           "(or one trait with a maternal genetic effect)")
        # categorical traits: None, one index, or a list (round 14: several, each in its own
        # residual group - checked by the driver)
        cs = [] if cat is None else ([int(cat)] if np.ndim(cat) == 0 else
                                     sorted(int(x) for x in cat))
        if len(set(cs)) != len(cs) or any(not 0 <= c < t for c in cs):
            raise ABPError("SPEC_INVALID", "categorical trait index out of range or repeated")
        self.cs = cs
        self.c = cs[0] if len(cs) == 1 else None      # the single categorical trait (if one)
        obs = ~np.isnan(Y)
        if np.any(~obs.any(axis=1)):
            raise ABPError("SCHEMA_TYPE", "every record needs at least one observed trait")
        self.Y, self.obs = Y, obs
        #: per categorical trait: categories, K, category index per record, observed mask
        self.catinfo = {}
        for c in cs:
            yc = Y[obs[:, c], c]
            if not np.all(yc == np.round(yc)):
                raise ABPError("SCHEMA_TYPE", f"categorical trait {c + 1} needs integer "
                               "category codes")
            cats = np.unique(yc)
            if cats.size < 2:
                raise ABPError("MODEL_NOT_IDENTIFIABLE", f"categorical trait {c + 1}: all "
                               "records are in one category")
            yk = np.zeros(self.n, dtype=np.int64)
            yk[obs[:, c]] = np.searchsorted(cats, Y[obs[:, c], c])
            self.catinfo[c] = {"cats": cats, "K": int(cats.size), "yk": yk, "obs": obs[:, c],
                               "free": np.arange(2, int(cats.size))}
        # single-categorical attributes (unchanged interface)
        first = self.catinfo[cs[0]] if cs else None
        self.cats = first["cats"] if first else np.array([])
        self.K = first["K"] if first else 1
        self.yk = first["yk"] if first else np.zeros(self.n, dtype=np.int64)
        self.cat_obs = first["obs"] if first else np.zeros(self.n, dtype=bool)
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
        if self.q <= 2 * self.r + 1:
            raise ABPError("MODEL_NOT_IDENTIFIABLE", "too few animals for the covariance priors")
        self.animal_col = np.asarray(animal_col, dtype=np.int64)
        if self.maternal:
            self.dam_col = np.asarray(dam_col, dtype=np.int64)
            if self.dam_col.shape != (self.n,) or self.dam_col.max() >= self.q:
                raise ValueError("dam_col needs one dam index (-1 = unknown) per record")
            if int((self.dam_col >= 0).sum()) <= 2 * t + 1:
                raise ABPError("MODEL_NOT_IDENTIFIABLE", "too few records with a known dam "
                               "for a maternal genetic effect")
        else:
            self.dam_col = None
        # optional permanent-environment (iid) term: pe ~ N(0, I_m (x) P0)
        if pe_col is None:
            self.pe_col, self.m = None, 0
        else:
            self.pe_col = np.asarray(pe_col, dtype=np.int64)
            if self.pe_col.shape != (self.n,) or self.pe_col.min() < 0:
                raise ValueError("pe_col needs one non-negative level index per record")
            self.m = int(self.pe_col.max()) + 1
            if self.m <= 2 * t + 1:
                raise ABPError("MODEL_NOT_IDENTIFIABLE", "too few permanent-environment levels")
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
        r = self.r
        zr, zc = [np.arange(self.n_obs)], [self.animal_col[rec_idx] * r + trait_idx]
        if self.maternal:                    # maternal effect of the record's dam
            known = self.dam_col[rec_idx] >= 0
            zr.append(np.flatnonzero(known))
            zc.append(self.dam_col[rec_idx[known]] * r + t + trait_idx[known])
        zr, zc = np.concatenate(zr), np.concatenate(zc)
        Zs = sp.csr_matrix((np.ones(zr.size), (zr, zc)), shape=(self.n_obs, self.q * r))
        blocks = [Xs, Zs]
        if self.m:
            blocks.append(sp.csr_matrix((np.ones(self.n_obs), (np.arange(self.n_obs),
                                                               self.pe_col[rec_idx] * t
                                                               + trait_idx)),
                                        shape=(self.n_obs, self.m * t)))
        self.W = sp.hstack(blocks, format="csr")
        self.Wt = self.W.T.tocsr()
        self.n_eq = self.P + self.q * r + self.m * t
        self.pe_off = self.P + self.q * r
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
        self._build_maps()
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
        for c, ci in self.catinfo.items():
            Xd = self.Xj[c].toarray()
            yk = ci["yk"][ci["obs"]]
            for k in range(Xd.shape[1]):
                col = Xd[:, k]
                if not np.all((col == 0) | (col == 1)) or np.all(col == 1):
                    continue
                sel = col == 1
                if sel.any() and (np.all(yk[sel] == 0) or np.all(yk[sel] == ci["K"] - 1)):
                    raise ABPError("MODEL_NOT_IDENTIFIABLE",
                                   f"categorical trait {c + 1}, fixed-effect column {k}: all "
                                   f"{int(sel.sum())} records are in one extreme category (its "
                                   "effect is not finite under a flat prior); merge the level")

    def _build_maps(self):
        """Fixed symmetric pattern of ``C`` and the linear maps from the values of
        ``R^{-1}`` (CSR slots) and of ``G0^{-1}`` (``t^2``) to the values of ``C`` on that
        pattern: ``C = W'R^{-1}W + blockdiag(0, K^{-1} (x) G0^{-1})`` is linear in both, so
        each iteration costs two sparse matrix-vector products instead of a sparse
        re-assembly."""
        from .cholesky import _symmetric_pattern
        t, W = self.t, self.W
        ones = np.ones((t, t))
        # structural pattern from absolute values: a numerical surrogate can cancel
        # exactly (repeated records: an entry of W'R^-1 W equal and opposite to the
        # prior entry), which would drop an entry that the actual C needs
        Wa = abs(W)
        r = self.r
        prior = [sp.csr_matrix((self.P, self.P)),
                 sp.kron(abs(sp.csr_matrix(self.k_inv)), np.ones((r, r)))]
        if self.m:
            prior.append(sp.kron(sp.identity(self.m, format="csr"), ones))
        surrogate = (Wa.T @ abs(self.rinv(np.eye(t) + 0.5 * ones)) @ Wa
                     + sp.block_diag(prior, format="csr")).tocsr()
        surrogate = surrogate + sp.diags(np.asarray(abs(surrogate).sum(axis=1)).ravel() + 1.0)
        pat = _symmetric_pattern(surrogate)
        self.cpat = pat
        #: SPD (diagonally dominant) values on the full pattern, used to build the factor
        self.surrogate_values = pat.data.copy()
        n_eq = self.n_eq
        prow = np.repeat(np.arange(n_eq, dtype=np.int64), np.diff(pat.indptr))
        pkey = prow * n_eq + pat.indices                         # sorted (canonical CSR)

        def locate(row, col):
            key = row.astype(np.int64) * n_eq + col
            pos = np.searchsorted(pkey, key)
            if np.any(pos >= pkey.size) or np.any(pkey[np.minimum(pos, pkey.size - 1)] != key):
                raise ValueError("entry outside the pattern of C")
            return pos
        # R^{-1} slots: rows a, cols b
        ind, ptr = self._rinv_ind
        S = ind.size
        a = np.repeat(np.arange(self.n_obs), np.diff(ptr))
        b = ind
        la = np.diff(W.indptr)[a]
        lb = np.diff(W.indptr)[b]
        cnt = la * lb
        slot = np.repeat(np.arange(S), cnt)
        start = np.repeat(np.cumsum(cnt) - cnt, cnt)
        p_ = np.arange(int(cnt.sum())) - start
        i = p_ // lb[slot]
        j = p_ % lb[slot]
        ka = W.indptr[a[slot]] + i
        kb = W.indptr[b[slot]] + j
        wv = W.data[ka] * W.data[kb]
        nz = wv != 0.0                    # stored zeros (e.g. a covariate value 0) add nothing
        self.M_R = sp.csr_matrix((wv[nz], (locate(W.indices[ka][nz], W.indices[kb][nz]),
                                           slot[nz])), shape=(pat.nnz, S))
        # K^{-1} (x) G0^{-1} (stored zeros of K^{-1}, e.g. exact cancellation in A^{-1}, dropped)
        Kc = sp.csr_matrix(self.k_inv, copy=True)
        Kc.eliminate_zeros()
        Kc = Kc.tocoo()
        abr = np.array([(x, y) for x in range(r) for y in range(r)])
        rows = (self.P + Kc.row[:, None] * r + abr[None, :, 0]).ravel()
        cols = (self.P + Kc.col[:, None] * r + abr[None, :, 1]).ravel()
        gidx = np.tile(np.arange(r * r), Kc.nnz)
        self.M_G = sp.csr_matrix((np.repeat(Kc.data, r * r), (locate(rows, cols), gidx)),
                                 shape=(pat.nnz, r * r))
        ab = np.array([(x, y) for x in range(t) for y in range(t)])
        # I_m (x) P0^{-1}
        self.M_P = None
        if self.m:
            lev = np.arange(self.m)
            rows = (self.pe_off + lev[:, None] * t + ab[None, :, 0]).ravel()
            cols = (self.pe_off + lev[:, None] * t + ab[None, :, 1]).ravel()
            self.M_P = sp.csr_matrix((np.ones(rows.size), (locate(rows, cols),
                                                           np.tile(np.arange(t * t), self.m))),
                                     shape=(pat.nnz, t * t))

    def coefficient_values(self, Rinv: sp.csr_matrix, G0inv: np.ndarray,
                           P0inv: np.ndarray | None = None) -> np.ndarray:
        """Values of ``C`` on :attr:`cpat` (``Rinv`` from :meth:`rinv`; ``P0inv`` only
        with a permanent-environment term)."""
        v = self.M_R @ Rinv.data + self.M_G @ np.ascontiguousarray(G0inv).ravel()
        if self.M_P is not None:
            v = v + self.M_P @ np.ascontiguousarray(P0inv).ravel()
        return v

    def rinv(self, R0: np.ndarray) -> sp.csr_matrix:
        """Block-diagonal ``R^{-1}``: ``R0[o_r, o_r]^{-1}`` for every record."""
        data = []
        for key, recs in self.patterns.items():
            blk = np.linalg.inv(R0[np.ix_(key, key)])
            data.append(np.broadcast_to(blk.ravel(), (recs.size, blk.size)).ravel())
        d = np.concatenate(data)[self._rinv_order]
        return sp.csr_matrix((d, *self._rinv_ind), shape=(self.n_obs, self.n_obs))

    def coefficient(self, Rinv: sp.csr_matrix, G0inv: np.ndarray,
                    P0inv: np.ndarray | None = None) -> sp.csr_matrix:
        blocks = [sp.csr_matrix((self.P, self.P)), sp.kron(sp.csr_matrix(self.k_inv), G0inv)]
        if self.m:
            blocks.append(sp.kron(sp.identity(self.m, format="csr"), P0inv))
        prior = sp.block_diag(blocks, format="csr")
        return (self.Wt @ Rinv @ self.W + prior).tocsr()

    def u(self, theta: np.ndarray) -> np.ndarray:
        """Genetic effects ``q x r`` (columns: direct traits, then maternal traits)."""
        return theta[self.P:self.pe_off].reshape(self.q, self.r)

    def incidence(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Observations that load genetic column ``k`` and the animal index of each:
        direct column ``k < t``: the record's animal for trait ``k``; maternal column
        ``t + j``: the record's dam (when known) for trait ``j``."""
        t = self.t
        sel = self.trait_idx == (k % t)
        if k < t:
            return np.flatnonzero(sel), self.animal_col[self.rec_idx[sel]]
        obs = np.flatnonzero(sel & (self.dam_col[self.rec_idx] >= 0))
        return obs, self.dam_col[self.rec_idx[obs]]

    def pe(self, theta: np.ndarray) -> np.ndarray:
        """Permanent-environment effects (``m x t``; empty without the term)."""
        return theta[self.pe_off:].reshape(self.m, self.t)


def draw_location(P: MTProblem, fac, y: np.ndarray, R0: np.ndarray, G0: np.ndarray,
                  rng, Rinv: sp.csr_matrix | None = None,
                  P0: np.ndarray | None = None) -> np.ndarray:
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
    Zq = rng.standard_normal((P.q, P.r)) @ Lg.T
    rhs[P.P:P.pe_off] += np.column_stack([P.root(Zq[:, j]) for j in range(P.r)]).ravel()
    if P.m:
        Lp = np.linalg.cholesky(np.linalg.inv(P0))            # L L' = P0^{-1}, root of I is I
        rhs[P.pe_off:] += (rng.standard_normal((P.m, t)) @ Lp.T).ravel()
    return fac.solve(rhs)


def draw_R0(E: np.ndarray, c, rng, groups: list | None = None,
            nu: float | None = None, R_prior: np.ndarray | None = None) -> np.ndarray:
    """Exact draw of ``R0 | residuals`` (``E``: ``n x t``, complete).  ``groups``: a
    partition of the traits into residual blocks (covariances between blocks are 0;
    default one block).  Each block independently: the block with the categorical trait
    ``c`` by the flat prior on ``(b, S)`` with ``R0[c, c] = 1`` (module notes, step 5),
    any other block from ``IW(E_B'E_B + nu R_prior_BB, n + nu)``; the default
    ``nu = -(|B| + 1)``, ``R_prior = 0`` is the flat prior on the PD matrices
    (``IW(E_B'E_B, n - |B| - 1)``); ``nu, R_prior`` give the block the marginal prior
    ``IW(nu, nu R_prior_BB)``."""
    n, t = E.shape
    if groups is None:
        groups = [list(range(t))]
    cset = set() if c is None else ({int(c)} if np.ndim(c) == 0 else {int(x) for x in c})
    R0 = np.zeros((t, t))
    for B in groups:
        B = list(B)
        EB = E[:, B]
        inb = [x for x in B if x in cset]
        if len(inb) > 1:
            raise ABPError("SPEC_INVALID", "at most one categorical trait per residual group")
        if inb:
            R0[np.ix_(B, B)] = _draw_R0_categorical(EB, B.index(inb[0]), rng)
        else:
            S = EB.T @ EB
            nb = -(len(B) + 1.0) if R_prior is None else float(nu)
            if R_prior is not None:
                S = S + nb * np.asarray(R_prior, dtype=np.float64)[np.ix_(B, B)]
            R0[np.ix_(B, B)] = np.atleast_2d(invwishart.rvs(
                df=n + nb, scale=0.5 * (S + S.T), random_state=rng))
    return R0


def _draw_R0_categorical(E: np.ndarray, c: int, rng) -> np.ndarray:
    """Korsgaard step for one block containing the categorical trait (index ``c``)."""
    n, t = E.shape
    if t == 1:
        return np.ones((1, 1))
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
    def __init__(self, P: MTProblem, cfg: MTThresholdGibbsConfig, rng, G0, R0, beta0,
                 P0=None):
        from .cholesky import SparseLDL
        self.P, self.cfg, self.rng = P, cfg, rng
        self.G0, self.R0 = G0.copy(), R0.copy()
        self.P0 = None if P0 is None else P0.copy()
        t = P.t
        self.theta = np.zeros(P.n_eq)
        self.theta[:P.P] = beta0
        self.taus = {c: np.concatenate([[-np.inf], P.tau0s[c], [np.inf]]) for c in P.cs}
        self.sds = {c: 0.1 for c in P.cs}
        self.n_props = {c: 0 for c in P.cs}
        self.n_accs = {c: 0 for c in P.cs}
        # the factor is built on the full pattern (a surrogate without exact zeros);
        # the actual C (zeros while R0, G0 are diagonal) is then refactorised on it
        surrogate = sp.csr_matrix((P.surrogate_values, P.cpat.indices, P.cpat.indptr),
                                  shape=P.cpat.shape)
        self.fac = SparseLDL(surrogate)
        if self.fac.C.nnz != P.cpat.nnz:
            raise ABPError("FACTORIZATION_FAILED", "internal: pattern of C changed")
        self._refactor()
        self.n_prop = self.n_acc = 0
        yo = P.Y[P.rec_idx, P.trait_idx]
        self.y = np.where(np.isin(P.trait_idx, P.cs), 0.0, yo)
        self.e_full = np.zeros((P.n, t))
        self._draw_categorical(tuning=False, it=1, mh=False)

    def _refactor(self):
        self.Rinv = self.P.rinv(self.R0)
        self.fac.refactor_values(self.P.coefficient_values(
            self.Rinv, np.linalg.inv(self.G0),
            None if self.P0 is None else np.linalg.inv(self.P0)))

    @property
    def tau(self):
        """Thresholds of the (first) categorical trait (single-categorical interface)."""
        return self.taus[self.P.cs[0]]

    # -- step 1: thresholds (Cowles) and categorical liabilities -----------------
    def _cond_cat(self, eta, c):
        """Mean and SD of ``l_rc`` given the record's other observed traits (records
        where categorical trait ``c`` is observed; other entries unused)."""
        P, R0 = self.P, self.R0
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
        """Step 1 for every categorical trait in turn, each given the current liabilities
        of the others (a Gibbs sweep over the categorical traits)."""
        for c in self.P.cs:
            self._draw_categorical_trait(c, tuning, it, mh)

    def _draw_categorical_trait(self, c: int, tuning: bool, it: int, mh: bool):
        P = self.P
        ci = P.catinfo[c]
        eta = P.W @ self.theta
        m, s = self._cond_cat(eta, c)
        ob = ci["obs"]
        mo, so, yk = m[ob], s[ob], ci["yk"][ob]
        free = ci["free"]
        if mh and free.size:
            old = self.taus[c]
            new = old.copy()
            sd = self.sds[c]
            for j in free:
                lo, hi = new[j - 1], old[j + 1]
                a, b = (lo - old[j]) / sd, (hi - old[j]) / sd
                pa, pb = ndtr(a), ndtr(b)
                new[j] = old[j] + sd * ndtri(pa + self.rng.random() * (pb - pa))
            log_q = 0.0
            for j in free:
                log_q += math.log(max(ndtr((old[j + 1] - old[j]) / sd)
                                      - ndtr((new[j - 1] - old[j]) / sd), 1e-300))
                log_q -= math.log(max(ndtr((new[j + 1] - new[j]) / sd)
                                      - ndtr((old[j - 1] - new[j]) / sd), 1e-300))
            lp_new, _ = _log_p((new[yk] - mo) / so, (new[yk + 1] - mo) / so)
            lp_old, _ = _log_p((old[yk] - mo) / so, (old[yk + 1] - mo) / so)
            log_r = float(lp_new.sum() - lp_old.sum()) + log_q
            self.n_prop += 1
            self.n_props[c] += 1
            if math.log(max(self.rng.random(), 1e-300)) < log_r:
                self.taus[c] = new
                self.n_acc += 1
                self.n_accs[c] += 1
            if tuning and it % 50 == 0 and self.n_props[c]:
                rate = self.n_accs[c] / self.n_props[c]
                self.sds[c] *= 0.7 if rate < 0.2 else (1.4 if rate > 0.5 else 1.0)
                self.n_props[c] = self.n_accs[c] = 0
        tau = self.taus[c]
        z = rtruncnorm(self.rng, np.zeros(mo.size), (tau[yk] - mo) / so,
                       (tau[yk + 1] - mo) / so)
        self.y[P.pos[ob, c]] = mo + so * z

    # -- step 2b: scale moves (parameter expansion) -------------------------------
    def _scale_move(self, j: int, term: str = "u"):
        """Generalised Gibbs step on the group ``x -> g x`` for column ``j`` of the
        genetic (``term = "u"``: ``u_j -> g u_j``, ``G0 -> D G0 D``) or the
        permanent-environment effects (``"pe"``: ``p_j -> g p_j``, ``P0 -> D P0 D``),
        ``D = diag(1, .., g, .., 1)``.  The density of the effects given the transformed
        covariance is invariant (the Jacobian ``g^q`` cancels ``|D|^{-q}``), so ``log g``
        has the density likelihood x IW-prior Jacobian ``g^{t+1}`` x prior, sampled by
        slice sampling (Liu & Sabatti 2000)."""
        P = self.P
        w = np.zeros(P.n_obs)
        if term == "u":
            V = P.u(self.theta)
            obs, lev = P.incidence(j)
            nu, Psi, M = P.nu, P.psi, self.G0
        else:
            V = P.pe(self.theta)
            obs = np.flatnonzero(P.trait_idx == j)
            lev = P.pe_col[P.rec_idx[obs]]
            nu = -(P.t + 1.0) if P.nu_pe is None else P.nu_pe
            Psi = None if P.p_prior is None else P.nu_pe * P.p_prior
            M = self.P0
        t = M.shape[0]                         # dimension of the transformed matrix
        w[obs] = V[lev, j]
        a2 = float(w @ (self.Rinv @ w))
        if a2 <= 0.0:
            return
        r = self.y - P.W @ self.theta + w
        a1 = float(w @ (self.Rinv @ r))
        if Psi is not None:
            Mi = np.linalg.inv(M)
            p2 = Psi[j, j] * Mi[j, j]
            p1 = 2.0 * float(Psi[j, :] @ Mi[:, j] - Psi[j, j] * Mi[j, j])

        def logf(x):
            g = math.exp(x)
            out = -0.5 * (g * g * a2 - 2.0 * g * a1) + (t + 1) * x - (nu + t + 1) * x
            if Psi is not None:
                out -= 0.5 * (p1 / g + p2 / (g * g))
            return out
        g = math.exp(_slice_sample(logf, 0.0, 0.5, self.rng))
        V[:, j] *= g                                  # view into theta
        D = np.ones(t)
        D[j] = g
        if term == "u":
            self.G0 = self.G0 * np.outer(D, D)
        else:
            self.P0 = self.P0 * np.outer(D, D)

    # -- step 2c: shear moves for the genetic covariances --------------------------
    def _shear_moves(self):
        P, t = self.P, self.P.r
        Psi = P.psi
        for i in range(t):
            obs, lev = P.incidence(i)
            for j in range(t):
                if i == j:
                    continue
                U = P.u(self.theta)
                w = np.zeros(P.n_obs)
                w[obs] = U[lev, j]
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
        self.theta = draw_location(P, self.fac, self.y, self.R0, self.G0, self.rng, self.Rinv,
                                   self.P0)
        if self.cfg.fix_covariances:
            return
        if self.cfg.scale_move:
            # categorical trait (identifies the liability scale) and, since round 10,
            # every trait and the permanent environment (variance <-> effects mixing)
            for k in range(P.r):
                self._scale_move(k, "u")
            for j in range(P.t if P.m else 0):
                self._scale_move(j, "pe")
        if self.cfg.shear_moves:
            self._shear_moves()
        self._draw_missing_residuals()
        self.G0 = draw_G0(P.u(self.theta), P.k_inv, self.rng, P.nu, P.g_prior)
        if P.m:
            self.P0 = draw_G0(P.pe(self.theta), sp.identity(P.m, format="csr"), self.rng,
                              P.nu_pe, P.p_prior)
        self.R0 = draw_R0(self.e_full, P.cs, self.rng, P.groups, P.nu_r, P.r_prior)
        self._refactor()


def _summ(A: np.ndarray) -> dict:
    return {"mean": A.mean(axis=0).tolist(), "sd": A.std(axis=0, ddof=1).tolist(),
            "median": np.median(A, axis=0).tolist(),
            "q025": np.quantile(A, 0.025, axis=0).tolist(),
            "q975": np.quantile(A, 0.975, axis=0).tolist()}


def mt_threshold_gibbs(Y, cat: int | None, X, animal_col, k_inv, cfg: MTThresholdGibbsConfig,
                       pe_col=None, dam_col=None) -> MTThresholdGibbsResult:
    """Run the sampler.  ``Y``: ``n x t`` (NaN = missing; column ``cat`` holds integer
    categories; ``cat = None``: all traits continuous - the Bayesian multi-trait linear
    model, ``R0`` from an inverse Wishart); ``X``: a list of per-trait designs (rows = records where the trait is
    observed, each with an intercept) or one design shared by the traits (rows =
    records); ``animal_col``: column of each record's animal in ``K``; ``k_inv``:
    ``K^{-1}``; ``pe_col``: optional level (0..m-1) of each record in a permanent-
    environment term ``pe ~ N(0, I_m (x) P0)`` (repeated records); ``dam_col``: optional
    column in ``K`` of each record's dam (-1 = unknown) for a maternal genetic effect:
    the genetic effects of an animal are then ``(direct_1..t, maternal_1..t)`` with a
    ``2t x 2t`` covariance matrix ``G0`` (direct-maternal covariances included)."""
    t0 = time.perf_counter()
    if cfg.chains < 2:
        raise ABPError("SPEC_INVALID", "at least 2 chains are required for R-hat (4 recommended)")
    if not (0 <= cfg.burn_in < cfg.iterations) or cfg.thin < 1:
        raise ABPError("SPEC_INVALID", "need 0 <= burn_in < iterations and thin >= 1")
    Y = np.asarray(Y, dtype=np.float64)
    Xs = list(X) if isinstance(X, (list, tuple)) else split_design(X, Y)
    P = MTProblem(Y, cat, Xs, animal_col, k_inv, pe_col, dam_col)
    t, r, cs = P.t, P.r, P.cs
    iscat = np.isin(np.arange(t), cs)
    P.nu = -(r + 1.0) if cfg.prior_nu is None else float(cfg.prior_nu)
    P.g_prior = None
    P.psi = None
    if cfg.prior_G0 is not None:
        if P.nu <= r - 1:
            raise ABPError("SPEC_INVALID", f"a proper inverse Wishart prior needs nu > {r - 1}")
        P.g_prior = _check_pd(cfg.prior_G0, "prior G0")
        if P.g_prior.shape != (r, r):
            raise ABPError("SPEC_INVALID", f"prior G0 must be {r} x {r}")
        P.psi = P.nu * P.g_prior
    elif cfg.prior_nu is not None:
        raise ABPError("SPEC_INVALID", "prior_nu needs prior_G0")
    P.nu_pe, P.p_prior = None, None
    if cfg.prior_P0 is not None:
        if P.m == 0:
            raise ABPError("SPEC_INVALID", "prior_P0 needs a permanent-environment term")
        if cfg.prior_nu_pe is None or cfg.prior_nu_pe <= t - 1:
            raise ABPError("SPEC_INVALID", f"a proper P0 prior needs prior_nu_pe > {t - 1}")
        P.nu_pe = float(cfg.prior_nu_pe)
        P.p_prior = _check_pd(cfg.prior_P0, "prior P0")
        if P.p_prior.shape != (t, t):
            raise ABPError("SPEC_INVALID", f"prior P0 must be {t} x {t}")
    elif cfg.prior_nu_pe is not None:
        raise ABPError("SPEC_INVALID", "prior_nu_pe needs prior_P0")
    P.nu_r, P.r_prior = None, None
    vy = np.array([np.nanvar(Y[:, j]) if not iscat[j] else 1.0 for j in range(t)])
    G0s = _check_pd(cfg.start_G0, "start G0") if cfg.start_G0 is not None else \
        np.diag(np.concatenate([np.where(iscat, 0.2, 0.3 * vy)]
                               + ([np.where(iscat, 0.1, 0.1 * vy)]
                                  if P.maternal else [])))
    if G0s.shape != (r, r):
        raise ABPError("SPEC_INVALID", f"start G0 must be {r} x {r}")
    R0s = _check_pd(cfg.start_R0, "start R0") if cfg.start_R0 is not None else \
        np.diag(np.where(iscat, 1.0, 0.7 * vy))
    if any(abs(R0s[c, c] - 1.0) > 1e-12 for c in cs):
        raise ABPError("SPEC_INVALID", "the residual variance of a categorical trait must be 1")
    P0s = None
    if P.m:
        P0s = _check_pd(cfg.start_P0, "start P0") if cfg.start_P0 is not None else \
            np.diag(np.where(iscat, 0.1, 0.1 * vy))
    P.groups = None
    if cfg.residual_groups is not None:
        flat = sorted(int(j) for g in cfg.residual_groups for j in g)
        if flat != list(range(t)) or any(len(g) == 0 for g in cfg.residual_groups):
            raise ABPError("SPEC_INVALID", "residual_groups must partition the traits")
        P.groups = [sorted(int(j) for j in g) for g in cfg.residual_groups]
        label = np.empty(t, dtype=np.int64)
        for k, g in enumerate(P.groups):
            label[g] = k
        if np.any((label[:, None] != label[None, :]) & (R0s != 0)):
            raise ABPError("SPEC_INVALID", "start R0 must be 0 between residual groups")
    if len(cs) > 1:
        # several categorical traits: each in a residual group of its own (no residual
        # covariance between two liabilities; the Korsgaard step handles one per block)
        blocks = P.groups or [list(range(t))]
        if any(sum(1 for x in B if x in cs) > 1 for B in blocks):
            raise ABPError("SPEC_INVALID", "with several categorical traits put each in a "
                           "different residual group (residual_groups); residual covariances "
                           "between two categorical traits are not estimated in this version")
    if cfg.prior_R0 is not None:
        if cfg.prior_nu_r is None:
            raise ABPError("SPEC_INVALID", "prior_R0 needs prior_nu_r")
        P.r_prior = _check_pd(cfg.prior_R0, "prior R0")
        if P.r_prior.shape != (t, t):
            raise ABPError("SPEC_INVALID", f"prior R0 must be {t} x {t}")
        blocks = P.groups or [list(range(t))]
        for B in blocks:
            if any(c in B for c in cs) and len(B) > 1:
                raise ABPError("SPEC_INVALID", "an R0 prior applies to blocks of continuous "
                               "traits; put the categorical trait in its own residual group")
            if not any(c in B for c in cs) and not cfg.prior_nu_r > len(B) - 1:
                raise ABPError("SPEC_INVALID", f"a proper R0 prior needs nu > {len(B) - 1}")
        P.nu_r = float(cfg.prior_nu_r)
    elif cfg.prior_nu_r is not None:
        raise ABPError("SPEC_INVALID", "prior_nu_r needs prior_R0")
    # starting thresholds from the category proportions, tau_1 = 0 (intercept absorbs it)
    raws = {}
    for c, ci in P.catinfo.items():
        cum = np.cumsum(np.bincount(ci["yk"][ci["obs"]], minlength=ci["K"]))[:-1] \
            / ci["obs"].sum()
        raws[c] = ndtri(np.clip(cum, 1e-6, 1 - 1e-6))
    P.tau0s = {c: raws[c] - raws[c][0] for c in cs}
    P.tau0 = P.tau0s[cs[0]] if cs else np.zeros(0)
    beta0 = np.zeros(P.P)
    for j in range(t):
        rows = P.obs[:, j]
        yj = Y[rows, j] if not iscat[j] else np.full(int(rows.sum()), -raws[j][0])
        beta0[P.p_off[j]:P.p_off[j + 1]] = lsqr(P.Xj[j], yj, atol=1e-10, btol=1e-10)[0]
    ss = np.random.SeedSequence(cfg.seed)
    child = ss.spawn(cfg.chains)
    chains = []
    for ch in child:
        rng = np.random.default_rng(ch)
        if cfg.fix_covariances:
            G0, R0, P0 = G0s, R0s, P0s
        else:
            f = np.exp(rng.uniform(-0.5, 0.5, r))
            G0 = G0s * np.outer(f, f)
            R0 = R0s.copy()
            P0 = None if P0s is None else P0s * np.outer(f[:t], f[:t])
        chains.append(_Chain(P, cfg, rng, G0, R0, beta0, P0))
    tri = [(i, j) for i in range(t) for j in range(i, t)]
    # free thresholds: tau_<k> with one categorical trait (unchanged names), tau<c>_<k>
    # (trait c, 0-based) with several
    tau_key = {(c, k): (f"tau_{k}" if len(cs) == 1 else f"tau{c}_{k}")
               for c in cs for k in P.catinfo[c]["free"]}
    tau_names = list(tau_key.values())
    trg = [(i, j) for i in range(r) for j in range(i, r)]
    scal = [f"G0_{i}_{j}" for i, j in trg] + \
        [f"R0_{i}_{j}" for i, j in tri if not (i == j and iscat[i])] + \
        ([f"P0_{i}_{j}" for i, j in tri] if P.m else []) + \
        [f"rG_{i}_{j}" for i, j in trg if i < j] + [f"h2_{j}" for j in range(t)] + \
        ([f"c2_{j}" for j in range(t)] if P.m else []) + \
        ([f"m2_{j}" for j in range(t)] if P.maternal else []) + \
        tau_names
    store = {k: [[] for _ in chains] for k in scal}
    Gs, Rs, Ps = [], [], []
    keep_ebv = P.q * r * (cfg.max_iterations // cfg.thin) * cfg.chains <= EBV_STORE_LIMIT
    ebv_draws = [[] for _ in chains]
    s_theta = np.zeros(P.n_eq)
    q_theta = np.zeros(P.n_eq)
    s_tau = {c: np.zeros(P.catinfo[c]["K"] - 1) for c in cs}
    s_uu = np.zeros((P.q, r, r))
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
                    Pm = chn.P0 if P.m else np.zeros((t, t))
                    vals = {f"G0_{i}_{j}": G[i, j] for i, j in trg}
                    vals.update({f"R0_{i}_{j}": R[i, j] for i, j in tri})
                    vals.update({f"P0_{i}_{j}": Pm[i, j] for i, j in tri})
                    vals.update({f"rG_{i}_{j}": G[i, j] / math.sqrt(G[i, i] * G[j, j])
                                 for i, j in trg if i < j})
                    # phenotypic variance of a record with maternal effects:
                    # var_A + var_M + cov_AM (+ P + R) (Willham 1972)
                    tot = np.diag(G)[:t] + np.diag(Pm) + np.diag(R)
                    if P.maternal:
                        tot = tot + np.diag(G)[t:] + np.array([G[j, t + j] for j in range(t)])
                        vals.update({f"m2_{j}": G[t + j, t + j] / tot[j] for j in range(t)})
                    vals.update({f"h2_{j}": G[j, j] / tot[j] for j in range(t)})
                    vals.update({f"c2_{j}": Pm[j, j] / tot[j] for j in range(t)})
                    vals.update({nm: chn.taus[c][k] for (c, k), nm in tau_key.items()})
                    for key in scal:
                        store[key][k].append(float(vals[key]))
                    Gs.append(G.copy())
                    Rs.append(R.copy())
                    if P.m:
                        Ps.append(Pm.copy())
                    Ud = P.u(chn.theta)
                    if keep_ebv:
                        ebv_draws[k].append(Ud.ravel().astype(np.float32))
                    s_theta += chn.theta
                    q_theta += chn.theta * chn.theta
                    s_uu += Ud[:, :, None] * Ud[:, None, :]
                    for c in cs:
                        s_tau[c] += chn.taus[c][1:P.catinfo[c]["K"]]
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
    gen_blocks = (s_uu / total - Um[:, :, None] * Um[:, None, :]) * total / max(total - 1, 1)
    pev_blocks = gen_blocks[:, :t, :t].copy()
    Uv = P.u(var)
    derived = {}
    for key in scal:
        if key.startswith(("rG_", "h2_", "c2_", "m2_")):
            v = np.array(store[key]).ravel()
            derived[key] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                            "q025": float(np.quantile(v, 0.025)),
                            "q975": float(np.quantile(v, 0.975))}
    acc_prop = sum(ch.n_prop for ch in chains)
    acc_n = sum(ch.n_acc for ch in chains)
    return MTThresholdGibbsResult(
        Um[:, :t].copy(), Uv[:, :t].copy(), pev_blocks,
        [mean[P.p_off[j]:P.p_off[j + 1]].copy() for j in range(t)],
        (s_tau[cs[0]] / total) if cs else np.zeros(0), P.cats,
        _summ(Ga), _summ(Ra), derived, summaries, ediag, bool(ok), it, n_saved,
        [int(s.generate_state(1)[0]) for s in child],
        (acc_n / acc_prop) if acc_prop else None, time.perf_counter() - t0, P.n_eq,
        traces={key: np.array(store[key]) for key in scal},
        P0=_summ(np.array(Ps)) if P.m else None,
        pe_mean=P.pe(mean).copy() if P.m else None,
        maternal_ebv=Um[:, t:].copy() if P.maternal else None,
        maternal_pev=Uv[:, t:].copy() if P.maternal else None,
        genetic_blocks=gen_blocks if P.maternal else None,
        thresholds_by_trait={c: s_tau[c] / total for c in cs},
        categories_by_trait={c: P.catinfo[c]["cats"] for c in cs})
