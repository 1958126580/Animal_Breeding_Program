"""Multi-trait REML: genetic and residual covariance matrices (method id ``reml.multi_trait``).

Model (as :mod:`abp.solvers.multitrait`)
-----------------------------------------
``y`` stacks the observed traits of each record (record-major);
``Var(u) = K (x) G0`` (animal-major), ``Var(e_r) = R0[o_r, o_r]`` for the observed
set ``o_r`` of record ``r``; each trait has its own full-rank fixed design.
Parameters ``theta = (vech G0, vech R0)``; the direction of parameter
``(j, k)`` is ``E_jk = e_j e_k' + e_k e_j'`` (``j != k``) or ``e_j e_j'``.

Likelihood (Henderson's form, derived for this implementation and tested
against the marginal V-form)::

    -2 logL = sum_r log|R0[o_r,o_r]| + t log|K| + q log|G0| + log|C| + y'Py,
    y'Py    = y'R^{-1}y - s'W'R^{-1}y,

with ``C`` the unscaled MME coefficient matrix and ``s`` its solution (the
constant ``(n - p) log 2pi`` is dropped).  Two quantities of ``C^{-1}`` are
needed, both on the pattern of ``C`` (so sparse selected inversion provides
them at scale, :mod:`abp.solvers.selinv`):

* ``T_jk = sum_ab K^{-1}_ab C^{(a,j),(b,k)}`` (``t x t``);
* ``Q_r = W_r C^{-1} W_r'`` for the rows ``W_r`` of every record.

Scores: for a genetic direction ``E``
``dlogL = 1/2 [tr(G0^{-1} E G0^{-1} (U'K^{-1}U + T)) - q tr(E G0^{-1})]``; for a
residual direction ``E``
``dlogL = 1/2 sum_r [e_r'R_r^{-1}E_r R_r^{-1}e_r + tr(R_r^{-1}E_r R_r^{-1}Q_r) - tr(R_r^{-1}E_r)]``
(``E_r = E[o_r, o_r]``).  The average-information matrix is ``1/2 F'PF`` with
working variates ``f_E = Z vec(U M')``, ``M = E G0^{-1}`` (genetic) and
``f_E = D_E R^{-1} e`` (residual), computed with one multi-RHS solve
(Gilmour, Thompson & Cullis 1995; Johnson & Thompson 1995 for the
multivariate form).

EM update (used when an AI step fails, and as the first steps)::

    G0 <- (U'K^{-1}U + T) / q,
    R0 <- (1/n_rec) sum_r E[e_r e_r' | y],

where the unobserved residuals of a record are filled by their conditional
expectation given the observed ones, plus the conditional covariance
(missing traits are handled exactly; nothing is imputed into ``y``).  EM keeps
both matrices positive semi-definite.  AI proposals that are not positive
definite or that decrease logL are halved, then replaced by EM.

Convergence: relative parameter change < ``tol`` and Newton decrement
``g'AI^{-1}g < tol``.  Non-convergence raises ``REML_NOT_CONVERGED``; a
``G0`` or ``R0`` that becomes singular (a genetic or residual variance or
correlation at the boundary) raises ``MODEL_NOT_IDENTIFIABLE``.

Reduced-rank genetic covariance (method id ``reml.multi_trait_reduced_rank``)
---------------------------------------------------------------------------
When the optimum has a singular ``G0`` (a genetic correlation of +-1, or a
genetic variance that is a combination of the others), ``G0 = Lambda Lambda'``
with ``Lambda`` ``t x r`` lower trapezoidal (``Lambda_jk = 0`` for ``k > j``)
is estimated instead (Kirkpatrick & Meyer 2004, Genetics 168:2295; Meyer &
Kirkpatrick 2005, Genet Sel Evol 37:1).  With latent factors
``f ~ N(0, K (x) I_r)`` and ``u = (I_q (x) Lambda) f`` the model is an ordinary
mixed model in ``f``, and Henderson's form of the likelihood becomes

    -2 logL = sum_r log|R0[o_r,o_r]| + r log|K| + log|C_f| + y'Py

(``C_f`` the MME of the ``f`` model; derived here, tested against the V-form
with a singular ``G0``).  ``(Lambda, chol R0)`` are unconstrained parameters
(``chol R0`` with log-diagonal); ``-2 logL`` is minimised by L-BFGS with the
analytic gradient (round 6): for the loadings ``2 T - 2 V'F`` with
``T_ab = sum_i sum_c (C^-1)_{f(i,b), c} (W_f'R^-1 Z)_{c,(i,a)}``, ``V = Z'R^-1 e``,
``F`` the factor solutions (envelope theorem for ``y'Py`` plus
``d log|C_f| = 2 tr(C^-1 W_f' R^-1 dW_f)``); for ``R0`` the matrix gradient
``sum_P [n_P R_P^-1 - R_P^-1 S_P R_P^-1]`` with ``S_P = sum_r (e_r e_r' + Q_r)``,
chained through ``chol R0``.  ``C^-1`` is needed only on the pattern of ``C``
(dense inverse or sparse selected inversion), which is full when the gradient is
evaluated at the rotated loadings ``Lambda Q`` (``Q`` orthogonal: same ``G0``).
Sampling errors are not reported for this parameterisation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from ..errors import ABPError
from .mme import DenseCholesky, dense_bytes, make_sparse_factor
from .multitrait import MTData, assemble_multitrait, check_covariance

DENSE_MAX = 2000    #: dense trace path up to this many equations (example 13, 6,390 equations:
                    #: dense 44.2 s, sparse selected inversion 7.9 s, identical logL)
MIN_EIG_REL = 1e-8  #: smallest eigenvalue / largest of G0 or R0 accepted at convergence
NEWTON_DEC_TOL_RR = 1e-6  #: reduced rank: Newton decrement g'H^-1 g of -2 logL at convergence


def _vech_pairs(t: int) -> list[tuple[int, int]]:
    return [(j, k) for j in range(t) for k in range(j, t)]


def _direction(t: int, j: int, k: int) -> np.ndarray:
    E = np.zeros((t, t))
    E[j, k] = E[k, j] = 1.0
    return E


def _pack(G0, R0):
    t = G0.shape[0]
    pr = _vech_pairs(t)
    return np.array([G0[j, k] for j, k in pr] + [R0[j, k] for j, k in pr])


def _unpack(theta, t):
    pr = _vech_pairs(t)
    m = len(pr)
    G0, R0 = np.zeros((t, t)), np.zeros((t, t))
    for i, (j, k) in enumerate(pr):
        G0[j, k] = G0[k, j] = theta[i]
        R0[j, k] = R0[k, j] = theta[m + i]
    return G0, R0


def _is_pd(S) -> bool:
    try:
        np.linalg.cholesky(S)
        return True
    except np.linalg.LinAlgError:
        return False


@dataclass
class MTPoint:
    theta: np.ndarray
    loglik: float
    score: np.ndarray
    ai: np.ndarray
    em: np.ndarray


@dataclass
class MTREMLFit:
    G0: np.ndarray
    R0: np.ndarray
    loglik: float
    iterations: int
    status: str
    se: dict | None
    trace_method: str
    start: dict
    history: list = field(default_factory=list)
    cov: np.ndarray | None = None       # inverse AI matrix of (vech G0, vech R0)

    def to_dict(self, traits: list[str]) -> dict:
        def corr(S):
            d = np.sqrt(np.diag(S))
            return (S / np.outer(d, d)).tolist()
        h2 = (np.diag(self.G0) / (np.diag(self.G0) + np.diag(self.R0))).tolist()
        return {"status": self.status, "algorithm": "AI-REML with EM fallback (multi-trait)",
                "traits": traits, "G0": self.G0.tolist(), "R0": self.R0.tolist(),
                "genetic_correlations": corr(self.G0), "residual_correlations": corr(self.R0),
                "heritabilities": dict(zip(traits, h2)), "loglik": self.loglik,
                "iterations": self.iterations, "se": self.se, "trace_method": self.trace_method,
                "start": self.start, "history": self.history,
                "note": "REML estimates; SE are asymptotic (inverse average information)."}


class MTREMLEvaluator:
    """logL, score, AI and EM update of the multi-trait model at given ``(G0, R0)``."""

    def __init__(self, data: MTData, k_inv, logdet_k: float | None,
                 memory_budget_bytes: int = 4 * 2**30):
        if logdet_k is None:
            raise ABPError("UNSUPPORTED_COMBINATION", "multi-trait REML needs log|K| of the "
                           "relationship structure")
        self.data = data
        self.k_inv = sp.csr_matrix(k_inv) if not sp.issparse(k_inv) else k_inv.tocsr().copy()
        # stored zeros (exact cancellation in A^-1, e.g. a son mated to his dam) are not in the
        # pattern of C and contribute nothing to the traces: drop them
        self.k_inv.eliminate_zeros()
        self.logdet_k = float(logdet_k)
        self.budget = memory_budget_bytes
        Y = np.asarray(data.Y, dtype=np.float64)
        self.t = Y.shape[1]
        self.q = self.k_inv.shape[0]
        obs = ~np.isnan(Y)
        self.obs = obs[obs.any(axis=1)]
        self.n_rec = int(self.obs.shape[0])
        # structure (constant across iterations): fix it from a unit-covariance assembly
        eye = np.eye(self.t)
        mts = assemble_multitrait(data, self.k_inv, eye, eye)
        sysm = mts.system
        self.rec_idx, self.trait_idx = mts.rec_idx, mts.trait_idx
        self.n_obs = self.rec_idx.size
        self.p = sysm.p
        self.a0 = sysm.offsets["animal"][0]
        n_eq = sysm.n_equations
        if self.n_obs - self.p <= 0:
            raise ABPError("MODEL_NOT_IDENTIFIABLE", "no residual degrees of freedom")
        self.trace_method = ("dense_inverse" if n_eq <= DENSE_MAX
                             and dense_bytes(n_eq, True) <= memory_budget_bytes
                             else "sparse_selected_inversion")
        self.W = sysm.W.tocsr()
        self.y = sysm.y
        # record blocks: observation rows of each record (contiguous, record-major)
        self.rstart = np.searchsorted(self.rec_idx, np.arange(self.n_rec))
        self.rend = np.searchsorted(self.rec_idx, np.arange(self.n_rec), side="right")
        pat = [tuple(self.trait_idx[a:b]) for a, b in zip(self.rstart, self.rend)]
        self.patterns = sorted(set(pat))
        self.pat_of = np.array([self.patterns.index(x) for x in pat])
        # pairs for Q_r = W_r C^-1 W_r': (obs i, obs j) of the same record x nonzeros
        Wc = self.W.tocsr()
        pi, pj, pa, pb, pw = [], [], [], [], []
        for a, b in zip(self.rstart, self.rend):
            rows = np.arange(a, b)
            ii, jj = np.meshgrid(rows, rows, indexing="ij")
            for i, j in zip(ii.ravel(), jj.ravel()):
                ci = Wc.indices[Wc.indptr[i]:Wc.indptr[i + 1]]
                vi = Wc.data[Wc.indptr[i]:Wc.indptr[i + 1]]
                cj = Wc.indices[Wc.indptr[j]:Wc.indptr[j + 1]]
                vj = Wc.data[Wc.indptr[j]:Wc.indptr[j + 1]]
                A, B = np.meshgrid(ci, cj, indexing="ij")
                pa.append(A.ravel())
                pb.append(B.ravel())
                pw.append(np.outer(vi, vj).ravel())
                pi.append(np.full(A.size, i))
                pj.append(np.full(A.size, j))
        nz = np.concatenate(pw) != 0.0
        pi, pj, pa, pb, pw = ([np.concatenate(x)[nz]] for x in (pi, pj, pa, pb, pw))
        self.q_i = np.concatenate(pi)
        self.q_j = np.concatenate(pj)
        self.q_a = np.concatenate(pa)
        self.q_b = np.concatenate(pb)
        self.q_w = np.concatenate(pw)
        # pairs for T: K^-1 nonzeros x (j, k)
        Kc = self.k_inv.tocoo()
        t = self.t
        jj, kk = np.meshgrid(np.arange(t), np.arange(t), indexing="ij")
        jj, kk = jj.ravel(), kk.ravel()
        self.t_rows = (self.a0 + Kc.row[:, None] * t + jj[None, :]).ravel()
        self.t_cols = (self.a0 + Kc.col[:, None] * t + kk[None, :]).ravel()
        self.t_w = np.repeat(Kc.data, t * t)
        self.t_jk = np.tile(jj * t + kk, Kc.nnz)
        self.Z = self.W[:, self.a0:self.a0 + self.q * t]

    # -------------------------------------------------------------- evaluate
    def evaluate(self, theta: np.ndarray, with_ai: bool = True) -> MTPoint:
        t, q = self.t, self.q
        G0, R0 = _unpack(theta, t)
        mts = assemble_multitrait(self.data, self.k_inv, G0, R0)
        sysm = mts.system
        dense = self.trace_method == "dense_inverse"
        fac = DenseCholesky(sysm.C.toarray()) if dense else make_sparse_factor(sysm.C, self.budget)
        s = fac.solve(sysm.rhs)
        if dense:
            Cinv = fac.inverse()
            ent = lambda r, c: Cinv[r, c]   # noqa: E731
        else:
            si = fac.selected_inverse()
            ent = si.entries
        e = self.y - self.W @ s
        Rinv_e = sysm.rinv @ e
        ypy = float(self.y @ (sysm.rinv @ self.y)) - float(s @ sysm.rhs)
        logdet_r = 0.0
        pat_inv = []
        for P in self.patterns:
            Rp = R0[np.ix_(P, P)]
            pat_inv.append(np.linalg.inv(Rp))
            logdet_r += np.linalg.slogdet(Rp)[1] * int(np.sum(self.pat_of == self.patterns.index(P)))
        m2ll = logdet_r + t * self.logdet_k + q * np.linalg.slogdet(G0)[1] + fac.logdet() + ypy
        U = s[self.a0:self.a0 + q * t].reshape(q, t)
        KU = self.k_inv @ U
        UKU = U.T @ KU
        Tflat = np.bincount(self.t_jk, weights=self.t_w * ent(self.t_rows, self.t_cols),
                            minlength=t * t)
        T = Tflat.reshape(t, t)
        T = 0.5 * (T + T.T)
        Qv = self.q_w * ent(self.q_a, self.q_b)
        Gi = np.linalg.inv(G0)
        pairs = _vech_pairs(t)
        m = len(pairs)
        score = np.zeros(2 * m)
        Sg = UKU + T
        for i, (j, k) in enumerate(pairs):
            E = _direction(t, j, k)
            score[i] = 0.5 * (np.trace(Gi @ E @ Gi @ Sg) - q * np.trace(E @ Gi))
        # per-record Q blocks (records x t x t, by position within the record)
        local_i = self.q_i - self.rstart[self.rec_idx[self.q_i]]
        local_j = self.q_j - self.rstart[self.rec_idx[self.q_j]]
        rec_of_pair = self.rec_idx[self.q_i]
        maxk = t
        key = (rec_of_pair * maxk + local_i) * maxk + local_j
        Qflat = np.bincount(key, weights=Qv, minlength=self.n_rec * maxk * maxk)
        Qr = Qflat.reshape(self.n_rec, maxk, maxk)
        Rnew = np.zeros((t, t))
        rscore = np.zeros(m)
        for pidx, P in enumerate(self.patterns):
            recs = np.flatnonzero(self.pat_of == pidx)
            k_ = len(P)
            Ri = pat_inv[pidx]
            Pl = list(P)
            miss = [x for x in range(t) if x not in Pl]
            # observed residuals of these records, k_ columns
            starts = self.rstart[recs]
            Eo = np.stack([e[starts + c] for c in range(k_)], axis=1)       # nrec x k_
            Qb = Qr[recs][:, :k_, :k_]
            S_oo = Eo.T @ Eo + Qb.sum(axis=0)                                # sum over records
            Rio = Ri
            for i, (j, k) in enumerate(pairs):     # residual scores
                if j in Pl and k in Pl:
                    Eloc = _direction(t, j, k)[np.ix_(Pl, Pl)]
                    M = Rio @ Eloc @ Rio
                    rscore[i] += 0.5 * (np.trace(M @ S_oo) - recs.size * np.trace(Rio @ Eloc))
            # EM: full-residual second moment
            full = np.zeros((t, t))
            full[np.ix_(Pl, Pl)] = S_oo
            if miss:
                Rmo = R0[np.ix_(miss, Pl)]
                B = Rmo @ Rio
                full[np.ix_(miss, Pl)] = B @ S_oo
                full[np.ix_(Pl, miss)] = (B @ S_oo).T
                full[np.ix_(miss, miss)] = B @ S_oo @ B.T + recs.size * (
                    R0[np.ix_(miss, miss)] - B @ Rmo.T)
            Rnew += full
        score[m:] = rscore
        Gnew = Sg / q
        Rnew /= self.n_rec
        em = _pack(0.5 * (Gnew + Gnew.T), 0.5 * (Rnew + Rnew.T))
        ai = np.zeros((2 * m, 2 * m))
        if with_ai:
            F = np.empty((self.n_obs, 2 * m))
            for i, (j, k) in enumerate(pairs):
                Mg = _direction(t, j, k) @ Gi
                F[:, i] = self.Z @ (U @ Mg.T).ravel()
                F[:, m + i] = self._apply_block(_direction(t, j, k), Rinv_e)
            RF = sysm.rinv @ F
            Sx = fac.solve(self.W.T @ RF)
            PF = RF - sysm.rinv @ (self.W @ Sx)
            ai = 0.5 * (F.T @ PF)
            ai = 0.5 * (ai + ai.T)
        return MTPoint(np.asarray(theta, dtype=np.float64), -0.5 * m2ll, score, ai, em)

    def _apply_block(self, E: np.ndarray, v: np.ndarray) -> np.ndarray:
        """``(D_E v)_obs`` = sum over observed traits k of the same record of E[trait, k] v_k."""
        out = np.zeros(self.n_obs)
        tr = self.trait_idx
        for a_rel in range(self.t):
            for b_rel in range(self.t):
                ia = self.rstart + a_rel
                ib = self.rstart + b_rel
                ok = (ia < self.rend) & (ib < self.rend)
                ia, ib = ia[ok], ib[ok]
                out[ia] += E[tr[ia], tr[ib]] * v[ib]
        return out


def default_start(data: MTData) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic start: phenotypic covariance of complete pairs split 1/3 : 2/3."""
    Y = np.asarray(data.Y, dtype=np.float64)
    t = Y.shape[1]
    P = np.zeros((t, t))
    for j in range(t):
        for k in range(t):
            ok = ~np.isnan(Y[:, j]) & ~np.isnan(Y[:, k])
            if ok.sum() > 2:
                P[j, k] = np.cov(Y[ok, j], Y[ok, k])[0, 1]
    d = np.sqrt(np.maximum(np.diag(P), 1e-12))
    off = np.clip(P / np.outer(d, d), -0.9, 0.9) * (1.0 - np.eye(t))
    S = (np.eye(t) + 0.5 * off) * np.outer(d, d)     # halved correlations: safely PD
    return S / 3.0, 2.0 * S / 3.0


def mt_reml_fit(data: MTData, k_inv, logdet_k: float | None, cfg: dict,
                memory_budget_bytes: int = 4 * 2**30, start=None) -> MTREMLFit:
    """Fit ``G0`` and ``R0`` by AI-REML with EM fallback (see module notes)."""
    ev = MTREMLEvaluator(data, k_inv, logdet_k, memory_budget_bytes)
    t = ev.t
    if start is None:
        G0, R0 = default_start(data)
        source = "data-based heuristic (phenotypic covariance split 1/3 genetic, 2/3 residual)"
    else:
        G0, R0 = (check_covariance(np.asarray(x), n) for x, n in zip(start, ("G0 start", "R0 start")))
        source = "user"
    theta = _pack(G0, R0)
    tol, max_iter = float(cfg["tol"]), int(cfg["max_iter"])
    history: list[dict] = []
    point = ev.evaluate(theta)
    n_em_first = 3
    pd_rejected: list[bool] = []          # was the AI proposal outside the PD region?
    for it in range(1, max_iter + 1):
        step_kind = "em"
        new = None
        rejected_pd = False
        if it > n_em_first:
            try:
                delta = np.linalg.solve(point.ai, point.score)
            except np.linalg.LinAlgError:
                delta = None
            if delta is not None:
                lam = 1.0
                for _ in range(6):
                    cand = point.theta + lam * delta
                    Gc, Rc = _unpack(cand, t)
                    if not (_is_pd(Gc) and _is_pd(Rc)):
                        rejected_pd = rejected_pd or lam == 1.0
                    if _is_pd(Gc) and _is_pd(Rc):
                        pc = ev.evaluate(cand)
                        if pc.loglik >= point.loglik - 1e-10 * abs(point.loglik):
                            new, step_kind = pc, f"ai(step {lam:g})"
                            break
                    lam *= 0.5
        if new is None:
            new = ev.evaluate(point.em)
            if new.loglik < point.loglik - 1e-8 * max(1.0, abs(point.loglik)):
                raise ABPError("REML_NOT_CONVERGED", "multi-trait EM step decreased the "
                               "likelihood (numerical problem)", iteration=it)
        rel = float(np.max(np.abs(new.theta - point.theta)) / max(np.max(np.abs(new.theta)), 1e-300))
        try:
            dec = float(new.score @ np.linalg.solve(new.ai, new.score))
        except np.linalg.LinAlgError:
            dec = float("inf")
        pd_rejected.append(it > n_em_first and rejected_pd)
        history.append({"iteration": it, "step": step_kind, "loglik": new.loglik,
                        "rel_change": rel, "newton_decrement": dec,
                        "ai_step_outside_pd_region": pd_rejected[-1]})
        point = new
        if rel < tol and abs(dec) < tol:
            break
    else:
        G_, R_ = _unpack(point.theta, t)
        ratios = {w: float(np.linalg.eigvalsh(S)[0] / np.linalg.eigvalsh(S)[-1])
                  for w, S in (("G0", G_), ("R0", R_))}
        recent = pd_rejected[-20:]
        if recent and sum(recent) >= len(recent) / 2:
            raise ABPError("MODEL_NOT_IDENTIFIABLE",
                           "multi-trait REML: the likelihood keeps increasing towards the "
                           "boundary of the parameter space (full-length AI steps leave the "
                           "positive-definite region; smallest/largest eigenvalue "
                           f"G0 {ratios['G0']:.3g}, R0 {ratios['R0']:.3g}): a correlation tends "
                           "to +-1 or a variance to 0, which this version does not estimate; "
                           "give known covariances or drop a trait",
                           eigenvalue_ratios=ratios, history=history[-5:])
        raise ABPError("REML_NOT_CONVERGED", f"multi-trait REML did not converge in {max_iter} "
                       "iterations", history=history[-5:])
    G0, R0 = _unpack(point.theta, t)
    for S, what in ((G0, "G0"), (R0, "R0")):
        ev_ = np.linalg.eigvalsh(S)
        if ev_[0] <= MIN_EIG_REL * ev_[-1]:
            raise ABPError("MODEL_NOT_IDENTIFIABLE",
                           f"the estimated {what} is (nearly) singular (smallest eigenvalue "
                           f"{ev_[0]:.3e}); a variance or correlation is at the boundary, which "
                           "multi-trait REML does not handle in this version",
                           matrix=what, eigenvalues=ev_.tolist())
    se = None
    cov = None
    try:
        cov = np.linalg.inv(point.ai)
        cov = 0.5 * (cov + cov.T)
        names = [f"G0[{j},{k}]" for j, k in _vech_pairs(t)] + [f"R0[{j},{k}]" for j, k in _vech_pairs(t)]
        se = {n: float(math.sqrt(cov[i, i])) if cov[i, i] > 0 else None for i, n in enumerate(names)}
        if np.linalg.eigvalsh(cov)[0] <= 0:
            cov = None
    except np.linalg.LinAlgError:
        pass
    return MTREMLFit(G0, R0, point.loglik, len(history), "converged", se, ev.trace_method,
                     {"source": source, "G0": _unpack(theta, t)[0].tolist(),
                      "R0": _unpack(theta, t)[1].tolist()}, history, cov)


# ---------------------------------------------------------------- reduced rank
def _lower_trapezoid(t: int, r: int) -> list[tuple[int, int]]:
    return [(j, k) for j in range(t) for k in range(min(j + 1, r))]


def _rr_unpack(x: np.ndarray, t: int, r: int) -> tuple[np.ndarray, np.ndarray]:
    lt = _lower_trapezoid(t, r)
    Lam = np.zeros((t, r))
    for v, (j, k) in zip(x[:len(lt)], lt):
        Lam[j, k] = v
    L = np.zeros((t, t))
    for v, (j, k) in zip(x[len(lt):], _lower_trapezoid(t, t)):
        L[j, k] = np.exp(v) if j == k else v
    return Lam, L @ L.T


def _rr_pack(Lam: np.ndarray, R0: np.ndarray) -> np.ndarray:
    t, r = Lam.shape
    # rotate Lambda to lower-trapezoidal form (G0 = Lambda Lambda' is unchanged)
    Qm, Rm = np.linalg.qr(Lam.T)
    Lt = Rm.T
    L = np.linalg.cholesky(R0)
    return np.array([Lt[j, k] for j, k in _lower_trapezoid(t, r)]
                    + [np.log(L[j, k]) if j == k else L[j, k] for j, k in _lower_trapezoid(t, t)])


class ReducedRankEvaluator:
    """``-2 logL`` of the reduced-rank model ``G0 = Lambda Lambda'`` (see module notes)."""

    def __init__(self, data: MTData, k_inv, logdet_k: float | None, rank: int,
                 memory_budget_bytes: int = 4 * 2**30):
        if logdet_k is None:
            raise ABPError("UNSUPPORTED_COMBINATION", "multi-trait REML needs log|K| of the "
                           "relationship structure")
        self.data = data
        self.k_inv = sp.csr_matrix(k_inv) if not sp.issparse(k_inv) else k_inv.tocsr().copy()
        self.logdet_k = float(logdet_k)
        self.t = int(np.asarray(data.Y).shape[1])
        if not 1 <= rank <= self.t:
            raise ABPError("SPEC_INVALID", f"reduced rank must be between 1 and {self.t}")
        self.r = int(rank)
        self.budget = memory_budget_bytes
        self.n_eval = 0

    def m2ll(self, Lam: np.ndarray, R0: np.ndarray) -> tuple[float, object]:
        self.n_eval += 1
        mts = assemble_multitrait(self.data, self.k_inv, None, R0, loadings=Lam)
        sysm = mts.system
        n_eq = sysm.n_equations
        dense = n_eq <= DENSE_MAX and dense_bytes(n_eq, False) <= self.budget
        fac = DenseCholesky(sysm.C.toarray()) if dense else make_sparse_factor(sysm.C, self.budget)
        s = fac.solve(sysm.rhs)
        ypy = float(sysm.y @ (sysm.rinv @ sysm.y)) - float(s @ sysm.rhs)
        tr = mts.trait_idx
        starts = np.searchsorted(mts.rec_idx, np.arange(mts.n_rec))
        ends = np.searchsorted(mts.rec_idx, np.arange(mts.n_rec), side="right")
        cache: dict[tuple, float] = {}
        logdet_r = 0.0
        for a, b in zip(starts, ends):
            if a == b:
                continue
            key = tuple(tr[a:b])
            if key not in cache:
                cache[key] = float(np.linalg.slogdet(R0[np.ix_(key, key)])[1])
            logdet_r += cache[key]
        return logdet_r + self.r * self.logdet_k + fac.logdet() + ypy, fac

    def value_and_gradient(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        """``-2 logL`` and its analytic gradient in the parameters ``x`` (see module
        notes: envelope theorem for ``y'Py``, ``d log|C_f| = 2 tr(C^-1 W_f'R^-1 dW_f)``,
        residual scores from ``Q_r = W_r C^-1 W_r'``; ``C^-1`` only on the pattern of
        ``C``, evaluated at the rotated loadings ``Lambda Q`` so that the pattern is full)."""
        t, r = self.t, self.r
        Lam, R0 = _rr_unpack(x, t, r)
        Qrot = self._rotation()
        Lt = Lam @ Qrot                                     # same G0, no structural zeros
        mts = assemble_multitrait(self.data, self.k_inv, None, R0, loadings=Lt)
        sysm = mts.system
        # sparse factor + selected inversion: only entries of C^-1 on the pattern of C are
        # needed (a dense inverse would cost O(n^3) per evaluation)
        fac = make_sparse_factor(sysm.C, self.budget)
        s = fac.solve(sysm.rhs)
        self.n_eval += 1
        ent = fac.selected_inverse().entries
        ypy = float(sysm.y @ (sysm.rinv @ sysm.y)) - float(s @ sysm.rhs)
        tr_idx, rec_idx = mts.trait_idx, mts.rec_idx
        n_rec = mts.n_rec
        starts = np.searchsorted(rec_idx, np.arange(n_rec))
        ends = np.searchsorted(rec_idx, np.arange(n_rec), side="right")
        pat = [tuple(tr_idx[a:b]) for a, b in zip(starts, ends)]
        patterns = sorted(set(pat))
        pat_of = np.array([patterns.index(p_) for p_ in pat])
        logdet_r = 0.0
        pinv = []
        for pidx, P_ in enumerate(patterns):
            Rp = R0[np.ix_(P_, P_)]
            pinv.append(np.linalg.inv(Rp))
            logdet_r += float(np.linalg.slogdet(Rp)[1]) * int(np.sum(pat_of == pidx))
        f_val = logdet_r + r * self.logdet_k + fac.logdet() + ypy
        # --- loadings: 2 T - 2 V'F (wrt Lambda~), then back to Lambda
        W = sysm.W.tocsr()
        a0 = sysm.offsets["animal"][0]
        q = self.k_inv.shape[0]
        n_obs = rec_idx.size
        Zo = sp.csr_matrix((np.ones(n_obs), (np.arange(n_obs),
                            np.asarray(self.data.animal_col)[rec_idx] * t + tr_idx)),
                           shape=(n_obs, q * t))
        e = sysm.y - W @ s
        Re = sysm.rinv @ e
        V = (Zo.T @ Re).reshape(q, t)
        F = s[a0:a0 + q * r].reshape(q, r)
        M = (W.T @ (sysm.rinv @ Zo)).tocoo()
        i_an, a_tr = M.col // t, M.col % t
        T = np.zeros((t, r))
        for b in range(r):
            vals = ent(a0 + i_an * r + b, M.row) * M.data
            T[:, b] = np.bincount(a_tr, weights=vals, minlength=t)
        g_lam = (2.0 * T - 2.0 * V.T @ F) @ Qrot.T
        # --- residual covariance: scores for symmetric directions, as in MTREMLEvaluator
        qi, qj, qa, qb, pa, pb = self._q_pairs(W, starts, ends, rec_idx)
        Qv = W.data[pa] * W.data[pb] * ent(qa, qb)
        li = qi - starts[rec_idx[qi]]
        lj = qj - starts[rec_idx[qj]]
        key = (rec_idx[qi] * t + li) * t + lj
        Qr = np.bincount(key, weights=Qv, minlength=n_rec * t * t).reshape(n_rec, t, t)
        Gam = np.zeros((t, t))                 # d(-2 logL)/dR0 as a symmetric matrix gradient
        for pidx, P_ in enumerate(patterns):
            recs = np.flatnonzero(pat_of == pidx)
            k_ = len(P_)
            Ri = pinv[pidx]
            Eo = np.stack([e[starts[recs] + c] for c in range(k_)], axis=1)
            S_oo = Eo.T @ Eo + Qr[recs][:, :k_, :k_].sum(axis=0)
            # d(-2logL)/dR_P = n R^-1 - R^-1 S R^-1   (restricted to the observed block)
            Gam[np.ix_(P_, P_)] += recs.size * Ri - Ri @ S_oo @ Ri
        Lr = np.linalg.cholesky(R0)
        g_L = 2.0 * Gam @ Lr
        grad = [g_lam[j, k] for j, k in _lower_trapezoid(t, r)]
        grad += [g_L[j, k] * (Lr[j, j] if j == k else 1.0) for j, k in _lower_trapezoid(t, t)]
        return float(f_val), np.array(grad)

    def _rotation(self) -> np.ndarray:
        if not hasattr(self, "_Q"):
            A = np.random.default_rng(12345).standard_normal((self.r, self.r))
            self._Q = np.linalg.qr(A)[0]
        return self._Q

    def _q_pairs(self, W, starts, ends, rec_idx):
        """Index arrays for ``Q_r``: pairs of observations of the same record x their
        non-zeros (positions into ``W.data``); cached while the pattern is unchanged."""
        sig = (W.indptr.tobytes(), W.indices.tobytes())
        if getattr(self, "_qsig", None) == sig:
            return self._qcache
        qi, qj, qa, qb, pa, pb = [], [], [], [], [], []
        for a, b in zip(starts, ends):
            for i in range(a, b):
                pi_ = np.arange(W.indptr[i], W.indptr[i + 1])
                for j in range(a, b):
                    pj_ = np.arange(W.indptr[j], W.indptr[j + 1])
                    A_, B_ = np.meshgrid(pi_, pj_, indexing="ij")
                    pa.append(A_.ravel())
                    pb.append(B_.ravel())
                    qi.append(np.full(A_.size, i))
                    qj.append(np.full(A_.size, j))
        pa, pb = np.concatenate(pa), np.concatenate(pb)
        qi, qj = np.concatenate(qi), np.concatenate(qj)
        qa, qb = W.indices[pa], W.indices[pb]
        self._qsig = sig
        self._qcache = (qi, qj, qa, qb, pa, pb)
        return self._qcache

    def objective(self, x: np.ndarray) -> float:
        Lam, R0 = _rr_unpack(x, self.t, self.r)
        try:
            return self.m2ll(Lam, R0)[0]
        except (ABPError, np.linalg.LinAlgError):
            return float("inf")


@dataclass
class ReducedRankFit:
    loadings: np.ndarray
    G0: np.ndarray
    R0: np.ndarray
    rank: int
    loglik: float
    evaluations: int
    status: str
    gradient_norm: float
    start: dict
    newton_decrement: float | None = None

    def to_dict(self, traits: list[str]) -> dict:
        def corr(S):
            d = np.sqrt(np.diag(S))
            return (S / np.outer(d, d)).tolist()
        h2 = (np.diag(self.G0) / (np.diag(self.G0) + np.diag(self.R0))).tolist()
        return {"status": self.status,
                "algorithm": f"reduced-rank REML, G0 = Lambda Lambda' of rank {self.rank} "
                             "(L-BFGS, analytic gradient)",
                "traits": traits, "G0": self.G0.tolist(), "R0": self.R0.tolist(),
                "loadings": self.loadings.tolist(), "rank": self.rank,
                "genetic_correlations": corr(self.G0), "residual_correlations": corr(self.R0),
                "heritabilities": dict(zip(traits, h2)), "loglik": self.loglik,
                "iterations": self.evaluations, "se": None, "gradient_norm": self.gradient_norm,
                "newton_decrement": self.newton_decrement,
                "start": self.start, "trace_method": "none (derivative-free likelihood)",
                "note": "REML estimates under a reduced-rank genetic covariance matrix; "
                        "the rank is an assumption, sampling errors are not reported."}


def mt_reml_fit_reduced_rank(data: MTData, k_inv, logdet_k: float | None, rank: int,
                             cfg: dict, memory_budget_bytes: int = 4 * 2**30,
                             start=None) -> ReducedRankFit:
    """REML with ``G0 = Lambda Lambda'`` of rank ``rank`` (see module notes)."""
    from scipy.optimize import minimize
    ev = ReducedRankEvaluator(data, k_inv, logdet_k, rank, memory_budget_bytes)
    t, r = ev.t, ev.r
    if start is None:
        G0s, R0s = default_start(data)
        source = "data-based heuristic, G0 truncated to its leading eigenvectors"
    else:
        G0s, R0s = (np.asarray(x, dtype=np.float64) for x in start)
        source = "user"
    w, V = np.linalg.eigh(0.5 * (G0s + G0s.T))
    idx = np.argsort(w)[::-1][:r]
    Lam0 = V[:, idx] * np.sqrt(np.maximum(w[idx], 1e-6 * max(w.max(), 1e-12)))
    x0 = _rr_pack(Lam0, check_covariance(R0s, "R0 start"))
    tol = float(cfg["tol"])

    def fg(x):
        try:
            return ev.value_and_gradient(x)
        except (ABPError, np.linalg.LinAlgError):
            return float("inf"), np.zeros_like(x)

    res = minimize(fg, x0, jac=True, method="L-BFGS-B",
                   options={"maxiter": int(cfg["max_iter"]), "gtol": 1e-7, "ftol": 1e-15})
    x = res.x
    f, g = fg(x)
    gnorm = float(np.max(np.abs(g)))
    # convergence: Newton decrement g'H^-1 g (predicted remaining decrease of -2 logL), with
    # H from central differences of the analytic gradient - scale-free, unlike |g|
    dec = float("inf")
    if np.isfinite(f):
        hh = 1e-5
        H = np.empty((x.size, x.size))
        for i in range(x.size):
            e = np.zeros_like(x)
            e[i] = hh
            H[:, i] = (fg(x + e)[1] - fg(x - e)[1]) / (2 * hh)
        H = 0.5 * (H + H.T)
        try:
            np.linalg.cholesky(H)
            dec = float(g @ np.linalg.solve(H, g))
        except np.linalg.LinAlgError:
            dec = float("inf")
    if not np.isfinite(f) or not dec <= NEWTON_DEC_TOL_RR:
        raise ABPError("REML_NOT_CONVERGED", f"reduced-rank multi-trait REML (rank {r}) did "
                       f"not converge (Newton decrement {dec:.2e}, largest gradient element "
                       f"{gnorm:.2e}; {res.message})", gradient=gnorm, newton_decrement=dec)
    Lam, R0 = _rr_unpack(x, t, r)
    ev_r = np.linalg.eigvalsh(R0)
    if ev_r[0] <= MIN_EIG_REL * ev_r[-1]:
        raise ABPError("MODEL_NOT_IDENTIFIABLE", "reduced-rank multi-trait REML: the residual "
                       "covariance matrix is (nearly) singular; reducing the rank of G0 does "
                       "not resolve this boundary", eigenvalues=ev_r.tolist())
    sv = np.linalg.svd(Lam, compute_uv=False)
    if sv[-1] <= 1e-6 * sv[0]:
        raise ABPError("MODEL_NOT_IDENTIFIABLE", f"reduced-rank multi-trait REML: rank {r} is "
                       f"still too high (the loadings have rank < {r}); use a lower rank",
                       singular_values=sv.tolist())
    Lam0f, R0f = _rr_unpack(x0, t, r)
    return ReducedRankFit(Lam, Lam @ Lam.T, R0, r, -0.5 * f, ev.n_eval, "converged", gnorm,
                          {"source": source, "G0": (Lam0f @ Lam0f.T).tolist(),
                           "R0": R0f.tolist()}, dec)
