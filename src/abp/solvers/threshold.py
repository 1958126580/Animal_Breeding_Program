"""Threshold (ordered probit) animal model for categorical traits (method id ``threshold.probit``).

Model (Gianola & Foulley 1983, Genet Sel Evol 15:201; Harville & Mee 1984,
Biometrics 40:393)
--------------------------------------------------------------------------
An observed category ``y_i in {1..K}`` (ordered) arises from a normal
liability ``l_i = x_i'b + sum_k z_ik'u_k + e_i``, ``e_i ~ N(0, 1)`` (the residual
variance is fixed at 1 to identify the scale), with thresholds
``tau_0 = -inf < tau_1 < ... < tau_{K-1} < tau_K = +inf``::

    P(y_i = k) = Phi(tau_k - eta_i) - Phi(tau_{k-1} - eta_i),   eta_i = x_i'b + z_i'u.

With an intercept, ``tau_1 = 0`` (otherwise the intercept and the thresholds
are confounded).  Random terms ``u_k ~ N(0, sigma_k^2 K_k)`` on the liability
scale, with **known** variances.

Estimation: the joint posterior mode of ``(b, u, tau)`` (flat prior on ``b`` and
``tau``) maximises

    L = sum_i log P(y_i) - 1/2 sum_k u_k' K_k^{-1} u_k / sigma_k^2,

a strictly concave function (the ordered-probit log-likelihood is concave in
``(b, tau)``, Pratt 1981).  Newton-Raphson with step-halving (keeping the
thresholds ordered) is iterated until the largest step is below ``tol``.
With ``a_i = tau_{y_i - 1} - eta_i``, ``b_i = tau_{y_i} - eta_i`` and
``P_i = Phi(b_i) - Phi(a_i)``, the derivatives of ``log P_i`` are

    g_a = -phi(a)/P,  g_b = phi(b)/P,
    H_aa = a phi(a)/P - phi(a)^2/P^2,  H_bb = -b phi(b)/P - phi(b)^2/P^2,
    H_ab = phi(a) phi(b)/P^2,

(with ``phi(+-inf) = (+-inf) phi(+-inf) = 0``); the chain rule through
``a = tau_{y-1} - eta`` and ``b = tau_y - eta`` gives the gradient and Hessian in
``(eta, tau)``.  The Newton system is the MME with record weights
``w_i = -(H_aa + 2 H_ab + H_bb)`` in place of ``R^{-1}``, bordered by the
threshold equations.

Uncertainty: the inverse of the negative Hessian at the mode approximates the
posterior covariance (Laplace approximation); ``PEV`` and reliabilities
``1 - PEV / (sigma^2 K_ii)`` on the liability scale are therefore
**approximate** and labelled so.  Categories with no records, or all records
in one category, are refused (thresholds not identifiable).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.special import ndtr

from ..errors import ABPError
from .blup import RELIABILITY_ROUNDING_BAND, RandomTerm, TermResult
from .mme import DenseCholesky, make_sparse_factor

_SQRT2PI = np.sqrt(2.0 * np.pi)


def _phi(x):
    with np.errstate(over="ignore", invalid="ignore"):
        out = np.exp(-0.5 * x * x) / _SQRT2PI
    return np.where(np.isfinite(x), out, 0.0)


def _xphi(x):
    return np.where(np.isfinite(x), x * _phi(np.where(np.isfinite(x), x, 0.0)), 0.0)


def _log_p(a, b):
    """log(Phi(b) - Phi(a)) evaluated stably (upper-tail form when both are large)."""
    upper = (a > 0) & np.isfinite(a)
    p = np.where(upper, ndtr(-a) - ndtr(-np.where(np.isfinite(b), b, np.inf)),
                 ndtr(b) - ndtr(a))
    return np.log(np.maximum(p, 1e-300)), np.maximum(p, 1e-300)


@dataclass
class ThresholdResult:
    fixed_solution: np.ndarray
    thresholds: np.ndarray            # tau_1 .. tau_{K-1}
    categories: np.ndarray            # observed category values (ordered)
    terms: dict                       # name -> TermResult (liability scale)
    iterations: int
    log_posterior: float
    max_step: float
    n_equations: int
    solver: str


def threshold_blup(y: np.ndarray, X: sp.csr_matrix, terms: list[RandomTerm],
                   variances: dict[str, float], intercept: bool, tol: float = 1e-10,
                   max_iter: int = 100, compute_pev: bool = True,
                   memory_budget_bytes: int = 4 * 2**30) -> ThresholdResult:
    """Posterior mode of the ordered-probit animal model (see module notes)."""
    y = np.asarray(y, dtype=np.float64)
    if not np.all(np.isfinite(y)) or not np.all(y == np.round(y)):
        raise ABPError("SCHEMA_TYPE", "a threshold trait needs integer category codes")
    cats = np.unique(y)
    K = cats.size
    if K < 2:
        raise ABPError("MODEL_NOT_IDENTIFIABLE", "all records are in one category; the "
                       "threshold model has nothing to estimate")
    yk = np.searchsorted(cats, y)                   # 0..K-1
    n = y.size
    if variances.get("residual", 1.0) != 1.0:
        raise ABPError("SPEC_INVALID", "the threshold model fixes the residual variance of the "
                       "liability at 1; give the other variances on that scale")
    for t_ in terms:
        v = variances.get(t_.name)
        if v is None or not v > 0:
            raise ABPError("COVARIANCE_NOT_PD", f"variance of {t_.name!r} must be > 0")
    X = sp.csr_matrix(X)
    p = X.shape[1]
    W = sp.hstack([X] + [t_.Z for t_ in terms], format="csr")
    offs, pos = {}, p
    for t_ in terms:
        offs[t_.name] = (pos, pos + t_.q)
        pos += t_.q
    n_loc = pos
    # free thresholds: tau_1 fixed at 0 with an intercept
    first_free = 1 if intercept else 0
    n_tau = (K - 1) - first_free
    Pblocks = [(offs[t_.name][0], sp.csr_matrix(t_.k_inv) / variances[t_.name]) for t_ in terms]
    # starting values: thresholds from cumulative proportions, everything else 0
    cum = np.cumsum(np.bincount(yk, minlength=K))[:-1] / n
    tau = np.concatenate([[-np.inf], _probit(cum), [np.inf]])
    if intercept:
        shift = tau[1]
        tau[1:-1] -= shift
        b0 = np.zeros(p)
        b0[0] = -shift                              # intercept column is the first one
    else:
        b0 = np.zeros(p)
    theta = np.concatenate([b0, np.zeros(n_loc - p)])

    def unpack_tau(tfree):
        tt = tau.copy()
        tt[1 + first_free:K] = tfree
        return tt

    def objective(th, tt):
        eta = W @ th
        a = tt[yk] - eta
        b = tt[yk + 1] - eta
        lp, _ = _log_p(a, b)
        pen = 0.0
        for a0, Ki in Pblocks:
            u = th[a0:a0 + Ki.shape[0]]
            pen += float(u @ (Ki @ u))
        return float(lp.sum()) - 0.5 * pen

    def derivatives(th, tt):
        eta = W @ th
        a = tt[yk] - eta
        b = tt[yk + 1] - eta
        _, P = _log_p(a, b)
        pa, pb = _phi(np.where(np.isfinite(a), a, 0.0)) * np.isfinite(a), \
            _phi(np.where(np.isfinite(b), b, 0.0)) * np.isfinite(b)
        ga, gb = -pa / P, pb / P
        Haa = _xphi(a) / P - (pa / P) ** 2
        Hbb = -_xphi(b) / P - (pb / P) ** 2
        Hab = pa * pb / P ** 2
        g_eta = -(ga + gb)
        w = -(Haa + 2 * Hab + Hbb)                    # >= 0: minus d2/deta2
        # gradient and negative Hessian over (loc, tau_free)
        grad_loc = W.T @ g_eta
        for a0, Ki in Pblocks:
            u = th[a0:a0 + Ki.shape[0]]
            grad_loc[a0:a0 + Ki.shape[0]] -= Ki @ u
        N = (W.T @ sp.diags(w) @ W).tocsr()
        for a0, Ki in Pblocks:
            Kc = Ki.tocoo()
            N = N + sp.csr_matrix((Kc.data, (Kc.row + a0, Kc.col + a0)), shape=N.shape)
        # threshold rows: tau index j (1..K-1) appears as lower bound (a) for category j and as
        # upper bound (b) for category j-1 (0-based categories)
        g_tau = np.zeros(K + 1)
        np.add.at(g_tau, yk, ga)
        np.add.at(g_tau, yk + 1, gb)
        # d2/deta dtau_lower = -(Haa + Hab), d2/deta dtau_upper = -(Hab + Hbb)  (of logP)
        c_low = sp.csr_matrix((-(Haa + Hab), (np.arange(n), yk)), shape=(n, K + 1))
        c_up = sp.csr_matrix((-(Hab + Hbb), (np.arange(n), yk + 1)), shape=(n, K + 1))
        cross = (W.T @ (c_low + c_up))                # of logP; negative Hessian uses minus
        Ht = np.zeros((K + 1, K + 1))
        np.add.at(Ht, (yk, yk), Haa)
        np.add.at(Ht, (yk + 1, yk + 1), Hbb)
        np.add.at(Ht, (yk, yk + 1), Hab)
        np.add.at(Ht, (yk + 1, yk), Hab)
        free = np.arange(1 + first_free, K)
        grad = np.concatenate([grad_loc, g_tau[free]])
        Ncross = -np.asarray(cross[:, free].todense())
        Ntt = -Ht[np.ix_(free, free)]
        full = sp.bmat([[N, sp.csr_matrix(Ncross)], [sp.csr_matrix(Ncross.T), sp.csr_matrix(Ntt)]],
                       format="csr")
        return grad, full

    tfree = tau[1 + first_free:K].copy()
    obj = objective(theta, unpack_tau(tfree))
    it, step = 0, np.inf
    for it in range(1, max_iter + 1):
        grad, NH = derivatives(theta, unpack_tau(tfree))
        fac = _factor(NH, memory_budget_bytes)
        delta = fac.solve(grad)
        lam, accepted = 1.0, False
        for _ in range(40):
            th_n = theta + lam * delta[:n_loc]
            tf_n = tfree + lam * delta[n_loc:]
            tt = unpack_tau(tf_n)
            if np.all(np.diff(tt[1:K]) > 0):            # thresholds stay strictly ordered
                o = objective(th_n, tt)
                if o >= obj - 1e-12 * abs(obj):
                    accepted = True
                    break
            lam *= 0.5
        if not accepted:
            raise ABPError("SOLVER_NOT_CONVERGED", "threshold model: Newton step could not "
                           "increase the log posterior", iteration=it)
        step = float(np.max(np.abs(lam * delta)))
        theta, tfree, obj = th_n, tf_n, o
        if step < tol:
            break
    else:
        raise ABPError("SOLVER_NOT_CONVERGED", f"threshold model did not converge in {max_iter} "
                       f"Newton iterations (last step {step:.2e})", last_step=step)
    tt = unpack_tau(tfree)
    grad, NH = derivatives(theta, tt)
    fac = _factor(NH, memory_budget_bytes)
    out: dict = {}
    for t_ in terms:
        a0, b0_ = offs[t_.name]
        sol = theta[a0:b0_].copy()
        pev = rel = None
        n_cl = 0
        if compute_pev:
            idx = np.arange(a0, b0_)
            if isinstance(fac, DenseCholesky):
                pev = fac.inverse_diagonal(idx)
            else:
                pev = fac.selected_inverse().diagonal(idx)
            if t_.genetic:
                prior = variances[t_.name] * np.asarray(t_.k_diag, dtype=np.float64)
                rr = 1.0 - pev / prior
                bad = (rr < -RELIABILITY_ROUNDING_BAND) | (rr > 1 + RELIABILITY_ROUNDING_BAND)
                if np.any(bad):
                    raise ABPError("RELIABILITY_OUT_OF_RANGE",
                                   "threshold model: approximate reliability outside [0, 1]",
                                   n_bad=int(bad.sum()))
                rel = np.clip(rr, 0.0, 1.0)
                n_cl = int(np.sum(rel != rr))
        out[t_.name] = TermResult(t_.name, t_.labels, sol, pev, rel, n_cl)
    return ThresholdResult(theta[:p].copy(), tt[1:K].copy(), cats, out, it, obj,
                           step, n_loc + n_tau, "dense" if isinstance(fac, DenseCholesky)
                           else "sparse_direct")


def _probit(p):
    from scipy.special import ndtri
    return ndtri(np.clip(p, 1e-12, 1 - 1e-12))


def _factor(M: sp.csr_matrix, budget: int):
    n = M.shape[0]
    if n <= 12000 and 16 * n * n <= budget:
        return DenseCholesky(M.toarray())
    return make_sparse_factor(M, budget)
