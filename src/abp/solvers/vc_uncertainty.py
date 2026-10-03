"""PEV including the uncertainty of estimated variance components (method id ``blup.pev_vc``).

BLUP computed with REML estimates ``theta_hat`` has prediction error
``u_hat(theta_hat) - u = [u_hat(theta) - u] + [u_hat(theta_hat) - u_hat(theta)]``.
The usual PEV (``C^ii``) covers only the first part.  Kackar & Harville (1984,
J Am Stat Assoc 79:853) showed that, to first order, the second part adds

    Delta_i = g_i' Sigma_theta g_i,     g_i = d u_hat_i / d theta,

with ``Sigma_theta`` the asymptotic covariance of the REML estimates (the
inverse average-information matrix).  ABP evaluates ``g`` by central
differences of the BLUP solution at ``theta_hat`` (two extra solves per
variance component, relative step ``1e-4``) and reports

    PEV_total_i = PEV_i + Delta_i,   reliability_total_i = 1 - PEV_total_i / (sigma_hat^2 K_ii)

in addition to the conditional values, never instead of them.  The
correction is a first-order (delta-method) approximation; it is not computed
at a REML boundary, where ``Sigma_theta`` is not valid.
"""

from __future__ import annotations

import numpy as np

from .blup import blup

REL_STEP = 1e-4


def kackar_harville_delta(y, X, terms, variances: dict[str, float], cov: np.ndarray,
                          names: list[str], genetic_term: str, method: str = "auto",
                          memory_budget_bytes: int = 4 * 2**30,
                          factorization: str = "auto") -> np.ndarray:
    """``Delta_i`` for every equation of ``genetic_term`` (see module notes)."""
    from .reml import SPARSE_REML_ABOVE, sparse_structures
    n_eq = X.shape[1] + sum(t.q for t in terms)
    if method == "auto" and n_eq > SPARSE_REML_ABOVE and sparse_structures(terms):
        method = "sparse_direct"       # solutions only: the sparse factor beats dense Cholesky
    p = len(names)
    grads = []
    for k in range(p):
        h = REL_STEP * variances[names[k]]
        sols = []
        for sign in (1.0, -1.0):
            vc = dict(variances)
            vc[names[k]] = variances[names[k]] + sign * h
            r = blup(y, X, terms, vc, method=method, compute_pev=False,
                     memory_budget_bytes=memory_budget_bytes, factorization=factorization)
            sols.append(r.terms[genetic_term].solution)
        grads.append((sols[0] - sols[1]) / (2.0 * h))
    Gm = np.column_stack(grads)                      # q x p
    return np.einsum("ik,kl,il->i", Gm, cov, Gm)


def kackar_harville_delta_multitrait(data, k_inv, k_diag, G0: np.ndarray, R0: np.ndarray,
                                    cov: np.ndarray, method: str = "auto",
                                    memory_budget_bytes: int = 4 * 2**30,
                                    factorization: str = "auto") -> np.ndarray:
    """Multi-trait version: ``Delta_i = J_i Sigma J_i'`` (``t x t`` per animal) with
    ``J_i = d u_hat_i / d theta``, ``theta = (vech G0, vech R0)`` in the order of
    :mod:`abp.solvers.multitrait_reml` and ``Sigma`` the inverse average-information
    matrix.  Central differences with a step ``REL_STEP * sqrt(S_jj S_kk)`` on the
    symmetric pair ``(j, k)`` of ``S = G0`` or ``R0``."""
    from .multitrait import build_and_solve
    if method == "auto":            # solutions only: the sparse factor beats dense Cholesky here
        method = "sparse_direct"
    t = G0.shape[0]
    pairs = [(j, k) for j in range(t) for k in range(j, t)]
    grads = []
    for which in ("G", "R"):
        S0 = G0 if which == "G" else R0
        for j, k in pairs:
            h = REL_STEP * float(np.sqrt(S0[j, j] * S0[k, k]))
            sols = []
            for sign in (1.0, -1.0):
                S = S0.copy()
                S[j, k] += sign * h
                if j != k:
                    S[k, j] += sign * h
                Gp, Rp = (S, R0) if which == "G" else (G0, S)
                r = build_and_solve(data, k_inv, k_diag, Gp, Rp, method=method,
                                    compute_pev=False, memory_budget_bytes=memory_budget_bytes,
                                    factorization=factorization)
                sols.append(r.ebv)
            grads.append((sols[0] - sols[1]) / (2.0 * h))
    J = np.stack(grads, axis=2)                     # q x t x n_par
    return np.einsum("iak,kl,ibl->iab", J, cov, J)


def kackar_harville_delta_reduced_rank(data, k_inv, k_diag, x: np.ndarray, t: int, r: int,
                                      cov_x: np.ndarray, method: str = "auto",
                                      memory_budget_bytes: int = 4 * 2**30,
                                      factorization: str = "auto") -> np.ndarray:
    """Reduced-rank version: ``Delta_i = J_i Sigma_x J_i'`` with ``x`` the parameters of
    :func:`abp.solvers.multitrait_reml.mt_reml_fit_reduced_rank` (lower-trapezoidal
    loadings, then ``chol R0`` with log diagonal), ``Sigma_x = 2 H^{-1}`` from the Hessian
    of ``-2 logL`` at the optimum and ``J_i = d u_hat_i / dx`` by central differences
    (step ``REL_STEP * max(1, |x_k|)``); ``u_hat_i = Lambda f_hat_i``."""
    from .multitrait import build_and_solve
    from .multitrait_reml import _rr_unpack
    if method == "auto":
        method = "sparse_direct"
    grads = []
    for k in range(x.size):
        h = REL_STEP * max(1.0, abs(float(x[k])))
        sols = []
        for sign in (1.0, -1.0):
            xx = np.array(x, dtype=np.float64, copy=True)
            xx[k] += sign * h
            Lam, R0 = _rr_unpack(xx, t, r)
            res = build_and_solve(data, k_inv, k_diag, None, R0, method=method,
                                  compute_pev=False, memory_budget_bytes=memory_budget_bytes,
                                  factorization=factorization, loadings=Lam)
            sols.append(res.ebv)
        grads.append((sols[0] - sols[1]) / (2.0 * h))
    J = np.stack(grads, axis=2)
    return np.einsum("iak,kl,ibl->iab", J, cov_x, J)


def kackar_harville_delta_maternal(data, k_inv, theta: np.ndarray, cov: np.ndarray,
                                   memory_budget_bytes: int = 4 * 2**30) -> np.ndarray:
    """Maternal animal model (round 13): ``Delta_a = J_a Sigma J_a'`` (``2 x 2`` per animal:
    direct, maternal) with ``J_a = d (a_hat, m_hat)_a / d theta``,
    ``theta = (s_a, s_am, s_m, s_p..., s_e)`` in the order of
    :mod:`abp.solvers.maternal_reml` and ``Sigma`` its inverse average-information matrix.
    Central differences with the step ``REL_STEP * sqrt(s_a s_m)`` for the covariance and
    ``REL_STEP * theta_k`` otherwise."""
    from .maternal_reml import MaternalREMLEvaluator
    ev = MaternalREMLEvaluator(data, k_inv, 0.0, memory_budget_bytes)
    theta = np.asarray(theta, dtype=np.float64)
    q = ev.q
    grads = []
    for k in range(theta.size):
        h = REL_STEP * (np.sqrt(theta[0] * theta[2]) if k == 1 else theta[k])
        sols = []
        for sign in (1.0, -1.0):
            th = theta.copy()
            th[k] += sign * h
            sols.append(ev.solve(th)[ev.g0:ev.g1].reshape(q, 2))
        grads.append((sols[0] - sols[1]) / (2.0 * h))
    J = np.stack(grads, axis=2)                      # q x 2 x p
    return np.einsum("ajk,kl,abl->ajb", J, cov, J)

