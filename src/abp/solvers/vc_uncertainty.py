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
