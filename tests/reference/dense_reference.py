"""Independent dense reference implementations used only by the test-suite.

These functions deliberately use *different* formulations from the
production code so that agreement is evidence of correctness rather than of
self-consistency:

* relationship matrix: the recursive tabular method (Emik & Terrill 1949;
  Henderson 1976) on a naively ordered pedigree, instead of Meuwissen-Luo +
  the sparse Henderson/Quaas inverse + Colleau products used in production;
* BLUP: the marginal (V-based) GLS/BLUP formulas, instead of the
  mixed-model equations (MME);
* REML: the V-based restricted log-likelihood with the projection matrix P,
  instead of the MME determinant formulation.

Explicit dense inverses are acceptable here because the problems are tiny.
Nothing in this module may import from ``abp``.
"""

from __future__ import annotations

import numpy as np


def naive_order(ids, sires, dams):
    """Order animals so that parents precede offspring (O(n^2), tiny inputs)."""
    placed: list[str] = []
    seen: set[str] = set()
    remaining = list(zip(ids, sires, dams))
    while remaining:
        progressed = False
        for rec in list(remaining):
            a, s, d = rec
            if (s is None or s in seen) and (d is None or d in seen):
                placed.append(a)
                seen.add(a)
                remaining.remove(rec)
                progressed = True
        if not progressed:
            raise ValueError("cycle")
    return placed


def tabular_a(ids, sires, dams):
    """Dense A by the tabular method; returned in the order of ``ids``."""
    parent = {a: (s, d) for a, s, d in zip(ids, sires, dams)}
    order = naive_order(ids, sires, dams)
    pos = {a: k for k, a in enumerate(order)}
    n = len(order)
    A = np.zeros((n, n))
    for i, a in enumerate(order):
        s, d = parent[a]
        si = pos[s] if s is not None else None
        di = pos[d] if d is not None else None
        for j in range(i):
            v = 0.0
            if si is not None:
                v += 0.5 * A[si, j]
            if di is not None:
                v += 0.5 * A[di, j]
            A[i, j] = A[j, i] = v
        A[i, i] = 1.0 + (0.5 * A[si, di] if si is not None and di is not None else 0.0)
    back = np.array([pos[a] for a in ids])
    return A[np.ix_(back, back)]


def blup_v_form(y, X, Z, G, R):
    """GLS/BLUP from the marginal model y ~ N(Xb, V), V = Z G Z' + R.

    Returns (b_hat, u_hat, PEV matrix Var(u - u_hat)).  PEV includes the
    uncertainty of the GLS estimate of b:  PEV = G - G Z' P Z G.
    """
    V = Z @ G @ Z.T + R
    Vi = np.linalg.inv(V)
    XtViX = X.T @ Vi @ X
    b = np.linalg.solve(XtViX, X.T @ Vi @ y)
    P = Vi - Vi @ X @ np.linalg.solve(XtViX, X.T @ Vi)
    u = G @ Z.T @ Vi @ (y - X @ b)
    pev = G - G @ Z.T @ P @ Z @ G
    return b, u, pev


def reml_loglik_v_form(y, X, covs, theta):
    """Restricted log-likelihood (constant dropped) of V = sum_k theta_k * covs_k.

    ``covs`` are the n x n matrices V_k (e.g. Z A Z', Z Z', I); X full rank.
    Returns (loglik, P).
    """
    V = sum(t * C for t, C in zip(theta, covs))
    Vi = np.linalg.inv(V)
    XtViX = X.T @ Vi @ X
    P = Vi - Vi @ X @ np.linalg.solve(XtViX, X.T @ Vi)
    s1, ld1 = np.linalg.slogdet(V)
    s2, ld2 = np.linalg.slogdet(XtViX)
    assert s1 > 0 and s2 > 0
    return -0.5 * (ld1 + ld2 + y @ P @ y), P


def reml_score_v_form(y, X, covs, theta):
    """Analytical REML score 0.5*(y'P V_k P y - tr(P V_k)) for each k."""
    _, P = reml_loglik_v_form(y, X, covs, theta)
    Py = P @ y
    return np.array([0.5 * (Py @ C @ Py - np.trace(P @ C)) for C in covs])
