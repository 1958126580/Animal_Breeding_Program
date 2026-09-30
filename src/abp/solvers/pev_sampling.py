"""PEV and reliability by sampling (method id ``blup.pev_sampled``).

For solvers that give solutions but not ``diag(C^{-1})`` (the matrix-free
single step), prediction error variances are estimated by simulation
(Garcia-Cortes, Moreno, Varona & Altarriba 1995, J Anim Breed Genet 112:176;
Hickey et al. 2009, Genet Sel Evol 41): for ``s = 1..N``

    u*_k ~ N(0, sigma_k^2 K_k),  e* ~ N(0, sigma_e^2 I),  y* = sum_k Z_k u*_k + e*,
    u_hat* = BLUP(y*)  (same MME, same solver),   d = u* - u_hat* ~ N(0, C^{-1}_uu),

because BLUP is translation invariant in the fixed effects (so ``y*`` needs
none) and ``u* - u_hat*`` is independent of ``u_hat*``.  ABP reports

    PEV_i = mean_s d_is^2,      reliability_i = 1 - mean_s d_is^2 / mean_s u*_is^2,

a ratio estimator that needs no ``diag(K)`` (not available on the
matrix-free path) and whose errors partly cancel.  Its Monte-Carlo standard
error follows from the delta method with the sample moments, and is written
next to every reliability.  Relative SE of PEV ``~ sqrt(2/N)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from ..errors import ABPError
from .blup import RandomTerm, build_system
from .mme import pcg


@dataclass
class SampledPEV:
    pev: np.ndarray
    reliability: np.ndarray
    reliability_se: np.ndarray
    n_samples: int
    seed: int
    pcg_iterations: list


def _term_sampler(t: RandomTerm, variance: float):
    from ..core.ssop import SingleStepHInverse
    sd = float(np.sqrt(variance))
    if isinstance(t.k_inv, SingleStepHInverse):
        return lambda rng: sd * t.k_inv.sample(rng)
    K = t.k_inv
    if sp.issparse(K):
        K = sp.csr_matrix(K)
        if (K - sp.diags(K.diagonal())).count_nonzero() == 0:
            s = sd / np.sqrt(K.diagonal())
            return lambda rng: s * rng.standard_normal(s.size)
    raise ABPError("UNSUPPORTED_COMBINATION", f"sampled PEV: no sampler for term {t.name!r} "
                   "(supported: matrix-free single step and iid terms)")


def sampled_pev(n: int, X, terms: list[RandomTerm], variances: dict, genetic_term: str,
                n_samples: int = 200, seed: int = 20260925, tol: float = 1e-8,
                max_iter: int = 10000) -> SampledPEV:
    """Estimate PEV and reliabilities of ``genetic_term`` by ``n_samples`` simulations."""
    if n_samples < 10:
        raise ABPError("SPEC_INVALID", "sampled PEV needs at least 10 samples")
    system = build_system(np.zeros(n), X, terms, variances)
    samplers = [_term_sampler(t, variances[t.name]) for t in terms]
    se = float(np.sqrt(variances["residual"]))
    a0, b0 = system.offsets[genetic_term]
    q = b0 - a0
    s_d2 = np.zeros(q)
    s_u2 = np.zeros(q)
    s_d4 = np.zeros(q)
    s_u4 = np.zeros(q)
    s_du = np.zeros(q)
    rng = np.random.default_rng(seed)
    diag = system.diagonal()
    its = []
    gi = [t.name for t in terms].index(genetic_term)
    for _ in range(n_samples):
        draws = [smp(rng) for smp in samplers]
        ystar = se * rng.standard_normal(n)
        for t, u in zip(terms, draws):
            ystar += t.Z @ u
        rhs = system.W.T @ (system.rinv @ ystar)
        sol, info = pcg(system.matvec, rhs, diag, tol=tol, max_iter=max_iter)
        if not info.converged:
            raise ABPError("SOLVER_NOT_CONVERGED", "sampled PEV: PCG did not converge",
                           rel_residual=info.rel_residual)
        its.append(info.iterations)
        d = draws[gi] - sol[a0:b0]
        u = draws[gi]
        d2, u2 = d * d, u * u
        s_d2 += d2
        s_u2 += u2
        s_d4 += d2 * d2
        s_u4 += u2 * u2
        s_du += d2 * u2
    N = float(n_samples)
    A, B = s_d2 / N, s_u2 / N
    vA = np.maximum(s_d4 / N - A * A, 0.0) / N
    vB = np.maximum(s_u4 / N - B * B, 0.0) / N
    cAB = (s_du / N - A * B) / N
    rel = 1.0 - A / B
    var_r = vA / B ** 2 + A * A * vB / B ** 4 - 2.0 * A * cAB / B ** 3
    return SampledPEV(A, np.clip(rel, 0.0, 1.0), np.sqrt(np.maximum(var_r, 0.0)),
                      n_samples, seed, its)
