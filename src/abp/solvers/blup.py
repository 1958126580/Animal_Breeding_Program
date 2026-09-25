"""Single-trait BLUP with known variance components.

Model contract (method registry id ``blup.single_trait``)
---------------------------------------------------------
``y = X b + sum_k Z_k u_k + e``, ``u_k ~ N(0, sigma_k^2 K_k)``,
``e ~ N(0, sigma_e^2 I)``; ``K_k`` is a relationship structure (``A``,
``G``, ``H`` or ``I``).  Variance components are treated as **known**; when
they come from REML the reported PEV is conditional on the estimates and
does not propagate their sampling error.

Outputs per random term: BLUP ``u_hat``; ``PEV_i = [C^{-1}]_ii`` (unscaled
MME, see :mod:`abp.solvers.mme`), ``SEP_i = sqrt(PEV_i)`` and, for genetic
terms, reliability ``1 - PEV_i / (sigma_k^2 K_k[i, i])`` where
``K_k[i, i] = 1 + F_i`` for the pedigree relationship.  Reliabilities
outside ``[-1e-8, 1 + 1e-8]`` indicate a scale/model inconsistency and raise
``RELIABILITY_OUT_OF_RANGE``; values within that round-off band are clamped
to ``[0, 1]`` and the number of clamped values is reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import scipy.sparse as sp

from ..errors import ABPError
from .mme import (MixedModelSystem, RandomEffect, SolveResult, assemble, pev_diagonal,
                  solve_system)

RELIABILITY_ROUNDING_BAND = 1e-8


@dataclass
class RandomTerm:
    """A random term before variance scaling.

    ``k_inv`` is the inverse structure matrix ``K^{-1}`` (``q x q``);
    ``k_diag`` is ``diag(K)`` (needed for reliabilities of genetic terms);
    ``logdet_k`` is ``log|K|`` (needed by REML).
    """

    name: str
    Z: sp.csr_matrix
    k_inv: sp.spmatrix | np.ndarray
    labels: Sequence[str]
    genetic: bool
    k_diag: np.ndarray | None = None
    logdet_k: float | None = None

    @property
    def q(self) -> int:
        return self.Z.shape[1]


@dataclass
class TermResult:
    name: str
    labels: Sequence[str]
    solution: np.ndarray
    pev: np.ndarray | None
    reliability: np.ndarray | None
    n_reliability_clamped: int = 0


@dataclass
class BLUPResult:
    fixed_solution: np.ndarray
    terms: dict[str, TermResult]
    solve: SolveResult
    system: MixedModelSystem
    variances: dict[str, float]
    residuals: np.ndarray
    extra: dict = field(default_factory=dict)


def _scaled_precision(k_inv, variance: float):
    return k_inv * (1.0 / variance)


def build_system(y: np.ndarray, X: sp.csr_matrix, terms: Sequence[RandomTerm],
                 variances: dict[str, float]) -> MixedModelSystem:
    """Assemble the unscaled MME for given variance components.

    ``variances`` maps each term name and ``"residual"`` to a variance > 0.
    """
    for name in [t.name for t in terms] + ["residual"]:
        v = variances.get(name)
        if v is None:
            raise ABPError("SPEC_INVALID", f"missing variance for {name!r}", term=name)
        if not np.isfinite(v) or v <= 0:
            raise ABPError("COVARIANCE_NOT_PD",
                           f"variance for {name!r} must be finite and > 0 (got {v!r}); a zero "
                           "variance means the term should be removed from the model",
                           term=name, value=v)
    n = y.shape[0]
    rinv = sp.identity(n, format="csr") * (1.0 / variances["residual"])
    randoms = [RandomEffect(t.name, t.Z, _scaled_precision(t.k_inv, variances[t.name]), t.labels)
               for t in terms]
    return assemble(y, X, randoms, rinv)


def blup(y: np.ndarray, X: sp.csr_matrix, terms: Sequence[RandomTerm],
         variances: dict[str, float], method: str = "auto", compute_pev: bool = True,
         tol: float = 1e-10, max_iter: int = 10000,
         memory_budget_bytes: int = 4 * 2**30) -> BLUPResult:
    """Solve the MME and derive PEV and reliabilities."""
    y = np.asarray(y, dtype=np.float64)
    if not np.all(np.isfinite(y)):
        raise ABPError("SCHEMA_TYPE", "response contains non-finite values")
    system = build_system(y, X, terms, variances)
    result = solve_system(system, method=method, need_inverse=compute_pev, tol=tol,
                          max_iter=max_iter, memory_budget_bytes=memory_budget_bytes)
    s = result.solution
    out: dict[str, TermResult] = {}
    for t in terms:
        a, b = system.offsets[t.name]
        sol = s[a:b].copy()
        pev = rel = None
        n_clamped = 0
        if compute_pev:
            pev = pev_diagonal(result, np.arange(a, b))
            if t.genetic:
                if t.k_diag is None:
                    raise ValueError(f"genetic term {t.name!r} needs k_diag for reliability")
                prior = variances[t.name] * np.asarray(t.k_diag, dtype=np.float64)
                rel_raw = 1.0 - pev / prior
                bad = (rel_raw < -RELIABILITY_ROUNDING_BAND) | (rel_raw > 1 + RELIABILITY_ROUNDING_BAND)
                if np.any(bad):
                    k = int(np.flatnonzero(bad)[0])
                    raise ABPError("RELIABILITY_OUT_OF_RANGE",
                                   f"reliability {rel_raw[k]:.6g} for {t.labels[k]!r} in term "
                                   f"{t.name!r} is outside [0, 1]",
                                   term=t.name, label=t.labels[k], value=float(rel_raw[k]),
                                   n_bad=int(bad.sum()))
                rel = np.clip(rel_raw, 0.0, 1.0)
                n_clamped = int(np.sum(rel != rel_raw))
        out[t.name] = TermResult(t.name, t.labels, sol, pev, rel, n_clamped)
    fixed = s[:system.p].copy()
    residuals = y - system.W @ s
    return BLUPResult(fixed, out, result, system, dict(variances), residuals)
