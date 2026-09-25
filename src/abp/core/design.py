"""Fixed-effect design matrices and deterministic handling of rank deficiency.

Model contract (method registry id ``lmm.fixed_design``)
--------------------------------------------------------
Fixed effects are an optional intercept, categorical factors (one column per
observed level) and numeric covariates (one column, used as given; users who
want a centred covariate must centre it in the data).  Such an ``X`` is
usually rank deficient (e.g. intercept + factor).  ABP makes it full column
rank by **setting dependent columns to zero**, scanning columns in a fixed,
documented order:

    intercept, then fixed terms in spec order, levels sorted as strings.

A column is dependent when its squared residual after projection on the
previously kept columns is ``<= rank_tol * ||x_j||^2`` (a scale-invariant,
angle-based criterion computed by a left-looking Cholesky of ``X'X``).  The
dropped columns are recorded; their "solutions" are reported as 0 with the
flag ``constrained``.  Individual fixed-effect level solutions therefore
depend on this constraint set and are **not estimable functions**; estimable
functions (e.g. ``X b``, differences of connected levels) and all random
effect predictions are invariant to the choice of constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.sparse as sp

from ..errors import ABPError

DEFAULT_RANK_TOL = 1e-9


@dataclass(frozen=True)
class FixedTerm:
    """One fixed term: ``kind`` is ``"factor"`` or ``"covariate"``."""

    column: str
    kind: str


@dataclass(frozen=True)
class FixedDesign:
    """Full-column-rank fixed design plus bookkeeping for reporting.

    Attributes
    ----------
    X:
        Sparse ``n x p_kept`` design containing only the kept columns.
    labels:
        ``(term, level)`` for *all* candidate columns (kept and dropped), in
        scan order; covariates use level ``""`` and the intercept is
        ``("intercept", "")``.
    kept:
        Boolean mask over ``labels``.
    rank_tol:
        Tolerance used for dependency detection.
    """

    X: sp.csr_matrix
    labels: tuple[tuple[str, str], ...]
    kept: np.ndarray
    rank_tol: float

    @property
    def rank(self) -> int:
        return int(self.kept.sum())

    @property
    def constrained_labels(self) -> list[tuple[str, str]]:
        return [lab for lab, k in zip(self.labels, self.kept) if not k]


def dependent_columns(xtx: np.ndarray, rank_tol: float = DEFAULT_RANK_TOL) -> np.ndarray:
    """Return a mask of columns kept by natural-order zero-pivot Cholesky.

    ``xtx`` is the dense symmetric positive semi-definite cross-product.
    Column ``j`` is kept iff its residual squared norm after projection on the
    previously kept columns exceeds ``rank_tol * xtx[j, j]``.  All-zero
    columns are always dropped.
    """
    p = xtx.shape[0]
    keep = np.zeros(p, dtype=bool)
    Lc = np.zeros((p, p), dtype=np.float64)  # compact: kept columns only
    k = 0
    for j in range(p):
        ajj = xtx[j, j]
        if ajj <= 0.0:
            continue
        lj = Lc[j, :k]
        v = ajj - lj @ lj
        if v <= rank_tol * ajj:
            continue
        piv = np.sqrt(v)
        Lc[j, k] = piv
        if j + 1 < p:
            Lc[j + 1:, k] = (xtx[j + 1:, j] - Lc[j + 1:, :k] @ lj) / piv
        keep[j] = True
        k += 1
    return keep


def build_fixed_design(columns: dict[str, Sequence], terms: Sequence[FixedTerm],
                       intercept: bool, n: int, rank_tol: float = DEFAULT_RANK_TOL,
                       max_dense_p: int = 20000) -> FixedDesign:
    """Build the full-column-rank fixed-effect design.

    Parameters
    ----------
    columns:
        Mapping column name -> sequence of length ``n`` (strings for factors,
        floats for covariates).  Missing values must have been handled by QC.
    terms:
        Fixed terms in spec order.
    intercept:
        Whether to include an intercept column first.
    """
    blocks: list[sp.csr_matrix] = []
    labels: list[tuple[str, str]] = []
    if intercept:
        blocks.append(sp.csr_matrix(np.ones((n, 1))))
        labels.append(("intercept", ""))
    for term in terms:
        values = columns[term.column]
        if term.kind == "factor":
            levels = sorted(set(values))
            pos = {lv: i for i, lv in enumerate(levels)}
            codes = np.fromiter((pos[v] for v in values), dtype=np.int64, count=n)
            blocks.append(sp.csr_matrix((np.ones(n), (np.arange(n), codes)),
                                        shape=(n, len(levels))))
            labels.extend((term.column, str(lv)) for lv in levels)
        elif term.kind == "covariate":
            x = np.asarray(values, dtype=np.float64).reshape(n, 1)
            if not np.all(np.isfinite(x)):
                raise ABPError("SCHEMA_TYPE", f"covariate {term.column!r} has non-finite values",
                               column=term.column)
            blocks.append(sp.csr_matrix(x))
            labels.append((term.column, ""))
        else:
            raise ABPError("SPEC_INVALID", f"unknown fixed term kind {term.kind!r}",
                           column=term.column)
    if not blocks:
        return FixedDesign(sp.csr_matrix((n, 0)), tuple(), np.zeros(0, dtype=bool), rank_tol)
    Xall = sp.hstack(blocks, format="csr")
    p = Xall.shape[1]
    if p > max_dense_p:
        raise ABPError("RESOURCE_MEMORY",
                       f"{p} fixed-effect columns exceed the dense rank-check limit {max_dense_p}",
                       p=p, limit=max_dense_p)
    xtx = (Xall.T @ Xall).toarray()
    keep = dependent_columns(xtx, rank_tol)
    if keep.sum() == 0 and p > 0:
        raise ABPError("MODEL_NOT_IDENTIFIABLE", "all fixed-effect columns are zero")
    X = Xall[:, np.flatnonzero(keep)].tocsr()
    return FixedDesign(X, tuple(labels), keep, rank_tol)
