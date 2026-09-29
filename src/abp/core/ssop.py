"""Matrix-free single step: ``H^{-1}`` applied as an operator (method id ``gen.ssgblup_matrix_free``).

Background
----------
``H^{-1} = A^{-1} + embed(G*^{-1} - A22^{-1})`` (Aguilar et al. 2010; Christensen &
Lund 2010).  The explicit construction in :func:`abp.core.genomic.single_step`
forms dense ``A22`` and ``A22^{-1}`` (``n2 x n2``), a dense ``n x n2`` block for
``diag(H)`` and a dense ``G*^{-1}``, so memory grows with ``n2^2`` and ``n n2``.
Iterative solvers only need products ``H^{-1} v``, which this module provides
with memory that grows with the number of non-zeros of ``A^{-1}`` plus, with
APY, ``c^2 + c n2`` (``c`` core animals):

* ``A22^{-1} v``: with ``A^{-1}`` partitioned into non-genotyped (1) and
  genotyped (2) animals, ``A22^{-1} = A^{22} - A^{21} (A^{11})^{-1} A^{12}``
  (Strandén & Mäntysaari 2014; Masuda et al. 2017) - a Schur complement of
  sparse blocks; ``A^{11}`` is factorised once by the sparse LDL' of
  :mod:`abp.solvers.cholesky`.
* ``G*^{-1} v``: either a dense inverse (small ``n2``) or the APY inverse as
  an operator, ``G_APY^{-1} = [[G_cc^{-1},0],[0,0]] + [[-P],[I]] M^{-1} [[-P', I]]``
  with ``P = G_cc^{-1} G_cn`` (never formed as an ``n2 x n2`` matrix).  The
  APY blocks are computed from the centred genotypes, so neither ``G`` nor
  ``A22`` is formed; the blend policy needs only the core columns of ``A``
  (Colleau products).

What is *not* available on this path: PEV/reliabilities (they need
``diag(C^{-1})``), REML (needs ``log|H|`` and traces) and ``diag(H)``.  These
are refused explicitly by the specification rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

from ..errors import ABPError


class A22InverseOperator:
    """``A22^{-1} v`` from sparse blocks of ``A^{-1}`` (see module notes)."""

    def __init__(self, ainv: sp.spmatrix, geno_index: np.ndarray,
                 memory_budget_bytes: int = 4 * 2**30):
        from ..solvers.mme import make_sparse_factor
        n = ainv.shape[0]
        g = np.asarray(geno_index, dtype=np.int64)
        non = np.setdiff1d(np.arange(n), g)
        A = sp.csr_matrix(ainv)
        self.a22 = A[g][:, g].tocsr()
        self.a12 = A[non][:, g].tocsr()
        self.n_non = non.size
        self.fac = make_sparse_factor(A[non][:, non].tocsc(), memory_budget_bytes) \
            if non.size else None

    def __call__(self, v: np.ndarray) -> np.ndarray:
        out = self.a22 @ v
        if self.fac is not None:
            out = out - self.a12.T @ self.fac.solve(self.a12 @ v)
        return out

    def diag_upper_bound(self) -> np.ndarray:
        """``diag(A^{22}) >= diag(A22^{-1})`` (the Schur term is positive semi-definite)."""
        return self.a22.diagonal()


class DenseInverseOperator:
    """``G*^{-1} v`` with an explicit dense inverse (small ``n2``)."""

    def __init__(self, g_inv: np.ndarray):
        self.g_inv = np.asarray(g_inv, dtype=np.float64)

    def __call__(self, v: np.ndarray) -> np.ndarray:
        return self.g_inv @ v

    def diag(self) -> np.ndarray:
        return np.diag(self.g_inv).copy()


class APYOperator:
    """``G_APY^{-1} v`` without forming ``G_APY^{-1}`` (Misztal et al. 2014)."""

    def __init__(self, g_cc: np.ndarray, g_cn: np.ndarray, g_nn_diag: np.ndarray,
                 core: np.ndarray, n2: int):
        self.core = np.asarray(core, dtype=np.int64)
        self.non = np.setdiff1d(np.arange(n2), self.core)
        self.n2 = n2
        try:
            L = sla.cholesky(np.asarray(g_cc, dtype=np.float64), lower=True)
        except np.linalg.LinAlgError:
            raise ABPError("RELATIONSHIP_SINGULAR", f"APY: G_cc ({self.core.size} core animals) "
                           "is not positive definite; choose fewer core animals or a "
                           "blend/ridge policy", n_core=int(self.core.size)) from None
        self.gcc_inv = sla.cho_solve((L, True), np.eye(self.core.size))
        self.gcc_inv = 0.5 * (self.gcc_inv + self.gcc_inv.T)
        self.P = self.gcc_inv @ g_cn                                  # c x n
        m = np.asarray(g_nn_diag, dtype=np.float64) - np.einsum("ij,ij->j", g_cn, self.P)
        if m.size and not np.all(m > 1e-10 * max(float(np.mean(np.diag(g_cc))), 1e-300)):
            k = int(np.argmin(m))
            raise ABPError("RELATIONSHIP_SINGULAR", f"APY: conditional variance of a non-core "
                           f"animal is {m[k]:.3e} (<= 0)", min_m=float(m[k]))
        self.m = m
        self.logdet = 2.0 * float(np.sum(np.log(np.diag(L)))) + float(np.sum(np.log(m)))

    def __call__(self, v: np.ndarray) -> np.ndarray:
        vc, vn = v[self.core], v[self.non]
        w = (self.P.T @ vc - vn) / self.m            # M^-1 (P' v_c - v_n)
        out = np.empty_like(v, dtype=np.float64)
        out[self.core] = self.gcc_inv @ vc + self.P @ w
        out[self.non] = -w
        return out

    def diag(self) -> np.ndarray:
        d = np.empty(self.n2)
        d[self.core] = np.diag(self.gcc_inv) + np.einsum("ij,ij->i", self.P, self.P / self.m)
        d[self.non] = 1.0 / self.m
        return d


@dataclass
class SingleStepHInverse:
    """``H^{-1}`` as ``A^{-1}`` (sparse, assembled into the MME) plus a correction
    ``embed(G*^{-1} - A22^{-1})`` applied as an operator inside PCG."""

    a_inv: sp.csr_matrix
    geno_index: np.ndarray
    g_op: object
    a22_op: A22InverseOperator
    meta: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.a_inv.shape

    def correction(self, u: np.ndarray) -> np.ndarray:
        """``embed(G*^{-1} - A22^{-1}) u`` for a full-length vector ``u``."""
        out = np.zeros_like(u, dtype=np.float64)
        ug = u[self.geno_index]
        out[self.geno_index] = self.g_op(ug) - self.a22_op(ug)
        return out

    def correction_diag(self) -> np.ndarray:
        """Diagonal used by the Jacobi preconditioner: ``diag(G*^{-1}) - diag(A^{22})``
        at the genotyped animals, so that the preconditioner's genotyped entries are
        ``diag(G*^{-1})`` (a lower bound of ``diag(H^{-1})``, always positive)."""
        out = np.zeros(self.a_inv.shape[0])
        out[self.geno_index] = self.g_op.diag() - self.a22_op.diag_upper_bound()
        return out

    def matvec(self, u: np.ndarray) -> np.ndarray:
        return self.a_inv @ u + self.correction(u)


def apy_blocks_from_genotypes(Wc: np.ndarray, scale_d: float, core: np.ndarray,
                              policy: str, alpha: float, ridge: float,
                              a22_core_cols: np.ndarray | None,
                              a22_diag: np.ndarray | None):
    """``G*_cc``, ``G*_cn`` and ``diag(G*)_n`` from centred genotypes ``Wc`` (``n2 x m``,
    ``G = Wc Wc' / d``) under the singular policy, without forming ``G``.

    ``a22_core_cols`` is ``A22[:, core]`` (``n2 x c``) and ``a22_diag`` ``diag(A22)``;
    both are required for ``policy = 'blend'`` only."""
    n2 = Wc.shape[0]
    core = np.asarray(core, dtype=np.int64)
    non = np.setdiff1d(np.arange(n2), core)
    Wk = Wc[core]
    g_cc = Wk @ Wk.T / scale_d
    g_cn = Wk @ Wc[non].T / scale_d
    g_nn = np.einsum("ij,ij->i", Wc[non], Wc[non]) / scale_d
    if policy == "blend":
        if a22_core_cols is None or a22_diag is None:
            raise ValueError("blend needs A22 core columns and diag(A22)")
        g_cc = (1 - alpha) * g_cc + alpha * a22_core_cols[core]
        g_cn = (1 - alpha) * g_cn + alpha * a22_core_cols[non].T
        g_nn = (1 - alpha) * g_nn + alpha * a22_diag[non]
    elif policy == "ridge":
        g_cc = g_cc + ridge * np.eye(core.size)
        g_nn = g_nn + ridge
    return g_cc, g_cn, g_nn
