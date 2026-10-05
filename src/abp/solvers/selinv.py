"""Sparse selected inversion (Takahashi equations) for exact PEV and REML traces.

Why
---
PEV and REML traces need entries of ``C^{-1}`` (the inverse of the mixed-model
coefficient matrix), but only on the sparsity pattern of ``C`` or of small
blocks.  Solving one unit vector per equation costs a full triangular solve
each, which is why ABP previously refused exact PEV above 30,000 equations
(finding F5).  The Takahashi, Fagan & Chin (1973) recurrence computes every
entry of ``C^{-1}`` on the pattern of the Cholesky factor from the factor
itself, at a cost comparable to the factorization (Erisman & Tinney 1975;
Campbell & Davis 1995; used in animal breeding by Misztal and co-workers,
e.g. Misztal & Perez-Enciso 1993, J Dairy Sci 76:1479).

Mathematics
-----------
With a symmetric permutation ``B = C[q][:, q]`` and ``B = L D L'`` (``L`` unit
lower triangular), ``Z = B^{-1}`` satisfies ``Z = D^{-1} L^{-1} + (I - L') Z``.
Processing columns ``j = n-1, ..., 0`` and writing ``S_j`` for the strictly
lower pattern of column ``j`` of ``L``:

    Z_ij = - sum_{k in S_j} L_kj Z_ik          (i in S_j)
    Z_jj = 1/d_j - sum_{k in S_j} L_kj Z_kj.

Every ``Z_ik`` needed has ``i, k in S_j``; for the *symbolic* Cholesky pattern
(closed under this recurrence: ``S_j minus {k and smaller} is a subset of S_k``
for ``k in S_j``) it has already been computed.  The numerical factor of
SuperLU omits entries that are exactly zero, which can break closure (e.g.
multi-trait systems with cancellations), so the factor is embedded into the
symbolic pattern computed from the elimination tree (Liu 1990), and every
SuperLU entry is verified to lie inside that pattern.

Result: all entries of ``C^{-1}`` on the pattern of ``L + L'``, which contains
the pattern of ``C`` and therefore every entry needed by diagonal PEV,
per-animal multi-trait PEV blocks and REML traces ``tr(K^{-1} C^{kk})``,
``tr(C^{-1} W'W)``.  Anything outside the pattern is refused, never guessed.

The Python functions are the reference implementation; the compiled kernels
``_native.symbolic_cholesky`` and ``_native.takahashi`` implement the same
algorithms and are cross-checked in the tests.
"""

from __future__ import annotations

import os

import numpy as np
import scipy.sparse as sp

from ..errors import ABPError

try:  # optional C++20 kernel (ADR 0002)
    from .. import _native  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    _native = None

#: bytes per stored entry of the symbolic factor (row index int64 + L value + Z value)
BYTES_PER_ENTRY = 24


def _native_ok(name: str) -> bool:
    return (_native is not None and hasattr(_native, name)
            and os.environ.get("ABP_DISABLE_NATIVE", "") != "1")


# ------------------------------------------------------------ symbolic phase
def symbolic_cholesky_python(indptr: np.ndarray, indices: np.ndarray, n: int
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Strictly lower pattern of the Cholesky factor of a symmetric matrix.

    Input: CSC structure (``indptr``, ``indices``) of a symmetric matrix (any
    triangle or both).  Column ``j`` of ``L`` is the union of the rows ``> j``
    of column ``j`` of the matrix and the patterns of the children of ``j`` in
    the elimination tree, less ``j``; the parent of ``j`` is the smallest row
    of its pattern.  Returns sorted CSC ``(colptr, rowidx)`` of ``L`` without
    the diagonal.
    """
    children: list[list[int]] = [[] for _ in range(n)]
    pats: list[np.ndarray] = []
    for j in range(n):
        rows = indices[indptr[j]:indptr[j + 1]]
        parts = [rows[rows > j]]
        for c in children[j]:
            pc = pats[c]
            parts.append(pc[pc > j])
        pj = np.unique(np.concatenate(parts)) if parts else np.empty(0, np.int64)
        pats.append(pj.astype(np.int64))
        if pj.size:
            children[int(pj[0])].append(j)
    colptr = np.zeros(n + 1, dtype=np.int64)
    colptr[1:] = np.cumsum([p.size for p in pats])
    rowidx = np.concatenate(pats) if pats else np.empty(0, np.int64)
    return colptr, rowidx.astype(np.int64)


def symbolic_cholesky(B: sp.csc_matrix) -> tuple[np.ndarray, np.ndarray, str]:
    B = B.tocsc()
    B.sort_indices()
    n = B.shape[0]
    ip = np.ascontiguousarray(B.indptr, dtype=np.int64)
    ix = np.ascontiguousarray(B.indices, dtype=np.int64)
    if _native_ok("symbolic_cholesky"):
        colptr_b, rowidx_b = _native.symbolic_cholesky(ip, ix, n)
        return (np.frombuffer(colptr_b, dtype=np.int64).copy(),
                np.frombuffer(rowidx_b, dtype=np.int64).copy(), "native_cpp")
    colptr, rowidx = symbolic_cholesky_python(ip, ix, n)
    return colptr, rowidx, "python"


# ----------------------------------------------------------- numeric phase
def takahashi_python(colptr: np.ndarray, rowidx: np.ndarray, lval: np.ndarray,
                     d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Takahashi recurrence on a closed pattern; returns ``(zdiag, zval)``.

    ``zval[p]`` is ``Z[rowidx[p], j]`` for ``colptr[j] <= p < colptr[j+1]``.
    """
    n = d.size
    zdiag = np.empty(n)
    zval = np.zeros(rowidx.size)
    acc = np.zeros(n)
    for j in range(n - 1, -1, -1):
        lo, hi = colptr[j], colptr[j + 1]
        S = rowidx[lo:hi]
        Lj = lval[lo:hi]
        for t, k in enumerate(S):
            lkj = Lj[t]
            acc[k] -= lkj * zdiag[k]
            # rows i in S with i > k: Z_ik is stored in column k
            klo, khi = colptr[k], colptr[k + 1]
            rows_k = rowidx[klo:khi]
            for u in range(t + 1, S.size):
                i = S[u]
                p = klo + np.searchsorted(rows_k, i)
                if p >= khi or rowidx[p] != i:
                    raise ABPError("FACTORIZATION_FAILED", "selected inversion: pattern not "
                                   "closed (internal error)", row=int(i), col=int(k))
                z = zval[p]
                acc[i] -= lkj * z
                acc[k] -= Lj[u] * z
        zval[lo:hi] = acc[S]
        zdiag[j] = 1.0 / d[j] - float(np.dot(Lj, acc[S]))
        acc[S] = 0.0
    return zdiag, zval


def takahashi(colptr, rowidx, lval, d) -> tuple[np.ndarray, np.ndarray, str]:
    if _native_ok("takahashi"):
        zd, zv = _native.takahashi(np.ascontiguousarray(colptr, dtype=np.int64),
                                   np.ascontiguousarray(rowidx, dtype=np.int64),
                                   np.ascontiguousarray(lval, dtype=np.float64),
                                   np.ascontiguousarray(d, dtype=np.float64))
        return (np.frombuffer(zd, dtype=np.float64).copy(),
                np.frombuffer(zv, dtype=np.float64).copy(), "native_cpp")
    zd, zv = takahashi_python(colptr, rowidx, lval, d)
    return zd, zv, "python"


# ---------------------------------------------------------------- interface
class SelectedInverse:
    """Entries of ``C^{-1}`` on the pattern of the Cholesky factor.

    Built from a SuperLU factorization with a symmetric permutation and no
    pivoting (:class:`abp.solvers.mme.SparseLU`).  ``memory_budget_bytes``
    bounds the symbolic factor; exceeding it raises ``RESOURCE_MEMORY`` before
    any numeric work.
    """

    def __init__(self, lu, C: sp.spmatrix, memory_budget_bytes: int = 4 * 2**30):
        n = C.shape[0]
        if not np.array_equal(lu.perm_r, lu.perm_c):
            raise ABPError("FACTORIZATION_FAILED", "selected inversion needs a symmetric "
                           "permutation without pivoting")
        self.n = n
        self.pos = np.asarray(lu.perm_c, dtype=np.int64)       # equation -> permuted position
        L = lu.L.tocsc()
        L.sort_indices()
        U = lu.U.tocsr()
        self.d = U.diagonal().copy()
        # U = D L' for a symmetric matrix factored without pivoting: verify
        Ls = sp.tril(L, -1).tocsc()
        dev = abs(sp.triu(U, 1) - (sp.diags(self.d) @ Ls.T)).max() if Ls.nnz else 0.0
        scale = max(float(np.max(np.abs(self.d))), 1.0)
        if dev > 1e-8 * scale:
            raise ABPError("FACTORIZATION_FAILED", "selected inversion: U is not D L' "
                           "(matrix not symmetric?)", deviation=float(dev))
        q = np.argsort(self.pos)
        B = sp.csc_matrix(C)[q][:, q]                              # the matrix SuperLU factored
        self.colptr, self.rowidx, self.kernel = symbolic_cholesky(B)
        need = BYTES_PER_ENTRY * self.rowidx.size
        if need > memory_budget_bytes:
            raise ABPError("RESOURCE_MEMORY",
                           f"selected inversion needs ~{need / 2**30:.2f} GiB for "
                           f"{self.rowidx.size} factor entries; budget is "
                           f"{memory_budget_bytes / 2**30:.2f} GiB", nnz_factor=int(self.rowidx.size))
        # embed SuperLU values into the symbolic pattern (verifying containment)
        self.lval = np.zeros(self.rowidx.size)
        cols = np.repeat(np.arange(n), np.diff(Ls.indptr))
        p = self._locate(Ls.indices.astype(np.int64), cols)
        if np.any(p < 0):
            raise ABPError("FACTORIZATION_FAILED", "selected inversion: numeric factor entry "
                           "outside the symbolic pattern (internal error)")
        self.lval[p] = Ls.data
        self.zdiag, self.zval, k2 = takahashi(self.colptr, self.rowidx, self.lval, self.d)
        self.kernel = f"{self.kernel}/{k2}"

    @property
    def nnz_factor(self) -> int:
        return int(self.rowidx.size)

    def _locate(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Storage position of permuted lower entries (row > col); -1 if absent."""
        out = np.full(rows.size, -1, dtype=np.int64)
        if rows.size == 0:
            return out
        order = np.lexsort((rows, cols))
        r, c = rows[order], cols[order]
        # search each column's segment; vectorized via a global key
        key_pat = np.repeat(np.arange(self.n, dtype=np.int64), np.diff(self.colptr)) * self.n \
            + self.rowidx
        key = c * self.n + r
        p = np.searchsorted(key_pat, key)
        ok = (p < key_pat.size)
        ok[ok] = key_pat[p[ok]] == key[ok]
        res = np.where(ok, p, -1)
        out[order] = res
        return out

    def diagonal(self, idx: np.ndarray) -> np.ndarray:
        """``C^{-1}[i, i]`` for equations ``idx``."""
        return self.zdiag[self.pos[np.asarray(idx, dtype=np.int64)]]

    def entries(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """``C^{-1}[rows[k], cols[k]]``; raises if an entry is outside the pattern."""
        a = self.pos[np.asarray(rows, dtype=np.int64)]
        b = self.pos[np.asarray(cols, dtype=np.int64)]
        out = np.empty(a.size)
        diag = a == b
        out[diag] = self.zdiag[a[diag]]
        lo = ~diag
        hi_ = np.maximum(a[lo], b[lo])
        lo_ = np.minimum(a[lo], b[lo])
        p = self._locate(hi_, lo_)
        if np.any(p < 0):
            raise ABPError("UNSUPPORTED_COMBINATION",
                           "requested entry of C^-1 lies outside the factor pattern; selected "
                           "inversion only provides entries on that pattern",
                           n_missing=int(np.sum(p < 0)))
        out[lo] = self.zval[p]
        return out

    def trace_product(self, M: sp.spmatrix, offset: int = 0) -> float:
        """``sum_ij M_ij C^{-1}[offset+i, offset+j]`` for sparse symmetric ``M``."""
        Mc = sp.coo_matrix(M)
        keep = Mc.data != 0.0         # stored zeros (e.g. exact cancellation in A^-1) add nothing
        Mc = sp.coo_matrix((Mc.data[keep], (Mc.row[keep], Mc.col[keep])), shape=Mc.shape)
        if Mc.nnz == 0:
            return 0.0
        return float(np.dot(Mc.data, self.entries(Mc.row + offset, Mc.col + offset)))
