"""Sparse LDL' factorization with minimum-degree ordering (method id ``num.sparse_ldl``).

Why
---
Exact PEV at scale (selected inversion, :mod:`abp.solvers.selinv`) spent most
of its time in SuperLU, a general LU code (15.2 of 16.9 s for 100,500
equations).  The mixed-model coefficient matrix is symmetric positive
definite, so a Cholesky-type factorization with a fill-reducing ordering does
the same job with less work, and its factor is directly on the symbolic
pattern that the Takahashi recurrence uses.

Algorithms (textbook forms, implemented here from their mathematical
description; George & Liu 1981, *Computer Solution of Large Sparse Positive
Definite Systems*; Liu 1990 for the elimination tree)
------------------------------------------------------------------------
* **Ordering**: minimum degree on the explicit elimination graph - repeatedly
  eliminate a node of smallest current degree (ties: smallest index), join
  its neighbours into a clique.  Dense rows (degree > max(16, 10 sqrt(n)),
  e.g. an intercept linked to every recorded animal) are set aside and
  eliminated last, as in AMD.  Deterministic.
* **Symbolic**: :func:`abp.solvers.selinv.symbolic_cholesky` on the permuted
  matrix ``B = C[q][:, q]`` (column patterns and elimination tree).
* **Numeric (up-looking LDL')**: for ``k = 0..n-1`` solve
  ``L[:k,:k] D[:k] y = B[:k, k]`` sparsely - the non-zeros of row ``k`` of ``L``
  are the elimination-tree reach of the pattern of ``B[:k, k]``, processed in
  topological order - then ``L[k, j] = y_j / d_j`` and
  ``d_k = B_kk - sum_j L[k, j] y_j``.  A pivot ``d_k <= 0`` means the matrix is
  not positive definite (``FACTORIZATION_FAILED``).
* **Dense trailing block** (round 12): when a dense factorization of the last ``m``
  columns (``128 <= m <= 6000``) saves work (:func:`dense_tail_split`), the up-looking
  kernel stops the reach at ``s = n - m`` and returns the Schur complement
  ``S = B22 - L21 D1 L21'``; ``S = Lc Lc'`` by LAPACK Cholesky, then
  ``d = diag(Lc)^2`` and ``L22 = Lc diag(Lc)^{-1}`` on the symbolic pattern (block
  elimination; Golub & Van Loan 2013, sec. 4.2; the dense-front idea of supernodal and
  multifrontal codes, Duff & Reid 1983).  ``dense_tail=False`` turns it off.
* **Solve**: ``x = P' L^{-T} D^{-1} L^{-1} P b`` for one or many right-hand sides.

The Python functions are the reference; the compiled kernels
``_native.mindegree_order``, ``_native.ldl_numeric`` and ``_native.ldl_solve``
implement the same algorithms and are cross-checked in the tests.  SuperLU
remains available (``solver.factorization = "superlu"``) as the independent
reference path.
"""

from __future__ import annotations

import heapq
import os

import numpy as np
import scipy.sparse as sp

from ..errors import ABPError
from .selinv import BYTES_PER_ENTRY, symbolic_cholesky, takahashi

try:  # optional C++20 kernel (ADR 0002)
    from .. import _native  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    _native = None


def native_ldl_available() -> bool:
    return (_native is not None and hasattr(_native, "ldl_numeric")
            and os.environ.get("ABP_DISABLE_NATIVE", "") != "1")


# ---------------------------------------------------------------- ordering
def mindegree_order_python(indptr: np.ndarray, indices: np.ndarray, n: int) -> np.ndarray:
    """Minimum-degree elimination order of a symmetric pattern (reference)."""
    adj = [set() for _ in range(n)]
    for j in range(n):
        for i in indices[indptr[j]:indptr[j + 1]]:
            if i != j:
                adj[j].add(int(i))
                adj[int(i)].add(j)
    # dense rows (degree > max(16, 10 sqrt(n))) are eliminated last, as in AMD
    limit = max(16, int(10.0 * np.sqrt(n)))
    dense = {v for v in range(n) if len(adj[v]) > limit}
    for v in range(n):
        if v in dense:
            adj[v] = set()
        else:
            adj[v] -= dense
    heap = [(len(adj[v]), v) for v in range(n) if v not in dense]
    heapq.heapify(heap)
    done = np.zeros(n, dtype=bool)
    order = []
    while heap:
        deg, v = heapq.heappop(heap)
        if done[v] or deg != len(adj[v]):
            continue                                   # stale entry
        done[v] = True
        order.append(v)
        nb = adj[v]
        for u in nb:
            adj[u].discard(v)
            adj[u].update(w for w in nb if w != u)
            heapq.heappush(heap, (len(adj[u]), u))
        adj[v] = set()
    order.extend(sorted(dense))
    return np.asarray(order, dtype=np.int64)


def mindegree_order(C: sp.spmatrix) -> tuple[np.ndarray, str]:
    Cc = sp.csc_matrix(C)
    Cc.sort_indices()
    ip = np.ascontiguousarray(Cc.indptr, dtype=np.int64)
    ix = np.ascontiguousarray(Cc.indices, dtype=np.int64)
    if native_ldl_available():
        return np.frombuffer(_native.mindegree_order(ip, ix, Cc.shape[0]), dtype=np.int64).copy(), \
            "native_cpp"
    return mindegree_order_python(ip, ix, Cc.shape[0]), "python"


# ----------------------------------------------------------------- numeric
def ldl_numeric_python(Bp, Bi, Bx, colptr, rowidx, n, split=None):
    """Up-looking LDL' of the permuted matrix (full symmetric CSC ``Bp, Bi, Bx``)
    on the symbolic pattern ``(colptr, rowidx)``; returns ``(d, lval)``.

    With ``split < n`` only the columns ``< split`` are factorized; for the rows
    ``k >= split`` the elimination-tree reach is cut at ``split``, which leaves row
    ``k`` of the Schur complement ``S = B22 - L21 D1 L21'`` of the trailing block in
    ``y[split:k]`` and ``dkk``.  Returns ``(d, lval, S)`` then (``S`` dense, lower
    triangle; ``d`` and ``lval`` of the trailing columns are zero)."""
    split = n if split is None else int(split)
    m = n - split
    S = np.zeros((m, m)) if split < n else None
    parent = np.full(n, -1, dtype=np.int64)
    for j in range(n):
        if colptr[j + 1] > colptr[j]:
            parent[j] = rowidx[colptr[j]]
    d = np.zeros(n)
    lval = np.zeros(rowidx.size)
    fill = colptr[:-1].copy()
    y = np.zeros(n)
    mark = np.full(n, -1, dtype=np.int64)
    for k in range(n):
        # scatter the upper part of column k and find the reach in the elimination tree
        stack = []
        dkk = 0.0
        for p in range(Bp[k], Bp[k + 1]):
            i = Bi[p]
            if i == k:
                dkk += Bx[p]
            elif i < k:
                y[i] += Bx[p]
                path = []
                while i != -1 and i < k and i < split and mark[i] != k:
                    path.append(i)
                    mark[i] = k
                    i = parent[i]
                stack.extend(reversed(path))       # ancestors after descendants when reversed
        # topological order: process descendants before ancestors (sort ascending is valid:
        # in an elimination tree a descendant has a smaller index than its ancestors)
        for j in sorted(stack):
            yj = y[j]
            y[j] = 0.0
            for p in range(colptr[j], fill[j]):
                y[rowidx[p]] -= lval[p] * yj
            lkj = yj / d[j]
            dkk -= lkj * yj
            p = fill[j]
            if p >= colptr[j + 1] or rowidx[p] != k:
                raise ABPError("FACTORIZATION_FAILED", "LDL': numeric entry outside the "
                               "symbolic pattern (internal error)", row=int(k), col=int(j))
            lval[p] = lkj
            fill[j] += 1
        if k >= split:
            S[k - split, :k - split] = y[split:k]
            y[split:k] = 0.0
            S[k - split, k - split] = dkk
            continue
        if not dkk > 0.0:
            raise ABPError("FACTORIZATION_FAILED", "coefficient matrix is not positive "
                           "definite (non-positive pivot in LDL')", pivot=float(dkk), index=int(k))
        d[k] = dkk
    return (d, lval) if S is None else (d, lval, S)


def ldl_solve_python(colptr, rowidx, lval, d, b):
    """Solve ``L D L' x = b`` (permuted system) for a vector or matrix ``b``."""
    x = np.array(b, dtype=np.float64, copy=True)
    n = d.size
    for j in range(n):                                  # L z = b
        xj = x[j]
        if np.any(xj != 0):
            for p in range(colptr[j], colptr[j + 1]):
                x[rowidx[p]] -= lval[p] * xj
    x = (x.T / d).T
    for j in range(n - 1, -1, -1):                      # L' x = z
        for p in range(colptr[j], colptr[j + 1]):
            x[j] -= lval[p] * x[rowidx[p]]
    return x


def _symmetric_pattern(C: sp.spmatrix) -> sp.csr_matrix:
    """``(C + C') / 2``: a matrix that is symmetric only to rounding can have an
    asymmetric sparsity pattern (a product that cancels to ~1e-17 on one side and to
    exactly 0 on the other); the symbolic factorization assumes a symmetric pattern."""
    C = sp.csr_matrix(C)
    S = ((C + C.T) * 0.5).tocsr()
    S.sum_duplicates()
    S.sort_indices()
    return S


def _on_pattern(C: sp.csr_matrix, pat: sp.csr_matrix) -> sp.csr_matrix:
    """``C`` stored on the (larger) pattern of ``pat``; every entry of ``C`` must lie on
    it (explicit zeros fill the rest), otherwise ``ValueError``."""
    n = pat.shape[1]
    pc = pat.tocoo()
    key = pc.row.astype(np.int64) * n + pc.col
    order = np.argsort(key)
    cc = C.tocoo()
    want = cc.row.astype(np.int64) * n + cc.col
    pos = np.searchsorted(key[order], want)
    if np.any(pos >= key.size) or np.any(key[order][np.minimum(pos, key.size - 1)] != want):
        raise ValueError("refactor needs a matrix with the same pattern")
    data = np.zeros(pat.nnz)
    np.add.at(data, order[pos], cc.data)
    return sp.csr_matrix((data, pat.indices.copy(), pat.indptr.copy()), shape=pat.shape)


# Dense trailing block (round 12).  With a fill-reducing order the last columns of L are
# often nearly dense (a separator of the pedigree graph; the maternal animal model on a
# pedigree with long-range links: 85% of the work in the last 20% of the columns).  The
# up-looking kernel does that work one scalar at a time; the trailing block is instead
# formed as the Schur complement S = B22 - L21 D1 L21' and factorized densely with LAPACK.
DENSE_TAIL_MIN = 128          # smallest trailing block worth a dense factorization
DENSE_TAIL_MAX = 6000         # 6000^2 doubles = 288 MB for S
DENSE_SPEED_RATIO = 8.0       # LAPACK potrf / up-looking kernel flop rate (build machine:
                              # 16-20 vs ~2 GFlop/s, one thread); see the validation report


def dense_tail_split(colptr: np.ndarray, n: int, *, min_size: int = DENSE_TAIL_MIN,
                     max_size: int = DENSE_TAIL_MAX,
                     speed_ratio: float = DENSE_SPEED_RATIO) -> int:
    """First column ``s = n - m`` of the trailing block whose dense factorization saves
    the most work, or ``n`` (no dense block) when none saves any.  The up-looking work
    of column ``j`` is about ``c_j^2`` (``c_j`` = entries below the diagonal); every
    entry of a trailing column lies in the block, so the work of the last ``m`` columns
    is a suffix sum; the dense work is ``m^3 / 3`` at ``speed_ratio`` times the rate:
    choose ``m`` (``min_size <= m <= max_size``) maximizing
    ``sum_{j >= n-m} c_j^2 - m^3 / (3 speed_ratio)``."""
    if n < min_size:
        return n
    cnt = np.diff(np.asarray(colptr, dtype=np.int64)).astype(np.float64)
    m = np.arange(1, n + 1, dtype=np.float64)                 # block sizes 1..n
    work = np.cumsum((cnt * cnt)[::-1])                       # up-looking work, last m columns
    gain = work - m ** 3 / (3.0 * speed_ratio)
    gain[(m < min_size) | (m > max_size)] = -np.inf
    best = int(np.argmax(gain))
    return n if not gain[best] > 0.0 else n - int(m[best])


def _dense_tail_factor(S: np.ndarray, split: int, colptr, rowidx, d, lval) -> None:
    """Cholesky of the Schur complement ``S`` (lower triangle given) and its
    ``L D L'`` written into ``d[split:]`` and the trailing columns of ``lval``."""
    import scipy.linalg as sla
    m = S.shape[0]
    try:                                       # potrf reads only the lower triangle
        Lc = sla.cholesky(S, lower=True, overwrite_a=True, check_finite=False)
    except np.linalg.LinAlgError:
        raise ABPError("FACTORIZATION_FAILED", "coefficient matrix is not positive definite "
                       "(dense trailing block of LDL')", index=int(split)) from None
    piv = np.diag(Lc).copy()
    if not (np.all(piv > 0.0) and np.all(np.isfinite(piv))):
        raise ABPError("FACTORIZATION_FAILED", "coefficient matrix is not positive definite "
                       "(dense trailing block of LDL')", index=int(split))
    d[split:] = piv * piv
    a = int(colptr[split])
    cols = np.repeat(np.arange(m), np.diff(np.asarray(colptr[split:], dtype=np.int64)))
    lval[a:] = Lc[np.asarray(rowidx[a:], dtype=np.int64) - split, cols] / piv[cols]


class SparseLDL:
    """``C = P' L D L' P`` with minimum-degree ``P``; same interface as ``SparseLU``."""

    kind = "sparse_direct"

    def __init__(self, C: sp.spmatrix, memory_budget_bytes: int = 4 * 2**30,
                 dense_tail: bool = True):
        C = _symmetric_pattern(C)
        self.C = C
        self.n = C.shape[0]
        self.memory_budget_bytes = memory_budget_bytes
        order, k1 = mindegree_order(C)
        self.pos = np.empty(self.n, dtype=np.int64)
        self.pos[order] = np.arange(self.n)             # equation -> permuted position
        self.q = order
        B = sp.csc_matrix(C)[order][:, order]
        B.sort_indices()
        self.colptr, self.rowidx, k2 = symbolic_cholesky(B)
        need = BYTES_PER_ENTRY * self.rowidx.size
        if need > memory_budget_bytes:
            raise ABPError("RESOURCE_MEMORY",
                           f"sparse LDL' needs ~{need / 2**30:.2f} GiB for {self.rowidx.size} "
                           f"factor entries; budget is {memory_budget_bytes / 2**30:.2f} GiB",
                           nnz_factor=int(self.rowidx.size))
        Bp = np.ascontiguousarray(B.indptr, dtype=np.int64)
        Bi = np.ascontiguousarray(B.indices, dtype=np.int64)
        Bx = np.ascontiguousarray(B.data, dtype=np.float64)
        # the dense block needs ~3 m^2 doubles (S, its factor, the copy from the kernel)
        cap = int(np.sqrt(max(memory_budget_bytes - need, 0) / 24.0))
        self.split = (dense_tail_split(self.colptr, self.n, min_size=DENSE_TAIL_MIN,
                                       max_size=min(DENSE_TAIL_MAX, cap),
                                       speed_ratio=DENSE_SPEED_RATIO)
                      if dense_tail else self.n)
        k3 = self._numeric(Bp, Bi, Bx)
        tail = (f", dense trailing block {self.n - self.split} (LAPACK)"
                if self.split < self.n else "")
        self.kernel = f"order {k1}, symbolic {k2}, numeric {k3}{tail}"
        self._selinv = None

    @property
    def nnz_factor(self) -> int:
        return int(self.rowidx.size)

    def _numeric(self, Bp, Bi, Bx) -> str:
        """Numeric LDL' of the permuted matrix into ``self.d`` and ``self.lval``; with a
        dense trailing block the leading columns and the Schur complement come from the
        up-looking kernel and the block from LAPACK.  Returns the kernel name."""
        split = self.split
        native = native_ldl_available()
        if native and split < self.n and not hasattr(_native, "ldl_numeric_split"):
            split = self.split = self.n                 # older compiled kernel
        if split < self.n:
            m = self.n - split
            if native:
                try:
                    raw_d, raw_l, raw_s = _native.ldl_numeric_split(Bp, Bi, Bx, self.colptr,
                                                                    self.rowidx, split)
                except ValueError as exc:
                    raise ABPError("FACTORIZATION_FAILED",
                                   f"sparse LDL' failed: {exc}") from None
                d = np.frombuffer(raw_d, dtype=np.float64).copy()
                lval = np.frombuffer(raw_l, dtype=np.float64).copy()
                S = np.frombuffer(raw_s, dtype=np.float64).reshape(m, m).copy()
            else:
                d, lval, S = ldl_numeric_python(Bp, Bi, Bx, self.colptr, self.rowidx, self.n,
                                                split=split)
            _dense_tail_factor(S, split, self.colptr, self.rowidx, d, lval)
            self.d, self.lval = d, lval
        elif native:
            try:
                raw_d, raw_l = _native.ldl_numeric(Bp, Bi, Bx, self.colptr, self.rowidx)
            except ValueError as exc:
                raise ABPError("FACTORIZATION_FAILED", f"sparse LDL' failed: {exc}") from None
            self.d = np.frombuffer(raw_d, dtype=np.float64).copy()
            self.lval = np.frombuffer(raw_l, dtype=np.float64).copy()
        else:
            self.d, self.lval = ldl_numeric_python(Bp, Bi, Bx, self.colptr, self.rowidx, self.n)
        return "native_cpp" if native else "python"

    def refactor(self, C: sp.spmatrix) -> None:
        """Numeric refactorization of a matrix with the **same sparsity pattern**
        (e.g. new variance ratios in a Gibbs sampler): ordering and symbolic
        pattern are reused; only ``d`` and ``L`` are recomputed."""
        C = _symmetric_pattern(C)
        if C.shape != self.C.shape:
            raise ValueError("refactor needs a matrix with the same pattern")
        if C.nnz != self.C.nnz or not (np.array_equal(C.indptr, self.C.indptr)
                                       and np.array_equal(C.indices, self.C.indices)):
            C = _on_pattern(C, self.C)          # entries that are exactly 0 this time
        B = sp.csc_matrix(C)[self.q][:, self.q]
        B.sort_indices()
        Bp = np.ascontiguousarray(B.indptr, dtype=np.int64)
        Bi = np.ascontiguousarray(B.indices, dtype=np.int64)
        Bx = np.ascontiguousarray(B.data, dtype=np.float64)
        self._numeric(Bp, Bi, Bx)
        self.C = C
        self._selinv = None

    def refactor_values(self, data: np.ndarray) -> None:
        """Numeric refactorization from the values of a matrix stored on **exactly**
        the pattern of ``self.C`` (CSR order of ``self.C.data``; the caller guarantees
        symmetry).  Skips the sparse re-assembly of :meth:`refactor`: the values are
        permuted into the factor's column order by an index computed once."""
        data = np.asarray(data, dtype=np.float64)
        if data.size != self.C.nnz:
            raise ValueError("refactor_values needs one value per stored entry of C")
        if getattr(self, "_bmap", None) is None:
            idx = sp.csr_matrix((np.arange(1, self.C.nnz + 1, dtype=np.float64),
                                 self.C.indices, self.C.indptr), shape=self.C.shape)
            B = sp.csc_matrix(idx)[self.q][:, self.q]
            B.sort_indices()
            self._bmap = B.data.astype(np.int64) - 1
            self._bp = np.ascontiguousarray(B.indptr, dtype=np.int64)
            self._bi = np.ascontiguousarray(B.indices, dtype=np.int64)
        Bx = np.ascontiguousarray(data[self._bmap])
        self._numeric(self._bp, self._bi, Bx)
        self.C = sp.csr_matrix((data, self.C.indices, self.C.indptr), shape=self.C.shape)
        self._selinv = None

    def l_times(self, v: np.ndarray) -> np.ndarray:
        """``L v`` in the permuted ordering (``L`` unit lower triangular)."""
        L = sp.csc_matrix((self.lval, self.rowidx, self.colptr), shape=(self.n, self.n))
        return v + L @ v

    def solve(self, b: np.ndarray) -> np.ndarray:
        b = np.asarray(b, dtype=np.float64)
        bp = b[self.q]
        if native_ldl_available():
            xb = np.ascontiguousarray(bp.T if bp.ndim == 2 else bp[None, :])
            raw = _native.ldl_solve(self.colptr, self.rowidx, self.lval, self.d, xb)
            xp = np.frombuffer(raw, dtype=np.float64).reshape(xb.shape)
            xp = xp.T if bp.ndim == 2 else xp[0]
        else:
            xp = ldl_solve_python(self.colptr, self.rowidx, self.lval, self.d, bp)
        return xp[self.pos]

    def logdet(self) -> float:
        return float(np.sum(np.log(self.d)))

    def selected_inverse(self):
        """Entries of ``C^{-1}`` on the factor pattern (Takahashi on this factor; cached)."""
        if self._selinv is None:
            self._selinv = _LDLSelectedInverse(self)
        return self._selinv

    def inverse_block(self, idx: np.ndarray, batch: int = 512) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int64)
        out = np.empty((idx.size, idx.size))
        for start in range(0, idx.size, batch):
            cols = idx[start:start + batch]
            E = np.zeros((self.n, cols.size))
            E[cols, np.arange(cols.size)] = 1.0
            out[:, start:start + cols.size] = self.solve(E)[idx, :]
        return 0.5 * (out + out.T)


class _LDLSelectedInverse:
    """Selected inverse from a :class:`SparseLDL` factor (same interface as SelectedInverse)."""

    def __init__(self, f: SparseLDL):
        from .selinv import SelectedInverse
        self.n = f.n
        self.pos = f.pos
        self.colptr, self.rowidx = f.colptr, f.rowidx
        self.zdiag, self.zval, k = takahashi(f.colptr, f.rowidx, f.lval, f.d)
        self.kernel = f"{f.kernel}; takahashi {k}"
        # reuse the lookup methods of SelectedInverse (same data layout)
        self._locate = SelectedInverse._locate.__get__(self)
        self.diagonal = SelectedInverse.diagonal.__get__(self)
        self.entries = SelectedInverse.entries.__get__(self)
        self.trace_product = SelectedInverse.trace_product.__get__(self)

    @property
    def nnz_factor(self) -> int:
        return int(self.rowidx.size)
