"""Pedigree ordering, inbreeding and the numerator relationship matrix A.

Scope and assumptions (method registry id ``ped.a_matrix``)
-----------------------------------------------------------
* Diploid, autosomal, additive inheritance; no unknown-parent groups and no
  metafounders (those are separate, not-yet-implemented extensions).
* Base population: every animal with an unknown parent contributes an
  independent, non-inbred base gamete.  **Different unknown parents are not
  one common ancestor** - each missing parent is a distinct unrelated founder
  gamete.  Consequently ``A`` is defined relative to that base.
* Animals are re-ordered so that parents precede offspring.  All matrices and
  vectors returned by :class:`Pedigree` use this internal order; use
  :attr:`Pedigree.ids` / :meth:`Pedigree.index_of` to map identifiers.

Mathematics
-----------
Write ``A = T D T'`` (Henderson 1976; Quaas 1976) where ``T = (I - P)^{-1}``,
``P[i, s_i] = P[i, d_i] = 1/2`` for known parents, and ``D`` is diagonal with
the Mendelian-sampling variance factors

    d_i = 1/2 - (F_s + F_d)/4    (both parents known)
    d_i = 3/4 - F_p/4            (one parent p known)
    d_i = 1                      (no parent known)

so that ``A^{-1} = (I - P)' D^{-1} (I - P)`` is sparse and ``log|A| = sum
log d_i`` (because ``|T| = 1``).  Inbreeding coefficients ``F_i = A_ii - 1``
use the algorithm of Meuwissen & Luo (1992, Genet. Sel. Evol. 24:305-313),
which traces the ancestors of each animal in decreasing internal order.
Products ``A x`` use the indirect method of Colleau (2002, Genet. Sel. Evol.
34:409-421): two sparse triangular solves and one diagonal scaling, never
forming ``A``.
"""

from __future__ import annotations

import heapq
import os
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve_triangular

from ..errors import ABPError

try:  # optional C++20 kernel (ADR 0002); the Python path is the reference
    from .. import _native  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - depends on the build environment
    _native = None

UNKNOWN = -1  #: internal code for an unknown parent


def native_kernel_available() -> bool:
    """True when the compiled kernel is importable and not disabled.

    Set the environment variable ``ABP_DISABLE_NATIVE=1`` to force the
    pure-Python reference kernels (e.g. for troubleshooting).
    """
    return _native is not None and os.environ.get("ABP_DISABLE_NATIVE", "") != "1"


def topological_order(parent_index: Sequence[tuple[int, int]]) -> np.ndarray:
    """Return an order in which every parent precedes its offspring.

    Ties are broken by the original position, so the result is deterministic
    and keeps the input order whenever the input is already valid.

    Parameters
    ----------
    parent_index:
        For each animal (by input position) the pair ``(sire_pos, dam_pos)``
        with ``-1`` for unknown.

    Raises
    ------
    ABPError(PEDIGREE_CYCLE)
        If the parent graph is not acyclic; ``details['animals']`` lists the
        input positions of the animals on (or downstream of) a cycle.
    """
    n = len(parent_index)
    indegree = np.zeros(n, dtype=np.int64)
    children: list[list[int]] = [[] for _ in range(n)]
    for i, (s, d) in enumerate(parent_index):
        for p in {s, d}:  # a set: s == d is rejected upstream (sex conflict)
            if p != UNKNOWN:
                indegree[i] += 1
                children[p].append(i)
    heap = [i for i in range(n) if indegree[i] == 0]
    heapq.heapify(heap)
    order: list[int] = []
    while heap:
        i = heapq.heappop(heap)
        order.append(i)
        for c in children[i]:
            indegree[c] -= 1
            if indegree[c] == 0:
                heapq.heappush(heap, c)
    if len(order) != n:
        stuck = sorted(set(range(n)) - set(order))
        raise ABPError("PEDIGREE_CYCLE",
                       f"{len(stuck)} animal(s) are on or below a pedigree cycle",
                       positions=stuck[:50], n_affected=len(stuck))
    return np.asarray(order, dtype=np.int64)


def inbreeding_meuwissen_luo(sire: np.ndarray, dam: np.ndarray) -> np.ndarray:
    """Inbreeding coefficients by Meuwissen & Luo (1992).

    ``sire``/``dam`` must be internal indices with parents preceding
    offspring and ``-1`` for unknown.  Complexity is proportional to the total
    number of (animal, ancestor) pairs traced; consecutive full sibs reuse the
    previous result.
    """
    n = sire.shape[0]
    # F_ext[n] = -1 encodes an unknown parent so that one formula gives d_i.
    F_ext = np.zeros(n + 1, dtype=np.float64)
    F_ext[n] = -1.0
    d = np.empty(n, dtype=np.float64)
    s_ext = np.where(sire < 0, n, sire)
    d_ext = np.where(dam < 0, n, dam)
    sire_l = sire.tolist()
    dam_l = dam.tolist()
    for i in range(n):
        si, di = s_ext[i], d_ext[i]
        d[i] = 0.5 - 0.25 * (F_ext[si] + F_ext[di])
        if si == n or di == n:
            F_ext[i] = 0.0  # one or both parents unknown -> not inbred
            continue
        if i > 0 and sire_l[i] == sire_l[i - 1] and dam_l[i] == dam_l[i - 1]:
            F_ext[i] = F_ext[i - 1]  # full sib of the previous animal
            continue
        # Trace row i of T: L[j] accumulates T[i, j] for ancestors j.
        coef = {i: 1.0}
        heap = [-i]
        acc = 0.0
        while heap:
            j = -heapq.heappop(heap)
            lj = coef.pop(j)
            acc += lj * lj * d[j]
            half = 0.5 * lj
            for p in (sire_l[j], dam_l[j]):
                if p >= 0:
                    if p in coef:
                        coef[p] += half
                    else:
                        coef[p] = half
                        heapq.heappush(heap, -p)
        F_ext[i] = acc - 1.0
    return F_ext[:n].copy()


def mendelian_factors(sire: np.ndarray, dam: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Diagonal ``D`` of ``A = T D T'`` (Mendelian sampling variance / sigma_a^2)."""
    Fs = np.where(sire >= 0, F[np.maximum(sire, 0)], -1.0)
    Fd = np.where(dam >= 0, F[np.maximum(dam, 0)], -1.0)
    return 0.5 - 0.25 * (Fs + Fd)


@dataclass(frozen=True)
class Pedigree:
    """An ordered, validated pedigree (parents precede offspring).

    Build instances with :meth:`from_parent_ids`; the constructor itself does
    not validate.  All numeric arrays use the internal order of :attr:`ids`.
    """

    ids: tuple[str, ...]
    sire: np.ndarray
    dam: np.ndarray
    generation: np.ndarray
    _index: dict[str, int] = field(repr=False, compare=False)
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    # ------------------------------------------------------------------ build
    @classmethod
    def from_parent_ids(cls, ids: Sequence[str], sires: Sequence[str | None],
                        dams: Sequence[str | None]) -> "Pedigree":
        """Create an ordered pedigree from identifier triplets.

        ``ids`` must be unique; every non-``None`` parent must itself appear
        in ``ids`` (the QC layer adds missing parents as founders *before*
        calling this).  Raises ``ABPError`` on duplicates, self-parenthood,
        unknown parents or cycles.
        """
        ids = list(ids)
        if not (len(ids) == len(sires) == len(dams)):
            raise ValueError("ids, sires and dams must have equal length")
        pos: dict[str, int] = {}
        for k, a in enumerate(ids):
            if a in pos:
                raise ABPError("PEDIGREE_CONFLICTING_DUPLICATE",
                               f"animal {a!r} appears more than once", animal=a)
            pos[a] = k
        pairs: list[tuple[int, int]] = []
        for k, (a, s, d) in enumerate(zip(ids, sires, dams)):
            sp_ = UNKNOWN if s is None else pos.get(s, None)
            dp_ = UNKNOWN if d is None else pos.get(d, None)
            if sp_ is None or dp_ is None:
                missing = s if sp_ is None else d
                raise ValueError(f"parent {missing!r} of {a!r} is not listed; "
                                 "add missing parents as founders first")
            if sp_ == k or dp_ == k:
                raise ABPError("PEDIGREE_SELF_PARENT",
                               f"animal {a!r} is recorded as its own parent", animal=a)
            if sp_ != UNKNOWN and sp_ == dp_:
                raise ABPError("PEDIGREE_SEX_CONFLICT",
                               f"animal {a!r} has the same animal {s!r} as sire and dam",
                               animal=a, parent=s)
            pairs.append((sp_, dp_))
        try:
            order = topological_order(pairs)
        except ABPError as exc:
            stuck = [ids[p] for p in exc.details.get("positions", [])]
            raise ABPError("PEDIGREE_CYCLE",
                           "pedigree contains a cycle; affected animals include "
                           + ", ".join(repr(x) for x in stuck[:10]),
                           animals=stuck, n_affected=exc.details.get("n_affected")) from None
        new_of_old = np.empty(len(ids), dtype=np.int64)
        new_of_old[order] = np.arange(len(ids))
        sire = np.full(len(ids), UNKNOWN, dtype=np.int64)
        dam = np.full(len(ids), UNKNOWN, dtype=np.int64)
        for new, old in enumerate(order):
            s, d = pairs[old]
            sire[new] = new_of_old[s] if s != UNKNOWN else UNKNOWN
            dam[new] = new_of_old[d] if d != UNKNOWN else UNKNOWN
        gen = np.zeros(len(ids), dtype=np.int64)
        for i in range(len(ids)):
            g = 0
            if sire[i] >= 0:
                g = max(g, gen[sire[i]] + 1)
            if dam[i] >= 0:
                g = max(g, gen[dam[i]] + 1)
            gen[i] = g
        ordered_ids = tuple(ids[o] for o in order)
        index = {a: i for i, a in enumerate(ordered_ids)}
        for arr in (sire, dam, gen):
            arr.setflags(write=False)
        return cls(ordered_ids, sire, dam, gen, index)

    # --------------------------------------------------------------- lookups
    @property
    def n(self) -> int:
        """Number of animals (including founders added for missing parents)."""
        return len(self.ids)

    def index_of(self, ids: Iterable[str]) -> np.ndarray:
        """Internal indices of the given identifiers (KeyError if absent)."""
        return np.fromiter((self._index[a] for a in ids), dtype=np.int64)

    def contains(self, animal: str) -> bool:
        return animal in self._index

    # ------------------------------------------------------ genetic quantities
    def inbreeding(self) -> np.ndarray:
        """Inbreeding coefficients ``F`` (read-only array, internal order).

        Uses the compiled Meuwissen-Luo kernel when available, otherwise the
        Python reference; :attr:`inbreeding_kernel` reports which one ran.
        """
        if "F" not in self._cache:
            if native_kernel_available():
                raw = _native.inbreeding_ml(np.ascontiguousarray(self.sire, dtype=np.int64),
                                            np.ascontiguousarray(self.dam, dtype=np.int64))
                F = np.frombuffer(raw, dtype=np.float64).copy()
                self._cache["F_kernel"] = "native_cpp_meuwissen_luo"
            else:
                F = inbreeding_meuwissen_luo(self.sire, self.dam)
                self._cache["F_kernel"] = "python_meuwissen_luo"
            F.setflags(write=False)
            self._cache["F"] = F
        return self._cache["F"]

    @property
    def inbreeding_kernel(self) -> str | None:
        """Name of the kernel that computed :meth:`inbreeding` (None if not yet run)."""
        return self._cache.get("F_kernel")

    def mendelian_d(self) -> np.ndarray:
        """Diagonal factors ``d_i`` of ``A = T D T'`` (read-only)."""
        if "D" not in self._cache:
            D = mendelian_factors(self.sire, self.dam, self.inbreeding())
            if np.any(D <= 0):  # impossible for valid F in [0,1); defensive
                raise ABPError("COVARIANCE_NOT_PD", "non-positive Mendelian sampling factor")
            D.setflags(write=False)
            self._cache["D"] = D
        return self._cache["D"]

    def logdet_a(self) -> float:
        """``log|A| = sum(log d_i)`` (exact; used by REML)."""
        return float(np.sum(np.log(self.mendelian_d())))

    def _l_matrix(self) -> sp.csr_matrix:
        """Sparse lower-triangular ``L = I - P`` (so that ``T = L^{-1}``)."""
        if "L" not in self._cache:
            n = self.n
            ks = np.flatnonzero(self.sire >= 0)
            kd = np.flatnonzero(self.dam >= 0)
            rows = np.concatenate([np.arange(n), ks, kd])
            cols = np.concatenate([np.arange(n), self.sire[ks], self.dam[kd]])
            vals = np.concatenate([np.ones(n), np.full(ks.size, -0.5), np.full(kd.size, -0.5)])
            L = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
            L.sort_indices()
            self._cache["L"] = L
            self._cache["LT"] = L.T.tocsr()
        return self._cache["L"]

    def ainv(self) -> sp.csr_matrix:
        """Sparse ``A^{-1} = L' D^{-1} L`` (Henderson/Quaas rules incl. inbreeding)."""
        if "Ainv" not in self._cache:
            n = self.n
            dinv = 1.0 / self.mendelian_d()
            # Each animal i contributes dinv_i * v v' with v = e_i - e_s/2 - e_d/2.
            members = [np.arange(n), self.sire, self.dam]
            weights = [np.ones(n), np.full(n, -0.5), np.full(n, -0.5)]
            rows, cols, vals = [], [], []
            for a in range(3):
                for b in range(3):
                    mask = (members[a] >= 0) & (members[b] >= 0)
                    rows.append(members[a][mask])
                    cols.append(members[b][mask])
                    vals.append(dinv[mask] * weights[a][mask] * weights[b][mask])
            Ainv = sp.csr_matrix((np.concatenate(vals),
                                  (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
            Ainv.sum_duplicates()
            Ainv.sort_indices()
            self._cache["Ainv"] = Ainv
        return self._cache["Ainv"]

    def a_times(self, x: np.ndarray) -> np.ndarray:
        """Compute ``A @ x`` without forming ``A`` (Colleau 2002).

        ``x`` may be a vector (n,) or a matrix (n, k).
        """
        L = self._l_matrix()
        LT = self._cache["LT"]
        x = np.asarray(x, dtype=np.float64)
        if x.shape[0] != self.n:
            raise ValueError(f"x has {x.shape[0]} rows, pedigree has {self.n} animals")
        v = spsolve_triangular(LT, x, lower=False, unit_diagonal=True)
        D = self.mendelian_d()
        w = v * (D if x.ndim == 1 else D[:, None])
        return spsolve_triangular(L, w, lower=True, unit_diagonal=True)

    def a_columns(self, idx: np.ndarray, block: int = 256) -> np.ndarray:
        """Dense columns ``A[:, idx]`` computed in blocks of ``block`` columns."""
        idx = np.asarray(idx, dtype=np.int64)
        out = np.empty((self.n, idx.size), dtype=np.float64)
        for start in range(0, idx.size, block):
            cols = idx[start:start + block]
            E = np.zeros((self.n, cols.size))
            E[cols, np.arange(cols.size)] = 1.0
            out[:, start:start + cols.size] = self.a_times(E)
        return out

    def a_submatrix(self, idx: np.ndarray) -> np.ndarray:
        """Dense ``A[idx][:, idx]`` (e.g. ``A22`` for genotyped animals)."""
        idx = np.asarray(idx, dtype=np.int64)
        sub = self.a_columns(idx)[idx, :]
        return 0.5 * (sub + sub.T)  # remove round-off asymmetry

    def a_dense(self, max_n: int = 20000) -> np.ndarray:
        """Full dense ``A``; refused above ``max_n`` animals to protect memory."""
        if self.n > max_n:
            raise ABPError("RESOURCE_MEMORY",
                           f"dense A for {self.n} animals exceeds the limit of {max_n}",
                           n=self.n, max_n=max_n)
        return self.a_submatrix(np.arange(self.n))

    def relationship(self, a: Sequence[str], b: Sequence[str]) -> np.ndarray:
        """Additive relationships ``A[a, b]`` between two lists of animal IDs."""
        ia, ib = self.index_of(a), self.index_of(b)
        return self.a_columns(ib)[ia, :]
