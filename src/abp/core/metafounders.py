"""Metafounders: related, possibly inbred base populations (method id ``ped.metafounders``).

Why
---
The ordinary pedigree relationship matrix ``A`` assumes that every unknown
parent is an unrelated, non-inbred base animal.  Genomic relationships built
with a fixed allele-frequency reference (``p = 0.5``) instead measure
relationships relative to an older, *related* base.  Combining the two in
single step then mixes two genetic bases, which biases EBVs of genotyped
animals and understates their PEV (finding F6 of the validation report).
Metafounders (Legarra, Christensen, Aguilar, Vitezica & Misztal 2015,
Genetics 200:455-468; Christensen 2012 for one metafounder) replace each base
population by a pseudo-individual whose self-relationship ``gamma`` expresses
the relatedness of the base, so both relationship matrices refer to the same
base.

Model contract
--------------
* ``k`` metafounders with symmetric positive-definite relationship matrix
  ``Gamma`` (k x k).  ``Gamma`` must come from a documented estimate (e.g.
  :func:`estimate_gamma_gls`) or a user file with stated provenance; it is
  never assigned from breed names.
* Every unknown parent of every animal is assigned to one metafounder.  A
  plain unknown parent would be a metafounder with ``gamma = 0``, whose
  inverse does not exist, so unassigned unknown parents are refused.
* Breeding values ``u`` of animals and ``u_mf`` of metafounders satisfy
  ``u_i = (u_s + u_d)/2 + m_i`` where a parent may be a metafounder, and
  ``u_mf ~ N(0, sigma^2 Gamma)``.  The extended relationship matrix over
  (animals, metafounders) is defined recursively (Legarra et al. 2015):

      A[mf, mf'] = Gamma[mf, mf'],
      A[i, j]    = (A[j, s_i] + A[j, d_i]) / 2    (j older than i),
      A[i, i]    = 1 + A[s_i, d_i] / 2.

Computation (derived for this implementation)
---------------------------------------------
Write the animal-only transmission matrix ``T = (I - P)^{-1}`` (``P`` holds
1/2 for *animal* parents) and ``Q`` the expected gene fractions from each
metafounder (:func:`abp.core.upg.group_fractions`).  Then

    A^Gamma = T D T' + Q Gamma Q',
    d_i = 1 - (A_ss + A_dd) / 4,     A_pp := Gamma_pp for a metafounder parent,

so that ``Var(m_i) = sigma^2 d_i``; for a founder whose two parents come from
metafounder ``p`` this gives ``d_i = 1 - Gamma_pp / 2`` and
``A_ii = 1 + Gamma_pp / 2``.  Henderson's rules with metafounders acting as
parents plus ``Gamma^{-1}`` on the metafounder block give the sparse inverse of
the extended matrix, and ``log|A_ext| = log|Gamma| + sum_i log d_i``.
``diag(A^Gamma)`` uses a generalised Meuwissen-Luo trace
(:func:`ml_general`): ``A_ii = 1 + A_sd/2`` directly for animals with a
metafounder parent, and ``A_ii = sum_j T_ij^2 d_j + q_i' Gamma q_i`` otherwise.

With ``Gamma = 0`` and every metafounder parent read as "unknown" the
formulae reduce to the ordinary ``A`` (this is used as a regression test).
"""

from __future__ import annotations

import csv
import heapq
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve_triangular

from ..errors import ABPError
from .pedigree import Pedigree, native_kernel_available
from .upg import GroupAssignment, group_fractions

try:  # optional C++20 kernel (ADR 0002)
    from .. import _native  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    _native = None

#: smallest admissible Mendelian sampling factor d_i (below: Gamma inconsistent with pedigree)
MIN_MENDELIAN_FACTOR = 1e-8
#: relative eigenvalue threshold for Gamma and for the GLS information matrix
EIG_RTOL = 1e-10


# ----------------------------------------------------------------- kernels
def ml_general_python(sire: np.ndarray, dam: np.ndarray, c: np.ndarray, e: np.ndarray,
                      fext: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Generalised Meuwissen-Luo trace (Python reference of ``_native.ml_general``).

    For animals in order (parents first):

    * ``d_i = 1 - (A_s + A_d)/4 - c_i`` where ``A_s`` is the diagonal of an
      *animal* sire and 0 otherwise (``c_i`` carries metafounder parents);
    * if a parent is not an animal: ``A_ii = 1 + fext_i``;
    * otherwise ``A_ii = sum_j T_ij^2 d_j + e_i`` over animal ancestors ``j``
      (including ``i``) - the ordinary trace plus the metafounder term.

    With ``c = e = fext = 0`` this is the ordinary Meuwissen & Luo (1992)
    algorithm and returns ``(1 + F, d)``.  Returns ``(diag_A, d)``.
    """
    n = sire.shape[0]
    adiag = np.zeros(n)
    d = np.zeros(n)
    sire_l, dam_l = sire.tolist(), dam.tolist()
    family: dict[tuple[int, int], float] = {}
    for i in range(n):
        s, m = sire_l[i], dam_l[i]
        a_s = adiag[s] if s >= 0 else 0.0
        a_m = adiag[m] if m >= 0 else 0.0
        d[i] = 1.0 - 0.25 * (a_s + a_m) - c[i]
        if s < 0 or m < 0:
            adiag[i] = 1.0 + fext[i]
            continue
        key = (min(s, m), max(s, m))
        if key in family:                       # full sibs share the diagonal
            adiag[i] = family[key]
            continue
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
        adiag[i] = acc + e[i]
        family[key] = adiag[i]
    return adiag, d


def ml_general(sire: np.ndarray, dam: np.ndarray, c: np.ndarray, e: np.ndarray,
               fext: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """Dispatch to the C++ kernel when available; returns ``(diag_A, d, kernel_name)``."""
    if native_kernel_available() and hasattr(_native, "ml_general"):
        arrs = [np.ascontiguousarray(x, dtype=np.int64) for x in (sire, dam)]
        arrs += [np.ascontiguousarray(x, dtype=np.float64) for x in (c, e, fext)]
        raw = np.frombuffer(_native.ml_general(*arrs), dtype=np.float64)
        n = sire.shape[0]
        return raw[:n].copy(), raw[n:].copy(), "native_cpp_ml_general"
    a, d = ml_general_python(sire, dam, c, e, fext)
    return a, d, "python_ml_general"


# --------------------------------------------------------------- Gamma input
def validate_gamma(gamma: np.ndarray, labels: tuple[str, ...]) -> np.ndarray:
    """Check that ``Gamma`` is finite, symmetric and positive definite."""
    g = np.asarray(gamma, dtype=np.float64)
    k = len(labels)
    if g.shape != (k, k):
        raise ABPError("SPEC_INVALID", f"Gamma must be {k} x {k} for metafounders {list(labels)}")
    if not np.all(np.isfinite(g)):
        raise ABPError("SPEC_INVALID", "Gamma contains non-finite values")
    if not np.allclose(g, g.T, rtol=0, atol=1e-12):
        raise ABPError("SPEC_INVALID", "Gamma must be symmetric")
    g = 0.5 * (g + g.T)
    ev = np.linalg.eigvalsh(g)
    if not ev[0] > EIG_RTOL * max(ev[-1], 1.0):
        raise ABPError("RELATIONSHIP_SINGULAR",
                       f"Gamma is not positive definite (smallest eigenvalue {ev[0]:.3e}); "
                       "metafounder relationships must form a valid covariance matrix",
                       min_eigenvalue=float(ev[0]))
    return g


def read_gamma_file(path: Path, labels: tuple[str, ...]) -> np.ndarray:
    """Read ``Gamma`` from a CSV with columns ``metafounder_1, metafounder_2, gamma``.

    Each unordered pair may appear once (either order); every diagonal element
    is required and missing off-diagonal pairs are refused rather than set to 0.
    """
    idx = {lab: k for k, lab in enumerate(labels)}
    k = len(labels)
    g = np.full((k, k), np.nan)
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        need = {"metafounder_1", "metafounder_2", "gamma"}
        if reader.fieldnames is None or not need <= set(reader.fieldnames):
            raise ABPError("SCHEMA_MISSING_COLUMN", f"{path.name} needs columns {sorted(need)}",
                           path=str(path))
        for line, row in enumerate(reader, start=2):
            a, b = row["metafounder_1"].strip(), row["metafounder_2"].strip()
            if a not in idx or b not in idx:
                raise ABPError("SPEC_INVALID", f"{path.name} line {line}: unknown metafounder "
                               f"{a if a not in idx else b!r} (pedigree uses {list(labels)})")
            if not np.isnan(g[idx[a], idx[b]]):
                raise ABPError("SPEC_INVALID", f"{path.name} line {line}: pair ({a}, {b}) repeated")
            try:
                v = float(row["gamma"])
            except ValueError:
                raise ABPError("SPEC_INVALID", f"{path.name} line {line}: gamma is not a number") from None
            g[idx[a], idx[b]] = g[idx[b], idx[a]] = v
    if np.isnan(g).any():
        miss = [(labels[i], labels[j]) for i in range(k) for j in range(i, k) if np.isnan(g[i, j])]
        raise ABPError("SPEC_INVALID", f"{path.name} lacks Gamma entries for {miss[:10]}",
                       missing=[list(x) for x in miss[:50]])
    return validate_gamma(g, labels)


# ---------------------------------------------------------------- structure
@dataclass
class MetafounderPedigree:
    """Pedigree + metafounder assignment + ``Gamma`` with the derived quantities.

    Equation order everywhere: animals in pedigree order, then metafounders
    in ``groups.labels`` order.
    """

    ped: Pedigree
    groups: GroupAssignment
    gamma: np.ndarray
    Q: np.ndarray = field(init=False, repr=False)
    adiag: np.ndarray = field(init=False, repr=False)
    d: np.ndarray = field(init=False, repr=False)
    kernel: str = field(init=False)

    def __post_init__(self):
        ped, grp = self.ped, self.groups
        labels = tuple(grp.labels)
        if not labels:
            raise ABPError("SPEC_INVALID", "no metafounder codes were found in the pedigree")
        self.gamma = validate_gamma(self.gamma, labels)
        unassigned = np.flatnonzero(((ped.sire < 0) & (grp.sire_group < 0))
                                    | ((ped.dam < 0) & (grp.dam_group < 0)))
        if unassigned.size:
            raise ABPError("PEDIGREE_UNASSIGNED_BASE",
                           f"{unassigned.size} animal(s) have an unknown parent that is not "
                           "assigned to a metafounder; code those parents or set "
                           "metafounders.default",
                           animals=[ped.ids[i] for i in unassigned[:20]],
                           n_affected=int(unassigned.size))
        self.Q = group_fractions(ped, grp)
        G = self.gamma
        gdiag = np.diag(G)
        sg, dg = grp.sire_group, grp.dam_group
        smf, dmf = ped.sire < 0, ped.dam < 0
        c = 0.25 * (np.where(smf, gdiag[np.maximum(sg, 0)], 0.0)
                    + np.where(dmf, gdiag[np.maximum(dg, 0)], 0.0))
        QG = self.Q @ G
        e = np.einsum("ij,ij->i", QG, self.Q)
        # F_i = A_sd / 2 for animals with at least one metafounder parent
        a_sd = np.zeros(ped.n)
        both = smf & dmf
        a_sd[both] = G[sg[both], dg[both]]
        one_s = smf & ~dmf                       # sire is a metafounder, dam an animal
        a_sd[one_s] = QG[ped.dam[one_s], sg[one_s]]
        one_d = dmf & ~smf
        a_sd[one_d] = QG[ped.sire[one_d], dg[one_d]]
        fext = 0.5 * a_sd
        self.adiag, self.d, self.kernel = ml_general(ped.sire, ped.dam, c, e, fext)
        bad = np.flatnonzero(self.d < MIN_MENDELIAN_FACTOR)
        if bad.size:
            raise ABPError("RELATIONSHIP_SINGULAR",
                           f"Gamma implies a non-positive Mendelian sampling variance for "
                           f"{bad.size} animal(s) (smallest d = {self.d.min():.3e}); Gamma is "
                           "inconsistent with this pedigree (self-relationships too large)",
                           animals=[ped.ids[i] for i in bad[:20]])

    # -- sizes and labels ---------------------------------------------------
    @property
    def n(self) -> int:
        return self.ped.n

    @property
    def k(self) -> int:
        return len(self.groups.labels)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(self.ped.ids) + tuple(self.groups.labels)

    def inbreeding(self) -> np.ndarray:
        """``F_i = A_ii - 1`` relative to the metafounder base (may be > 0 for founders)."""
        return self.adiag - 1.0

    # -- inverse and determinant ------------------------------------------
    def ainv_ext(self) -> sp.csr_matrix:
        """Sparse inverse of the extended relationship matrix (animals, metafounders)."""
        ped, grp = self.ped, self.groups
        n, k = self.n, self.k
        dinv = 1.0 / self.d
        s_col = np.where(ped.sire >= 0, ped.sire, n + grp.sire_group)
        d_col = np.where(ped.dam >= 0, ped.dam, n + grp.dam_group)
        members = [np.arange(n), s_col, d_col]
        weights = [1.0, -0.5, -0.5]
        rows, cols, vals = [], [], []
        for a in range(3):
            for b in range(3):
                rows.append(members[a])
                cols.append(members[b])
                vals.append(dinv * weights[a] * weights[b])
        gi = np.linalg.inv(self.gamma)
        rr, cc = np.meshgrid(np.arange(n, n + k), np.arange(n, n + k), indexing="ij")
        rows.append(rr.ravel())
        cols.append(cc.ravel())
        vals.append(gi.ravel())
        M = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(n + k, n + k))
        M.sum_duplicates()
        M.sort_indices()
        return M

    def logdet_ext(self) -> float:
        """``log|A_ext| = log|Gamma| + sum log d_i``."""
        sign, ld = np.linalg.slogdet(self.gamma)
        return float(ld + np.sum(np.log(self.d)))

    def cov_mf(self) -> np.ndarray:
        """``A_ext[:, metafounders]`` ((n + k) x k): ``Q Gamma`` for animals, ``Gamma`` below."""
        return np.vstack([self.Q @ self.gamma, self.gamma])

    def diag_ext(self) -> np.ndarray:
        """``diag(A_ext)``: animals then metafounders (``Gamma_pp``)."""
        return np.concatenate([self.adiag, np.diag(self.gamma)])

    # -- products ------------------------------------------------------------
    def a_times(self, x: np.ndarray) -> np.ndarray:
        """``A^Gamma x`` over animals: ``T D T' x + Q Gamma Q' x`` (never forms A)."""
        ped = self.ped
        L = ped._l_matrix()
        LT = ped._cache["LT"]
        x = np.asarray(x, dtype=np.float64)
        v = spsolve_triangular(LT, x, lower=False, unit_diagonal=True)
        w = v * (self.d if x.ndim == 1 else self.d[:, None])
        out = spsolve_triangular(L, w, lower=True, unit_diagonal=True)
        return out + self.Q @ (self.gamma @ (self.Q.T @ x))

    def ext_columns(self, idx: np.ndarray, block: int = 256) -> np.ndarray:
        """Dense columns ``A_ext[:, idx]`` for *animal* indices ``idx`` ((n + k) x len(idx))."""
        idx = np.asarray(idx, dtype=np.int64)
        n = self.n
        out = np.empty((n + self.k, idx.size))
        for start in range(0, idx.size, block):
            cols = idx[start:start + block]
            E = np.zeros((n, cols.size))
            E[cols, np.arange(cols.size)] = 1.0
            out[:n, start:start + cols.size] = self.a_times(E)
        out[n:, :] = self.gamma @ self.Q[idx].T          # A[mf, i] = (Gamma q_i)_mf
        return out

    def a_submatrix(self, idx: np.ndarray) -> np.ndarray:
        """Dense ``A^Gamma[idx][:, idx]`` (e.g. ``A22`` of genotyped animals)."""
        idx = np.asarray(idx, dtype=np.int64)
        sub = self.ext_columns(idx)[idx]
        return 0.5 * (sub + sub.T)

    def a_ext_dense(self, max_n: int = 20000) -> np.ndarray:
        """Full dense extended matrix (tests and small examples only)."""
        if self.n + self.k > max_n:
            raise ABPError("RESOURCE_MEMORY", f"dense A_ext above {max_n} equations refused")
        cols = self.ext_columns(np.arange(self.n))
        full = np.empty((self.n + self.k, self.n + self.k))
        full[:, :self.n] = cols
        full[:self.n, self.n:] = cols[self.n:].T
        full[self.n:, self.n:] = self.gamma
        return 0.5 * (full + full.T)


# ---------------------------------------------------------- Gamma estimation
@dataclass
class GammaEstimate:
    """Result of :func:`estimate_gamma_gls` (all quantities are written to the manifest)."""

    gamma: np.ndarray
    gamma_uncorrected: np.ndarray
    base_frequencies: np.ndarray        # k x m
    information_inverse: np.ndarray     # C = (Q2' A22^-1 Q2)^-1
    sampling_correction: bool
    n_clipped: int
    missing_iterations: int
    notes: list[str]

    def to_dict(self) -> dict:
        return {"gamma": self.gamma.tolist(), "gamma_uncorrected": self.gamma_uncorrected.tolist(),
                "gls_information_inverse": self.information_inverse.tolist(),
                "sampling_correction": self.sampling_correction,
                "n_base_frequencies_clipped": self.n_clipped,
                "missing_dosage_iterations": self.missing_iterations,
                "n_markers": int(self.base_frequencies.shape[1]),
                "mean_base_frequency": self.base_frequencies.mean(axis=1).tolist(),
                "notes": self.notes}


def estimate_gamma_gls(ped: Pedigree, groups: GroupAssignment, geno_index: np.ndarray,
                       M: np.ndarray, missing: np.ndarray | None = None,
                       sampling_correction: bool = True, tol: float = 1e-10,
                       max_iter: int = 500) -> GammaEstimate:
    """Estimate ``Gamma`` from genotypes via GLS base allele frequencies.

    Method (Gengler, Mayeres & Szydlowski 2007 for GLS base frequencies;
    Garcia-Baccino et al. 2017, Genet Sel Evol 49:34 for ``Gamma``):

    * expected dosages of genotyped animals are ``E[M] = 2 Q2 P`` with ``P``
      (k x m) the base frequencies of each metafounder population, and the
      covariance of one marker's dosages is approximately ``2p(1-p) A22``
      where ``A22`` is the ordinary pedigree matrix of genotyped animals;
    * GLS: ``P_hat = C Q2' A22^{-1} M / 2`` with ``C = (Q2' A22^{-1} Q2)^{-1}``;
    * ``Gamma_raw = 8 (P_hat - 0.5)(P_hat - 0.5)' / m``, which matches genomic
      relationships built with allele frequency 0.5 (``G05``);
    * ``Var(P_hat_{.j}) = p_j(1-p_j) C / 2`` inflates ``Gamma_raw`` by
      ``4 mean(p(1-p)) C``; with ``sampling_correction`` this expectation is
      subtracted (a pooled mean over metafounders and markers; derived here).

    Missing dosages are filled by their expectation ``2 Q2 P_hat`` and the
    estimate is iterated to a fixed point.  Frequencies outside [0, 1] are
    clipped (counted).  ABP refuses (``MODEL_NOT_IDENTIFIABLE``) when a
    metafounder is not estimable from the genotyped animals, and
    (``RELATIONSHIP_SINGULAR``) when the corrected ``Gamma`` is not positive
    definite, rather than repairing it.
    """
    geno_index = np.asarray(geno_index, dtype=np.int64)
    M = np.asarray(M, dtype=np.float64)
    labels = tuple(groups.labels)
    k = len(labels)
    Q2 = group_fractions(ped, groups)[geno_index]
    A22 = ped.a_submatrix(geno_index)
    try:
        cf = sla.cho_factor(A22, lower=True)
    except np.linalg.LinAlgError:
        raise ABPError("RELATIONSHIP_SINGULAR", "A22 is not positive definite") from None
    AiQ = sla.cho_solve(cf, Q2)
    info = Q2.T @ AiQ
    ev = np.linalg.eigvalsh(info)
    if not ev[0] > EIG_RTOL * max(ev[-1], 1e-300):
        contrib = [labels[j] for j in range(k) if Q2[:, j].sum() == 0.0]
        raise ABPError("MODEL_NOT_IDENTIFIABLE",
                       "base allele frequencies of the metafounders are not estimable from the "
                       "genotyped animals" + (f" (no genotyped descendants: {contrib})" if contrib
                                              else " (gene fractions are collinear)"),
                       metafounders=contrib, smallest_eigenvalue=float(ev[0]))
    C = np.linalg.inv(info)
    C = 0.5 * (C + C.T)
    B = C @ AiQ.T                                     # k x n2
    X = M.copy()
    iters = 0
    if missing is not None and missing.any():
        obs = ~missing
        colmean = np.where(obs, M, 0.0).sum(0) / np.maximum(obs.sum(0), 1)
        X = np.where(missing, colmean[None, :], M)
        for iters in range(1, max_iter + 1):
            P = 0.5 * (B @ X)
            Xn = np.where(missing, 2.0 * (Q2 @ P), M)
            delta = float(np.max(np.abs(Xn - X)))
            X = Xn
            if delta < tol:
                break
        else:
            raise ABPError("SOLVER_NOT_CONVERGED", "missing-dosage iteration of base frequencies did "
                           f"not converge in {max_iter} iterations", last_change=delta)
    P = 0.5 * (B @ X)
    n_clip = int(np.count_nonzero((P < 0) | (P > 1)))
    P = np.clip(P, 0.0, 1.0)
    m = P.shape[1]
    Dv = P - 0.5
    g_raw = 8.0 * (Dv @ Dv.T) / m
    g_raw = 0.5 * (g_raw + g_raw.T)
    notes = ["Gamma refers to genomic relationships computed with allele frequency 0.5 "
             "(G = (M - 1)(M - 1)' / (m/2)); use genomic.frequency_source = 'fixed_0.5'."]
    if n_clip:
        notes.append(f"{n_clip} base-frequency estimates outside [0, 1] were clipped")
    g = g_raw
    if sampling_correction:
        s = float(np.mean(P * (1.0 - P)))
        g = g_raw - 4.0 * s * C
        notes.append(f"sampling correction subtracted 4 * {s:.6f} * C (pooled p(1-p))")
    try:
        g = validate_gamma(g, labels)
    except ABPError as exc:
        raise ABPError("RELATIONSHIP_SINGULAR",
                       "the estimated Gamma is not positive definite"
                       + (" after the sampling correction (too few genotyped descendants or "
                          "markers per metafounder); merge metafounders or supply Gamma from "
                          "a documented file" if sampling_correction else ""),
                       gamma=g.tolist(), gamma_uncorrected=g_raw.tolist()) from exc
    return GammaEstimate(g, g_raw, P, C, sampling_correction, n_clip, iters, notes)


# ---------------------------------------------------------------- single step
@dataclass
class SingleStepMF:
    h_inv: sp.csr_matrix       # (n + k) x (n + k)
    h_diag: np.ndarray
    logdet_h: float
    a22: np.ndarray
    cov_mf: np.ndarray         # H[:, metafounders], (n + k) x k


def single_step_mf(mfp: MetafounderPedigree, geno_index: np.ndarray, Gstar: np.ndarray,
                   a22: np.ndarray | None = None,
                   max_dense_bytes: int = 2 * 2**30) -> SingleStepMF:
    """Single-step ``H^{-1}`` over (animals, metafounders) with ``A^Gamma``.

    ``H^{-1} = A_ext^{-1} + embed(G*^{-1} - (A22^Gamma)^{-1})`` (Legarra et
    al. 2015, eq. for ssGBLUP with metafounders), where ``G*`` must be on the
    metafounder base (``G05``).  ``diag(H)`` of non-genotyped equations uses
    ``H_ii = A_ii + b_i'(G* - A22) b_i`` with ``b_i = A22^{-1} A[2, i]``, and
    ``log|H| = log|A_ext| + log|G*| - log|A22|``.
    """
    from .genomic import spd_inverse_and_logdet
    geno_index = np.asarray(geno_index, dtype=np.int64)
    n2 = geno_index.size
    n, k = mfp.n, mfp.k
    if Gstar.shape != (n2, n2):
        raise ValueError("G* does not match the genotyped index")
    if 8 * (n + k) * n2 > max_dense_bytes:
        raise ABPError("RESOURCE_MEMORY",
                       f"single-step diag(H) needs an {n + k} x {n2} dense block "
                       f"(~{8 * (n + k) * n2 / 2**30:.2f} GiB)", n=n, n_genotyped=n2)
    cols = mfp.ext_columns(geno_index)
    if a22 is None:
        a22 = 0.5 * (cols[geno_index] + cols[geno_index].T)
    a22_inv, logdet_a22 = spd_inverse_and_logdet(a22, "A22 (metafounder base)")
    g_inv, logdet_g = spd_inverse_and_logdet(Gstar, "G*")
    block = g_inv - a22_inv
    rr, cc = np.meshgrid(geno_index, geno_index, indexing="ij")
    embed = sp.csr_matrix((block.ravel(), (rr.ravel(), cc.ravel())), shape=(n + k, n + k))
    h_inv = (mfp.ainv_ext() + embed).tocsr()
    h_inv.sum_duplicates()
    h_inv.sort_indices()
    Bm = cols @ a22_inv
    BmD = Bm @ (Gstar - a22)
    h_diag = mfp.diag_ext() + np.einsum("ij,ij->i", BmD, Bm)
    h_diag[geno_index] = np.diag(Gstar)
    # H[:, p] = A[:, p] + b' (G* - A22) b_p: needed for contrasts with a metafounder
    cov_mf = mfp.cov_mf() + BmD @ Bm[n:].T
    logdet_h = mfp.logdet_ext() + logdet_g - logdet_a22
    return SingleStepMF(h_inv, h_diag, logdet_h, a22, cov_mf)
