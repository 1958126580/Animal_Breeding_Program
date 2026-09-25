"""Genomic relationships (VanRaden G), G policies and the single-step H^{-1}.

Model contract (method registry ids ``gen.g_vanraden1``, ``gen.h_inverse``)
--------------------------------------------------------------------------
Diploid, autosomal, additive.  ``M`` holds dosages (0..2 copies) of the
declared *counted* allele; ``p_j`` is the counted-allele frequency from a
declared reference sample.  With ``W = M - 2p`` (missing dosages set to the
reference mean, i.e. ``W_ij = 0``)

    G = W W' / d,        d = 2 sum_j p_j (1 - p_j)       (VanRaden 2008, method 1).

``d = 0`` is refused.  ``G`` is symmetric positive *semi*-definite and is
singular whenever the number of animals exceeds the rank of ``W``; ABP never
repairs it silently.  An explicit ``singular_policy`` must be chosen:

* ``blend``: ``G* = (1 - alpha) G + alpha A22`` (needs a pedigree);
* ``ridge``: ``G* = G + lambda I``;
* ``error`` (default): stop with the smallest eigenvalue reported.

Optional tuning ``match_a22`` (Christensen et al. 2012) rescales
``G* <- a + b G*`` so that the mean diagonal and mean off-diagonal equal
those of ``A22``.  All adjustments and their magnitudes are recorded.

Single step (Aguilar et al. 2010; Christensen & Lund 2010):

    H^{-1} = A^{-1} + [[0, 0], [0, G*^{-1} - A22^{-1}]],

where ``A22^{-1}`` is the inverse of the pedigree relationship sub-matrix of
the genotyped animals (computed from ``A22`` itself; it is *not* the 22-block
of ``A^{-1}``).  ``|H| = |A| |G*| / |A22|`` gives ``log|H|`` for REML.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

from ..errors import ABPError
from .pedigree import Pedigree


def allele_frequencies(M: np.ndarray, missing: np.ndarray | None = None) -> np.ndarray:
    """Counted-allele frequency per marker from non-missing dosages."""
    if missing is None:
        return M.mean(axis=0) / 2.0
    obs = ~missing
    n = obs.sum(axis=0)
    s = np.where(obs, M, 0.0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = s / (2.0 * n)
    return np.where(n > 0, p, np.nan)


def minor_allele_frequency(p: np.ndarray) -> np.ndarray:
    """MAF = min(p, 1 - p) in [0, 0.5] (the minimum; see spec erratum T01)."""
    p = np.asarray(p, dtype=np.float64)
    return np.minimum(p, 1.0 - p)


def centered(M: np.ndarray, p: np.ndarray, missing: np.ndarray | None = None) -> np.ndarray:
    """``W = M - 2p``; missing dosages contribute 0 (mean imputation at ``p``)."""
    W = M - 2.0 * p[None, :]
    if missing is not None:
        W = np.where(missing, 0.0, W)
    return W


def scaling_d(p: np.ndarray) -> float:
    d = float(2.0 * np.sum(p * (1.0 - p)))
    if not d > 0.0:
        raise ABPError("GENOTYPE_ZERO_SCALING",
                       "2*sum(p*(1-p)) is zero: no polymorphic markers in the reference sample")
    return d


def vanraden_g(M: np.ndarray, p: np.ndarray, missing: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Return ``(G, d)`` for VanRaden's first method."""
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ABPError("GENOTYPE_ALLELE_MISMATCH", "allele frequencies must be finite and in [0, 1]")
    d = scaling_d(p)
    W = centered(M, p, missing)
    G = (W @ W.T) / d
    return 0.5 * (G + G.T), d


@dataclass
class GPolicyRecord:
    """What was done to G before use (written to the manifest)."""

    singular_policy: str
    tuning: str
    blend_alpha: float | None = None
    ridge: float | None = None
    tuning_a: float | None = None
    tuning_b: float | None = None
    min_eigenvalue_before: float | None = None
    min_eigenvalue_after: float | None = None
    notes: list[str] = field(default_factory=list)


def _min_eig(S: np.ndarray) -> float:
    return float(sla.eigvalsh(S, subset_by_index=[0, 0])[0])


def tune_to_a22(G: np.ndarray, A22: np.ndarray) -> tuple[np.ndarray, float, float]:
    """``G* = a + b G`` matching mean diagonal and mean off-diagonal of ``A22``."""
    n = G.shape[0]
    if n < 2:
        raise ABPError("MODEL_NOT_IDENTIFIABLE", "tuning needs at least two genotyped animals")
    off = ~np.eye(n, dtype=bool)
    gd, go = np.mean(np.diag(G)), np.mean(G[off])
    ad, ao = np.mean(np.diag(A22)), np.mean(A22[off])
    if abs(gd - go) < 1e-12:
        raise ABPError("MODEL_NOT_IDENTIFIABLE", "G has equal mean diagonal and off-diagonal")
    b = (ad - ao) / (gd - go)
    a = ad - b * gd
    return a + b * G, float(a), float(b)


def apply_g_policy(G: np.ndarray, policy: str, tuning: str, A22: np.ndarray | None,
                   alpha: float, ridge: float) -> tuple[np.ndarray, GPolicyRecord]:
    """Apply the declared tuning and singularity policy; verify positive definiteness."""
    rec = GPolicyRecord(singular_policy=policy, tuning=tuning)
    Gs = G.copy()
    if tuning == "match_a22":
        if A22 is None:
            raise ABPError("UNSUPPORTED_COMBINATION", "genomic.tuning = 'match_a22' needs a pedigree")
        Gs, rec.tuning_a, rec.tuning_b = tune_to_a22(Gs, A22)
    rec.min_eigenvalue_before = _min_eig(Gs)
    if policy == "blend":
        if A22 is None:
            raise ABPError("UNSUPPORTED_COMBINATION",
                           "genomic.singular_policy = 'blend' needs a pedigree (A22)")
        Gs = (1.0 - alpha) * Gs + alpha * A22
        rec.blend_alpha = alpha
    elif policy == "ridge":
        Gs = Gs + ridge * np.eye(Gs.shape[0])
        rec.ridge = ridge
    elif policy != "error":
        raise ABPError("SPEC_INVALID", f"unknown genomic.singular_policy {policy!r}")
    rec.min_eigenvalue_after = _min_eig(Gs)
    try:
        sla.cholesky(Gs, lower=True)
    except np.linalg.LinAlgError:
        raise ABPError("RELATIONSHIP_SINGULAR",
                       f"G is not positive definite (smallest eigenvalue "
                       f"{rec.min_eigenvalue_after:.3e}); with {G.shape[0]} genotyped animals "
                       "choose genomic.singular_policy = 'blend' or 'ridge' explicitly",
                       min_eigenvalue=rec.min_eigenvalue_after, n=G.shape[0]) from None
    return Gs, rec


def spd_inverse_and_logdet(S: np.ndarray, what: str) -> tuple[np.ndarray, float]:
    """Inverse and log-determinant of an SPD matrix via Cholesky (LAPACK potri)."""
    try:
        L = sla.cholesky(S, lower=True)
    except np.linalg.LinAlgError:
        raise ABPError("RELATIONSHIP_SINGULAR", f"{what} is not positive definite") from None
    inv, info = sla.lapack.dpotri(L, lower=1)
    if info != 0:
        raise ABPError("RELATIONSHIP_SINGULAR", f"inversion of {what} failed (info={info})")
    inv = np.tril(inv) + np.tril(inv, -1).T
    return inv, float(2.0 * np.sum(np.log(np.diag(L))))


@dataclass
class SingleStep:
    """Single-step structure in pedigree order."""

    h_inv: sp.csr_matrix
    h_diag: np.ndarray
    logdet_h: float
    geno_index: np.ndarray       # pedigree indices of genotyped animals (G order)
    a22: np.ndarray
    a22_inv: np.ndarray


def single_step(ped: Pedigree, geno_index: np.ndarray, Gstar: np.ndarray,
                a22: np.ndarray | None = None, max_dense_bytes: int = 2 * 2**30) -> SingleStep:
    """Build ``H^{-1}``, ``diag(H)`` and ``log|H|``.

    ``Gstar`` must be ordered like ``geno_index``.  ``diag(H)`` for
    non-genotyped animals uses ``H_ii = A_ii + b_i'(G* - A22) b_i`` with
    ``b_i = A22^{-1} A[2, i]``.
    """
    geno_index = np.asarray(geno_index, dtype=np.int64)
    n2 = geno_index.size
    if Gstar.shape != (n2, n2):
        raise ValueError("G* does not match the genotyped index")
    n = ped.n
    if 8 * n * n2 > max_dense_bytes:
        raise ABPError("RESOURCE_MEMORY",
                       f"single-step diag(H) needs an {n} x {n2} dense block "
                       f"(~{8 * n * n2 / 2**30:.2f} GiB)", n=n, n_genotyped=n2)
    Acols = ped.a_columns(geno_index)                      # A[:, 2]
    if a22 is None:
        a22 = 0.5 * (Acols[geno_index] + Acols[geno_index].T)
    a22_inv, logdet_a22 = spd_inverse_and_logdet(a22, "A22")
    g_inv, logdet_g = spd_inverse_and_logdet(Gstar, "G*")
    block = g_inv - a22_inv
    rr, cc = np.meshgrid(geno_index, geno_index, indexing="ij")
    embed = sp.csr_matrix((block.ravel(), (rr.ravel(), cc.ravel())), shape=(n, n))
    h_inv = (ped.ainv() + embed).tocsr()
    h_inv.sum_duplicates()
    h_inv.sort_indices()
    h_diag = 1.0 + ped.inbreeding()
    B = Acols @ a22_inv                                     # rows: all animals
    h_diag = h_diag + np.einsum("ij,ij->i", B @ (Gstar - a22), B)
    h_diag[geno_index] = np.diag(Gstar)                     # exact for genotyped
    logdet_h = ped.logdet_a() + logdet_g - logdet_a22
    return SingleStep(h_inv, h_diag, logdet_h, geno_index, a22, a22_inv)
