"""Mixed-model equations (MME): assembly and linear solvers.

Model contract (method registry id ``lmm.mme``)
-----------------------------------------------
``y = X b + sum_k Z_k u_k + e`` with ``Var(u_k) = Sigma_k`` (positive
definite), ``Var(e) = R`` (positive definite), ``u_k`` and ``e`` mutually
uncorrelated, and ``X`` of full column rank (see :mod:`abp.core.design`).
With ``W = [X, Z_1, ..., Z_K]`` the MME are

    C s = r,   C = W' R^{-1} W + blockdiag(0, Sigma_1^{-1}, ..., Sigma_K^{-1}),
               r = W' R^{-1} y.

``C`` is **not** scaled by the residual variance, so the random-effect block
of ``C^{-1}`` is directly the prediction-error covariance
``Var(u - u_hat)`` *including* the uncertainty of the GLS estimate of ``b``
(Henderson 1975), conditional on the (co)variance parameters.

Solvers
-------
``dense``          LAPACK Cholesky; exact log-determinant and inverse (PEV).
``sparse_direct``  SuperLU with symmetric minimum-degree ordering and no
                   pivoting (valid because ``C`` is symmetric positive
                   definite); PEV by solving for selected unit vectors.
``pcg``            Jacobi-preconditioned conjugate gradients; solutions only.
Every solution is checked against the *original* system: the relative
residual ``||C s - r|| / ||r||`` is always recomputed and reported.
Explicit inverses are formed only when PEV or REML traces are requested
on the dense path; the solve itself never uses an inverse.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from ..errors import ABPError

SOLVER_METHODS = ("auto", "dense", "sparse_direct", "pcg")


@dataclass
class RandomEffect:
    """One random-effect block of the MME.

    ``precision`` is ``Sigma_k^{-1}`` (sparse or dense, ``q x q``), i.e. it
    already includes the variance scaling (e.g. ``A^{-1} / sigma_a^2``).
    """

    name: str
    Z: sp.csr_matrix
    precision: sp.spmatrix | np.ndarray
    labels: Sequence

    @property
    def q(self) -> int:
        return self.Z.shape[1]


@dataclass
class MixedModelSystem:
    """Assembled MME with the bookkeeping needed to interpret solutions."""

    C: sp.csr_matrix
    rhs: np.ndarray
    W: sp.csr_matrix
    y: np.ndarray
    rinv: sp.spmatrix
    p: int
    offsets: dict[str, tuple[int, int]]  # name -> (start, stop) in the solution vector

    @property
    def n_equations(self) -> int:
        return self.C.shape[0]


def assemble(y: np.ndarray, X: sp.csr_matrix, randoms: Sequence[RandomEffect],
             rinv: sp.spmatrix) -> MixedModelSystem:
    """Assemble ``C`` and ``r``; validates all dimensions explicitly."""
    y = np.asarray(y, dtype=np.float64)
    n = y.shape[0]
    if X.shape[0] != n or rinv.shape != (n, n):
        raise ValueError(f"dimension mismatch: y {n}, X {X.shape}, R^-1 {rinv.shape}")
    blocks = [X]
    offsets: dict[str, tuple[int, int]] = {}
    pos = X.shape[1]
    offsets["fixed"] = (0, pos)
    for re in randoms:
        if re.Z.shape[0] != n:
            raise ValueError(f"random effect {re.name!r}: Z has {re.Z.shape[0]} rows, y has {n}")
        if re.precision.shape != (re.q, re.q):
            raise ValueError(f"random effect {re.name!r}: precision {re.precision.shape} "
                             f"does not match q={re.q}")
        if re.name in offsets:
            raise ValueError(f"duplicate random effect name {re.name!r}")
        blocks.append(re.Z)
        offsets[re.name] = (pos, pos + re.q)
        pos += re.q
    W = sp.hstack(blocks, format="csr")
    RW = (rinv @ W).tocsr()
    C = (W.T @ RW).tocsr()
    for re in randoms:
        a, b = offsets[re.name]
        prec = re.precision
        prec_sp = sp.csr_matrix(prec) if not sp.issparse(prec) else prec.tocoo()
        prec_sp = prec_sp.tocoo()
        C = C + sp.csr_matrix((prec_sp.data, (prec_sp.row + a, prec_sp.col + a)),
                              shape=C.shape)
    C = C.tocsr()
    C.sum_duplicates()
    C.sort_indices()
    rhs = np.asarray(RW.T @ y).ravel()
    return MixedModelSystem(C=C, rhs=rhs, W=W, y=y, rinv=rinv, p=X.shape[1], offsets=offsets)


# --------------------------------------------------------------------------
# Factorizations
# --------------------------------------------------------------------------
class DenseCholesky:
    """LAPACK Cholesky of a dense SPD matrix, with cached inverse."""

    kind = "dense"

    def __init__(self, C: np.ndarray):
        try:
            self.L = sla.cholesky(C, lower=True, overwrite_a=False, check_finite=True)
        except np.linalg.LinAlgError as exc:
            raise ABPError("FACTORIZATION_FAILED",
                           "coefficient matrix is not positive definite (dense Cholesky failed); "
                           "check variance components and fixed-effect dependencies",
                           lapack=str(exc)) from None
        self.n = C.shape[0]
        self._inv: np.ndarray | None = None

    def solve(self, b: np.ndarray) -> np.ndarray:
        return sla.cho_solve((self.L, True), b, check_finite=False)

    def logdet(self) -> float:
        return float(2.0 * np.sum(np.log(np.diag(self.L))))

    def inverse(self) -> np.ndarray:
        """Full ``C^{-1}`` via LAPACK ``potri`` (cached; needed for REML traces)."""
        if self._inv is None:
            inv, info = sla.lapack.dpotri(self.L, lower=1)
            if info != 0:
                raise ABPError("FACTORIZATION_FAILED", f"dpotri failed (info={info})")
            n, bs = inv.shape[0], 1024
            for j0 in range(0, n, bs):  # mirror the lower triangle in place, by blocks
                j1 = min(n, j0 + bs)
                blk = inv[j0:j1, j0:j1]
                blk[np.triu_indices(j1 - j0, 1)] = blk.T[np.triu_indices(j1 - j0, 1)]
                if j1 < n:
                    inv[j0:j1, j1:] = inv[j1:, j0:j1].T
            self._inv = inv
        return self._inv

    def _linv(self) -> np.ndarray:
        """``L^{-1}`` (lower triangular) via LAPACK ``trtri``."""
        linv, info = sla.lapack.dtrtri(self.L, lower=1)
        if info != 0:
            raise ABPError("FACTORIZATION_FAILED", f"dtrtri failed (info={info})")
        return linv

    def inverse_diagonal(self, idx: np.ndarray) -> np.ndarray:
        """``diag(C^{-1})[idx]`` = column sums of squares of ``L^{-1}`` (no full inverse)."""
        if self._inv is not None:
            return np.diag(self._inv)[np.asarray(idx)].copy()
        linv = self._linv()
        return np.einsum("ij,ij->j", linv[:, idx], linv[:, idx])

    def inverse_diagonal_blocks(self, start: int, q: int, t: int) -> np.ndarray:
        """``t x t`` diagonal blocks of ``C^{-1}`` for equations ``start + i*t + (0..t-1)``."""
        out = np.empty((q, t, t))
        if self._inv is not None:
            for i in range(q):
                s = start + i * t
                out[i] = self._inv[s:s + t, s:s + t]
            return out
        linv = self._linv()
        for i in range(q):
            s = start + i * t
            B = linv[s:, s:s + t]       # L^{-1} is lower triangular: rows < s are zero
            out[i] = B.T @ B
        return out

    def inverse_block(self, idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx)
        return self.inverse()[np.ix_(idx, idx)]


class SparseLU:
    """SuperLU factorization of a sparse SPD matrix (symmetric ordering, no pivoting)."""

    kind = "sparse_direct"

    def __init__(self, C: sp.csr_matrix):
        try:
            self.lu = splu(C.tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                           options={"SymmetricMode": True})
        except RuntimeError as exc:
            raise ABPError("FACTORIZATION_FAILED",
                           "sparse factorization failed (matrix singular?)", superlu=str(exc)) from None
        self.n = C.shape[0]
        diagU = self.lu.U.diagonal()
        if np.any(diagU <= 0):
            raise ABPError("FACTORIZATION_FAILED",
                           "coefficient matrix is not positive definite (non-positive pivot)",
                           min_pivot=float(diagU.min()))
        self._diagU = diagU

    def solve(self, b: np.ndarray) -> np.ndarray:
        return self.lu.solve(np.asarray(b, dtype=np.float64))

    def logdet(self) -> float:
        # With a symmetric permutation P C P' = L U, det(C) = prod(diag U) > 0.
        if not np.array_equal(self.lu.perm_r, self.lu.perm_c):
            raise ABPError("FACTORIZATION_FAILED",
                           "unexpected row pivoting in SuperLU; log-determinant unavailable")
        return float(np.sum(np.log(self._diagU)))

    def inverse_block(self, idx: np.ndarray, batch: int = 512) -> np.ndarray:
        """``C^{-1}[idx, idx]`` by solving for unit vectors in batches."""
        idx = np.asarray(idx, dtype=np.int64)
        out = np.empty((idx.size, idx.size))
        for start in range(0, idx.size, batch):
            cols = idx[start:start + batch]
            E = np.zeros((self.n, cols.size))
            E[cols, np.arange(cols.size)] = 1.0
            out[:, start:start + cols.size] = self.solve(E)[idx, :]
        return 0.5 * (out + out.T)


# --------------------------------------------------------------------------
# Preconditioned conjugate gradients
# --------------------------------------------------------------------------
@dataclass
class PCGInfo:
    iterations: int
    converged: bool
    rel_residual: float
    history: list[tuple[int, float]] = field(default_factory=list)


def pcg(matvec: Callable[[np.ndarray], np.ndarray], b: np.ndarray, diag: np.ndarray,
        tol: float = 1e-10, max_iter: int = 10000, x0: np.ndarray | None = None,
        record_every: int = 10) -> tuple[np.ndarray, PCGInfo]:
    """Jacobi-preconditioned CG for SPD systems.

    Converged means the **true** relative residual ``||b - A x|| / ||b||``
    is ``<= tol`` (the recurrence residual is re-verified, and CG restarts
    from the true residual when the two disagree).
    """
    if np.any(diag <= 0):
        raise ABPError("FACTORIZATION_FAILED", "non-positive diagonal in PCG preconditioner")
    minv = 1.0 / diag
    bnorm = float(np.linalg.norm(b))
    x = np.zeros_like(b) if x0 is None else np.array(x0, dtype=np.float64)
    if bnorm == 0.0:
        return np.zeros_like(b), PCGInfo(0, True, 0.0, [])
    history: list[tuple[int, float]] = []
    it = 0
    while True:
        r = b - matvec(x)
        rel = float(np.linalg.norm(r)) / bnorm
        if rel <= tol:
            return x, PCGInfo(it, True, rel, history)
        if it >= max_iter:
            return x, PCGInfo(it, False, rel, history)
        z = minv * r
        d = z.copy()
        rz = float(r @ z)
        while it < max_iter:
            q = matvec(d)
            dq = float(d @ q)
            if dq <= 0:
                raise ABPError("FACTORIZATION_FAILED",
                               "PCG found a non-positive curvature direction; system is not SPD")
            alpha = rz / dq
            x += alpha * d
            r -= alpha * q
            it += 1
            rel_rec = float(np.linalg.norm(r)) / bnorm
            if it % record_every == 0:
                history.append((it, rel_rec))
            if rel_rec <= tol:
                break  # verify with the true residual in the outer loop
            z = minv * r
            rz_new = float(r @ z)
            d = z + (rz_new / rz) * d
            rz = rz_new


# --------------------------------------------------------------------------
# Solver front-end
# --------------------------------------------------------------------------
@dataclass
class SolveResult:
    solution: np.ndarray
    method: str
    selection_reason: str
    rel_residual: float
    iterations: int | None
    wall_seconds: float
    factor: DenseCholesky | SparseLU | None
    pcg_history: list[tuple[int, float]] = field(default_factory=list)


def dense_bytes(n_eq: int, with_inverse: bool) -> int:
    """Approximate peak bytes of the dense path (matrix + factor [+ inverse])."""
    return 8 * n_eq * n_eq * (3 if with_inverse else 2)


EXACT_PEV_LIMIT = 30000  #: max equations for exact PEV by selected solves (sparse path)


def choose_method(n_eq: int, need_inverse: bool, memory_budget_bytes: int,
                  requested: str = "auto") -> tuple[str, str]:
    """Pick a solver and explain why (the reason is written to the manifest).

    Rules (measured in docs/benchmarks.md): small systems -> dense LAPACK
    (exact PEV, fastest below ~12k equations); large systems without PEV ->
    Jacobi-PCG (with automatic fallback to sparse direct if it does not
    converge); large systems with exact PEV -> sparse direct, refused above
    ``EXACT_PEV_LIMIT`` equations.
    """
    if requested not in SOLVER_METHODS:
        raise ABPError("SPEC_INVALID", f"solver.method must be one of {SOLVER_METHODS}",
                       value=requested)
    need = dense_bytes(n_eq, need_inverse)
    if requested != "auto":
        if requested == "dense" and need > memory_budget_bytes:
            raise ABPError("RESOURCE_MEMORY",
                           f"dense solver needs ~{need / 2**30:.2f} GiB, budget is "
                           f"{memory_budget_bytes / 2**30:.2f} GiB", n_equations=n_eq)
        if requested == "sparse_direct" and need_inverse and n_eq > EXACT_PEV_LIMIT:
            raise _pev_too_large(n_eq)
        return requested, "requested explicitly in spec"
    if n_eq <= 12000 and need <= memory_budget_bytes:
        return "dense", (f"auto: {n_eq} equations <= 12000 and dense memory "
                         f"{need / 2**30:.2f} GiB within budget")
    if need_inverse:
        if n_eq > EXACT_PEV_LIMIT:
            raise _pev_too_large(n_eq)
        return "sparse_direct", (f"auto: {n_eq} equations too large for dense path; "
                                 "PEV requested -> sparse direct with selected solves")
    return "pcg", (f"auto: {n_eq} equations, no PEV -> Jacobi-PCG "
                   "(fallback: sparse direct if not converged)")


def _pev_too_large(n_eq: int) -> ABPError:
    return ABPError("UNSUPPORTED_COMBINATION",
                    f"exact PEV for {n_eq} equations exceeds the 0.1 limit of {EXACT_PEV_LIMIT}; "
                    "set solver.pev = \"none\" (EBVs without reliabilities) - approximate "
                    "reliabilities are on the roadmap", n_equations=n_eq, limit=EXACT_PEV_LIMIT)


def solve_system(system: MixedModelSystem, method: str = "auto", need_inverse: bool = False,
                 tol: float = 1e-10, max_iter: int = 10000,
                 memory_budget_bytes: int = 4 * 2**30,
                 residual_limit: float = 1e-8) -> SolveResult:
    """Solve the MME and verify the solution against the original system."""
    t0 = time.perf_counter()
    n_eq = system.n_equations
    chosen, reason = choose_method(n_eq, need_inverse, memory_budget_bytes, method)
    factor = None
    iterations = None
    history: list[tuple[int, float]] = []
    if chosen == "dense":
        factor = DenseCholesky(system.C.toarray())
        s = factor.solve(system.rhs)
    elif chosen == "sparse_direct":
        factor = SparseLU(system.C)
        s = factor.solve(system.rhs)
    else:
        if need_inverse:
            raise ABPError("UNSUPPORTED_COMBINATION",
                           "PCG does not provide PEV; use solver.pev = \"none\" or a direct solver")
        C = system.C
        s, info = pcg(lambda v: C @ v, system.rhs, C.diagonal(), tol=tol, max_iter=max_iter)
        iterations, history = info.iterations, info.history
        if not info.converged:
            if method == "auto":  # documented fallback to the verified direct solver
                reason += (f"; PCG did not converge in {info.iterations} iterations "
                           f"(rel. residual {info.rel_residual:.2e}) -> fell back to sparse direct")
                chosen = "sparse_direct"
                factor = SparseLU(system.C)
                s = factor.solve(system.rhs)
            else:
                raise ABPError("SOLVER_NOT_CONVERGED",
                               f"PCG stopped after {info.iterations} iterations with relative "
                               f"residual {info.rel_residual:.3e} > tol {tol:.1e}",
                               iterations=info.iterations, rel_residual=info.rel_residual, tol=tol)
    rnorm = float(np.linalg.norm(system.C @ s - system.rhs))
    bnorm = float(np.linalg.norm(system.rhs))
    rel = rnorm / bnorm if bnorm > 0 else rnorm
    limit = max(residual_limit, tol) if chosen == "pcg" else residual_limit
    if not np.all(np.isfinite(s)) or rel > limit:
        raise ABPError("BACKWARD_ERROR_TOO_LARGE",
                       f"relative residual {rel:.3e} of the {chosen} solution exceeds {limit:.1e}",
                       rel_residual=rel, limit=limit, method=chosen)
    return SolveResult(s, chosen, reason, rel, iterations, time.perf_counter() - t0, factor, history)


def prediction_error_covariance(result: SolveResult, idx: np.ndarray) -> np.ndarray:
    """``C^{-1}[idx, idx]`` = ``Var(u - u_hat)`` for the random equations ``idx``."""
    if result.factor is None:
        raise ABPError("UNSUPPORTED_COMBINATION", "PEV requires a direct solver")
    return result.factor.inverse_block(idx)


def pev_diagonal(result: SolveResult, idx: np.ndarray) -> np.ndarray:
    """Diagonal PEV for equations ``idx`` (dense: from the cached inverse)."""
    if isinstance(result.factor, DenseCholesky):
        return result.factor.inverse_diagonal(np.asarray(idx))
    if isinstance(result.factor, SparseLU):
        idx = np.asarray(idx, dtype=np.int64)
        out = np.empty(idx.size)
        for start in range(0, idx.size, 512):
            cols = idx[start:start + 512]
            E = np.zeros((result.factor.n, cols.size))
            E[cols, np.arange(cols.size)] = 1.0
            out[start:start + cols.size] = result.factor.solve(E)[cols, np.arange(cols.size)]
        return out
    raise ABPError("UNSUPPORTED_COMBINATION", "PEV requires a direct solver")
