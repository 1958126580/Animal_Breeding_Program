"""Multi-trait BLUP with known (co)variance matrices (module M09, part 1).

Model contract (method registry id ``blup.multi_trait``)
--------------------------------------------------------
For ``t`` traits, each record ``r`` (one animal, one occasion) carries the
observed subset ``o_r`` of the traits.  With ``U`` the ``q x t`` matrix of
additive effects and ``K`` the relationship structure,

    Var(vec_animal_major(U)) = K (x) G0,     Var(e_r) = R0[o_r, o_r],

records independent of each other.  **Stacking order** of the random
equations is animal-major: equation ``offset + i * t + j`` holds animal
``i``, trait ``j`` (so the precision block is ``K^{-1} (x) G0^{-1}``).  The
trait-major convention ``G0 (x) K`` describes the same model with permuted
equations; tests compare against an independently built trait-major
reference.  Observations are stacked record-major.

Each trait has its own fixed design (full column rank per trait, see
:mod:`abp.core.design`), so ``X`` is block-diagonal by trait.  Missing traits
are handled through the observed-data likelihood (sub-matrices of ``R0``);
nothing is imputed.  Residual covariances must only link traits that can
covary within a record; a structural zero (e.g. a lamb trait and a later
ewe trait) must be entered as 0.

Outputs: EBV per animal and trait, the ``t x t`` PEV block of every animal
(for index reliabilities), and reliabilities ``1 - PEV_jj / (G0_jj K_ii)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

from ..errors import ABPError
from .mme import DenseCholesky, RandomEffect, SolveResult, SparseLU, assemble, solve_system

STACKING_ORDER = "animal-major: equation = offset + animal_index * n_traits + trait_index"


def check_covariance(S: np.ndarray, what: str) -> np.ndarray:
    S = np.asarray(S, dtype=np.float64)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ABPError("COVARIANCE_NOT_PD", f"{what} must be a square matrix")
    if not np.allclose(S, S.T, atol=1e-12 * max(1.0, np.abs(S).max())):
        raise ABPError("COVARIANCE_NOT_PD", f"{what} must be symmetric")
    try:
        sla.cholesky(S, lower=True)
    except np.linalg.LinAlgError:
        ev = np.linalg.eigvalsh(S)
        raise ABPError("COVARIANCE_NOT_PD", f"{what} is not positive definite "
                       f"(smallest eigenvalue {ev.min():.3e})", eigenvalues=ev.tolist()) from None
    return 0.5 * (S + S.T)


@dataclass
class MTData:
    """Observed data in record-major stacking."""

    Y: np.ndarray                      # n_rec x t, NaN = missing
    X_per_trait: list[sp.csr_matrix]   # X_j rows = records where trait j is observed
    animal_col: np.ndarray             # n_rec, column index of the animal in K


@dataclass
class MTResult:
    ebv: np.ndarray                    # q x t
    pev_blocks: np.ndarray | None      # q x t x t
    reliability: np.ndarray | None     # q x t
    fixed: list[np.ndarray]            # per trait
    solve: SolveResult
    n_obs: int


def build_and_solve(data: MTData, k_inv, k_diag: np.ndarray, G0: np.ndarray, R0: np.ndarray,
                    method: str = "auto", compute_pev: bool = True, tol: float = 1e-10,
                    max_iter: int = 10000, memory_budget_bytes: int = 4 * 2**30) -> MTResult:
    Y = np.asarray(data.Y, dtype=np.float64)
    n_rec, t = Y.shape
    q = k_inv.shape[0]
    G0 = check_covariance(G0, "genetic covariance matrix G0")
    R0 = check_covariance(R0, "residual covariance matrix R0")
    if G0.shape != (t, t) or R0.shape != (t, t):
        raise ABPError("SPEC_INVALID", f"covariance matrices must be {t}x{t}")
    obs = ~np.isnan(Y)
    rec_idx, trait_idx = np.nonzero(obs)          # row-major -> record-major stacking
    n_obs = rec_idx.size
    if n_obs == 0:
        raise ABPError("NO_USABLE_RECORDS", "no observed trait values")
    y = Y[rec_idx, trait_idx]
    # fixed effects: block-diagonal by trait
    p_off = np.cumsum([0] + [X.shape[1] for X in data.X_per_trait])
    rows, cols, vals = [], [], []
    for j, Xj in enumerate(data.X_per_trait):
        recs_j = np.flatnonzero(obs[:, j])
        if Xj.shape[0] != recs_j.size:
            raise ValueError(f"X for trait {j} has {Xj.shape[0]} rows, expected {recs_j.size}")
        obs_row_of_rec = np.full(n_rec, -1)
        mask = trait_idx == j
        obs_row_of_rec[rec_idx[mask]] = np.flatnonzero(mask)
        Xc = Xj.tocoo()
        rows.append(obs_row_of_rec[recs_j[Xc.row]])
        cols.append(Xc.col + p_off[j])
        vals.append(Xc.data)
    X = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n_obs, int(p_off[-1])))
    Z = sp.csr_matrix((np.ones(n_obs), (np.arange(n_obs), data.animal_col[rec_idx] * t + trait_idx)),
                      shape=(n_obs, q * t))
    # residual precision: one block per record, inverse of R0 restricted to observed traits
    starts = np.searchsorted(rec_idx, np.arange(n_rec))
    ends = np.searchsorted(rec_idx, np.arange(n_rec), side="right")
    cache: dict[tuple, np.ndarray] = {}
    rr, rc, rv = [], [], []
    for r in range(n_rec):
        a, b = starts[r], ends[r]
        if a == b:
            continue
        key = tuple(trait_idx[a:b])
        if key not in cache:
            cache[key] = np.linalg.inv(R0[np.ix_(key, key)])
        blk = cache[key]
        ii, jj = np.meshgrid(np.arange(a, b), np.arange(a, b), indexing="ij")
        rr.append(ii.ravel())
        rc.append(jj.ravel())
        rv.append(blk.ravel())
    rinv = sp.csr_matrix((np.concatenate(rv), (np.concatenate(rr), np.concatenate(rc))),
                         shape=(n_obs, n_obs))
    g0_inv = np.linalg.inv(G0)
    kin = sp.csr_matrix(k_inv) if not sp.issparse(k_inv) else k_inv.tocsr()
    precision = sp.kron(kin, sp.csr_matrix(g0_inv), format="csr")
    system = assemble(y, X, [RandomEffect("animal", Z, precision, list(range(q * t)))], rinv)
    res = solve_system(system, method=method, need_inverse=compute_pev, tol=tol,
                       max_iter=max_iter, memory_budget_bytes=memory_budget_bytes)
    a0, _ = system.offsets["animal"]
    ebv = res.solution[a0:a0 + q * t].reshape(q, t)
    pev = rel = None
    if compute_pev:
        pev = _pev_blocks(res, a0, q, t)
        prior = np.asarray(k_diag, dtype=np.float64)[:, None] * np.diag(G0)[None, :]
        rel_raw = 1.0 - np.einsum("ijj->ij", pev) / prior
        if np.any(rel_raw < -1e-8) or np.any(rel_raw > 1 + 1e-8):
            i, j = np.unravel_index(np.argmax(np.abs(rel_raw - np.clip(rel_raw, 0, 1))), rel_raw.shape)
            raise ABPError("RELIABILITY_OUT_OF_RANGE",
                           f"multi-trait reliability {rel_raw[i, j]:.6g} outside [0, 1]",
                           animal_index=int(i), trait_index=int(j))
        rel = np.clip(rel_raw, 0.0, 1.0)
    fixed = [res.solution[p_off[j]:p_off[j + 1]].copy() for j in range(t)]
    return MTResult(ebv, pev, rel, fixed, res, n_obs)


def _pev_blocks(res: SolveResult, a0: int, q: int, t: int) -> np.ndarray:
    out = np.empty((q, t, t))
    if isinstance(res.factor, DenseCholesky):
        return res.factor.inverse_diagonal_blocks(a0, q, t)
    if isinstance(res.factor, SparseLU):
        n = res.factor.n
        batch = max(t, (512 // t) * t)
        idx = np.arange(a0, a0 + q * t)
        for start in range(0, idx.size, batch):
            cols = idx[start:start + batch]
            E = np.zeros((n, cols.size))
            E[cols, np.arange(cols.size)] = 1.0
            S = res.factor.solve(E)
            for k in range(0, cols.size, t):
                i = (cols[k] - a0) // t
                s = cols[k]
                out[i] = 0.5 * (S[s:s + t, k:k + t] + S[s:s + t, k:k + t].T)
        return out
    raise ABPError("UNSUPPORTED_COMBINATION", "PEV requires a direct solver")
