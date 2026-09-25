"""Selection indices (module M11): Smith-Hazel, restricted, and EBV aggregates.

Smith-Hazel index (method registry id ``dec.smith_hazel``)
---------------------------------------------------------
Breeding objective ``H = a'g`` (``a``: economic weights per unit of each
objective trait; ``g``: true breeding values).  Information ``x`` (centred)
with ``P = Var(x)`` and ``C = Cov(x, g')`` (``n_x x n_g``).  Then

    q = C a,   P b = q,   I = b'x,
    Var(I) = b'P b = q'P^{-1}q,   Var(H) = a' G_H a,
    r_IH^2 = Var(I) / Var(H)       (reliability; accuracy is its square root).

**Accuracy is the square root of the ratio of explained to objective
variance - never a product of variances** (spec erratum for a textbook
shortcut).  Expected response to truncation selection with intensity ``i``:
``R_H = i sigma_I`` and per objective trait ``R_j = i (C'b)_j / sigma_I``,
valid under the usual assumptions (multivariate normality, one round of
selection, no Bulmer effect, infinite population).

Restricted index (Kempthorne & Nordskog 1959): zero expected change in the
restricted traits ``g_r`` requires ``C_r' b = 0`` with ``C_r = Cov(x, g_r')``:

    b_r = b - P^{-1} C_r (C_r' P^{-1} C_r)^{-1} C_r' b.

Infeasible restrictions (singular ``C_r' P^{-1} C_r``) are reported, never
silently relaxed.  ``P`` must be positive definite: duplicated or collinear
information sources are reported with the offending eigen-direction.

EBV aggregate: when ``u_hat`` comes from a *multi-trait* BLUP that includes
all objective traits, ``I = a'u_hat`` is the BLUP of ``H`` and its
reliability is ``1 - a' PEV_i a / (a' G0 a K_ii)``.  Combining EBVs from
separate single-trait analyses is not optimal and is not offered here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import ndtri

from ..errors import ABPError


def selection_intensity(proportion_selected: float) -> float:
    """Truncation-selection intensity ``i = phi(z) / p`` (infinite population)."""
    p = float(proportion_selected)
    if not 0.0 < p < 1.0:
        raise ABPError("SPEC_INVALID", "proportion selected must lie in (0, 1)", value=p)
    z = float(ndtri(1.0 - p))                       # upper-tail standard normal quantile
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi) / p


def _check_pd(P: np.ndarray, names: list[str]) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    if P.shape[0] != P.shape[1] or not np.allclose(P, P.T, atol=1e-12 * max(1, np.abs(P).max())):
        raise ABPError("COVARIANCE_NOT_PD", "P = Var(x) must be square and symmetric")
    w, v = np.linalg.eigh(P)
    if w.min() <= 1e-12 * max(1.0, w.max()):
        k = int(np.argmin(w))
        combo = {names[i]: round(float(v[i, k]), 4) for i in range(len(names))
                 if abs(v[i, k]) > 1e-6}
        raise ABPError("MODEL_NOT_IDENTIFIABLE",
                       "information sources are collinear (duplicated information): the "
                       f"combination {combo} has (near) zero variance",
                       eigenvalue=float(w.min()), combination=combo)
    return 0.5 * (P + P.T)


@dataclass
class IndexResult:
    b: np.ndarray
    var_index: float
    var_objective: float
    reliability: float
    accuracy: float
    response_objective: float | None
    response_traits: np.ndarray | None
    intensity: float | None
    restricted: list[str]
    restriction_residual: float | None

    def to_dict(self, info_names, trait_names) -> dict:
        return {"coefficients": dict(zip(info_names, self.b.tolist())),
                "var_index": self.var_index, "var_objective": self.var_objective,
                "reliability": self.reliability, "accuracy": self.accuracy,
                "selection_intensity": self.intensity,
                "response_objective": self.response_objective,
                "response_per_trait": None if self.response_traits is None else
                dict(zip(trait_names, self.response_traits.tolist())),
                "restricted_traits": self.restricted,
                "max_abs_restriction_residual": self.restriction_residual,
                "assumptions": "multivariate normal; one round of truncation selection; "
                               "infinite population; no Bulmer effect"}


def smith_hazel(P: np.ndarray, C: np.ndarray, a: np.ndarray, G_H: np.ndarray,
                info_names: list[str], trait_names: list[str],
                restrict: list[str] | None = None,
                proportion_selected: float | None = None) -> IndexResult:
    """Optimal (optionally restricted) linear index; see module docstring."""
    P = _check_pd(P, info_names)
    C = np.asarray(C, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    G_H = np.asarray(G_H, dtype=np.float64)
    nx, ng = C.shape
    if P.shape != (nx, nx) or a.shape != (ng,) or G_H.shape != (ng, ng):
        raise ABPError("SPEC_INVALID", f"dimension mismatch: P {P.shape}, C {C.shape}, "
                       f"a {a.shape}, G_H {G_H.shape}")
    var_h = float(a @ G_H @ a)
    if not var_h > 0:
        raise ABPError("COVARIANCE_NOT_PD", "Var(H) = a'G_H a must be > 0")
    q = C @ a
    b = np.linalg.solve(P, q)
    restrict = list(restrict or [])
    resid = None
    if restrict:
        unknown = [r for r in restrict if r not in trait_names]
        if unknown:
            raise ABPError("SPEC_INVALID", f"restricted traits not in the objective: {unknown}")
        Cr = C[:, [trait_names.index(r) for r in restrict]]
        PiCr = np.linalg.solve(P, Cr)
        M = Cr.T @ PiCr
        if np.linalg.matrix_rank(M) < M.shape[0]:
            raise ABPError("MODEL_NOT_IDENTIFIABLE",
                           "restrictions are infeasible or redundant (Cr' P^-1 Cr is singular)",
                           restricted=restrict)
        b = b - PiCr @ np.linalg.solve(M, Cr.T @ b)
        resid = float(np.max(np.abs(Cr.T @ b)))
    var_i = float(b @ P @ b)
    if restrict:
        # accuracy of the restricted index with respect to H
        cov_ih = float(b @ q)
        rel = cov_ih**2 / (var_i * var_h) if var_i > 0 else 0.0
    else:
        rel = var_i / var_h
    if rel > 1 + 1e-10:
        raise ABPError("RELIABILITY_OUT_OF_RANGE",
                       f"index reliability {rel:.6g} > 1: P, C and G_H are inconsistent")
    intensity = resp = resp_t = None
    if proportion_selected is not None and var_i > 0:
        intensity = selection_intensity(proportion_selected)
        sd_i = math.sqrt(var_i)
        resp_t = intensity * (C.T @ b) / sd_i
        resp = float(a @ resp_t)
    return IndexResult(b, var_i, var_h, float(min(rel, 1.0)), float(math.sqrt(min(rel, 1.0))),
                       resp, resp_t, intensity, restrict, resid)


def ebv_index(ebv: np.ndarray, pev_blocks: np.ndarray | None, weights: np.ndarray,
              G0: np.ndarray, k_diag: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """``I_i = a'u_hat_i`` and its reliability ``1 - a'PEV_i a / (a'G0 a K_ii)``."""
    a = np.asarray(weights, dtype=np.float64)
    idx = ebv @ a
    if pev_blocks is None:
        return idx, None
    pev_i = np.einsum("j,ijk,k->i", a, pev_blocks, a)
    var_h = float(a @ G0 @ a) * np.asarray(k_diag, dtype=np.float64)
    rel = 1.0 - pev_i / var_h
    if np.any(rel < -1e-8) or np.any(rel > 1 + 1e-8):
        raise ABPError("RELIABILITY_OUT_OF_RANGE", "index reliability outside [0, 1]")
    return idx, np.clip(rel, 0.0, 1.0)
