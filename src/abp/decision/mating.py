"""Mating allocation under hard constraints (module M12, part 2).

Model contract (method registry id ``dec.mating``)
--------------------------------------------------
Given integer mating counts ``n_s`` for sires and ``m_d`` for dams
(``sum n_s = sum m_d = N``, e.g. from :func:`abp.decision.ocs.integer_matings`),
choose ``x_sd in {0, 1}`` (a pair is mated at most once) with

    sum_d x_sd = n_s,     sum_s x_sd = m_d,
    x_sd = 0 for forbidden pairs,
    minimize sum x_sd * F_sd,     F_sd = A_sd / 2  (inbreeding of the progeny).

Because every parent's number of matings is fixed, the expected merit of the
progeny, ``sum x_sd (g_s + g_d)/2``, is the same for every feasible plan, so
the plan only minimises progeny inbreeding. Forbidden pairs are those with
``A_sd > max_pair_relationship``, those whose risk of an affected progeny for
a declared autosomal recessive (``p_s p_d / 4`` with carrier probabilities
``p``) exceeds ``max_affected_risk``, and any pair listed explicitly.
Carriers are never removed from selection by this module; only risky
pairings are avoided.

Solution: the constraint matrix is that of a bipartite transportation
problem (totally unimodular), so the LP relaxation has integral optimal
vertices; ABP solves it with the HiGHS dual simplex and **verifies** that the
returned plan is integral and satisfies every constraint.  If no feasible plan
exists, an elastic LP locates the parents that cannot be matched; the
constraints are never relaxed silently.  The plan is a proposal for the
breeder; ABP never triggers matings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from ..errors import ABPError


@dataclass
class MatingPlan:
    pairs: list[tuple[int, int]]          # (sire index, dam index) into the inputs
    offspring_inbreeding: np.ndarray      # F of each pair
    affected_risk: np.ndarray             # p_s p_d / 4 of each pair (0 if no carrier data)
    total_inbreeding: float
    mean_inbreeding: float
    checks: dict


def forbidden_mask(A_sd: np.ndarray, max_pair_relationship: float | None,
                   carrier_s: np.ndarray | None, carrier_d: np.ndarray | None,
                   max_affected_risk: float | None,
                   explicit: set[tuple[int, int]] | None = None) -> tuple[np.ndarray, dict]:
    """Boolean matrix of forbidden pairs and the count per reason."""
    forb = np.zeros(A_sd.shape, dtype=bool)
    reasons = {}
    if max_pair_relationship is not None:
        m = A_sd > max_pair_relationship
        reasons["relationship"] = int(m.sum())
        forb |= m
    if carrier_s is not None and carrier_d is not None and max_affected_risk is not None:
        risk = np.outer(carrier_s, carrier_d) / 4.0
        m = risk > max_affected_risk
        reasons["recessive_risk"] = int(m.sum())
        forb |= m
    if explicit:
        for s, d in explicit:
            forb[s, d] = True
        reasons["explicit"] = len(explicit)
    return forb, reasons


def allocate(n_s: np.ndarray, m_d: np.ndarray, A_sd: np.ndarray, forbidden: np.ndarray,
             carrier_s: np.ndarray | None = None, carrier_d: np.ndarray | None = None) -> MatingPlan:
    """Minimum-inbreeding mating plan (see module docstring)."""
    n_s = np.asarray(n_s, dtype=np.int64)
    m_d = np.asarray(m_d, dtype=np.int64)
    S, D = A_sd.shape
    if n_s.size != S or m_d.size != D:
        raise ValueError("counts do not match the relationship block")
    if n_s.sum() != m_d.sum():
        raise ABPError("MODEL_NOT_IDENTIFIABLE",
                       f"sire matings ({int(n_s.sum())}) and dam matings ({int(m_d.sum())}) differ")
    if np.any(m_d > S) or np.any(n_s > D):
        raise ABPError("MODEL_NOT_IDENTIFIABLE", "a parent needs more distinct mates than exist")
    si, di = np.nonzero(~forbidden & (n_s[:, None] > 0) & (m_d[None, :] > 0))
    F = A_sd / 2.0
    cost = F[si, di]
    nv = si.size
    rows_s = sp.csr_matrix((np.ones(nv), (si, np.arange(nv))), shape=(S, nv))
    rows_d = sp.csr_matrix((np.ones(nv), (di, np.arange(nv))), shape=(D, nv))
    A_eq = sp.vstack([rows_s, rows_d]).tocsr()
    b_eq = np.concatenate([n_s, m_d]).astype(float)
    res = linprog(cost, A_eq=A_eq, b_eq=b_eq, bounds=(0.0, 1.0), method="highs-ds") \
        if nv else None
    if res is None or res.status != 0:
        raise _infeasibility(n_s, m_d, si, di, S, D)
    x = res.x
    if np.max(np.abs(x - np.round(x))) > 1e-7:
        raise ABPError("SOLVER_NOT_CONVERGED", "mating LP returned a fractional vertex",
                       max_fraction=float(np.max(np.abs(x - np.round(x)))))
    chosen = np.flatnonzero(np.round(x) == 1)
    pairs = [(int(si[k]), int(di[k])) for k in chosen]
    f = np.array([F[s, d] for s, d in pairs])
    risk = np.array([carrier_s[s] * carrier_d[d] / 4.0 for s, d in pairs]) \
        if carrier_s is not None and carrier_d is not None else np.zeros(len(pairs))
    # independent verification of every constraint on the integral plan
    used_s = np.bincount([s for s, _ in pairs], minlength=S)
    used_d = np.bincount([d for _, d in pairs], minlength=D)
    checks = {
        "sire_counts_match": bool(np.array_equal(used_s, n_s)),
        "dam_counts_match": bool(np.array_equal(used_d, m_d)),
        "no_forbidden_pair": bool(not any(forbidden[s, d] for s, d in pairs)),
        "no_repeated_pair": len(set(pairs)) == len(pairs),
        "lp_objective": float(res.fun),
        "plan_objective": float(f.sum()),
    }
    if not all(v for k, v in checks.items() if isinstance(v, bool)) or \
            abs(checks["lp_objective"] - checks["plan_objective"]) > 1e-9:
        raise ABPError("INTERNAL", "mating plan failed its own constraint verification",
                       checks=checks)
    return MatingPlan(pairs, f, risk, float(f.sum()), float(f.mean()) if f.size else 0.0, checks)


def _infeasibility(n_s, m_d, si, di, S, D) -> ABPError:
    """Elastic LP: minimise unmet matings to locate parents that cannot be matched."""
    nv = si.size
    rows_s = sp.csr_matrix((np.ones(nv), (si, np.arange(nv))), shape=(S, nv))
    rows_d = sp.csr_matrix((np.ones(nv), (di, np.arange(nv))), shape=(D, nv))
    slack = sp.identity(S + D, format="csr")
    A_eq = sp.hstack([sp.vstack([rows_s, rows_d]), slack]).tocsr()
    c = np.concatenate([np.zeros(nv), np.ones(S + D)])
    res = linprog(c, A_eq=A_eq, b_eq=np.concatenate([n_s, m_d]).astype(float),
                  bounds=[(0, 1)] * nv + [(0, None)] * (S + D), method="highs-ds")
    unmet_s = unmet_d = []
    if res.status == 0:
        sl = res.x[nv:]
        unmet_s = [int(i) for i in np.flatnonzero(sl[:S] > 1e-7)]
        unmet_d = [int(j) for j in np.flatnonzero(sl[S:] > 1e-7)]
    return ABPError("MODEL_NOT_IDENTIFIABLE",
                    f"no mating plan satisfies all hard constraints: {len(unmet_s)} sire(s) and "
                    f"{len(unmet_d)} dam(s) cannot receive their matings; constraints were not "
                    "relaxed", unmet_sire_indices=unmet_s, unmet_dam_indices=unmet_d)
