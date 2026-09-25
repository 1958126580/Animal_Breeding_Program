"""Unknown-parent groups (UPG) by the QP transformation (module M03, extension).

Model contract (method registry id ``ped.upg``)
-----------------------------------------------
Unknown parents may be assigned to *genetic groups* (e.g. by birth period,
origin and sex of the missing parent).  With ``Q`` (animals x groups) the
expected fraction of each animal's genes coming from each group
(``Q_i = (Q_s + Q_d)/2``, a group parent contributing its unit vector and a
plain unknown parent contributing nothing), the breeding value is

    u*_i = u_i + Q_i g,     u ~ N(0, sigma_a^2 A),

where ``A`` treats every unknown parent (grouped or not) as an unrelated
base animal.  Following Quaas (1988, J Dairy Sci 71:1338) and Westell,
Quaas & Van Vleck (1988, J Dairy Sci 71:1310), the mixed-model equations
for ``(u*, g)`` use

    A*^{-1} = [[A^{-1},      -A^{-1} Q    ],
               [-Q' A^{-1},   Q' A^{-1} Q ]],

which is built directly by Henderson's rules with groups acting as parents
(``d_i`` counts only *animal* parents).  Groups are:

* **random** (default): add ``I / ratio`` to the group block, with
  ``ratio = sigma_g^2 / sigma_a^2`` declared by the user; then
  ``Var(u*_i) = sigma_a^2 (A_ii + ratio * Q_i Q_i')``;
* **fixed**: nothing is added.  The system for ``(b, u*, g)`` is a
  non-singular transformation of the one for ``(b, g, u)`` with fixed design
  ``[X, Z Q]``, so it is solvable exactly when ``[X, Z Q]`` has full column
  rank.  This fails e.g. with an intercept when every recorded lineage ends
  in a group (the rows of ``Z Q`` then sum to one) or when a group has no
  recorded descendants.  :func:`check_fixed_groups_estimable` tests the rank
  before solving and ABP stops (``MODEL_NOT_IDENTIFIABLE``) instead of
  choosing an arbitrary constraint.  With fixed groups ``Var(u*_i)`` is not
  defined, so no reliabilities are reported (``PEV`` of ``u*`` includes the
  estimation error of ``g``); REML is refused (``K`` is improper).

Animal solutions are ``u*`` (they include the group contributions); group
solutions are reported separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from ..errors import ABPError
from .model import GeneticStructure
from .pedigree import Pedigree

#: relative singular-value threshold of the fixed-group estimability test
ESTIMABILITY_RTOL = 1e-9


@dataclass
class GroupAssignment:
    labels: tuple[str, ...]          # group labels, in equation order
    sire_group: np.ndarray           # per animal (pedigree order): group index or -1
    dam_group: np.ndarray


def ainv_with_groups(ped: Pedigree, groups: GroupAssignment) -> sp.csr_matrix:
    """``A*^{-1}`` of size (n + n_groups) by Henderson's rules with groups as parents."""
    n, ng = ped.n, len(groups.labels)
    dinv = 1.0 / ped.mendelian_d()          # groups count as unknown parents for d_i
    s_col = np.where(ped.sire >= 0, ped.sire, np.where(groups.sire_group >= 0,
                                                       n + groups.sire_group, -1))
    d_col = np.where(ped.dam >= 0, ped.dam, np.where(groups.dam_group >= 0,
                                                     n + groups.dam_group, -1))
    members = [np.arange(n), s_col, d_col]
    weights = [1.0, -0.5, -0.5]
    rows, cols, vals = [], [], []
    for a in range(3):
        for b in range(3):
            mask = (members[a] >= 0) & (members[b] >= 0)
            rows.append(members[a][mask])
            cols.append(members[b][mask])
            vals.append(dinv[mask] * weights[a] * weights[b])
    M = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n + ng, n + ng))
    M.sum_duplicates()
    M.sort_indices()
    return M


def group_fractions(ped: Pedigree, groups: GroupAssignment) -> np.ndarray:
    """``Q`` (n x n_groups): expected gene fraction from each group (pedigree order)."""
    n, ng = ped.n, len(groups.labels)
    Q = np.zeros((n, ng))
    # parents are in earlier generations, so one vectorized pass per generation suffices
    for g in range(int(ped.generation.max()) + 1 if n else 0):
        idx = np.flatnonzero(ped.generation == g)
        for par, grp in ((ped.sire[idx], groups.sire_group[idx]),
                         (ped.dam[idx], groups.dam_group[idx])):
            known = par >= 0
            Q[idx[known]] += 0.5 * Q[par[known]]
            ing = ~known & (grp >= 0)
            np.add.at(Q, (idx[ing], grp[ing]), 0.5)
    return Q


def upg_structure_parts(ped: Pedigree, groups: GroupAssignment, effect: str,
                        ratio: float | None) -> tuple[sp.csr_matrix, np.ndarray, float | None]:
    """``K^{-1}`` (in units of sigma_a^2), ``diag(K)`` for reliabilities and ``log|K|``.

    Random groups: ``K = Var([u*; g]) / sigma_a^2`` with ``log|K| = log|A| + ng log(ratio)``.
    Fixed groups: ``K^{-1} = A*^{-1}`` (improper; ``log|K|`` undefined -> REML refused) and
    ``diag(K)`` is ``NaN`` (reliabilities undefined, see module notes).
    """
    n, ng = ped.n, len(groups.labels)
    Ks = ainv_with_groups(ped, groups)
    Q = group_fractions(ped, groups)
    base = 1.0 + ped.inbreeding()
    if effect == "random":
        if ratio is None or not ratio > 0:
            raise ABPError("SPEC_INVALID", "random unknown-parent groups need upg.variance_ratio > 0")
        add = sp.csr_matrix((np.full(ng, 1.0 / ratio), (np.arange(n, n + ng), np.arange(n, n + ng))),
                            shape=(n + ng, n + ng))
        k_inv = (Ks + add).tocsr()
        k_diag = np.concatenate([base + ratio * np.sum(Q * Q, axis=1), np.full(ng, ratio)])
        logdet = ped.logdet_a() + ng * float(np.log(ratio))
    elif effect == "fixed":
        k_inv = Ks
        k_diag = np.full(n + ng, np.nan)          # Var(u*) undefined: no reliabilities
        logdet = None
    else:
        raise ABPError("SPEC_INVALID", "upg.effect must be 'random' or 'fixed'")
    return k_inv, k_diag, logdet


def check_fixed_groups_estimable(X: sp.spmatrix | np.ndarray, ZQ: np.ndarray,
                                 labels: tuple[str, ...]) -> dict:
    """Refuse fixed groups that are not estimable jointly with the fixed effects.

    ``X`` is the full-rank fixed design (records x p) and ``ZQ`` the group
    design (records x n_groups).  The combined matrix must have full column
    rank; the test uses singular values of the column-equilibrated matrix
    (threshold ``ESTIMABILITY_RTOL`` times the largest).  Returns the
    diagnostics (rank, smallest relative singular value).
    """
    Xd = X.toarray() if sp.issparse(X) else np.asarray(X, dtype=np.float64)
    M = np.hstack([Xd, ZQ])
    norms = np.linalg.norm(M, axis=0)
    empty = [labels[k] for k in range(ZQ.shape[1]) if norms[Xd.shape[1] + k] == 0.0]
    if empty:
        raise ABPError("MODEL_NOT_IDENTIFIABLE",
                       f"fixed unknown-parent groups without recorded descendants cannot be "
                       f"estimated: {empty[:10]}; merge them or declare upg.effect = 'random'",
                       groups=empty)
    sv = np.linalg.svd(M / norms, compute_uv=False)
    rel_min = float(sv[-1] / sv[0])
    rank = int(np.sum(sv > ESTIMABILITY_RTOL * sv[0]))
    if rank < M.shape[1]:
        raise ABPError("MODEL_NOT_IDENTIFIABLE",
                       f"fixed unknown-parent group effects are confounded with the fixed "
                       f"effects (rank {rank} of {M.shape[1]}; e.g. an intercept while every "
                       "recorded lineage traces to groups).  Remove the intercept/confounded "
                       "term, merge groups, or declare upg.effect = 'random'",
                       rank=rank, columns=M.shape[1], smallest_relative_singular_value=rel_min)
    return {"rank": rank, "columns": int(M.shape[1]),
            "smallest_relative_singular_value": rel_min}


def upg_structure(ped: Pedigree, groups: GroupAssignment, effect: str,
                  ratio: float | None, meta: dict | None = None) -> GeneticStructure:
    """Genetic structure over ``(u*, g)``: animals in pedigree order, then groups."""
    k_inv, k_diag, logdet = upg_structure_parts(ped, groups, effect, ratio)
    Q = group_fractions(ped, groups)
    info = {"inbreeding_kernel": ped.inbreeding_kernel,
            "base": "unknown parents: genetic groups (QP transformation) where declared, "
                    "otherwise unrelated, non-inbred base animals",
            "upg_effect": effect, "upg_variance_ratio": ratio,
            "n_animals": ped.n, "n_groups": len(groups.labels),
            "groups": list(groups.labels), "group_fractions": Q}
    info.update(meta or {})
    return GeneticStructure("pedigree_upg", tuple(ped.ids) + tuple(groups.labels), k_inv,
                            k_diag, logdet, info)
