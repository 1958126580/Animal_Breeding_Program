"""Unknown-parent groups: the QP identity, and BLUP of u* = u + Qg against an
independent V-form model with explicit group effects."""

import numpy as np
import pytest
import scipy.sparse as sp

from abp.core.design import build_fixed_design
from abp.core.pedigree import Pedigree
from abp.core.upg import (GroupAssignment, ainv_with_groups, check_fixed_groups_estimable,
                          group_fractions, upg_structure_parts)
from abp.errors import ABPError
from abp.solvers.blup import RandomTerm, blup
from tests.reference.dense_reference import blup_v_form, tabular_a

# animals 1-3 founders from groups; 4 has a sire and an unknown dam of group G2
PED = [("1", "G1", "G2"), ("2", "G1", "G2"), ("3", "G2", "G2"), ("4", "1", "G2"),
       ("5", "3", "2"), ("6", "1", "2"), ("7", "4", "5"), ("8", "3", "6"), ("9", "8", None)]
GROUPS = ("G1", "G2")


def _setup():
    ids = [a for a, _, _ in PED]
    sires = [s if s in ids else None for _, s, _ in PED]
    dams = [d if d in ids else None for _, _, d in PED]
    ped = Pedigree.from_parent_ids(ids, sires, dams)
    gpos = {g: k for k, g in enumerate(GROUPS)}
    sg = np.full(ped.n, -1)
    dg = np.full(ped.n, -1)
    for a, s, d in PED:
        i = ped.index_of([a])[0]
        if s in gpos:
            sg[i] = gpos[s]
        if d in gpos:
            dg[i] = gpos[d]
    return ids, sires, dams, ped, GroupAssignment(GROUPS, sg, dg)


def _q_independent(ids):
    """Gene fractions by direct recursion on the input order (parents listed first)."""
    q = {}
    for a, s, d in PED:
        v = np.zeros(2)
        for p in (s, d):
            if p in GROUPS:
                v[GROUPS.index(p)] += 0.5
            elif p is not None:
                v += 0.5 * q[p]
        q[a] = v
    return np.array([q[a] for a in ids])


def test_qp_identity():
    ids, sires, dams, ped, grp = _setup()
    idx = ped.index_of(ids)
    A = tabular_a(ids, sires, dams)
    Ai = np.linalg.inv(A)
    Q = _q_independent(ids)
    np.testing.assert_allclose(group_fractions(ped, grp)[idx], Q, atol=1e-15)
    expected = np.block([[Ai, -Ai @ Q], [-Q.T @ Ai, Q.T @ Ai @ Q]])
    got = ainv_with_groups(ped, grp).toarray()
    order = np.concatenate([idx, [ped.n, ped.n + 1]])
    np.testing.assert_allclose(got[np.ix_(order, order)], expected, atol=1e-12)


def test_random_groups_blup_equals_explicit_model():
    """y = mu + Z(u + Q g) + e with g ~ N(0, s_g I): V-form reference vs the
    QP-transformed MME (animal solutions = u*, PEV of u*, group solutions)."""
    ids, sires, dams, ped, grp = _setup()
    rng = np.random.default_rng(3)
    rec = ["4", "5", "6", "7", "8", "9", "5", "8"]
    y = rng.normal(10, 2, len(rec))
    sa, se, ratio = 2.0, 4.0, 0.5
    k_inv, k_diag, logdet = upg_structure_parts(ped, grp, "random", ratio)
    n, ng = ped.n, 2
    cols = ped.index_of(rec)
    Z = sp.csr_matrix((np.ones(len(rec)), (np.arange(len(rec)), cols)), shape=(len(rec), n + ng))
    labels = list(ped.ids) + list(GROUPS)
    X = build_fixed_design({}, [], True, len(rec)).X
    res = blup(y, X, [RandomTerm("animal", Z, k_inv, labels, True, k_diag=k_diag,
                                 logdet_k=logdet)], {"animal": sa, "residual": se}, method="dense")
    # independent reference on the vector (u*, g) with its explicit covariance
    A = tabular_a(ids, sires, dams)
    Q = _q_independent(ids)
    Vu = sa * A + sa * ratio * Q @ Q.T
    Cug = sa * ratio * Q
    Gfull = np.block([[Vu, Cug], [Cug.T, sa * ratio * np.eye(2)]])
    pos = {a: k for k, a in enumerate(ids)}
    Zr = np.zeros((len(rec), 11))
    Zr[np.arange(len(rec)), [pos[a] for a in rec]] = 1
    _, u_ref, pev_ref = blup_v_form(y, np.ones((len(rec), 1)), Zr, Gfull, se * np.eye(len(rec)))
    sol = res.terms["animal"].solution
    order = np.concatenate([ped.index_of(ids), [n, n + 1]])
    np.testing.assert_allclose(sol[order], u_ref, atol=1e-9)
    np.testing.assert_allclose(res.terms["animal"].pev[order], np.diag(pev_ref), atol=1e-9)
    # log|K| used by REML equals the log-determinant of the explicit covariance / sigma_a^2
    assert logdet == pytest.approx(np.linalg.slogdet(Gfull / sa)[1], abs=1e-10)


def _fixed_setup(rec):
    ids, sires, dams, ped, grp = _setup()
    k_inv, k_diag, logdet = upg_structure_parts(ped, grp, "fixed", None)
    Z = sp.csr_matrix((np.ones(len(rec)), (np.arange(len(rec)), ped.index_of(rec))),
                      shape=(len(rec), ped.n + 2))
    X = build_fixed_design({}, [], True, len(rec)).X
    return ids, sires, dams, ped, grp, k_inv, k_diag, logdet, Z, X


def test_fixed_groups_confounded_with_intercept_are_refused():
    """Every recorded lineage (animals 4-8) ends in groups: the rows of ZQ sum
    to one, so the group effects are confounded with the intercept."""
    rec = ["4", "5", "6", "7", "8"]
    ids, sires, dams, ped, grp, k_inv, k_diag, logdet, Z, X = _fixed_setup(rec)
    assert logdet is None and np.all(np.isnan(k_diag))
    ZQ = Z[:, :ped.n] @ group_fractions(ped, grp)
    np.testing.assert_allclose(ZQ.sum(axis=1), 1.0)
    with pytest.raises(ABPError) as exc:
        check_fixed_groups_estimable(X, ZQ, GROUPS)
    assert exc.value.code == "ABP-E300"
    # a group without recorded descendants is not estimable either
    with pytest.raises(ABPError, match="without recorded descendants"):
        check_fixed_groups_estimable(X[:1], ZQ[:1] * [1.0, 0.0], GROUPS)


def test_fixed_groups_blup_equals_explicit_model():
    """Animal 9 has a plain unknown dam, so with its record [X, ZQ] has full
    rank; the QP-transformed MME must reproduce the explicit model
    y = mu + ZQ g + Z u + e with (mu, g) fixed: u* = u + Q g and
    PEV(u*) = Var(T [b; g; u]) from the explicit MME inverse."""
    rec = ["4", "5", "6", "7", "8", "9", "9", "7"]
    ids, sires, dams, ped, grp, k_inv, k_diag, logdet, Z, X = _fixed_setup(rec)
    Q = group_fractions(ped, grp)
    info = check_fixed_groups_estimable(X, Z[:, :ped.n] @ Q, GROUPS)
    assert info["rank"] == 3
    y = np.array([3.0, 5.0, 4.0, 6.5, 2.0, 7.0, 6.0, 5.5])
    sa, se = 1.5, 3.0
    labels = list(ped.ids) + list(GROUPS)
    res = blup(y, X, [RandomTerm("animal", Z, k_inv, labels, True, k_diag=k_diag)],
               {"animal": sa, "residual": se}, method="dense")
    tr = res.terms["animal"]
    assert np.all(np.isnan(tr.reliability))
    # explicit MME in (mu, g, u), pedigree order, with A^-1 from the tabular method
    n = ped.n
    order = ped.index_of(ids)
    A = tabular_a(ids, sires, dams)
    Ainv = np.zeros((n, n))
    Ainv[np.ix_(order, order)] = np.linalg.inv(A)
    Zu = Z[:, :n].toarray()
    Wf = np.hstack([X.toarray(), Zu @ Q])
    W = np.hstack([Wf, Zu])
    C = W.T @ W / se
    C[3:, 3:] += Ainv / sa
    sol = np.linalg.solve(C, W.T @ y / se)
    Cinv = np.linalg.inv(C)
    T = np.zeros((n + 2, 3 + n))                    # (u*, g) = T (mu, g, u)
    T[:n, 1:3] = Q
    T[:n, 3:] = np.eye(n)
    T[n:, 1:3] = np.eye(2)
    np.testing.assert_allclose(tr.solution, T @ sol, atol=1e-10)
    np.testing.assert_allclose(res.fixed_solution, sol[:1], atol=1e-10)
    np.testing.assert_allclose(tr.pev, np.diag(T @ Cinv @ T.T), atol=1e-10)
