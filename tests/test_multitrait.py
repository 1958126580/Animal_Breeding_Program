"""Multi-trait BLUP: Kronecker stacking gold standard, degeneration, missing
patterns and PEV blocks against an independent trait-major V-form reference."""

import numpy as np
import pytest
import scipy.sparse as sp

from abp.core.design import FixedTerm, build_fixed_design
from abp.core.pedigree import Pedigree
from abp.errors import ABPError
from abp.solvers.blup import blup
from abp.solvers.multitrait import MTData, build_and_solve
from tests.reference.dense_reference import tabular_a
from tests.test_blup import animal_term


def _reference(Y, X_blocks, animal_of_rec, A, G0, R0):
    """Trait-major V-form BLUP: vec(U) = [u_trait1 (all animals); u_trait2; ...]."""
    n_rec, t = Y.shape
    q = A.shape[0]
    obs = [(r, j) for r in range(n_rec) for j in range(t) if not np.isnan(Y[r, j])]
    y = np.array([Y[r, j] for r, j in obs])
    p_off = np.cumsum([0] + [Xj.shape[1] for Xj in X_blocks])
    X = np.zeros((len(obs), p_off[-1]))
    row_in_trait = {j: {r: k for k, r in enumerate(np.flatnonzero(~np.isnan(Y[:, j])))}
                    for j in range(t)}
    for o, (r, j) in enumerate(obs):
        X[o, p_off[j]:p_off[j + 1]] = X_blocks[j][row_in_trait[j][r]]
    Z = np.zeros((len(obs), t * q))
    for o, (r, j) in enumerate(obs):
        Z[o, j * q + animal_of_rec[r]] = 1.0          # trait-major column
    Gfull = np.kron(G0, A)                            # Var(vec_trait_major(U))
    R = np.zeros((len(obs), len(obs)))
    for a, (ra, ja) in enumerate(obs):
        for b, (rb, jb) in enumerate(obs):
            if ra == rb:
                R[a, b] = R0[ja, jb]
    V = Z @ Gfull @ Z.T + R
    Vi = np.linalg.inv(V)
    P = Vi - Vi @ X @ np.linalg.solve(X.T @ Vi @ X, X.T @ Vi) if X.shape[1] else Vi
    u = Gfull @ Z.T @ P @ y
    pev = Gfull - Gfull @ Z.T @ P @ Z @ Gfull
    return u.reshape(t, q).T, pev  # animal x trait; pev in trait-major order


def test_two_animals_two_traits_kronecker_order():
    """Spec item 5: 2 animals x 2 traits; animal 2 is the offspring of animal 1."""
    ped = Pedigree.from_parent_ids(["1", "2"], [None, "1"], [None, None])
    A = tabular_a(["1", "2"], [None, "1"], [None, None])
    G0 = np.array([[1.0, 0.5], [0.5, 2.0]])
    R0 = np.array([[1.0, 0.2], [0.2, 1.5]])
    Y = np.array([[1.0, 2.0], [0.5, np.nan]])
    idx = ped.index_of(["1", "2"])
    Xb = [sp.csr_matrix(np.zeros((2, 0))), sp.csr_matrix(np.zeros((1, 0)))]
    res = build_and_solve(MTData(Y, Xb, idx), ped.ainv(), 1 + ped.inbreeding(), G0, R0,
                          method="dense")
    u_ref, pev_ref = _reference(Y, [np.zeros((2, 0)), np.zeros((1, 0))], [0, 1], A, G0, R0)
    np.testing.assert_allclose(res.ebv[idx], u_ref, atol=1e-12)
    for a in range(2):
        blk = pev_ref[np.ix_([a, 2 + a], [a, 2 + a])]
        np.testing.assert_allclose(res.pev_blocks[idx[a]], blk, atol=1e-12)
    # The genetic covariance must propagate information to the unobserved trait:
    assert res.ebv[idx[1], 1] != 0.0


def _random_mt(seed, n_anim=50, n_rec=70, t=3):
    rng = np.random.default_rng(seed)
    ids = [f"a{i}" for i in range(n_anim)]
    sires, dams = [None] * n_anim, [None] * n_anim
    for i in range(8, n_anim):
        s, d = rng.choice(np.arange(max(0, i - 20), i), 2, replace=False)
        sires[i], dams[i] = ids[s], ids[d]
    rec_anim = rng.choice(np.arange(n_anim), n_rec, replace=False) if n_rec <= n_anim else \
        rng.integers(0, n_anim, n_rec)
    Y = rng.normal(size=(n_rec, t)) + np.arange(t)
    Y[rng.random((n_rec, t)) < 0.3] = np.nan
    Y[np.all(np.isnan(Y), axis=1), 0] = 1.0
    grp = [f"g{k}" for k in rng.integers(0, 3, n_rec)]
    return ids, sires, dams, rec_anim, Y, grp, rng


def test_missing_patterns_fixed_per_trait_against_reference():
    ids, sires, dams, rec_anim, Y, grp, rng = _random_mt(3, n_anim=45, n_rec=45)
    ped = Pedigree.from_parent_ids(ids, sires, dams)
    A = tabular_a(ids, sires, dams)
    t = Y.shape[1]
    G0 = np.array([[2.0, 0.6, -0.3], [0.6, 1.0, 0.2], [-0.3, 0.2, 0.8]])
    R0 = np.array([[3.0, 0.9, 0.0], [0.9, 2.0, 0.0], [0.0, 0.0, 1.0]])  # trait 3: structural zero
    Xb, Xd = [], []
    for j in range(t):
        rows = np.flatnonzero(~np.isnan(Y[:, j]))
        terms = [FixedTerm("g", "factor")] if j < 2 else []
        fd = build_fixed_design({"g": [grp[r] for r in rows]}, terms, True, rows.size)
        Xb.append(fd.X)
        Xd.append(fd.X.toarray())
    idx_rec = ped.index_of([ids[a] for a in rec_anim])
    res = build_and_solve(MTData(Y, Xb, idx_rec), ped.ainv(), 1 + ped.inbreeding(), G0, R0,
                          method="dense")
    u_ref, pev_ref = _reference(Y, Xd, list(rec_anim), A, G0, R0)
    order = ped.index_of(ids)
    np.testing.assert_allclose(res.ebv[order], u_ref, atol=1e-9)
    q = len(ids)
    for a in range(q):
        sel = [j * q + a for j in range(t)]
        np.testing.assert_allclose(res.pev_blocks[order[a]], pev_ref[np.ix_(sel, sel)], atol=1e-9)
    # sparse direct path gives the same blocks
    res2 = build_and_solve(MTData(Y, Xb, idx_rec), ped.ainv(), 1 + ped.inbreeding(), G0, R0,
                           method="sparse_direct")
    np.testing.assert_allclose(res2.pev_blocks, res.pev_blocks, atol=1e-9)
    np.testing.assert_allclose(res2.ebv, res.ebv, atol=1e-9)


def test_single_trait_degeneration_and_independent_traits():
    ids, sires, dams, rec_anim, Y, grp, rng = _random_mt(5, n_anim=60, n_rec=90, t=2)
    ped = Pedigree.from_parent_ids(ids, sires, dams)
    rec_ids = [ids[a] for a in rec_anim]
    idx_rec = ped.index_of(rec_ids)
    G0 = np.diag([2.0, 0.5])
    R0 = np.diag([3.0, 1.0])
    Xb = []
    for j in range(2):
        rows = np.flatnonzero(~np.isnan(Y[:, j]))
        Xb.append(build_fixed_design({"g": [grp[r] for r in rows]}, [FixedTerm("g", "factor")],
                                     True, rows.size).X)
    mt = build_and_solve(MTData(Y, Xb, idx_rec), ped.ainv(), 1 + ped.inbreeding(), G0, R0)
    for j in range(2):
        rows = np.flatnonzero(~np.isnan(Y[:, j]))
        st = blup(Y[rows, j], Xb[j], [animal_term(ped, [rec_ids[r] for r in rows])],
                  {"animal": G0[j, j], "residual": R0[j, j]})
        np.testing.assert_allclose(mt.ebv[:, j], st.terms["animal"].solution, atol=1e-10)
        np.testing.assert_allclose(mt.pev_blocks[:, j, j], st.terms["animal"].pev, atol=1e-10)
        np.testing.assert_allclose(mt.reliability[:, j], st.terms["animal"].reliability, atol=1e-10)
    # one trait = single-trait model
    mt1 = build_and_solve(MTData(Y[:, :1], Xb[:1], idx_rec), ped.ainv(), 1 + ped.inbreeding(),
                          G0[:1, :1], R0[:1, :1])
    np.testing.assert_allclose(mt1.ebv[:, 0], mt.ebv[:, 0], atol=1e-10)


def test_unit_change_of_one_trait():
    ids, sires, dams, rec_anim, Y, grp, rng = _random_mt(8, n_anim=40, n_rec=40, t=2)
    ped = Pedigree.from_parent_ids(ids, sires, dams)
    idx_rec = ped.index_of([ids[a] for a in rec_anim])
    G0 = np.array([[2.0, 0.4], [0.4, 0.5]])
    R0 = np.array([[3.0, 0.5], [0.5, 1.0]])
    Xb = [build_fixed_design({}, [], True, int((~np.isnan(Y[:, j])).sum())).X for j in range(2)]
    base = build_and_solve(MTData(Y, Xb, idx_rec), ped.ainv(), 1 + ped.inbreeding(), G0, R0)
    D = np.diag([1000.0, 1.0])                                  # trait 1: kg -> g
    Y2 = Y * np.array([1000.0, 1.0])
    scaled = build_and_solve(MTData(Y2, Xb, idx_rec), ped.ainv(), 1 + ped.inbreeding(),
                             D @ G0 @ D, D @ R0 @ D)
    np.testing.assert_allclose(scaled.ebv, base.ebv * np.array([1000.0, 1.0]), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(scaled.reliability, base.reliability, atol=1e-10)


def test_invalid_covariance_rejected():
    ped = Pedigree.from_parent_ids(["1", "2"], [None, None], [None, None])
    Y = np.array([[1.0, 2.0], [0.5, 1.0]])
    Xb = [sp.csr_matrix(np.ones((2, 1))), sp.csr_matrix(np.ones((2, 1)))]
    bad = np.array([[1.0, 1.5], [1.5, 1.0]])  # correlation 1.5
    with pytest.raises(ABPError) as exc:
        build_and_solve(MTData(Y, Xb, ped.index_of(["1", "2"])), ped.ainv(),
                        1 + ped.inbreeding(), bad, np.eye(2))
    assert exc.value.code == "ABP-E301"
