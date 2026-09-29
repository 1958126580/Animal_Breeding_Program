"""Sparse selected inversion (Takahashi): kernels against dense inverses, pattern
closure with exact cancellations, and the sparse PEV / REML paths against the
dense paths."""

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from abp.core.pedigree import Pedigree
from abp.errors import ABPError
from abp.solvers import selinv as S
from abp.solvers.blup import blup
from abp.solvers.mme import SparseLU, choose_method
from abp.solvers.multitrait import MTData, build_and_solve
from abp.solvers.reml import REMLEvaluator
from tests.test_blup import animal_term

NATIVE = S._native is not None and hasattr(S._native, "takahashi")


def _spd(n, density, seed):
    A = sp.random(n, n, density=density, random_state=seed)
    return ((A + A.T) + sp.identity(n) * n).tocsc()


def _lu(A):
    return splu(A.tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                options={"SymmetricMode": True})


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_selected_inverse_equals_dense_inverse_on_pattern(seed):
    A = _spd(250, 0.03, seed)
    si = S.SelectedInverse(_lu(A), A)
    Ai = np.linalg.inv(A.toarray())
    np.testing.assert_allclose(si.diagonal(np.arange(250)), np.diag(Ai), rtol=0, atol=1e-15)
    c = A.tocoo()
    np.testing.assert_allclose(si.entries(c.row, c.col), Ai[c.row, c.col], rtol=0, atol=1e-15)
    # every stored entry (the whole factor pattern, not only C's pattern)
    q = np.argsort(si.pos)
    cols = np.repeat(np.arange(250), np.diff(si.colptr))
    np.testing.assert_allclose(si.zval, Ai[q[si.rowidx], q[cols]], atol=1e-15)


@pytest.mark.skipif(not NATIVE, reason="native kernel not built")
def test_native_kernels_equal_python_reference():
    A = _spd(400, 0.02, 7)
    B = A.tocsc()
    B.sort_indices()
    ip, ix = B.indptr.astype(np.int64), B.indices.astype(np.int64)
    cp, ri = S.symbolic_cholesky_python(ip, ix, 400)
    cpn, rin = (np.frombuffer(x, dtype=np.int64) for x in S._native.symbolic_cholesky(ip, ix, 400))
    np.testing.assert_array_equal(cp, cpn)
    np.testing.assert_array_equal(ri, rin)
    rng = np.random.default_rng(1)
    lval = rng.normal(0, 0.1, ri.size)
    d = rng.uniform(1, 2, 400)
    zd, zv = S.takahashi_python(cp, ri, lval, d)
    zdn, zvn = (np.frombuffer(x, dtype=np.float64) for x in S._native.takahashi(cp, ri, lval, d))
    np.testing.assert_allclose(zdn, zd, rtol=0, atol=1e-13)
    # same algorithm, different summation order: absolute agreement at the value scale (~1)
    np.testing.assert_allclose(zvn, zv, rtol=0, atol=1e-13)
    # the selected inverse of L D L' built from these values is its true inverse
    n = 400
    L = sp.csc_matrix((lval, ri, cp), shape=(n, n)) + sp.identity(n)
    Zi = np.linalg.inv((L @ sp.diags(d) @ L.T).toarray())
    np.testing.assert_allclose(zd, np.diag(Zi), atol=1e-12)


def test_symbolic_pattern_closes_over_numeric_cancellation():
    """A matrix whose numeric factor has an exact zero inside the symbolic
    pattern: SuperLU drops it, the symbolic pattern keeps it, and the result
    is still the exact inverse."""
    # arrow + chain: eliminating 0 creates fill (1,2) that cancels exactly with C[1,2]
    C = np.array([[2.0, 1.0, 1.0, 0.0],
                  [1.0, 3.0, 0.5, 1.0],
                  [1.0, 0.5, 3.0, 1.0],
                  [0.0, 1.0, 1.0, 3.0]])
    Cs = sp.csc_matrix(C)
    lu = splu(Cs, permc_spec="NATURAL", diag_pivot_thresh=0.0, options={"SymmetricMode": True})
    assert sp.tril(lu.L, -1).nnz < 5          # the (2,1) entry of L is exactly zero -> dropped
    si = S.SelectedInverse(lu, Cs)
    assert si.nnz_factor == 5                 # symbolic pattern keeps it
    Ci = np.linalg.inv(C)
    r, c = np.nonzero(np.ones((4, 4)) - np.eye(4) * 0)
    keep = ~((r == 0) & (c == 3)) & ~((r == 3) & (c == 0))
    np.testing.assert_allclose(si.entries(r[keep], c[keep]), Ci[r[keep], c[keep]], atol=1e-14)


def test_entries_outside_the_pattern_are_refused():
    C = sp.csc_matrix(np.diag([2.0, 3.0, 4.0]))
    si = S.SelectedInverse(_lu(C), C)
    with pytest.raises(ABPError, match="outside the factor pattern"):
        si.entries(np.array([0]), np.array([2]))


def test_memory_guard():
    A = _spd(200, 0.05, 3)
    with pytest.raises(ABPError) as exc:
        S.SelectedInverse(_lu(A), A, memory_budget_bytes=1000)
    assert exc.value.code == "ABP-E500"


def test_no_equation_limit_for_exact_pev():
    method, reason = choose_method(500_000, True, 8 * 2**30)
    assert method == "sparse_direct" and "selected inversion" in reason


def _pedigree_problem(n=600, seed=4):
    rng = np.random.default_rng(seed)
    ids = [str(i) for i in range(n)]
    sires, dams = [], []
    for i in range(n):
        if i < 40:
            sires.append(None)
            dams.append(None)
        else:
            s, d = rng.choice(i, 2, replace=False)
            sires.append(str(s))
            dams.append(str(d) if rng.random() > 0.1 else None)
    ped = Pedigree.from_parent_ids(ids, sires, dams)
    rec = [ids[k] for k in rng.integers(40, n, 900)]
    groups = rng.integers(0, 12, len(rec))
    X = sp.csr_matrix((np.ones(len(rec)), (np.arange(len(rec)), groups)), shape=(len(rec), 12))
    y = rng.normal(0, 2, len(rec)) + groups * 0.3
    return ped, rec, X, y


def test_sparse_pev_and_reliability_equal_dense_path():
    ped, rec, X, y = _pedigree_problem()
    term = animal_term(ped, rec)
    vc = {"animal": 1.5, "residual": 3.0}
    dense = blup(y, X, [term], vc, method="dense")
    sparse = blup(y, X, [term], vc, method="sparse_direct")
    np.testing.assert_allclose(sparse.terms["animal"].solution, dense.terms["animal"].solution,
                               atol=1e-10)
    np.testing.assert_allclose(sparse.terms["animal"].pev, dense.terms["animal"].pev, atol=1e-12)
    np.testing.assert_allclose(sparse.terms["animal"].reliability,
                               dense.terms["animal"].reliability, atol=1e-12)


def test_sparse_reml_traces_equal_dense_path(monkeypatch):
    ped, rec, X, y = _pedigree_problem(400, 9)
    term = animal_term(ped, rec)
    theta = np.array([1.2, 2.7])
    dense = REMLEvaluator(y, X, [term], 2**34)
    assert dense.trace_method == "dense_inverse"
    p_d = dense.evaluate(theta)
    import abp.solvers.reml as R
    monkeypatch.setattr(R, "DENSE_REML_MAX", 10)
    sparse = REMLEvaluator(y, X, [term], 2**34)
    assert sparse.trace_method == "sparse_selected_inversion"
    p_s = sparse.evaluate(theta)
    assert p_s.loglik == pytest.approx(p_d.loglik, abs=1e-8)
    np.testing.assert_allclose(p_s.score, p_d.score, atol=1e-9)
    np.testing.assert_allclose(p_s.em, p_d.em, rtol=1e-11)
    np.testing.assert_allclose(p_s.ai, p_d.ai, rtol=1e-10)


def test_multitrait_pev_blocks_sparse_equal_dense():
    ped, rec, X, y = _pedigree_problem(300, 5)
    rng = np.random.default_rng(2)
    Y = np.column_stack([y, 0.5 * y + rng.normal(0, 1, y.size)])
    Y[rng.random(Y.shape) < 0.15] = np.nan                   # missing records
    Y[np.isnan(Y).all(axis=1), 0] = 1.0
    X = X.tocsr()
    data = MTData(Y, [X[np.flatnonzero(~np.isnan(Y[:, j]))] for j in range(2)],
                  ped.index_of(rec))
    kd = 1.0 + ped.inbreeding()

    def both(G0, R0):
        d = build_and_solve(data, ped.ainv(), kd, G0, R0, method="dense")
        s = build_and_solve(data, ped.ainv(), kd, G0, R0, method="sparse_direct")
        np.testing.assert_allclose(s.ebv, d.ebv, atol=1e-10)
        np.testing.assert_allclose(s.pev_blocks, d.pev_blocks, atol=1e-12)

    both(np.array([[1.0, 0.4], [0.4, 0.8]]), np.array([[3.0, 0.6], [0.6, 2.0]]))
    # uncorrelated traits: the blocks are off the factor pattern -> solve fallback, still exact
    both(np.diag([1.0, 0.8]), np.diag([3.0, 2.0]))


def test_sparse_lu_selected_inverse_is_cached():
    A = _spd(50, 0.1, 1)
    f = SparseLU(A.tocsr())
    assert f.selected_inverse() is f.selected_inverse()


def test_sparse_reml_ignores_stored_zeros_in_k_inverse(monkeypatch):
    """Single-trait sparse REML with an explicit zero stored in K^-1 (regression)."""
    ped, rec, X, y = _pedigree_problem(300, 3)
    term = animal_term(ped, rec)
    Ki = term.k_inv.tocoo()
    i, j = 0, ped.n - 1
    Kz = sp.csr_matrix((np.r_[Ki.data, 0.0, 0.0], (np.r_[Ki.row, i, j], np.r_[Ki.col, j, i])),
                       shape=Ki.shape)
    term.k_inv = Kz
    import abp.solvers.reml as R
    theta = np.array([1.2, 2.7])
    dense = REMLEvaluator(y, X, [term], 2**34).evaluate(theta)
    monkeypatch.setattr(R, "DENSE_REML_MAX", 10)
    sparse = REMLEvaluator(y, X, [term], 2**34).evaluate(theta)
    assert sparse.loglik == pytest.approx(dense.loglik, abs=1e-8)
    np.testing.assert_allclose(sparse.score, dense.score, atol=1e-9)
