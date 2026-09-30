"""Sparse LDL' with minimum-degree ordering: kernels against Python references
and dense algebra, ordering properties, refusals, and agreement with SuperLU on
full BLUP problems."""

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from abp.errors import ABPError
from abp.solvers import cholesky as CH
from abp.solvers.blup import blup
from abp.solvers.mme import make_sparse_factor
from tests.test_blup import animal_term
from tests.test_selinv import _pedigree_problem

NATIVE = CH._native is not None and hasattr(CH._native, "ldl_numeric")


def _spd(n, density, seed, dense_row=False):
    A = sp.random(n, n, density=density, random_state=seed)
    A = (A + A.T) + sp.identity(n) * n
    if dense_row:                                      # an "intercept" linked to everything
        r = sp.csr_matrix((np.ones(n), (np.zeros(n), np.arange(n))), shape=(n, n))
        A = A + r + r.T + sp.identity(n) * 2
    return A.tocsr()


@pytest.mark.parametrize("native", [True, False])
@pytest.mark.parametrize("dense_row", [False, True])
def test_ldl_solve_logdet_inverse_equal_dense(native, dense_row, monkeypatch):
    if native and not CH.native_ldl_available():
        pytest.skip("native kernel not built or disabled (ABP_DISABLE_NATIVE)")
    if not native:
        monkeypatch.setenv("ABP_DISABLE_NATIVE", "1")
    A = _spd(300, 0.02, 4, dense_row)
    f = CH.SparseLDL(A)
    assert ("native_cpp" in f.kernel) == native
    Ad = A.toarray()
    Ai = np.linalg.inv(Ad)
    b = np.random.default_rng(1).standard_normal((300, 3))
    np.testing.assert_allclose(f.solve(b), np.linalg.solve(Ad, b), atol=1e-12)
    np.testing.assert_allclose(f.solve(b[:, 0]), np.linalg.solve(Ad, b[:, 0]), atol=1e-12)
    assert f.logdet() == pytest.approx(np.linalg.slogdet(Ad)[1], abs=1e-9)
    si = f.selected_inverse()
    c = A.tocoo()
    np.testing.assert_allclose(si.entries(c.row, c.col), Ai[c.row, c.col], atol=1e-15)
    if dense_row:                                      # the dense row is ordered last
        assert f.q[-1] == 0


@pytest.mark.skipif(not NATIVE, reason="native kernel not built")
def test_native_ordering_and_numeric_equal_python_reference():
    A = _spd(400, 0.015, 9, True)
    Ac = sp.csc_matrix(A)
    Ac.sort_indices()
    ip, ix = Ac.indptr.astype(np.int64), Ac.indices.astype(np.int64)
    o_py = CH.mindegree_order_python(ip, ix, 400)
    o_nat = np.frombuffer(CH._native.mindegree_order(ip, ix, 400), dtype=np.int64)
    np.testing.assert_array_equal(o_py, o_nat)             # same deterministic order
    B = Ac[o_py][:, o_py]
    B.sort_indices()
    from abp.solvers.selinv import symbolic_cholesky_python
    cp, ri = symbolic_cholesky_python(B.indptr.astype(np.int64), B.indices.astype(np.int64), 400)
    d_py, l_py = CH.ldl_numeric_python(B.indptr, B.indices, B.data, cp, ri, 400)
    d_n, l_n = (np.frombuffer(x, dtype=np.float64) for x in CH._native.ldl_numeric(
        B.indptr.astype(np.int64), B.indices.astype(np.int64), B.data, cp, ri))
    np.testing.assert_allclose(d_n, d_py, rtol=1e-13)
    np.testing.assert_allclose(l_n, l_py, atol=1e-14)


def test_minimum_degree_reduces_fill_against_natural_order():
    ped, rec, X, y = _pedigree_problem(600, 4)
    from abp.solvers.blup import build_system
    S = build_system(y, X, [animal_term(ped, rec)], {"animal": 1.0, "residual": 3.0})
    from abp.solvers.selinv import symbolic_cholesky
    f = CH.SparseLDL(S.C)
    Cn = sp.csc_matrix(S.C)
    Cn.sort_indices()
    _, ri_nat, _ = symbolic_cholesky(Cn)
    assert f.nnz_factor < 0.5 * ri_nat.size
    lu = spla.splu(sp.csc_matrix(S.C), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                   options={"SymmetricMode": True})
    assert f.nnz_factor <= 1.2 * (lu.L.nnz - S.C.shape[0])   # comparable to SuperLU's MMD


def test_not_positive_definite_is_refused():
    A = sp.csr_matrix(np.array([[1.0, 2.0], [2.0, 1.0]]))
    with pytest.raises(ABPError) as exc:
        CH.SparseLDL(A)
    assert exc.value.code == "ABP-E404"


def test_blup_ldl_equals_superlu_and_dense():
    ped, rec, X, y = _pedigree_problem(500, 6)
    term = animal_term(ped, rec)
    vc = {"animal": 1.3, "residual": 2.9}
    d = blup(y, X, [term], vc, method="dense")
    a = blup(y, X, [term], vc, method="sparse_direct", factorization="ldl")
    b = blup(y, X, [term], vc, method="sparse_direct", factorization="superlu")
    assert "LDL'" in a.solve.selection_reason and "SuperLU" in b.solve.selection_reason
    for r in (a, b):
        np.testing.assert_allclose(r.terms["animal"].solution, d.terms["animal"].solution, atol=1e-10)
        np.testing.assert_allclose(r.terms["animal"].pev, d.terms["animal"].pev, atol=1e-12)
    with pytest.raises(ABPError):
        make_sparse_factor(sp.identity(3, format="csr"), 2**30, "cholmod")


def test_numerically_symmetric_matrix_with_asymmetric_pattern():
    """Regression (round 6): an entry of ~1e-17 stored on one side only (cancellation in
    W'R^-1W) made the numeric LDL' step leave the symbolic pattern."""
    import numpy as np
    import scipy.sparse as sp
    from abp.solvers.cholesky import SparseLDL
    rng = np.random.default_rng(0)
    B = sp.random(60, 60, density=0.05, random_state=1)
    C = (B @ B.T + 60 * sp.identity(60)).tolil()
    C[5, 40] = 1e-17                     # stored on one side only
    C = C.tocsr()
    f = SparseLDL(C)
    b = rng.standard_normal(60)
    Cs = 0.5 * (C + C.T).toarray()
    np.testing.assert_allclose(f.solve(b), np.linalg.solve(Cs, b), rtol=1e-10, atol=1e-12)
