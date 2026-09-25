"""Pedigree kernel: gold standards, independent reference, failure cases."""

import numpy as np
import pytest

from abp.core.pedigree import Pedigree
from abp.errors import ABPError
from tests.reference.dense_reference import tabular_a

ATOL, RTOL = 1e-10, 1e-8


def _random_pedigree(n, n_founders, seed, window=40, p_unknown=0.05):
    """Random pedigree with close matings (hence inbreeding) and missing parents."""
    rng = np.random.default_rng(seed)
    ids = [f"a{i}" for i in range(n)]
    sires, dams = [None] * n, [None] * n
    for i in range(n_founders, n):
        lo = max(0, i - window)
        s, d = rng.choice(np.arange(lo, i), size=2, replace=False)
        sires[i] = None if rng.random() < p_unknown else ids[s]
        dams[i] = None if rng.random() < p_unknown else ids[d]
    return ids, sires, dams


def test_t03_hand_derived_gold_standard():
    """Spec T03: founders 1,2; full sibs 3,4 from 1x2; 5 from 3x4."""
    ped = Pedigree.from_parent_ids(["1", "2", "3", "4", "5"],
                                   [None, None, "1", "1", "3"],
                                   [None, None, "2", "2", "4"])
    idx = ped.index_of(["1", "2", "3", "4", "5"])
    F = ped.inbreeding()[idx]
    np.testing.assert_allclose(F, [0, 0, 0, 0, 0.25], atol=ATOL, rtol=RTOL)
    A = ped.a_dense()[np.ix_(idx, idx)]
    np.testing.assert_allclose(A[4], [0.5, 0.5, 0.75, 0.75, 1.25], atol=ATOL, rtol=RTOL)
    Ainv_exact = np.array([[2, 1, -1, -1, 0],
                           [1, 2, -1, -1, 0],
                           [-1, -1, 2.5, 0.5, -1],
                           [-1, -1, 0.5, 2.5, -1],
                           [0, 0, -1, -1, 2]], dtype=float)
    Ainv = ped.ainv().toarray()[np.ix_(idx, idx)]
    np.testing.assert_allclose(Ainv, Ainv_exact, atol=ATOL, rtol=RTOL)


def test_classic_relationships():
    """Half sibs 1/4, full sibs 1/2, parent-offspring 1/2, half-sib mating F=1/8."""
    ped = Pedigree.from_parent_ids(
        ["S", "D1", "D2", "D3", "H1", "H2", "F1", "X"],
        [None, None, None, None, "S", "S", "S", "H1"],
        [None, None, None, None, "D1", "D2", "D1", "H2"])
    A = ped.relationship(["H1", "H1", "H1", "S", "X"], ["H2", "F1", "S", "H1", "X"])
    np.testing.assert_allclose(np.diag(A), [0.25, 0.5, 0.5, 0.5, 1.125], atol=ATOL)


def test_unknown_parents_are_distinct_base_animals():
    """Two animals sharing a sire, each with an unknown dam, are half sibs (1/4),
    not full sibs: different unknown parents are not one common ancestor."""
    ped = Pedigree.from_parent_ids(["S", "X", "Y", "Z"], [None, "S", "S", "X"],
                                   [None, None, None, "Y"])
    A = ped.relationship(["X", "Z"], ["Y", "Z"])
    assert A[0, 0] == pytest.approx(0.25, abs=ATOL)
    assert A[1, 1] == pytest.approx(1.125, abs=ATOL)  # F_Z = A_XY/2
    # d for a single known non-inbred parent is 3/4
    assert ped.mendelian_d()[ped.index_of(["X"])[0]] == pytest.approx(0.75)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_against_tabular_reference(seed):
    ids, sires, dams = _random_pedigree(250, 12, seed)
    ped = Pedigree.from_parent_ids(ids, sires, dams)
    idx = ped.index_of(ids)
    A_ref = tabular_a(ids, sires, dams)
    assert A_ref.diagonal().max() > 1.1, "test pedigree should contain inbreeding"
    F = ped.inbreeding()[idx]
    np.testing.assert_allclose(F, A_ref.diagonal() - 1, atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(ped.a_dense()[np.ix_(idx, idx)], A_ref, atol=ATOL, rtol=RTOL)
    Ainv = ped.ainv().toarray()[np.ix_(idx, idx)]
    np.testing.assert_allclose(Ainv @ A_ref, np.eye(len(ids)), atol=1e-9)
    sign, logdet = np.linalg.slogdet(A_ref)
    assert sign > 0
    assert ped.logdet_a() == pytest.approx(logdet, rel=1e-10, abs=1e-9)
    # Sub-matrix (Colleau) for an arbitrary subset.
    sub = np.array([5, 17, 100, 249, 3])
    np.testing.assert_allclose(ped.a_submatrix(idx[sub]), A_ref[np.ix_(sub, sub)], atol=ATOL)


def test_input_order_invariance():
    ids, sires, dams = _random_pedigree(120, 8, seed=11)
    ped1 = Pedigree.from_parent_ids(ids, sires, dams)
    perm = np.random.default_rng(5).permutation(len(ids))
    ped2 = Pedigree.from_parent_ids([ids[k] for k in perm], [sires[k] for k in perm],
                                    [dams[k] for k in perm])
    A1 = ped1.a_dense()[np.ix_(ped1.index_of(ids), ped1.index_of(ids))]
    A2 = ped2.a_dense()[np.ix_(ped2.index_of(ids), ped2.index_of(ids))]
    np.testing.assert_allclose(A1, A2, atol=ATOL)
    np.testing.assert_allclose(ped1.inbreeding()[ped1.index_of(ids)],
                               ped2.inbreeding()[ped2.index_of(ids)], atol=ATOL)


def test_parents_always_precede_offspring():
    ids, sires, dams = _random_pedigree(200, 10, seed=7)
    rev = list(range(len(ids)))[::-1]  # offspring listed before parents
    ped = Pedigree.from_parent_ids([ids[k] for k in rev], [sires[k] for k in rev],
                                   [dams[k] for k in rev])
    i = np.arange(ped.n)
    assert np.all((ped.sire < i)) and np.all(ped.dam < i)


def test_cycle_is_rejected():
    with pytest.raises(ABPError) as exc:
        Pedigree.from_parent_ids(["A", "B", "C"], ["C", "A", "B"], [None, None, None])
    assert exc.value.code == "ABP-E200"
    assert set(exc.value.details["animals"]) == {"A", "B", "C"}


def test_self_parent_and_same_sire_dam_are_rejected():
    with pytest.raises(ABPError) as exc:
        Pedigree.from_parent_ids(["A"], ["A"], [None])
    assert exc.value.code == "ABP-E201"
    with pytest.raises(ABPError) as exc:
        Pedigree.from_parent_ids(["P", "A"], [None, "P"], [None, "P"])
    assert exc.value.code == "ABP-E202"


def test_a_times_matches_dense_for_matrix_rhs():
    ids, sires, dams = _random_pedigree(80, 6, seed=3)
    ped = Pedigree.from_parent_ids(ids, sires, dams)
    X = np.random.default_rng(0).standard_normal((ped.n, 4))
    np.testing.assert_allclose(ped.a_times(X), ped.a_dense() @ X, atol=1e-10)


def test_native_kernel_matches_python_reference():
    """The optional C++ kernel must reproduce the Python reference exactly
    (up to round-off) on an inbred pedigree, and must reject bad ordering."""
    from abp.core import pedigree as pmod
    if pmod._native is None:
        pytest.skip("native kernel not compiled in this environment (recorded as not_run)")
    ids, sires, dams = _random_pedigree(3000, 30, seed=21, window=60)
    ped = Pedigree.from_parent_ids(ids, sires, dams)
    F_py = pmod.inbreeding_meuwissen_luo(ped.sire, ped.dam)
    F_cpp = np.frombuffer(pmod._native.inbreeding_ml(ped.sire.copy(), ped.dam.copy()))
    assert F_py.max() > 0.05
    np.testing.assert_allclose(F_cpp, F_py, atol=1e-13, rtol=0)
    bad = np.array([-1, 1], dtype=np.int64)  # animal 1 is its own parent
    with pytest.raises(ValueError):
        pmod._native.inbreeding_ml(bad, np.array([-1, -1], dtype=np.int64))


def test_disable_native_switch(monkeypatch):
    monkeypatch.setenv("ABP_DISABLE_NATIVE", "1")
    ped = Pedigree.from_parent_ids(["1", "2", "3", "4", "5"],
                                   [None, None, "1", "1", "3"], [None, None, "2", "2", "4"])
    assert ped.inbreeding()[ped.index_of(["5"])[0]] == pytest.approx(0.25)
    assert ped.inbreeding_kernel == "python_meuwissen_luo"
