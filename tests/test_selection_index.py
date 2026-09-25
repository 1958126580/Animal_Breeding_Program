"""Selection index: spec T02 gold standard, unit invariance, restricted index
KKT check, collinearity and response."""

import numpy as np
import pytest

from abp.decision.selection_index import ebv_index, selection_intensity, smith_hazel
from abp.errors import ABPError

P = np.array([[4.0, 1.0], [1.0, 9.0]])


def test_t02_gold_standard():
    r = smith_hazel(P, np.array([[2.0], [3.0]]), np.array([1.0]), np.array([[5.0]]),
                    ["x1", "x2"], ["H"])
    np.testing.assert_allclose(r.b, [3 / 7, 2 / 7], atol=1e-14)
    assert float(r.b @ [2.0, -1.0]) == pytest.approx(4 / 7, abs=1e-14)
    assert r.reliability == pytest.approx(12 / 35, abs=1e-14)
    assert r.accuracy == pytest.approx(0.585540043769, abs=1e-12)
    # target scaling: q x10, Var(H) x100 -> identical reliability
    r10 = smith_hazel(P, np.array([[20.0], [30.0]]), np.array([1.0]), np.array([[500.0]]),
                      ["x1", "x2"], ["H"])
    assert r10.reliability == pytest.approx(r.reliability, abs=1e-14)


def test_unit_change_of_information_source():
    D = np.diag([1000.0, 1.0])  # x1 recorded in g instead of kg
    C = np.array([[2.0, 0.5], [3.0, 1.0]])
    a = np.array([1.0, 2.0])
    G_H = np.array([[5.0, 1.0], [1.0, 2.0]])
    r1 = smith_hazel(P, C, a, G_H, ["x1", "x2"], ["t1", "t2"])
    r2 = smith_hazel(D @ P @ D, D @ C, a, G_H, ["x1", "x2"], ["t1", "t2"])
    np.testing.assert_allclose(r2.b, np.linalg.solve(D, r1.b), rtol=1e-12)
    assert r2.reliability == pytest.approx(r1.reliability, rel=1e-12)


def test_restricted_index_satisfies_kkt():
    P3 = np.array([[4.0, 1.0, 0.5], [1.0, 9.0, 1.0], [0.5, 1.0, 2.0]])
    C = np.array([[2.0, 0.5], [3.0, -1.0], [0.4, 0.8]])
    a = np.array([1.0, 1.5])
    G_H = np.array([[5.0, 0.3], [0.3, 1.2]])
    r = smith_hazel(P3, C, a, G_H, ["x1", "x2", "x3"], ["t1", "t2"], restrict=["t2"],
                    proportion_selected=0.2)
    assert r.restriction_residual < 1e-12
    assert abs(r.response_traits[1]) < 1e-12        # no change in the restricted trait
    # independent KKT solution of  min Var(I - H)  s.t.  Cr'b = 0
    Cr = C[:, [1]]
    K = np.block([[P3, Cr], [Cr.T, np.zeros((1, 1))]])
    sol = np.linalg.solve(K, np.concatenate([C @ a, [0.0]]))
    np.testing.assert_allclose(r.b, sol[:3], atol=1e-12)
    unres = smith_hazel(P3, C, a, G_H, ["x1", "x2", "x3"], ["t1", "t2"])
    assert r.reliability <= unres.reliability + 1e-12


def test_collinear_information_is_reported():
    Pbad = np.array([[4.0, 2.0], [2.0, 1.0]])  # x2 = x1 / 2 : duplicated information
    with pytest.raises(ABPError) as exc:
        smith_hazel(Pbad, np.array([[2.0], [1.0]]), np.array([1.0]), np.array([[5.0]]),
                    ["x1", "x2"], ["H"])
    assert exc.value.code == "ABP-E300"
    assert set(exc.value.details["combination"]) == {"x1", "x2"}


def test_selection_intensity_and_response():
    assert selection_intensity(0.1) == pytest.approx(1.75498, abs=1e-5)
    assert selection_intensity(0.5) == pytest.approx(0.79788, abs=1e-5)
    r = smith_hazel(P, np.array([[2.0], [3.0]]), np.array([1.0]), np.array([[5.0]]),
                    ["x1", "x2"], ["H"], proportion_selected=0.1)
    assert r.response_objective == pytest.approx(selection_intensity(0.1) * np.sqrt(12 / 7))


def test_ebv_index_reliability():
    ebv = np.array([[1.0, 2.0], [0.0, -1.0]])
    pev = np.array([[[0.5, 0.1], [0.1, 0.4]], [[0.9, 0.0], [0.0, 0.8]]])
    G0 = np.array([[1.0, 0.2], [0.2, 1.0]])
    idx, rel = ebv_index(ebv, pev, np.array([2.0, 1.0]), G0, np.array([1.0, 1.0]))
    np.testing.assert_allclose(idx, [4.0, -1.0])
    var_h = 4 * 1 + 2 * 2 * 1 * 0.2 + 1
    np.testing.assert_allclose(rel, [1 - (4 * 0.5 + 4 * 0.1 + 0.4) / var_h,
                                     1 - (4 * 0.9 + 0.8) / var_h])
