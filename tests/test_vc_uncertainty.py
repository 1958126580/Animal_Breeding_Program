"""Kackar-Harville PEV correction for estimated variances: against derivatives
of the independent V-form BLUP, and in the workflow."""

import csv
from pathlib import Path

import numpy as np

from abp.solvers.vc_uncertainty import kackar_harville_delta
from abp.workflows.evaluate import run_evaluation
from tests.reference.dense_reference import blup_v_form, tabular_a
from tests.test_blup import animal_term

ROOT = Path(__file__).resolve().parents[1]


def test_delta_equals_v_form_derivatives():
    from abp.core.design import build_fixed_design
    from abp.core.pedigree import Pedigree
    rng = np.random.default_rng(3)
    n = 40
    ids = [str(i) for i in range(n)]
    s = [None] * 6 + [ids[rng.integers(0, i)] for i in range(6, n)]
    d = [None] * 6 + [ids[rng.integers(0, i)] for i in range(6, n)]
    d = [x if x != sx else None for x, sx in zip(d, s)]
    ped = Pedigree.from_parent_ids(ids, s, d)
    rec = [ids[k] for k in rng.integers(0, n, 70)]
    y = rng.normal(5, 2, 70)
    X = build_fixed_design({}, [], True, 70).X
    term = animal_term(ped, rec)
    vc = {"animal": 1.4, "residual": 3.1}
    cov = np.array([[0.30, -0.12], [-0.12, 0.25]])
    delta = kackar_harville_delta(y, X, [term], vc, cov, ["animal", "residual"], "animal")
    A = tabular_a(ids, s, d)
    order = ped.index_of(ids)
    Z = np.zeros((70, n))
    Z[np.arange(70), [ids.index(a) for a in rec]] = 1

    def u(sa, se):
        return blup_v_form(y, np.ones((70, 1)), Z, sa * A, se * np.eye(70))[1]
    h = 1e-5
    g = np.column_stack([(u(1.4 + h, 3.1) - u(1.4 - h, 3.1)) / (2 * h),
                         (u(1.4, 3.1 + h) - u(1.4, 3.1 - h)) / (2 * h)])
    ref = np.einsum("ik,kl,il->i", g, cov, g)
    np.testing.assert_allclose(delta[order], ref, rtol=1e-5, atol=1e-12)
    assert np.all(delta >= 0)


def test_workflow_reports_pev_including_vc_uncertainty(tmp_path):
    out = run_evaluation(ROOT / "examples" / "02_sheep_wwt_reml" / "analysis.toml",
                         tmp_path / "o", console=False)
    with open(out.out_dir / "ebv_wwt.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    pev = np.array([float(r["pev"]) for r in rows])
    pev_t = np.array([float(r["pev_incl_vc_uncertainty"]) for r in rows])
    rel_t = np.array([float(r["reliability_incl_vc_uncertainty"]) for r in rows])
    assert np.all(pev_t >= pev - 1e-12) and np.any(pev_t > pev)
    assert np.all((rel_t >= 0) & (rel_t <= 1))
    # known variances: no extra columns
    out2 = run_evaluation(ROOT / "examples" / "01_textbook_mrode_3_1" / "analysis.toml",
                          tmp_path / "p", console=False)
    with open(out2.out_dir / "ebv_wwg.csv", encoding="utf-8") as fh:
        assert "pev_incl_vc_uncertainty" not in fh.readline()


def test_multitrait_delta_equals_v_form_derivatives():
    from abp.solvers.vc_uncertainty import kackar_harville_delta_multitrait
    from tests.test_multitrait_reml import _problem
    ped, data, A = _problem(3, n_anim=40, n_rec=70)
    G0 = np.array([[1.6, 0.4], [0.4, 1.1]])
    R0 = np.array([[2.5, 0.6], [0.6, 2.0]])
    rng = np.random.default_rng(1)
    M = rng.standard_normal((6, 6))
    cov = 0.01 * (M @ M.T + 6 * np.eye(6))
    delta = kackar_harville_delta_multitrait(data, ped.ainv(), 1 + ped.inbreeding(), G0, R0,
                                             cov, method="dense")
    Y = data.Y
    ri, ti = np.nonzero(~np.isnan(Y))
    y = Y[ri, ti]
    n, q, t = y.size, A.shape[0], 2
    Z = np.zeros((n, q * t))
    Z[np.arange(n), data.animal_col[ri] * t + ti] = 1
    X = np.zeros((n, 6))
    for j, Xj in enumerate(data.X_per_trait):
        X[np.flatnonzero(ti == j), 3 * j:3 * j + 3] = Xj.toarray()

    def u_hat(Gm, Rm):
        R = np.zeros((n, n))
        for r in np.unique(ri):
            idx = np.flatnonzero(ri == r)
            R[np.ix_(idx, idx)] = Rm[np.ix_(ti[idx], ti[idx])]
        Gu = np.kron(A, Gm)
        Vi = np.linalg.inv(Z @ Gu @ Z.T + R)
        P = Vi - Vi @ X @ np.linalg.solve(X.T @ Vi @ X, X.T @ Vi)
        return (Gu @ Z.T @ P @ y).reshape(q, t)
    J = []
    h = 1e-5
    for which in "GR":
        for j, k in [(0, 0), (0, 1), (1, 1)]:
            E = np.zeros((2, 2))
            E[j, k] = E[k, j] = h
            if which == "G":
                J.append((u_hat(G0 + E, R0) - u_hat(G0 - E, R0)) / (2 * h))
            else:
                J.append((u_hat(G0, R0 + E) - u_hat(G0, R0 - E)) / (2 * h))
    J = np.stack(J, axis=2)
    ref = np.einsum("iak,kl,ibl->iab", J, cov, J)
    np.testing.assert_allclose(delta, ref, rtol=1e-4, atol=1e-10)


def test_example13_reports_multitrait_pev_including_vc_uncertainty(tmp_path):
    out = run_evaluation(ROOT / "examples" / "13_sheep_multitrait_reml" / "analysis.toml",
                         tmp_path / "o", console=False)
    with open(out.out_dir / "ebv_multitrait.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for tr in ("wwt", "fat", "fec"):
        sep = np.array([float(r[f"sep_{tr}"]) for r in rows])
        pev_t = np.array([float(r[f"pev_incl_vc_uncertainty_{tr}"]) for r in rows])
        rel_t = np.array([float(r[f"reliability_incl_vc_uncertainty_{tr}"]) for r in rows])
        assert np.all(pev_t >= sep ** 2 - 1e-9) and np.any(pev_t > sep ** 2)
        assert np.all((rel_t >= 0) & (rel_t <= 1))


def test_reduced_rank_delta_equals_v_form_derivatives():
    from abp.solvers.multitrait_reml import _rr_unpack
    from abp.solvers.vc_uncertainty import kackar_harville_delta_reduced_rank
    from tests.test_multitrait_reml import _problem
    ped, data, A = _problem(3, n_anim=40, n_rec=70)
    x = np.array([1.2, 0.5, 0.9, 0.3, 0.6])          # Lambda (2 x 1), chol R0 (log diagonal)
    rng = np.random.default_rng(2)
    M = rng.standard_normal((5, 5))
    cov = 0.01 * (M @ M.T + 5 * np.eye(5))
    delta = kackar_harville_delta_reduced_rank(data, ped.ainv(), 1 + ped.inbreeding(), x, 2, 1,
                                               cov, method="dense")
    Y = data.Y
    ri, ti = np.nonzero(~np.isnan(Y))
    y = Y[ri, ti]
    n, q, t = y.size, A.shape[0], 2
    Z = np.zeros((n, q * t))
    Z[np.arange(n), data.animal_col[ri] * t + ti] = 1
    X = np.zeros((n, 6))
    for j, Xj in enumerate(data.X_per_trait):
        X[np.flatnonzero(ti == j), 3 * j:3 * j + 3] = Xj.toarray()

    def u_hat(xx):
        Lam, Rm = _rr_unpack(xx, 2, 1)
        R = np.zeros((n, n))
        for r in np.unique(ri):
            idx = np.flatnonzero(ri == r)
            R[np.ix_(idx, idx)] = Rm[np.ix_(ti[idx], ti[idx])]
        Gu = np.kron(A, Lam @ Lam.T)
        Vi = np.linalg.inv(Z @ Gu @ Z.T + R)
        P = Vi - Vi @ X @ np.linalg.solve(X.T @ Vi @ X, X.T @ Vi)
        return (Gu @ Z.T @ P @ y).reshape(q, t)
    h = 1e-6
    J = np.stack([(u_hat(x + h * e) - u_hat(x - h * e)) / (2 * h) for e in np.eye(5)], axis=2)
    ref = np.einsum("iak,kl,ibl->iab", J, cov, J)
    np.testing.assert_allclose(delta, ref, rtol=1e-4, atol=1e-10)
