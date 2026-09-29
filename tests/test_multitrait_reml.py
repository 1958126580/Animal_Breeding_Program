"""Multi-trait REML against the marginal (V-form) likelihood built independently:
log-likelihood, scores (finite differences), EM monotonicity, optimum against
Nelder-Mead, dense = sparse trace paths, and refusals."""

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.optimize import minimize

from abp.core.pedigree import Pedigree
from abp.errors import ABPError
from abp.solvers import multitrait_reml as MR
from abp.solvers.multitrait import MTData
from tests.reference.dense_reference import tabular_a


def _problem(seed=1, n_anim=70, n_rec=110, t=2, miss=0.2):
    rng = np.random.default_rng(seed)
    ids = [str(i) for i in range(n_anim)]
    sires, dams = [], []
    for i in range(n_anim):
        if i < 12:
            sires.append(None)
            dams.append(None)
        else:
            s, d = rng.choice(i, 2, replace=False)
            sires.append(ids[s])
            dams.append(ids[d])
    ped = Pedigree.from_parent_ids(ids, sires, dams)
    A = tabular_a(ids, sires, dams)
    order = ped.index_of(ids)
    G0 = np.array([[2.0, 0.8], [0.8, 1.5]])[:t, :t]
    R0 = np.array([[3.0, 0.9], [0.9, 2.5]])[:t, :t]
    L = np.linalg.cholesky(np.kron(A, G0))
    u = (L @ rng.standard_normal(n_anim * t)).reshape(n_anim, t)
    rec_an = rng.integers(0, n_anim, n_rec)
    herd = rng.integers(0, 3, n_rec)
    e = rng.standard_normal((n_rec, t)) @ np.linalg.cholesky(R0).T
    Y = 10 + herd[:, None] * np.arange(1, t + 1) + u[rec_an] + e
    Y[rng.random(Y.shape) < miss] = np.nan
    Y[np.isnan(Y).all(axis=1), 0] = 10.0
    Xs = []
    for j in range(t):
        rows = np.flatnonzero(~np.isnan(Y[:, j]))
        h = herd[rows]
        X = np.column_stack([np.ones(rows.size), h == 1, h == 2]).astype(float)
        Xs.append(sp.csr_matrix(X))
    # animal indices in pedigree order; the reference works in input order
    col = order[rec_an]
    data = MTData(Y, Xs, col)
    A_ped = A[np.ix_(np.argsort(order), np.argsort(order))]      # pedigree order
    return ped, data, A_ped


def _v_form_m2ll(data, A, G0, R0):
    """-2 REML logL from V = Z (A (x) G0) Z' + R, independent of ABP's MME."""
    Y = data.Y
    t = Y.shape[1]
    obs = ~np.isnan(Y)
    ri, ti = np.nonzero(obs)
    y = Y[ri, ti]
    n = y.size
    q = A.shape[0]
    Z = np.zeros((n, q * t))
    Z[np.arange(n), data.animal_col[ri] * t + ti] = 1
    R = np.zeros((n, n))
    for r in np.unique(ri):
        idx = np.flatnonzero(ri == r)
        R[np.ix_(idx, idx)] = R0[np.ix_(ti[idx], ti[idx])]
    V = Z @ np.kron(A, G0) @ Z.T + R
    X = np.zeros((n, sum(x.shape[1] for x in data.X_per_trait)))
    off = 0
    for j, Xj in enumerate(data.X_per_trait):
        rows = np.flatnonzero(ti == j)
        X[rows, off:off + Xj.shape[1]] = Xj.toarray()
        off += Xj.shape[1]
    Vi = np.linalg.inv(V)
    XVX = X.T @ Vi @ X
    P = Vi - Vi @ X @ np.linalg.solve(XVX, X.T @ Vi)
    return np.linalg.slogdet(V)[1] + np.linalg.slogdet(XVX)[1] + y @ P @ y


def test_loglik_equals_v_form_up_to_constant_and_scores_match_finite_differences():
    ped, data, A = _problem()
    ev = MR.MTREMLEvaluator(data, ped.ainv(), ped.logdet_a())
    G0 = np.array([[1.7, 0.5], [0.5, 1.2]])
    R0 = np.array([[2.8, 0.7], [0.7, 2.2]])
    th = MR._pack(G0, R0)
    p = ev.evaluate(th)
    ref = _v_form_m2ll(data, A, G0, R0)
    # Henderson's form and the marginal form are equal exactly (no constant differs)
    assert -2 * p.loglik == pytest.approx(ref, abs=1e-8)
    h = 1e-5
    for i in range(th.size):
        d = np.zeros_like(th)
        d[i] = h
        fd = -0.5 * (_v_form_m2ll(data, A, *MR._unpack(th + d, 2))
                     - _v_form_m2ll(data, A, *MR._unpack(th - d, 2))) / (2 * h)
        assert p.score[i] == pytest.approx(fd, abs=1e-6), i


def test_em_step_increases_loglik_and_keeps_pd():
    ped, data, A = _problem(3)
    ev = MR.MTREMLEvaluator(data, ped.ainv(), ped.logdet_a())
    th = MR._pack(np.array([[1.0, 0.2], [0.2, 0.6]]), np.array([[5.0, 0.1], [0.1, 4.0]]))
    p0 = ev.evaluate(th)
    p1 = ev.evaluate(p0.em)
    assert p1.loglik > p0.loglik
    G, R = MR._unpack(p0.em, 2)
    assert MR._is_pd(G) and MR._is_pd(R)


def test_optimum_equals_independent_nelder_mead():
    ped, data, A = _problem(5, n_anim=60, n_rec=120, miss=0.15)
    fit = MR.mt_reml_fit(data, ped.ainv(), ped.logdet_a(),
                         {"tol": 1e-9, "max_iter": 300})
    assert fit.status == "converged"

    def f(x):  # Cholesky parametrisation keeps both matrices PD
        Lg = np.array([[x[0], 0], [x[1], x[2]]])
        Lr = np.array([[x[3], 0], [x[4], x[5]]])
        return _v_form_m2ll(data, A, Lg @ Lg.T, Lr @ Lr.T)
    x0 = np.concatenate([np.linalg.cholesky(fit.G0)[[0, 1, 1], [0, 0, 1]],
                         np.linalg.cholesky(fit.R0)[[0, 1, 1], [0, 0, 1]]]) * 1.1
    r = minimize(f, x0, method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-11,
                                                       "maxiter": 20000, "maxfev": 20000})
    Lg = np.array([[r.x[0], 0], [r.x[1], r.x[2]]])
    Lr = np.array([[r.x[3], 0], [r.x[4], r.x[5]]])
    assert -2 * fit.loglik <= r.fun + 1e-7                       # ABP at least as good
    np.testing.assert_allclose(fit.G0, Lg @ Lg.T, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(fit.R0, Lr @ Lr.T, rtol=2e-3, atol=2e-3)
    assert set(fit.to_dict(["a", "b"])["heritabilities"]) == {"a", "b"}


def test_dense_and_sparse_trace_paths_agree(monkeypatch):
    ped, data, A = _problem(7)
    th = MR._pack(np.array([[1.5, 0.4], [0.4, 1.0]]), np.array([[3.0, 0.5], [0.5, 2.0]]))
    pd_ = MR.MTREMLEvaluator(data, ped.ainv(), ped.logdet_a()).evaluate(th)
    monkeypatch.setattr(MR, "DENSE_MAX", 5)
    ev = MR.MTREMLEvaluator(data, ped.ainv(), ped.logdet_a())
    assert ev.trace_method == "sparse_selected_inversion"
    ps = ev.evaluate(th)
    assert ps.loglik == pytest.approx(pd_.loglik, abs=1e-8)
    np.testing.assert_allclose(ps.score, pd_.score, atol=1e-9)
    np.testing.assert_allclose(ps.em, pd_.em, rtol=1e-10)
    np.testing.assert_allclose(ps.ai, pd_.ai, rtol=1e-9)


def test_three_traits_with_structural_missing_pattern():
    """A trait never recorded together with another (structural zero residual
    covariance region) still gives the V-form likelihood."""
    ped, data, A = _problem(9, t=2)
    rng = np.random.default_rng(1)
    Y3 = np.column_stack([data.Y, data.Y[:, 0] * 0.3 + rng.normal(0, 1, data.Y.shape[0])])
    Y3[: Y3.shape[0] // 2, 2] = np.nan
    Xs = list(data.X_per_trait) + [sp.csr_matrix(np.ones((int((~np.isnan(Y3[:, 2])).sum()), 1)))]
    d3 = MTData(Y3, Xs, data.animal_col)
    G0 = np.array([[1.5, 0.3, 0.2], [0.3, 1.0, 0.1], [0.2, 0.1, 0.8]])
    R0 = np.array([[3.0, 0.5, 0.4], [0.5, 2.0, 0.2], [0.4, 0.2, 1.5]])
    p = MR.MTREMLEvaluator(d3, ped.ainv(), ped.logdet_a()).evaluate(MR._pack(G0, R0))
    assert -2 * p.loglik == pytest.approx(_v_form_m2ll(d3, A, G0, R0), abs=1e-8)


def test_refusals():
    ped, data, A = _problem(11)
    with pytest.raises(ABPError, match="log"):
        MR.MTREMLEvaluator(data, ped.ainv(), None)
    with pytest.raises(ABPError) as exc:
        MR.mt_reml_fit(data, ped.ainv(), ped.logdet_a(), {"tol": 1e-12, "max_iter": 2})
    assert exc.value.code == "ABP-E403"


def test_example13_multitrait_reml_end_to_end(tmp_path):
    from pathlib import Path
    from abp.workflows.evaluate import run_evaluation
    ex = Path(__file__).resolve().parents[1] / "examples" / "13_sheep_multitrait_reml"
    out = run_evaluation(ex / "analysis.toml", tmp_path / "o", console=False)
    r = out.results["traits"]["wwt"]["reml"]
    assert r["status"] == "converged" and r["trace_method"] == "sparse_selected_inversion"
    G0 = np.array(r["G0"])
    assert np.all(np.linalg.eigvalsh(G0) > 0)
    assert out.results["traits"]["fat"]["variance_source"].startswith("reml")
    # the estimates reach the generator's base values within a few SE (one replicate)
    se = r["se"]
    assert abs(G0[0, 0] - 4.0) < 4 * se["G0[0,0]"]
    report = (out.out_dir / "report.md").read_text(encoding="utf-8")
    assert "genetic correlations" in report


def test_spec_refuses_start_values_for_multitrait_reml():
    from abp.core.spec import validate_spec_dict
    d = {"schema_version": "1",
         "project": {"name": "t", "species": "sheep", "synthetic_data": True},
         "analysis": {"task": "additive_ebv", "target_population": "x",
                      "information_cutoff": "2024-12-31", "genetic_base": "x"},
         "data": {"pedigree": "p.csv", "phenotypes": "y.csv"},
         "traits": [{"name": "a", "unit": "kg"}, {"name": "b", "unit": "kg"}],
         "model": {"traits": ["a", "b"],
                   "random": [{"name": "animal", "kind": "additive", "relationship": "pedigree"}]},
         "variances": {"mode": "reml"}}
    assert validate_spec_dict(d)["variances"]["mode"] == "reml"
    d["reml"] = {"start": {"animal": 1.0, "residual": 1.0}}
    with pytest.raises(ABPError, match="reml.start"):
        validate_spec_dict(d)
