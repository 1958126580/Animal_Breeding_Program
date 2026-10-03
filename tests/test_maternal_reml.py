"""REML for the maternal animal model (round 13): Henderson-form likelihood, scores and
average information against the marginal (V) form with dense algebra, the optimum
against a generic optimizer of the V-form likelihood, sparse against dense traces, BLUP
and PEV against the V-form predictor, and refusals."""

import math

import numpy as np
import pytest
import scipy.optimize as so
import scipy.sparse as sp

from abp.core.pedigree import Pedigree
from abp.errors import ABPError
from abp.solvers.maternal_reml import (MaternalData, MaternalREMLEvaluator, maternal_blup,
                                       maternal_design, maternal_reml_fit)

G0_TRUE = np.array([[4.0, -1.0], [-1.0, 2.0]])


def _problem(n=90, seed=3, n_groups=3, pe=True):
    """Random sex-consistent pedigree; one record per non-founder (dam known for 90%,
    drawn from the most recent n/10 earlier females, preferably ones with records, so that
    each dam has several offspring); intercept + a 3-level factor; maternal pe on the
    dam."""
    rng = np.random.default_rng(seed)
    ids = [f"a{i}" for i in range(n)]
    male = np.arange(n) % 2 == 0
    nf = n // 6
    sires, dams = [], []
    for i in range(n):
        if i < nf:
            sires.append(None)
            dams.append(None)
        else:
            sires.append(ids[rng.choice(np.flatnonzero(male[:i]))])
            # dams: the most recent n/10 earlier females (own records when possible), so that
            # each dam has several offspring and dams are related to their offspring's dams
            fem = np.flatnonzero(~male[:i])
            rec_f = fem[fem >= nf]
            pool = (rec_f if rec_f.size >= 4 else fem)[-max(4, n // 10):]
            dams.append(None if rng.random() < 0.1 else ids[rng.choice(pool)])
    ped = Pedigree.from_parent_ids(ids, sires, dams)
    A = np.linalg.inv(ped.ainv().toarray())
    L = np.linalg.cholesky(np.kron(A, G0_TRUE))
    am = (L @ rng.standard_normal(2 * n)).reshape(n, 2)
    dam = np.asarray(ped.dam, dtype=np.int64)
    rec = np.arange(nf, n)
    grp = rng.integers(0, n_groups, rec.size)
    X = sp.csr_matrix(np.column_stack([np.ones(rec.size)]
                                      + [(grp == g).astype(float) for g in range(1, n_groups)]))
    damr = dam[rec]
    y = 20.0 + 1.5 * grp + am[rec, 0] + np.where(damr >= 0, am[np.maximum(damr, 0), 1], 0.0)
    iid = []
    if pe:
        known = damr >= 0
        lev = np.where(known, damr, -1)
        levels, inv = np.unique(lev, return_inverse=True)
        # records with an unknown dam get their own level (a pe effect that is not shared)
        c = rng.normal(0, math.sqrt(1.5), levels.size)
        y = y + np.where(known, c[inv], 0.0)
        iid = [("mpe", inv, levels.size)]
    y = y + rng.normal(0, math.sqrt(6.0), rec.size)
    data = MaternalData(y, X, rec, damr, iid)
    return data, ped


def _vform(data, ped, theta):
    """-2 logL (constant dropped), V, P and the derivative matrices dV/dtheta, densely."""
    q = ped.ainv().shape[0]
    A = np.linalg.inv(ped.ainv().toarray())
    Zg, Zp = maternal_design(data, q)
    Zg = Zg.toarray()
    G0 = np.array([[theta[0], theta[1]], [theta[1], theta[2]]])
    n_iid = len(Zp)
    vp, se = theta[3:3 + n_iid], theta[3 + n_iid]
    n = data.y.size
    dV = []
    for j, k in ((0, 0), (0, 1), (1, 1)):
        E = np.zeros((2, 2))
        E[j, k] = E[k, j] = 1.0
        dV.append(Zg @ np.kron(A, E) @ Zg.T)
    V = Zg @ np.kron(A, G0) @ Zg.T + se * np.eye(n)
    for Z, v in zip(Zp, vp):
        Zd = Z.toarray()
        V += v * Zd @ Zd.T
        dV.append(Zd @ Zd.T)
    dV.append(np.eye(n))
    X = data.X.toarray()
    Vi = np.linalg.inv(V)
    XVX = X.T @ Vi @ X
    P = Vi - Vi @ X @ np.linalg.solve(XVX, X.T @ Vi)
    m2 = np.linalg.slogdet(V)[1] + np.linalg.slogdet(XVX)[1] + float(data.y @ P @ data.y)
    return m2, V, P, dV, Zg, A, G0


THETA = np.array([3.5, -0.8, 2.2, 1.2, 6.5])


@pytest.mark.parametrize("dense", [True, False])
def test_likelihood_score_and_ai_equal_the_marginal_form(dense):
    data, ped = _problem()
    ev = MaternalREMLEvaluator(data, ped.ainv(), ped.logdet_a(), dense=dense)
    pt = ev.evaluate(THETA)
    m2, V, P, dV, *_ = _vform(data, ped, THETA)
    # log|C| + q log|G0| + 2 log|A| + ... = log|V| + log|X'V^-1X| up to the same constant
    # (Henderson's identity); the constants agree because both drop (n - p) log 2 pi
    assert -2.0 * pt.loglik == pytest.approx(m2, rel=1e-10, abs=1e-8)
    Py = P @ data.y
    score = np.array([0.5 * (Py @ D @ Py - np.trace(P @ D)) for D in dV])
    np.testing.assert_allclose(pt.score, score, rtol=1e-8, atol=1e-9)
    ai = np.array([[0.5 * Py @ Di @ P @ Dj @ Py for Dj in dV] for Di in dV])
    np.testing.assert_allclose(pt.ai, ai, rtol=1e-8, atol=1e-10)


def test_score_equals_finite_differences_of_the_likelihood():
    data, ped = _problem(seed=5)
    ev = MaternalREMLEvaluator(data, ped.ainv(), ped.logdet_a())
    pt = ev.evaluate(THETA, with_ai=False)
    h = 1e-5
    for i in range(THETA.size):
        d = np.zeros_like(THETA)
        d[i] = h
        num = (ev.evaluate(THETA + d, False).loglik - ev.evaluate(THETA - d, False).loglik) / (2 * h)
        assert pt.score[i] == pytest.approx(num, rel=1e-5, abs=1e-7)


def test_sparse_traces_equal_dense_traces():
    data, ped = _problem(seed=7)
    d = MaternalREMLEvaluator(data, ped.ainv(), ped.logdet_a(), dense=True)
    s = MaternalREMLEvaluator(data, ped.ainv(), ped.logdet_a(), dense=False)
    for th in (THETA, np.array([3.0, 0.0, 1.0, 0.5, 7.0])):   # also a zero covariance
        a, b = d.evaluate(th), s.evaluate(th)
        assert a.loglik == pytest.approx(b.loglik, rel=1e-12)
        np.testing.assert_allclose(a.score, b.score, rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(a.em, b.em, rtol=1e-10)
        np.testing.assert_allclose(a.ai, b.ai, rtol=1e-9)


def test_reml_optimum_equals_a_generic_optimizer_of_the_marginal_likelihood():
    data, ped = _problem(n=200, seed=21)          # an interior optimum
    fit = maternal_reml_fit(data, ped.ainv(), ped.logdet_a(), {"tol": 1e-10, "max_iter": 300})
    assert fit.status == "converged"

    def f(x):                     # unconstrained: chol(G0) with log diagonal, log variances
        Lg = np.array([[math.exp(x[0]), 0.0], [x[1], math.exp(x[2])]])
        G = Lg @ Lg.T
        th = np.array([G[0, 0], G[0, 1], G[1, 1], math.exp(x[3]), math.exp(x[4])])
        return _vform(data, ped, th)[0]
    Lf = np.linalg.cholesky(fit.G0)
    x0 = np.array([math.log(Lf[0, 0]) + 0.2, Lf[1, 0] - 0.3, math.log(Lf[1, 1]) - 0.2,
                   math.log(fit.iid["mpe"]) + 0.3, math.log(fit.residual) - 0.2])
    res = so.minimize(f, x0, method="Nelder-Mead",
                      options={"xatol": 1e-9, "fatol": 1e-11, "maxiter": 20000, "maxfev": 40000})
    assert -2.0 * fit.loglik <= res.fun + 1e-7            # REML optimum at least as good
    Lg = np.array([[math.exp(res.x[0]), 0.0], [res.x[1], math.exp(res.x[2])]])
    np.testing.assert_allclose(fit.G0, Lg @ Lg.T, rtol=2e-3, atol=2e-3)
    assert fit.iid["mpe"] == pytest.approx(math.exp(res.x[3]), rel=5e-3, abs=2e-3)
    assert fit.residual == pytest.approx(math.exp(res.x[4]), rel=2e-3)
    # standard errors and derived quantities are reported
    assert set(fit.se) == {"direct", "direct-maternal covariance", "maternal", "mpe", "residual"}
    for k in ("h2", "m2", "direct_maternal_correlation"):
        assert k in fit.derived and fit.derived[k + "_se"] > 0


def test_blup_and_pev_equal_the_marginal_form():
    data, ped = _problem(seed=13)
    th = THETA
    beta, U, pev, (c,), info = maternal_blup(data, ped.ainv(), np.array([[th[0], th[1]], [th[1], th[2]]]),
                                       [th[3]], th[4])
    m2, V, P, dV, Zg, A, G0 = _vform(data, ped, th)
    G = np.kron(A, G0)
    X = data.X.toarray()
    Vi = np.linalg.inv(V)
    b = np.linalg.solve(X.T @ Vi @ X, X.T @ Vi @ data.y)
    u = G @ Zg.T @ Vi @ (data.y - X @ b)
    assert info["relative_residual"] < 1e-10
    np.testing.assert_allclose(beta, b, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(U.ravel(), u, rtol=1e-8, atol=1e-9)
    PEV = G - G @ Zg.T @ P @ Zg @ G                     # prediction-error covariance
    q = U.shape[0]
    for j in range(2):
        for k in range(2):
            idx_j = np.arange(q) * 2 + j
            idx_k = np.arange(q) * 2 + k
            np.testing.assert_allclose(pev[:, j, k], PEV[idx_j, idx_k], rtol=1e-8, atol=1e-10)


def test_out_of_space_start_and_bad_indices_are_refused():
    data, ped = _problem(seed=17)
    with pytest.raises(ABPError) as e:
        maternal_reml_fit(data, ped.ainv(), ped.logdet_a(), {"tol": 1e-8},
                          start=np.array([1.0, 2.0, 1.0, 1.0, 1.0]))     # |r| > 1
    assert e.value.code == "ABP-E302" or "start" in str(e.value)
    bad = MaternalData(data.y, data.X, data.animal, np.full(data.y.size, 10_000), data.iid)
    with pytest.raises(ABPError):
        MaternalREMLEvaluator(bad, ped.ainv(), ped.logdet_a())


def test_boundary_of_the_maternal_pe_variance_satisfies_kuhn_tucker():
    """A data set whose optimum has a zero maternal-pe variance: the term is fixed at 0,
    the score at zero (Henderson form) equals the marginal-form derivative and is <= 0,
    and no point with s_p >= 0 has a higher likelihood (generic optimizer)."""
    from abp.solvers.maternal_reml import MaternalREMLEvaluator as Ev
    from abp.solvers.maternal_reml import _score_at_zero
    data, ped = _problem(n=200, seed=12)
    fit = maternal_reml_fit(data, ped.ainv(), ped.logdet_a(), {"tol": 1e-10, "max_iter": 300})
    assert fit.status == "converged_boundary" and fit.iid["mpe"] == 0.0
    assert fit.se is None                                   # withheld at a boundary
    th = np.array([fit.G0[0, 0], fit.G0[0, 1], fit.G0[1, 1], 0.0, fit.residual])
    m2, V, P, dV, *_ = _vform(data, ped, th)
    Py = P @ data.y
    g_v = 0.5 * (Py @ dV[3] @ Py - np.trace(P @ dV[3]))
    sub = MaternalData(data.y, data.X, data.animal, data.dam, [])
    ev = Ev(sub, ped.ainv(), ped.logdet_a())
    Zd = maternal_design(data, ped.ainv().shape[0])[1][0]
    g_h = _score_at_zero(ev, np.delete(th, 3), Zd)
    assert g_h == pytest.approx(g_v, rel=1e-8, abs=1e-10)
    assert g_v <= 0.0
    assert -2.0 * fit.loglik == pytest.approx(m2, rel=1e-10)

    def f(x):
        Lg = np.array([[math.exp(x[0]), 0.0], [x[1], math.exp(x[2])]])
        G = Lg @ Lg.T
        return _vform(data, ped, np.array([G[0, 0], G[0, 1], G[1, 1], x[3] ** 2,
                                           math.exp(x[4])]))[0]
    Lf = np.linalg.cholesky(fit.G0)
    best = min(so.minimize(f, np.array([math.log(Lf[0, 0]), Lf[1, 0], math.log(Lf[1, 1]), x3,
                                        math.log(fit.residual)]), method="Nelder-Mead",
                           options={"xatol": 1e-9, "fatol": 1e-11, "maxiter": 20000}).fun
               for x3 in (0.3, 1.0))
    assert -2.0 * fit.loglik <= best + 1e-6


def test_singular_genetic_covariance_is_refused():
    """A small data set whose likelihood increases towards |r_am| = 1."""
    data, ped = _problem(n=200, seed=18)
    with pytest.raises(ABPError) as e:
        maternal_reml_fit(data, ped.ainv(), ped.logdet_a(), {"tol": 1e-8, "max_iter": 300})
    assert e.value.code == "ABP-E300"


def test_maternal_variances_are_recovered_on_average():
    """Mean REML estimates over the converged fits of 12 replicates (600 animals) within
    3 MC SE of the simulated values; at most 2 replicates refused (singular G0)."""
    est, refused = [], 0
    for seed in range(100, 112):
        data, ped = _problem(n=600, seed=seed)
        try:
            fit = maternal_reml_fit(data, ped.ainv(), ped.logdet_a(),
                                    {"tol": 1e-8, "max_iter": 300})
        except ABPError as exc:
            assert exc.code == "ABP-E300"
            refused += 1
            continue
        est.append([fit.G0[0, 0], fit.G0[0, 1], fit.G0[1, 1], fit.iid["mpe"], fit.residual])
    assert refused <= 2
    est = np.array(est)
    truth = np.array([4.0, -1.0, 2.0, 1.5, 6.0])
    se = est.std(axis=0, ddof=1) / math.sqrt(len(est))
    assert np.all(np.abs(est.mean(axis=0) - truth) <= 3 * se + 1e-9), (est.mean(axis=0), se)
