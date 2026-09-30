"""Multi-trait threshold + linear Gibbs sampler against independent references:
the exact conditional of the location block (dense algebra from the model
definition), the R0 step against quadrature and closed-form moments, the G0 step
against an independent random-walk Metropolis sampler, the posterior with fixed
covariances against importance sampling of the closed-form posterior, and the
reduction to the single-trait threshold sampler and to BLUP when the traits are
uncorrelated."""

import math

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.special import log_ndtr

from abp.core.pedigree import Pedigree
from abp.errors import ABPError
from abp.solvers import mt_threshold_gibbs as MT
from abp.solvers.cholesky import SparseLDL


def _ped(n, seed, founders):
    rng = np.random.default_rng(seed)
    ids = [f"a{i}" for i in range(n)]
    male = np.arange(n) % 2 == 0
    s, d = [], []
    for i in range(n):
        if i < founders:
            s.append(None)
            d.append(None)
        else:
            s.append(ids[rng.choice(np.flatnonzero(male[:i]))])
            d.append(ids[rng.choice(np.flatnonzero(~male[:i]))])
    return Pedigree.from_parent_ids(ids, s, d), rng


def _dense_c(Y, X, Kinv, R0, G0):
    """C and W'R^-1 from the model definition (independent of the module): unknowns
    (beta_1 .. beta_t, u animal-major), observations record-major over the observed
    (record, trait) pairs, residual block R0[o, o]^-1 per record; X shared by the
    traits, one record per animal (record r = animal r)."""
    X = np.asarray(X.toarray() if sp.issparse(X) else X)
    n, t = Y.shape
    p, q = X.shape[1], Kinv.shape[0]
    obs = [(r, j) for r in range(n) for j in range(t) if not np.isnan(Y[r, j])]
    W = np.zeros((len(obs), p * t + q * t))
    for i, (r, j) in enumerate(obs):
        W[i, j * p:(j + 1) * p] = X[r]
        W[i, p * t + r * t + j] = 1.0
    Rinv = np.zeros((len(obs), len(obs)))
    for r in range(n):
        idx = [i for i, (rr, _) in enumerate(obs) if rr == r]
        tr = [obs[i][1] for i in idx]
        Rinv[np.ix_(idx, idx)] = np.linalg.inv(R0[np.ix_(tr, tr)])
    prior = np.zeros((W.shape[1], W.shape[1]))
    prior[p * t:, p * t:] = np.kron(Kinv, np.linalg.inv(G0))
    return W.T @ Rinv @ W + prior, W.T @ Rinv


def test_location_draw_has_the_exact_conditional_mean_and_covariance():
    ped, rng = _ped(10, 1, 4)
    n = ped.n
    Y = np.column_stack([rng.integers(1, 3, n).astype(float), rng.normal(5, 2, n)])
    Y[[2, 7], 1] = np.nan                                       # two missing values
    Y[4, 0] = np.nan
    X = sp.csr_matrix(np.column_stack([np.ones(n), np.arange(n) % 2]))
    P = MT.MTProblem(Y, 0, MT.split_design(X, Y), np.arange(n), ped.ainv())
    G0 = np.array([[0.4, 0.3], [0.3, 1.5]])
    R0 = np.array([[1.0, 0.5], [0.5, 3.0]])
    yobs = rng.normal(0, 1, P.n_obs)
    C, WR = _dense_c(Y, X, ped.ainv().toarray(), R0, G0)
    mean = np.linalg.solve(C, WR @ yobs)
    cov = np.linalg.inv(C)
    fac = SparseLDL(P.coefficient(P.rinv(R0), np.linalg.inv(G0)))
    D = np.array([MT.draw_location(P, fac, yobs, R0, G0, rng) for _ in range(40000)])
    se_m = np.sqrt(np.diag(cov) / D.shape[0])
    assert np.max(np.abs(D.mean(axis=0) - mean) / se_m) < 5.0
    emp = np.cov(D.T)
    se_c = np.sqrt((np.outer(np.diag(cov), np.diag(cov)) + cov ** 2) / D.shape[0])
    z = (emp - cov) / se_c
    # the z-scores of covariance entries are correlated with one another (entries share
    # variables), so their pooled SD can lie well below 1; the bound on max |z| is the test
    assert np.abs(z).max() < 5.5 and 0.6 < z.std() < 1.3


def test_r0_step_matches_quadrature_and_closed_form_moments():
    rng = np.random.default_rng(3)
    # two traits: E[b] and E[S + b^2] by quadrature of p(b, S | e) with flat priors
    n = 25
    E = rng.standard_normal((n, 2)) @ np.linalg.cholesky([[1.0, 0.6], [0.6, 2.0]]).T
    ec, eo = E[:, 0], E[:, 1]
    bs = np.linspace(-1.5, 2.5, 801)
    Ss = np.linspace(0.05, 8.0, 1600)
    Bg, Sg = np.meshgrid(bs, Ss, indexing="ij")
    rss = (eo @ eo) - 2 * Bg * (eo @ ec) + Bg ** 2 * (ec @ ec)
    logp = -0.5 * n * np.log(Sg) - 0.5 * rss / Sg
    w = np.exp(logp - logp.max())
    w /= w.sum()
    ref_b, ref_r11 = float((w * Bg).sum()), float((w * (Sg + Bg ** 2)).sum())
    D = np.array([MT.draw_R0(E, 0, rng) for _ in range(60000)])
    for got, ref, k in ((D[:, 0, 1], ref_b, "R01"), (D[:, 1, 1], ref_r11, "R11")):
        assert abs(got.mean() - ref) < 4.0 * got.std() / math.sqrt(got.size), k
    assert np.all(D[:, 0, 0] == 1.0) and np.allclose(D[:, 0, 1], D[:, 1, 0])
    # three traits, categorical in the middle: E[b] = bhat, E[S] = Sres / (df - m - 1)
    n = 40
    E = rng.standard_normal((n, 3))
    c, o = 1, [0, 2]
    cc = E[:, c] @ E[:, c]
    bhat = E[:, o].T @ E[:, c] / cc
    Sres = E[:, o].T @ E[:, o] - np.outer(bhat, bhat) * cc
    D = np.array([MT.draw_R0(E, c, rng) for _ in range(40000)])
    b = D[:, o, c]
    S = D[:, 0, 0] - b[:, 0] ** 2
    assert np.all(np.abs(b.mean(axis=0) - bhat) < 4 * b.std(axis=0) / math.sqrt(b.shape[0]))
    ES = Sres[0, 0] / (n - 2 - 2 - 2 - 1)
    assert abs(S.mean() - ES) < 4 * S.std() / math.sqrt(S.size)


def test_g0_step_matches_an_independent_metropolis_sampler():
    rng = np.random.default_rng(5)
    q = 14
    U = rng.standard_normal((q, 2)) @ np.linalg.cholesky([[1.0, 0.4], [0.4, 0.5]]).T
    Kinv = np.eye(q)
    S = U.T @ U
    D = np.array([MT.draw_G0(U, Kinv, rng) for _ in range(40000)])

    def logp(g):                                  # flat prior on PD matrices
        G = np.array([[g[0], g[1]], [g[1], g[2]]])
        if g[0] <= 0 or g[2] <= 0 or g[0] * g[2] - g[1] ** 2 <= 0:
            return -np.inf
        return -0.5 * q * math.log(np.linalg.det(G)) - 0.5 * np.trace(np.linalg.solve(G, S))
    g = np.array([S[0, 0], S[0, 1], S[1, 1]]) / q
    lp = logp(g)
    step = np.array([0.35, 0.2, 0.18]) * g[[0, 0, 2]] ** 0.5 * 0.9
    out = []
    for i in range(300000):
        prop = g + step * rng.standard_normal(3)
        lq = logp(prop)
        if math.log(rng.random()) < lq - lp:
            g, lp = prop, lq
        if i >= 20000 and i % 5 == 0:
            out.append(g.copy())
    M = np.array(out)
    ref = M.mean(axis=0)
    got = np.array([D[:, 0, 0].mean(), D[:, 0, 1].mean(), D[:, 1, 1].mean()])
    # MH draws are autocorrelated: tolerance 3% of the value (relative MC error ~1%)
    assert np.all(np.abs(got - ref) < 0.03 * np.abs(ref) + 0.01), (got, ref)
    # and the closed form of the inverse Wishart mean S / (q - 2t - 2)
    np.testing.assert_allclose(got, np.array([S[0, 0], S[0, 1], S[1, 1]]) / (q - 6), rtol=0.03)
    # proper prior IW(nu, nu G_prior): MH on |G|^-(q + nu + t + 1)/2 exp(-tr((S + nu Gp) G^-1)/2)
    nu, Gp = 5.0, np.array([[0.8, 0.1], [0.1, 0.6]])
    D = np.array([MT.draw_G0(U, Kinv, rng, nu, Gp) for _ in range(40000)])
    S2 = S + nu * Gp

    def logp2(g):
        G = np.array([[g[0], g[1]], [g[1], g[2]]])
        if g[0] <= 0 or g[2] <= 0 or g[0] * g[2] - g[1] ** 2 <= 0:
            return -np.inf
        return (-0.5 * (q + nu + 3) * math.log(np.linalg.det(G))
                - 0.5 * np.trace(np.linalg.solve(G, S2)))
    g = np.array([S2[0, 0], S2[0, 1], S2[1, 1]]) / (q + nu)
    lp = logp2(g)
    out = []
    for i in range(300000):
        prop = g + step * rng.standard_normal(3)
        lq = logp2(prop)
        if math.log(rng.random()) < lq - lp:
            g, lp = prop, lq
        if i >= 20000 and i % 5 == 0:
            out.append(g.copy())
    ref = np.array(out).mean(axis=0)
    got = np.array([D[:, 0, 0].mean(), D[:, 0, 1].mean(), D[:, 1, 1].mean()])
    assert np.all(np.abs(got - ref) < 0.03 * np.abs(ref) + 0.01), (got, ref)


def _log_post(theta, Y, R0, G0, Kinv):
    """log p(theta | y) up to a constant, liabilities integrated out (binary trait
    first, one continuous trait, one intercept per trait, record r = animal r;
    flat prior on beta).  theta = (beta_1, beta_2, u animal-major)."""
    n = Y.shape[0]
    U = theta[2:].reshape(n, 2)
    eta = theta[:2][None, :] + U
    lp = -0.5 * np.trace(np.linalg.solve(G0, U.T @ Kinv @ U))
    ob_o = ~np.isnan(Y[:, 1])
    r = Y[:, 1] - eta[:, 1]
    lp += -0.5 * np.sum(r[ob_o] ** 2) / R0[1, 1]
    m = eta[:, 0].copy()
    s = np.ones(n)
    m[ob_o] += R0[0, 1] / R0[1, 1] * r[ob_o]
    s[ob_o] = math.sqrt(1.0 - R0[0, 1] ** 2 / R0[1, 1])
    sign = np.where(Y[:, 0] == 2, 1.0, -1.0)         # category 2: l > 0, category 1: l <= 0
    lp += np.sum(log_ndtr(sign * m / s))
    return lp


def test_posterior_with_fixed_covariances_equals_importance_sampling():
    ped, rng = _ped(8, 7, 3)
    n = ped.n
    Y = np.column_stack([np.array([1, 2, 2, 1, 2, 1, 2, 2], float),
                         np.array([3.1, 4.0, np.nan, 2.2, 5.3, 2.9, np.nan, 4.4])])
    X = sp.csr_matrix(np.ones((n, 1)))
    G0 = np.array([[0.5, 0.4], [0.4, 1.0]])
    R0 = np.array([[1.0, 0.5], [0.5, 1.2]])
    cfg = MT.MTThresholdGibbsConfig(chains=4, iterations=40000, burn_in=2000, thin=2,
                                    max_iterations=40000, seed=11, start_G0=G0, start_R0=R0,
                                    fix_covariances=True)
    res = MT.mt_threshold_gibbs(Y, 0, X, np.arange(n), ped.ainv(), cfg)
    Kinv = ped.ainv().toarray()
    # proposal: multivariate t around the sampler's moments (IS is consistent for any
    # proposal with heavier tails; the estimate does not rely on the sampler)
    theta_m = np.concatenate([np.concatenate(res.fixed_mean), res.ebv.ravel()])
    C, _ = _dense_c(Y, X, Kinv, R0, G0)
    V = 2.0 * np.linalg.inv(C + 1e-9 * np.eye(C.shape[0]))
    Lc = np.linalg.cholesky(V)
    N, df = 400000, 5
    zz = rng.standard_normal((N, theta_m.size))
    chi = rng.chisquare(df, N) / df
    T = theta_m + (zz @ Lc.T) / np.sqrt(chi)[:, None]
    logq = -0.5 * (df + theta_m.size) * np.log1p(np.sum(zz ** 2, axis=1) / chi / df)
    logp = np.array([_log_post(th, Y, R0, G0, Kinv) for th in T])
    lw = logp - logq
    w = np.exp(lw - lw.max())
    w /= w.sum()
    ess = 1.0 / np.sum(w ** 2)
    assert ess > 5000
    is_mean = w @ T
    is_var = w @ (T - is_mean) ** 2
    u_is = is_mean[2:].reshape(n, 2)
    sd_u = np.sqrt(is_var[2:].reshape(n, 2))
    # MC error of the sampler mean: sd / sqrt(effective draws); generous bound 0.05 sd
    assert np.max(np.abs(res.ebv - u_is) / sd_u) < 0.05, np.max(np.abs(res.ebv - u_is) / sd_u)
    np.testing.assert_allclose(res.pev, sd_u ** 2, rtol=0.06)


def test_uncorrelated_traits_reduce_to_single_trait_models():
    from abp.solvers.blup import RandomTerm, blup
    from abp.solvers.threshold_gibbs import ThresholdGibbsConfig, threshold_gibbs
    ped, rng = _ped(40, 13, 8)
    n = ped.n
    yc = rng.integers(1, 4, n).astype(float)                   # three categories
    yo = rng.normal(10, 2, n)
    Y = np.column_stack([yc, yo])
    X = sp.csr_matrix(np.ones((n, 1)))
    Z = sp.identity(n, format="csr")
    anim = np.arange(n)
    G0 = np.diag([0.4, 2.0])
    R0 = np.diag([1.0, 3.0])
    cfg = MT.MTThresholdGibbsConfig(chains=4, iterations=12000, burn_in=2000, thin=2,
                                    max_iterations=12000, seed=5, start_G0=G0, start_R0=R0,
                                    fix_covariances=True)
    mt = MT.mt_threshold_gibbs(Y, 0, X, anim, ped.ainv(), cfg)
    term = [RandomTerm("animal", Z, ped.ainv(), ped.ids, True, k_diag=1 + ped.inbreeding())]
    st = threshold_gibbs(yc, X, term, True, ThresholdGibbsConfig(
        chains=4, iterations=12000, burn_in=2000, thin=2, max_iterations=12000, seed=6,
        start={"animal": 0.4}, fix_variances=True))
    u_st = st.terms["animal"].solution
    sd = np.sqrt(st.terms["animal"].pev)
    # both are MC estimates (~20,000 draws each, autocorrelated): 0.1 posterior SD
    assert np.max(np.abs(mt.ebv[:, 0] - u_st) / sd) < 0.1
    np.testing.assert_allclose(mt.thresholds_mean, st.thresholds_mean, atol=0.03)
    lin = blup(yo, X, term, {"animal": 2.0, "residual": 3.0}, method="dense")
    u_lin = lin.terms["animal"].solution
    sd_lin = np.sqrt(lin.terms["animal"].pev)
    assert np.max(np.abs(mt.ebv[:, 1] - u_lin) / sd_lin) < 0.05
    np.testing.assert_allclose(mt.pev[:, 1], lin.terms["animal"].pev, rtol=0.05)


def test_refusals():
    ped, rng = _ped(20, 1, 5)
    n = ped.n
    X = sp.csr_matrix(np.ones((n, 1)))
    anim = np.arange(n)
    Y = np.column_stack([np.ones(n), rng.normal(size=n)])

    def prob(Y, X=X):
        return MT.MTProblem(Y, 0, MT.split_design(X, Y), anim, ped.ainv())
    with pytest.raises(ABPError) as e:
        prob(Y)                                                # one category only
    assert e.value.code == "ABP-E300"
    Y[:, 0] = np.linspace(0, 1, n)
    with pytest.raises(ABPError):
        prob(Y)                                                # not integer codes
    Y[:, 0] = rng.integers(1, 3, n)
    Y[3] = np.nan
    with pytest.raises(ABPError):
        prob(Y)                                                # empty record
    Y[3] = [1.0, 0.0]
    cfg = MT.MTThresholdGibbsConfig(chains=2, iterations=10, burn_in=5, max_iterations=10,
                                    start_R0=np.diag([2.0, 1.0]))
    with pytest.raises(ABPError):
        MT.mt_threshold_gibbs(Y, 0, X, anim, ped.ainv(), cfg)   # R0[c, c] must be 1
    # extreme category: a 0/1 column whose categorical records are all in category 1
    Y[:, 0] = np.where(np.arange(n) < 5, 1.0, rng.integers(1, 3, n))
    X2 = sp.csr_matrix(np.column_stack([np.ones(n), np.arange(n) < 5]).astype(float))
    with pytest.raises(ABPError) as e:
        prob(Y, X2)
    assert "extreme category" in e.value.message
    # a level without records of one trait: not estimable for that trait
    Y[:5, 1] = np.nan
    Y[:, 0] = rng.integers(1, 3, n)
    with pytest.raises(ABPError) as e:
        prob(Y, X2)
    assert "full column rank" in e.value.message


def test_expansion_moves_leave_the_posterior_unchanged():
    """With unknown G0 and R0 (proper inverse Wishart prior on G0: one categorical
    record per animal makes the flat-prior posterior improper), the chains with and
    without the parameter-expanded moves (scale and shear) target the same posterior:
    their posterior means agree within Monte-Carlo error; the moves improve mixing."""
    ped, rng = _ped(100, 21, 20)
    n = ped.n
    from scipy.sparse.linalg import spsolve_triangular
    F = spsolve_triangular(ped._l_matrix(), np.sqrt(ped.mendelian_d())[:, None]
                           * rng.standard_normal((n, 2)), lower=True, unit_diagonal=True)
    U = F @ np.linalg.cholesky([[0.5, 0.4], [0.4, 1.5]]).T
    E = rng.standard_normal((n, 2)) @ np.linalg.cholesky([[1.0, 0.4], [0.4, 2.0]]).T
    L = U + E + np.array([0.0, 5.0])
    Y = L.copy()
    Y[:, 0] = np.digitize(L[:, 0], [-0.4, 0.5]) + 1
    Y[rng.random(n) < 0.2, 1] = np.nan
    X = sp.csr_matrix(np.ones((n, 1)))
    out = {}
    for move in (True, False):
        cfg = MT.MTThresholdGibbsConfig(chains=4, iterations=3000, burn_in=500, thin=2,
                                        max_iterations=3000, seed=31 if move else 32,
                                        scale_move=move, shear_moves=move, prior_nu=6.0,
                                        prior_G0=np.diag([0.3, 1.5]))
        r = MT.mt_threshold_gibbs(Y, 0, X, np.arange(n), ped.ainv(), cfg)
        out[move] = r
    for key in ("G0_0_0", "G0_0_1", "G0_1_1", "R0_0_1", "R0_1_1"):
        a, b = out[True].summaries[key], out[False].summaries[key]
        diff = abs(a["mean"] - b["mean"])
        assert diff < 4.0 * math.hypot(a["mcse_mean"], b["mcse_mean"]), (key, a, b)
    # the moves improve the mixing of the categorical trait's genetic variance and of
    # the genetic correlation
    for key in ("G0_0_0", "rG_0_1"):
        assert out[True].summaries[key]["ess_bulk"] > 1.3 * out[False].summaries[key]["ess_bulk"]
