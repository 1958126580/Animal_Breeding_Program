"""Bayesian marker models: exact inclusion probabilities, the conjugate
Gaussian posterior (BRR with fixed variances) against the MME, recovery of a
sparse architecture, and convergence gating."""

import math

import numpy as np
import pytest

from abp.errors import ABPError
from abp.solvers.bayes import BayesConfig, run_bayes, sweep_python


def _geno(n, m, seed):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.1, 0.9, m)
    M = (rng.random((n, m)) < p).astype(float) + (rng.random((n, m)) < p)
    phat = M.mean(0) / 2
    return M - 2 * phat, phat, rng


def test_inclusion_probability_matches_exact_marginal_likelihood():
    """BayesC sweep for one marker: P(delta = 1 | r) from the code's
    log-odds equals the exact ratio of Gaussian marginal likelihoods
    N(r; 0, s2 I + v w w') vs N(r; 0, s2 I)."""
    rng = np.random.default_rng(0)
    n = 30
    w = rng.normal(size=n)
    r = rng.normal(size=n)
    s2, v, pi0 = 1.3, 0.4, 0.7
    from scipy.stats import multivariate_normal as mvn
    l1 = mvn(mean=np.zeros(n), cov=s2 * np.eye(n) + v * np.outer(w, w)).logpdf(r)
    l0 = mvn(mean=np.zeros(n), cov=s2 * np.eye(n)).logpdf(r)
    p_exact = 1.0 / (1.0 + math.exp(math.log(pi0) + l0 - math.log(1 - pi0) - l1))
    # run the sweep with u just below / above p_exact: inclusion flips exactly there
    for u, expect in ((p_exact - 1e-9, 1), (p_exact + 1e-9, 0)):
        e = r.copy()
        beta = np.zeros(1)
        delta = np.zeros(1, dtype=np.int64)
        sweep_python(w.reshape(n, 1).copy(order="F"), np.array([w @ w]), e, beta, delta,
                     np.array([v]), np.log([pi0, 1 - pi0]), np.array([0.0, v]), s2, 1,
                     np.zeros(1), np.array([u]))
        assert delta[0] == expect


def test_brr_fixed_variances_matches_exact_gaussian_posterior():
    """BRR with fixed sigma_b^2 and sigma_e^2: the posterior of the GEBVs is
    Gaussian with mean and covariance given by the MME; MCMC means must lie
    within 5 Monte-Carlo SE and posterior SDs within 10%."""
    W, p, rng = _geno(60, 40, 1)
    X = np.ones((60, 1))
    y = 3.0 + W @ rng.normal(0, 0.3, 40) + rng.normal(0, 1.0, 60)
    sb2, se2 = 0.09, 1.0
    C = np.block([[X.T @ X, X.T @ W], [W.T @ X, W.T @ W + np.eye(40) * se2 / sb2]])
    sol = np.linalg.solve(C, np.concatenate([X.T @ y, W.T @ y]))
    Cinv = np.linalg.inv(C) * se2
    g_exact = W @ sol[1:]
    g_sd = np.sqrt(np.einsum("ij,jk,ik->i", W, Cinv[1:, 1:], W))
    cfg = BayesConfig(method="BRR", chains=4, iterations=6000, burn_in=500, thin=1, seed=11,
                      fix_variances={"sigma_b2": sb2, "sigma_e2": se2})
    res = run_bayes(y, X, W, W, cfg, float(2 * np.sum(p * (1 - p))))
    draws = 4 * res.draws_per_chain
    mcse = g_sd / np.sqrt(draws / 10.0)          # conservative ESS: 1 in 10 draws
    assert np.all(np.abs(res.gebv_mean - g_exact) < 5 * mcse)
    np.testing.assert_allclose(res.gebv_sd, g_sd, rtol=0.1)
    assert res.fixed_mean[0] == pytest.approx(sol[0], abs=0.05)


def test_bayescpi_recovers_sparse_architecture():
    W, p, rng = _geno(400, 200, 3)
    beta = np.zeros(200)
    qtl = rng.choice(200, 8, replace=False)
    beta[qtl] = rng.normal(0, 1.0, 8)
    y = W @ beta + rng.normal(0, 1.0, 400)
    cfg = BayesConfig(method="BayesCpi", chains=4, iterations=3000, burn_in=500, thin=2, seed=5,
                      max_iterations=3000)
    res = run_bayes(y, np.ones((400, 1)), W, W, cfg, float(2 * np.sum(p * (1 - p))))
    top = set(np.argsort(-res.inclusion_prob)[:8])
    assert len(top & set(qtl)) >= 6
    pi0 = res.summaries["pi0"]["mean"]
    assert 0.85 < pi0 < 1.0
    assert np.corrcoef(res.gebv_mean, W @ beta)[0, 1] > 0.9


@pytest.mark.parametrize("method", ["BRR", "BayesA", "BayesB", "BayesC", "BayesR"])
def test_all_methods_run_and_report_diagnostics(method):
    W, p, rng = _geno(150, 60, 7)
    y = W @ rng.normal(0, 0.2, 60) + rng.normal(0, 1.0, 150)
    cfg = BayesConfig(method=method, chains=4, iterations=600, burn_in=100, thin=1, seed=1,
                      max_iterations=600, pi0=0.8)
    res = run_bayes(y, np.ones((150, 1)), W, W[:20], cfg, float(2 * np.sum(p * (1 - p))))
    assert res.gebv_mean.shape == (20,) and np.all(np.isfinite(res.gebv_mean))
    assert "sigma_e2" in res.summaries and res.summaries["sigma_e2"]["rhat"] > 0.9
    assert res.priors.derivation["prior_r2"] == 0.5


def test_seed_determinism_and_chain_independence():
    W, p, rng = _geno(80, 30, 9)
    y = W @ rng.normal(0, 0.3, 30) + rng.normal(size=80)
    cfg = BayesConfig(method="BayesC", chains=2, iterations=300, burn_in=50, thin=1, seed=3,
                      max_iterations=300)
    a = run_bayes(y, np.ones((80, 1)), W, W, cfg, 1.0)
    b = run_bayes(y, np.ones((80, 1)), W, W, cfg, 1.0)
    np.testing.assert_array_equal(a.gebv_mean, b.gebv_mean)
    assert len(set(a.seeds)) == 2


def test_invalid_configuration_is_rejected():
    W, p, rng = _geno(20, 5, 1)
    with pytest.raises(ABPError):
        run_bayes(np.zeros(20) + rng.normal(size=20), np.ones((20, 1)), W, W,
                  BayesConfig(method="BayesZ"), 1.0)
    with pytest.raises(ABPError):
        run_bayes(rng.normal(size=20), np.ones((20, 1)), W, W,
                  BayesConfig(method="BRR", chains=1), 1.0)


@pytest.mark.parametrize("code", [0, 1, 2])
def test_native_sweep_matches_python_reference(code):
    from abp.core import pedigree as pmod
    if pmod._native is None or not hasattr(pmod._native, "bayes_sweep"):
        pytest.skip("native kernel not compiled (recorded as not_run)")
    W, p, rng = _geno(70, 50, 4)
    Wf = np.asfortranarray(W)
    wtw = np.einsum("ij,ij->j", W, W)
    e0 = rng.normal(size=70)
    beta0 = rng.normal(0, 0.1, 50) * (rng.random(50) < 0.5)
    var_j = rng.uniform(0.01, 0.1, 50)
    if code == 2:
        log_pi = np.log([0.6, 0.2, 0.15, 0.05])
        comp_var = np.array([0.0, 1e-4, 1e-3, 1e-2]) * 5.0
    else:
        log_pi = np.log([0.7, 0.3])
        comp_var = np.array([0.0, 1.0])
    z, u = rng.normal(size=50), rng.random(50)
    outs = []
    for native in (False, True):
        e, beta = e0.copy(), beta0.copy()
        delta = np.zeros(50, dtype=np.int64)
        if native:
            pmod._native.bayes_sweep(Wf.T, wtw, e, beta, delta, var_j, log_pi, comp_var, 1.2,
                                     code, z, u)
        else:
            sweep_python(Wf, wtw, e, beta, delta, var_j, log_pi, comp_var, 1.2, code, z, u)
        outs.append((e, beta, delta))
    np.testing.assert_allclose(outs[1][0], outs[0][0], atol=1e-12)
    np.testing.assert_allclose(outs[1][1], outs[0][1], atol=1e-12)
    np.testing.assert_array_equal(outs[1][2], outs[0][2])


def test_skipped_gebv_diagnostics_are_reported_not_silent(monkeypatch):
    import abp.solvers.bayes as bm
    monkeypatch.setattr(bm, "GEBV_STORE_LIMIT", 10)
    W, p, rng = _geno(60, 20, 2)
    y = W @ rng.normal(0, 0.3, 20) + rng.normal(size=60)
    cfg = BayesConfig(method="BRR", chains=2, iterations=200, burn_in=50, thin=1, seed=4,
                      max_iterations=200)
    res = run_bayes(y, np.ones((60, 1)), W, W, cfg, float(2 * np.sum(p * (1 - p))))
    assert "not_computed" in res.gebv_diagnostics and res.gebv_diagnostics["n_animals"] == 60


def test_explicit_priors_traces_and_draws():
    """Explicit hyper-parameters are used as given (SBC needs this), traces and
    draws have the documented shapes, and posterior means equal the mean of
    the returned draws."""
    from abp.solvers.bayes import BayesPriors
    W, p, rng = _geno(50, 12, 6)
    y = W @ rng.normal(0, 0.3, 12) + rng.normal(size=50)
    pri = BayesPriors(nu=5.0, s2=0.02, nu_e=5.0, s2_e=0.6, pi0=0.8, gamma=(0.0, 1.0),
                      dirichlet=(1.0, 1.0), derivation={"source": "test"})
    cfg = BayesConfig(method="BayesC", chains=2, iterations=300, burn_in=100, thin=2, seed=9,
                      max_iterations=300, priors=pri, return_draws=True)
    res = run_bayes(y, np.ones((50, 1)), W, W[:7], cfg, 123.0)
    assert res.priors is pri
    assert res.traces["sigma_e2"].shape == (2, 100)
    assert res.gebv_draws.shape == (2, 100, 7) and res.beta_draws.shape == (2, 100, 12)
    np.testing.assert_allclose(res.gebv_mean, res.gebv_draws.reshape(200, 7).mean(0), rtol=1e-12)
    np.testing.assert_allclose(res.beta_mean, res.beta_draws.reshape(200, 12).mean(0),
                               rtol=1e-12, atol=1e-15)
    assert set(res.ppc["statistics"]) == {"sd", "skewness", "min", "max"}
    bad = BayesPriors(nu=2.0, s2=0.02, nu_e=5.0, s2_e=0.6, pi0=0.8, gamma=(0.0, 1.0),
                      dirichlet=(1.0, 1.0))
    with pytest.raises(ABPError):
        run_bayes(y, np.ones((50, 1)), W, W, BayesConfig(method="BayesC", chains=2,
                                                         priors=bad), 1.0)
