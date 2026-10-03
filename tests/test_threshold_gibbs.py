"""Gibbs sampler for the threshold model against independent references:
truncated-normal draws against scipy.stats.truncnorm (incl. far tails), the
block draw of the location effects against its exact normal conditional,
posterior means against numerical quadrature of the exact posterior on tiny
models, the precision root, and the workflow."""

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.stats import kstest, norm, truncnorm

from abp.core.design import build_fixed_design
from abp.core.pedigree import Pedigree
from abp.solvers.blup import RandomTerm
from abp.solvers.threshold_gibbs import (ThresholdGibbsConfig, _Problem, draw_location,
                                         precision_root, rtruncnorm, threshold_gibbs)


@pytest.mark.parametrize("lo,hi", [(-0.5, 1.0), (8.0, np.inf), (-np.inf, -9.0), (3.0, 3.5),
                                   (-np.inf, np.inf)])
def test_truncated_normal_draws_match_scipy(lo, hi):
    rng = np.random.default_rng(1)
    x = rtruncnorm(rng, np.full(20000, 0.3), np.full(20000, lo), np.full(20000, hi))
    assert np.all((x >= lo) & (x <= hi))
    ref = truncnorm(lo - 0.3, hi - 0.3, loc=0.3)
    assert kstest(x, ref.cdf).pvalue > 1e-3


def test_precision_root_reproduces_k_inverse():
    ids = [str(i) for i in range(12)]
    sires = [None] * 4 + ["0", "0", "1", "1", "4", "6", "8", "8"]
    dams = [None] * 4 + ["2", "3", "2", "3", "5", "7", "9", "5"]
    Ainv = Pedigree.from_parent_ids(ids, sires, dams).ainv()
    for K in (Ainv, Ainv.toarray(), sp.identity(12, format="csr") * 2.0):
        root = precision_root(K)
        F = np.column_stack([root(e) for e in np.eye(12)])
        Kd = K.toarray() if sp.issparse(K) else K
        np.testing.assert_allclose(F @ F.T, Kd, atol=1e-12)


def _tiny(seed=3, n=40, levels=2, K=2, intercept=True):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, levels, n)
    liab = 0.3 + np.array([0.6, -0.4, 0.2])[:levels][g] + rng.standard_normal(n)
    cuts = [0.0, 0.9][:K - 1]
    y = 1.0 + np.searchsorted(cuts, liab)
    X = build_fixed_design({}, [], intercept, n).X
    Z = sp.csr_matrix((np.ones(n), (np.arange(n), g)), shape=(n, levels))
    term = RandomTerm("u", Z, sp.identity(levels, format="csr"), [f"g{k}" for k in range(levels)],
                      True, k_diag=np.ones(levels))
    return y, X, term, g


def test_block_draw_has_the_exact_conditional_mean_and_covariance():
    y, X, term, g = _tiny(levels=3)
    P = _Problem(y, X, [term], True, sample_variances=False)
    from abp.solvers.cholesky import SparseLDL
    var = np.array([0.7])
    C = P.coefficient(var)
    fac = SparseLDL(C)
    rng = np.random.default_rng(5)
    liab = rng.standard_normal(y.size)
    D = np.array([draw_location(P, fac, liab, var, rng) for _ in range(40000)])
    Cd = C.toarray()
    mean = np.linalg.solve(Cd, P.Wt @ liab)
    cov = np.linalg.inv(Cd)
    se = np.sqrt(np.diag(cov) / D.shape[0])
    assert np.all(np.abs(D.mean(axis=0) - mean) < 4.5 * se)
    np.testing.assert_allclose(np.cov(D.T), cov, atol=0.03 * np.abs(cov).max())


def _quadrature_binary(y, g, sigma2):
    """Exact posterior means of (b, u1, u2) for y = 1{b + u_g + e > 0} on a grid."""
    grid = np.linspace(-4, 4, 161)
    B, U1, U2 = np.meshgrid(grid, grid, grid, indexing="ij")
    logp = -0.5 * (U1 ** 2 + U2 ** 2) / sigma2
    for yi, gi in zip(y, g):
        eta = B + (U1 if gi == 0 else U2)
        logp = logp + (norm.logcdf(eta) if yi == 2 else norm.logcdf(-eta))
    w = np.exp(logp - logp.max())
    w /= w.sum()
    return np.array([(w * B).sum(), (w * U1).sum(), (w * U2).sum()])


def test_binary_posterior_means_equal_quadrature():
    y, X, term, g = _tiny(seed=3, n=40, levels=2, K=2)
    ref = _quadrature_binary(y, g, 0.5)
    cfg = ThresholdGibbsConfig(chains=4, iterations=20000, burn_in=1000, thin=1,
                               max_iterations=20000, start={"u": 0.5}, fix_variances=True,
                               seed=11)
    r = threshold_gibbs(y, X, [term], True, cfg)
    got = np.r_[r.fixed_mean, r.terms["u"].solution]
    sd = np.sqrt(np.r_[np.nan, r.terms["u"].pev])
    # Monte-Carlo SE from the effective sample size of the draws (>= several thousand here)
    tol = 0.03
    np.testing.assert_allclose(got, ref, atol=tol)
    assert np.all(np.isfinite(sd[1:]))


def test_three_category_threshold_posterior_equals_quadrature():
    """No random effect beyond a tiny fixed-variance term: posterior mean of the
    free threshold tau_2 and the intercept by 2-D quadrature (the random term's
    effect is integrated analytically because its variance is ~0)."""
    y, X, term, g = _tiny(seed=8, n=60, levels=3, K=3)
    tiny = RandomTerm("u", term.Z, sp.identity(3, format="csr"), term.labels, True,
                      k_diag=np.ones(3))
    cfg = ThresholdGibbsConfig(chains=4, iterations=20000, burn_in=2000, thin=1,
                               max_iterations=20000, start={"u": 1e-6}, fix_variances=True,
                               seed=3)
    r = threshold_gibbs(y, X, [tiny], True, cfg)
    bg = np.linspace(-2.5, 3.5, 601)
    tg = np.linspace(0.01, 4.0, 400)
    Bm, Tm = np.meshgrid(bg, tg, indexing="ij")
    logp = np.zeros_like(Bm)
    for yi in y:
        if yi == 1:
            logp += norm.logcdf(-Bm)
        elif yi == 2:
            logp += np.log(np.maximum(norm.cdf(Tm - Bm) - norm.cdf(-Bm), 1e-300))
        else:
            logp += norm.logcdf(Bm - Tm)
    w = np.exp(logp - logp.max())
    w /= w.sum()
    assert r.thresholds_mean[0] == 0.0
    assert r.thresholds_mean[1] == pytest.approx((w * Tm).sum(), abs=0.02)
    assert r.fixed_mean[0] == pytest.approx((w * Bm).sum(), abs=0.02)
    assert 0.2 < r.acceptance < 0.6


def _rwm_reference(y, g, levels, nu, s2, n_iter=120000, seed=0):
    """Independent reference: adaptive random-walk Metropolis on the full joint
    posterior of (b, u_1..u_q, log sigma^2, tau_2) written with scipy.stats.norm."""
    yk = (y - 1).astype(int)
    rng = np.random.default_rng(seed)

    def logpost(x):
        b, u, ls, t2 = x[0], x[1:1 + levels], x[1 + levels], x[2 + levels]
        if t2 <= 0:
            return -np.inf
        s = np.exp(ls)
        eta = b + u[g]
        tau = np.array([-np.inf, 0.0, t2, np.inf])
        P = norm.cdf(tau[yk + 1] - eta) - norm.cdf(tau[yk] - eta)
        if np.any(P <= 0):
            return -np.inf
        return (np.log(P).sum() - 0.5 * u @ u / s - 0.5 * levels * ls
                - (nu / 2 + 1) * ls - nu * s2 / (2 * s) + ls)      # + ls: Jacobian of log
    x = np.r_[0.0, np.zeros(levels), np.log(0.5), 1.0]
    lp = logpost(x)
    d = x.size
    cov = np.eye(d) * 0.05
    draws = []
    mean_run, m2 = x.copy(), np.zeros((d, d))
    for it in range(1, n_iter + 1):
        prop = x + rng.multivariate_normal(np.zeros(d), cov * 2.38 ** 2 / d)
        lq = logpost(prop)
        if np.log(rng.random()) < lq - lp:
            x, lp = prop, lq
        if it < n_iter // 4:                         # adaptation during burn-in only
            delta = x - mean_run
            mean_run += delta / it
            m2 += np.outer(delta, x - mean_run)
            if it > 2000 and it % 500 == 0:
                cov = m2 / (it - 1) + 1e-6 * np.eye(d)
        else:
            draws.append(x.copy())
    return np.array(draws)


def test_variance_posterior_equals_independent_metropolis_reference():
    from abp.solvers import mcmc_diagnostics as dg
    levels = 6
    y, X, term, g = _tiny(seed=21, n=150, levels=3, K=3)
    rng = np.random.default_rng(4)
    g = rng.integers(0, levels, y.size)
    liab = 0.2 + rng.normal(0, 0.8, levels)[g] + rng.standard_normal(y.size)
    y = 1.0 + np.searchsorted([0.0, 0.9], liab)
    Z = sp.csr_matrix((np.ones(y.size), (np.arange(y.size), g)), shape=(y.size, levels))
    term = RandomTerm("u", Z, sp.identity(levels, format="csr"),
                      [f"g{k}" for k in range(levels)], True, k_diag=np.ones(levels))
    nu, s2 = 4.0, 0.5
    cfg = ThresholdGibbsConfig(chains=4, iterations=12000, burn_in=1000, thin=1,
                               max_iterations=12000, nu=nu, s2=s2, seed=7)
    r = threshold_gibbs(y, X, [term], True, cfg)
    ref = _rwm_reference(y, g, levels, nu, s2)
    s_ref = np.exp(ref[:, 1 + levels])
    se_ref = dg.mcse_mean(s_ref[None, :])
    tr = r.traces["var_u"]
    se_g = dg.mcse_mean(tr)
    diff = r.variances["u"]["mean"] - s_ref.mean()
    assert abs(diff) < 4 * np.hypot(se_ref, se_g), (r.variances["u"]["mean"], s_ref.mean())
    t_ref = ref[:, 2 + levels]
    diff_t = r.thresholds_mean[1] - t_ref.mean()
    assert abs(diff_t) < 4 * np.hypot(dg.mcse_mean(t_ref[None, :]),
                                      dg.mcse_mean(r.traces["tau_2"]))


def _write_threshold_case(tmp_path, bayes_lines, n=400, seed=2):
    """Sex-consistent pedigree, one categorical record per animal (3 categories)."""
    rng = np.random.default_rng(seed)
    ids = [f"a{i}" for i in range(n)]
    male = np.arange(n) % 2 == 0
    s, d = [], []
    for i in range(n):
        if i < 40:
            s.append(None)
            d.append(None)
        else:
            s.append(ids[rng.choice(np.flatnonzero(male[:i]))])
            d.append(ids[rng.choice(np.flatnonzero(~male[:i]))])
    ped = Pedigree.from_parent_ids(ids, s, d)
    u = np.linalg.cholesky(0.6 * ped.a_dense()) @ rng.standard_normal(n)
    liab = u[ped.index_of(ids)] + rng.standard_normal(n)
    y = 1 + np.searchsorted([0.0, 0.8], liab)
    with open(tmp_path / "ped.csv", "w", encoding="utf-8") as fh:
        fh.write("id,sire,dam,sex\n")
        for a, s_, d_, m_ in zip(ids, s, d, male):
            fh.write(f"{a},{s_ or '0'},{d_ or '0'},{'M' if m_ else 'F'}\n")
    with open(tmp_path / "phe.csv", "w", encoding="utf-8") as fh:
        fh.write("id,score\n")
        for a, v in zip(ids, y):
            fh.write(f"{a},{v}\n")
    spec = tmp_path / "a.toml"
    spec.write_text(f"""schema_version = "1"
[project]
name = "tg"
species = "sheep"
synthetic_data = true
[analysis]
task = "additive_ebv"
target_population = "test"
information_cutoff = "2025-12-31"
genetic_base = "unknown parents"
[data]
pedigree = "{(tmp_path / 'ped.csv').as_posix()}"
phenotypes = "{(tmp_path / 'phe.csv').as_posix()}"
[data.pedigree_columns]
sex = "sex"
[[traits]]
name = "score"
unit = "class"
type = "categorical"
[model]
traits = ["score"]
random = [{{ name = "animal", kind = "additive", relationship = "pedigree" }}]
[variances]
mode = "bayes"
[bayes]
method = "threshold"
{bayes_lines}
""", encoding="utf-8")
    return spec


def test_workflow_threshold_gibbs_converges_and_writes_evidence(tmp_path):
    import csv
    import json
    from abp.workflows.evaluate import run_evaluation
    spec = _write_threshold_case(tmp_path, "iterations = 3000\nburn_in = 500\nthin = 2\n"
                                           "max_iterations = 12000")
    out = run_evaluation(spec, tmp_path / "o", console=False)
    t = out.results["traits"]["score"]
    assert t["variance_source"].startswith("bayes (threshold-model Gibbs")
    assert t["bayes"]["converged"] and t["variance_components"]["residual"] == 1.0
    diag = json.loads((out.out_dir / "mcmc_diagnostics_score.json").read_text(encoding="utf-8"))
    assert diag["summaries"]["var_animal"]["rhat"] < 1.01
    assert diag["ebv_diagnostics"]["max_rhat"] < 1.01
    with open(out.out_dir / "mcmc_trace_score.csv", encoding="utf-8") as fh:
        assert "var_animal" in next(csv.reader(fh))
    assert "Threshold-model Gibbs sampler" in (out.out_dir / "report.md").read_text(encoding="utf-8")
    assert 0 <= t["reliability_summary"]["min"] <= t["reliability_summary"]["max"] <= 1


def test_workflow_threshold_gibbs_withholds_unconverged_results(tmp_path):
    from abp.errors import ABPError
    from abp.workflows.evaluate import run_evaluation
    spec = _write_threshold_case(tmp_path, "iterations = 60\nburn_in = 20\nthin = 1\n"
                                           "max_iterations = 60")
    with pytest.raises(ABPError) as exc:
        run_evaluation(spec, tmp_path / "o", console=False)
    assert exc.value.code == "ABP-E405"


def test_spec_rules_for_threshold_gibbs():
    from abp.core.spec import validate_spec_dict
    from abp.errors import ABPError
    base = {"schema_version": "1",
            "project": {"name": "t", "species": "sheep", "synthetic_data": True},
            "analysis": {"task": "additive_ebv", "target_population": "x",
                         "information_cutoff": "2024-12-31", "genetic_base": "x"},
            "data": {"pedigree": "p.csv", "phenotypes": "y.csv"},
            "traits": [{"name": "y", "unit": "c", "type": "categorical"}],
            "model": {"traits": ["y"], "random": [{"name": "animal", "kind": "additive",
                                                   "relationship": "pedigree"}]},
            "variances": {"mode": "bayes"}}
    assert validate_spec_dict(dict(base, bayes={"method": "threshold"}))["bayes"]["method"] \
        == "threshold"
    with pytest.raises(ABPError, match="threshold"):
        validate_spec_dict(dict(base, bayes={"method": "BayesC"}))
    cont = dict(base, traits=[{"name": "y", "unit": "c"}])
    with pytest.raises(ABPError, match="categorical"):
        validate_spec_dict(dict(cont, bayes={"method": "threshold"}))


def test_per_term_prior_scales_equal_the_scalar_prior():
    y, X, term, g = _tiny(seed=21, n=150, levels=3, K=3)
    common = dict(chains=2, iterations=300, burn_in=100, thin=1, max_iterations=300, nu=4.0,
                  seed=9)
    a = threshold_gibbs(y, X, [term], True, ThresholdGibbsConfig(s2=0.5, **common))
    b = threshold_gibbs(y, X, [term], True, ThresholdGibbsConfig(s2={"u": 0.5}, **common))
    np.testing.assert_array_equal(a.traces["var_u"], b.traces["var_u"])


def test_spec_rules_for_threshold_priors():
    from abp.core.spec import validate_spec_dict
    from abp.errors import ABPError
    base = {"schema_version": "1",
            "project": {"name": "t", "species": "sheep", "synthetic_data": True},
            "analysis": {"task": "additive_ebv", "target_population": "x",
                         "information_cutoff": "2024-12-31", "genetic_base": "x"},
            "data": {"pedigree": "p.csv", "phenotypes": "y.csv"},
            "traits": [{"name": "y", "unit": "c", "type": "categorical"}],
            "model": {"traits": ["y"], "random": [{"name": "animal", "kind": "additive",
                                                   "relationship": "pedigree"}]},
            "variances": {"mode": "bayes"}}
    ok = validate_spec_dict(dict(base, bayes={"method": "threshold",
                                              "variance_prior": "scaled_inv_chi2",
                                              "prior_variances": {"animal": 0.1}}))
    assert ok["bayes"]["prior_variances"] == {"animal": 0.1}
    for bad in ({"method": "threshold", "variance_prior": "scaled_inv_chi2"},
                {"method": "threshold", "variance_prior": "scaled_inv_chi2",
                 "prior_variances": {"pe": 0.1}},
                {"method": "threshold", "prior_variances": {"animal": 0.1}}):
        with pytest.raises(ABPError, match="prior_variances"):
            validate_spec_dict(dict(base, bayes=bad))


def test_workflow_threshold_gibbs_with_informative_prior(tmp_path):
    import json
    from abp.workflows.evaluate import run_evaluation
    spec = _write_threshold_case(tmp_path, "iterations = 3000\nburn_in = 500\nthin = 2\n"
                                           "max_iterations = 12000\n"
                                           'variance_prior = "scaled_inv_chi2"\nnu = 6\n'
                                           "prior_variances = { animal = 0.5 }")
    out = run_evaluation(spec, tmp_path / "o", console=False)
    diag = json.loads((out.out_dir / "mcmc_diagnostics_score.json").read_text(encoding="utf-8"))
    assert "nu = 6" in diag["priors"]["variances"]
    assert "scaled inverse chi-square" in (out.out_dir / "report.md").read_text(encoding="utf-8")
