"""Threshold (ordered probit) animal model against an independent dense
optimizer written with scipy.stats.norm, Laplace PEV against a numerical
Hessian, invariances and refusals."""

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.stats import norm

from abp.core.design import FixedTerm, build_fixed_design
from abp.core.pedigree import Pedigree
from abp.errors import ABPError
from abp.solvers.blup import RandomTerm
from abp.solvers.threshold import threshold_blup
from tests.reference.dense_reference import tabular_a


def _problem(seed=1, n_anim=40, n_rec=120, K=3):
    rng = np.random.default_rng(seed)
    ids = [str(i) for i in range(n_anim)]
    sires, dams = [], []
    for i in range(n_anim):
        if i < 8:
            sires.append(None)
            dams.append(None)
        else:
            s, d = rng.choice(i, 2, replace=False)
            sires.append(ids[s])
            dams.append(ids[d])
    ped = Pedigree.from_parent_ids(ids, sires, dams)
    A = tabular_a(ids, sires, dams)
    u = np.linalg.cholesky(0.4 * A) @ rng.standard_normal(n_anim)
    rec = rng.integers(0, n_anim, n_rec)
    grp = rng.integers(0, 3, n_rec)
    liab = 0.3 * grp + u[rec] + rng.standard_normal(n_rec)
    cuts = np.quantile(liab, np.linspace(0, 1, K + 1)[1:-1])
    y = 1 + np.searchsorted(cuts, liab).astype(float)
    fd = build_fixed_design({"g": [f"g{k}" for k in grp]}, [FixedTerm("g", "factor")], True, n_rec)
    col = ped.index_of([ids[r] for r in rec])
    Z = sp.csr_matrix((np.ones(n_rec), (np.arange(n_rec), col)), shape=(n_rec, n_anim))
    term = RandomTerm("animal", Z, ped.ainv(), ped.ids, True, k_diag=1 + ped.inbreeding(),
                      logdet_k=ped.logdet_a())
    Ainv_ped = ped.ainv().toarray()
    return y, fd.X, term, Z.toarray(), Ainv_ped, K


def _reference(y, X, Z, Ainv, s2, K):
    """Negative log posterior with norm.cdf in the natural parameterisation
    (b, u, tau_2..tau_{K-1}), tau_1 = 0; minimised by BFGS."""
    X = X.toarray()
    cats = np.unique(y)
    yk = np.searchsorted(cats, y)
    p, q = X.shape[1], Z.shape[1]

    def f(x):
        b, u, tf = x[:p], x[p:p + q], x[p + q:]
        tau = np.concatenate([[-np.inf, 0.0], tf, [np.inf]])
        eta = X @ b + Z @ u
        P = norm.cdf(tau[yk + 1] - eta) - norm.cdf(tau[yk] - eta)
        return -(np.sum(np.log(P)) - 0.5 * u @ Ainv @ u / s2)
    x0 = np.concatenate([np.zeros(p + q), np.arange(1, K - 1, dtype=float)])
    r = minimize(f, x0, method="BFGS", options={"gtol": 1e-10, "maxiter": 10000})
    return r.x, f


def test_mode_equals_independent_optimizer_and_laplace_pev_equals_numerical_hessian():
    y, X, term, Z, Ainv, K = _problem()
    res = threshold_blup(y, X, [term], {"animal": 0.4, "residual": 1.0}, intercept=True)
    x_ref, f = _reference(y, X, Z, Ainv, 0.4, K)
    p = X.shape[1]
    np.testing.assert_allclose(res.fixed_solution, x_ref[:p], atol=2e-5)
    np.testing.assert_allclose(res.terms["animal"].solution, x_ref[p:p + Z.shape[1]], atol=2e-5)
    np.testing.assert_allclose(res.thresholds, np.r_[0.0, x_ref[p + Z.shape[1]:]], atol=2e-5)
    # Laplace PEV = diag of the inverse numerical Hessian of the reference objective
    x = np.concatenate([res.fixed_solution, res.terms["animal"].solution, res.thresholds[1:]])
    h = 1e-4
    n = x.size
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            e_i, e_j = np.eye(n)[i] * h, np.eye(n)[j] * h
            H[i, j] = H[j, i] = (f(x + e_i + e_j) - f(x + e_i - e_j) - f(x - e_i + e_j)
                                 + f(x - e_i - e_j)) / (4 * h * h)
    pev_ref = np.diag(np.linalg.inv(H))[p:p + Z.shape[1]]
    np.testing.assert_allclose(res.terms["animal"].pev, pev_ref, rtol=2e-3)
    assert np.all((res.terms["animal"].reliability >= 0) & (res.terms["animal"].reliability <= 1))


def test_binary_trait_and_category_relabelling():
    y, X, term, Z, Ainv, K = _problem(3, K=2)
    r1 = threshold_blup(y, X, [term], {"animal": 0.5, "residual": 1.0}, intercept=True)
    assert r1.thresholds.tolist() == [0.0]
    # codes 1/2 -> 10/20 give the same result (only the order of categories matters)
    r2 = threshold_blup(y * 10, X, [term], {"animal": 0.5, "residual": 1.0}, intercept=True)
    np.testing.assert_allclose(r2.terms["animal"].solution, r1.terms["animal"].solution, atol=1e-12)
    # reversing the category order negates the liability solutions
    r3 = threshold_blup(3 - y, X, [term], {"animal": 0.5, "residual": 1.0}, intercept=True)
    np.testing.assert_allclose(r3.terms["animal"].solution, -r1.terms["animal"].solution, atol=1e-8)


def test_refusals():
    y, X, term, Z, Ainv, K = _problem(5)
    with pytest.raises(ABPError, match="residual"):
        threshold_blup(y, X, [term], {"animal": 0.4, "residual": 2.0}, intercept=True)
    with pytest.raises(ABPError, match="one category"):
        threshold_blup(np.ones_like(y), X, [term], {"animal": 0.4, "residual": 1.0}, intercept=True)
    with pytest.raises(ABPError, match="integer"):
        threshold_blup(y + 0.5, X, [term], {"animal": 0.4, "residual": 1.0}, intercept=True)


def test_example14_threshold_workflow(tmp_path):
    from pathlib import Path
    import csv
    from abp.workflows.evaluate import run_evaluation
    ex = Path(__file__).resolve().parents[1] / "examples" / "14_sheep_nlb_threshold"
    out = run_evaluation(ex / "analysis.toml", tmp_path / "o", console=False)
    t = out.results["traits"]["nlb"]
    tm = t["threshold_model"]
    assert tm["categories"] == [1.0, 2.0, 3.0] and tm["thresholds"][0] == 0.0
    assert tm["thresholds"][1] > 0
    with open(out.out_dir / "thresholds_nlb.csv", encoding="utf-8") as fh:
        assert len(list(csv.DictReader(fh))) == 2
    assert (out.out_dir / "random_pe_nlb.csv").exists()
    report = (out.out_dir / "report.md").read_text(encoding="utf-8")
    assert "threshold (probit) model" in report and "liability" in report
    assert 0 <= t["reliability_summary"]["min"] <= t["reliability_summary"]["max"] <= 1


def test_spec_rules_for_categorical_traits():
    from abp.core.spec import validate_spec_dict
    base = {"schema_version": "1",
            "project": {"name": "t", "species": "sheep", "synthetic_data": True},
            "analysis": {"task": "additive_ebv", "target_population": "x",
                         "information_cutoff": "2024-12-31", "genetic_base": "x"},
            "data": {"pedigree": "p.csv", "phenotypes": "y.csv"},
            "traits": [{"name": "y", "unit": "c", "type": "categorical"}],
            "model": {"traits": ["y"], "random": [{"name": "animal", "kind": "additive",
                                                   "relationship": "pedigree"}]},
            "variances": {"mode": "known", "values": {"animal": 0.2, "residual": 1.0}}}
    assert validate_spec_dict(base)["traits"][0]["type"] == "categorical"
    bad = dict(base, variances={"mode": "known", "values": {"animal": 0.2, "residual": 2.0}})
    with pytest.raises(ABPError, match="residual"):
        validate_spec_dict(bad)
    with pytest.raises(ABPError, match="known"):
        validate_spec_dict(dict(base, variances={"mode": "reml"}))
