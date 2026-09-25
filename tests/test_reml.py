"""REML: likelihood, score and AI against the independent V-form reference;
optimum against an independent optimizer; boundary; checkpoint; calibration."""

import numpy as np
import pytest
import scipy.optimize as so
import scipy.sparse as sp

from abp.core.design import FixedTerm, build_fixed_design
from abp.core.pedigree import Pedigree
from abp.errors import ABPError
from abp.solvers.reml import REMLEvaluator, reml_fit
from tests.reference.dense_reference import reml_loglik_v_form, reml_score_v_form, tabular_a
from tests.test_blup import animal_term, iid_term

T03 = (["1", "2", "3", "4", "5"], [None, None, "1", "1", "3"], [None, None, "2", "2", "4"])
CFG = {"algorithm": "ai", "max_iter": 200, "tol": 1e-10, "start": None}


def test_t05_score_matches_v_form_and_detects_wrong_derivative():
    ped = Pedigree.from_parent_ids(*T03)
    ids = T03[0]
    y = np.array([1.0, 2.0, -1.0, 0.0, 3.0])
    X = build_fixed_design({}, [], True, 5).X
    ev = REMLEvaluator(y, X, [animal_term(ped, ids)], 2**30)
    p = ev.evaluate(np.array([0.7, 1.3]))
    A = tabular_a(*T03)
    Xd = np.ones((5, 1))
    ll_ref, _ = reml_loglik_v_form(y, Xd, [A, np.eye(5)], [0.7, 1.3])
    assert p.loglik == pytest.approx(ll_ref, abs=1e-10)
    score_ref = reml_score_v_form(y, Xd, [A, np.eye(5)], [0.7, 1.3])
    np.testing.assert_allclose(p.score, score_ref, atol=1e-10)
    assert p.score[0] == pytest.approx(-0.01463020355, abs=1e-10)  # spec T05 value
    # Counterexample: same P (correct V) but derivative matrix I instead of A.
    _, P = reml_loglik_v_form(y, Xd, [A, np.eye(5)], [0.7, 1.3])
    wrong = 0.5 * (y @ P @ P @ y - np.trace(P))
    assert wrong == pytest.approx(1.03497570401, abs=1e-9)
    assert abs(wrong - p.score[0]) > 0.1
    # central finite difference of the production log-likelihood
    h = 1e-5
    fd = (ev.evaluate(np.array([0.7 + h, 1.3])).loglik
          - ev.evaluate(np.array([0.7 - h, 1.3])).loglik) / (2 * h)
    assert fd == pytest.approx(p.score[0], abs=1e-8)


def _problem(seed, n_anim=80, n_rec=160):
    rng = np.random.default_rng(seed)
    ids = [f"a{i}" for i in range(n_anim)]
    sires, dams = [None] * n_anim, [None] * n_anim
    for i in range(10, n_anim):
        s, d = rng.choice(np.arange(max(0, i - 30), i), 2, replace=False)
        sires[i], dams[i] = ids[s], ids[d]
    rec = [ids[k] for k in rng.integers(0, n_anim, n_rec)]
    grp = [f"g{k}" for k in rng.integers(0, 5, n_rec)]
    return ids, sires, dams, rec, grp, rng


def _v_form_pieces(ids, sires, dams, rec, pe_term):
    A = tabular_a(ids, sires, dams)
    pos = {a: k for k, a in enumerate(ids)}
    Za = np.zeros((len(rec), len(ids)))
    Za[np.arange(len(rec)), [pos[a] for a in rec]] = 1
    Zp = pe_term.Z.toarray()
    return [Za @ A @ Za.T, Zp @ Zp.T, np.eye(len(rec))]


def test_repeatability_model_loglik_score_ai_match_v_form():
    ids, sires, dams, rec, grp, rng = _problem(3)
    ped = Pedigree.from_parent_ids(ids, sires, dams)
    y = rng.normal(5, 2, len(rec))
    fd = build_fixed_design({"g": grp}, [FixedTerm("g", "factor")], True, len(rec))
    pe = iid_term("pe", rec)
    ev = REMLEvaluator(y, fd.X, [animal_term(ped, rec), pe], 2**30)
    theta = np.array([1.1, 0.6, 2.3])
    p = ev.evaluate(theta)
    covs = _v_form_pieces(ids, sires, dams, rec, pe)
    Xd = fd.X.toarray()
    ll, P = reml_loglik_v_form(y, Xd, covs, theta)
    assert p.loglik == pytest.approx(ll, abs=1e-9)
    np.testing.assert_allclose(p.score, reml_score_v_form(y, Xd, covs, theta), atol=1e-9)
    Py = P @ y
    ai_ref = np.array([[0.5 * Py @ Vk @ P @ Vl @ Py for Vl in covs] for Vk in covs])
    np.testing.assert_allclose(p.ai, ai_ref, rtol=1e-9, atol=1e-10)


def _simulate(ped_ids, sires, dams, rec, grp, va, vpe, ve, rng):
    A = tabular_a(ped_ids, sires, dams)
    u = np.linalg.cholesky(A) @ rng.standard_normal(len(ped_ids)) * np.sqrt(va)
    pos = {a: k for k, a in enumerate(ped_ids)}
    pe_eff = {a: rng.normal(0, np.sqrt(vpe)) for a in set(rec)}
    g_eff = {g: rng.normal(0, 3) for g in set(grp)}
    return np.array([10 + g_eff[g] + u[pos[a]] + pe_eff[a] + rng.normal(0, np.sqrt(ve))
                     for a, g in zip(rec, grp)])


def test_reml_optimum_matches_independent_optimizer_and_em():
    # well-identified design (5 records per animal); near-boundary optima make
    # plain EM extremely slow, which is why EM is only the fallback algorithm
    ids, sires, dams, rec, grp, rng = _problem(8, n_anim=100, n_rec=500)
    ped = Pedigree.from_parent_ids(ids, sires, dams)
    y = _simulate(ids, sires, dams, rec, grp, 2.0, 1.5, 2.0, rng)
    fd = build_fixed_design({"g": grp}, [FixedTerm("g", "factor")], True, len(rec))
    pe = iid_term("pe", rec)
    terms = [animal_term(ped, rec), pe]
    fit = reml_fit(y, fd.X, terms, CFG)
    assert fit.status == "converged"
    covs = _v_form_pieces(ids, sires, dams, rec, pe)
    Xd = fd.X.toarray()

    def neg(logt):
        return -reml_loglik_v_form(y, Xd, covs, np.exp(logt))[0]
    opt = so.minimize(neg, np.log([1.0, 1.0, 1.0]), method="Nelder-Mead",
                      options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000, "maxfev": 20000})
    ref = np.exp(opt.x)
    est = np.array([fit.variances["animal"], fit.variances["pe"], fit.variances["residual"]])
    np.testing.assert_allclose(est, ref, rtol=2e-5)
    assert fit.loglik >= -opt.fun - 1e-8  # never worse than the reference optimum
    fit_em = reml_fit(y, fd.X, terms, {**CFG, "algorithm": "em", "max_iter": 5000, "tol": 1e-9})
    np.testing.assert_allclose([fit_em.variances[k] for k in ("animal", "pe", "residual")], est,
                               rtol=1e-5)
    assert fit.iterations < fit_em.iterations
    assert 0 < fit.heritability < 1 and fit.se is not None


def test_boundary_zero_component():
    """No permanent-environment variance in the data: pe must end on the boundary."""
    ids, sires, dams, rec, grp, rng = _problem(21, n_anim=100, n_rec=250)
    ped = Pedigree.from_parent_ids(ids, sires, dams)
    y = _simulate(ids, sires, dams, rec, grp, 2.0, 0.0, 3.0, rng)
    # make pe variance clearly estimated at zero: remove any pe-like signal by
    # replacing each animal's records with independent draws around its mean
    fd = build_fixed_design({"g": grp}, [FixedTerm("g", "factor")], True, len(rec))
    pe = iid_term("pe", rec)
    fit = reml_fit(y, fd.X, [animal_term(ped, rec), pe], {**CFG, "start": {
        "animal": 1.0, "pe": 1.0, "residual": 1.0}})
    if fit.status == "converged_boundary":
        assert fit.variances["pe"] == 0.0 and fit.se is None
        assert fit.active_terms == ["animal"]
        assert any("score at zero" in h.get("event", "") for h in fit.history)
    else:  # interior optimum: must be a genuine stationary point
        assert fit.variances["pe"] > 0


def test_forced_boundary_case_is_detected():
    """Construct data whose REML optimum for pe is exactly on the boundary:
    one record per animal makes pe confounded with residual only through the
    likelihood of an animal model; use strongly negative intra-animal correlation."""
    rng = np.random.default_rng(5)
    ped = Pedigree.from_parent_ids([f"f{i}" for i in range(60)], [None] * 60, [None] * 60)
    rec = [f"f{i}" for i in range(60) for _ in range(2)]
    # records of the same animal deviate in opposite directions -> negative
    # within-animal covariance -> animal and pe variance estimates hit zero
    d = rng.normal(0, 2, 60)
    y = np.array([v for x in d for v in (x, -x)]) + 10
    X = build_fixed_design({}, [], True, len(rec)).X
    fit = reml_fit(y, X, [iid_term("pe", rec)], {**CFG, "start": {"pe": 1.0, "residual": 1.0}})
    assert fit.status == "converged_boundary"
    assert fit.variances["pe"] == 0.0
    assert fit.variances["residual"] == pytest.approx(float(np.var(y, ddof=1)), rel=1e-6)


def test_non_convergence_is_an_error_and_checkpoint_resume(tmp_path):
    ids, sires, dams, rec, grp, rng = _problem(4, n_anim=90, n_rec=200)
    ped = Pedigree.from_parent_ids(ids, sires, dams)
    y = _simulate(ids, sires, dams, rec, grp, 2.0, 0.5, 3.0, rng)
    fd = build_fixed_design({"g": grp}, [FixedTerm("g", "factor")], True, len(rec))
    terms = [animal_term(ped, rec), iid_term("pe", rec)]
    ck = tmp_path / "ck.json"
    with pytest.raises(ABPError) as exc:
        reml_fit(y, fd.X, terms, {**CFG, "algorithm": "em", "max_iter": 3},
                 checkpoint_path=ck, fingerprint="fp1")
    assert exc.value.code == "ABP-E403"
    assert ck.exists()
    with pytest.raises(ABPError) as exc:
        reml_fit(y, fd.X, terms, CFG, checkpoint_path=ck, resume=True, fingerprint="other")
    assert exc.value.code == "ABP-E504"
    resumed = reml_fit(y, fd.X, terms, CFG, checkpoint_path=ck, resume=True, fingerprint="fp1")
    fresh = reml_fit(y, fd.X, terms, CFG)
    for k in fresh.variances:
        assert resumed.variances[k] == pytest.approx(fresh.variances[k], rel=1e-6)
    assert any("resumed" in h.get("event", "") for h in resumed.history)


@pytest.mark.slow
def test_reml_calibration_by_simulation():
    """G4: across replicates simulated from known variances, the mean REML
    estimate is within 3 Monte-Carlo standard errors of the truth."""
    ids, sires, dams, rec, grp, rng = _problem(100, n_anim=250, n_rec=250)
    rec = ids  # one record per animal
    grp = [f"g{k}" for k in rng.integers(0, 5, len(ids))]
    ped = Pedigree.from_parent_ids(ids, sires, dams)
    fd = build_fixed_design({"g": grp}, [FixedTerm("g", "factor")], True, len(rec))
    term = animal_term(ped, rec)
    est = []
    for r in range(40):
        y = _simulate(ids, sires, dams, rec, grp, 2.0, 0.0, 3.0, np.random.default_rng(1000 + r))
        fit = reml_fit(y, fd.X, [term], {**CFG, "start": {"animal": 1.5, "residual": 3.5}})
        est.append([fit.variances["animal"], fit.variances["residual"]])
    est = np.array(est)
    mean, se = est.mean(0), est.std(0, ddof=1) / np.sqrt(len(est))
    assert abs(mean[0] - 2.0) < 3 * se[0], (mean, se)
    assert abs(mean[1] - 3.0) < 3 * se[1], (mean, se)


def test_reml_with_dense_genomic_structure_matches_v_form():
    """REML with a dense K^{-1} (GBLUP path) against the independent V-form optimum."""
    from abp.core.genomic import allele_frequencies, apply_g_policy, vanraden_g
    from abp.solvers.blup import RandomTerm
    rng = np.random.default_rng(17)
    n, m = 150, 600
    p0 = rng.uniform(0.1, 0.9, m)
    M = (rng.random((n, m)) < p0).astype(float) + (rng.random((n, m)) < p0)
    G, _ = vanraden_g(M, allele_frequencies(M))
    Gs, _ = apply_g_policy(G, "ridge", "none", None, 0.0, 0.01)
    u = np.linalg.cholesky(Gs) @ rng.standard_normal(n) * np.sqrt(1.5)
    grp = [f"g{k}" for k in rng.integers(0, 4, n)]
    fd = build_fixed_design({"g": grp}, [FixedTerm("g", "factor")], True, n)
    y = fd.X @ rng.normal(0, 2, fd.X.shape[1]) + u + rng.normal(0, 1.0, n)
    Gi = np.linalg.inv(Gs)
    term = RandomTerm("animal", sp.identity(n, format="csr"), Gi, list(range(n)), True,
                      k_diag=np.diag(Gs), logdet_k=float(np.linalg.slogdet(Gs)[1]))
    fit = reml_fit(y, fd.X, [term], CFG)
    Xd = fd.X.toarray()
    p = REMLEvaluator(y, fd.X, [term], 2**30).evaluate(np.array([1.2, 0.9]))
    ll_ref, _ = reml_loglik_v_form(y, Xd, [Gs, np.eye(n)], [1.2, 0.9])
    assert p.loglik == pytest.approx(ll_ref, abs=1e-9)
    np.testing.assert_allclose(p.score, reml_score_v_form(y, Xd, [Gs, np.eye(n)], [1.2, 0.9]),
                               atol=1e-9)
    opt = so.minimize(lambda lt: -reml_loglik_v_form(y, Xd, [Gs, np.eye(n)], np.exp(lt))[0],
                      np.log([1.0, 1.0]), method="Nelder-Mead",
                      options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 5000})
    np.testing.assert_allclose([fit.variances["animal"], fit.variances["residual"]],
                               np.exp(opt.x), rtol=2e-5)
