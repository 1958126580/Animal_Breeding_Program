"""Matrix-free single step (H^-1 as an operator in PCG) against the explicit
construction: A22^-1 operator vs the dense inverse, APY operator vs the dense
APY inverse (incl. the blend policy built from genotypes), and whole
evaluations with identical EBVs."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from abp.core.genomic import apply_g_policy, apy_inverse, centered, scaling_d
from abp.core.pedigree import Pedigree
from abp.core.ssop import A22InverseOperator, APYOperator, apy_blocks_from_genotypes
from abp.errors import ABPError

ROOT = Path(__file__).resolve().parents[1]


def _ped(n=300, seed=4):
    rng = np.random.default_rng(seed)
    ids = [f"x{i}" for i in range(n)]
    male = np.arange(n) % 2 == 0
    s, d = [], []
    for i in range(n):
        if i < 30:
            s.append(None)
            d.append(None)
        else:
            s.append(ids[rng.choice(np.flatnonzero(male[:i]))])
            d.append(ids[rng.choice(np.flatnonzero(~male[:i]))])
    return Pedigree.from_parent_ids(ids, s, d), rng


def test_a22_inverse_operator_equals_dense_inverse():
    ped, rng = _ped()
    g = np.sort(rng.choice(ped.n, 90, replace=False))
    A22 = ped.a_submatrix(g)
    op = A22InverseOperator(ped.ainv(), g)
    V = rng.standard_normal((g.size, 3))
    ref = np.linalg.solve(A22, V)
    got = np.column_stack([op(V[:, k]) for k in range(3)])
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-9)
    assert np.all(op.diag_upper_bound() >= np.diag(np.linalg.inv(A22)) - 1e-12)


@pytest.mark.parametrize("policy", ["ridge", "blend"])
def test_apy_operator_from_genotypes_equals_dense_apy(policy):
    ped, rng = _ped()
    g = np.sort(rng.choice(ped.n, 120, replace=False))
    M = rng.integers(0, 3, (g.size, 400)).astype(float)
    p = M.mean(axis=0) / 2
    W = centered(M, p)
    d = scaling_d(p)
    G = W @ W.T / d
    A22 = ped.a_submatrix(g)
    Gs, _ = apply_g_policy(G, policy, "none", A22, 0.05, 0.01, check_pd=False)
    core = np.sort(rng.choice(g.size, 40, replace=False))
    ref = apy_inverse(Gs, core)
    A_cols = ped.a_columns(g[core])[g]
    gcc, gcn, gnn = apy_blocks_from_genotypes(W, d, core, policy, 0.05, 0.01, A_cols,
                                              1.0 + ped.inbreeding()[g])
    op = APYOperator(gcc, gcn, gnn, core, g.size)
    v = rng.standard_normal(g.size)
    np.testing.assert_allclose(op(v), ref.g_inv @ v, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(op.diag(), np.diag(ref.g_inv), rtol=1e-8)
    assert op.logdet == pytest.approx(ref.logdet, abs=1e-8)


def _spec(tmp_path, extra_genomic, name):
    ex = ROOT / "examples" / "05_sheep_wwt_single_step" / "analysis.toml"
    txt = ex.read_text(encoding="utf-8")
    txt = txt.replace("../sheep_data", (ROOT / "examples" / "sheep_data").as_posix())
    txt = txt.replace('[variances]\nmode = "reml"',
                      '[variances]\nmode = "known"\nvalues = { animal = 4.0, residual = 12.25 }')
    txt = txt.replace('tuning = "match_a22"\n', "tuning = \"none\"\n" + extra_genomic)
    txt += '\n[solver]\npev = "none"\ntol = 1e-11\n'
    p = tmp_path / f"{name}.toml"
    p.write_text(txt, encoding="utf-8")
    return p


def _ebv(out):
    with open(out.out_dir / "ebv_wwt.csv", encoding="utf-8") as fh:
        return {r["animal"]: float(r["ebv"]) for r in csv.DictReader(fh)}


@pytest.mark.parametrize("apy", [0, 150])
def test_matrix_free_evaluation_equals_explicit(tmp_path, apy):
    from abp.workflows.evaluate import run_evaluation
    g = f"apy_core_size = {apy}\n"
    ex = run_evaluation(_spec(tmp_path, g, "ex"), tmp_path / "ex", console=False)
    mf = run_evaluation(_spec(tmp_path, g + 'single_step_mode = "matrix_free"\n', "mf"),
                        tmp_path / "mf", console=False)
    e1, e2 = _ebv(ex), _ebv(mf)
    assert e1.keys() == e2.keys()
    a = np.array([e1[k] for k in e1])
    b = np.array([e2[k] for k in e1])
    np.testing.assert_allclose(b, a, atol=1e-7 * np.abs(a).max())
    s = mf.results["traits"]["wwt"]["solver"]
    assert s["method"] == "pcg" and "matrix-free" in s["selection_reason"]
    assert mf.manifest["relationship"]["single_step_mode"] == "matrix_free"


def test_spec_rules_for_matrix_free(tmp_path):
    from abp.core.spec import load_spec
    base = _spec(tmp_path, 'single_step_mode = "matrix_free"\n', "ok")
    load_spec(base)
    bad = base.read_text(encoding="utf-8").replace('pev = "none"', 'pev = "exact"')
    (tmp_path / "bad.toml").write_text(bad, encoding="utf-8")
    with pytest.raises(ABPError, match="pev"):
        load_spec(tmp_path / "bad.toml")


def test_h_sampler_has_the_single_step_covariance():
    from abp.core.genomic import single_step
    from abp.core.ssop import DenseInverseOperator, SingleStepHInverse
    ped, rng = _ped(n=80, seed=9)
    g = np.sort(rng.choice(ped.n, 25, replace=False))
    M = rng.integers(0, 3, (g.size, 300)).astype(float)
    p = M.mean(axis=0) / 2
    W = centered(M, p)
    G = W @ W.T / scaling_d(p)
    A22 = ped.a_submatrix(g)
    Gs, _ = apply_g_policy(G, "blend", "none", A22, 0.05, 0.01)
    H = np.linalg.inv(single_step(ped, g, Gs, a22=A22).h_inv.toarray())
    op = SingleStepHInverse(ped.ainv().tocsr(), g,
                            DenseInverseOperator(np.linalg.inv(Gs), np.linalg.cholesky(Gs)),
                            A22InverseOperator(ped.ainv(), g), ped=ped)
    S = np.array([op.sample(rng) for _ in range(40000)])
    emp = S.T @ S / S.shape[0]
    # Monte-Carlo SE of a covariance entry: sqrt((H_ii H_jj + H_ij^2) / N)
    se = np.sqrt((np.outer(np.diag(H), np.diag(H)) + H ** 2) / S.shape[0])
    z = (emp - H) / se
    assert np.abs(z).max() < 5.5 and abs(z.mean()) < 0.05 and 0.9 < z.std() < 1.1


def test_apy_sampler_has_the_apy_covariance():
    ped, rng = _ped()
    g = np.sort(rng.choice(ped.n, 60, replace=False))
    M = rng.integers(0, 3, (g.size, 400)).astype(float)
    p = M.mean(axis=0) / 2
    W = centered(M, p)
    d = scaling_d(p)
    core = np.sort(rng.choice(g.size, 20, replace=False))
    Gs, _ = apply_g_policy(W @ W.T / d, "ridge", "none", None, 0.05, 0.01, check_pd=False)
    ref = apy_inverse(Gs, core).g_apy
    gcc, gcn, gnn = apy_blocks_from_genotypes(W, d, core, "ridge", 0.05, 0.01, None, None)
    op = APYOperator(gcc, gcn, gnn, core, g.size)
    S = np.array([op.sample(rng) for _ in range(30000)])
    emp = S.T @ S / S.shape[0]
    se = np.sqrt((np.outer(np.diag(ref), np.diag(ref)) + ref ** 2) / S.shape[0])
    z = (emp - ref) / se
    assert np.abs(z).max() < 5.5 and abs(z.mean()) < 0.05 and 0.9 < z.std() < 1.1


@pytest.mark.parametrize("apy", [0, 150])
def test_sampled_reliabilities_agree_with_exact_ones(tmp_path, apy):
    from abp.workflows.evaluate import run_evaluation
    g = f"apy_core_size = {apy}\n"
    ex_spec = _spec(tmp_path, g, "ex")
    ex_spec.write_text(ex_spec.read_text(encoding="utf-8").replace('pev = "none"', 'pev = "exact"'),
                       encoding="utf-8")
    ex = run_evaluation(ex_spec, tmp_path / "ex", console=False)
    mf_spec = _spec(tmp_path, g + 'single_step_mode = "matrix_free"\n', "mf")
    mf_spec.write_text(mf_spec.read_text(encoding="utf-8").replace(
        'pev = "none"', 'pev = "sampled"\npev_samples = 300'), encoding="utf-8")
    mf = run_evaluation(mf_spec, tmp_path / "mf", console=False)

    def rel(out, col):
        with open(out.out_dir / "ebv_wwt.csv", encoding="utf-8") as fh:
            return {r["animal"]: float(r[col]) for r in csv.DictReader(fh)}
    r_ex = rel(ex, "reliability")
    r_mf = rel(mf, "reliability")
    se = rel(mf, "reliability_mc_se")
    keys = [k for k in r_ex if se[k] > 0]
    z = np.array([(r_mf[k] - r_ex[k]) / se[k] for k in keys])
    # the Monte-Carlo SE is calibrated (spread of z ~ 1); the mean of z is not used because
    # every SE is estimated from the same samples as its estimate (self-normalised statistics
    # have a small mean shift) and all animals share the samples; bias is checked on the
    # reliability scale instead
    assert 0.8 < z.std() < 1.25
    assert abs(np.mean([r_mf[k] - r_ex[k] for k in keys])) < 0.15 * np.mean([se[k] for k in keys])
    # and its size matches the observed errors: E|diff| = sqrt(2/pi) SE for normal errors
    diff = np.array([r_mf[k] - r_ex[k] for k in keys])
    expected = np.sqrt(2 / np.pi) * np.mean([se[k] for k in keys])
    assert 0.8 * expected < np.abs(diff).mean() < 1.25 * expected
    # regression (round 7): the sampling record survives in the manifest next to the solver
    diag = json.loads((mf.out_dir / "manifest.json").read_text(encoding="utf-8"))["diagnostics"]
    assert diag["wwt"]["pev_sampling"]["n_samples"] == 300 and "solver" in diag["wwt"]


def test_apy_blocks_from_int8_dosage_equal_the_float_version():
    from abp.core.ssop import apy_blocks_from_dosage
    ped, rng = _ped()
    g = np.sort(rng.choice(ped.n, 90, replace=False))
    M = rng.integers(0, 3, (g.size, 350)).astype(np.int8)
    M[rng.random(M.shape) < 0.02] = -1                         # missing calls
    miss = M < 0
    Mf = np.where(miss, 0, M).astype(float)
    p = np.array([Mf[~miss[:, j], j].mean() / 2 for j in range(M.shape[1])])
    W = centered(Mf, p, miss)
    core = np.sort(rng.choice(g.size, 30, replace=False))
    cols = ped.a_columns(g[core])[g]
    diagA = 1.0 + ped.inbreeding()[g]
    ref = apy_blocks_from_genotypes(W, scaling_d(p), core, "blend", 0.05, 0.01, cols, diagA)
    got = apy_blocks_from_dosage(M, p, core, "blend", 0.05, 0.01, cols, diagA, block=17)
    for a_, b_ in zip(got, ref):
        np.testing.assert_allclose(a_, b_, rtol=1e-12, atol=1e-12)


def test_a_block_equals_columns_of_a():
    from abp.core.ssop import a_block
    ped, rng = _ped()
    rows = np.sort(rng.choice(ped.n, 50, replace=False))
    cols = np.sort(rng.choice(ped.n, 40, replace=False))
    np.testing.assert_allclose(a_block(ped, rows, cols, block=7),
                               ped.a_columns(cols)[rows], atol=1e-13)


@pytest.mark.parametrize("mode", ["explicit", "matrix_free"])
def test_int8_genotype_storage_gives_identical_ebvs(tmp_path, mode):
    from abp.workflows.evaluate import run_evaluation
    g = 'apy_core_size = 150\n' + ('single_step_mode = "matrix_free"\n' if mode == "matrix_free"
                                   else "")
    a = run_evaluation(_spec(tmp_path, g, "f64"), tmp_path / "f64", console=False)
    b = run_evaluation(_spec(tmp_path, g + 'genotype_storage = "int8"\n', "i8"), tmp_path / "i8",
                       console=False)
    e1, e2 = _ebv(a), _ebv(b)
    x = np.array([e1[k] for k in e1])
    np.testing.assert_allclose(np.array([e2[k] for k in e1]), x, atol=1e-8 * np.abs(x).max())
    assert b.manifest["relationship"]["genotype_storage"] == "int8"


def test_orthogonal_reliability_estimator_is_unbiased_and_less_noisy():
    """Round 8: reliability = mean h^2 / (mean h^2 + mean d^2) against the exact
    reliabilities of a small single step, over 60 seeds, next to the round-6 ratio
    estimator 1 - mean d^2 / mean u^2 on the same draws.  Checks: both centred on the
    truth, the orthogonal one less noisy (theory: by sqrt(r)), and its reported SE
    matches the spread across seeds."""
    import scipy.sparse as sp
    from abp.core.genomic import single_step
    from abp.core.ssop import DenseInverseOperator, SingleStepHInverse
    from abp.solvers.blup import RandomTerm
    from abp.solvers.pev_sampling import sampled_pev
    ped, rng = _ped(n=80, seed=9)
    g = np.sort(rng.choice(ped.n, 25, replace=False))
    M = rng.integers(0, 3, (g.size, 300)).astype(float)
    p = M.mean(axis=0) / 2
    W = centered(M, p)
    Gs, _ = apply_g_policy(W @ W.T / scaling_d(p), "blend", "none", ped.a_submatrix(g), 0.05,
                           0.01)
    h_inv = single_step(ped, g, Gs, a22=ped.a_submatrix(g)).h_inv.toarray()
    rec = np.sort(rng.choice(ped.n, 50, replace=False))          # 50 records, 80 animals
    Z = sp.csr_matrix((np.ones(rec.size), (np.arange(rec.size), rec)), shape=(rec.size, ped.n))
    X = sp.csr_matrix(np.ones((rec.size, 1)))
    va, ve = 2.0, 4.0
    # exact: C = [X'X X'Z; Z'X Z'Z + H^-1 ve/va] / ve, PEV = C^uu
    Zd, Xd = Z.toarray(), X.toarray()
    C = np.block([[Xd.T @ Xd, Xd.T @ Zd], [Zd.T @ Xd, Zd.T @ Zd + h_inv * ve / va]]) / ve
    pev = np.diag(np.linalg.inv(C))[1:]
    r_true = 1.0 - pev / (va * np.diag(np.linalg.inv(h_inv)))
    op = SingleStepHInverse(ped.ainv().tocsr(), g,
                            DenseInverseOperator(np.linalg.inv(Gs), np.linalg.cholesky(Gs)),
                            A22InverseOperator(ped.ainv(), g), ped=ped)
    term = [RandomTerm("animal", Z, op, ped.ids, True)]
    est = {"orthogonal": [], "ratio": []}
    se = []
    for seed in range(60):
        for e in est:
            s = sampled_pev(rec.size, X, term, {"animal": va, "residual": ve}, "animal",
                            n_samples=40, seed=seed, tol=1e-12, estimator=e)
            est[e].append(s.reliability)
            if e == "orthogonal":
                se.append(s.reliability_se)
    R = {e: np.array(v) for e, v in est.items()}               # seeds x animals
    sd = {e: (R[e] - r_true).std(axis=0, ddof=1) for e in R}
    for e in R:                                                 # centred: |bias| < 4 MC SE
        bias = (R[e] - r_true).mean(axis=0)
        assert np.abs(bias.mean()) < 4 * np.sqrt(np.mean(sd[e] ** 2) / (60 * ped.n)) + 0.01
    gain = sd["orthogonal"].mean() / sd["ratio"].mean()
    predicted = np.mean(np.sqrt(r_true) * sd["ratio"]) / sd["ratio"].mean()
    assert gain < 0.9 and abs(gain - predicted) < 0.15, (gain, predicted)
    # the reported SE matches the spread across seeds (to 20%)
    assert 0.8 < np.mean(se) / sd["orthogonal"].mean() < 1.2
