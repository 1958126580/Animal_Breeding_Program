"""Metafounders: extended relationships, inverse, determinant, kernels, single
step and Gamma estimation, each checked against an independent reference."""

import numpy as np
import pytest
import scipy.sparse as sp

from abp.core import metafounders as mfm
from abp.core.design import build_fixed_design
from abp.core.genomic import vanraden_g
from abp.core.metafounders import (MetafounderPedigree, estimate_gamma_gls, ml_general_python,
                                   read_gamma_file, single_step_mf)
from abp.core.pedigree import Pedigree, native_kernel_available
from abp.core.upg import GroupAssignment
from abp.errors import ABPError
from abp.solvers.blup import RandomTerm, blup
from abp.solvers.reml import REMLEvaluator
from tests.reference.dense_reference import (blup_v_form, reml_loglik_v_form, reml_score_v_form,
                                             tabular_a_metafounders)

MF = ("MF1", "MF2")
GAMMA = np.array([[0.40, 0.15], [0.15, 0.30]])
# founders from each metafounder, a crossbred, one parent known + one from a
# metafounder, and inbreeding (7 x 8 share ancestors; 10 is a sib mating)
PED = [("1", "MF1", "MF1"), ("2", "MF1", "MF1"), ("3", "MF2", "MF2"), ("4", "1", "MF2"),
       ("5", "3", "2"), ("6", "1", "2"), ("7", "4", "5"), ("8", "4", "6"), ("9", "7", "8"),
       ("10", "7", "6"), ("11", "MF2", "5")]


def _build(records=PED, labels=MF):
    ids = [a for a, _, _ in records]
    ped = Pedigree.from_parent_ids(ids, [s if s in ids else None for _, s, _ in records],
                                   [d if d in ids else None for _, _, d in records])
    pos = {g: k for k, g in enumerate(labels)}
    sg = np.full(ped.n, -1)
    dg = np.full(ped.n, -1)
    for a, s, d in records:
        i = ped.index_of([a])[0]
        sg[i] = pos.get(s, -1)
        dg[i] = pos.get(d, -1)
    return ped, GroupAssignment(tuple(labels), sg, dg)


def _reference(mfp):
    labels, A = tabular_a_metafounders(PED, MF, GAMMA)
    order = [labels.index(x) for x in mfp.labels]
    return A[np.ix_(order, order)]


@pytest.fixture(params=["native", "python"])
def kernel(request, monkeypatch):
    if request.param == "native":
        if not native_kernel_available():
            pytest.skip("native kernel not built")
    else:
        monkeypatch.setenv("ABP_DISABLE_NATIVE", "1")
    return request.param


def test_extended_matrix_inverse_and_logdet_match_tabular_reference(kernel):
    ped, grp = _build()
    mfp = MetafounderPedigree(ped, grp, GAMMA)
    assert mfp.kernel.startswith("native" if kernel == "native" else "python")
    A = _reference(mfp)
    np.testing.assert_allclose(mfp.diag_ext(), np.diag(A), atol=1e-14)
    np.testing.assert_allclose(mfp.a_ext_dense(), A, atol=1e-13)
    np.testing.assert_allclose(mfp.ainv_ext().toarray(), np.linalg.inv(A), atol=1e-11)
    assert mfp.logdet_ext() == pytest.approx(np.linalg.slogdet(A)[1], abs=1e-12)
    # founder self-relationship 1 + gamma/2 and relationship gamma between founders
    i1, i2, i3 = ped.index_of(["1", "2", "3"])
    assert mfp.adiag[i1] == pytest.approx(1 + GAMMA[0, 0] / 2)
    assert A[i1, i2] == pytest.approx(GAMMA[0, 0])
    assert A[i1, i3] == pytest.approx(GAMMA[0, 1])
    idx = ped.index_of(["9", "4", "11"])
    np.testing.assert_allclose(mfp.a_submatrix(idx), A[np.ix_(idx, idx)], atol=1e-13)


def test_zero_inputs_reduce_to_ordinary_meuwissen_luo(kernel):
    """c = e = fext = 0 must give diag(A) = 1 + F of the ordinary pedigree."""
    rng = np.random.default_rng(11)
    n = 3000
    ids = [str(i) for i in range(n)]
    sires, dams = [], []
    for i in range(n):
        if i < 60:
            sires.append(None)
            dams.append(None)
        else:
            s, d = rng.choice(i, 2, replace=False)
            sires.append(str(s) if rng.random() > 0.05 else None)
            dams.append(str(d))
    ped = Pedigree.from_parent_ids(ids, sires, dams)
    z = np.zeros(n)
    a, d, name = mfm.ml_general(ped.sire, ped.dam, z, z, z)
    assert name.startswith("native" if kernel == "native" else "python")
    np.testing.assert_allclose(a, 1.0 + ped.inbreeding(), atol=1e-13)
    np.testing.assert_allclose(d, ped.mendelian_d(), atol=1e-14)


@pytest.mark.skipif(not native_kernel_available(), reason="native kernel not built")
def test_native_and_python_kernels_agree_on_a_random_pedigree():
    rng = np.random.default_rng(5)
    n, k = 4000, 3
    ids = [str(i) for i in range(n)]
    records = []
    labels = tuple(f"M{j}" for j in range(k))
    for i in range(n):
        if i < 90:
            records.append((ids[i], labels[i % k], labels[(i // k) % k]))
        else:
            s, d = rng.choice(i, 2, replace=False)
            records.append((ids[i], ids[s] if rng.random() > 0.1 else labels[rng.integers(k)],
                            ids[d]))
    ped, grp = _build(records, labels)
    gamma = np.array([[0.5, 0.2, 0.1], [0.2, 0.4, 0.05], [0.1, 0.05, 0.3]])
    mfp = MetafounderPedigree(ped, grp, gamma)
    assert mfp.kernel == "native_cpp_ml_general"
    sm, dmf = ped.sire < 0, ped.dam < 0
    gd = np.diag(gamma)
    c = 0.25 * (np.where(sm, gd[np.maximum(grp.sire_group, 0)], 0)
                + np.where(dmf, gd[np.maximum(grp.dam_group, 0)], 0))
    e = np.einsum("ij,ij->i", mfp.Q @ gamma, mfp.Q)
    fext = mfp.adiag - 1.0      # the kernels must reproduce the pre-computed ones exactly
    fext[(~sm) & (~dmf)] = 0.0
    a_py, d_py = ml_general_python(ped.sire, ped.dam, c, e, fext)
    np.testing.assert_allclose(a_py, mfp.adiag, atol=1e-13)
    np.testing.assert_allclose(d_py, mfp.d, atol=1e-14)
    # Colleau product equals the extended inverse's solve
    x = rng.standard_normal(n)
    ext = np.concatenate([x, np.zeros(k)])
    Ai = mfp.ainv_ext().tocsc()
    y = mfp.a_times(x)
    r = Ai @ np.concatenate([y, mfp.gamma @ (mfp.Q.T @ x)]) - ext
    assert np.max(np.abs(r)) < 1e-9


def test_invalid_inputs_are_refused():
    ped, grp = _build()
    with pytest.raises(ABPError) as exc:
        MetafounderPedigree(ped, grp, np.array([[0.4, 0.5], [0.5, 0.3]]))  # not PD
    assert exc.value.code == "ABP-E302"
    with pytest.raises(ABPError, match="non-positive Mendelian"):
        MetafounderPedigree(ped, grp, np.array([[2.2, 0.0], [0.0, 2.2]]))
    bad = GroupAssignment(MF, grp.sire_group.copy(), grp.dam_group.copy())
    bad.dam_group[ped.index_of(["11"])[0]] = -1  # dam of 11 is an animal -> still fine
    bad.sire_group[ped.index_of(["11"])[0]] = -1  # sire of 11 unknown and unassigned
    with pytest.raises(ABPError) as exc:
        MetafounderPedigree(ped, bad, GAMMA)
    assert exc.value.code == "ABP-E205"
    with pytest.raises(ABPError, match="symmetric"):
        mfm.validate_gamma(np.array([[0.4, 0.1], [0.2, 0.3]]), MF)


def test_gamma_file_reader(tmp_path):
    f = tmp_path / "gamma.csv"
    f.write_text("metafounder_1,metafounder_2,gamma\nMF1,MF1,0.4\nMF2,MF1,0.15\nMF2,MF2,0.3\n",
                 encoding="utf-8")
    np.testing.assert_allclose(read_gamma_file(f, MF), GAMMA)
    f.write_text("metafounder_1,metafounder_2,gamma\nMF1,MF1,0.4\nMF2,MF2,0.3\n", encoding="utf-8")
    with pytest.raises(ABPError, match="lacks Gamma entries"):
        read_gamma_file(f, MF)


def test_blup_with_metafounders_equals_v_form_model():
    """Animal and metafounder solutions and PEV from the sparse extended
    inverse equal GLS/BLUP from the marginal model with the tabular matrix."""
    ped, grp = _build()
    mfp = MetafounderPedigree(ped, grp, GAMMA)
    rec = ["4", "5", "6", "7", "8", "9", "10", "11", "5", "9"]
    y = np.random.default_rng(2).normal(20, 3, len(rec))
    sa, se = 2.5, 6.0
    n, k = ped.n, 2
    Z = sp.csr_matrix((np.ones(len(rec)), (np.arange(len(rec)), ped.index_of(rec))),
                      shape=(len(rec), n + k))
    X = build_fixed_design({}, [], True, len(rec)).X
    res = blup(y, X, [RandomTerm("animal", Z, mfp.ainv_ext(), list(mfp.labels), True,
                                 k_diag=mfp.diag_ext(), logdet_k=mfp.logdet_ext())],
               {"animal": sa, "residual": se}, method="dense")
    A = _reference(mfp)
    _, u_ref, pev_ref = blup_v_form(y, np.ones((len(rec), 1)), Z.toarray(), sa * A,
                                    se * np.eye(len(rec)))
    tr = res.terms["animal"]
    np.testing.assert_allclose(tr.solution, u_ref, atol=1e-9)
    np.testing.assert_allclose(tr.pev, np.diag(pev_ref), atol=1e-9)
    np.testing.assert_allclose(tr.reliability, 1 - np.diag(pev_ref) / (sa * np.diag(A)), atol=1e-9)


def test_single_step_with_metafounders_matches_dense_h():
    ped, grp = _build()
    mfp = MetafounderPedigree(ped, grp, GAMMA)
    A = _reference(mfp)
    gi = ped.index_of(["7", "8", "9", "11"])
    A22 = A[np.ix_(gi, gi)]
    rng = np.random.default_rng(4)
    W = rng.standard_normal((4, 30))
    G = 0.7 * A22 + 0.3 * (W @ W.T) / 30         # some SPD G on a comparable scale
    ss = single_step_mf(mfp, gi, G)
    A2 = A[:, gi]
    Ai22 = np.linalg.inv(A22)
    H = A + A2 @ Ai22 @ (G - A22) @ Ai22 @ A2.T
    np.testing.assert_allclose(ss.h_inv.toarray(), np.linalg.inv(H), atol=1e-9)
    np.testing.assert_allclose(ss.h_diag, np.diag(H), atol=1e-11)
    assert ss.logdet_h == pytest.approx(np.linalg.slogdet(H)[1], abs=1e-10)


def _gene_drop(seed, m=4000, n_founders=80, gens=3, per_gen=160):
    """Two metafounder populations with known base frequencies; returns the
    pedigree, groups, genotyped indices, dosages and the empirical true Gamma."""
    rng = np.random.default_rng(seed)
    P = np.vstack([rng.beta(0.6, 0.6, m), rng.beta(2.0, 2.0, m)])
    P[1] = np.clip(0.6 * P[0] + 0.4 * P[1], 0, 1)   # correlated populations
    records, hap = [], {}
    for i in range(n_founders):
        g = i % 2
        hap[str(i)] = (rng.random((2, m)) < P[g]).astype(np.int8)
        records.append((str(i), MF[g], MF[g]))
    prev = [str(i) for i in range(n_founders)]
    nxt = n_founders
    geno_ids = []
    for _ in range(gens):
        cur = []
        for _ in range(per_gen):
            s, d = rng.choice(len(prev), 2, replace=False)
            s, d = prev[s], prev[d]
            pick = lambda h: h[rng.integers(0, 2, m), np.arange(m)]  # unlinked loci
            a = str(nxt)
            nxt += 1
            hap[a] = np.vstack([pick(hap[s]), pick(hap[d])])
            records.append((a, s, d))
            cur.append(a)
        geno_ids += cur
        prev = cur
    ped, grp = _build(records, MF)
    gi = ped.index_of(geno_ids)
    M = np.array([hap[a].sum(0) for a in geno_ids], dtype=float)
    true_gamma = 8 * (P - 0.5) @ (P - 0.5).T / m
    return ped, grp, gi, M, true_gamma


def test_gamma_estimation_recovers_simulated_base_and_correction_reduces_bias():
    errs_raw, errs_cor = [], []
    for seed in range(4):
        ped, grp, gi, M, true_gamma = _gene_drop(seed)
        est = estimate_gamma_gls(ped, grp, gi, M)
        errs_cor.append(est.gamma - true_gamma)
        errs_raw.append(est.gamma_uncorrected - true_gamma)
        assert np.max(np.abs(est.gamma - true_gamma)) < 0.03
    bias_raw = np.mean(errs_raw, axis=0)
    bias_cor = np.mean(errs_cor, axis=0)
    # the uncorrected estimator is inflated on the diagonal by ~4 p(1-p) C
    assert np.all(np.diag(bias_raw) > 0)
    assert np.all(np.abs(np.diag(bias_cor)) < np.abs(np.diag(bias_raw)))


def test_gamma_estimation_with_missing_dosages_and_identifiability():
    ped, grp, gi, M, true_gamma = _gene_drop(9, m=2000)
    full = estimate_gamma_gls(ped, grp, gi, M)
    miss = np.random.default_rng(1).random(M.shape) < 0.05
    est = estimate_gamma_gls(ped, grp, gi, np.where(miss, np.nan, M), miss)
    assert est.missing_iterations > 0
    assert np.max(np.abs(est.gamma - full.gamma)) < 0.01
    # a metafounder without genotyped descendants cannot be estimated
    records = PED + [("12", "MF1", "MF1")]
    labels = ("MF1", "MF2", "MF3")
    records[2] = ("3", "MF3", "MF3")
    ped2, grp2 = _build(records, labels)
    with pytest.raises(ABPError) as exc:
        estimate_gamma_gls(ped2, grp2, ped2.index_of(["1", "2", "12"]), np.ones((3, 10)))
    assert exc.value.code == "ABP-E300"


def test_g05_is_on_the_metafounder_scale():
    """Mean off-diagonal of G05 among unrelated founders of one metafounder
    estimates gamma (Legarra et al. 2015), which the GLS estimate matches."""
    ped, grp, gi, M, true_gamma = _gene_drop(3)
    G05, d = vanraden_g(M, np.full(M.shape[1], 0.5))
    assert d == pytest.approx(M.shape[1] / 2)
    mfp = MetafounderPedigree(ped, grp, estimate_gamma_gls(ped, grp, gi, M).gamma)
    A22 = mfp.a_submatrix(gi)
    off = ~np.eye(gi.size, dtype=bool)
    assert np.mean(G05[off]) == pytest.approx(np.mean(A22[off]), abs=0.02)
    assert np.mean(np.diag(G05)) == pytest.approx(np.mean(np.diag(A22)), abs=0.03)


def test_reml_loglik_and_score_under_metafounders_match_v_form():
    """log|A_ext| enters the REML likelihood; compare with the marginal model."""
    ped, grp = _build()
    mfp = MetafounderPedigree(ped, grp, GAMMA)
    rec = ["4", "5", "6", "7", "8", "9", "10", "11", "2", "3"]
    y = np.random.default_rng(8).normal(0, 2, len(rec))
    Z = sp.csr_matrix((np.ones(len(rec)), (np.arange(len(rec)), ped.index_of(rec))),
                      shape=(len(rec), ped.n + 2))
    X = build_fixed_design({}, [], True, len(rec)).X
    term = RandomTerm("animal", Z, mfp.ainv_ext(), list(mfp.labels), True,
                      k_diag=mfp.diag_ext(), logdet_k=mfp.logdet_ext())
    theta = np.array([1.7, 2.9])
    p = REMLEvaluator(y, X, [term], 2**30).evaluate(theta)
    A = _reference(mfp)
    ZAZ = Z.toarray() @ A @ Z.toarray().T
    ll_ref, _ = reml_loglik_v_form(y, np.ones((len(rec), 1)), [ZAZ, np.eye(len(rec))], theta)
    assert p.loglik == pytest.approx(ll_ref, abs=1e-10)
    np.testing.assert_allclose(
        p.score, reml_score_v_form(y, np.ones((len(rec), 1)), [ZAZ, np.eye(len(rec))], theta),
        atol=1e-10)
