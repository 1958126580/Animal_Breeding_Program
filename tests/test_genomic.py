"""Genomic relationships, GBLUP/SNP-BLUP duality and single-step H^{-1}."""

import numpy as np
import pytest
import scipy.sparse as sp

from abp.core.design import build_fixed_design
from abp.core.genomic import (allele_frequencies, apply_g_policy, minor_allele_frequency,
                              single_step, vanraden_g)
from abp.core.pedigree import Pedigree
from abp.errors import ABPError
from abp.solvers.blup import RandomTerm, blup
from tests.reference.dense_reference import tabular_a

T03 = (["1", "2", "3", "4", "5"], [None, None, "1", "1", "3"], [None, None, "2", "2", "4"])


def test_t01_maf_is_minimum():
    np.testing.assert_allclose(minor_allele_frequency([0.1, 0.0, 0.5, 0.9, 1.0]),
                               [0.1, 0.0, 0.5, 0.1, 0.0], atol=1e-15)


def test_g_hand_example_and_properties():
    M = np.array([[0, 1, 2], [1, 1, 0], [2, 0, 1]], dtype=float)
    p = np.array([0.5, 1 / 3, 0.5])
    G, d = vanraden_g(M, p)
    W = M - 2 * p
    np.testing.assert_allclose(G, W @ W.T / (2 * np.sum(p * (1 - p))), atol=1e-14)
    assert d == pytest.approx(2 * (0.25 + 2 / 9 + 0.25))
    assert np.allclose(G, G.T) and np.linalg.eigvalsh(G).min() > -1e-12
    with pytest.raises(ABPError) as exc:
        vanraden_g(np.array([[0.0, 2.0], [0.0, 2.0]]), np.array([0.0, 1.0]))
    assert exc.value.code == "ABP-E221"


def test_allele_flip_invariance():
    """Spec property 2: flipping the counted allele (M -> 2 - M, p -> 1 - p)."""
    rng = np.random.default_rng(0)
    M = rng.integers(0, 3, (30, 200)).astype(float)
    miss = rng.random(M.shape) < 0.02
    p = allele_frequencies(M, miss)
    G1, _ = vanraden_g(M, p, miss)
    flip = rng.random(200) < 0.5
    M2 = np.where(flip, 2 - M, M)
    p2 = np.where(flip, 1 - p, p)
    G2, _ = vanraden_g(M2, p2, miss)
    np.testing.assert_allclose(G1, G2, atol=1e-12)


def _geno_problem(seed, n=60, m=400):
    rng = np.random.default_rng(seed)
    p0 = rng.uniform(0.1, 0.9, m)
    M = (rng.random((n, m)) < p0).astype(float) + (rng.random((n, m)) < p0)
    return M, rng


def test_gblup_equals_snp_blup_including_unphenotyped():
    M, rng = _geno_problem(1)
    n, m = M.shape
    p = allele_frequencies(M)
    G, d = vanraden_g(M, p)
    Gs, _ = apply_g_policy(G, "ridge", "none", None, 0.0, 1e-3)
    phen = np.arange(40)  # 20 genotyped animals without phenotypes
    y = rng.normal(0, 1, 40)
    X = build_fixed_design({}, [], True, 40).X
    va, ve = 1.5, 2.0
    Z = sp.csr_matrix((np.ones(40), (np.arange(40), phen)), shape=(40, n))
    g_inv = np.linalg.inv(Gs)
    r1 = blup(y, X, [RandomTerm("animal", Z, g_inv, [str(i) for i in range(n)], True,
                                k_diag=np.diag(Gs))], {"animal": va, "residual": ve})
    # SNP-BLUP on the same (ridge-adjusted) covariance: G* = WW'/d + lam I needs an
    # extra iid term with variance va*lam to be exactly equivalent.
    W = M - 2 * p
    Zs = sp.csr_matrix(W[phen])
    terms = [RandomTerm("snp", Zs, sp.identity(m, format="csr"), [f"m{j}" for j in range(m)], False,
                        k_diag=np.ones(m)),
             RandomTerm("res_poly", Z, sp.identity(n, format="csr"), [str(i) for i in range(n)],
                        False, k_diag=np.ones(n))]
    r2 = blup(y, X, terms, {"snp": va / d, "res_poly": va * 1e-3, "residual": ve}, method="dense")
    gebv_snp = W @ r2.terms["snp"].solution + r2.terms["res_poly"].solution
    np.testing.assert_allclose(gebv_snp, r1.terms["animal"].solution, atol=1e-8)
    # back-solved marker effects reproduce the GBLUP of every genotyped animal
    beta = (W.T @ np.linalg.solve(Gs, r1.terms["animal"].solution)) / d
    np.testing.assert_allclose(W @ beta + 1e-3 * np.linalg.solve(Gs, r1.terms["animal"].solution),
                               r1.terms["animal"].solution, atol=1e-8)


def test_pure_gblup_snp_blup_duality_when_g_is_pd():
    """m > n and full rank: G is PD without adjustment, equivalence is exact."""
    M, rng = _geno_problem(2, n=30, m=500)
    p = allele_frequencies(M)
    # centre with frequencies from a *larger* reference so W has full row rank
    p = np.clip(p + rng.normal(0, 0.02, p.size), 0.05, 0.95)
    G, d = vanraden_g(M, p)
    Gs, rec = apply_g_policy(G, "error", "none", None, 0.0, 0.0)
    assert rec.min_eigenvalue_after > 0
    y = rng.normal(size=20)
    X = build_fixed_design({}, [], True, 20).X
    Z = sp.csr_matrix((np.ones(20), (np.arange(20), np.arange(20))), shape=(20, 30))
    r1 = blup(y, X, [RandomTerm("animal", Z, np.linalg.inv(Gs), list(range(30)), True,
                                k_diag=np.diag(Gs))], {"animal": 1.0, "residual": 1.0})
    W = M - 2 * p
    r2 = blup(y, X, [RandomTerm("snp", sp.csr_matrix(W[:20]), sp.identity(M.shape[1], format="csr"),
                                list(range(M.shape[1])), False, k_diag=np.ones(M.shape[1]))],
              {"snp": 1.0 / d, "residual": 1.0}, method="dense")
    np.testing.assert_allclose(W @ r2.terms["snp"].solution, r1.terms["animal"].solution, atol=1e-8)


def test_singular_g_requires_explicit_policy():
    M, _ = _geno_problem(3, n=50, m=30)  # more animals than markers -> singular
    G, _ = vanraden_g(M, allele_frequencies(M))
    with pytest.raises(ABPError) as exc:
        apply_g_policy(G, "error", "none", None, 0.05, 0.01)
    assert exc.value.code == "ABP-E302"
    Gs, rec = apply_g_policy(G, "ridge", "none", None, 0.05, 0.01)
    assert rec.min_eigenvalue_after > 0 and rec.ridge == 0.01


def test_t06_single_step_identities():
    ped = Pedigree.from_parent_ids(*T03)
    idx = ped.index_of(T03[0])
    A = tabular_a(*T03)
    geno = ped.index_of(["3", "4", "5"])
    A22 = A[2:, 2:]
    # (1) G* = A22 collapses H^{-1} to A^{-1}
    ss = single_step(ped, geno, A22.copy())
    np.testing.assert_allclose(ss.h_inv.toarray(), ped.ainv().toarray(), atol=1e-12)
    np.testing.assert_allclose(ss.h_diag, 1 + ped.inbreeding(), atol=1e-12)
    # (2) non-trivial G*: H^{-1} equals the inverse of the explicitly built H
    G2 = 0.8 * A22 + 0.2 * np.eye(3)
    ss2 = single_step(ped, geno, G2)
    A11, A12 = A[:2, :2], A[:2, 2:]
    B = A12 @ np.linalg.inv(A22)
    H = np.block([[A11 + B @ (G2 - A22) @ B.T, B @ G2], [G2 @ B.T, G2]])
    Hinv = ss2.h_inv.toarray()[np.ix_(idx, idx)]
    np.testing.assert_allclose(Hinv, np.linalg.inv(H), atol=1e-12)
    np.testing.assert_allclose(ss2.h_diag[idx], np.diag(H), atol=1e-12)
    assert ss2.logdet_h == pytest.approx(np.linalg.slogdet(H)[1], abs=1e-12)
    # (3) counterexample: the 22-block of A^{-1} is NOT A22^{-1}
    Ainv22 = ped.ainv().toarray()[np.ix_(geno, geno)]
    assert np.max(np.abs(Ainv22 - np.linalg.inv(A22))) > 0.1


def test_tuning_matches_a22_means():
    ped = Pedigree.from_parent_ids(*T03)
    A22 = ped.a_submatrix(ped.index_of(["3", "4", "5"]))
    G = np.array([[1.1, 0.2, 0.4], [0.2, 0.9, 0.5], [0.4, 0.5, 1.3]])
    Gs, rec = apply_g_policy(G, "error", "match_a22", A22, 0.0, 0.0)
    off = ~np.eye(3, dtype=bool)
    assert np.mean(np.diag(Gs)) == pytest.approx(np.mean(np.diag(A22)))
    assert np.mean(Gs[off]) == pytest.approx(np.mean(A22[off]))
    assert rec.tuning_b is not None
