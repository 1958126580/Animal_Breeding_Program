"""APY: the inverse equals the dense inverse of the implied G_APY, reduces to
G^-1 with all animals in the core, the log-determinant, single step with APY
against an explicit dense H, singular-G handling, refusals and the workflow."""

import numpy as np
import pytest

from abp.core.genomic import apy_inverse, single_step, vanraden_g
from abp.core.pedigree import Pedigree
from abp.errors import ABPError


def _g(n=60, m=400, seed=1):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.1, 0.9, m)
    M = (rng.random((n, m)) < p).astype(float) + (rng.random((n, m)) < p)
    G, _ = vanraden_g(M, p)
    return 0.95 * G + 0.05 * np.eye(n)


def test_apy_inverse_equals_inverse_of_implied_g_and_logdet():
    G = _g()
    core = np.arange(0, 60, 3)
    r = apy_inverse(G, core)
    np.testing.assert_allclose(r.g_inv, np.linalg.inv(r.g_apy), atol=1e-9)
    assert r.logdet == pytest.approx(np.linalg.slogdet(r.g_apy)[1], abs=1e-9)
    nonc = np.setdiff1d(np.arange(60), core)
    # G_APY keeps core blocks and non-core diagonals
    np.testing.assert_allclose(r.g_apy[np.ix_(core, core)], G[np.ix_(core, core)])
    np.testing.assert_allclose(r.g_apy[np.ix_(core, nonc)], G[np.ix_(core, nonc)])
    np.testing.assert_allclose(np.diag(r.g_apy), np.diag(G))


def test_all_animals_in_core_reproduce_the_exact_inverse():
    G = _g(30)
    r = apy_inverse(G, np.arange(29))       # one non-core animal: still exact (m = Schur complement)
    np.testing.assert_allclose(r.g_inv, np.linalg.inv(G), atol=1e-9)


def test_singular_g_is_handled_when_core_is_full_rank():
    """More genotyped animals than markers: G is singular, APY with a core
    below its rank gives a proper G_APY without blending."""
    rng = np.random.default_rng(3)
    n, m = 80, 40
    p = rng.uniform(0.2, 0.8, m)
    M = (rng.random((n, m)) < p).astype(float) + (rng.random((n, m)) < p)
    G, _ = vanraden_g(M, p)
    assert np.linalg.matrix_rank(G) < n
    r = apy_inverse(G, np.arange(25))
    assert np.all(np.linalg.eigvalsh(r.g_apy) > 0)
    with pytest.raises(ABPError) as exc:
        apy_inverse(G, np.arange(60))        # core beyond the rank: G_cc singular
    assert exc.value.code == "ABP-E302"


def test_single_step_with_apy_matches_explicit_h():
    ids = [str(i) for i in range(12)]
    s = [None, None, None, None, "0", "0", "2", "2", "4", "6", "4", "6"]
    d = [None, None, None, None, "1", "1", "3", "3", "5", "7", "7", "5"]
    ped = Pedigree.from_parent_ids(ids, s, d)
    gi = ped.index_of(ids[4:])
    A = ped.a_dense()
    A22 = A[np.ix_(gi, gi)]
    rng = np.random.default_rng(2)
    W = rng.standard_normal((8, 30))
    G = 0.6 * A22 + 0.4 * W @ W.T / 30
    r = apy_inverse(G, np.array([0, 2, 4]))
    ss = single_step(ped, gi, r.g_apy, a22=A22, g_inverse=(r.g_inv, r.logdet))
    A2 = A[:, gi]
    Ai22 = np.linalg.inv(A22)
    H = A + A2 @ Ai22 @ (r.g_apy - A22) @ Ai22 @ A2.T
    np.testing.assert_allclose(ss.h_inv.toarray(), np.linalg.inv(H), atol=1e-8)
    np.testing.assert_allclose(ss.h_diag, np.diag(H), atol=1e-10)
    assert ss.logdet_h == pytest.approx(np.linalg.slogdet(H)[1], abs=1e-9)


def test_workflow_example05_with_apy(tmp_path):
    from pathlib import Path
    from abp.workflows.evaluate import run_evaluation
    root = Path(__file__).resolve().parents[1]
    txt = (root / "examples" / "05_sheep_wwt_single_step" / "analysis.toml").read_text(
        encoding="utf-8").replace("../sheep_data", (root / "examples" / "sheep_data").as_posix())
    txt = txt.replace('blend_alpha = 0.05', 'blend_alpha = 0.05\napy_core_size = 300')
    txt = txt.replace('mode = "reml"', 'mode = "known"\nvalues = { animal = 3.5, residual = 12.5 }')
    (tmp_path / "a.toml").write_text(txt, encoding="utf-8")
    out = run_evaluation(tmp_path / "a.toml", tmp_path / "o", console=False)
    apy = out.manifest["relationship"]["apy"]
    assert apy["n_core"] == 300 and len(apy["core_ids"]) == 300 and apy["min_m"] > 0
    # the same run without APY ranks the candidates almost identically
    (tmp_path / "b.toml").write_text(txt.replace("apy_core_size = 300", ""), encoding="utf-8")
    ref = run_evaluation(tmp_path / "b.toml", tmp_path / "p", console=False)
    import csv
    e1 = {r["animal"]: float(r["ebv"]) for r in csv.DictReader(open(out.out_dir / "ebv_wwt.csv"))}
    e2 = {r["animal"]: float(r["ebv"]) for r in csv.DictReader(open(ref.out_dir / "ebv_wwt.csv"))}
    ks = sorted(e1)
    assert np.corrcoef([e1[k] for k in ks], [e2[k] for k in ks])[0, 1] > 0.98
