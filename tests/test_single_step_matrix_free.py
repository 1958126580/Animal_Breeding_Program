"""Matrix-free single step (H^-1 as an operator in PCG) against the explicit
construction: A22^-1 operator vs the dense inverse, APY operator vs the dense
APY inverse (incl. the blend policy built from genotypes), and whole
evaluations with identical EBVs."""

import csv
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
