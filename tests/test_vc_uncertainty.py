"""Kackar-Harville PEV correction for estimated variances: against derivatives
of the independent V-form BLUP, and in the workflow."""

import csv
from pathlib import Path

import numpy as np

from abp.solvers.vc_uncertainty import kackar_harville_delta
from abp.workflows.evaluate import run_evaluation
from tests.reference.dense_reference import blup_v_form, tabular_a
from tests.test_blup import animal_term

ROOT = Path(__file__).resolve().parents[1]


def test_delta_equals_v_form_derivatives():
    from abp.core.design import build_fixed_design
    from abp.core.pedigree import Pedigree
    rng = np.random.default_rng(3)
    n = 40
    ids = [str(i) for i in range(n)]
    s = [None] * 6 + [ids[rng.integers(0, i)] for i in range(6, n)]
    d = [None] * 6 + [ids[rng.integers(0, i)] for i in range(6, n)]
    d = [x if x != sx else None for x, sx in zip(d, s)]
    ped = Pedigree.from_parent_ids(ids, s, d)
    rec = [ids[k] for k in rng.integers(0, n, 70)]
    y = rng.normal(5, 2, 70)
    X = build_fixed_design({}, [], True, 70).X
    term = animal_term(ped, rec)
    vc = {"animal": 1.4, "residual": 3.1}
    cov = np.array([[0.30, -0.12], [-0.12, 0.25]])
    delta = kackar_harville_delta(y, X, [term], vc, cov, ["animal", "residual"], "animal")
    A = tabular_a(ids, s, d)
    order = ped.index_of(ids)
    Z = np.zeros((70, n))
    Z[np.arange(70), [ids.index(a) for a in rec]] = 1

    def u(sa, se):
        return blup_v_form(y, np.ones((70, 1)), Z, sa * A, se * np.eye(70))[1]
    h = 1e-5
    g = np.column_stack([(u(1.4 + h, 3.1) - u(1.4 - h, 3.1)) / (2 * h),
                         (u(1.4, 3.1 + h) - u(1.4, 3.1 - h)) / (2 * h)])
    ref = np.einsum("ik,kl,il->i", g, cov, g)
    np.testing.assert_allclose(delta[order], ref, rtol=1e-5, atol=1e-12)
    assert np.all(delta >= 0)


def test_workflow_reports_pev_including_vc_uncertainty(tmp_path):
    out = run_evaluation(ROOT / "examples" / "02_sheep_wwt_reml" / "analysis.toml",
                         tmp_path / "o", console=False)
    with open(out.out_dir / "ebv_wwt.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    pev = np.array([float(r["pev"]) for r in rows])
    pev_t = np.array([float(r["pev_incl_vc_uncertainty"]) for r in rows])
    rel_t = np.array([float(r["reliability_incl_vc_uncertainty"]) for r in rows])
    assert np.all(pev_t >= pev - 1e-12) and np.any(pev_t > pev)
    assert np.all((rel_t >= 0) & (rel_t <= 1))
    # known variances: no extra columns
    out2 = run_evaluation(ROOT / "examples" / "01_textbook_mrode_3_1" / "analysis.toml",
                          tmp_path / "p", console=False)
    with open(out2.out_dir / "ebv_wwg.csv", encoding="utf-8") as fh:
        assert "pev_incl_vc_uncertainty" not in fh.readline()
