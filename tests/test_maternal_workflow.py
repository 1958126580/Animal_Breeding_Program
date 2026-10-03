"""Maternal animal model with REML or known (co)variances through ``abp run`` (round 13):
example 18 with REML; the same model with the REML estimates given as known values gives
the same EBVs; outputs; spec rules."""

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from abp.core.spec import validate_spec_dict
from abp.errors import ABPError
from abp.workflows.evaluate import run_evaluation

EX18 = Path(__file__).resolve().parents[1] / "examples" / "18_sheep_wwt_maternal_bayes"


def _rows(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_example18_reml_and_known_variances_agree(tmp_path):
    out = run_evaluation(EX18 / "analysis_reml.toml", tmp_path / "reml", console=False)
    t = out.results["traits"]["wwt"]
    r = t["reml"]
    assert r["status"] == "converged" and r["se"] is not None
    vc = t["variance_components"]
    assert set(vc) == {"animal", "maternal", "animal-maternal covariance", "mpe", "residual"}
    sP = sum(vc.values())
    assert t["heritability"] == pytest.approx(vc["animal"] / sP)
    assert t["maternal"]["m2"] == pytest.approx(vc["maternal"] / sP)
    assert t["maternal"]["direct_maternal_correlation"] == pytest.approx(
        vc["animal-maternal covariance"] / math.sqrt(vc["animal"] * vc["maternal"]))
    assert t["solver"]["relative_residual"] < 1e-10
    ebv = _rows(out.out_dir / "ebv_wwt.csv")
    assert len(ebv) == 2108
    assert {"ebv", "reliability", "sep", "mebv", "mreliability", "msep",
            "pev_incl_vc_uncertainty", "mreliability_incl_vc_uncertainty"} <= set(ebv[0])
    for a, b in (("sep", "pev_incl_vc_uncertainty"), ("msep", "mpev_incl_vc_uncertainty")):
        assert all(float(x[b]) >= float(x[a]) ** 2 * (1 - 1e-12) for x in ebv)
    rel = np.array([float(x["reliability"]) for x in ebv])
    mrel = np.array([float(x["mreliability"]) for x in ebv])
    assert np.all((rel >= 0) & (rel <= 1)) and np.all((mrel >= 0) & (mrel <= 1))
    assert len(_rows(out.out_dir / "mpe_wwt.csv")) == 454
    report = (out.out_dir / "report.md").read_text(encoding="utf-8")
    assert "Maternal genetic effect" in report and "m2 =" in report
    # the same model with the REML estimates as known values: identical EBVs
    spec = (EX18 / "analysis_reml.toml").read_text(encoding="utf-8")
    a, b = spec.index("[variances]"), spec.index("[qc]")
    known = (f'[variances]\nmode = "known"\nvalues = {{ animal = [[{vc["animal"]!r}, '
             f'{vc["animal-maternal covariance"]!r}], [{vc["animal-maternal covariance"]!r}, '
             f'{vc["maternal"]!r}]], mpe = {vc["mpe"]!r}, residual = {vc["residual"]!r} }}\n\n')
    spec_k = spec[:a] + known + spec[b:]
    spec_k = spec_k.replace('"../sheep_data/', f'"{(EX18 / "../sheep_data").resolve().as_posix()}/')
    spec_k = spec_k.replace('"data/lambs_maternal.csv"',
                            f'"{(EX18 / "data/lambs_maternal.csv").as_posix()}"')
    p = tmp_path / "known.toml"
    p.write_text(spec_k, encoding="utf-8")
    out_k = run_evaluation(p, tmp_path / "known", console=False)
    assert out_k.results["traits"]["wwt"]["reml"] is None
    ebv_k = _rows(out_k.out_dir / "ebv_wwt.csv")
    for col in ("ebv", "mebv", "reliability", "mreliability"):
        np.testing.assert_allclose([float(x[col]) for x in ebv_k], [float(x[col]) for x in ebv],
                                   rtol=1e-9, atol=1e-10)


BASE = {
    "schema_version": "1",
    "project": {"name": "x", "species": "sheep", "synthetic_data": True},
    "analysis": {"task": "additive_ebv", "target_population": "t",
                 "information_cutoff": "2025-12-31", "genetic_base": "b"},
    "data": {"pedigree": "p.csv", "phenotypes": "y.csv"},
    "traits": [{"name": "w", "unit": "kg"}],
    "model": {"traits": ["w"],
              "random": [{"name": "animal", "kind": "additive", "relationship": "pedigree"},
                         {"name": "mat", "kind": "maternal"},
                         {"name": "mpe", "kind": "iid", "column": "dam"}]},
    "variances": {"mode": "reml"},
}


def test_spec_rules_for_maternal_reml_and_known_variances():
    assert validate_spec_dict(BASE)
    ok_known = dict(BASE, variances={"mode": "known", "values": {
        "animal": [[4.0, -1.0], [-1.0, 2.0]], "mpe": 1.5, "residual": 8.0}})
    assert validate_spec_dict(ok_known)
    bad = [
        dict(BASE, reml={"start": {"animal": 1.0, "mpe": 1.0, "residual": 1.0}}),
        dict(BASE, variances={"mode": "known", "values": {      # additive needs a 2x2 matrix
            "animal": 4.0, "mpe": 1.5, "residual": 8.0}}),
        dict(BASE, variances={"mode": "known", "values": {      # maternal is inside the matrix
            "animal": [[4.0, -1.0], [-1.0, 2.0]], "mat": 2.0, "mpe": 1.5, "residual": 8.0}}),
        dict(BASE, upg={"prefix": "UPG:"}),
    ]
    for b in bad:
        with pytest.raises(ABPError):
            validate_spec_dict(b)
