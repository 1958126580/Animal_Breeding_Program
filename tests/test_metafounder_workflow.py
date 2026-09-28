"""Metafounders end to end: spec rules, pedigree codes and default, Gamma file
provenance, outputs against an independent V-form reference, refusals, and the
shipped example 12."""

import csv
from pathlib import Path

import numpy as np
import pytest

from abp.core.spec import validate_spec_dict
from abp.errors import ABPError
from abp.workflows.evaluate import run_evaluation
from tests.reference.dense_reference import blup_v_form, tabular_a_metafounders

ROOT = Path(__file__).resolve().parents[1]
EX12 = ROOT / "examples" / "12_sheep_wwt_single_step_metafounder"


def _read(p):
    with open(p, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _spec(**over):
    d = {"schema_version": "1",
         "project": {"name": "t", "species": "sheep", "synthetic_data": True},
         "analysis": {"task": "additive_ebv", "target_population": "x",
                      "information_cutoff": "2024-12-31", "genetic_base": "metafounders"},
         "data": {"pedigree": "p.csv", "phenotypes": "y.csv"},
         "traits": [{"name": "y", "unit": "kg"}],
         "model": {"traits": ["y"],
                   "random": [{"name": "animal", "kind": "additive",
                               "relationship": "pedigree"}]},
         "variances": {"mode": "known", "values": {"animal": 1.0, "residual": 2.0}},
         "metafounders": {"prefix": "MF:", "gamma_source": "file", "gamma_file": "g.csv",
                          "gamma_provenance": "test values"}}
    d.update(over)
    return d


@pytest.mark.parametrize("over, match", [
    ({"upg": {"prefix": "UPG:", "effect": "random", "variance_ratio": 1.0}}, "either"),
    ({"metafounders": {"prefix": "MF:", "gamma_source": "file", "gamma_file": "g.csv"}},
     "gamma_provenance"),
    ({"metafounders": {"prefix": "MF:", "gamma_source": "genotypes_gls"}}, "needs genotypes"),
    ({"metafounders": {"prefix": "MF:", "default": "BASE", "gamma_source": "file",
                       "gamma_file": "g.csv", "gamma_provenance": "x"}}, "start with"),
    ({"metafounders": {"prefix": "0", "gamma_source": "file", "gamma_file": "g.csv",
                       "gamma_provenance": "x"}}, "unknown_parent_values"),
    ({"variances": {"mode": "bayes"}, "bayes": {"method": "BRR"}}, None),
])
def test_spec_rules(over, match):
    with pytest.raises(ABPError, match=match):
        validate_spec_dict(_spec(**over))


def test_single_step_needs_g05_without_tuning():
    geno = {"pedigree": "p.csv", "phenotypes": "y.csv", "genotypes": "g.csv",
            "marker_map": "m.csv"}
    ss = {"random": [{"name": "animal", "kind": "additive", "relationship": "single_step"}],
          "traits": ["y"]}
    with pytest.raises(ABPError, match="fixed_0.5"):
        validate_spec_dict(_spec(data=geno, model=ss))
    with pytest.raises(ABPError, match="tuning"):
        validate_spec_dict(_spec(data=geno, model=ss,
                                 genomic={"frequency_source": "fixed_0.5",
                                          "tuning": "match_a22"}))
    ok = validate_spec_dict(_spec(data=geno, model=ss, genomic={"frequency_source": "fixed_0.5"}))
    assert ok["metafounders"]["sampling_correction"] is True


PED = [("a", "MF:X", "MF:X"), ("b", "MF:X", "MF:Y"), ("c", "MF:Y", "0"), ("d", "a", "b"),
       ("e", "a", "c"), ("f", "d", "e"), ("g", "d", "MF:Y")]
GAMMA = np.array([[0.5, 0.2], [0.2, 0.35]])
RECORDS = [("d", 11.0), ("e", 9.5), ("f", 12.0), ("g", 10.0), ("c", 13.5), ("b", 8.0), ("a", 10.5)]


def _write_case(tmp_path, default="MF:Y", gamma_rows=None):
    (tmp_path / "p.csv").write_text("id,sire,dam\n" + "\n".join(",".join(r) for r in PED) + "\n",
                                    encoding="utf-8")
    (tmp_path / "y.csv").write_text("id,y\n" + "\n".join(f"{a},{v}" for a, v in RECORDS) + "\n",
                                    encoding="utf-8")
    rows = gamma_rows or [("MF:X", "MF:X", 0.5), ("MF:X", "MF:Y", 0.2), ("MF:Y", "MF:Y", 0.35)]
    (tmp_path / "g.csv").write_text("metafounder_1,metafounder_2,gamma\n"
                                    + "\n".join(f"{a},{b},{g}" for a, b, g in rows) + "\n",
                                    encoding="utf-8")
    mf = {"prefix": "MF:", "gamma_source": "file", "gamma_file": "g.csv",
          "gamma_provenance": "values for a unit test"}
    if default:
        mf["default"] = default
    toml = f'''schema_version = "1"
[project]
name = "mf"
species = "sheep"
synthetic_data = true
[analysis]
task = "additive_ebv"
target_population = "x"
information_cutoff = "2024-12-31"
genetic_base = "metafounders X and Y"
[data]
pedigree = "p.csv"
phenotypes = "y.csv"
[[traits]]
name = "y"
unit = "kg"
[model]
traits = ["y"]
random = [{{ name = "animal", kind = "additive", relationship = "pedigree" }}]
[variances]
mode = "known"
values = {{ animal = 2.0, residual = 3.0 }}
[metafounders]
''' + "\n".join(f'{k} = "{v}"' for k, v in mf.items()) + "\n"
    (tmp_path / "a.toml").write_text(toml, encoding="utf-8")
    return tmp_path / "a.toml"


def test_pedigree_metafounder_run_matches_v_form_reference(tmp_path):
    out = run_evaluation(_write_case(tmp_path), tmp_path / "o", console=False)
    rel = out.manifest["relationship"]
    assert rel["kind"] == "pedigree_mf" and rel["gamma"] == GAMMA.tolist()
    assert rel["gamma_provenance"] == "values for a unit test"
    assert any(i["role"] == "metafounder_gamma" for i in out.manifest["inputs"])
    # the plain unknown dam "0" of c goes to the default metafounder MF:Y
    records = [(a, s, "MF:Y" if d == "0" else d) for a, s, d in PED]
    labels, A = tabular_a_metafounders(records, ("MF:X", "MF:Y"), GAMMA)
    Z = np.zeros((len(RECORDS), len(labels)))
    Z[np.arange(len(RECORDS)), [labels.index(a) for a, _ in RECORDS]] = 1
    y = np.array([v for _, v in RECORDS])
    _, u, pev = blup_v_form(y, np.ones((len(y), 1)), Z, 2.0 * A, 3.0 * np.eye(len(y)))
    ebv = {r["animal"]: r for r in _read(out.out_dir / "ebv_y.csv")}
    assert set(ebv) == {a for a, _, _ in PED}
    for k, lab in enumerate(labels[:len(PED)]):
        assert float(ebv[lab]["ebv"]) == pytest.approx(u[k], abs=1e-9)
        assert float(ebv[lab]["pev"]) == pytest.approx(pev[k, k], abs=1e-9)
        assert float(ebv[lab]["inbreeding"]) == pytest.approx(A[k, k] - 1, abs=1e-12)
    mf = {r["metafounder"]: r for r in _read(out.out_dir / "metafounder_solutions_y.csv")}
    for lab in ("MF:X", "MF:Y"):
        k = labels.index(lab)
        assert float(mf[lab]["solution"]) == pytest.approx(u[k], abs=1e-9)
        assert float(mf[lab]["gamma_self"]) == pytest.approx(A[k, k])
    report = (out.out_dir / "report.md").read_text(encoding="utf-8")
    assert "metafounders MF:X, MF:Y" in report and "values for a unit test" in report


def test_unassigned_unknown_parent_is_refused(tmp_path):
    with pytest.raises(ABPError) as exc:
        run_evaluation(_write_case(tmp_path, default=None), tmp_path / "o", console=False)
    assert exc.value.code == "ABP-E205"
    assert "c" in exc.value.details["animals"]


def test_invalid_gamma_file_is_refused(tmp_path):
    bad = [("MF:X", "MF:X", 0.3), ("MF:X", "MF:Y", 0.5), ("MF:Y", "MF:Y", 0.3)]
    with pytest.raises(ABPError) as exc:
        run_evaluation(_write_case(tmp_path, gamma_rows=bad), tmp_path / "o", console=False)
    assert exc.value.code == "ABP-E302"


def test_example12_single_step_on_metafounder_base(tmp_path):
    out = run_evaluation(EX12 / "analysis.toml", tmp_path / "o", console=False)
    rel = out.manifest["relationship"]
    assert rel["kind"] == "single_step_mf" and rel["metafounders"] == ["MF:BASE"]
    est = rel["gamma_estimate"]
    # the sampling correction lowers gamma; both are recorded
    assert est["gamma"][0][0] < est["gamma_uncorrected"][0][0]
    # G05 and A22 on the metafounder base agree without any rescaling of G
    c = rel["compatibility"]
    assert c["mean_diag_Gstar"] == pytest.approx(c["mean_diag_A22_gamma"], rel=0.03)
    assert c["mean_offdiag_Gstar"] == pytest.approx(c["mean_offdiag_A22_gamma"], rel=0.05)
    t = out.results["traits"]["wwt"]
    assert t["metafounders"]["file"] == "metafounder_solutions_wwt.csv"
    rel_ = [float(r["reliability"]) for r in _read(out.out_dir / "ebv_wwt.csv")]
    assert 0 <= min(rel_) and max(rel_) <= 1
