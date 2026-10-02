"""Multi-trait threshold model in the workflow (``abp run``): outputs, evidence,
withholding of unconverged chains, and the spec rules."""

import csv
import json

import numpy as np
import pytest
from scipy.sparse.linalg import spsolve_triangular

from abp.core.pedigree import Pedigree
from abp.core.spec import validate_spec_dict
from abp.errors import ABPError


def _write(tmp_path, bayes_lines: str, n: int = 150, fixed: str = "",
           method: str = "threshold") -> object:
    rng = np.random.default_rng(8)
    ids = [f"a{i}" for i in range(n)]
    male = np.arange(n) % 2 == 0
    s, d = [], []
    for i in range(n):
        if i < 25:
            s.append(None)
            d.append(None)
        else:
            s.append(ids[rng.choice(np.flatnonzero(male[:i]))])
            d.append(ids[rng.choice(np.flatnonzero(~male[:i]))])
    ped = Pedigree.from_parent_ids(ids, s, d)
    F = spsolve_triangular(ped._l_matrix(), np.sqrt(ped.mendelian_d())[:, None]
                           * rng.standard_normal((n, 2)), lower=True, unit_diagonal=True)
    U = F @ np.linalg.cholesky([[4.0, 0.5], [0.5, 0.4]]).T
    E = rng.standard_normal((n, 2)) @ np.linalg.cholesky([[8.0, 0.8], [0.8, 1.0]]).T
    L = np.array([30.0, 0.0]) + U + E
    wt = L[:, 0]
    score = np.digitize(L[:, 1], [-0.4, 0.5]) + 1
    grp = rng.integers(0, 3, n)
    with open(tmp_path / "ped.csv", "w", encoding="utf-8") as fh:
        fh.write("id,sire,dam,sex\n")
        for a, s_, d_, m_ in zip(ids, s, d, male):
            fh.write(f"{a},{s_ or '0'},{d_ or '0'},{'M' if m_ else 'F'}\n")
    with open(tmp_path / "phe.csv", "w", encoding="utf-8") as fh:
        fh.write("id,grp,wt,score,liab\n")
        for i, a in enumerate(ids):
            w = "NA" if i % 7 == 0 else f"{wt[i]:.3f}"          # some missing weights
            sc = "NA" if male[i] else str(score[i])            # score on females only
            lb = "NA" if male[i] else f"{L[i, 1]:.4f}"
            fh.write(f"{a},g{grp[i]},{w},{sc},{lb}\n")
    spec = tmp_path / "a.toml"
    spec.write_text(f"""schema_version = "1"
[project]
name = "mtt"
species = "sheep"
synthetic_data = true
[analysis]
task = "additive_ebv"
target_population = "test"
information_cutoff = "2025-12-31"
genetic_base = "unknown parents"
[data]
pedigree = "{(tmp_path / 'ped.csv').as_posix()}"
phenotypes = "{(tmp_path / 'phe.csv').as_posix()}"
[data.pedigree_columns]
sex = "sex"
[[traits]]
name = "wt"
unit = "kg"
[[traits]]
name = "score"
unit = "class"
type = "categorical"
[[traits]]
name = "liab"
unit = "sd"
[model]
traits = {'["wt", "score"]' if method == "threshold" else '["wt", "liab"]'}
{fixed}
random = [{{ name = "animal", kind = "additive", relationship = "pedigree" }}]
[variances]
mode = "bayes"
[bayes]
method = "{method}"
{bayes_lines}
""", encoding="utf-8")
    return spec


PRIOR = """variance_prior = "inverse_wishart"
nu = 6.0
prior_covariance = { animal = [[3.0, 0.0], [0.0, 0.3]] }"""


def test_workflow_multitrait_threshold_writes_outputs_and_evidence(tmp_path):
    from abp.workflows.evaluate import run_evaluation
    spec = _write(tmp_path, PRIOR + """
chains = 4
iterations = 3000
burn_in = 500
thin = 2
max_iterations = 12000
rhat_max = 1.05
ess_min = 100""", fixed='fixed = [{ column = "grp", type = "factor", traits = ["wt"] }]')
    out = run_evaluation(spec, tmp_path / "o", console=False)
    assert out.status == "passed"
    r = out.results["traits"]
    assert set(r) == {"wt", "score"}
    assert r["score"]["unit"] == "liability" and r["score"]["bayes"]["converged"]
    assert r["score"]["bayes"]["variance_prior"].startswith("inverse Wishart")
    assert r["wt"]["n_records"] == 150 - len(range(0, 150, 7))
    assert r["score"]["n_records"] == 75
    rows = list(csv.DictReader(open(out.out_dir / "ebv_multitrait.csv", encoding="utf-8")))
    assert len(rows) == 150 and {"ebv_wt", "ebv_score", "reliability_score"} <= set(rows[0])
    rel = np.array([float(x["reliability_score"]) for x in rows])
    assert np.all((rel >= 0) & (rel <= 1))
    fe = list(csv.DictReader(open(out.out_dir / "fixed_effects_wt.csv", encoding="utf-8")))
    assert {x["term"] for x in fe} == {"intercept", "grp"}
    diag = json.loads((out.out_dir / "mcmc_diagnostics_multitrait.json").read_text("utf-8"))
    assert diag["categorical_trait"] == "score" and diag["converged"]
    assert "rG_0_1" in diag["summaries"] and diag["ebv_diagnostics"]["max_rhat"] < 1.05
    assert (out.out_dir / "mcmc_trace_multitrait.csv").exists()
    report = (out.out_dir / "report.md").read_text(encoding="utf-8")
    assert "Threshold-model Gibbs sampler" in report


def test_workflow_multitrait_threshold_withholds_unconverged_chains(tmp_path):
    from abp.workflows.evaluate import run_evaluation
    spec = _write(tmp_path, PRIOR + """
chains = 2
iterations = 40
burn_in = 10
thin = 1
max_iterations = 40""")
    with pytest.raises(ABPError) as e:
        run_evaluation(spec, tmp_path / "o", console=False)
    assert e.value.code == "ABP-E405"
    failed = [p for p in tmp_path.iterdir() if p.name.startswith("o.failed")]
    assert failed and (failed[0] / "mcmc_diagnostics_multitrait.json").exists()


def test_workflow_bayesian_multitrait_linear_model(tmp_path):
    """bayes.method = "multitrait": continuous traits, (co)variances sampled by Gibbs;
    residual_groups fixes the residual covariance at exactly 0."""
    from abp.workflows.evaluate import run_evaluation
    spec = _write(tmp_path, PRIOR.replace("0.3]]", "0.4]]") + """
chains = 4
iterations = 2000
burn_in = 400
thin = 2
max_iterations = 8000
rhat_max = 1.05
ess_min = 100""", method="multitrait")
    out = run_evaluation(spec, tmp_path / "o", console=False)
    assert out.status == "passed"
    r = out.results["traits"]
    assert set(r) == {"wt", "liab"} and r["liab"]["unit"] == "sd"
    assert r["wt"]["bayes"]["method"] == "multitrait" and r["wt"]["bayes"]["thresholds"] is None
    assert r["liab"]["n_records"] == 75
    diag = json.loads((out.out_dir / "mcmc_diagnostics_multitrait.json").read_text("utf-8"))
    assert diag["categorical_trait"] is None and diag["converged"]
    assert diag["model"] == "Bayesian multi-trait linear model"
    assert "IW(E'E" in diag["priors"]["R0"] and diag["residual_groups"] is None
    assert abs(diag["R0"]["mean"][0][1]) > 0
    report = (out.out_dir / "report.md").read_text(encoding="utf-8")
    assert "Bayesian multi-trait linear model, Gibbs sampler" in report

    spec = _write(tmp_path, PRIOR.replace("0.3]]", "0.4]]") + """
residual_groups = { wt = 1, liab = 2 }
chains = 4
iterations = 2000
burn_in = 400
thin = 2
max_iterations = 8000
rhat_max = 1.05
ess_min = 100""", method="multitrait")
    out = run_evaluation(spec, tmp_path / "o2", console=False)
    diag = json.loads((out.out_dir / "mcmc_diagnostics_multitrait.json").read_text("utf-8"))
    assert diag["residual_groups"] == [["wt"], ["liab"]]
    assert diag["R0"]["mean"][0][1] == 0.0 and "fixed at 0" in diag["priors"]["R0"]
    assert "['liab']: flat, R0_BB | E_B ~ IW" in diag["priors"]["R0"]


def test_spec_rules_for_the_multitrait_threshold_model():
    base = {
        "schema_version": "1",
        "project": {"name": "x", "species": "sheep", "synthetic_data": True},
        "analysis": {"task": "additive_ebv", "target_population": "t",
                     "information_cutoff": "2025-12-31", "genetic_base": "b"},
        "data": {"pedigree": "p.csv", "phenotypes": "y.csv"},
        "traits": [{"name": "w", "unit": "kg"}, {"name": "s", "unit": "c",
                                                  "type": "categorical"}],
        "model": {"traits": ["w", "s"],
                  "random": [{"name": "animal", "kind": "additive",
                              "relationship": "pedigree"}]},
        "variances": {"mode": "bayes"},
        "bayes": {"method": "threshold"},
    }
    assert validate_spec_dict(base)
    iw = dict(base, bayes={"method": "threshold", "variance_prior": "inverse_wishart",
                           "nu": 5.0, "prior_covariance": {"animal": [[1.0, 0.0],
                                                                      [0.0, 0.2]]}})
    assert validate_spec_dict(iw)
    bad = [
        dict(base, variances={"mode": "known", "values": {"animal": [[1, 0], [0, 1]],
                                                          "residual": [[1, 0], [0, 1]]}}),
        dict(base, traits=[{"name": "w", "unit": "kg", "type": "categorical"},
                           {"name": "s", "unit": "c", "type": "categorical"}]),
        dict(base, bayes={"method": "threshold", "variance_prior": "inverse_wishart"}),
        dict(base, bayes={"method": "threshold", "variance_prior": "inverse_wishart",
                          "prior_covariance": {"animal": [[1.0, 2.0], [2.0, 1.0]]}}),
        dict(base, bayes={"method": "threshold", "variance_prior": "scaled_inv_chi2",
                          "prior_variances": {"animal": 0.2}}),
        dict(base, model=dict(base["model"], random=base["model"]["random"]
                              + [{"name": "pe", "kind": "iid"},
                                 {"name": "pe2", "kind": "iid", "column": "id"}])),
    ]
    for b in bad:
        with pytest.raises(ABPError):
            validate_spec_dict(b)
    # round 10: one permanent-environment term is allowed in the multi-trait Gibbs models
    assert validate_spec_dict(dict(base, model=dict(base["model"], random=base["model"]["random"]
                                                    + [{"name": "pe", "kind": "iid"}])))
    # bayes.method = "multitrait" and residual_groups
    lin = dict(base, traits=[{"name": "w", "unit": "kg"}, {"name": "s", "unit": "c"}],
               bayes={"method": "multitrait"})
    assert validate_spec_dict(lin)
    assert validate_spec_dict(dict(lin, bayes={"method": "multitrait",
                                               "residual_groups": {"w": 1, "s": 2}}))
    assert validate_spec_dict(dict(base, bayes={"method": "threshold",
                                                "residual_groups": {"w": 1, "s": 2}}))
    bad = [
        dict(base, bayes={"method": "multitrait"}),                   # categorical trait
        dict(lin, model=dict(lin["model"], traits=["w"])),            # one trait
        dict(lin, bayes={"method": "multitrait", "residual_groups": {"w": 1, "s": 1}}),
        dict(lin, bayes={"method": "multitrait", "residual_groups": {"w": 1}}),
        dict(lin, bayes={"method": "multitrait", "residual_groups": {"w": 1, "s": 1.5}}),
    ]
    for b in bad:
        with pytest.raises(ABPError):
            validate_spec_dict(b)


@pytest.mark.parametrize("name", ["15_sheep_wwt_nlb1_threshold", "16_sheep_multitrait_bayes",
                                  "17_sheep_ewe_repeated_bayes", "18_sheep_wwt_maternal_bayes",
                                  "18_sheep_wwt_maternal_bayes/analysis_equal_prior.toml"])
def test_examples_15_16_spec_and_data_validate(name):
    """Examples 15-18 (and the weak-prior variant of example 18) pass spec and data
    validation (the full runs take minutes and are part of the examples log)."""
    from pathlib import Path

    from abp.cli import main
    ex = Path(__file__).resolve().parents[1] / "examples" / name
    assert main(["validate", str(ex if ex.suffix == ".toml" else ex / "analysis.toml")]) == 0


def _write_repeated(tmp_path, bayes_lines: str, n: int = 120, n_rec: int = 3):
    """Two continuous traits, n_rec records per non-founder animal, a permanent
    environment shared by an animal's records."""
    rng = np.random.default_rng(21)
    ids = [f"a{i}" for i in range(n)]
    male = np.arange(n) % 2 == 0
    s, d = [], []
    for i in range(n):
        if i < 20:
            s.append(None)
            d.append(None)
        else:
            s.append(ids[rng.choice(np.flatnonzero(male[:i]))])
            d.append(ids[rng.choice(np.flatnonzero(~male[:i]))])
    ped = Pedigree.from_parent_ids(ids, s, d)
    F = spsolve_triangular(ped._l_matrix(), np.sqrt(ped.mendelian_d())[:, None]
                           * rng.standard_normal((n, 2)), lower=True, unit_diagonal=True)
    U = F @ np.linalg.cholesky([[1.0, 0.3], [0.3, 0.6]]).T
    PE = rng.standard_normal((n, 2)) @ np.linalg.cholesky([[0.5, 0.1], [0.1, 0.4]]).T
    with open(tmp_path / "ped.csv", "w", encoding="utf-8") as fh:
        fh.write("id,sire,dam,sex\n")
        for a, s_, d_, m_ in zip(ids, s, d, male):
            fh.write(f"{a},{s_ or '0'},{d_ or '0'},{'M' if m_ else 'F'}\n")
    with open(tmp_path / "phe.csv", "w", encoding="utf-8") as fh:
        fh.write("rec,id,parity,y1,y2\n")
        k = 0
        for i in range(n):
            for p in range(n_rec):
                e = rng.standard_normal(2) @ np.linalg.cholesky([[1.5, 0.3], [0.3, 1.0]]).T
                y = np.array([10.0, 4.0]) + 0.3 * p + U[i] + PE[i] + e
                y2 = "NA" if rng.random() < 0.2 else f"{y[1]:.4f}"
                fh.write(f"r{k},{ids[i]},p{p + 1},{y[0]:.4f},{y2}\n")
                k += 1
    spec = tmp_path / "rep.toml"
    spec.write_text(f"""schema_version = "1"
[project]
name = "mtpe"
species = "sheep"
synthetic_data = true
[analysis]
task = "additive_ebv"
target_population = "test"
information_cutoff = "2025-12-31"
genetic_base = "unknown parents"
[data]
pedigree = "{(tmp_path / 'ped.csv').as_posix()}"
phenotypes = "{(tmp_path / 'phe.csv').as_posix()}"
[data.pedigree_columns]
sex = "sex"
[data.phenotype_columns]
record_id = "rec"
[[traits]]
name = "y1"
unit = "kg"
[[traits]]
name = "y2"
unit = "kg"
[model]
traits = ["y1", "y2"]
fixed = [{{ column = "parity", type = "factor" }}]
random = [{{ name = "animal", kind = "additive", relationship = "pedigree" }},
          {{ name = "pe", kind = "iid" }}]
[variances]
mode = "bayes"
[bayes]
method = "multitrait"
{bayes_lines}
""", encoding="utf-8")
    return spec


def test_workflow_multitrait_with_permanent_environment(tmp_path):
    """Round 10: repeated records with a permanent-environment term in the Bayesian
    multi-trait linear model; P0 summaries, c2, pe solutions and the IW priors for the
    pe and residual matrices are written."""
    from abp.workflows.evaluate import run_evaluation
    spec = _write_repeated(tmp_path, """variance_prior = "inverse_wishart"
nu = 6.0
prior_covariance = { animal = [[1.0, 0.0], [0.0, 0.5]], pe = [[0.5, 0.0], [0.0, 0.5]], residual = [[1.5, 0.0], [0.0, 1.0]] }
chains = 4
iterations = 2000
burn_in = 400
thin = 2
max_iterations = 8000
rhat_max = 1.05
ess_min = 100""")
    out = run_evaluation(spec, tmp_path / "o", console=False)
    assert out.status == "passed"
    r = out.results["traits"]
    assert r["y1"]["n_records"] == 360 and "pe" in r["y1"]["variance_components"]
    diag = json.loads((out.out_dir / "mcmc_diagnostics_multitrait.json").read_text("utf-8"))
    pe = diag["permanent_environment"]
    assert pe["levels"] == 120 and pe["prior"].startswith("inverse Wishart")
    assert diag["priors"]["R0"].startswith("inverse Wishart")
    assert "P0_0_1" in diag["summaries"] and "c2_0" in diag["derived"]
    rows = list(csv.DictReader(open(out.out_dir / "pe_multitrait.csv", encoding="utf-8")))
    assert len(rows) == 120 and {"pe_y1", "pe_y2"} <= set(rows[0])


def test_spec_rules_for_permanent_environment_and_r0_prior():
    base = {
        "schema_version": "1",
        "project": {"name": "x", "species": "sheep", "synthetic_data": True},
        "analysis": {"task": "additive_ebv", "target_population": "t",
                     "information_cutoff": "2025-12-31", "genetic_base": "b"},
        "data": {"pedigree": "p.csv", "phenotypes": "y.csv"},
        "traits": [{"name": "w", "unit": "kg"}, {"name": "s", "unit": "c"}],
        "model": {"traits": ["w", "s"],
                  "random": [{"name": "animal", "kind": "additive", "relationship": "pedigree"},
                             {"name": "pe", "kind": "iid"}]},
        "variances": {"mode": "bayes"},
        "bayes": {"method": "multitrait"},
    }
    assert validate_spec_dict(base)
    I2 = [[1.0, 0.0], [0.0, 1.0]]
    assert validate_spec_dict(dict(base, bayes={
        "method": "multitrait", "variance_prior": "inverse_wishart", "nu": 5.0,
        "prior_covariance": {"animal": I2, "pe": I2, "residual": I2}}))
    cat = dict(base, traits=[{"name": "w", "unit": "kg"},
                             {"name": "s", "unit": "c", "type": "categorical"}])
    ok_cat = dict(cat, bayes={"method": "threshold", "variance_prior": "inverse_wishart",
                              "nu": 5.0, "residual_groups": {"w": 1, "s": 2},
                              "prior_covariance": {"animal": I2, "residual": I2}})
    assert validate_spec_dict(ok_cat)
    bad = [
        dict(base, variances={"mode": "reml"}, bayes=None),            # pe needs the Gibbs model
        dict(base, bayes={"method": "multitrait", "variance_prior": "inverse_wishart",
                          "nu": 5.0, "prior_covariance": {"pe": I2}}),   # additive missing
        dict(base, bayes={"method": "multitrait", "variance_prior": "inverse_wishart",
                          "nu": 5.0, "prior_covariance": {"animal": I2, "other": I2}}),
        dict(base, model=dict(base["model"], random=base["model"]["random"]
                              + [{"name": "pe2", "kind": "iid", "column": "id"}])),
        dict(cat, bayes={"method": "threshold", "variance_prior": "inverse_wishart",
                         "nu": 5.0, "prior_covariance": {"animal": I2, "residual": I2}}),
    ]
    for b in bad:
        b = {k: v for k, v in b.items() if v is not None}
        with pytest.raises(ABPError):
            validate_spec_dict(b)


def test_workflow_maternal_effects_single_trait(tmp_path):
    """Round 11: one trait with direct and maternal genetic effects (dams from the
    pedigree): mebv/mreliability columns, maternal variance, direct-maternal
    correlation and m2 are reported; the run goes through the Gibbs workflow."""
    from abp.workflows.evaluate import run_evaluation
    rng = np.random.default_rng(4)
    n = 180
    ids = [f"a{i}" for i in range(n)]
    male = np.arange(n) % 2 == 0
    s, d = [], []
    for i in range(n):
        if i < 30:
            s.append(None)
            d.append(None)
        else:
            s.append(ids[rng.choice(np.flatnonzero(male[:i]))])
            d.append(ids[rng.choice(np.flatnonzero(~male[:i]))])
    ped = Pedigree.from_parent_ids(ids, s, d)
    F = spsolve_triangular(ped._l_matrix(), np.sqrt(ped.mendelian_d())[:, None]
                           * rng.standard_normal((n, 2)), lower=True, unit_diagonal=True)
    U = F @ np.linalg.cholesky([[1.0, -0.2], [-0.2, 0.6]]).T
    with open(tmp_path / "ped.csv", "w", encoding="utf-8") as fh:
        fh.write("id,sire,dam,sex\n")
        for a, s_, d_, m_ in zip(ids, s, d, male):
            fh.write(f"{a},{s_ or '0'},{d_ or '0'},{'M' if m_ else 'F'}\n")
    with open(tmp_path / "phe.csv", "w", encoding="utf-8") as fh:
        fh.write("id,sex,wt\n")
        for i in range(30, n):
            k = ped.index_of([ids[i]])[0]
            dam = int(ped.dam[k])
            y = 25 + U[k, 0] + (U[dam, 1] if dam >= 0 else 0.0) + 1.1 * rng.standard_normal()
            fh.write(f"{ids[i]},{'M' if male[i] else 'F'},{y:.3f}\n")
    spec = tmp_path / "mat.toml"
    spec.write_text(f"""schema_version = "1"
[project]
name = "mat"
species = "sheep"
synthetic_data = true
[analysis]
task = "additive_ebv"
target_population = "test"
information_cutoff = "2025-12-31"
genetic_base = "unknown parents"
[data]
pedigree = "{(tmp_path / 'ped.csv').as_posix()}"
phenotypes = "{(tmp_path / 'phe.csv').as_posix()}"
[data.pedigree_columns]
sex = "sex"
[[traits]]
name = "wt"
unit = "kg"
[model]
traits = ["wt"]
fixed = [{{ column = "sex", type = "factor" }}]
random = [{{ name = "animal", kind = "additive", relationship = "pedigree" }},
          {{ name = "mat", kind = "maternal" }}]
[variances]
mode = "bayes"
[bayes]
method = "multitrait"
chains = 4
iterations = 1200
burn_in = 300
thin = 1
max_iterations = 4800
rhat_max = 1.05
ess_min = 100
""", encoding="utf-8")
    out = run_evaluation(spec, tmp_path / "o", console=False)
    assert out.status == "passed"
    r = out.results["traits"]["wt"]
    assert "mat" in r["variance_components"] and "maternal" in r
    assert -1 <= r["maternal"]["direct_maternal_correlation"]["mean"] <= 1
    rows = list(csv.DictReader(open(out.out_dir / "ebv_multitrait.csv", encoding="utf-8")))
    assert {"ebv_wt", "mebv_wt", "mreliability_wt", "msep_wt"} <= set(rows[0])
    diag = json.loads((out.out_dir / "mcmc_diagnostics_multitrait.json").read_text("utf-8"))
    assert diag["genetic_effects"] == ["animal:wt", "mat:wt"]
    assert diag["model"] == "Bayesian linear animal model with maternal genetic effects"


def test_spec_rules_for_maternal_effects():
    base = {
        "schema_version": "1",
        "project": {"name": "x", "species": "sheep", "synthetic_data": True},
        "analysis": {"task": "additive_ebv", "target_population": "t",
                     "information_cutoff": "2025-12-31", "genetic_base": "b"},
        "data": {"pedigree": "p.csv", "phenotypes": "y.csv"},
        "traits": [{"name": "w", "unit": "kg"}],
        "model": {"traits": ["w"],
                  "random": [{"name": "animal", "kind": "additive", "relationship": "pedigree"},
                             {"name": "mat", "kind": "maternal"}]},
        "variances": {"mode": "bayes"},
        "bayes": {"method": "multitrait"},
    }
    assert validate_spec_dict(base)
    G2 = [[1.0, 0.0], [0.0, 0.5]]
    assert validate_spec_dict(dict(base, bayes={"method": "multitrait",
                                                "variance_prior": "inverse_wishart",
                                                "nu": 4.0, "prior_covariance": {"animal": G2}}))
    bad = [
        dict(base, variances={"mode": "reml"}, bayes=None),                  # REML: refused
        dict(base, bayes={"method": "multitrait", "variance_prior": "inverse_wishart",
                          "nu": 4.0, "prior_covariance": {"animal": [[1.0]]}}),  # not 2x2
        dict(base, bayes={"method": "multitrait", "variance_prior": "inverse_wishart",
                          "nu": 4.0, "prior_covariance": {"animal": G2, "mat": [[1.0]]}}),
        dict(base, model=dict(base["model"], random=base["model"]["random"]
                              + [{"name": "mat2", "kind": "maternal"}])),
        dict(base, model=dict(base["model"], random=[
            {"name": "animal", "kind": "additive", "relationship": "pedigree"},
            {"name": "mat", "kind": "maternal", "column": "dam"}])),
        dict(base, model=dict(base["model"], random=[
            {"name": "animal", "kind": "additive", "relationship": "genomic"},
            {"name": "mat", "kind": "maternal"}])),
    ]
    for b in bad:
        b = {k: v for k, v in b.items() if v is not None}
        with pytest.raises(ABPError):
            validate_spec_dict(b)
