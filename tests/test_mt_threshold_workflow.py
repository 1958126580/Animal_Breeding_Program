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


def _write(tmp_path, bayes_lines: str, n: int = 150, fixed: str = "") -> object:
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
        fh.write("id,grp,wt,score\n")
        for i, a in enumerate(ids):
            w = "NA" if i % 7 == 0 else f"{wt[i]:.3f}"          # some missing weights
            sc = "NA" if male[i] else str(score[i])            # score on females only
            fh.write(f"{a},g{grp[i]},{w},{sc}\n")
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
[model]
traits = ["wt", "score"]
{fixed}
random = [{{ name = "animal", kind = "additive", relationship = "pedigree" }}]
[variances]
mode = "bayes"
[bayes]
method = "threshold"
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
                              + [{"name": "pe", "kind": "iid"}])),
    ]
    for b in bad:
        with pytest.raises(ABPError):
            validate_spec_dict(b)


def test_example15_spec_and_data_validate():
    """Example 15 (weaning weight + first-parity litter size) passes spec and data
    validation (the full run takes minutes and is part of the examples log)."""
    from pathlib import Path

    from abp.cli import main
    ex = Path(__file__).resolve().parents[1] / "examples" / "15_sheep_wwt_nlb1_threshold"
    assert main(["validate", str(ex / "analysis.toml")]) == 0
