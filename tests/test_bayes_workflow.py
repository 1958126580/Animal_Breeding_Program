"""Bayesian marker regression through the workflow: outputs, gating, spec rules."""

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from abp.cli import main
from abp.errors import ABPError
from abp.workflows.evaluate import run_evaluation

ROOT = Path(__file__).resolve().parents[1]
EX10 = ROOT / "examples" / "10_sheep_fec_bayesc" / "analysis.toml"


def _copy_spec(tmp_path, text_edit=None):
    d = tmp_path / "ex"
    d.mkdir()
    text = EX10.read_text(encoding="utf-8").replace("../sheep_data", str(ROOT / "examples" / "sheep_data").replace("\\", "/"))
    if text_edit:
        text = text_edit(text)
    p = d / "analysis.toml"
    p.write_text(text, encoding="utf-8")
    return p


@pytest.mark.slow
def test_example10_converges_and_writes_outputs(tmp_path):
    out = run_evaluation(EX10, tmp_path / "o", console=False)
    t = out.results["traits"]["fec"]
    assert t["bayes"]["converged"] and t["bayes"]["kernel"] in ("native_cpp", "python_reference")
    assert 0.05 < t["heritability"] < 0.8
    for f in ("ebv_fec.csv", "marker_effects_fec.csv", "mcmc_diagnostics_fec.json"):
        assert (out.out_dir / f).exists()
    diag = json.loads((out.out_dir / "mcmc_diagnostics_fec.json").read_text(encoding="utf-8"))
    assert all(v["rhat"] < 1.01 for k, v in diag["summaries"].items() if k != "n_nonzero")
    assert diag["gebv_diagnostics"]["max_rhat"] < 1.01
    assert len(set(diag["chain_seeds"])) == 4
    # traces: one row per chain and saved draw, one column per monitored scalar
    with open(out.out_dir / "mcmc_trace_fec.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 4 * diag["draws_per_chain"]
    assert set(diag["summaries"]) <= set(rows[0])
    h2 = np.array([float(r["h2"]) for r in rows])
    assert np.mean(h2) == pytest.approx(diag["summaries"]["h2"]["mean"], rel=1e-12)
    pp = diag["posterior_predictive"]["statistics"]
    assert set(pp) == {"sd", "skewness", "min", "max"}
    assert all(0.0 <= v["p_value"] <= 1.0 for v in pp.values())
    assert 0.01 < pp["sd"]["p_value"] < 0.99      # the model reproduces the spread of y


def test_nonconvergence_withholds_results(tmp_path):
    spec = _copy_spec(tmp_path, lambda t: t.replace("iterations = 4000", "iterations = 60")
                      .replace("burn_in = 1000", "burn_in = 10").replace("max_iterations = 16000",
                                                                        "max_iterations = 60")
                      .replace("thin = 3", "thin = 1"))
    rc = main(["run", str(spec), "--out", str(tmp_path / "r"), "--quiet"])
    assert rc == 6 and not (tmp_path / "r").exists()
    failed = next(tmp_path.glob("r.failed-*"))
    man = json.loads((failed / "manifest.json").read_text(encoding="utf-8"))
    assert man["error"]["code"] == "ABP-E405"
    assert (failed / "mcmc_diagnostics_fec.json").exists()
    assert not (failed / "ebv_fec.csv").exists()


@pytest.mark.parametrize("edit,fragment", [
    (lambda t: t.replace('mode = "bayes"', 'mode = "reml"'), "go together"),
    (lambda t: t.replace('relationship = "genomic"', 'relationship = "pedigree"'), "genomic"),
    (lambda t: t.replace("burn_in = 1000", "burn_in = 5000"), "burn_in"),
])
def test_bayes_spec_rules(tmp_path, edit, fragment):
    spec = _copy_spec(tmp_path, edit)
    with pytest.raises(ABPError) as exc:
        run_evaluation(spec, tmp_path / "o", console=False)
    assert fragment in exc.value.message
