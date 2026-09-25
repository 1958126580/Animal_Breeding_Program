"""Every shipped example runs end to end (the manual's commands are real)."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from abp.cli import main
from abp.core.spec import json_schema
from abp.workflows.evaluate import run_evaluation

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples"


def _read(p):
    with open(p, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_example02_reml_and_quarantine(tmp_path):
    out = run_evaluation(EX / "02_sheep_wwt_reml" / "analysis.toml", tmp_path / "o", console=False)
    t = out.results["traits"]["wwt"]
    assert t["reml"]["status"] == "converged"
    assert 0.05 < t["heritability"] < 0.6
    assert len(_read(out.out_dir / "qc_excluded_records.csv")) == 3
    rel = [float(r["reliability"]) for r in _read(out.out_dir / "ebv_wwt.csv")]
    assert 0 <= min(rel) and max(rel) <= 1


def test_example03_repeatability(tmp_path):
    out = run_evaluation(EX / "03_sheep_nlb_repeatability" / "analysis.toml", tmp_path / "o",
                         console=False)
    t = out.results["traits"]["nlb"]
    assert t["reml"]["status"] in ("converged", "converged_boundary")
    assert "pe" in t["variance_components"]


def test_example04_gblup(tmp_path):
    out = run_evaluation(EX / "04_sheep_fec_gblup" / "analysis.toml", tmp_path / "o", console=False)
    assert out.manifest["relationship"]["kind"] == "genomic"
    assert out.manifest["relationship"]["candidate_genotype_use"] == "transductive_unsupervised"
    excl = {r["reason"] for r in _read(out.out_dir / "qc_excluded_records.csv")}
    assert excl == {"PHE-UNGENOTYPED"}
    assert (out.out_dir / "qc_genotypes.json").exists()


def test_example05_single_step(tmp_path):
    out = run_evaluation(EX / "05_sheep_wwt_single_step" / "analysis.toml", tmp_path / "o",
                         console=False)
    rel = out.manifest["relationship"]
    assert rel["kind"] == "single_step" and rel["g_policy"]["tuning"] == "match_a22"
    c = rel["compatibility"]
    assert c["mean_diag_Gstar"] == pytest.approx(c["mean_diag_A22"])


def test_example06_multitrait_index(tmp_path):
    out = run_evaluation(EX / "06_sheep_multitrait_index" / "analysis.toml", tmp_path / "o",
                         console=False)
    assert set(out.results["traits"]) == {"wwt", "fat", "fec", "nlb1"}
    ebv = {r["animal"]: r for r in _read(out.out_dir / "ebv_multitrait.csv")}
    w = {"wwt": 2.0, "fat": -3.0, "fec": -4.0, "nlb1": 25.0}
    for r in _read(out.out_dir / "index.csv")[:50]:
        expect = sum(w[t] * float(ebv[r["animal"]][f"ebv_{t}"]) for t in w)
        assert float(r["index"]) == pytest.approx(expect, abs=1e-9)
    assert "synthetic placeholders" in (out.out_dir / "report.md").read_text(encoding="utf-8")
    assert out.manifest["diagnostics"]["multi_trait"]["stacking_order"].startswith("animal-major")


def test_example07_index_cli(capsys):
    assert main(["index", str(EX / "07_selection_index" / "index.toml")]) == 0
    res = json.loads(capsys.readouterr().out)
    assert res["reliability"] == pytest.approx(12 / 35)
    np.testing.assert_allclose(list(res["coefficients"].values()), [3 / 7, 2 / 7])


def test_selftest_command(capsys):
    assert main(["selftest"]) == 0
    assert "RESULT: PASS" in capsys.readouterr().out


def test_published_json_schema_is_current():
    on_disk = json.loads((ROOT / "contracts" / "analysis_spec.schema.json").read_text(encoding="utf-8"))
    assert on_disk == json_schema(), "regenerate with: python -m abp.core.spec"


def test_errors_command_lists_all_codes(capsys):
    assert main(["errors"]) == 0
    text = capsys.readouterr().out
    assert "ABP-E200" in text and "ABP-E600" in text
