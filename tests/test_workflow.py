"""End-to-end workflow: outputs, manifest, failure statuses, atomicity, paths."""

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from abp.cli import main
from abp.errors import ABPError
from abp.workflows.evaluate import run_evaluation
from tests.reference.dense_reference import blup_v_form, tabular_a

ROOT = Path(__file__).resolve().parents[1]
EX01 = ROOT / "examples" / "01_textbook_mrode_3_1"


def _copy_example(dst: Path) -> Path:
    shutil.copytree(EX01, dst)
    return dst / "analysis.toml"


def _read_csv(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_example01_end_to_end_matches_independent_reference(tmp_path):
    out = run_evaluation(EX01 / "analysis.toml", tmp_path / "out", console=False)
    assert out.status == "passed"
    rows = {r["animal"]: r for r in _read_csv(out.out_dir / "ebv_wwg.csv")}
    ids = [str(i) for i in range(1, 9)]
    sires = [None, None, None, "1", "3", "1", "4", "3"]
    dams = [None, None, None, None, "2", "2", "5", "6"]
    A = tabular_a(ids, sires, dams)
    Z = np.zeros((5, 8))
    for r, a in enumerate(["4", "5", "6", "7", "8"]):
        Z[r, int(a) - 1] = 1
    X = np.array([[1, 0], [0, 1], [0, 1], [1, 0], [1, 0]], dtype=float)  # M, F
    y = np.array([4.5, 2.9, 3.9, 3.5, 5.0])
    _, u, pev = blup_v_form(y, X, Z, 20 * A, 40 * np.eye(5))
    for k, a in enumerate(ids):
        assert float(rows[a]["ebv"]) == pytest.approx(u[k], abs=1e-10)
        assert float(rows[a]["pev"]) == pytest.approx(pev[k, k], abs=1e-9)
        assert float(rows[a]["reliability"]) == pytest.approx(1 - pev[k, k] / (20 * A[k, k]), abs=1e-10)
    man = json.loads((out.out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert man["status"] == "passed" and man["error"] is None
    assert {i["role"] for i in man["inputs"]} == {"pedigree", "phenotypes"}
    listed = {o["file"] for o in man["outputs"]}
    assert {"ebv_wwg.csv", "report.md", "results.json", "qc_pedigree.json"} <= listed
    # manifest hashes match the files actually published
    from abp.workflows.manifest import sha256_path
    for o in man["outputs"]:
        assert sha256_path(out.out_dir / o["file"]) == o["sha256"]
    assert "SYNTHETIC DATA" in (out.out_dir / "report.md").read_text(encoding="utf-8")
    # no staging folders left behind
    assert not list(tmp_path.glob("out.partial-*"))


def test_rerun_is_deterministic(tmp_path):
    a = run_evaluation(EX01 / "analysis.toml", tmp_path / "a", console=False)
    b = run_evaluation(EX01 / "analysis.toml", tmp_path / "b", console=False)
    for name in ("ebv_wwg.csv", "fixed_effects_wwg.csv"):
        assert (a.out_dir / name).read_bytes() == (b.out_dir / name).read_bytes()


def test_output_exists_requires_force(tmp_path):
    run_evaluation(EX01 / "analysis.toml", tmp_path / "o", console=False)
    with pytest.raises(ABPError) as exc:
        run_evaluation(EX01 / "analysis.toml", tmp_path / "o", console=False)
    assert exc.value.code == "ABP-E503"
    out = run_evaluation(EX01 / "analysis.toml", tmp_path / "o", force=True, console=False)
    assert out.status == "passed"
    assert not list(tmp_path.glob("o.replaced-*"))


def test_pedigree_cycle_fails_without_publishing(tmp_path):
    spec = _copy_example(tmp_path / "ex")
    (spec.parent / "pedigree.csv").write_text(
        "id,sire,dam,sex\n1,8,0,M\n2,0,0,F\n3,0,0,M\n4,1,0,M\n5,3,2,F\n6,1,2,F\n7,4,5,M\n8,3,6,M\n",
        encoding="utf-8")
    rc = main(["run", str(spec), "--out", str(tmp_path / "res"), "--quiet"])
    assert rc == 4
    assert not (tmp_path / "res").exists()
    failed = list(tmp_path.glob("res.failed-*"))
    assert len(failed) == 1
    man = json.loads((failed[0] / "manifest.json").read_text(encoding="utf-8"))
    assert man["status"] == "failed" and man["error"]["code"] == "ABP-E200"
    assert "ebv_wwg.csv" not in {p.name for p in failed[0].iterdir()}


def test_out_of_range_error_then_quarantine(tmp_path):
    spec = _copy_example(tmp_path / "ex")
    phe = spec.parent / "phenotypes.csv"
    phe.write_text("id,sex,wwg\n4,M,4.5\n5,F,2.9\n6,F,39\n7,M,3.5\n8,M,5.0\n", encoding="utf-8")
    assert main(["run", str(spec), "--out", str(tmp_path / "r1"), "--quiet"]) == 4
    text = spec.read_text(encoding="utf-8") + '\n[qc]\nout_of_range = "quarantine"\n'
    spec.write_text(text, encoding="utf-8")
    out = run_evaluation(spec, tmp_path / "r2", console=False)
    excl = _read_csv(out.out_dir / "qc_excluded_records.csv")
    assert len(excl) == 1 and excl[0]["animal"] == "6" and excl[0]["reason"] == "PHE-RANGE"
    rows = {r["animal"]: r for r in _read_csv(out.out_dir / "ebv_wwg.csv")}
    assert rows["6"]["n_records"] == "0"


def test_unknown_spec_key_is_rejected(tmp_path):
    spec = _copy_example(tmp_path / "ex")
    spec.write_text(spec.read_text(encoding="utf-8").replace('method = "dense"', 'methd = "dense"'),
                    encoding="utf-8")
    with pytest.raises(ABPError) as exc:
        run_evaluation(spec, tmp_path / "r", console=False)
    assert exc.value.code == "ABP-E104" and "methd" in exc.value.message


def test_repeated_records_need_permanent_environment(tmp_path):
    spec = _copy_example(tmp_path / "ex")
    (spec.parent / "phenotypes.csv").write_text(
        "id,sex,wwg\n4,M,4.5\n4,M,4.0\n5,F,2.9\n6,F,3.9\n7,M,3.5\n8,M,5.0\n", encoding="utf-8")
    with pytest.raises(ABPError) as exc:
        run_evaluation(spec, tmp_path / "r", console=False)
    assert exc.value.code == "ABP-E303"
    text = spec.read_text(encoding="utf-8").replace(
        'random = [{ name = "animal", kind = "additive", relationship = "pedigree" }]',
        'random = [{ name = "animal", kind = "additive", relationship = "pedigree" },\n'
        '          { name = "pe", kind = "iid" }]').replace(
        "values = { animal = 20.0, residual = 40.0 }",
        "values = { animal = 20.0, pe = 5.0, residual = 35.0 }")
    spec.write_text(text, encoding="utf-8")
    out = run_evaluation(spec, tmp_path / "r2", console=False)
    assert (out.out_dir / "random_pe_wwg.csv").exists()


def test_unicode_and_space_paths(tmp_path):
    """Chinese characters and spaces in folder and file names (Windows requirement)."""
    folder = tmp_path / "育种 数据 test"
    spec = _copy_example(folder)
    (spec.parent / "pedigree.csv").rename(spec.parent / "系谱 文件.csv")
    spec.write_text(spec.read_text(encoding="utf-8").replace('pedigree = "pedigree.csv"',
                                                             'pedigree = "系谱 文件.csv"'),
                    encoding="utf-8")
    rc = main(["run", str(spec), "--out", str(folder / "结果 输出"), "--quiet"])
    assert rc == 0
    rep = (folder / "结果 输出" / "report.md").read_text(encoding="utf-8")
    assert "系谱 文件.csv" in rep


def test_bad_utf8_and_ragged_rows(tmp_path):
    spec = _copy_example(tmp_path / "ex")
    phe = spec.parent / "phenotypes.csv"
    phe.write_bytes("id,sex,wwg\n4,M,4.5\n5,F,2.9\n".encode("utf-8") + b"6,F,\xff3.9\n")
    with pytest.raises(ABPError) as exc:
        run_evaluation(spec, tmp_path / "r", console=False)
    assert exc.value.code == "ABP-E101"
    phe.write_text("id,sex,wwg\n4,M,4.5\n5,F\n", encoding="utf-8")
    with pytest.raises(ABPError) as exc:
        run_evaluation(spec, tmp_path / "r", console=False)
    assert exc.value.code == "ABP-E103" and exc.value.details["line"] == 3


def test_cuda_request_is_never_faked(tmp_path):
    spec = _copy_example(tmp_path / "ex")
    spec.write_text(spec.read_text(encoding="utf-8") + '\n[backend]\ndevice = "cuda"\n',
                    encoding="utf-8")
    assert main(["run", str(spec), "--out", str(tmp_path / "r"), "--quiet"]) == 7
    spec.write_text(spec.read_text(encoding="utf-8") + 'on_unavailable = "fallback_cpu"\n',
                    encoding="utf-8")
    out = run_evaluation(spec, tmp_path / "r2", console=False)
    assert out.manifest["backend"] == {"requested": "cuda", "used": "cpu", "fallback": True}


def test_pedigree_command(tmp_path):
    rc = main(["pedigree", str(EX01 / "pedigree.csv"), "--out", str(tmp_path / "p"),
               "--sex", "sex"])
    assert rc == 0
    rows = _read_csv(tmp_path / "p" / "inbreeding.csv")
    assert len(rows) == 8
    trip = _read_csv(tmp_path / "p" / "ainv_triplets.csv")
    assert all(int(r["row"]) >= int(r["col"]) for r in trip)


def test_validate_command(tmp_path, capsys):
    assert main(["validate", str(EX01 / "analysis.toml")]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "passed"
