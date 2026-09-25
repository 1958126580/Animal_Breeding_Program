"""manifest.json of passed and failed runs satisfies the published contract."""

import json
from pathlib import Path

from abp.cli import main

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "contracts" / "run_manifest.schema.json").read_text(encoding="utf-8"))


def _check(man: dict):
    for key in SCHEMA["required"]:
        assert key in man, key
    assert man["status"] in SCHEMA["properties"]["status"]["enum"]
    for i in man["inputs"]:
        assert set(SCHEMA["properties"]["inputs"]["items"]["required"]) <= set(i)
        assert i["role"] in SCHEMA["properties"]["inputs"]["items"]["properties"]["role"]["enum"]
    for o in man["outputs"]:
        assert {"file", "bytes", "sha256"} <= set(o)


def test_manifest_contract_for_passed_and_failed_runs(tmp_path):
    ex = ROOT / "examples" / "01_textbook_mrode_3_1" / "analysis.toml"
    assert main(["run", str(ex), "--out", str(tmp_path / "ok"), "--quiet"]) == 0
    _check(json.loads((tmp_path / "ok" / "manifest.json").read_text(encoding="utf-8")))
    bad = tmp_path / "bad"
    bad.mkdir()
    for f in ("analysis.toml", "pedigree.csv"):
        (bad / f).write_bytes((ex.parent / f).read_bytes())
    (bad / "phenotypes.csv").write_text("id,sex,wwg\n4,M,4.5\n99,F,2.9\n", encoding="utf-8")
    assert main(["run", str(bad / "analysis.toml"), "--out", str(tmp_path / "r"), "--quiet"]) == 4
    failed = next(tmp_path.glob("r.failed-*"))
    man = json.loads((failed / "manifest.json").read_text(encoding="utf-8"))
    _check(man)
    assert man["status"] == "failed" and man["error"]["code"] == "ABP-E210"
