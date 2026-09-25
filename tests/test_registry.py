"""The method registry is complete, uses only allowed statuses, and every
evidence test it cites exists (so the registry cannot claim phantom tests)."""

import ast
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {"method_id", "name", "estimand", "scope", "math", "sources", "errata",
            "exact_or_approximate", "complexity", "code", "tests", "gates",
            "implementation_status", "maturity", "known_limits", "dependencies"}
STATUS = {"passed", "failed", "blocked", "not_run"}
MATURITY = {"stable", "validated", "experimental"}


def _test_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}


def test_registry_entries():
    reg = tomllib.loads((ROOT / "docs" / "method_registry.toml").read_text(encoding="utf-8"))
    ids = set()
    for m in reg["method"]:
        missing = REQUIRED - set(m)
        assert not missing, (m.get("method_id"), missing)
        assert m["method_id"] not in ids
        ids.add(m["method_id"])
        assert m["implementation_status"] in STATUS
        assert m["maturity"] in MATURITY
        assert m["maturity"] != "validated" or m["gates"].get("G5") == "passed"
        assert set(m["gates"]) == {f"G{k}" for k in range(8)}
        for c in m["code"]:
            assert (ROOT / c).exists(), c
        assert m["tests"], m["method_id"]
        for t in m["tests"]:
            file, func = t.split("::")
            assert func in _test_names(ROOT / file), t
    for p in reg["planned"]:
        assert p["implementation_status"] == "not_run"
