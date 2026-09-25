"""Unknown-parent groups end to end: pedigree codes, spec rules, outputs,
identifiability refusal and the shipped example 11."""

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from abp.core.spec import validate_spec_dict
from abp.errors import ABPError
from abp.io.tables import read_table
from abp.qc.pedigree import PedigreeColumns, load_pedigree
from abp.workflows.evaluate import run_evaluation

ROOT = Path(__file__).resolve().parents[1]
EX11 = ROOT / "examples" / "11_sheep_upg"


def _read(p):
    with open(p, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _module(name):
    spec = importlib.util.spec_from_file_location(name, EX11 / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ped(tmp_path, rows):
    p = tmp_path / "ped.csv"
    p.write_text("id,sire,dam\n" + "\n".join(",".join(r) for r in rows) + "\n", encoding="utf-8")
    return read_table(p, ",")


def test_group_codes_are_parsed_as_groups_not_animals(tmp_path):
    t = _ped(tmp_path, [("a", "UPG:x", "UPG:y"), ("b", "UPG:x", "0"), ("c", "a", "b"),
                        ("d", "c", "UPG:y")])
    pd = load_pedigree(t, PedigreeColumns(), {"0"}, set(), group_prefix="UPG:")
    assert pd.pedigree.n == 4 and pd.added_founders == []
    g = pd.groups
    assert g.labels == ("UPG:x", "UPG:y")
    i = {a: k for k, a in enumerate(pd.pedigree.ids)}
    assert (g.sire_group[i["a"]], g.dam_group[i["a"]]) == (0, 1)
    assert (g.sire_group[i["b"]], g.dam_group[i["b"]]) == (0, -1)
    assert g.dam_group[i["d"]] == 1 and g.sire_group[i["c"]] == -1
    info = next(f for f in pd.qc.findings if f.check == "PED-UPG")
    assert {r["group"]: (r["n_as_sire"], r["n_as_dam"]) for r in info.examples} == \
        {"UPG:x": (2, 0), "UPG:y": (0, 2)}
    assert pd.qc.stats["n_groups"] == 2 and pd.qc.stats["n_parents_in_groups"] == 4
    # without a prefix the same codes are ordinary (added) founder parents
    pd2 = load_pedigree(t, PedigreeColumns(), {"0"}, set())
    assert pd2.pedigree.n == 6 and pd2.groups is None


def test_animal_ids_using_the_group_prefix_are_rejected(tmp_path):
    t = _ped(tmp_path, [("UPG:a", "0", "0"), ("b", "UPG:a", "0")])
    with pytest.raises(ABPError) as exc:
        load_pedigree(t, PedigreeColumns(), {"0"}, set(), group_prefix="UPG:")
    assert exc.value.code == "ABP-E105"


def _spec(**over):
    d = {"schema_version": "1",
         "project": {"name": "t", "species": "sheep", "synthetic_data": True},
         "analysis": {"task": "additive_ebv", "target_population": "x",
                      "information_cutoff": "2024-12-31", "genetic_base": "x"},
         "data": {"pedigree": "p.csv", "phenotypes": "y.csv"},
         "traits": [{"name": "y", "unit": "kg"}],
         "model": {"traits": ["y"],
                   "random": [{"name": "animal", "kind": "additive",
                               "relationship": "pedigree"}]},
         "variances": {"mode": "known", "values": {"animal": 1.0, "residual": 2.0}},
         "upg": {"prefix": "UPG:", "effect": "random", "variance_ratio": 0.5}}
    for k, v in over.items():
        d[k] = v
    return d


@pytest.mark.parametrize("over, code", [
    ({"upg": {"prefix": "UPG:", "effect": "random"}}, "ABP-E104"),
    ({"upg": {"prefix": "UPG:", "effect": "fixed", "variance_ratio": 1.0}}, "ABP-E104"),
    ({"upg": {"prefix": "UPG:", "effect": "fixed"}, "variances": {"mode": "reml"}}, "ABP-E303"),
    ({"upg": {"prefix": "0", "effect": "random", "variance_ratio": 1.0}}, "ABP-E104"),
    ({"model": {"traits": ["y"], "random": [{"name": "animal", "kind": "additive",
                                             "relationship": "genomic"}]},
      "data": {"pedigree": "p.csv", "phenotypes": "y.csv", "genotypes": "g.csv",
               "marker_map": "m.csv"}}, "ABP-E303"),
])
def test_spec_rules(over, code):
    with pytest.raises(ABPError) as exc:
        validate_spec_dict(_spec(**over))
    assert exc.value.code == code


def test_valid_upg_spec_passes():
    d = validate_spec_dict(_spec())
    assert d["upg"] == {"prefix": "UPG:", "effect": "random", "variance_ratio": 0.5}
    assert validate_spec_dict({k: v for k, v in _spec().items() if k != "upg"})["upg"] is None


def test_example11_data_are_reproducible(tmp_path):
    mk = _module("make_data")
    mk.write(mk.simulate(), tmp_path)
    for rel in ("data/pedigree.csv", "data/weaning.csv", "truth/tbv.csv", "truth/groups.csv"):
        assert (tmp_path / rel).read_bytes() == (EX11 / rel).read_bytes(), rel


def test_example11_random_groups(tmp_path):
    out = run_evaluation(EX11 / "analysis.toml", tmp_path / "o", console=False)
    t = out.results["traits"]["wwt"]
    assert out.manifest["relationship"]["kind"] == "pedigree_upg"
    assert out.manifest["relationship"]["n_groups"] == 3
    assert t["reml"]["status"] == "converged" and t["upg"]["effect"] == "random"
    ebv = _read(out.out_dir / "ebv_wwt.csv")
    assert len(ebv) == 1533 and not any(r["animal"].startswith("UPG:") for r in ebv)
    rel = np.array([float(r["reliability"]) for r in ebv])
    assert np.all((rel >= 0) & (rel <= 1))
    grp = _read(out.out_dir / "upg_solutions_wwt.csv")
    assert [g["group"] for g in grp] == ["UPG:A_16_19", "UPG:A_20_24", "UPG:B"]
    assert all(0 < float(g["reliability"]) < 1 for g in grp)
    # purchased rams: u* = u + Q g; each ram has both parents in one group
    sol = {g["group"]: float(g["solution"]) for g in grp}
    assert sol["UPG:A_20_24"] > sol["UPG:A_16_19"] > 0 > sol["UPG:B"]
    assert "upg" in out.manifest["diagnostics"]["wwt"]


def test_example11_fixed_groups_have_no_reliabilities(tmp_path):
    out = run_evaluation(EX11 / "analysis_fixed.toml", tmp_path / "o", console=False)
    t = out.results["traits"]["wwt"]
    assert t["upg"]["estimability"]["rank"] == t["upg"]["estimability"]["columns"]
    assert t["reliability_summary"] is None
    ebv = _read(out.out_dir / "ebv_wwt.csv")
    assert all(r["reliability"] == "" and float(r["sep"]) > 0 for r in ebv)
    assert "reliabilities are not defined" in (out.out_dir / "report.md").read_text()


def test_fixed_groups_confounded_with_intercept_stop_the_run(tmp_path):
    """Grouping the base flock too makes every lineage end in a group: the
    group effects are then confounded with the intercept."""
    rows = _read(EX11 / "data" / "pedigree.csv")
    (tmp_path / "data").mkdir()
    with open(tmp_path / "data" / "pedigree.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        for r in rows:
            r["sire"] = "UPG:BASE" if r["sire"] == "0" else r["sire"]
            r["dam"] = "UPG:BASE" if r["dam"] == "0" else r["dam"]
            w.writerow(r)
    text = (EX11 / "analysis_fixed.toml").read_text(encoding="utf-8")
    text = text.replace('phenotypes = "data/weaning.csv"',
                        f'phenotypes = "{(EX11 / "data" / "weaning.csv").as_posix()}"')
    spec = tmp_path / "spec.toml"
    spec.write_text(text, encoding="utf-8")
    with pytest.raises(ABPError) as exc:
        run_evaluation(spec, tmp_path / "o", console=False)
    assert exc.value.code == "ABP-E300"
    assert "confounded" in exc.value.message
    assert not (tmp_path / "o").exists()


def test_example11_groups_reduce_bias_of_purchased_rams(tmp_path):
    """One replicate (fixed seed): ignoring groups under-predicts purchased
    rams from a breeder with genetic progress; groups remove most of that."""
    res = _module("compare").main(tmp_path)
    m = res["metrics"]
    assert abs(m["none"]["bias_rams"]) > abs(m["random"]["bias_rams"])
    assert abs(m["none"]["bias_rams"]) > abs(m["fixed"]["bias_rams"])
    assert m["random"]["corr_lambs"] > m["none"]["corr_lambs"]


def test_validate_and_pedigree_commands_understand_group_codes(tmp_path, capsys):
    import json
    from abp.cli import main
    assert main(["validate", str(EX11 / "analysis.toml")]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert any(f["check"] == "PED-UPG" for f in summary["pedigree"]["findings"])
    assert main(["pedigree", str(EX11 / "data" / "pedigree.csv"), "--out", str(tmp_path / "p"),
                 "--group-prefix", "UPG:"]) == 0
    qc = json.loads((tmp_path / "p" / "qc_pedigree.json").read_text(encoding="utf-8"))
    assert qc["stats"]["n_groups"] == 3 and qc["stats"]["n_animals"] == 1533
    # without the prefix the codes are read as one animal used as sire and dam
    assert main(["pedigree", str(EX11 / "data" / "pedigree.csv"), "--out",
                 str(tmp_path / "q")]) == 4
