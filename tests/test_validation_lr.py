"""LR forward-in-time validation: statistics, date splitting, leakage, bootstrap."""

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from abp.errors import ABPError
from abp.examples.sheep import SheepSimConfig, write_sheep_example
from abp.qc.phenotype import RecordSet
from abp.workflows.evaluate import run_evaluation
from abp.workflows.validation_lr import cluster_bootstrap, lr_statistics, split_by_cutoff

SMALL = dict(n_founder_rams=4, n_founder_ewes=50, rams_per_year=4, years=(2021, 2022, 2023, 2024),
             snp_per_chrom=30, qtl_per_chrom=20, n_chrom=5, n_wwt_typos=0)


def test_lr_statistics_definitions():
    rng = np.random.default_rng(0)
    u_p = rng.normal(0, 1, 200)
    u_w = 0.2 + 1.3 * u_p + rng.normal(0, 0.5, 200)
    s = lr_statistics(u_p, u_w)
    assert s["bias_delta_p"] == pytest.approx(u_p.mean() - u_w.mean(), abs=1e-14)
    c = np.cov(u_w, u_p)
    assert s["dispersion_b_w_p"] == pytest.approx(c[0, 1] / c[1, 1], rel=1e-12)
    assert s["rho_wp"] == pytest.approx(np.corrcoef(u_w, u_p)[0, 1], rel=1e-12)
    # identical evaluations: no bias, unit slope, correlation 1
    s0 = lr_statistics(u_p, u_p)
    assert s0["bias_delta_p"] == 0 and s0["dispersion_b_w_p"] == pytest.approx(1.0)
    assert s0["rho_wp"] == pytest.approx(1.0)


def _records(dates):
    n = len(dates)
    return RecordSet([str(i) for i in range(n)], [f"a{i}" for i in range(n)], list(range(2, n + 2)),
                     {"t": np.ones(n)}, {}, {}, {"t": "kg"}, list(dates))


def test_split_by_cutoff_rules():
    hidden = split_by_cutoff(_records(["2023", "2024", "2025", "2025-01-02", "2024-12-31"]),
                             "2024-12-31")
    assert hidden.tolist() == [False, False, True, True, False]
    with pytest.raises(ABPError) as exc:
        split_by_cutoff(_records(["2024", "2025"]), "2024-06-30")      # 2024 straddles
    assert "straddle" in exc.value.message
    with pytest.raises(ABPError) as exc:
        split_by_cutoff(_records(["24/05/2025"]), "2024-12-31")
    assert exc.value.code == "ABP-E103"


def test_cluster_bootstrap_is_seeded_and_counts_clusters():
    rng = np.random.default_rng(1)
    u_p = rng.normal(size=60)
    u_w = u_p + rng.normal(0, 0.3, 60)
    clusters = [f"s{k % 12}" for k in range(60)]
    a = cluster_bootstrap(u_p, u_w, clusters, 200, seed=5)
    b = cluster_bootstrap(u_p, u_w, clusters, 200, seed=5)
    assert a == b and a["n_clusters"] == 12
    lo, hi = a["dispersion_b_w_p"]["ci95"]
    assert lo < lr_statistics(u_p, u_w)["dispersion_b_w_p"] < hi


@pytest.fixture(scope="module")
def flock(tmp_path_factory):
    out = tmp_path_factory.mktemp("lrflock") / "sheep"
    write_sheep_example(out, seed=3, cfg=SheepSimConfig(**SMALL))
    return out


def _spec(folder: Path, extra_validation: str = "", variances: str | None = None) -> Path:
    variances = variances or 'mode = "known"\nvalues = { animal = 4.0, residual = 12.25 }'
    p = folder / "lr.toml"
    p.write_text(f'''schema_version = "1"
[project]
name = "lr"
species = "sheep"
synthetic_data = true
[analysis]
task = "additive_ebv"
target_population = "t"
information_cutoff = "2025-12-31"
genetic_base = "founders"
[data]
pedigree = "sheep/data/pedigree.csv"
phenotypes = "sheep/data/lambs.csv"
[data.pedigree_columns]
sex = "sex"
birth_date = "birth_date"
[data.phenotype_columns]
date = "year"
[[traits]]
name = "wwt"
unit = "kg"
[model]
traits = ["wwt"]
fixed = [{{ column = "cg", type = "factor" }}, {{ column = "birth_type", type = "factor" }}]
random = [{{ name = "animal", kind = "additive", relationship = "pedigree" }}]
[variances]
{variances}
[validation]
method = "lr"
cutoff = "2023-12-31"
bootstrap_replicates = 200
{extra_validation}
''', encoding="utf-8")
    return p


def _read(p):
    with open(p, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_hidden_phenotypes_cannot_leak_into_partial_evaluation(flock, tmp_path):
    work = tmp_path / "w"
    shutil.copytree(flock, work / "sheep")
    spec = _spec(work)
    a = run_evaluation(spec, work / "a", console=False)
    lr = json.loads((a.out_dir / "lr_validation.json").read_text(encoding="utf-8"))
    assert lr["n_focal"] > 20 and lr["n_records_hidden"] > 20
    assert abs(lr["statistics"]["dispersion_b_w_p"] - 1) < 0.5
    rows = _read(work / "sheep" / "data" / "lambs.csv")
    for r in rows:
        if r["year"] == "2024" and r["wwt"] != "NA":
            r["wwt"] = str(float(r["wwt"]) + 25.0)          # tamper with hidden records only
    with open(work / "sheep" / "data" / "lambs.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    b = run_evaluation(spec, work / "b", console=False)
    fa = {r["animal"]: r for r in _read(a.out_dir / "lr_focal_wwt.csv")}
    fb = {r["animal"]: r for r in _read(b.out_dir / "lr_focal_wwt.csv")}
    assert fa.keys() == fb.keys()
    assert all(fa[k]["ebv_partial"] == fb[k]["ebv_partial"] for k in fa)      # untouched
    assert any(fa[k]["ebv_whole"] != fb[k]["ebv_whole"] for k in fa)          # whole sees them


def test_partial_reml_ignores_hidden_records(flock, tmp_path):
    work = tmp_path / "w"
    shutil.copytree(flock, work / "sheep")
    spec = _spec(work, variances='mode = "reml"')
    a = run_evaluation(spec, work / "a", console=False)
    lr = a.results["validation"]
    assert lr["variance_source"].startswith("REML on the partial data only")
    assert lr["variance_components"] != a.results["traits"]["wwt"]["variance_components"]


def test_validation_spec_rules(flock, tmp_path):
    work = tmp_path / "w"
    shutil.copytree(flock, work / "sheep")
    spec = _spec(work)
    text = spec.read_text(encoding="utf-8")
    spec.write_text(text.replace('[data.phenotype_columns]\ndate = "year"\n', ""), encoding="utf-8")
    with pytest.raises(ABPError) as exc:
        run_evaluation(spec, work / "o1", console=False)
    assert "phenotype_columns.date" in exc.value.message
    spec.write_text(text.replace('cutoff = "2023-12-31"', 'cutoff = "2026-01-01"'), encoding="utf-8")
    with pytest.raises(ABPError) as exc:
        run_evaluation(spec, work / "o2", console=False)
    assert "information_cutoff" in exc.value.message
