"""GBLUP and single-step through the full workflow, checked against an
independent dense computation built directly from the CSV files."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from abp.errors import ABPError
from abp.examples.sheep import SheepSimConfig, write_sheep_example
from abp.workflows.evaluate import run_evaluation
from tests.reference.dense_reference import blup_v_form, tabular_a

SMALL = dict(n_founder_rams=4, n_founder_ewes=40, rams_per_year=4, years=(2021, 2022, 2023),
             snp_per_chrom=40, qtl_per_chrom=20, n_chrom=5, n_wwt_typos=0)


@pytest.fixture(scope="module")
def flock(tmp_path_factory):
    out = tmp_path_factory.mktemp("flock") / "sheep"
    write_sheep_example(out, seed=7, cfg=SheepSimConfig(**SMALL))
    return out


def _spec(folder: Path, relationship: str, extra: str = "") -> Path:
    txt = f'''schema_version = "1"
[project]
name = "t"
species = "sheep"
synthetic_data = true
[analysis]
task = "additive_ebv"
target_population = "test"
information_cutoff = "2025-12-31"
genetic_base = "test"
[data]
pedigree = "sheep/data/pedigree.csv"
phenotypes = "sheep/data/lambs.csv"
genotypes = "sheep/data/genotypes.csv"
marker_map = "sheep/data/markers.csv"
[[traits]]
name = "wwt"
unit = "kg"
[model]
traits = ["wwt"]
fixed = [{{ column = "cg", type = "factor" }}]
random = [{{ name = "animal", kind = "additive", relationship = "{relationship}" }}]
[variances]
mode = "known"
values = {{ animal = 3.0, residual = 10.0 }}
[genomic]
singular_policy = "blend"
blend_alpha = 0.1
[solver]
method = "dense"
{extra}
'''
    p = folder / f"spec_{relationship}.toml"
    p.write_text(txt, encoding="utf-8")
    return p


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _independent_inputs(flock):
    ped = _read(flock / "data" / "pedigree.csv")
    ids = [r["id"] for r in ped]
    sires = [None if r["sire"] == "0" else r["sire"] for r in ped]
    dams = [None if r["dam"] == "0" else r["dam"] for r in ped]
    A = tabular_a(ids, sires, dams)
    pos = {a: k for k, a in enumerate(ids)}
    g = _read(flock / "data" / "genotypes.csv")
    gid = [r["id"] for r in g]
    markers = [c for c in g[0] if c != "id"]
    M = np.array([[np.nan if r[m] == "NA" else float(r[m]) for m in markers] for r in g])
    miss = np.isnan(M)
    p = np.nanmean(M, axis=0) / 2
    keep = (p > 0) & (p < 1)
    W = np.where(miss, 0.0, M - 2 * p)[:, keep]
    G = W @ W.T / (2 * np.sum(p[keep] * (1 - p[keep])))
    lambs = _read(flock / "data" / "lambs.csv")
    return ids, pos, A, gid, G, lambs


def test_gblup_workflow_matches_independent_reference(flock):
    folder = flock.parent
    spec = _spec(folder, "genomic", '[qc]\nungenotyped_records = "exclude"')
    out = run_evaluation(spec, folder / "out_g", console=False)
    ids, pos, A, gid, G, lambs = _independent_inputs(flock)
    gi = [pos[a] for a in gid]
    Gs = 0.9 * G + 0.1 * A[np.ix_(gi, gi)]
    recs = [r for r in lambs if r["id"] in set(gid) and r["wwt"] != "NA"]
    y = np.array([float(r["wwt"]) for r in recs])
    cgs = sorted({r["cg"] for r in recs})
    X = np.array([[1.0 if r["cg"] == c else 0.0 for c in cgs] for r in recs])
    X = np.column_stack([np.ones(len(recs)), X])
    gpos = {a: k for k, a in enumerate(gid)}
    Z = np.zeros((len(recs), len(gid)))
    Z[np.arange(len(recs)), [gpos[r["id"]] for r in recs]] = 1
    # generalized-inverse reference (X rank deficient) via pinv
    V = Z @ (3.0 * Gs) @ Z.T + 10.0 * np.eye(len(recs))
    Vi = np.linalg.inv(V)
    P = Vi - Vi @ X @ np.linalg.pinv(X.T @ Vi @ X) @ X.T @ Vi
    u_ref = 3.0 * Gs @ Z.T @ P @ y
    pev_ref = np.diag(3.0 * Gs - 9.0 * Gs @ Z.T @ P @ Z @ Gs)
    rows = {r["animal"]: r for r in _read(out.out_dir / "ebv_wwt.csv")}
    assert set(rows) == set(gid)
    for a in gid:
        assert float(rows[a]["ebv"]) == pytest.approx(u_ref[gpos[a]], abs=1e-8)
        assert float(rows[a]["pev"]) == pytest.approx(pev_ref[gpos[a]], abs=1e-8)
    man = json.loads((out.out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert man["relationship"]["candidate_genotype_use"] == "transductive_unsupervised"
    assert man["relationship"]["g_policy"]["blend_alpha"] == 0.1


def test_excluded_records_cannot_influence_results(flock, tmp_path):
    """Leakage check: altering phenotypes that the approved rule excludes
    (non-genotyped animals in GBLUP) leaves every output byte unchanged."""
    import shutil
    work = tmp_path / "w"
    shutil.copytree(flock, work / "sheep")
    spec = _spec(work, "genomic", '[qc]\nungenotyped_records = "exclude"')
    a = run_evaluation(spec, work / "a", console=False)
    geno = {r["id"] for r in _read(work / "sheep" / "data" / "genotypes.csv")}
    path = work / "sheep" / "data" / "lambs.csv"
    rows = _read(path)
    changed = 0
    for r in rows:
        if r["id"] not in geno and r["wwt"] != "NA":
            r["wwt"] = str(float(r["wwt"]) + 7.5)
            changed += 1
    assert changed > 10
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    b = run_evaluation(spec, work / "b", console=False)
    assert (a.out_dir / "ebv_wwt.csv").read_bytes() == (b.out_dir / "ebv_wwt.csv").read_bytes()


def test_gblup_refuses_ungenotyped_records_by_default(flock):
    spec = _spec(flock.parent, "genomic")
    with pytest.raises(ABPError) as exc:
        run_evaluation(spec, flock.parent / "out_refuse", console=False)
    assert exc.value.code == "ABP-E210"


def test_single_step_workflow_matches_explicit_h(flock):
    folder = flock.parent
    spec = _spec(folder, "single_step")
    out = run_evaluation(spec, folder / "out_ss", console=False)
    ids, pos, A, gid, G, lambs = _independent_inputs(flock)
    gi = np.array([pos[a] for a in gid])
    ng = np.array([k for k in range(len(ids)) if k not in set(gi)])
    A22 = A[np.ix_(gi, gi)]
    Gs = 0.9 * G + 0.1 * A22
    order = np.concatenate([ng, gi])
    Ao = A[np.ix_(order, order)]
    n1 = ng.size
    A11, A12 = Ao[:n1, :n1], Ao[:n1, n1:]
    B = A12 @ np.linalg.inv(A22)
    H = np.block([[A11 + B @ (Gs - A22) @ B.T, B @ Gs], [Gs @ B.T, Gs]])
    recs = [r for r in lambs if r["wwt"] != "NA"]
    y = np.array([float(r["wwt"]) for r in recs])
    cgs = sorted({r["cg"] for r in recs})
    X = np.column_stack([np.ones(len(recs))] + [[1.0 if r["cg"] == c else 0.0 for r in recs]
                                               for c in cgs])
    opos = {ids[k]: j for j, k in enumerate(order)}
    Z = np.zeros((len(recs), len(ids)))
    Z[np.arange(len(recs)), [opos[r["id"]] for r in recs]] = 1
    V = Z @ (3.0 * H) @ Z.T + 10.0 * np.eye(len(recs))
    Vi = np.linalg.inv(V)
    P = Vi - Vi @ X @ np.linalg.pinv(X.T @ Vi @ X) @ X.T @ Vi
    u_ref = 3.0 * H @ Z.T @ P @ y
    rows = {r["animal"]: r for r in _read(out.out_dir / "ebv_wwt.csv")}
    for a in ids:
        assert float(rows[a]["ebv"]) == pytest.approx(u_ref[opos[a]], abs=1e-8)
        rel_ref = 1 - (3.0 * H - 9.0 * H @ Z.T @ P @ Z @ H)[opos[a], opos[a]] / (3.0 * H[opos[a], opos[a]])
        assert float(rows[a]["reliability"]) == pytest.approx(rel_ref, abs=1e-8)


def test_sheep_generator_is_deterministic(tmp_path):
    a = write_sheep_example(tmp_path / "a", seed=11, cfg=SheepSimConfig(**SMALL))
    b = write_sheep_example(tmp_path / "b", seed=11, cfg=SheepSimConfig(**SMALL))
    for f in ("data/pedigree.csv", "data/lambs.csv", "data/genotypes.csv", "truth/tbv.csv"):
        assert (a / f).read_bytes() == (b / f).read_bytes()
