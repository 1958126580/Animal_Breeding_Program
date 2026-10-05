#!/usr/bin/env python3
"""Workflow-level large single step: ``abp run`` on generated files (round 7).

Writes a synthetic data set - pedigree CSV (``--gens`` x ``--per-gen`` animals, discrete
generations), phenotypes CSV (one record per non-founder, 400 contemporary groups),
and a PLINK 1 binary file with the last ``--genotyped`` animals and ``--markers``
random SNPs (written from int8, never held as float) - and runs the complete
workflow (QC, frequencies, APY, matrix-free single step, PCG, sampled
reliabilities, outputs, manifest) with ``genomic.genotype_storage = "int8"`` in a
child process whose peak resident memory is measured (``ru_maxrss``).

Usage: python benchmarks/ssmf_workflow_large.py [--gens 12 --per-gen 10000
       --genotyped 30000 --markers 20000 --core 5000 --pev-samples 10]
       [--out benchmarks/results/ssmf_workflow_large.json]
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

SPEC = """schema_version = "1"
[project]
name = "large_single_step"
species = "sheep"
synthetic_data = true
[analysis]
task = "additive_ebv"
target_population = "synthetic"
information_cutoff = "2025-12-31"
genetic_base = "founders"
[data]
pedigree = "{d}/pedigree.csv"
phenotypes = "{d}/phenotypes.csv"
plink = "{d}/geno"
genotype_assembly = "SYNTHETIC-1"
[[traits]]
name = "y"
unit = "kg"
[model]
traits = ["y"]
fixed = [{{ column = "cg", type = "factor" }}]
random = [{{ name = "animal", kind = "additive", relationship = "single_step" }}]
[variances]
mode = "known"
values = {{ animal = 4.0, residual = 12.0 }}
[genomic]
singular_policy = "blend"
blend_alpha = 0.05
apy_core_size = {core}
genotype_storage = "int8"
single_step_mode = "matrix_free"
min_call_rate_marker = 0.0
min_call_rate_animal = 0.0
[solver]
pev = "sampled"
pev_samples = {samples}
tol = 1e-8
max_iter = 20000
[resources]
max_memory_gb = 14
"""


def write_data(d: Path, a) -> dict:
    import numpy as np

    from abp.io.plink import write_bed
    from run_benchmarks import sim_pedigree
    ids, s, dm = sim_pedigree(a.gens, a.per_gen, 200, seed=5)
    with open(d / "pedigree.csv", "w", encoding="utf-8") as fh:
        fh.write("id,sire,dam\n")
        for i, x, y in zip(ids, s, dm):
            fh.write(f"{i},{x or '0'},{y or '0'}\n")
    rng = np.random.default_rng(3)
    with open(d / "phenotypes.csv", "w", encoding="utf-8") as fh:
        fh.write("id,cg,y\n")
        for i in ids[a.per_gen:]:
            fh.write(f"{i},c{rng.integers(0, 400)},{rng.normal(30, 4):.3f}\n")
    geno_ids = ids[len(ids) - a.genotyped:]
    p = rng.uniform(0.05, 0.95, a.markers)
    M = np.empty((a.genotyped, a.markers), dtype=np.int8)
    for i in range(0, a.genotyped, 1000):
        k = min(1000, a.genotyped - i)
        M[i:i + k] = (rng.random((k, a.markers)) < p) + (rng.random((k, a.markers)) < p)
    write_bed(d / "geno", geno_ids,
              [(f"snp{j}", "1", 1000 * (j + 1), "A", "G") for j in range(a.markers)], M)
    return {"animals": len(ids), "records": len(ids) - a.per_gen, "genotyped": a.genotyped,
            "markers": a.markers, "bed_bytes": (d / "geno.bed").stat().st_size}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=12)
    ap.add_argument("--per-gen", type=int, default=10000)
    ap.add_argument("--genotyped", type=int, default=30000)
    ap.add_argument("--markers", type=int, default=20000)
    ap.add_argument("--core", type=int, default=5000)
    ap.add_argument("--pev-samples", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "results" /
                                         "ssmf_workflow_large.json"))
    a = ap.parse_args()
    from abp.workflows.manifest import environment, utc_now
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        t0 = time.perf_counter()
        size = write_data(d, a)
        t_write = time.perf_counter() - t0
        (d / "a.toml").write_text(SPEC.format(d=d.as_posix(), core=a.core,
                                              samples=a.pev_samples), encoding="utf-8")
        t1 = time.perf_counter()
        run = subprocess.run([sys.executable, "-m", "abp", "run", str(d / "a.toml"), "--out",
                              str(d / "out")], capture_output=True, text=True)
        wall = time.perf_counter() - t1
        peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024
        manifest = json.loads((d / "out" / "manifest.json").read_text(encoding="utf-8")) \
            if run.returncode == 0 else None
        results = json.loads((d / "out" / "results.json").read_text(encoding="utf-8")) \
            if run.returncode == 0 else None
    doc = {"created_at": utc_now(), "environment": environment(), "size": size,
           "data_generation_s": t_write, "abp_run_exit_status": run.returncode,
           "abp_run_wall_s": wall, "abp_run_peak_rss_gb": peak / 2**30,
           "log_tail": run.stdout.splitlines()[-25:] + run.stderr.splitlines()[-10:]}
    if results is not None:
        t = results["traits"]["y"]
        doc["solver"] = t["solver"]
        doc["reliability_summary"] = t["reliability_summary"]
        doc["relationship"] = {k: manifest["relationship"].get(k) for k in (
            "kind", "n", "n_genotyped", "n_markers", "genotype_storage", "single_step_mode")}
        doc["pev_sampling"] = manifest["diagnostics"]["y"].get("pev_sampling")
    Path(a.out).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: doc[k] for k in ("size", "abp_run_exit_status", "abp_run_wall_s",
                                          "abp_run_peak_rss_gb")}, indent=1))
    print("\n".join(doc["log_tail"]))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
