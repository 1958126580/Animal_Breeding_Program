#!/usr/bin/env python3
"""Large matrix-free single step: 200,000 animals, 30,000 genotyped, int8 genotypes.

Synthetic problem at a size where the explicit single step cannot run on the
build machine (its n x n2 block alone would need 200,000 x 30,000 x 8 bytes =
48 GB).  Steps and what is measured (wall time, peak resident memory):

1. pedigree (``--gens`` x ``--per-gen``, discrete generations), inbreeding;
2. int8 genotypes of the last ``--genotyped`` animals, ``--markers`` SNPs;
3. APY blocks from the int8 dosages with the 5% A22 blend (``--core`` random core
   animals; A22 core columns by Colleau products in blocks), APY operator;
4. A22^-1 operator (sparse LDL' of A^11);
5. PCG solve of the MME (records on all non-founders, one contemporary-group factor);
6. sampled PEV with ``--pev-samples`` simulations (reliability and its MC SE).

Usage: python benchmarks/ssmf_large.py [--gens 20 --per-gen 10000 --genotyped 30000
       --markers 20000 --core 5000 --pev-samples 20] [--out benchmarks/results/ssmf_large.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=20)
    ap.add_argument("--per-gen", type=int, default=10000)
    ap.add_argument("--genotyped", type=int, default=30000)
    ap.add_argument("--markers", type=int, default=20000)
    ap.add_argument("--core", type=int, default=5000)
    ap.add_argument("--pev-samples", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "results" / "ssmf_large.json"))
    a = ap.parse_args()

    import numpy as np
    import scipy.sparse as sp

    from abp.core.design import FixedTerm, build_fixed_design
    from abp.core.pedigree import Pedigree
    from abp.core.ssop import (A22InverseOperator, APYOperator, SingleStepHInverse, a_block,
                               apy_blocks_from_dosage)
    from abp.solvers.blup import RandomTerm, blup
    from abp.solvers.pev_sampling import sampled_pev
    from abp.workflows.manifest import environment, peak_rss_bytes, utc_now
    from run_benchmarks import sim_pedigree

    steps = {}

    def step(name, fn):
        t = time.perf_counter()
        out = fn()
        steps[name] = {"wall_s": time.perf_counter() - t, "peak_rss_gb": peak_rss_bytes() / 2**30}
        print(name, json.dumps(steps[name]), flush=True)
        return out

    ids, s, d = sim_pedigree(a.gens, a.per_gen, 200, seed=5)
    ped = step("pedigree", lambda: Pedigree.from_parent_ids(ids, s, d))
    step("inbreeding", ped.inbreeding)
    n = ped.n
    rng = np.random.default_rng(3)
    g_index = np.sort(ped.index_of(ids[n - a.genotyped:]))
    p = rng.uniform(0.05, 0.95, a.markers)

    def genotypes():
        M = np.empty((a.genotyped, a.markers), dtype=np.int8)
        for i in range(0, a.genotyped, 1000):
            k = min(1000, a.genotyped - i)
            M[i:i + k] = (rng.random((k, a.markers)) < p) + (rng.random((k, a.markers)) < p)
        return M
    M = step("genotypes_int8", genotypes)
    core = np.sort(rng.choice(a.genotyped, a.core, replace=False))
    cols = step("a22_core_columns", lambda: a_block(ped, g_index, g_index[core]))
    def apy():
        blocks = apy_blocks_from_dosage(M, M.mean(axis=0) / 2, core, "blend", 0.05, 0.01, cols,
                                        1.0 + ped.inbreeding()[g_index])
        return APYOperator(*blocks, core, a.genotyped)
    g_op = step("apy_blocks_from_int8_and_operator", apy)
    M = cols = None                                  # release the genotypes and A columns
    a22 = step("a22_inverse_operator", lambda: A22InverseOperator(ped.ainv(), g_index))
    h = SingleStepHInverse(ped.ainv().tocsr(), g_index, g_op, a22, ped=ped)
    rec = np.arange(a.per_gen, n)
    y = rng.normal(0, 3, rec.size)
    cg = [f"c{k}" for k in rng.integers(0, 2000, rec.size)]
    fd = build_fixed_design({"cg": cg}, [FixedTerm("cg", "factor")], True, rec.size)
    Z = sp.csr_matrix((np.ones(rec.size), (np.arange(rec.size), rec)), shape=(rec.size, n))
    term = RandomTerm("animal", Z, h, ped.ids, True)
    res = step("pcg_solve", lambda: blup(y, fd.X, [term], {"animal": 4.0, "residual": 12.0},
                                         method="pcg", compute_pev=False, tol=1e-8,
                                         max_iter=20000))
    smp = step("sampled_pev", lambda: sampled_pev(rec.size, fd.X, [term],
                                                   {"animal": 4.0, "residual": 12.0}, "animal",
                                                   n_samples=a.pev_samples, tol=1e-8))
    geno = smp.reliability[g_index]
    doc = {"created_at": utc_now(), "environment": environment(),
           "size": {"animals": n, "genotyped": a.genotyped, "markers": a.markers,
                    "core": a.core, "records": int(rec.size),
                    "equations": int(res.solve.solution.size)},
           "pcg": {"iterations": res.solve.iterations,
                   "relative_residual": res.solve.rel_residual},
           "sampled_pev": {"n_samples": smp.n_samples,
                           "mean_pcg_iterations": float(np.mean(smp.pcg_iterations)),
                           "mean_reliability_genotyped": float(geno.mean()),
                           "mean_reliability_all": float(smp.reliability.mean()),
                           "mean_mc_se": float(smp.reliability_se.mean())},
           "steps": steps, "peak_rss_gb": peak_rss_bytes() / 2**30,
           "explicit_single_step_n_x_n2_block_gb": n * a.genotyped * 8 / 2**30}
    Path(a.out).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: doc[k] for k in ("size", "pcg", "sampled_pev", "peak_rss_gb")}))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
