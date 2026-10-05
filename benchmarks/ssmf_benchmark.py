#!/usr/bin/env python3
"""Explicit vs matrix-free single step: build time, solve time, peak memory.

One synthetic problem (random pedigree of ``--gens`` x ``--per-gen`` animals,
the last generations genotyped with ``--markers`` independent SNPs, records on
all non-founders, one contemporary-group factor).  Each mode runs in its own
process so that its peak resident memory (``ru_maxrss``) is measured alone:

* ``explicit``        dense G*^-1, dense A22 and A22^-1, n x n2 block for diag(H)
                      (:func:`abp.core.genomic.single_step`), PCG solve;
* ``explicit_apy``    the same with the dense APY inverse;
* ``matrix_free``     H^-1 as an operator with dense G*^-1 (:mod:`abp.core.ssop`);
* ``matrix_free_apy`` H^-1 as an operator with the APY operator.

G* = 0.95 G + 0.05 A22 (blend) in every mode.  The solutions of each mode are
compared with ``explicit`` (same model) and ``explicit_apy`` (APY model).

Usage: python benchmarks/ssmf_benchmark.py [--gens 10 --per-gen 5000 --genotyped 6000
       --markers 5000 --core 2000] [--out benchmarks/results/ssmf.json]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))
MODES = ("explicit", "explicit_apy", "matrix_free", "matrix_free_apy")


def run_mode(mode: str, a) -> dict:
    import numpy as np
    import scipy.sparse as sp

    from abp.core.design import FixedTerm, build_fixed_design
    from abp.core.genomic import (apply_g_policy, apy_inverse, centered, scaling_d,
                                  single_step)
    from abp.core.pedigree import Pedigree
    from abp.core.ssop import (A22InverseOperator, APYOperator, DenseInverseOperator,
                               SingleStepHInverse, apy_blocks_from_genotypes)
    from abp.core.genomic import spd_inverse_and_logdet
    from abp.solvers.blup import RandomTerm, blup
    from abp.workflows.manifest import peak_rss_bytes
    from run_benchmarks import sim_pedigree

    ids, s, d = sim_pedigree(a.gens, a.per_gen, 100, seed=7)
    ped = Pedigree.from_parent_ids(ids, s, d)
    rng = np.random.default_rng(11)
    n = ped.n
    g_index = np.sort(ped.index_of(ids[n - a.genotyped:]))
    p = rng.uniform(0.05, 0.95, a.markers)
    M = (rng.random((a.genotyped, a.markers)) < p).astype(np.float64) + \
        (rng.random((a.genotyped, a.markers)) < p)
    rec = np.arange(a.per_gen, n)
    y = rng.normal(0, 3, rec.size)
    cg = [f"c{k}" for k in rng.integers(0, 400, rec.size)]
    fd = build_fixed_design({"cg": cg}, [FixedTerm("cg", "factor")], True, rec.size)
    Z = sp.csr_matrix((np.ones(rec.size), (np.arange(rec.size), rec)), shape=(rec.size, n))
    core = np.sort(rng.choice(a.genotyped, a.core, replace=False))
    alpha = 0.05
    t0 = time.perf_counter()
    pf = M.mean(axis=0) / 2
    W = centered(M, pf)
    dd = scaling_d(pf)
    del M
    if mode.startswith("explicit"):
        G = W @ W.T / dd
        A22 = ped.a_submatrix(g_index)
        Gs, _ = apply_g_policy(G, "blend", "none", A22, alpha, 0.01, check_pd=(mode == "explicit"))
        del G
        ginv = None
        if mode == "explicit_apy":
            r = apy_inverse(Gs, core)
            ginv = (r.g_inv, r.logdet)
            Gs = r.g_apy
        ss = single_step(ped, g_index, Gs, a22=A22, max_dense_bytes=64 * 2**30, g_inverse=ginv)
        k_inv = ss.h_inv
    else:
        if mode == "matrix_free_apy":
            cols = ped.a_columns(g_index[core])[g_index]
            gcc, gcn, gnn = apy_blocks_from_genotypes(W, dd, core, "blend", alpha, 0.01, cols,
                                                      1.0 + ped.inbreeding()[g_index])
            g_op = APYOperator(gcc, gcn, gnn, core, a.genotyped)
        else:
            G = W @ W.T / dd
            Gs, _ = apply_g_policy(G, "blend", "none", ped.a_submatrix(g_index), alpha, 0.01)
            del G
            g_op = DenseInverseOperator(spd_inverse_and_logdet(Gs, "G*")[0])
            del Gs
        k_inv = SingleStepHInverse(ped.ainv().tocsr(), g_index, g_op,
                                   A22InverseOperator(ped.ainv(), g_index))
    t_build = time.perf_counter() - t0
    term = RandomTerm("animal", Z, k_inv, ped.ids, True)
    t1 = time.perf_counter()
    res = blup(y, fd.X, [term], {"animal": 4.0, "residual": 12.0}, method="pcg",
               compute_pev=False, tol=1e-10, max_iter=20000)
    t_solve = time.perf_counter() - t1
    sol_path = Path(a.tmp) / f"{mode}.npy"
    np.save(sol_path, res.terms["animal"].solution)
    return {"mode": mode, "n_animals": n, "n_genotyped": a.genotyped, "n_markers": a.markers,
            "n_core": a.core if mode.endswith("apy") else None, "n_records": int(rec.size),
            "n_equations": int(res.solve.solution.size), "build_s": t_build,
            "solve_s": t_solve, "pcg_iterations": res.solve.iterations,
            "relative_residual": res.solve.rel_residual, "peak_rss_bytes": peak_rss_bytes(),
            "solution_file": str(sol_path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=10)
    ap.add_argument("--per-gen", type=int, default=5000)
    ap.add_argument("--genotyped", type=int, default=6000)
    ap.add_argument("--markers", type=int, default=5000)
    ap.add_argument("--core", type=int, default=2000)
    ap.add_argument("--mode", choices=MODES)
    ap.add_argument("--tmp")
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "results" / "ssmf.json"))
    a = ap.parse_args()
    if a.mode:                                     # child process
        print(json.dumps(run_mode(a.mode, a)))
        return
    import numpy as np
    from abp.workflows.manifest import environment, utc_now
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for mode in MODES:
            cmd = [sys.executable, __file__, "--mode", mode, "--tmp", tmp, "--gens", str(a.gens),
                   "--per-gen", str(a.per_gen), "--genotyped", str(a.genotyped),
                   "--markers", str(a.markers), "--core", str(a.core)]
            out = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ))
            if out.returncode != 0:
                rows.append({"mode": mode, "failed": out.stderr.strip().splitlines()[-1:]})
                print(mode, "FAILED", out.stderr[-500:])
                continue
            r = json.loads(out.stdout.strip().splitlines()[-1])
            rows.append(r)
            print(json.dumps({k: v for k, v in r.items() if k != "solution_file"}))
        sols = {r["mode"]: np.load(r["solution_file"]) for r in rows if "solution_file" in r}
        for r in rows:
            ref = "explicit_apy" if r["mode"].endswith("apy") else "explicit"
            if r["mode"] in sols and ref in sols and ref != r["mode"]:
                r["max_abs_diff_vs_" + ref] = float(np.max(np.abs(sols[r["mode"]] - sols[ref])))
            r.pop("solution_file", None)
    doc = {"created_at": utc_now(), "environment": environment(), "cases": rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    for r in rows:
        print(r["mode"], {k: r.get(k) for k in ("build_s", "solve_s", "pcg_iterations",
                                                  "peak_rss_bytes") },
              {k: v for k, v in r.items() if k.startswith("max_abs")})
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
