#!/usr/bin/env python3
"""Reproducible performance measurements for ABP kernels (synthetic inputs).

Usage:  python benchmarks/run_benchmarks.py [--full] [--only GROUP ...] [--out FILE]

Groups: pedigree, blup, reml, g (round 1); bayes, ocs, upg, plink (round 2).

Every case builds its own synthetic data from a fixed seed, times the step
with ``time.perf_counter`` (wall clock, single run unless stated), and checks
the numerical result where a cheap check exists.  ``--full`` adds the slow
pure-Python reference timings that motivated the C++ kernel (ADR 0002).
Numbers are only comparable on the recorded hardware/software.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abp.core.design import FixedTerm, build_fixed_design  # noqa: E402
from abp.core.genomic import allele_frequencies, vanraden_g  # noqa: E402
from abp.core.pedigree import Pedigree, inbreeding_meuwissen_luo  # noqa: E402
from abp.solvers.blup import RandomTerm, blup  # noqa: E402
from abp.solvers.reml import reml_fit  # noqa: E402
from abp.workflows.manifest import environment, peak_rss_bytes, utc_now  # noqa: E402


def sim_pedigree(n_gen, per_gen, n_sires, seed=1):
    rng = np.random.default_rng(seed)
    ids, sires, dams, prev_m, prev_f = [], [], [], [], []
    for g in range(n_gen):
        cur_m, cur_f = [], []
        for k in range(per_gen):
            a = f"g{g}_{k}"
            ids.append(a)
            if g == 0:
                sires.append(None)
                dams.append(None)
            else:
                sires.append(prev_m[rng.integers(min(n_sires, len(prev_m)))])
                dams.append(prev_f[rng.integers(len(prev_f))])
            (cur_m if k % 2 == 0 else cur_f).append(a)
        prev_m, prev_f = cur_m, cur_f
    return ids, sires, dams


def timed(fn):
    t = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t


def case_pedigree(n_gen, per_gen, python_too):
    ids, s, d = sim_pedigree(n_gen, per_gen, 50)
    ped, t_build = timed(lambda: Pedigree.from_parent_ids(ids, s, d))
    F, t_native = timed(ped.inbreeding)
    _, t_ainv = timed(ped.ainv)
    rec = {"case": f"pedigree_{len(ids)}x{n_gen}gen", "n_animals": len(ids),
           "generations": n_gen, "build_order_s": t_build, "inbreeding_s": t_native,
           "inbreeding_kernel": ped.inbreeding_kernel, "ainv_s": t_ainv,
           "ainv_nnz": int(ped.ainv().nnz), "mean_F": float(F.mean())}
    if python_too:
        Fp, t_py = timed(lambda: inbreeding_meuwissen_luo(ped.sire, ped.dam))
        rec["inbreeding_python_reference_s"] = t_py
        rec["max_abs_diff_native_vs_python"] = float(np.max(np.abs(Fp - F)))
    return rec


def _animal_problem(n_anim, n_rec, n_cg, seed=3):
    ids, s, d = sim_pedigree(10, n_anim // 10, 50, seed)  # 10 discrete generations
    ped = Pedigree.from_parent_ids(ids, s, d)
    rng = np.random.default_rng(seed)
    rec = rng.choice(np.arange(ped.n // 10, ped.n), n_rec, replace=False)
    cg = [f"c{k}" for k in rng.integers(0, n_cg, n_rec)]
    y = rng.normal(0, 3, n_rec)
    fd = build_fixed_design({"cg": cg}, [FixedTerm("cg", "factor")], True, n_rec)
    Z = sp.csr_matrix((np.ones(n_rec), (np.arange(n_rec), rec)), shape=(n_rec, ped.n))
    term = RandomTerm("animal", Z, ped.ainv(), ped.ids, True, k_diag=1 + ped.inbreeding(),
                      logdet_k=ped.logdet_a())
    return ped, y, fd, term


def case_blup(n_anim, n_rec, method, pev):
    ped, y, fd, term = _animal_problem(n_anim, n_rec, 500)
    res, t = timed(lambda: blup(y, fd.X, [term], {"animal": 1.0, "residual": 3.0}, method=method,
                                compute_pev=pev, tol=1e-10, memory_budget_bytes=8 * 2**30))
    return {"case": f"blup_{method}{'_pev' if pev else ''}_{ped.n}animals", "n_animals": ped.n,
            "n_records": n_rec, "n_equations": int(res.solve.solution.size), "method": method,
            "pev": pev, "wall_s": t, "relative_residual": res.solve.rel_residual,
            "iterations": res.solve.iterations}


def case_reml(n_anim, n_rec):
    ped, y, fd, term = _animal_problem(n_anim, n_rec, 50, seed=5)
    rng = np.random.default_rng(9)
    # simulate u ~ N(0, 2 A) through A = T D T', i.e. u = T (sqrt(2 D) z)
    from scipy.sparse.linalg import spsolve_triangular
    u = spsolve_triangular(ped._l_matrix(), np.sqrt(2.0 * ped.mendelian_d())
                           * rng.standard_normal(ped.n), lower=True, unit_diagonal=True)
    yy = term.Z @ u + rng.normal(0, np.sqrt(4.0), y.size) + fd.X @ rng.normal(0, 2, fd.X.shape[1])
    fit, t = timed(lambda: reml_fit(yy, fd.X, [term], {"algorithm": "ai", "max_iter": 100,
                                                       "tol": 1e-8, "start": None},
                                    memory_budget_bytes=8 * 2**30))
    return {"case": f"reml_ai_{ped.n}animals", "n_animals": ped.n, "n_records": int(y.size),
            "wall_s": t, "iterations": fit.iterations, "status": fit.status,
            "estimates": fit.variances, "simulated": {"animal": 2.0, "residual": 4.0}}


def case_g(n, m):
    rng = np.random.default_rng(11)
    p0 = rng.uniform(0.05, 0.95, m)
    M = (rng.random((n, m)) < p0).astype(float) + (rng.random((n, m)) < p0)
    p = allele_frequencies(M)
    (G, d), t = timed(lambda: vanraden_g(M, p))
    return {"case": f"g_matrix_{n}x{m}", "n_animals": n, "n_markers": m, "wall_s": t,
            "mean_diag": float(np.mean(np.diag(G)))}


def case_bayes_sweep(n, m, reps):
    """One Gibbs marker sweep (BayesC code path): C++ kernel vs Python reference."""
    from abp.core import pedigree as pmod
    from abp.solvers.bayes import sweep_python
    rng = np.random.default_rng(13)
    Wf = np.asfortranarray(rng.integers(0, 3, (n, m)).astype(float) - 1.0)
    wtw = np.einsum("ij,ij->j", Wf, Wf)
    args = (np.full(m, 0.01), np.log([0.95, 0.05]), np.array([0.0, 0.01]), 1.0, 1,
            rng.normal(size=m), rng.random(m))
    rec = {"case": f"bayes_sweep_{n}x{m}", "records": n, "markers": m}
    runs = [("python_reference", lambda e, b, d: sweep_python(Wf, wtw, e, b, d, *args))]
    if pmod._native is not None and hasattr(pmod._native, "bayes_sweep"):
        runs.append(("native_cpp", lambda e, b, d: pmod._native.bayes_sweep(Wf.T, wtw, e, b, d,
                                                                            *args)))
    outs = {}
    e0, b0, d0 = rng.normal(size=n), np.zeros(m), np.zeros(m, dtype=np.int64)   # same start
    for name, fn in runs:
        e, b, d = e0.copy(), b0.copy(), d0.copy()
        _, t = timed(lambda: [fn(e, b, d) for _ in range(reps)])
        rec[f"{name}_s_per_sweep"] = t / reps
        e, b, d = e0.copy(), b0.copy(), d0.copy()
        fn(e, b, d)
        outs[name] = b
    if len(outs) == 2:
        rec["max_abs_diff_native_vs_python"] = float(np.max(np.abs(outs["native_cpp"]
                                                                   - outs["python_reference"])))
        rec["speedup"] = rec["python_reference_s_per_sweep"] / rec["native_cpp_s_per_sweep"]
    return rec


def case_ocs(n_male, n_female, n_matings):
    from abp.decision.mating import allocate, forbidden_mask
    from abp.decision.ocs import coancestry_target_from_delta_f, integer_matings, solve_ocs
    ids, s, d = sim_pedigree(6, 2 * (n_male + n_female), 20, seed=17)
    ped = Pedigree.from_parent_ids(ids, s, d)
    last = ped.index_of(ids[-2 * (n_male + n_female):])
    male_all = np.arange(last.size) % 2 == 0
    cand = np.concatenate([last[male_all][:n_male], last[~male_all][:n_female]])
    male = np.arange(cand.size) < n_male
    A = ped.a_submatrix(cand)
    g = np.random.default_rng(19).normal(size=cand.size)
    cap = np.where(male, 20, 1)
    _, ct = coancestry_target_from_delta_f(A, 0.01)
    cmax = ct                                      # ceiling at the candidates' mean coancestry
    oc, t_ocs = timed(lambda: solve_ocs(g, A, male, np.zeros(cand.size), cap / (2.0 * n_matings),
                                        cmax))
    counts = integer_matings(oc.c, cap, male, n_matings)
    si, di = np.flatnonzero(male & (counts > 0)), np.flatnonzero(~male & (counts > 0))
    A_sd = A[np.ix_(si, di)]
    forb, _ = forbidden_mask(A_sd, 0.25, None, None, None)
    plan, t_mate = timed(lambda: allocate(counts[si], counts[di], A_sd, forb))
    return {"case": f"ocs_mating_{cand.size}candidates", "candidates": int(cand.size),
            "matings": n_matings, "ocs_s": t_ocs, "ocs_status": oc.status,
            "kkt_stationarity": oc.kkt["stationarity_max_abs"], "mating_lp_s": t_mate,
            "sires_x_dams": int(si.size * di.size), "plan_checks": plan.checks}


def case_upg(n_gen, per_gen, n_groups):
    from abp.core.upg import GroupAssignment, ainv_with_groups, group_fractions
    ids, s, d = sim_pedigree(n_gen, per_gen, 50)
    ped = Pedigree.from_parent_ids(ids, s, d)
    rng = np.random.default_rng(23)
    unknown = ped.sire < 0
    sg = np.where(unknown, rng.integers(0, n_groups, ped.n), -1)
    dg = np.where(ped.dam < 0, rng.integers(0, n_groups, ped.n), -1)
    grp = GroupAssignment(tuple(f"G{k}" for k in range(n_groups)), sg, dg)
    _, t_f = timed(ped.inbreeding)                # shared by both A-inverse variants (cached)
    M, t_ainv = timed(lambda: ainv_with_groups(ped, grp))
    Q, t_q = timed(lambda: group_fractions(ped, grp))
    _, t_plain = timed(ped.ainv)
    return {"case": f"upg_{ped.n}animals_{n_groups}groups", "n_animals": ped.n,
            "groups": n_groups, "inbreeding_s": t_f, "ainv_with_groups_s": t_ainv,
            "group_fractions_s": t_q,
            "plain_ainv_s": t_plain, "nnz": int(M.nnz),
            "q_rows_sum_to_one": bool(np.allclose(Q.sum(axis=1), 1.0))}


def case_plink(n, m):
    from abp.io.plink import decode_bed
    rng = np.random.default_rng(29)
    raw = bytes([0x6C, 0x1B, 0x01]) + rng.integers(0, 256, m * ((n + 3) // 4),
                                                    dtype=np.uint8).tobytes()
    M, t = timed(lambda: decode_bed(raw, n, m))
    return {"case": f"plink_decode_{n}x{m}", "samples": n, "variants": m, "wall_s": t,
            "bytes": len(raw), "missing_fraction": float(np.isnan(M).mean())}


GROUPS = ("pedigree", "blup", "reml", "g", "bayes", "ocs", "upg", "plink")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="include slow Python reference timings")
    ap.add_argument("--only", nargs="+", choices=GROUPS, default=list(GROUPS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    results = {"created_at": utc_now(), "environment": environment(), "groups": args.only,
               "cases": []}
    cases = [
        ("pedigree", lambda: case_pedigree(10, 1000, True)),
        ("pedigree", lambda: case_pedigree(10, 10000, args.full)),
        ("pedigree", lambda: case_pedigree(20, 5000, args.full)),
        ("blup", lambda: case_blup(5000, 4000, "dense", True)),
        ("blup", lambda: case_blup(100000, 80000, "sparse_direct", False)),
        ("blup", lambda: case_blup(100000, 80000, "pcg", False)),
        ("reml", lambda: case_reml(3000, 2500)),
        ("g", lambda: case_g(2000, 50000)),
        ("bayes", lambda: case_bayes_sweep(1000, 10000, 3)),
        ("bayes", lambda: case_bayes_sweep(5000, 50000, 1)),
        ("ocs", lambda: case_ocs(100, 400, 400)),
        ("ocs", lambda: case_ocs(300, 1200, 1200)),
        ("upg", lambda: case_upg(10, 10000, 50)),
        ("plink", lambda: case_plink(5000, 50000)),
    ]
    cases = [c for grp, c in cases if grp in args.only]
    for c in cases:
        rec = c()
        print(json.dumps(rec))
        results["cases"].append(rec)
    results["peak_rss_bytes"] = peak_rss_bytes()
    out = Path(args.out) if args.out else ROOT / "benchmarks" / "results" / \
        f"{time.strftime('%Y%m%d')}-{sys.platform}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
