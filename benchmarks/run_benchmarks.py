#!/usr/bin/env python3
"""Reproducible performance measurements for ABP kernels (synthetic inputs).

Usage:  python benchmarks/run_benchmarks.py [--full] [--only GROUP ...] [--out FILE]
        python benchmarks/run_benchmarks.py --preset ci [--out FILE]

``--preset ci`` (round 16) runs a fixed set of mid-size cases (a few minutes) used in
continuous integration on Windows and Linux, so that the job logs carry comparable
timings for both platforms.

Groups: pedigree, blup, reml, g (round 1); bayes, ocs, upg, plink (round 2);
selinv, metafounders (round 3); ldl, apy (round 4).

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


GROUPS = ("pedigree", "blup", "reml", "g", "bayes", "ocs", "upg", "plink", "selinv", "metafounders", "ldl", "apy", "ci")


def case_selinv_pev(n_anim, n_rec, n_check=300):
    """Exact PEV by selected inversion; spot-check against unit-vector solves."""
    ped, y, fd, term = _animal_problem(n_anim, n_rec, 500)
    res, t = timed(lambda: blup(y, fd.X, [term], {"animal": 1.0, "residual": 3.0},
                                method="sparse_direct", compute_pev=True,
                                memory_budget_bytes=8 * 2**30))
    fac = res.solve.factor
    si = fac.selected_inverse()
    rng = np.random.default_rng(1)
    a, _ = res.system.offsets["animal"]
    idx = rng.choice(ped.n, n_check, replace=False)
    E = np.zeros((fac.n, n_check))
    E[a + idx, np.arange(n_check)] = 1.0
    direct = fac.solve(E)[a + idx, np.arange(n_check)]
    err = float(np.max(np.abs(res.terms["animal"].pev[idx] - direct)))
    return {"case": f"blup_sparse_direct_exact_pev_{ped.n}animals", "n_animals": ped.n,
            "n_records": n_rec, "n_equations": int(res.solve.solution.size),
            "wall_s_total": t, "nnz_factor": si.nnz_factor, "kernel": si.kernel,
            "max_abs_diff_vs_solves_300_equations": err,
            "mean_reliability": float(res.terms["animal"].reliability.mean())}


def case_reml_sparse(n_anim, n_rec):
    out = case_reml(n_anim, n_rec)
    out["case"] = f"reml_ai_sparse_selinv_{out['n_animals']}animals"
    return out


def case_metafounders(n_gen, per_gen, k):
    from abp.core.metafounders import MetafounderPedigree
    from abp.core.upg import GroupAssignment
    ids, s, d = sim_pedigree(n_gen, per_gen, 100, seed=21)
    ped = Pedigree.from_parent_ids(ids, s, d)
    rng = np.random.default_rng(3)
    sg = np.where(ped.sire < 0, rng.integers(0, k, ped.n), -1)
    dg = np.where(ped.dam < 0, rng.integers(0, k, ped.n), -1)
    W = rng.normal(size=(k, 3 * k))
    gamma = 0.3 * np.eye(k) + 0.05 * W @ W.T / (3 * k)
    grp = GroupAssignment(tuple(f"MF{j}" for j in range(k)), sg, dg)
    out = {"case": f"metafounders_{ped.n}animals_{k}mf", "n_animals": ped.n}
    for name, env in (("native", None), ("python", "1")):
        if env:
            os.environ["ABP_DISABLE_NATIVE"] = env
        try:
            mfp, t = timed(lambda: MetafounderPedigree(ped, grp, gamma))
            _, t_inv = timed(mfp.ainv_ext)
        finally:
            os.environ.pop("ABP_DISABLE_NATIVE", None)
        out[f"{name}_diag_s"] = t
        out[f"{name}_inverse_s"] = t_inv
        out[f"{name}_kernel"] = mfp.kernel
        out[f"{name}_mean_diag"] = float(mfp.adiag.mean())
    return out


def case_ldl_pev(n_anim, n_rec):
    """Exact PEV with ABP's LDL' (minimum degree) vs SuperLU on the same system."""
    ped, y, fd, term = _animal_problem(n_anim, n_rec, 500)
    out = {"case": f"exact_pev_ldl_vs_superlu_{ped.n}animals", "n_animals": ped.n}
    sols = {}
    for fac in ("ldl", "superlu"):
        res, t = timed(lambda: blup(y, fd.X, [term], {"animal": 1.0, "residual": 3.0},
                                    method="sparse_direct", compute_pev=True,
                                    memory_budget_bytes=8 * 2**30, factorization=fac))
        out[f"{fac}_wall_s"] = t
        out[f"{fac}_nnz_factor"] = res.solve.factor.selected_inverse().nnz_factor
        sols[fac] = res
    out["max_abs_diff_ebv"] = float(np.max(np.abs(sols["ldl"].terms["animal"].solution
                                                  - sols["superlu"].terms["animal"].solution)))
    out["max_abs_diff_pev"] = float(np.max(np.abs(sols["ldl"].terms["animal"].pev
                                                  - sols["superlu"].terms["animal"].pev)))
    return out


def case_apy(n, m, n_core):
    from abp.core.genomic import apy_inverse, spd_inverse_and_logdet
    rng = np.random.default_rng(13)
    p = rng.uniform(0.05, 0.95, m)
    M = (rng.random((n, m)) < p).astype(float) + (rng.random((n, m)) < p)
    G, _ = vanraden_g(M, p)
    G = 0.99 * G + 0.01 * np.eye(n)
    (_, _), t_full = timed(lambda: spd_inverse_and_logdet(G, "G"))
    r, t_apy = timed(lambda: apy_inverse(G, np.arange(0, n, n // n_core)[:n_core]))
    return {"case": f"g_inverse_{n}genotyped_{m}markers", "full_inverse_s": t_full,
            "apy_inverse_s": t_apy, "n_core": n_core, "min_m": float(r.m.min())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="include slow Python reference timings")
    ap.add_argument("--only", nargs="+", choices=GROUPS, default=[g for g in GROUPS if g != "ci"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--preset", choices=("ci",), default=None,
                    help="'ci': fixed mid-size cases for Windows/Linux CI timings")
    args = ap.parse_args()
    if args.preset == "ci":
        args.only = ["ci"]
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
        ("selinv", lambda: case_selinv_pev(100000, 80000)),
        ("selinv", lambda: case_reml_sparse(20000, 16000)),
        ("metafounders", lambda: case_metafounders(20, 5000, 5)),
        ("ldl", lambda: case_ldl_pev(100000, 80000)),
        ("apy", lambda: case_apy(8000, 10000, 2000)),
        ("ci", lambda: case_pedigree(10, 10000, False)),
        ("ci", lambda: case_blup(5000, 4000, "dense", True)),
        ("ci", lambda: case_reml(3000, 2500)),
        ("ci", lambda: case_selinv_pev(20000, 16000)),
        ("ci", lambda: case_reml_sparse(20000, 16000)),
        ("ci", lambda: case_g(2000, 10000)),
        ("ci", lambda: case_apy(4000, 10000, 1000)),
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
