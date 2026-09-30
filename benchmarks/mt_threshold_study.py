#!/usr/bin/env python3
"""Calibration study of the multi-trait threshold model (round 8).

Per replicate: a random sex-consistent pedigree (``--animals``, 10% founders);
a three-category trait (liability ``l = mu + u_1 + e_1``, thresholds at -0.3 and
0.7) recorded on the females (single record per animal) and a continuous trait
recorded on every non-founder; breeding values by gene dropping from
``G0 = [[0.25, 0.5], [0.5, 4.0]]`` (liability h2 0.2, genetic correlation 0.5),
residual ``R0 = [[1, 0.849], [0.849, 8]]`` (correlation 0.3).  Fitted:

* ``mt``  multi-trait threshold model (``mt_threshold_gibbs``), inverse Wishart prior
          ``IW(5, 5 G_prior)`` with ``G_prior = diag(0.2, 3.0)`` (near, not at, the truth);
* ``st``  single-trait threshold model on the categorical trait (``threshold_gibbs``),
          scaled inverse chi-square prior ``nu = 5, s2 = 0.2`` (the same marginal prior).

Scored with the true values (never used by the fits): posterior means of the
(co)variances, realized accuracy of the EBVs (all animals), MSE / mean posterior
variance and coverage of 95% intervals, and the ratio model / realized accuracy.
Chains: 4 x ``--iterations`` (burn-in 1/5, thin 2); whether R-hat/ESS passed is
recorded but results are scored either way (the workflow would withhold the
unconverged ones).  Means +- Monte-Carlo SE over replicates.

Usage: python benchmarks/mt_threshold_study.py [--replicates 30] [--workers 4]
       [--animals 800] [--iterations 3000] [--out docs/validation/mt_threshold_study.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abp.core.pedigree import Pedigree  # noqa: E402
from abp.solvers.blup import RandomTerm  # noqa: E402
from abp.solvers.mt_threshold_gibbs import MTThresholdGibbsConfig, mt_threshold_gibbs  # noqa: E402
from abp.solvers.threshold_gibbs import ThresholdGibbsConfig, threshold_gibbs  # noqa: E402

G0 = np.array([[0.25, 0.5], [0.5, 4.0]])
R0 = np.array([[1.0, 0.3 * np.sqrt(8.0)], [0.3 * np.sqrt(8.0), 8.0]])
CUTS = [-0.3, 0.7]
N_ANIMALS = 800
ITER = 3000


def simulate(seed: int, n: int):
    from scipy.sparse.linalg import spsolve_triangular
    rng = np.random.default_rng(seed)
    ids = [f"a{i}" for i in range(n)]
    male = np.arange(n) % 2 == 0
    nf = n // 10
    s, d = [], []
    for i in range(n):
        if i < nf:
            s.append(None)
            d.append(None)
        else:
            s.append(ids[rng.choice(np.flatnonzero(male[:i]))])
            d.append(ids[rng.choice(np.flatnonzero(~male[:i]))])
    ped = Pedigree.from_parent_ids(ids, s, d)
    F = spsolve_triangular(ped._l_matrix(), np.sqrt(ped.mendelian_d())[:, None]
                           * rng.standard_normal((n, 2)), lower=True, unit_diagonal=True)
    U = F @ np.linalg.cholesky(G0).T
    E = rng.standard_normal((n, 2)) @ np.linalg.cholesky(R0).T
    L = np.array([0.0, 30.0]) + U + E
    Y = np.full((n, 2), np.nan)
    rec = np.arange(nf, n)
    fem = rec[~male[rec]]
    Y[fem, 0] = np.digitize(L[fem, 0], CUTS) + 1
    Y[rec, 1] = L[rec, 1]
    keep = ~np.all(np.isnan(Y), axis=1)
    return ped, Y, U, keep


def _score(U, ebv, pev, kd, gjj):
    err = U - ebv
    rel = np.clip(1.0 - pev / (kd * gjj), 0, 1)
    acc = float(np.corrcoef(U, ebv)[0, 1])
    return {"realized_accuracy": acc, "pev_ratio": float(np.mean(err ** 2) / np.mean(pev)),
            "coverage95": float(np.mean(np.abs(err) <= 1.96 * np.sqrt(pev))),
            "model_accuracy": float(np.mean(np.sqrt(rel))),
            "model_to_realized": float(np.mean(np.sqrt(rel)) / acc)}


def replicate(seed: int) -> dict:
    ped, Y, U, keep = simulate(seed, N_ANIMALS)
    n = ped.n
    kd = 1.0 + ped.inbreeding()
    rows = np.flatnonzero(keep)
    Yk = Y[rows]
    Xs = [sp.csr_matrix(np.ones((int((~np.isnan(Yk[:, j])).sum()), 1))) for j in range(2)]
    it = ITER
    out = {"seed": seed}
    t0 = time.time()
    mt = mt_threshold_gibbs(Yk, 0, Xs, rows, ped.ainv(), MTThresholdGibbsConfig(
        chains=4, iterations=it, burn_in=it // 5, thin=2, max_iterations=it, seed=seed,
        prior_nu=5.0, prior_G0=np.diag([0.2, 3.0])))
    Gm = np.array(mt.G0["mean"])
    out["mt"] = {"converged": mt.converged, "wall_s": time.time() - t0,
                 "G0": mt.G0["mean"], "R0": mt.R0["mean"],
                 "rG": mt.derived["rG_0_1"]["mean"], "h2_cat": mt.derived["h2_0"]["mean"],
                 "cat": _score(U[:, 0], mt.ebv[:, 0], mt.pev[:, 0], kd, Gm[0, 0]),
                 "cont": _score(U[:, 1], mt.ebv[:, 1], mt.pev[:, 1], kd, Gm[1, 1])}
    t0 = time.time()
    cat_rows = np.flatnonzero(~np.isnan(Y[:, 0]))
    Z = sp.csr_matrix((np.ones(cat_rows.size), (np.arange(cat_rows.size), cat_rows)),
                      shape=(cat_rows.size, n))
    st = threshold_gibbs(Y[cat_rows, 0], sp.csr_matrix(np.ones((cat_rows.size, 1))),
                         [RandomTerm("animal", Z, ped.ainv(), ped.ids, True, k_diag=kd)], True,
                         ThresholdGibbsConfig(chains=4, iterations=it, burn_in=it // 5, thin=2,
                                              max_iterations=it, seed=seed, nu=5.0,
                                              s2={"animal": 0.2}))
    g = st.terms["animal"]
    va = st.variances["animal"]["mean"]
    out["st"] = {"converged": st.converged, "wall_s": time.time() - t0, "var_cat": va,
                 "cat": _score(U[:, 0], g.solution, g.pev, kd, va)}
    return out


def _ms(v):
    v = np.asarray(v, dtype=float)
    return {"mean": float(v.mean()), "mc_se": float(v.std(ddof=1) / np.sqrt(v.size)),
            "n": int(v.size)}


def main():
    global N_ANIMALS, ITER
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=30)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--animals", type=int, default=800)
    ap.add_argument("--iterations", type=int, default=3000)
    ap.add_argument("--out", default=str(ROOT / "docs" / "validation" / "mt_threshold_study.json"))
    a = ap.parse_args()
    N_ANIMALS, ITER = a.animals, a.iterations
    t0 = time.time()
    seeds = list(range(1, a.replicates + 1))
    if a.workers > 1:
        from multiprocessing import Pool
        with Pool(a.workers) as pool:
            reps = list(pool.imap(replicate, seeds))
    else:
        reps = [replicate(s) for s in seeds]
    s = {"mt_converged": sum(r["mt"]["converged"] for r in reps),
         "st_converged": sum(r["st"]["converged"] for r in reps)}
    for k, (i, j) in {"G0_cat": (0, 0), "G0_cov": (0, 1), "G0_cont": (1, 1)}.items():
        s[f"mt_{k}"] = {**_ms([r["mt"]["G0"][i][j] for r in reps]), "true": float(G0[i, j])}
    for k, (i, j) in {"R0_cov": (0, 1), "R0_cont": (1, 1)}.items():
        s[f"mt_{k}"] = {**_ms([r["mt"]["R0"][i][j] for r in reps]), "true": float(R0[i, j])}
    s["mt_rG"] = {**_ms([r["mt"]["rG"] for r in reps]), "true": 0.5}
    s["st_var_cat"] = {**_ms([r["st"]["var_cat"] for r in reps]), "true": float(G0[0, 0])}
    for m, tr in (("mt", "cat"), ("mt", "cont"), ("st", "cat")):
        s[f"{m}_{tr}"] = {k: _ms([r[m][tr][k] for r in reps])
                          for k in reps[0][m][tr]}
    s["accuracy_gain_cat_mt_minus_st"] = _ms([r["mt"]["cat"]["realized_accuracy"]
                                              - r["st"]["cat"]["realized_accuracy"] for r in reps])
    doc = {"study": "multi-trait threshold model vs single-trait threshold model",
           "replicates": a.replicates, "seeds": f"1..{a.replicates}", "animals": N_ANIMALS,
           "iterations": ITER, "true": {"G0": G0.tolist(), "R0": R0.tolist(), "cuts": CUTS},
           "wall_seconds": time.time() - t0, "summary": s, "replicate_results": reps}
    Path(a.out).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(s, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
