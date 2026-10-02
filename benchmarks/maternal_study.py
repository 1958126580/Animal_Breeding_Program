#!/usr/bin/env python3
"""Calibration study of the Bayesian maternal animal model (round 11).

Per replicate: a random sex-consistent pedigree (``--animals``, 10% founders, dams
drawn among the earlier females, so dams have several offspring); one record per
non-founder animal: ``y = mu + a_animal + m_dam + c_dam + e`` with
``(a, m) ~ N(0, A (x) G0)``, ``G0 = [[4, -1], [-1, 2]]`` (direct-maternal correlation
-0.35), maternal permanent environment ``c`` (variance 1.5), residual 8 (the values
of example 18).  Fitted with ``mt_threshold_gibbs`` (one trait, ``dam_col``, iid term
on the dam), flat priors, 4 chains x ``--iterations`` (burn-in 1/4, thin 4); whether
R-hat/ESS passed is recorded and every fit is scored.

Scores (truth used only for scoring): posterior means of the variances; for the
direct and the maternal EBVs of all animals: MSE / mean PEV, coverage of nominal 95%
intervals, realized accuracy.  Means +- Monte-Carlo SE over replicates.

Usage: python benchmarks/maternal_study.py [--replicates 20] [--workers 3]
       [--animals 1500] [--iterations 4000] [--out docs/validation/maternal_study.json]
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
from abp.solvers.mt_threshold_gibbs import MTThresholdGibbsConfig, mt_threshold_gibbs  # noqa: E402

G0 = np.array([[4.0, -1.0], [-1.0, 2.0]])
VAR_C, VAR_E = 1.5, 8.0
N_ANIMALS, ITER = 1500, 4000


def simulate(seed: int, n: int):
    from scipy.sparse.linalg import spsolve_triangular
    rng = np.random.default_rng(seed)
    ids = [f"a{i}" for i in range(n)]
    male = np.arange(n) % 2 == 0
    nf = n // 10
    sires, dams = [], []
    for i in range(n):
        if i < nf:
            sires.append(None)
            dams.append(None)
        else:
            sires.append(ids[rng.choice(np.flatnonzero(male[:i]))])
            dams.append(ids[rng.choice(np.flatnonzero(~male[:i]))])
    ped = Pedigree.from_parent_ids(ids, sires, dams)
    F = spsolve_triangular(ped._l_matrix(), np.sqrt(ped.mendelian_d())[:, None]
                           * rng.standard_normal((n, 2)), lower=True, unit_diagonal=True)
    AM = F @ np.linalg.cholesky(G0).T
    dam = np.asarray(ped.dam, dtype=np.int64)
    rec = np.flatnonzero(dam >= 0)
    c = rng.normal(0, np.sqrt(VAR_C), n)
    y = 30 + AM[rec, 0] + AM[dam[rec], 1] + c[dam[rec]] + rng.normal(0, np.sqrt(VAR_E), rec.size)
    return ped, rec, dam, y, AM


def _score(true, ebv, pev):
    err = true - ebv
    return {"pev_ratio": float(np.mean(err ** 2) / np.mean(pev)),
            "coverage95": float(np.mean(np.abs(err) <= 1.96 * np.sqrt(pev))),
            "realized_accuracy": float(np.corrcoef(true, ebv)[0, 1])}


def replicate(seed: int) -> dict:
    ped, rec, dam, y, AM = simulate(seed, N_ANIMALS)
    damrec = dam[rec]
    levels, pe_col = np.unique(damrec, return_inverse=True)
    t0 = time.time()
    g = mt_threshold_gibbs(y[:, None], None, sp.csr_matrix(np.ones((rec.size, 1))), rec,
                           ped.ainv(), MTThresholdGibbsConfig(
                               chains=4, iterations=ITER, burn_in=ITER // 4, thin=4,
                               max_iterations=ITER, seed=seed),
                           pe_col=pe_col, dam_col=damrec)
    return {"seed": seed, "converged": g.converged, "wall_s": time.time() - t0,
            "max_rhat": float(max(v["rhat"] for v in g.summaries.values())),
            "min_ess_bulk": float(min(v["ess_bulk"] for v in g.summaries.values())),
            "G0": g.G0["mean"], "P0": g.P0["mean"][0][0], "R0": g.R0["mean"][0][0],
            "direct": _score(AM[:, 0], g.ebv[:, 0], g.pev[:, 0]),
            "maternal": _score(AM[:, 1], g.maternal_ebv[:, 0], g.maternal_pev[:, 0])}


def _ms(v):
    v = np.asarray(v, dtype=float)
    return {"mean": float(v.mean()), "mc_se": float(v.std(ddof=1) / np.sqrt(v.size)),
            "n": int(v.size)}


def main():
    global N_ANIMALS, ITER
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=20)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--animals", type=int, default=1500)
    ap.add_argument("--iterations", type=int, default=4000)
    ap.add_argument("--out", default=str(ROOT / "docs" / "validation" / "maternal_study.json"))
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
    s = {"converged": sum(r["converged"] for r in reps),
         "direct_variance": {**_ms([r["G0"][0][0] for r in reps]), "true": 4.0},
         "maternal_variance": {**_ms([r["G0"][1][1] for r in reps]), "true": 2.0},
         "direct_maternal_covariance": {**_ms([r["G0"][0][1] for r in reps]), "true": -1.0},
         "maternal_pe_variance": {**_ms([r["P0"] for r in reps]), "true": VAR_C},
         "residual_variance": {**_ms([r["R0"] for r in reps]), "true": VAR_E}}
    for eff in ("direct", "maternal"):
        s[eff] = {k: _ms([r[eff][k] for r in reps]) for k in reps[0][eff]}
    doc = {"study": "Bayesian maternal animal model (direct + maternal genetic + maternal pe)",
           "replicates": a.replicates, "seeds": f"1..{a.replicates}", "animals": N_ANIMALS,
           "iterations": ITER, "true": {"G0": G0.tolist(), "var_c": VAR_C, "var_e": VAR_E},
           "wall_seconds": time.time() - t0, "summary": s, "replicate_results": reps}
    Path(a.out).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(s, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
