#!/usr/bin/env python3
"""Calibration study: two categorical traits in one multi-trait threshold model (round 14).

Per replicate: a random sex-consistent pedigree (``--animals``, 10% founders); on every
non-founder a three-category trait (liability thresholds -0.3 and 0.7), a binary trait
(threshold 0.2) and a continuous trait; breeding values by gene dropping from

    G0 = [[0.25, 0.10, 0.50], [0.10, 0.16, 0.40], [0.50, 0.40, 4.00]]

(liability h2 0.2 and 0.14; genetic correlation 0.5 between the two categorical traits,
0.5 between each and the continuous trait), residuals independent (``R0 = diag(1, 1, 8)``,
the model's assumption: every trait in its own residual group).  Fitted:

* ``mt``  multi-trait threshold model with both categorical traits
          (``mt_threshold_gibbs(Y, [0, 1], ...)``, residual groups ``[[0], [1], [2]]``),
          inverse Wishart prior ``IW(5, 5 G_prior)``, ``G_prior = diag(0.2, 0.2, 3.0)``
          (zero covariances: the prior does not pull towards the true correlations);
* ``st``  two single-trait threshold models (``threshold_gibbs``), scaled inverse
          chi-square prior ``nu = 5, s2 = 0.2`` (the same marginal prior).

Scored with the true values (never used by the fits): posterior means of the (co)variances
and genetic correlations, MSE / mean posterior variance, coverage of 95% intervals and
realized accuracy of the EBVs (all animals) per categorical trait.  Chains: 4 x
``--iterations`` (burn-in 1/5, thin 2); R-hat/ESS recorded, every fit scored.  Means
+- Monte-Carlo SE over replicates.

Usage: python benchmarks/two_categorical_study.py [--replicates 30] [--workers 3]
       [--animals 800] [--iterations 3000] [--out docs/validation/two_categorical_study.json]
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

G0 = np.array([[0.25, 0.10, 0.50], [0.10, 0.16, 0.40], [0.50, 0.40, 4.00]])
R0 = np.diag([1.0, 1.0, 8.0])
CUTS = ([-0.3, 0.7], [0.2])
G_PRIOR = np.diag([0.2, 0.2, 3.0])
N_ANIMALS, ITER = 800, 3000


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
                           * rng.standard_normal((n, 3)), lower=True, unit_diagonal=True)
    U = F @ np.linalg.cholesky(G0).T
    L = np.array([0.0, 0.0, 30.0]) + U + rng.standard_normal((n, 3)) * np.sqrt(np.diag(R0))
    rec = np.arange(nf, n)
    Y = np.column_stack([np.digitize(L[rec, 0], CUTS[0]) + 1.0,
                         np.digitize(L[rec, 1], CUTS[1]) + 1.0, L[rec, 2]])
    return ped, rec, Y, U


def _score(U, ebv, pev):
    err = U - ebv
    return {"realized_accuracy": float(np.corrcoef(U, ebv)[0, 1]),
            "pev_ratio": float(np.mean(err ** 2) / np.mean(pev)),
            "coverage95": float(np.mean(np.abs(err) <= 1.96 * np.sqrt(pev)))}


def replicate(seed: int) -> dict:
    ped, rec, Y, U = simulate(seed, N_ANIMALS)
    n, kd = ped.n, 1.0 + ped.inbreeding()
    X = [sp.csr_matrix(np.ones((rec.size, 1)))] * 3
    it = ITER
    out = {"seed": seed}
    t0 = time.time()
    mt = mt_threshold_gibbs(Y, [0, 1], X, rec, ped.ainv(), MTThresholdGibbsConfig(
        chains=4, iterations=it, burn_in=it // 5, thin=2, max_iterations=it, seed=seed,
        prior_nu=5.0, prior_G0=G_PRIOR, residual_groups=[[0], [1], [2]]))
    out["mt"] = {"converged": mt.converged, "wall_s": time.time() - t0,
                 "max_rhat": float(max(v["rhat"] for v in mt.summaries.values())),
                 "G0": mt.G0["mean"],
                 "rG": {k: mt.derived[k]["mean"] for k in ("rG_0_1", "rG_0_2", "rG_1_2")},
                 "rG_cover95": {k: bool(mt.derived[k]["q025"] <= t <= mt.derived[k]["q975"])
                                for k, t in (("rG_0_1", G0[0, 1] / np.sqrt(G0[0, 0] * G0[1, 1])),
                                             ("rG_0_2", G0[0, 2] / np.sqrt(G0[0, 0] * G0[2, 2])),
                                             ("rG_1_2", G0[1, 2] / np.sqrt(G0[1, 1] * G0[2, 2])))},
                 "thresholds": {str(c): v.tolist() for c, v in mt.thresholds_by_trait.items()},
                 "cat3": _score(U[:, 0], mt.ebv[:, 0], mt.pev[:, 0]),
                 "bin": _score(U[:, 1], mt.ebv[:, 1], mt.pev[:, 1])}
    Z = sp.csr_matrix((np.ones(rec.size), (np.arange(rec.size), rec)), shape=(rec.size, n))
    for j, key in ((0, "cat3"), (1, "bin")):
        t0 = time.time()
        st = threshold_gibbs(Y[:, j], sp.csr_matrix(np.ones((rec.size, 1))),
                             [RandomTerm("animal", Z, ped.ainv(), ped.ids, True, k_diag=kd)],
                             True, ThresholdGibbsConfig(chains=4, iterations=it,
                                                        burn_in=it // 5, thin=2,
                                                        max_iterations=it, seed=seed + 1000 * j,
                                                        nu=5.0, s2={"animal": 0.2}))
        g = st.terms["animal"]
        out[f"st_{key}"] = {"converged": st.converged, "wall_s": time.time() - t0,
                            "var": st.variances["animal"]["mean"],
                            "score": _score(U[:, j], g.solution, g.pev)}
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
    ap.add_argument("--out", default=str(ROOT / "docs" / "validation" /
                                         "two_categorical_study.json"))
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
         "mt_max_rhat_median": float(np.median([r["mt"]["max_rhat"] for r in reps]))}
    for i, j in ((0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2)):
        s[f"G0_{i}_{j}"] = {**_ms([r["mt"]["G0"][i][j] for r in reps]), "true": float(G0[i, j])}
    for k in ("rG_0_1", "rG_0_2", "rG_1_2"):
        i, j = int(k[3]), int(k[5])
        s[k] = {**_ms([r["mt"]["rG"][k] for r in reps]),
                "true": float(G0[i, j] / np.sqrt(G0[i, i] * G0[j, j])),
                "interval_coverage95": _ms([r["mt"]["rG_cover95"][k] for r in reps])}
    for key in ("cat3", "bin"):
        s[f"mt_{key}"] = {k: _ms([r["mt"][key][k] for r in reps]) for k in reps[0]["mt"][key]}
        s[f"st_{key}"] = {k: _ms([r[f"st_{key}"]["score"][k] for r in reps])
                          for k in reps[0][f"st_{key}"]["score"]}
        s[f"accuracy_gain_{key}_mt_minus_st"] = _ms(
            [r["mt"][key]["realized_accuracy"] - r[f"st_{key}"]["score"]["realized_accuracy"]
             for r in reps])
    doc = {"study": "two categorical traits in one multi-trait threshold model vs single-trait "
                    "threshold models", "replicates": a.replicates, "seeds": f"1..{a.replicates}",
           "animals": N_ANIMALS, "iterations": ITER,
           "true": {"G0": G0.tolist(), "R0": R0.tolist(), "cuts": [list(c) for c in CUTS]},
           "g_prior": G_PRIOR.tolist(), "wall_seconds": time.time() - t0, "summary": s,
           "replicate_results": reps}
    Path(a.out).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(s, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
