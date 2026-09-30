#!/usr/bin/env python3
"""Calibration study of reduced-rank multi-trait REML (round 6, finding F10).

Two traits whose genetic covariance matrix has rank 1 (genetic correlation
exactly 1): ``G0 = lambda lambda'`` with ``lambda = (1.4, 0.6)``, residual
covariance ``R0 = [[3.0, 0.9], [0.9, 2.0]]``.  Per replicate: a random
sex-consistent pedigree (``--animals``, 10% founders), one record per animal,
trait 2 missing for 30% of the records, one mean per trait; the breeding values
are simulated by gene dropping ``f = T D^{1/2} z``, ``u = f lambda'`` (exactly
``N(0, A (x) G0)``).  Evaluated:

* ``full_rank``   multi-trait AI-REML (how it ends: converged / ``ABP-E300`` / other);
* ``rank1_reml``  reduced-rank REML with rank 1; estimates of G0 and R0, then BLUP
                  (``loadings``) and, for all animals, MSE / mean PEV and the
                  coverage of nominal 95% intervals per trait;
* ``rank1_true``  the same BLUP with the true lambda and R0 (reference).

Means +- Monte-Carlo SE over replicates.  Truth is used only for scoring.

Usage: python benchmarks/rr_calibration_study.py [--replicates 100] [--workers 4]
       [--animals 1000] [--out docs/validation/rr_calibration_study.json]
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
from abp.errors import ABPError  # noqa: E402
from abp.solvers.multitrait import MTData, build_and_solve  # noqa: E402
from abp.solvers.multitrait_reml import mt_reml_fit, mt_reml_fit_reduced_rank  # noqa: E402

LAM = np.array([[1.4], [0.6]])
G0 = LAM @ LAM.T
R0 = np.array([[3.0, 0.9], [0.9, 2.0]])
N_ANIMALS = 1000


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
    f = spsolve_triangular(ped._l_matrix(), np.sqrt(ped.mendelian_d()) * rng.standard_normal(n),
                           lower=True, unit_diagonal=True)
    U = f[:, None] * LAM[:, 0][None, :]                     # pedigree order, n x 2
    E = rng.standard_normal((n, 2)) @ np.linalg.cholesky(R0).T
    Y = 10.0 + U + E
    Y[rng.random(n) < 0.3, 1] = np.nan
    Xs = [sp.csr_matrix(np.ones((int((~np.isnan(Y[:, j])).sum()), 1))) for j in range(2)]
    return ped, MTData(Y, Xs, np.arange(n)), U


def _scores(res, U):
    out = {}
    for j in range(2):
        err = U[:, j] - res.ebv[:, j]
        pev = res.pev_blocks[:, j, j]
        out[f"trait{j + 1}"] = {"pev_ratio": float(np.mean(err ** 2) / np.mean(pev)),
                                "coverage95": float(np.mean(np.abs(err) <= 1.96 * np.sqrt(pev))),
                                "realized_accuracy": float(np.corrcoef(U[:, j],
                                                                       res.ebv[:, j])[0, 1])}
    return out


def replicate(seed: int) -> dict:
    ped, data, U = simulate(seed, N_ANIMALS)
    kd = 1.0 + ped.inbreeding()
    cfg = {"tol": 1e-8, "max_iter": 200}
    out = {"seed": seed}
    try:
        fr = mt_reml_fit(data, ped.ainv(), ped.logdet_a(), cfg)
        out["full_rank"] = {"status": "converged", "G0": fr.G0.tolist(),
                            "rG": float(fr.G0[0, 1] / np.sqrt(fr.G0[0, 0] * fr.G0[1, 1]))}
    except ABPError as exc:
        out["full_rank"] = {"status": exc.code}
    try:
        rr = mt_reml_fit_reduced_rank(data, ped.ainv(), ped.logdet_a(), 1, cfg)
        res = build_and_solve(data, ped.ainv(), kd, None, rr.R0, method="dense",
                              loadings=rr.loadings)
        out["rank1_reml"] = {"status": "converged", "G0": rr.G0.tolist(), "R0": rr.R0.tolist(),
                             "evaluations": rr.evaluations, **_scores(res, U)}
    except ABPError as exc:
        out["rank1_reml"] = {"status": exc.code}
    res_t = build_and_solve(data, ped.ainv(), kd, None, R0, method="dense", loadings=LAM)
    out["rank1_true"] = _scores(res_t, U)
    return out


def _ms(v):
    v = np.asarray(v, dtype=float)
    return {"mean": float(v.mean()), "mc_se": float(v.std(ddof=1) / np.sqrt(v.size)),
            "n": int(v.size)}


def main():
    global N_ANIMALS
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=100)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--animals", type=int, default=1000)
    ap.add_argument("--out", default=str(ROOT / "docs" / "validation" / "rr_calibration_study.json"))
    a = ap.parse_args()
    N_ANIMALS = a.animals
    t0 = time.time()
    seeds = list(range(1, a.replicates + 1))
    if a.workers > 1:
        from multiprocessing import Pool
        with Pool(a.workers) as pool:
            reps = list(pool.imap(replicate, seeds))
    else:
        reps = [replicate(s) for s in seeds]
    for r in reps:
        print(r["seed"], r["full_rank"]["status"], r["rank1_reml"]["status"], flush=True)
    ok = [r for r in reps if r["rank1_reml"]["status"] == "converged"]
    Gs = np.array([r["rank1_reml"]["G0"] for r in ok])
    Rs = np.array([r["rank1_reml"]["R0"] for r in ok])
    fr_status = {}
    for r in reps:
        fr_status[r["full_rank"]["status"]] = fr_status.get(r["full_rank"]["status"], 0) + 1
    summary = {"full_rank_status_counts": fr_status,
               "rank1_converged": len(ok),
               "rank1_G0": {f"G0[{i},{j}]": {**_ms(Gs[:, i, j]), "true": float(G0[i, j])}
                            for i in range(2) for j in range(i, 2)},
               "rank1_R0": {f"R0[{i},{j}]": {**_ms(Rs[:, i, j]), "true": float(R0[i, j])}
                            for i in range(2) for j in range(i, 2)}}
    for sc, rows in (("rank1_reml", [r["rank1_reml"] for r in ok]),
                     ("rank1_true", [r["rank1_true"] for r in reps])):
        summary[sc] = {tr: {k: _ms([row[tr][k] for row in rows])
                            for k in ("pev_ratio", "coverage95", "realized_accuracy")}
                       for tr in ("trait1", "trait2")}
    conv = [r["full_rank"]["rG"] for r in reps if r["full_rank"]["status"] == "converged"]
    if conv:
        summary["full_rank_converged_rG"] = _ms(conv)
    doc = {"study": "reduced-rank multi-trait REML, true G0 of rank 1",
           "replicates": a.replicates, "seeds": f"1..{a.replicates}", "animals": N_ANIMALS,
           "true": {"lambda": LAM[:, 0].tolist(), "G0": G0.tolist(), "R0": R0.tolist()},
           "wall_seconds": time.time() - t0, "summary": summary, "replicate_results": reps}
    Path(a.out).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
