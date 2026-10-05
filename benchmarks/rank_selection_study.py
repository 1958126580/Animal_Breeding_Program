#!/usr/bin/env python3
"""Rank-selection study: what the EBVs of the chosen model are worth (round 8).

``reml.rank_selection = "aic"`` (``select_rank``) fits every rank of the genetic
covariance matrix and keeps a lower rank only when its AIC is smaller by at least
2.  Round 7 counted the chosen ranks; this study scores the BLUP of the model that
the rule actually chooses - including the full-rank fits kept when the truth has a
reduced rank - with the plug-in PEV and with the Kackar-Harville PEV, next to BLUP
with the true parameters.

Scenarios (``G0 = L L'``, the simulated rank is the number of columns of ``L``):

* ``t2_rank1``  two traits, rank 1 (genetic correlation 1), as in round 7;
* ``t2_full``   two traits, full rank (genetic correlation 0.595);
* ``t3_rank2``  three traits, rank 2;
* ``t3_full``   three traits, full rank.

Per replicate: random sex-consistent pedigree (``--animals``, 10% founders), one
record per animal, traits after the first missing at random for 30% of records,
one mean per trait; breeding values by gene dropping ``F = T D^{1/2} Z``,
``U = F L'`` (exactly ``N(0, A (x) G0)``).  Scores per trait over all animals:
MSE/mean PEV, coverage of nominal 95% intervals, realized accuracy.  Truth is used
only for scoring.  Means +- Monte-Carlo SE over replicates.

Usage: python benchmarks/rank_selection_study.py [--scenarios t2_rank1 ...]
       [--replicates 100] [--workers 4] [--animals 1000]
       [--out docs/validation/rank_selection_study.json]
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
from abp.solvers.multitrait_reml import select_rank  # noqa: E402
from abp.solvers.vc_uncertainty import (kackar_harville_delta_multitrait,  # noqa: E402
                                        kackar_harville_delta_reduced_rank)

L_T2_RANK1 = np.array([[1.4], [0.6]])
L_T2_FULL = np.linalg.cholesky(np.array([[1.96, 0.5], [0.5, 0.36]]))
R0_T2 = np.array([[3.0, 0.9], [0.9, 2.0]])
L_T3_RANK2 = np.array([[1.4, 0.0], [0.6, 0.5], [0.3, -0.6]])
L_T3_FULL = np.linalg.cholesky(np.array([[1.96, 0.5, 0.3], [0.5, 0.6, 0.1], [0.3, 0.1, 0.5]]))
R0_T3 = np.array([[3.0, 0.9, 0.4], [0.9, 2.0, 0.3], [0.4, 0.3, 1.5]])
SCENARIOS = {"t2_rank1": (L_T2_RANK1, R0_T2), "t2_full": (L_T2_FULL, R0_T2),
             "t3_rank2": (L_T3_RANK2, R0_T3), "t3_full": (L_T3_FULL, R0_T3)}
N_ANIMALS = 1000
SCENARIO = "t2_rank1"                 # set by main(); inherited by forked workers


def simulate(seed: int, n: int, L: np.ndarray, R0: np.ndarray):
    from scipy.sparse.linalg import spsolve_triangular
    rng = np.random.default_rng(seed)
    t, r = L.shape
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
                           * rng.standard_normal((n, r)), lower=True, unit_diagonal=True)
    U = F.reshape(n, r) @ L.T
    E = rng.standard_normal((n, t)) @ np.linalg.cholesky(R0).T
    Y = 10.0 + U + E
    for j in range(1, t):
        Y[rng.random(n) < 0.3, j] = np.nan
    Xs = [sp.csr_matrix(np.ones((int((~np.isnan(Y[:, j])).sum()), 1))) for j in range(t)]
    return ped, MTData(Y, Xs, np.arange(n)), U


def _scores(res, U, delta=None):
    out = {}
    for j in range(U.shape[1]):
        err = U[:, j] - res.ebv[:, j]
        pev = res.pev_blocks[:, j, j] + (0.0 if delta is None else delta[:, j, j])
        out[f"trait{j + 1}"] = {"pev_ratio": float(np.mean(err ** 2) / np.mean(pev)),
                                "coverage95": float(np.mean(np.abs(err) <= 1.96 * np.sqrt(pev))),
                                "realized_accuracy": float(np.corrcoef(U[:, j],
                                                                       res.ebv[:, j])[0, 1])}
    return out


def replicate(seed: int) -> dict:
    L, R0 = SCENARIOS[SCENARIO]
    t = L.shape[0]
    ped, data, U = simulate(seed, N_ANIMALS, L, R0)
    ainv, kd = ped.ainv(), 1.0 + ped.inbreeding()
    out = {"seed": seed}
    try:
        sel = select_rank(data, ainv, ped.logdet_a(), {"tol": 1e-8, "max_iter": 200})
    except ABPError as exc:
        out["selection"] = {"status": exc.code}
        return out
    chosen = sel["chosen_rank"]
    fit = sel["fits"][chosen]
    out["selection"] = {"status": "ok", "chosen_rank": chosen,
                        "table": [{k: v for k, v in r_.items() if k != "error"}
                                  | ({"error": r_["error"]["code"]} if "error" in r_ else {})
                                  for r_ in sel["table"]]}
    if chosen == t:
        res = build_and_solve(data, ainv, kd, fit.G0, fit.R0, method="dense")
        delta = (kackar_harville_delta_multitrait(data, ainv, kd, fit.G0, fit.R0, fit.cov,
                                                  method="dense")
                 if fit.cov is not None else None)
    else:
        res = build_and_solve(data, ainv, kd, None, fit.R0, method="dense",
                              loadings=fit.loadings)
        delta = (kackar_harville_delta_reduced_rank(data, ainv, kd, fit.x, t, chosen,
                                                    fit.cov_x, method="dense")
                 if fit.cov_x is not None else None)
    out["chosen_plugin"] = _scores(res, U)
    if delta is not None:
        out["chosen_kh"] = _scores(res, U, delta)
    if L.shape[1] == t:
        res_t = build_and_solve(data, ainv, kd, L @ L.T, R0, method="dense")
    else:
        res_t = build_and_solve(data, ainv, kd, None, R0, method="dense", loadings=L)
    out["true_parameters"] = _scores(res_t, U)
    return out


def _ms(v):
    v = np.asarray(v, dtype=float)
    return {"mean": float(v.mean()),
            "mc_se": float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else None,
            "n": int(v.size)}


def summarize(reps, t):
    ok = [r for r in reps if r["selection"]["status"] == "ok"]
    counts = {}
    for r in reps:
        k = str(r["selection"].get("chosen_rank", r["selection"]["status"]))
        counts[k] = counts.get(k, 0) + 1
    s = {"chosen_rank_counts": counts}
    groups = {"all": ok}
    for rank in sorted({r["selection"]["chosen_rank"] for r in ok}):
        groups[f"chosen_rank_{rank}"] = [r for r in ok if r["selection"]["chosen_rank"] == rank]
    for g, rows in groups.items():
        s[g] = {}
        for sc in ("chosen_plugin", "chosen_kh", "true_parameters"):
            rr = [r[sc] for r in rows if sc in r]
            if rr:
                s[g][sc] = {f"trait{j + 1}": {k: _ms([x[f"trait{j + 1}"][k] for x in rr])
                                              for k in ("pev_ratio", "coverage95",
                                                        "realized_accuracy")}
                            for j in range(t)}
    return s


def main():
    global N_ANIMALS, SCENARIO
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS))
    ap.add_argument("--replicates", type=int, default=100)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--animals", type=int, default=1000)
    ap.add_argument("--out", default=str(ROOT / "docs" / "validation" /
                                         "rank_selection_study.json"))
    a = ap.parse_args()
    N_ANIMALS = a.animals
    doc = {"study": "BLUP of the model chosen by select_rank (AIC, margin 2)",
           "replicates": a.replicates, "seeds": f"1..{a.replicates}", "animals": N_ANIMALS,
           "scenarios": {}}
    for name in a.scenarios:
        SCENARIO = name
        L, R0 = SCENARIOS[name]
        t0 = time.time()
        seeds = list(range(1, a.replicates + 1))
        if a.workers > 1:
            from multiprocessing import Pool
            with Pool(a.workers) as pool:
                reps = list(pool.imap(replicate, seeds))
        else:
            reps = [replicate(s) for s in seeds]
        summ = summarize(reps, L.shape[0])
        doc["scenarios"][name] = {"true": {"G0": (L @ L.T).tolist(), "rank": int(L.shape[1]),
                                           "R0": R0.tolist()},
                                  "wall_seconds": time.time() - t0, "summary": summ,
                                  "replicate_results": reps}
        print(name, json.dumps(summ["chosen_rank_counts"]), flush=True)
        for g in summ:
            if g == "chosen_rank_counts":
                continue
            for sc, tr in summ[g].items():
                print(f"  {g:>14} {sc:>15}", " ".join(
                    f"{k}: MSE/PEV {v['pev_ratio']['mean']:.3f} cov {v['coverage95']['mean']:.3f}"
                    f" acc {v['realized_accuracy']['mean']:.3f}" for k, v in tr.items()),
                    flush=True)
    Path(a.out).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
