#!/usr/bin/env python3
"""Real-data validation (gate G5) on the Holstein data of example 20 (round 15).

Data: ``examples/20_holstein_milk_real/data`` written by ``fetch_data.py`` (the
``milk`` and ``pedCowsR`` data sets of the R package pedigreemm 0.3-5, USDA AIPL
records; 3,397 lactations of 1,359 cows in 57 herds; pedigree of 6,547 animals; sire
repair documented in the example's README).  Two parts:

1. **REML against an independent implementation.**  For three models ABP's REML
   (``abp.solvers.reml.reml_fit``) is compared with a dense marginal (V-form) REML
   written here from the model definition: ``A`` by the tabular method (not ABP's
   ``A^-1``), ``V = s_a Z A Z' + s_p Z_p Z_p' + s_e I``,
   ``-2 logL = log|V| + log|X'V^-1X| + y'Py`` with ``X`` of full column rank (QR),
   maximised by Nelder-Mead over ``sqrt`` of the variances (so that a variance can reach
   0).  Models: (a) first-lactation milk, herd fixed, animal; (b) somatic cell score,
   all lactations, herd + lactation fixed, animal + permanent environment; (c) milk, all
   lactations, the same repeatability model (ABP: additive variance at the boundary).
2. **Predictive ability by cross-validation over cows.**  The cows are split at random
   (seed 2026) into 5 folds; for each fold all records of its cows are hidden, the
   animal model (herd fixed) is fitted by REML on the first lactations of the other
   cows and BLUP gives EBVs for the hidden cows (from relatives only: no own record, no
   permanent-environment leakage).  The target is the hidden cow's mean yield over all
   her lactations adjusted for herd and lactation (least squares on all records).  Scores:
   correlation and regression of the target on the EBV over the 5 folds (bootstrap SE
   over cows, 2,000 resamples), for milk, fat and protein.

Usage: python benchmarks/real_milk_validation.py [--out docs/validation/real_milk_validation.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
import scipy.optimize as so  # noqa: E402
import scipy.sparse as sp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DATA = ROOT / "examples" / "20_holstein_milk_real" / "data"

from abp.core.pedigree import Pedigree  # noqa: E402
from abp.solvers.blup import RandomTerm, blup  # noqa: E402
from abp.solvers.reml import reml_fit  # noqa: E402


def load():
    if not (DATA / "lactations.csv").exists():
        sys.exit("run examples/20_holstein_milk_real/fetch_data.py first")
    prow = list(csv.DictReader(open(DATA / "pedigree.csv", encoding="utf-8")))
    ped = Pedigree.from_parent_ids([r["id"] for r in prow],
                                   [None if r["sire"] == "0" else r["sire"] for r in prow],
                                   [None if r["dam"] == "0" else r["dam"] for r in prow])
    recs = list(csv.DictReader(open(DATA / "lactations.csv", encoding="utf-8")))
    return prow, ped, recs


def design(recs, factors):
    """Intercept + dummies of the factors, dependent columns dropped by pivoted QR."""
    import scipy.linalg as sla
    cols = [np.ones(len(recs))]
    for f in factors:
        lev = sorted({r[f] for r in recs})
        for lv in lev[1:]:
            cols.append(np.array([r[f] == lv for r in recs], dtype=float))
    X = np.column_stack(cols)
    _, R, piv = sla.qr(X, mode="economic", pivoting=True)
    rank = int(np.sum(np.abs(np.diag(R)) > 1e-9 * abs(R[0, 0])))
    return X[:, np.sort(piv[:rank])]


def tabular_a(prow):
    """Numerator relationship matrix by the tabular method (Henderson 1976), dense."""
    n = len(prow)
    pos = {r["id"]: i for i, r in enumerate(prow)}
    s = [pos.get(r["sire"], -1) for r in prow]
    d = [pos.get(r["dam"], -1) for r in prow]
    A = np.zeros((n, n))
    for i in range(n):
        si, di = s[i], d[i]
        if i:
            row = np.zeros(i)
            if si >= 0:
                row += 0.5 * A[si, :i]
            if di >= 0:
                row += 0.5 * A[di, :i]
            A[i, :i] = row
            A[:i, i] = row
        A[i, i] = 1.0 + (0.5 * A[si, di] if si >= 0 and di >= 0 else 0.0)
    return A


def vform_reml(y, X, covs):
    """Maximise the V-form REML log-likelihood over sqrt-variances (Nelder-Mead)."""
    n = y.size

    def m2ll(x):
        th = np.asarray(x) ** 2
        V = sum(t * C for t, C in zip(th, covs))
        try:
            L = np.linalg.cholesky(V)
        except np.linalg.LinAlgError:
            return 1e300
        Vi_X = np.linalg.solve(L.T, np.linalg.solve(L, X))
        Vi_y = np.linalg.solve(L.T, np.linalg.solve(L, y))
        XVX = X.T @ Vi_X
        b = np.linalg.solve(XVX, X.T @ Vi_y)
        Py = Vi_y - Vi_X @ b
        return (2.0 * np.sum(np.log(np.diag(L))) + np.linalg.slogdet(XVX)[1]
                + float(y @ Py))
    return m2ll, n


def fit_vform(y, X, covs, start):
    f, _ = vform_reml(y, X, covs)
    best, runs = None, []
    # two starts away from the supplied point (half and twice every variance): each
    # dense evaluation costs about 2 s with 3,397 records, so a third start is not run
    for scale in (0.5, 2.0):
        x0 = np.sqrt(np.asarray(start) * scale)
        r = so.minimize(f, x0, method="Nelder-Mead",
                        options={"xatol": 1e-6, "fatol": 1e-7, "maxiter": 3000,
                                 "maxfev": 6000})
        runs.append({"start_scale": scale, "loglik": -0.5 * float(r.fun),
                     "variances": (r.x ** 2).tolist(), "evaluations": int(r.nfev)})
        if best is None or r.fun < best.fun:
            best = r
    return best.x ** 2, -0.5 * best.fun, runs


def terms_for(ped, recs, with_pe):
    n = len(recs)
    cols = ped.index_of([r["id"] for r in recs])
    Z = sp.csr_matrix((np.ones(n), (np.arange(n), cols)), shape=(n, ped.n))
    t = [RandomTerm("animal", Z, ped.ainv(), ped.ids, True, k_diag=1.0 + ped.inbreeding(),
                    logdet_k=ped.logdet_a())]
    if with_pe:
        cows = sorted({r["id"] for r in recs})
        p = {c: i for i, c in enumerate(cows)}
        Zp = sp.csr_matrix((np.ones(n), (np.arange(n), [p[r["id"]] for r in recs])),
                           shape=(n, len(cows)))
        t.append(RandomTerm("pe", Zp, sp.identity(len(cows), format="csr"), cows, False,
                            k_diag=np.ones(len(cows)), logdet_k=0.0))
    return t


def reml_comparison(prow, ped, recs, A):
    out = {}
    pos = {r["id"]: i for i, r in enumerate(prow)}
    cfg = {"algorithm": "ai", "max_iter": 200, "tol": 1e-9, "start": None}
    cases = {"milk_first_lactation": ("milk", [r for r in recs if r["lact"] == "1"],
                                      ["herd"], False),
             "scs_repeatability": ("scs", recs, ["herd", "lact"], True),
             "milk_repeatability": ("milk", recs, ["herd", "lact"], True)}
    for name, (trait, rr, fac, pe) in cases.items():
        t0 = time.time()
        y = np.array([float(r[trait]) for r in rr])
        X = design(rr, fac)
        terms = terms_for(ped, rr, pe)
        fit = reml_fit(y, sp.csr_matrix(X), terms, cfg)
        t_abp = time.time() - t0
        idx = np.array([pos[r["id"]] for r in rr])
        covs = [A[np.ix_(idx, idx)]]
        if pe:
            cow = np.array([r["id"] for r in rr])
            covs.append((cow[:, None] == cow[None, :]).astype(float))
        covs.append(np.eye(y.size))
        names = ["animal"] + (["pe"] if pe else []) + ["residual"]
        start = [max(fit.variances[k], 0.05 * fit.variances["residual"]) for k in names]
        t0 = time.time()
        th, ll, runs = fit_vform(y, X, covs, start)
        out[name] = {"n_records": int(y.size), "abp": {"status": fit.status,
                                                        "variances": fit.variances,
                                                        "loglik": fit.loglik,
                                                        "seconds": t_abp},
                     "vform": {"variances": dict(zip(names, map(float, th))), "loglik": ll,
                               "starts": runs,
                               "seconds": time.time() - t0},
                     "loglik_difference_abp_minus_vform": fit.loglik - ll}
        print(name, json.dumps(out[name]["abp"]["variances"]), json.dumps(out[name]["vform"]),
              flush=True)
    return out


def adjusted_targets(recs, trait):
    """Mean over a cow's lactations of y adjusted for herd and lactation (LS on all)."""
    X = design(recs, ["herd", "lact"])
    y = np.array([float(r[trait]) for r in recs])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    acc: dict = {}
    for r, v in zip(recs, e):
        acc.setdefault(r["id"], []).append(v)
    return {c: float(np.mean(v)) for c, v in acc.items()}


def cross_validation(ped, recs, folds=5, seed=2026):
    rng = np.random.default_rng(seed)
    cows = sorted({r["id"] for r in recs})
    fold = {c: k for c, k in zip(cows, rng.permutation(np.arange(len(cows)) % folds))}
    cfg = {"algorithm": "ai", "max_iter": 200, "tol": 1e-9, "start": None}
    out = {}
    for trait in ("milk", "fat", "prot"):
        target = adjusted_targets(recs, trait)
        ebv_all, tgt_all, h2 = [], [], []
        for k in range(folds):
            train = [r for r in recs if r["lact"] == "1" and fold[r["id"]] != k]
            test = [c for c in cows if fold[c] == k]
            y = np.array([float(r[trait]) for r in train])
            X = sp.csr_matrix(design(train, ["herd"]))
            terms = terms_for(ped, train, False)
            fit = reml_fit(y, X, terms, cfg)
            vc = fit.variances
            h2.append(vc["animal"] / (vc["animal"] + vc["residual"]))
            res = blup(y, X, terms, vc, compute_pev=False)
            sol = res.terms["animal"].solution
            idx = ped.index_of(test)
            ebv_all.extend(sol[idx])
            tgt_all.extend(target[c] for c in test)
        e, t = np.array(ebv_all), np.array(tgt_all)
        r = float(np.corrcoef(e, t)[0, 1])
        slope = float(np.cov(e, t)[0, 1] / np.var(e, ddof=1))
        bs = np.random.default_rng(seed + 1)
        rb, sb = [], []
        for _ in range(2000):
            i = bs.integers(0, e.size, e.size)
            rb.append(np.corrcoef(e[i], t[i])[0, 1])
            sb.append(np.cov(e[i], t[i])[0, 1] / np.var(e[i], ddof=1))
        out[trait] = {"n_cows": int(e.size), "correlation": r,
                      "correlation_se": float(np.std(rb, ddof=1)),
                      "regression_slope": slope, "slope_se": float(np.std(sb, ddof=1)),
                      "h2_first_lactation_by_fold": h2,
                      "n_cows_with_nonzero_ebv": int(np.sum(np.abs(e) > 0))}
        print(trait, json.dumps(out[trait]), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "validation" /
                                         "real_milk_validation.json"))
    a = ap.parse_args()
    t0 = time.time()
    prow, ped, recs = load()
    A = tabular_a(prow)
    doc = {"data": "pedigreemm 0.3-5 'milk' and 'pedCowsR' (USDA AIPL Holstein records), "
                   "via examples/20_holstein_milk_real/fetch_data.py",
           "n_records": len(recs), "n_animals": ped.n,
           "reml_vs_independent_vform": reml_comparison(prow, ped, recs, A),
           "cross_validation": cross_validation(ped, recs)}
    doc["wall_seconds"] = time.time() - t0
    Path(a.out).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
