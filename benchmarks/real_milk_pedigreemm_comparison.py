#!/usr/bin/env python3
"""ABP against the R package pedigreemm on the real Holstein data (round 16).

pedigreemm (Bates & Vazquez; Vazquez et al. 2010, J Anim Sci 88:497) fits animal models
through lme4 with the relationship matrix entering as a Cholesky factor of A; it was
published with these very data (``milk``). This script fits the three models of
``real_milk_validation.py`` with ABP (``reml_fit`` + ``blup``) and with pedigreemm
(``benchmarks/r/pedigreemm_fit.R``, REML) and compares variance components, REML
log-likelihoods and the additive BLUPs (EBVs) of the cows with records.

Requirements for the R side (not ABP dependencies): R >= 4, lme4 and Matrix (e.g. Ubuntu
``r-base-core r-cran-lme4``), and pedigreemm built from its CRAN sources
(``git clone --branch 0.3-5 https://github.com/cran/pedigreemm && R CMD INSTALL -l <lib>
pedigreemm``); point ``R_LIBS`` at <lib>.

The likelihoods are compared up to the constant ``(n - rank X) log(2 pi) / 2``, which
lme4's REML criterion includes and ABP's log-likelihood omits (checked here, not assumed:
both differences are reported).

Usage: R_LIBS=<lib> python benchmarks/real_milk_pedigreemm_comparison.py
       [--out docs/validation/real_milk_pedigreemm.json]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))
DATA = ROOT / "examples" / "20_holstein_milk_real" / "data"

from abp.solvers.blup import blup  # noqa: E402
from abp.solvers.reml import reml_fit  # noqa: E402
from real_milk_validation import design, load, tabular_a, terms_for  # noqa: E402

CFG = {"algorithm": "ai", "max_iter": 200, "tol": 1e-10, "start": None}
CASES = {"milk_first_lactation": ("milk", True, ["herd"], False),
         "scs_repeatability": ("scs", False, ["herd", "lact"], True),
         "milk_repeatability": ("milk", False, ["herd", "lact"], True)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "validation" /
                                         "real_milk_pedigreemm.json"))
    ap.add_argument("--rscript", default="Rscript")
    ap.add_argument("--r-json", default=None,
                    help="output of benchmarks/r/pedigreemm_fit.R from an earlier run "
                         "(skips the R fits)")
    a = ap.parse_args()
    prow, ped, recs = load()
    r_wall = None
    if a.r_json:
        R = json.loads(Path(a.r_json).read_text())
    else:
        with tempfile.TemporaryDirectory() as td:
            rjson = Path(td) / "pedigreemm.json"
            t0 = time.time()
            subprocess.run([a.rscript, str(ROOT / "benchmarks" / "r" / "pedigreemm_fit.R"),
                            str(DATA), str(rjson)], check=True)
            r_wall = time.time() - t0
            R = json.loads(rjson.read_text())
    doc = {"data": "pedigreemm 0.3-5 'milk' and 'pedCowsR' via "
                   "examples/20_holstein_milk_real/fetch_data.py (sire repair applied)",
           "r_versions": R["versions"], "r_wall_seconds": r_wall, "models": {}}
    for name, (trait, first_only, fac, pe) in CASES.items():
        rr = [r for r in recs if r["lact"] == "1"] if first_only else recs
        y = np.array([float(r[trait]) for r in rr])
        X = sp.csr_matrix(design(rr, fac))
        tt = terms_for(ped, rr, pe)
        t0 = time.time()
        fit = reml_fit(y, X, tt, CFG)
        b = blup(y, X, tt, fit.variances, compute_pev=False)
        t_abp = time.time() - t0
        ebv = dict(zip(b.terms["animal"].labels, b.terms["animal"].solution))
        Rm = R[name]
        rv = {("animal" if k == "id" else k): float(v) for k, v in Rm["variances"].items()}
        n, rank = len(y), int(Rm["rank_x"])
        ll_r = -0.5 * float(Rm["reml_criterion"])
        const = 0.5 * (n - rank) * math.log(2 * math.pi)
        ids = sorted(Rm["ebv"])
        e_r = np.array([float(Rm["ebv"][i]) for i in ids])
        e_a = np.array([ebv[i] for i in ids])
        e_rf = np.array([float(Rm["ebv_ranef"][i]) for i in ids])
        doc["models"][name] = {
            "n_records": n, "rank_x_abp": int(X.shape[1]), "rank_x_r": rank,
            "abp": {"status": fit.status, "variances": fit.variances, "loglik": fit.loglik,
                    "seconds": t_abp},
            "pedigreemm": {"variances": rv, "loglik_reml": ll_r, "seconds": Rm["seconds"]},
            "relative_difference_of_variances": {
                k: (fit.variances[k] - rv[k]) / rv[k] for k in rv},
            "loglik_abp_minus_pedigreemm": fit.loglik - ll_r,
            "loglik_abp_minus_pedigreemm_without_2pi_constant": fit.loglik - (ll_r + const),
            "ebv": {"n_animals": len(ids), "correlation": float(np.corrcoef(e_a, e_r)[0, 1]),
                    "max_abs_difference": float(np.max(np.abs(e_a - e_r))),
                    "sd_ebv_pedigreemm": float(np.std(e_r)),
                    "max_abs_difference_over_sd": float(np.max(np.abs(e_a - e_r))
                                                         / np.std(e_r)),
                    "pedigreemm_ranef_as_returned": {
                        "note": "pedigreemm 0.3-5 ranef() = relfac %*% b; the additive BLUP "
                                "is t(relfac) %*% b (compared above)",
                        "correlation_with_abp": float(np.corrcoef(e_a, e_rf)[0, 1])}}}
        if first_only:      # dense V-form BLUP at ABP's variances (third implementation)
            A = tabular_a(prow)
            pos = {r["id"]: k for k, r in enumerate(prow)}
            idx = np.array([pos[r["id"]] for r in rr])
            Aj = A[np.ix_(idx, idx)]
            va, ve = fit.variances["animal"], fit.variances["residual"]
            Xd = X.toarray()
            Vi = np.linalg.inv(va * Aj + ve * np.eye(n))
            beta = np.linalg.solve(Xd.T @ Vi @ Xd, Xd.T @ Vi @ y)
            u_v = va * Aj @ Vi @ (y - Xd @ beta)
            e_v = np.array([ebv[r["id"]] for r in rr]) - u_v
            doc["models"][name]["ebv"]["abp_minus_vform_max_abs"] = float(np.max(np.abs(e_v)))
        print(name, json.dumps(doc["models"][name], default=float), flush=True)
    Path(a.out).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
