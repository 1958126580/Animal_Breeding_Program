#!/usr/bin/env python3
"""ABP against the R packages rrBLUP and sommer on the real mouse data (round 16).

rrBLUP (Endelman 2011, Plant Genome 4:250) and sommer (Covarrubias-Pazaran 2016, PLoS ONE
11:e0156744) are widely used R packages for GBLUP by REML. ``benchmarks/r/mice_gblup_fit.R``
computes the genomic relationship matrix with ``rrBLUP::A.mat`` (VanRaden 2008, method 1;
independent of ABP's code) and fits, for body weight and body length,

* ``rrBLUP::mixed.solve``: sex + year x season fixed, animal (G + 0.01 I), residual;
* ``sommer::mmer``:        the same + cage as an independent random effect (example 21).

This script compares, against ABP's ``reml_fit`` + ``blup`` on the same models:
ABP's G (``vanraden_g`` with ``allele_frequencies``) with rrBLUP's ``A.mat``; variance
components; REML log-likelihoods where the packages report them on a comparable scale
(rrBLUP; sommer's ``monitor`` log-likelihood uses its own constants and is not compared);
and the additive BLUPs (GEBVs) of all mice.

Requirements for the R side (not ABP dependencies): R >= 4, rrBLUP 4.6.3 and sommer 4.3.6
built from their CRAN sources (sommer needs Rcpp, RcppArmadillo, RcppProgress, Matrix,
MASS, crayon; Ubuntu ``r-cran-*`` packages); point ``R_LIBS`` at the library.

Usage: R_LIBS=<lib> python benchmarks/real_mice_software_comparison.py
       [--r-dir <dir with G_rrBLUP.bin and r_results.json from an earlier run>]
       [--out docs/validation/real_mice_software.json]
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
    os.environ.setdefault(_var, "2")

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))
DATA = ROOT / "examples" / "21_mice_bodyweight_real" / "data"

from abp.core.genomic import allele_frequencies, spd_inverse_and_logdet, vanraden_g  # noqa: E402
from abp.solvers.blup import RandomTerm, blup  # noqa: E402
from abp.solvers.reml import reml_fit  # noqa: E402
from real_mice_validation import design, load  # noqa: E402

CFG = {"algorithm": "ai", "max_iter": 200, "tol": 1e-10, "start": None}


def abp_fit(ph, trait, Kinv, logdet, with_cage):
    n = len(ph)
    y = np.array([float(r[trait]) for r in ph])
    X = sp.csr_matrix(design(ph, list(range(n))))
    tt = [RandomTerm("animal", sp.identity(n, format="csr"), Kinv, [r["id"] for r in ph],
                     True, k_diag=np.ones(n), logdet_k=logdet)]
    if with_cage:
        lv = sorted({r["cage"] for r in ph})
        p = {c: k for k, c in enumerate(lv)}
        Zc = sp.csr_matrix((np.ones(n), (np.arange(n), [p[r["cage"]] for r in ph])),
                           shape=(n, len(lv)))
        tt.append(RandomTerm("cage", Zc, sp.identity(len(lv), format="csr"), lv, False,
                             k_diag=np.ones(len(lv)), logdet_k=0.0))
    t0 = time.time()
    f = reml_fit(y, X, tt, CFG)
    b = blup(y, X, tt, f.variances, compute_pev=False)
    return f, b.terms["animal"].solution, time.time() - t0, n, X.shape[1]


def compare_ebv(a, r):
    return {"correlation": float(np.corrcoef(a, r)[0, 1]),
            "max_abs_difference": float(np.max(np.abs(a - r))),
            "max_abs_difference_over_sd": float(np.max(np.abs(a - r)) / np.std(r))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "validation" / "real_mice_software.json"))
    ap.add_argument("--r-dir", default=None)
    ap.add_argument("--rscript", default="Rscript")
    a = ap.parse_args()
    ph, M, _ = load()
    n = len(ph)
    with tempfile.TemporaryDirectory() as td:
        rdir = Path(a.r_dir) if a.r_dir else Path(td)
        r_wall = None
        if not a.r_dir:
            t0 = time.time()
            subprocess.run([a.rscript, str(ROOT / "benchmarks" / "r" / "mice_gblup_fit.R"),
                            str(DATA), str(rdir)], check=True)
            r_wall = time.time() - t0
        R = json.loads((rdir / "r_results.json").read_text())
        G_r = np.fromfile(rdir / "G_rrBLUP.bin", dtype=np.float64).reshape(n, n, order="F")
    G, _ = vanraden_g(M, allele_frequencies(M))
    Kinv, logdet = spd_inverse_and_logdet(G + 0.01 * np.eye(n), "G")
    doc = {"data": "BGLR 1.1.4 'mice' via examples/21_mice_bodyweight_real/fetch_data.py",
           "r_versions": R["versions"], "r_wall_seconds": r_wall,
           "G_abp_vs_rrBLUP_A_mat": {"max_abs_difference": float(np.max(np.abs(G - G_r))),
                                     "mean_diag_abp": float(np.mean(np.diag(G))),
                                     "mean_diag_rrBLUP": float(np.mean(np.diag(G_r)))},
           "traits": {}}
    ids = [r["id"] for r in ph]
    for trait in ("bw", "body_length"):
        out = {}
        for key, with_cage in (("rrBLUP_animal_only", False), ("sommer_animal_cage", True)):
            f, u, sec, nn, p = abp_fit(ph, trait, Kinv, logdet, with_cage)
            Rm = R[trait][key]
            rv = {k: float(v) for k, v in Rm["variances"].items()}
            e_r = np.array([float(Rm["ebv"][i]) for i in ids])
            row = {"abp": {"status": f.status, "variances": f.variances, "loglik": f.loglik,
                           "seconds": sec},
                   "r": {"variances": rv, "loglik": float(Rm["loglik"]),
                         "seconds": float(Rm["seconds"])},
                   "relative_difference_of_variances": {
                       k: (f.variances[k] - rv[k]) / rv[k] for k in rv},
                   "ebv": compare_ebv(u, e_r)}
            if key == "rrBLUP_animal_only":
                const = 0.5 * (nn - p) * math.log(2 * math.pi)
                row["loglik_abp_minus_r"] = f.loglik - float(Rm["loglik"])
                row["loglik_abp_minus_r_without_2pi_constant"] = (
                    f.loglik - (float(Rm["loglik"]) + const))
            out[key] = row
            print(trait, key, json.dumps(row, default=float), flush=True)
        doc["traits"][trait] = out
    Path(a.out).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
