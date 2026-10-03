#!/usr/bin/env python3
"""Real-data validation (gate G5) on the heterogeneous-stock mice of example 21 (round 15).

Data: ``examples/21_mice_bodyweight_real/data`` written by ``fetch_data.py`` (the
``mice`` data of the R package BGLR 1.1.4: 1,814 mice, 10,346 SNPs, pedigree
relationship matrix ``mice.A``; Valdar et al. 2006; Legarra et al. 2008).

Models, for body weight and body length: sex and test year x season fixed, cage as an
independent random effect, and an additive effect with covariance

* ``gblup``  VanRaden G (allele frequencies of all genotyped mice) + 0.01 I;
* ``pblup``  the pedigree relationship matrix shipped with the data;
* ``fixed``  no additive effect (cage only) - the baseline.

1. REML on all records: variances, heritabilities (additive / (additive + cage +
   residual)), log-likelihoods (the two additive models have the same number of
   parameters, so their logL compare directly).
2. 5-fold cross-validation over mice (random folds, seed 2026): REML and BLUP on the
   training mice; for the hidden mice the prediction is the additive BLUP (no own
   record); the target is the hidden mouse's phenotype minus the training fixed-effect
   solution.  Scores: correlation (predictive ability) and regression slope over the 5
   folds, bootstrap SE over mice (2,000 resamples), paired difference gblup - pblup.
3. The same with whole families hidden together (``across_family``: families are the
   connected components of A >= 0.25, assigned to 5 folds at random, seed 2028).  The
   pedigree matrix of these data links only full sibs, so pedigree EBVs of a hidden
   family are 0 and only GBLUP can predict across families.

Usage: python benchmarks/real_mice_validation.py [--out docs/validation/real_mice_validation.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DATA = ROOT / "examples" / "21_mice_bodyweight_real" / "data"

from abp.core.genomic import allele_frequencies, spd_inverse_and_logdet, vanraden_g  # noqa: E402
from abp.solvers.blup import RandomTerm, blup  # noqa: E402
from abp.solvers.reml import reml_fit  # noqa: E402

CFG = {"algorithm": "ai", "max_iter": 200, "tol": 1e-9, "start": None}


def load():
    if not (DATA / "phenotypes.csv").exists():
        sys.exit("run examples/21_mice_bodyweight_real/fetch_data.py first")
    ph = list(csv.DictReader(open(DATA / "phenotypes.csv", encoding="utf-8")))
    with open(DATA / "genotypes.csv", encoding="utf-8") as fh:
        rd = csv.reader(fh)
        next(rd)
        rows = [r for r in rd]
    ids = [r[0] for r in rows]
    M = np.array([[float(v) for v in r[1:]] for r in rows])
    if ids != [r["id"] for r in ph]:
        sys.exit("genotype and phenotype order differ")
    A = np.load(DATA / "pedigree_A.npy")
    return ph, M, A


def design(ph, rows):
    cols = [np.ones(len(rows))]
    for f in ("sex", "year_season"):
        lev = sorted({ph[i][f] for i in rows})
        for lv in lev[1:]:
            cols.append(np.array([ph[i][f] == lv for i in rows], dtype=float))
    X = np.column_stack(cols)
    keep = np.linalg.matrix_rank(X) == X.shape[1]
    if not keep:
        raise RuntimeError("fixed design not of full rank")
    return X


def terms(ph, rows, n, Kinv, logdet):
    m = len(rows)
    out = []
    if Kinv is not None:
        Z = sp.csr_matrix((np.ones(m), (np.arange(m), rows)), shape=(m, n))
        out.append(RandomTerm("animal", Z, Kinv, [str(i) for i in range(n)], True,
                              k_diag=np.ones(n), logdet_k=logdet))
    cages = sorted({ph[i]["cage"] for i in rows})
    p = {c: k for k, c in enumerate(cages)}
    Zc = sp.csr_matrix((np.ones(m), (np.arange(m), [p[ph[i]["cage"]] for i in rows])),
                       shape=(m, len(cages)))
    out.append(RandomTerm("cage", Zc, sp.identity(len(cages), format="csr"), cages, False,
                          k_diag=np.ones(len(cages)), logdet_k=0.0))
    return out


def fit(ph, rows, trait, n, Kinv, logdet):
    y = np.array([float(ph[i][trait]) for i in rows])
    X = design(ph, rows)
    tt = terms(ph, rows, n, Kinv, logdet)
    f = reml_fit(y, sp.csr_matrix(X), tt, CFG)
    r = blup(y, sp.csr_matrix(X), tt, f.variances, compute_pev=False)
    return f, r, X


def _corr(e, t):
    """Correlation; None when the EBVs are constant (no information)."""
    if np.std(e) < 1e-12 * max(1.0, np.std(t)):
        return None
    return float(np.corrcoef(e, t)[0, 1])


def cross_validate(ph, n, trait, fold, structures):
    preds = {k: np.zeros(n) for k in ("gblup", "pblup")}
    target = np.zeros(n)
    for k in range(5):
        train = [i for i in range(n) if fold[i] != k]
        test = np.flatnonzero(fold == k)
        for name in ("gblup", "pblup"):
            Ki, ld = structures[name]
            f, r, X = fit(ph, train, trait, n, Ki, ld)
            preds[name][test] = r.terms["animal"].solution[test]
            if name == "gblup":
                # fixed-effect solution applied to the hidden mice (same coding)
                levs = {fct: sorted({ph[i][fct] for i in train})
                        for fct in ("sex", "year_season")}
                for i in test:
                    x = [1.0]
                    for fct in ("sex", "year_season"):
                        x += [float(ph[i][fct] == lv) for lv in levs[fct][1:]]
                    target[i] = float(ph[i][trait]) - float(np.array(x) @ r.fixed_solution)
    res = {}
    bs = np.random.default_rng(2027)
    boot_idx = [bs.integers(0, n, n) for _ in range(2000)]
    for name, e in preds.items():
        rr = _corr(e, target)
        if rr is None:
            res[name] = {"correlation": None, "sd_ebv": float(np.std(e)),
                         "note": "EBVs of hidden mice are constant: no relationship "
                                 "with the training mice in this matrix"}
            continue
        rb = [_corr(e[i], target[i]) for i in boot_idx]
        res[name] = {"correlation": rr, "correlation_se": float(np.std(rb, ddof=1)),
                     "regression_slope": float(np.cov(e, target)[0, 1] / np.var(e, ddof=1))}
    if res["pblup"]["correlation"] is not None:
        diff = [_corr(preds["gblup"][i], target[i]) - _corr(preds["pblup"][i], target[i])
                for i in boot_idx]
        res["gblup_minus_pblup"] = {
            "correlation": res["gblup"]["correlation"] - res["pblup"]["correlation"],
            "se": float(np.std(diff, ddof=1))}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "validation" /
                                         "real_mice_validation.json"))
    a = ap.parse_args()
    t0 = time.time()
    ph, M, A = load()
    n = len(ph)
    p = allele_frequencies(M)
    G, d = vanraden_g(M, p)
    G = G + 0.01 * np.eye(n)
    structures = {}
    for name, K in (("gblup", G), ("pblup", A)):
        Ki, ld = spd_inverse_and_logdet(K, name)
        structures[name] = (Ki, ld)
    structures["fixed"] = (None, None)
    doc = {"data": "BGLR 1.1.4 'mice' (Valdar et al. 2006), via "
                   "examples/21_mice_bodyweight_real/fetch_data.py",
           "n_mice": n, "n_snps": int(M.shape[1]), "vanraden_d": float(d),
           "mean_diag_G": float(np.mean(np.diag(G))), "full_data": {}, "cross_validation": {}}
    allrows = list(range(n))
    for trait in ("bw", "body_length"):
        doc["full_data"][trait] = {}
        for name, (Ki, ld) in structures.items():
            f, _, _ = fit(ph, allrows, trait, n, Ki, ld)
            vc = f.variances
            tot = sum(vc.values())
            doc["full_data"][trait][name] = {"variances": vc, "loglik": f.loglik,
                                             "status": f.status,
                                             "h2": vc.get("animal", 0.0) / tot,
                                             "cage_share": vc["cage"] / tot}
            print(trait, name, json.dumps(doc["full_data"][trait][name]), flush=True)
    rng = np.random.default_rng(2026)
    designs = {"random": rng.permutation(np.arange(n) % 5)}
    # families: connected components of A >= 0.25 (the pedigree matrix of these data is
    # block diagonal by full-sib family); whole families are hidden together
    nfam, fam = connected_components(sp.csr_matrix(A >= 0.25), directed=False)
    fam_fold = np.random.default_rng(2028).permutation(np.arange(nfam) % 5)
    designs["across_family"] = fam_fold[fam]
    doc["n_families"] = int(nfam)
    for dname, fold in designs.items():
        doc["cross_validation"][dname] = {}
        for trait in ("bw", "body_length"):
            res = cross_validate(ph, n, trait, fold, structures)
            doc["cross_validation"][dname][trait] = res
            print(dname, trait, json.dumps(res), flush=True)
    doc["wall_seconds"] = time.time() - t0
    Path(a.out).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
