#!/usr/bin/env python3
"""F22 factor study (round 16): why are GBLUP EBVs over-dispersed across families?

Data: example 21 (BGLR ``mice``; ``examples/21_mice_bodyweight_real/fetch_data.py``).
Design: the across-family 5-fold cross-validation of ``real_mice_validation.py``
(families = connected components of the pedigree matrix ``A >= 0.25``, 169 full-sib
families; folds seed 2028). For each variant the regression of the target (phenotype
minus the training fixed-effect solution) on the EBV of the hidden mice is computed;
1 is expected for unbiased, correctly dispersed EBVs (round 15: 0.66 for body weight,
0.52 for body length). Variants change one factor of the baseline:

* ``baseline``          sex + year x season fixed, cage iid, GBLUP (VanRaden G with the
                        observed allele frequencies + 0.01 I), variances by REML per fold;
* ``family_effect``     + a full-sib-family iid effect (common environment of littermates:
                        litter, dam, early cage), which the baseline cannot separate from
                        the additive effect because full sibs share both;
* ``known_variances``   variances fixed at the full-data REML estimates (removes
                        fold-to-fold estimation noise);
* ``p_half``            G with allele frequencies 0.5 instead of the observed ones;
* ``ridge_0.05``        G + 0.05 I instead of 0.01 I;
* ``markers_half``, ``markers_quarter``: G from a random half / quarter of the SNPs
                        (seed 2029; with the family effect): if the remaining
                        over-dispersion comes from imperfect linkage between markers and
                        causal loci across families, fewer markers should lower the slope;
* ``family_all_markers`` the family-effect model again, as the reference for the two
                        marker-density variants.

Also reported per variant: full-data REML variances and logL (``family_effect`` vs
``baseline`` differ by one parameter, so twice the logL difference is a likelihood-ratio
statistic for the family variance; on the boundary its null distribution is a 50:50
mixture of 0 and chi2(1)).

Usage: python benchmarks/f22_dispersion_study.py [--out docs/validation/f22_dispersion_study.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "2")

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from abp.core.genomic import spd_inverse_and_logdet, vanraden_g, allele_frequencies  # noqa: E402
from abp.solvers.blup import RandomTerm, blup  # noqa: E402
from abp.solvers.reml import reml_fit  # noqa: E402
from real_mice_validation import CFG, _corr, design, load  # noqa: E402

TRAITS = ("bw", "body_length")


def _iid(name, labels_of_rows):
    lv = sorted(set(labels_of_rows))
    p = {v: k for k, v in enumerate(lv)}
    m = len(labels_of_rows)
    Z = sp.csr_matrix((np.ones(m), (np.arange(m), [p[v] for v in labels_of_rows])),
                      shape=(m, len(lv)))
    return RandomTerm(name, Z, sp.identity(len(lv), format="csr"), lv, False,
                      k_diag=np.ones(len(lv)), logdet_k=0.0)


def build_terms(ph, rows, n, Kinv, logdet, fam, family):
    m = len(rows)
    Z = sp.csr_matrix((np.ones(m), (np.arange(m), rows)), shape=(m, n))
    out = [RandomTerm("animal", Z, Kinv, [str(i) for i in range(n)], True,
                      k_diag=np.ones(n), logdet_k=logdet),
           _iid("cage", [ph[i]["cage"] for i in rows])]
    if family:
        out.append(_iid("family", [f"F{fam[i]}" for i in rows]))
    return out


def fit(ph, rows, trait, n, K, fam, family, variances=None):
    Kinv, logdet = K
    y = np.array([float(ph[i][trait]) for i in rows])
    X = sp.csr_matrix(design(ph, rows))
    tt = build_terms(ph, rows, n, Kinv, logdet, fam, family)
    f = None
    if variances is None:
        f = reml_fit(y, X, tt, CFG)
        variances = f.variances
    r = blup(y, X, tt, variances, compute_pev=False)
    return f, r


def across_family(ph, n, trait, fold, K, fam, family, variances=None):
    pred, target = np.zeros(n), np.zeros(n)
    h2 = []
    for k in range(5):
        train = [i for i in range(n) if fold[i] != k]
        test = np.flatnonzero(fold == k)
        f, r = fit(ph, train, trait, n, K, fam, family, variances)
        if f is not None:
            v = f.variances
            h2.append(v["animal"] / sum(v.values()))
        pred[test] = r.terms["animal"].solution[test]
        levs = {c: sorted({ph[i][c] for i in train}) for c in ("sex", "year_season")}
        for i in test:
            x = [1.0]
            for c in ("sex", "year_season"):
                x += [float(ph[i][c] == lv) for lv in levs[c][1:]]
            target[i] = float(ph[i][trait]) - float(np.array(x) @ r.fixed_solution)
    bs = np.random.default_rng(2027)
    idx = [bs.integers(0, n, n) for _ in range(1000)]

    def slope(e, t):
        return float(np.cov(e, t)[0, 1] / np.var(e, ddof=1))
    return {"regression_slope": slope(pred, target),
            "regression_slope_se": float(np.std([slope(pred[i], target[i]) for i in idx],
                                                ddof=1)),
            "correlation": _corr(pred, target),
            "correlation_se": float(np.std([_corr(pred[i], target[i]) for i in idx], ddof=1)),
            "sd_ebv": float(np.std(pred)), "h2_by_fold": h2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "validation" /
                                         "f22_dispersion_study.json"))
    a = ap.parse_args()
    t0 = time.time()
    ph, M, A = load()
    n = len(ph)
    nfam, fam = connected_components(sp.csr_matrix(A >= 0.25), directed=False)
    fold = np.random.default_rng(2028).permutation(np.arange(nfam) % 5)[fam]
    p_obs = allele_frequencies(M)
    Gs = {}
    for name, p, ridge in (("obs_0.01", p_obs, 0.01), ("half_0.01", np.full_like(p_obs, 0.5), 0.01),
                           ("obs_0.05", p_obs, 0.05)):
        G, _ = vanraden_g(M, p)
        Gs[name] = spd_inverse_and_logdet(G + ridge * np.eye(n), name)
    perm = np.random.default_rng(2029).permutation(M.shape[1])
    for name, k in (("half_markers", M.shape[1] // 2), ("quarter_markers", M.shape[1] // 4)):
        cols = np.sort(perm[:k])
        G, _ = vanraden_g(M[:, cols], p_obs[cols])
        Gs[name] = spd_inverse_and_logdet(G + 0.01 * np.eye(n), name)
    variants = {"baseline": ("obs_0.01", False, False),
                "family_effect": ("obs_0.01", True, False),
                "known_variances": ("obs_0.01", False, True),
                "p_half": ("half_0.01", False, False),
                "ridge_0.05": ("obs_0.05", False, False),
                "family_all_markers": ("obs_0.01", True, False),
                "markers_half": ("half_markers", True, False),
                "markers_quarter": ("quarter_markers", True, False)}
    doc = {"data": "BGLR 1.1.4 'mice' via examples/21_mice_bodyweight_real/fetch_data.py",
           "n_mice": n, "n_families": int(nfam), "design": "across-family 5-fold CV, seed 2028",
           "full_data": {}, "across_family": {}}
    allrows = list(range(n))
    for trait in TRAITS:
        doc["full_data"][trait], doc["across_family"][trait] = {}, {}
        full = {}
        for vname, (gname, family, known) in variants.items():
            if known:
                continue
            f, _ = fit(ph, allrows, trait, n, Gs[gname], fam, family)
            full[vname] = f.variances
            v = f.variances
            doc["full_data"][trait][vname] = {"variances": v, "loglik": f.loglik,
                                              "status": f.status,
                                              "h2": v["animal"] / sum(v.values())}
            print(trait, "full", vname, json.dumps(doc["full_data"][trait][vname]), flush=True)
        for vname, (gname, family, known) in variants.items():
            res = across_family(ph, n, trait, fold, Gs[gname], fam, family,
                                full["baseline"] if known else None)
            doc["across_family"][trait][vname] = res
            print(trait, "cv", vname, json.dumps(res), flush=True)
    doc["wall_seconds"] = time.time() - t0
    Path(a.out).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
