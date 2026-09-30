#!/usr/bin/env python3
"""Sampled reliabilities: orthogonal vs ratio estimator against exact values (round 8).

Small single step (80 animals, 25 genotyped with 300 SNPs, 5% A22 blend, 50
records, sigma_a^2 = 2, sigma_e^2 = 4) whose exact reliabilities come from the
dense inverse of the mixed-model equations.  For ``--seeds`` independent seeds,
``sampled_pev`` with ``--samples`` simulations is run with both estimators on the
same draws (same seed):

* ``ratio``       1 - mean d^2 / mean u*^2          (round 6)
* ``orthogonal``  mean h^2 / (mean h^2 + mean d^2)  (round 8; h = u_hat*, d = u* - h)

Reported per estimator: mean error, SD of the error across seeds (averaged over
animals), mean reported SE, and the theoretical SD ratio ``mean(sqrt(r) SD_ratio) /
mean(SD_ratio)``.  The same design is checked in
``tests/test_single_step_matrix_free.py::test_orthogonal_reliability_estimator_is_unbiased_and_less_noisy``.

Usage: python benchmarks/pev_estimator_study.py [--seeds 300] [--samples 40]
       [--out docs/validation/pev_estimator_study.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abp.core.genomic import apply_g_policy, centered, scaling_d, single_step  # noqa: E402
from abp.core.pedigree import Pedigree  # noqa: E402
from abp.core.ssop import (A22InverseOperator, DenseInverseOperator,  # noqa: E402
                           SingleStepHInverse)
from abp.solvers.blup import RandomTerm  # noqa: E402
from abp.solvers.pev_sampling import sampled_pev  # noqa: E402


def problem():
    rng = np.random.default_rng(9)
    n = 80
    ids = [f"x{i}" for i in range(n)]
    male = np.arange(n) % 2 == 0
    s, d = [], []
    for i in range(n):
        if i < 30:
            s.append(None)
            d.append(None)
        else:
            s.append(ids[rng.choice(np.flatnonzero(male[:i]))])
            d.append(ids[rng.choice(np.flatnonzero(~male[:i]))])
    ped = Pedigree.from_parent_ids(ids, s, d)
    g = np.sort(rng.choice(n, 25, replace=False))
    M = rng.integers(0, 3, (g.size, 300)).astype(float)
    p = M.mean(axis=0) / 2
    W = centered(M, p)
    Gs, _ = apply_g_policy(W @ W.T / scaling_d(p), "blend", "none", ped.a_submatrix(g),
                           0.05, 0.01)
    h_inv = single_step(ped, g, Gs, a22=ped.a_submatrix(g)).h_inv.toarray()
    rec = np.sort(rng.choice(n, 50, replace=False))
    Z = sp.csr_matrix((np.ones(rec.size), (np.arange(rec.size), rec)), shape=(rec.size, n))
    X = sp.csr_matrix(np.ones((rec.size, 1)))
    op = SingleStepHInverse(ped.ainv().tocsr(), g,
                            DenseInverseOperator(np.linalg.inv(Gs), np.linalg.cholesky(Gs)),
                            A22InverseOperator(ped.ainv(), g), ped=ped)
    return ped, h_inv, Z, X, op, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=300)
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--out", default=str(ROOT / "docs" / "validation" /
                                         "pev_estimator_study.json"))
    a = ap.parse_args()
    ped, h_inv, Z, X, op, rec = problem()
    va, ve = 2.0, 4.0
    Zd, Xd = Z.toarray(), X.toarray()
    C = np.block([[Xd.T @ Xd, Xd.T @ Zd], [Zd.T @ Xd, Zd.T @ Zd + h_inv * ve / va]]) / ve
    pev = np.diag(np.linalg.inv(C))[1:]
    r_true = 1.0 - pev / (va * np.diag(np.linalg.inv(h_inv)))
    term = [RandomTerm("animal", Z, op, ped.ids, True)]
    R = {"orthogonal": [], "ratio": []}
    SE = {"orthogonal": [], "ratio": []}
    for seed in range(a.seeds):
        for e in R:
            s = sampled_pev(rec.size, X, term, {"animal": va, "residual": ve}, "animal",
                            n_samples=a.samples, seed=seed, tol=1e-12, estimator=e)
            R[e].append(s.reliability)
            SE[e].append(s.reliability_se)
    out = {"design": {"animals": ped.n, "genotyped": 25, "records": int(rec.size),
                      "seeds": a.seeds, "samples_per_seed": a.samples,
                      "mean_exact_reliability": float(r_true.mean()),
                      "exact_reliability_range": [float(r_true.min()), float(r_true.max())]}}
    sd = {}
    for e in R:
        err = np.array(R[e]) - r_true
        sd[e] = err.std(axis=0, ddof=1)
        bias = err.mean(axis=0)
        out[e] = {"mean_error": float(bias.mean()),
                  "mc_se_of_mean_error": float(np.sqrt(np.mean(sd[e] ** 2) / (a.seeds * ped.n))),
                  "max_abs_animal_bias_in_se": float(np.max(np.abs(bias) /
                                                            (sd[e] / np.sqrt(a.seeds)))),
                  "sd_across_seeds": float(sd[e].mean()),
                  "mean_reported_se": float(np.mean(SE[e]))}
    out["sd_ratio_orthogonal_to_ratio"] = float(sd["orthogonal"].mean() / sd["ratio"].mean())
    out["sd_ratio_predicted_sqrt_r"] = float(np.mean(np.sqrt(r_true) * sd["ratio"])
                                             / sd["ratio"].mean())
    out["equivalent_sample_factor"] = float(1.0 / out["sd_ratio_orthogonal_to_ratio"] ** 2)
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
