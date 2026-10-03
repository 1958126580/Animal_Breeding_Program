#!/usr/bin/env python3
"""Cross-check of the missing-data handling of the multi-trait threshold sampler (round 8).

Two samplers of the same posterior, built differently, on one data set with missing
values (categorical trait on half the animals, 20% of the continuous values missing):

* current (``abp.solvers.mt_threshold_gibbs``): partially collapsed Gibbs sampler on
  the observed-data equations, missing residuals drawn from their conditional;
* first implementation (commit ``b0d3242``, loaded with ``git show``): imputes the
  missing **observations** (data augmentation) and works on complete records.

Proper IW prior, 4 chains each, independent seeds; the posterior means of every
monitored scalar must agree within Monte-Carlo error (z = difference / combined MCSE).

Usage: python benchmarks/mt_threshold_crosscheck.py [--iterations 8000]
       [--out docs/validation/mt_threshold_crosscheck.json]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abp.core.pedigree import Pedigree  # noqa: E402
from abp.solvers import mt_threshold_gibbs as V2  # noqa: E402


def data():
    from scipy.sparse.linalg import spsolve_triangular
    rng = np.random.default_rng(22)
    n = 150
    ids = [f"a{i}" for i in range(n)]
    male = np.arange(n) % 2 == 0
    s, d = [], []
    for i in range(n):
        if i < 25:
            s.append(None)
            d.append(None)
        else:
            s.append(ids[rng.choice(np.flatnonzero(male[:i]))])
            d.append(ids[rng.choice(np.flatnonzero(~male[:i]))])
    ped = Pedigree.from_parent_ids(ids, s, d)
    F = spsolve_triangular(ped._l_matrix(), np.sqrt(ped.mendelian_d())[:, None]
                           * rng.standard_normal((n, 2)), lower=True, unit_diagonal=True)
    U = F @ np.linalg.cholesky([[0.5, 0.5], [0.5, 2.0]]).T
    E = rng.standard_normal((n, 2)) @ np.linalg.cholesky([[1.0, 0.5], [0.5, 2.0]]).T
    L = U + E + np.array([0.0, 5.0])
    Y = L.copy()
    Y[:, 0] = np.digitize(L[:, 0], [-0.4, 0.5]) + 1
    Y[np.arange(n) % 2 == 0, 0] = np.nan
    Y[(rng.random(n) < 0.2) & (np.arange(n) % 2 == 1), 1] = np.nan
    return ped, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=8000)
    ap.add_argument("--out", default=str(ROOT / "docs" / "validation" /
                                         "mt_threshold_crosscheck.json"))
    a = ap.parse_args()
    src = subprocess.run(["git", "show", "b0d3242:src/abp/solvers/mt_threshold_gibbs.py"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location("abp.solvers.mt_threshold_gibbs_v1", fh.name)
    V1 = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = V1
    spec.loader.exec_module(V1)
    ped, Y = data()
    n = ped.n
    X = sp.csr_matrix(np.ones((n, 1)))
    kw = dict(chains=4, iterations=a.iterations, burn_in=1000, thin=2,
              max_iterations=a.iterations, prior_nu=6.0, prior_G0=np.diag([0.4, 1.5]))
    r2 = V2.mt_threshold_gibbs(Y, 0, X, np.arange(n), ped.ainv(),
                               V2.MTThresholdGibbsConfig(seed=41, **kw))
    r1 = V1.mt_threshold_gibbs(Y, 0, X, sp.identity(n, format="csr"), ped.ainv(),
                               V1.MTThresholdGibbsConfig(seed=42, **kw))
    rows = {}
    for k, b in r2.summaries.items():
        s1 = r1.summaries.get(k)
        if s1 is None:
            continue
        z = (s1["mean"] - b["mean"]) / math.hypot(s1["mcse_mean"], b["mcse_mean"])
        rows[k] = {"first_implementation": s1["mean"], "current": b["mean"],
                   "ess_first": s1["ess_bulk"], "ess_current": b["ess_bulk"], "z": z}
        print(f"{k:8s} first {s1['mean']:.4f}  current {b['mean']:.4f}  z {z:+.2f}")
    doc = {"check": "posterior means: imputation of missing observations (b0d3242) vs "
                    "partially collapsed sampler (current)", "iterations": a.iterations,
           "max_abs_z": max(abs(r["z"]) for r in rows.values()), "quantities": rows}
    Path(a.out).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"max |z| {doc['max_abs_z']:.2f}\nwrote {a.out}")


if __name__ == "__main__":
    main()
