#!/usr/bin/env python3
"""Cross-check ABP's MCMC diagnostics against ArviZ (independent implementation).

ABP implements rank-normalized split R-hat, bulk/tail ESS and the MCSE of the
mean itself (``abp.solvers.mcmc_diagnostics``).  This script compares them
with ArviZ (``az.rhat(method="rank")``, ``az.ess(method="bulk" | "tail")``,
``az.mcse(method="mean")``) on seeded synthetic chains covering the cases
that matter: independent draws, AR(1) chains with increasing autocorrelation,
chains stuck at different locations, heavy tails, a near-constant discrete
quantity, and different numbers of chains and draws.

ArviZ is **not** an ABP dependency; install it in a separate location and
put it on the path, e.g.::

    python -m pip install --target /tmp/arviz_pkg arviz
    ARVIZ_PATH=/tmp/arviz_pkg python benchmarks/mcmc_diagnostics_crosscheck.py

The result (versions, every case, largest relative differences) is written to
``docs/validation/mcmc_diagnostics_crosscheck.json``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if os.environ.get("ARVIZ_PATH"):
    sys.path.append(os.environ["ARVIZ_PATH"])      # after the system NumPy/SciPy

import arviz as az  # noqa: E402

from abp.solvers import mcmc_diagnostics as dg  # noqa: E402


def ar1(phi: float, chains: int, n: int, rng, shift=None) -> np.ndarray:
    x = np.zeros((chains, n))
    x[:, 0] = rng.normal(size=chains) / np.sqrt(1 - phi**2)
    e = rng.normal(size=(chains, n))
    for t in range(1, n):
        x[:, t] = phi * x[:, t - 1] + e[:, t]
    if shift is not None:
        x += np.asarray(shift)[:, None]
    return x


def cases(rng):
    yield "iid_4x1000", rng.normal(size=(4, 1000))
    yield "iid_2x200", rng.normal(size=(2, 200))
    yield "ar1_0.5_4x2000", ar1(0.5, 4, 2000, rng)
    yield "ar1_0.9_4x2000", ar1(0.9, 4, 2000, rng)
    yield "ar1_0.99_4x5000", ar1(0.99, 4, 5000, rng)
    yield "ar1_0.9_8x500", ar1(0.9, 8, 500, rng)
    yield "stuck_chains_4x1000", ar1(0.3, 4, 1000, rng, shift=[0.0, 0.0, 0.0, 1.5])
    yield "cauchy_4x1000", rng.standard_cauchy(size=(4, 1000))
    yield "lognormal_4x1000", np.exp(ar1(0.7, 4, 1000, rng))
    yield "discrete_counts_4x1000", rng.poisson(3.0, size=(4, 1000)).astype(float)
    yield "odd_length_3x999", ar1(0.8, 3, 999, rng)


def rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-300)


def main():
    rng = np.random.default_rng(20260925)
    rows = []
    for name, x in cases(rng):
        ours = {"rhat": dg.rhat(x), "ess_bulk": dg.bulk_ess(x), "ess_tail": dg.tail_ess(x),
                "mcse_mean": dg.mcse_mean(x)}
        ref = {"rhat": float(az.rhat(x, method="rank")),
               "ess_bulk": float(az.ess(x, method="bulk")),
               "ess_tail": float(az.ess(x, method="tail")),
               "mcse_mean": float(az.mcse(x, method="mean"))}
        diff = {k: rel(ours[k], ref[k]) for k in ours}
        rows.append({"case": name, "shape": list(x.shape), "abp": ours, "arviz": ref,
                     "relative_difference": diff})
        print(f"{name:24s} " + "  ".join(
            f"{k} {ours[k]:.6g}/{ref[k]:.6g} ({diff[k]:.1e})" for k in ours))
    worst = {k: max(r["relative_difference"][k] for r in rows)
             for k in ("rhat", "ess_bulk", "ess_tail", "mcse_mean")}
    doc = {"purpose": "independent cross-check of abp.solvers.mcmc_diagnostics against ArviZ",
           "arviz_version": az.__version__, "numpy_version": np.__version__,
           "seed": 20260925, "cases": rows, "max_relative_difference": worst}
    out = ROOT / "docs" / "validation" / "mcmc_diagnostics_crosscheck.json"
    out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print("max relative difference:", {k: f"{v:.2e}" for k, v in worst.items()})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
