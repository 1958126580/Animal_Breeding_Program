#!/usr/bin/env python3
"""Simulation-based check (acceptance gate G4, single replicate) for the
synthetic sheep examples.

Runs examples 02 (pedigree BLUP + REML), 05 (single-step) and 06
(multi-trait) and compares the published EBVs with the simulated true
breeding values in ``examples/sheep_data/truth`` - the only place in the
project where the truth is read.  Reported per group of animals:

* realized accuracy  corr(EBV, TBV);
* model-based accuracy  sqrt(mean reliability) (should be similar if the
  reliabilities are calibrated);
* dispersion  slope of the regression of TBV on EBV (1 = no over/under-dispersion);
* bias  mean(TBV - EBV) (relative to the founder base).

This is ONE replicate of ONE synthetic scenario: it can reveal gross
errors, it cannot establish calibration in general.

Usage:  python benchmarks/simulation_check.py [--out docs/validation/simulation_check.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abp.workflows.evaluate import run_evaluation  # noqa: E402

DATA = ROOT / "examples" / "sheep_data"


def _read(p):
    with open(p, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _stats(e, t, rel):
    b = float(np.cov(t, e)[0, 1] / np.var(e, ddof=1))
    return {"n": int(e.size), "realized_accuracy": float(np.corrcoef(e, t)[0, 1]),
            "model_accuracy_sqrt_mean_rel": float(np.sqrt(np.mean(rel))),
            "dispersion_slope_tbv_on_ebv": b, "bias_mean_tbv_minus_ebv": float(np.mean(t - e))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "validation" / "simulation_check.json"))
    args = ap.parse_args()
    tbv = {r["id"]: r for r in _read(DATA / "truth" / "tbv.csv")}
    ped = {r["id"]: r for r in _read(DATA / "data" / "pedigree.csv")}
    geno = {r["id"] for r in _read(DATA / "data" / "genotypes.csv")}
    groups = {
        "all animals": lambda a: True,
        "lambs born 2025 (candidates)": lambda a: ped[a]["birth_date"].startswith("2025"),
        "2025 lambs, genotyped": lambda a: ped[a]["birth_date"].startswith("2025") and a in geno,
        "2025 lambs, not genotyped": lambda a: ped[a]["birth_date"].startswith("2025") and a not in geno,
    }
    out = {"scope": "single replicate of one synthetic scenario (seed 20260925); indicative only",
           "results": []}
    with tempfile.TemporaryDirectory() as tmp:
        runs = [("02_sheep_wwt_reml", "wwt", "ebv_wwt.csv", "ebv", "reliability", "wwt"),
                ("05_sheep_wwt_single_step", "wwt", "ebv_wwt.csv", "ebv", "reliability", "wwt"),
                ("06_sheep_multitrait_index", "wwt", "ebv_multitrait.csv", "ebv_wwt", "reliability_wwt", "wwt"),
                ("06_sheep_multitrait_index", "fat", "ebv_multitrait.csv", "ebv_fat", "reliability_fat", "fat"),
                ("06_sheep_multitrait_index", "fec", "ebv_multitrait.csv", "ebv_fec", "reliability_fec", "fec"),
                ("06_sheep_multitrait_index", "nlb1", "ebv_multitrait.csv", "ebv_nlb1", "reliability_nlb1", "nlb")]
        done = {}
        for ex, trait, fname, ecol, rcol, tcol in runs:
            if ex not in done:
                res = run_evaluation(ROOT / "examples" / ex / "analysis.toml", Path(tmp) / ex,
                                     console=False)
                done[ex] = res.out_dir
            rows = _read(done[ex] / fname)
            for gname, sel in groups.items():
                ids = [r["animal"] for r in rows if sel(r["animal"])]
                if len(ids) < 10:
                    continue
                r_by = {r["animal"]: r for r in rows}
                e = np.array([float(r_by[a][ecol]) for a in ids])
                rel = np.array([float(r_by[a][rcol]) for a in ids])
                t = np.array([float(tbv[a][f"tbv_{tcol}"]) for a in ids])
                rec = {"example": ex, "trait": trait, "group": gname, **_stats(e, t, rel)}
                if trait == "nlb1":
                    rec["note"] = ("TBV on the liability scale, EBV from a linear model on "
                                   "observed counts: accuracy comparable, scale/slope/bias not")
                out["results"].append(rec)
                print(f"{ex:28s} {trait:5s} {gname:28s} n={rec['n']:5d} "
                      f"r={rec['realized_accuracy']:.3f} model={rec['model_accuracy_sqrt_mean_rel']:.3f} "
                      f"slope={rec['dispersion_slope_tbv_on_ebv']:.3f} bias={rec['bias_mean_tbv_minus_ebv']:+.3f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
