#!/usr/bin/env python3
"""Linear vs threshold model for a categorical trait (finding F2).

Number of lambs born (1/2/3 per lambing, several lambings per ewe) in R
replicates of the synthetic flock is analysed with

* ``linear_reml``      the linear repeatability model of example 03 (REML);
* ``threshold_true``   the threshold (probit) repeatability model with the
                       generator's liability variances rescaled to a residual
                       SD of 1 (example 14).

For the ewes with records the script reports the realized accuracy
corr(EBV, TBV) against the simulated liability TBV, the model accuracy
sqrt(mean reliability) and their ratio (1 if the model's reliabilities are
calibrated).  Means ± Monte-Carlo SE over the replicates in which both models
issued results; replicates where ABP withheld a result (e.g. REML variance at
the boundary) are counted per model.

Usage: python benchmarks/threshold_study.py [--replicates 30] [--workers 4]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abp.errors import ABPError  # noqa: E402
from abp.examples.sheep import write_sheep_example  # noqa: E402
from abp.workflows.evaluate import run_evaluation  # noqa: E402

EX03 = ROOT / "examples" / "03_sheep_nlb_repeatability" / "analysis.toml"
EX14 = ROOT / "examples" / "14_sheep_nlb_threshold" / "analysis.toml"
SCEN = {"linear_reml": EX03, "threshold_true": EX14}


def replicate(seed: int, work: Path) -> dict:
    write_sheep_example(work / "sheep_data", seed=seed, force=True)
    with open(work / "sheep_data" / "truth" / "tbv.csv", encoding="utf-8") as fh:
        tbv = {r["id"]: float(r["tbv_nlb"]) for r in csv.DictReader(fh)}
    out = {"seed": seed}
    for name, src in SCEN.items():
        spec = work / "cases" / f"{name}.toml"
        spec.parent.mkdir(exist_ok=True)
        spec.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        try:
            res = run_evaluation(spec, work / f"out_{name}", force=True, console=False)
        except ABPError as exc:          # e.g. REML at the boundary: ABP withholds the ranking
            out[name] = {"withheld": exc.code, "message": exc.message}
            continue
        with open(res.out_dir / "ebv_nlb.csv", encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh) if int(r["n_records"]) > 0]
        e = np.array([float(r["ebv"]) for r in rows])
        rel = np.array([float(r["reliability"]) for r in rows])
        t = np.array([tbv[r["animal"]] for r in rows])
        acc = float(np.corrcoef(t, e)[0, 1])
        macc = float(np.sqrt(rel.mean()))
        out[name] = {"n": int(e.size), "realized_accuracy": acc, "model_accuracy": macc,
                     "ratio_model_to_realized": macc / acc}
    return out


def _run(seed):
    with tempfile.TemporaryDirectory() as tmp:
        return replicate(seed, Path(tmp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=30)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=str(ROOT / "docs" / "validation" / "threshold_study.json"))
    args = ap.parse_args()
    t0 = time.time()
    seeds = list(range(1, args.replicates + 1))
    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            reps = list(pool.imap(_run, seeds))
    else:
        reps = [_run(s) for s in seeds]
    both = [r for r in reps if all("withheld" not in r[n] for n in SCEN)]
    summary = {"n_replicates_both_models": len(both)}
    for name in SCEN:
        summary[name] = {"n_withheld": sum("withheld" in r[name] for r in reps)}
        for k in ("realized_accuracy", "model_accuracy", "ratio_model_to_realized"):
            v = np.array([r[name][k] for r in both])
            summary[name][k] = {"mean": float(v.mean()), "mc_se": float(v.std(ddof=1) / np.sqrt(v.size))}
    d = np.array([r["threshold_true"]["realized_accuracy"] - r["linear_reml"]["realized_accuracy"]
                  for r in both])
    summary["paired_difference_realized_accuracy_threshold_minus_linear"] = {
        "mean": float(d.mean()), "mc_se": float(d.std(ddof=1) / np.sqrt(d.size))}
    doc = {"study": "linear vs threshold model, number of lambs born", "replicates": args.replicates,
           "seeds": f"1..{args.replicates}", "specs": {k: str(v.relative_to(ROOT)) for k, v in SCEN.items()},
           "wall_seconds": time.time() - t0, "summary": summary, "replicate_results": reps}
    Path(args.out).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    for k, v in summary.items():
        if isinstance(v, dict) and "mean" not in v:
            print(k, {a: (f"{b['mean']:.3f}±{b['mc_se']:.3f}" if isinstance(b, dict) else b)
                      for a, b in v.items()})
        else:
            print(k, v if not isinstance(v, dict) else f"{v['mean']:.4f}±{v['mc_se']:.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
