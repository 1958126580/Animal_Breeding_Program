#!/usr/bin/env python3
"""Multi-replicate calibration study of EBVs (acceptance gate G4).

For R independent replicates of the synthetic sheep flock (seeds 1..R) this
script evaluates weaning weight three ways and compares the selection
candidates' EBVs (lambs of the last season) with the simulated true
breeding values:

* ``pedigree_true``  pedigree BLUP with the true base-population variances
                     (sigma_a^2 = 4.0, sigma_e^2 = 12.25);
* ``pedigree_reml``  pedigree BLUP with variances estimated by REML;
* ``single_step_true`` ssGBLUP (G tuned to A22, 5% A22 blend) with true variances.

Per replicate: dispersion slope b(TBV | EBV), bias mean(TBV - EBV),
realized accuracy, model accuracy sqrt(mean reliability), the ratio
mean((TBV - EBV)^2) / mean(PEV) (1 if PEV is calibrated), and the coverage of
nominal 95% prediction intervals EBV +- 1.96 SEP.  Across replicates the
script reports means with Monte-Carlo standard errors (SD / sqrt(R)).

Expected under a correctly specified model with true variances: slope 1,
bias 0, PEV ratio 1, coverage 0.95 (up to finite-locus and selection effects,
which the infinitesimal model ignores).  Truth is read only here.

Usage: python benchmarks/calibration_study.py [--replicates 50] [--out docs/validation/calibration_study.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abp.examples.sheep import write_sheep_example  # noqa: E402
from abp.workflows.evaluate import run_evaluation  # noqa: E402

SPEC = """schema_version = "1"
[project]
name = "calibration"
species = "sheep"
synthetic_data = true
[analysis]
task = "additive_ebv"
target_population = "synthetic replicate"
information_cutoff = "2025-12-31"
genetic_base = "founders"
[data]
pedigree = "sheep/data/pedigree.csv"
phenotypes = "sheep/data/lambs.csv"
{geno}
[data.pedigree_columns]
sex = "sex"
birth_date = "birth_date"
[[traits]]
name = "wwt"
unit = "kg"
min = 5.0
max = 80.0
[model]
traits = ["wwt"]
fixed = [{{ column = "cg", type = "factor" }}, {{ column = "birth_type", type = "factor" }},
         {{ column = "dam_age", type = "factor" }}]
random = [{{ name = "animal", kind = "additive", relationship = "{rel}" }}]
[variances]
{variances}
[qc]
out_of_range = "quarantine"
{genomic}
"""

TRUE = 'mode = "known"\nvalues = { animal = 4.0, residual = 12.25 }'
REML = 'mode = "reml"'
GENO = 'genotypes = "sheep/data/genotypes.csv"\nmarker_map = "sheep/data/markers.csv"'
GCFG = '[genomic]\ntuning = "match_a22"\nsingular_policy = "blend"\nblend_alpha = 0.05'

SCENARIOS = {
    "pedigree_true": dict(rel="pedigree", variances=TRUE, geno="", genomic=""),
    "pedigree_reml": dict(rel="pedigree", variances=REML, geno="", genomic=""),
    "single_step_true": dict(rel="single_step", variances=TRUE, geno=GENO, genomic=GCFG),
}


def _read(p):
    with open(p, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def replicate(seed: int, work: Path) -> dict:
    write_sheep_example(work / "sheep", seed=seed, force=True)
    tbv = {r["id"]: float(r["tbv_wwt"]) for r in _read(work / "sheep" / "truth" / "tbv.csv")}
    ped = {r["id"]: r["birth_date"] for r in _read(work / "sheep" / "data" / "pedigree.csv")}
    last = max(d[:4] for d in ped.values())
    out = {"seed": seed}
    for name, kw in SCENARIOS.items():
        spec = work / f"{name}.toml"
        spec.write_text(SPEC.format(**kw), encoding="utf-8")
        res = run_evaluation(spec, work / f"out_{name}", force=True, console=False)
        rows = [r for r in _read(res.out_dir / "ebv_wwt.csv") if ped[r["animal"]].startswith(last)]
        e = np.array([float(r["ebv"]) for r in rows])
        pev = np.array([float(r["pev"]) for r in rows])
        rel = np.array([float(r["reliability"]) for r in rows])
        t = np.array([tbv[r["animal"]] for r in rows])
        err = t - e
        out[name] = {
            "n": int(e.size),
            "slope": float(np.cov(t, e)[0, 1] / np.var(e, ddof=1)),
            "bias": float(err.mean()),
            "realized_accuracy": float(np.corrcoef(t, e)[0, 1]),
            "model_accuracy": float(np.sqrt(rel.mean())),
            "pev_ratio": float(np.mean(err**2) / np.mean(pev)),
            "coverage95": float(np.mean(np.abs(err) <= 1.96 * np.sqrt(pev))),
            "sigma_a2": res.results["traits"]["wwt"]["variance_components"]["animal"],
        }
    return out


def summarize(reps: list[dict]) -> dict:
    summary = {}
    for name in SCENARIOS:
        s = {}
        for key in ("slope", "bias", "realized_accuracy", "model_accuracy", "pev_ratio",
                    "coverage95", "sigma_a2"):
            v = np.array([r[name][key] for r in reps])
            s[key] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                      "mc_se": float(v.std(ddof=1) / np.sqrt(v.size))}
        summary[name] = s
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=50)
    ap.add_argument("--out", default=str(ROOT / "docs" / "validation" / "calibration_study.json"))
    args = ap.parse_args()
    t0 = time.time()
    reps = []
    with tempfile.TemporaryDirectory() as tmp:
        for seed in range(1, args.replicates + 1):
            reps.append(replicate(seed, Path(tmp)))
            r = reps[-1]
            print(f"seed {seed:3d}: " + "  ".join(
                f"{k}: slope {r[k]['slope']:.3f} bias {r[k]['bias']:+.3f} cov {r[k]['coverage95']:.3f}"
                for k in SCENARIOS), flush=True)
    summary = summarize(reps)
    doc = {"study": "EBV calibration for last-season candidates, weaning weight",
           "replicates": args.replicates, "seeds": f"1..{args.replicates}",
           "generator": "abp.examples.sheep default configuration",
           "true_variances": {"animal": 4.0, "residual": 12.25},
           "wall_seconds": time.time() - t0, "summary": summary, "replicate_results": reps}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    for name, s in summary.items():
        print(name, {k: f"{v['mean']:.3f} +- {v['mc_se']:.3f}" for k, v in s.items()})
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
