#!/usr/bin/env python3
"""Multi-trait calibration study (acceptance gate G4, finding F4).

For R replicates of the synthetic sheep flock (seeds 1..R), weaning weight,
fat depth and log faecal egg count are evaluated jointly by multi-trait BLUP

* ``mt_true``  with the generator's base-population covariance matrices
               (read from ``truth/simulation_parameters.json`` - only this
               script reads the truth);
* ``mt_reml``  with G0 and R0 estimated by multi-trait REML.

For the selection candidates (lambs of the last season) and each trait the
script reports the dispersion slope b(TBV | EBV), bias mean(TBV - EBV),
realized accuracy, model accuracy sqrt(mean reliability), MSE / mean PEV and
the coverage of nominal 95% intervals, as ``benchmarks/calibration_study.py``
does for one trait.  Means ± Monte-Carlo SE across replicates.  From round 6
``mt_reml`` also reports ``pev_ratio_incl_vc`` and ``coverage95_incl_vc``
with the Kackar-Harville PEV that includes the uncertainty of G0 and R0.

Usage: python benchmarks/mt_calibration_study.py [--replicates 50] [--workers 4]
       [--out docs/validation/mt_calibration_study.json]
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
    os.environ.setdefault(_var, "1")      # one BLAS thread per worker (before NumPy)

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abp.examples.sheep import write_sheep_example  # noqa: E402
from abp.workflows.evaluate import run_evaluation  # noqa: E402

TRAITS = ["wwt", "fat", "fec"]
TRAIT_BLOCK = """[[traits]]
name = "wwt"
unit = "kg"
min = 5.0
max = 80.0
[[traits]]
name = "fat"
unit = "mm"
min = 0.0
max = 15.0
[[traits]]
name = "fec"
unit = "ln(epg+1)"
min = 0.0
max = 12.0
"""
SPEC = """schema_version = "1"
[project]
name = "mt_calibration"
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
[data.pedigree_columns]
sex = "sex"
birth_date = "birth_date"
{traits}
[model]
traits = ["wwt", "fat", "fec"]
fixed = [{{ column = "cg", type = "factor", traits = ["wwt", "fat", "fec"] }},
         {{ column = "birth_type", type = "factor", traits = ["wwt", "fat"] }},
         {{ column = "dam_age", type = "factor", traits = ["wwt"] }}]
random = [{{ name = "animal", kind = "additive", relationship = "pedigree" }}]
[variances]
{variances}
[qc]
out_of_range = "quarantine"
"""


def _read(p):
    with open(p, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _matrix(M):
    return "[" + ", ".join("[" + ", ".join(repr(float(x)) for x in row) + "]" for row in M) + "]"


def replicate(seed: int, work: Path) -> dict:
    write_sheep_example(work / "sheep", seed=seed, force=True)
    params = json.loads((work / "sheep" / "truth" / "simulation_parameters.json").read_text())
    G = np.array(params["founder_genetic_covariance"])[:3, :3]
    R = np.array(params["residual_covariance_lamb_traits"])
    tbv = {r["id"]: r for r in _read(work / "sheep" / "truth" / "tbv.csv")}
    ped = {r["id"]: r["birth_date"] for r in _read(work / "sheep" / "data" / "pedigree.csv")}
    last = max(d[:4] for d in ped.values())
    scen = {"mt_true": f'mode = "known"\nvalues.animal = {_matrix(G)}\nvalues.residual = {_matrix(R)}',
            "mt_reml": 'mode = "reml"'}
    out = {"seed": seed}
    for name, var in scen.items():
        spec = work / f"{name}.toml"
        spec.write_text(SPEC.format(traits=TRAIT_BLOCK, variances=var), encoding="utf-8")
        res = run_evaluation(spec, work / f"out_{name}", force=True, console=False)
        rows = [r for r in _read(res.out_dir / "ebv_multitrait.csv")
                if ped[r["animal"]].startswith(last)]
        o = {}
        for j, tr in enumerate(TRAITS):
            e = np.array([float(r[f"ebv_{tr}"]) for r in rows])
            sep = np.array([float(r[f"sep_{tr}"]) for r in rows])
            rel = np.array([float(r[f"reliability_{tr}"]) for r in rows])
            t = np.array([float(tbv[r["animal"]][f"tbv_{tr}"]) for r in rows])
            err = t - e
            o[tr] = {"slope": float(np.cov(t, e)[0, 1] / np.var(e, ddof=1)),
                     "bias": float(err.mean()),
                     "realized_accuracy": float(np.corrcoef(t, e)[0, 1]),
                     "model_accuracy": float(np.sqrt(rel.mean())),
                     "pev_ratio": float(np.mean(err ** 2) / np.mean(sep ** 2)),
                     "coverage95": float(np.mean(np.abs(err) <= 1.96 * sep))}
            if f"pev_incl_vc_uncertainty_{tr}" in rows[0]:      # round 6: Kackar-Harville PEV
                pv = np.array([float(r[f"pev_incl_vc_uncertainty_{tr}"]) for r in rows])
                o[tr]["pev_ratio_incl_vc"] = float(np.mean(err ** 2) / np.mean(pv))
                o[tr]["coverage95_incl_vc"] = float(np.mean(np.abs(err) <= 1.96 * np.sqrt(pv)))
        if name == "mt_reml":
            r = res.results["traits"]["wwt"]["reml"]
            o["G0"] = r["G0"]
            o["R0"] = r["R0"]
            o["iterations"] = r["iterations"]
        out[name] = o
    out["true_G0"] = G.tolist()
    out["true_R0"] = R.tolist()
    return out


def _run_seed(seed: int) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        return replicate(seed, Path(tmp))


def summarize(reps):
    s = {}
    for name in ("mt_true", "mt_reml"):
        s[name] = {}
        for tr in TRAITS:
            s[name][tr] = {}
            for k in reps[0][name][tr]:
                v = np.array([r[name][tr][k] for r in reps if k in r[name][tr]])
                s[name][tr][k] = {"mean": float(v.mean()), "mc_se": float(v.std(ddof=1) / np.sqrt(v.size))}
    G = np.array([r["mt_reml"]["G0"] for r in reps])
    R = np.array([r["mt_reml"]["R0"] for r in reps])
    s["mt_reml"]["G0_mean"] = G.mean(0).tolist()
    s["mt_reml"]["G0_mc_se"] = (G.std(0, ddof=1) / np.sqrt(len(reps))).tolist()
    s["mt_reml"]["R0_mean"] = R.mean(0).tolist()
    s["mt_reml"]["R0_mc_se"] = (R.std(0, ddof=1) / np.sqrt(len(reps))).tolist()
    s["true_G0"] = reps[0]["true_G0"]
    s["true_R0"] = reps[0]["true_R0"]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=50)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=str(ROOT / "docs" / "validation" / "mt_calibration_study.json"))
    args = ap.parse_args()
    t0 = time.time()
    seeds = list(range(1, args.replicates + 1))
    reps = []
    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            for r in pool.imap(_run_seed, seeds):
                reps.append(r)
                print(f"seed {r['seed']:3d}: " + "  ".join(
                    f"{n}/{tr}: pev {r[n][tr]['pev_ratio']:.3f}" for n in ("mt_true", "mt_reml")
                    for tr in TRAITS), flush=True)
    else:
        for seed in seeds:
            reps.append(_run_seed(seed))
            print(f"seed {seed} done", flush=True)
    doc = {"study": "multi-trait EBV calibration (wwt, fat, fec) for last-season candidates",
           "replicates": args.replicates, "seeds": f"1..{args.replicates}",
           "generator": "abp.examples.sheep default configuration", "workers": args.workers,
           "wall_seconds": time.time() - t0, "summary": summarize(reps), "replicate_results": reps}
    Path(args.out).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    for name in ("mt_true", "mt_reml"):
        for tr in TRAITS:
            v = doc["summary"][name][tr]
            print(name, tr, {k: f"{x['mean']:.3f}±{x['mc_se']:.3f}" for k, x in v.items()})
    print("REML G0 mean", np.round(doc["summary"]["mt_reml"]["G0_mean"], 3).tolist(),
          "true", np.round(doc["summary"]["true_G0"], 3).tolist())
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
