#!/usr/bin/env python3
"""Calibration study with two base populations (crossbred / composite design).

A gene-drop simulator written here, independently of ABP's relationship code,
creates two breeds A and B whose base allele frequencies differ (so their
genetic levels and within-breed relationships differ), four generations of
purebred, F1 and composite matings, phenotypes on generations 1-3, and
genotyped, unrecorded candidates in generation 4.  Truth: TBV relative to the
mean of breed A's base population.

Single-step models compared (variances by REML in every scenario):

* ``ss_conventional`` unknown parents plain, G tuned to A22 (match_a22) + 5% blend;
* ``ss_one_mf``       one metafounder for all unknown parents, G05, 5% blend;
* ``ss_two_mf``       metafounders MF:A and MF:B by breed of the founder, G05,
                      5% blend, reference MF:A (``ebv_vs_base``).

Only ``ss_two_mf`` estimates the target (TBV relative to breed A's base)
directly, so bias and coverage are interpretable for it; realized accuracy and
slope are comparable across all three.  Means ± Monte-Carlo SE; because the
breed-level difference averages zero across replicates, the root-mean-square
bias across replicates is reported as well.

Usage: python benchmarks/two_metafounder_study.py [--replicates 30] [--workers 4]
       [--out docs/validation/two_metafounder_study.json]
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

from abp.workflows.evaluate import run_evaluation  # noqa: E402

N_SNP, N_QTL = 2000, 300
N_FOUNDERS = 60          # per breed (half male)
PER_GEN = 200
H2 = 0.3                 # within breed A
SIGMA_A2 = 1.0           # additive variance within breed A's base (QTL effects scaled to it)

SPEC = """schema_version = "1"
[project]
name = "two_mf"
species = "sheep"
synthetic_data = true
[analysis]
task = "additive_ebv"
target_population = "synthetic composite"
information_cutoff = "2025-12-31"
genetic_base = "{base}"
[data]
pedigree = "{ped}"
phenotypes = "phen.csv"
genotypes = "geno.csv"
marker_map = "markers.csv"
[[traits]]
name = "y"
unit = "u"
[model]
traits = ["y"]
fixed = [{{ column = "gen", type = "factor" }}]
random = [{{ name = "animal", kind = "additive", relationship = "single_step" }}]
[variances]
mode = "reml"
{genomic}
{mf}
"""
G_CONV = '[genomic]\nfrequency_source = "genotyped_all"\ntuning = "match_a22"\nsingular_policy = "blend"\nblend_alpha = 0.05'
G_05 = '[genomic]\nfrequency_source = "fixed_0.5"\ntuning = "none"\nsingular_policy = "blend"\nblend_alpha = 0.05'
MF_ONE = '[metafounders]\nprefix = "MF:"\ndefault = "MF:ALL"\ngamma_source = "genotypes_gls"'
MF_TWO = '[metafounders]\nprefix = "MF:"\nreference = "MF:A"\ngamma_source = "genotypes_gls"'
SCENARIOS = {
    "ss_conventional": dict(ped="ped_plain.csv", genomic=G_CONV, mf="", base="unrelated founders"),
    "ss_one_mf": dict(ped="ped_plain.csv", genomic=G_05, mf=MF_ONE, base="one metafounder"),
    "ss_two_mf": dict(ped="ped_mf.csv", genomic=G_05, mf=MF_TWO, base="breeds A and B"),
}


def simulate(seed: int, work: Path) -> dict:
    rng = np.random.default_rng(seed)
    m = N_SNP + N_QTL
    pA = rng.uniform(0.05, 0.95, m)
    pB = np.clip(pA + rng.normal(0, 0.15, m), 0.02, 0.98)
    q = np.arange(N_SNP, m)
    a = rng.normal(0, 1, N_QTL)
    a *= np.sqrt(SIGMA_A2 / np.sum(2 * pA[q] * (1 - pA[q]) * a ** 2))
    base_A = 2 * pA[q] @ a
    sigma_e = np.sqrt(SIGMA_A2 * (1 - H2) / H2)
    hap, rows = {}, []
    breed_of = {}
    males, females = {"A": [], "B": []}, {"A": [], "B": []}
    for br, p in (("A", pA), ("B", pB)):
        for k in range(N_FOUNDERS):
            aid = f"{br}{k:03d}"
            hap[aid] = (rng.random((2, m)) < p).astype(np.int8)
            rows.append((aid, None, None, 0, br))
            breed_of[aid] = br
            (males if k % 2 == 0 else females)[br].append(aid)
    prev_m = males["A"] + males["B"]
    prev_f = females["A"] + females["B"]
    for g in range(1, 5):
        cur_m, cur_f = [], []
        for k in range(PER_GEN):
            if g == 1:                        # purebreds and F1s in equal numbers
                kind = k % 4
                s = rng.choice(males["A"] if kind in (0, 2) else males["B"])
                d = rng.choice(females["A"] if kind in (0, 3) else females["B"])
            else:
                s, d = rng.choice(prev_m), rng.choice(prev_f)
            aid = f"G{g}_{k:03d}"
            gam = lambda h: h[rng.integers(0, 2, m), np.arange(m)]   # noqa: E731 (unlinked loci)
            hap[aid] = np.vstack([gam(hap[s]), gam(hap[d])])
            rows.append((aid, s, d, g, None))
            (cur_m if k % 2 == 0 else cur_f).append(aid)
        prev_m, prev_f = cur_m, cur_f
    tbv = {aid: float(h[:, q].sum(0) @ a - base_A) for aid, h in hap.items()}
    # files
    with open(work / "ped_plain.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "sire", "dam"])
        for aid, s, d, g, br in rows:
            w.writerow([aid, s or "0", d or "0"])
    with open(work / "ped_mf.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "sire", "dam"])
        for aid, s, d, g, br in rows:
            w.writerow([aid, s or f"MF:{br}", d or f"MF:{br}"])
    gen_of = {aid: g for aid, s, d, g, br in rows}
    with open(work / "phen.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "gen", "y"])
        for aid, s, d, g, br in rows:
            if 1 <= g <= 3:
                w.writerow([aid, f"g{g}", f"{10 + 0.5 * g + tbv[aid] + rng.normal(0, sigma_e):.6f}"])
    geno = [aid for aid, s, d, g, br in rows if g >= 2 and (g == 4 or rng.random() < 0.5)]
    with open(work / "markers.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["marker_id", "chrom", "pos", "ref", "alt", "counted_allele", "assembly"])
        for j in range(N_SNP):
            w.writerow([f"s{j}", 1 + j // 200, 1000 * (j % 200 + 1), "A", "G", "A", "SYNTH-0"])
    with open(work / "geno.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id"] + [f"s{j}" for j in range(N_SNP)])
        for aid in geno:
            w.writerow([aid] + hap[aid][:, :N_SNP].sum(0).tolist())
    gamma_true = 8 * np.array([[np.mean((x - .5) * (y - .5)) for y in (pA[:N_SNP], pB[:N_SNP])]
                               for x in (pA[:N_SNP], pB[:N_SNP])])
    return {"tbv": tbv, "gen": gen_of, "gamma_true_snp": gamma_true.tolist()}


def replicate(seed: int, work: Path) -> dict:
    sim = simulate(seed, work)
    out = {"seed": seed, "gamma_true_snp": sim["gamma_true_snp"]}
    for name, kw in SCENARIOS.items():
        spec = work / f"{name}.toml"
        spec.write_text(SPEC.format(**kw), encoding="utf-8")
        res = run_evaluation(spec, work / f"out_{name}", force=True, console=False)
        with open(res.out_dir / "ebv_y.csv", encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh) if sim["gen"][r["animal"]] == 4]
        sfx = "_vs_base" if "ebv_vs_base" in rows[0] else ""
        e = np.array([float(r["ebv" + sfx]) for r in rows])
        pev = np.array([float(r["pev" + sfx]) for r in rows])
        t = np.array([sim["tbv"][r["animal"]] for r in rows])
        err = t - e
        o = {"n": int(e.size), "realized_accuracy": float(np.corrcoef(t, e)[0, 1]),
             "slope": float(np.cov(t, e)[0, 1] / np.var(e, ddof=1)), "bias": float(err.mean()),
             "pev_ratio": float(np.mean(err ** 2) / np.mean(pev)),
             "coverage95": float(np.mean(np.abs(err) <= 1.96 * np.sqrt(pev)))}
        rel = res.manifest["relationship"]
        if "gamma" in rel:
            o["gamma"] = rel["gamma"]
        out[name] = o
    return out


def _run_seed(seed):
    with tempfile.TemporaryDirectory() as tmp:
        return replicate(seed, Path(tmp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=30)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=str(ROOT / "docs" / "validation" / "two_metafounder_study.json"))
    args = ap.parse_args()
    t0 = time.time()
    seeds = list(range(1, args.replicates + 1))
    reps = []
    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            for r in pool.imap(_run_seed, seeds):
                reps.append(r)
                print(f"seed {r['seed']}: " + "  ".join(
                    f"{k}: acc {r[k]['realized_accuracy']:.3f} bias {r[k]['bias']:+.3f}"
                    for k in SCENARIOS), flush=True)
    else:
        for s in seeds:
            reps.append(_run_seed(s))
            print(f"seed {s} done", flush=True)
    summary = {}
    for name in SCENARIOS:
        summary[name] = {}
        for k in ("realized_accuracy", "slope", "bias", "pev_ratio", "coverage95"):
            v = np.array([r[name][k] for r in reps])
            summary[name][k] = {"mean": float(v.mean()), "mc_se": float(v.std(ddof=1) / np.sqrt(v.size))}
        # the breed-level difference has mean 0 across replicates, so a wrong base shows as
        # scatter of the bias; root-mean-square bias across replicates measures it
        b = np.array([r[name]["bias"] for r in reps])
        summary[name]["rms_bias"] = {"mean": float(np.sqrt(np.mean(b ** 2))), "mc_se": float("nan")}
    g_est = np.array([r["ss_two_mf"]["gamma"] for r in reps])
    g_true = np.array([r["gamma_true_snp"] for r in reps])
    summary["ss_two_mf"]["gamma_estimate_mean"] = g_est.mean(0).tolist()
    summary["ss_two_mf"]["gamma_true_snp_mean"] = g_true.mean(0).tolist()
    summary["ss_two_mf"]["gamma_error_mean"] = (g_est - g_true).mean(0).tolist()
    doc = {"study": "two base populations (breeds A, B), composite generations, genotyped candidates",
           "replicates": args.replicates, "seeds": f"1..{args.replicates}",
           "design": {"snp": N_SNP, "qtl": N_QTL, "founders_per_breed": N_FOUNDERS,
                      "per_generation": PER_GEN, "h2_within_A": H2,
                      "pB": "clip(pA + N(0, 0.15^2), 0.02, 0.98)"},
           "truth": "TBV relative to the mean of breed A's base population",
           "wall_seconds": time.time() - t0, "summary": summary, "replicate_results": reps}
    Path(args.out).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    for name, s in summary.items():
        print(name, {k: (f"{v['mean']:.3f}±{v['mc_se']:.3f}" if isinstance(v, dict) else
                         np.round(v, 3).tolist()) for k, v in s.items()})
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
