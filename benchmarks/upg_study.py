#!/usr/bin/env python3
"""Multi-replicate study of unknown-parent groups (example 11 scenario).

For R independent replicates of the example-11 flock (seeds 1..R; purchased
rams from a breeder with genetic progress and from a breeder below the base)
this script evaluates weaning weight three ways - random groups (declared
ratio 1.0, REML variances), fixed groups (simulation variances) and no groups
(group codes replaced by unknown parents) - and scores each against the true
breeding values:

* ``corr_lambs``  correlation of EBV and TBV over all lambs;
* ``slope_lambs`` regression b(TBV | EBV) over all lambs (1 = no dispersion bias);
* ``bias_rams``   mean(EBV - TBV) of purchased rams (their level is what
                  groups are meant to capture);
* ``bias_2024``   mean(EBV - TBV) of the last lamb crop (selection candidates).

Across replicates it reports means with Monte-Carlo standard errors
(SD / sqrt(R)) and paired differences against ``none``.  Truth is read only
here.  The scenario is one synthetic design; results do not transfer to other
populations without their own study.

``--trend step`` is the control scenario: origin A's mean is constant within
each purchase period (2.1 kg for 2016-2019, 4.8 kg for 2020-2024), so the
group model is correctly specified and fixed groups should be unbiased.
The default ``linear`` trend (+0.6 kg per purchase year) also varies
*within* each period, which a period group cannot represent.

``--ratio R`` overrides the declared sigma_g^2/sigma_a^2 of the random-group
model (1.0 in example 11); as R grows the random-group solutions approach the
fixed-group ones (less shrinkage of the group effects towards zero).

Usage: python benchmarks/upg_study.py [--replicates 20] [--trend linear|step] [--ratio R] [--out FILE]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EX11 = ROOT / "examples" / "11_sheep_upg"
sys.path.insert(0, str(ROOT / "src"))

MODELS = ("random", "fixed", "none")
METRICS = ("corr_lambs", "slope_lambs", "bias_rams", "bias_2024")


def _module(name: str):
    spec = importlib.util.spec_from_file_location(name, EX11 / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mc(values) -> dict:
    v = np.asarray(values, dtype=np.float64)
    return {"mean": float(v.mean()), "mc_se": float(v.std(ddof=1) / np.sqrt(v.size))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=20)
    ap.add_argument("--trend", choices=("linear", "step"), default="linear")
    ap.add_argument("--ratio", type=float, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        name = "upg_study" + ("" if args.trend == "linear" else "_step") + (
            "" if args.ratio is None else f"_ratio{args.ratio:g}") + ".json"
        args.out = str(ROOT / "docs" / "validation" / name)
    make, compare = _module("make_data"), _module("compare")
    if args.trend == "step":
        linear = make.origin_mean
        make.origin_mean = lambda o, y: (linear("A", 2016 + 1.5) if y <= 2019
                                         else linear("A", 2022)) if o == "A" else linear(o, y)
    t0 = time.time()
    reps = []
    with tempfile.TemporaryDirectory() as tmp:
        for seed in range(1, args.replicates + 1):
            folder = Path(tmp) / f"rep{seed}"
            make.write(make.simulate(seed), folder)
            for f in ("analysis.toml", "analysis_fixed.toml"):
                shutil.copy(EX11 / f, folder / f)
            if args.ratio is not None:
                spec = folder / "analysis.toml"
                text = spec.read_text(encoding="utf-8")
                assert "variance_ratio = 1.0" in text
                spec.write_text(text.replace("variance_ratio = 1.0",
                                             f"variance_ratio = {args.ratio!r}"), encoding="utf-8")
            r = compare.evaluate_all(folder, folder / "out")
            r["seed"] = seed
            reps.append(r)
            m = r["metrics"]
            print(f"seed {seed:3d}: " + "  ".join(
                f"{k}: corr {m[k]['corr_lambs']:.3f} bias_rams {m[k]['bias_rams']:+.3f} "
                f"bias_2024 {m[k]['bias_2024']:+.3f}" for k in MODELS), flush=True)
            shutil.rmtree(folder)
    summary = {k: {mt: _mc([r["metrics"][k][mt] for r in reps]) for mt in METRICS}
               for k in MODELS}
    paired = {k: {"abs_bias_rams_reduction": _mc([abs(r["metrics"]["none"]["bias_rams"])
                                                  - abs(r["metrics"][k]["bias_rams"])
                                                  for r in reps]),
                  "abs_bias_2024_reduction": _mc([abs(r["metrics"]["none"]["bias_2024"])
                                                  - abs(r["metrics"][k]["bias_2024"])
                                                  for r in reps]),
                  "corr_gain": _mc([r["metrics"][k]["corr_lambs"]
                                    - r["metrics"]["none"]["corr_lambs"] for r in reps])}
              for k in ("random", "fixed")}
    groups = {}
    for g in reps[0]["realised"]:
        groups[g] = {"realised_mean_tbv": _mc([r["realised"][g] for r in reps]),
                     **{f"{k}_minus_realised": _mc([r["groups"][k][g] - r["realised"][g]
                                                    for r in reps]) for k in ("random", "fixed")}}
    doc = {"study": "unknown-parent groups, example-11 scenario, weaning weight",
           "trend_of_origin_A": args.trend,
           "random_group_variance_ratio": 1.0 if args.ratio is None else args.ratio,
           "replicates": args.replicates, "seeds": f"1..{args.replicates}",
           "generator": "examples/11_sheep_upg/make_data.py (simulate(seed))",
           "true_variances": {"animal": 4.0, "residual": 12.0},
           "wall_seconds": time.time() - t0, "summary": summary,
           "paired_vs_none": paired, "groups": groups,
           "replicate_results": [{"seed": r["seed"], "metrics": r["metrics"],
                                  "groups": r["groups"], "realised": r["realised"]}
                                 for r in reps]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    for name, s in summary.items():
        print(name, {k: f"{v['mean']:.3f} +- {v['mc_se']:.3f}" for k, v in s.items()})
    for name, s in paired.items():
        print(name, "vs none", {k: f"{v['mean']:.3f} +- {v['mc_se']:.3f}" for k, v in s.items()})
    for g, s in groups.items():
        print(g, {k: f"{v['mean']:.3f} +- {v['mc_se']:.3f}" for k, v in s.items()})
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
