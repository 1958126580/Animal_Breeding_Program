#!/usr/bin/env python3
"""Compare evaluations with and without unknown-parent groups against the
simulation truth of example 11 (SYNTHETIC data; one replicate only).

Three evaluations of the same records:

* ``random`` - ``analysis.toml`` (random groups, ratio 1.0, REML variances);
* ``fixed``  - ``analysis_fixed.toml`` (fixed groups, simulation variances);
* ``none``   - the random-group spec without ``[upg]`` on a copy of the
  pedigree in which every group code is replaced by ``0``, i.e. purchased
  rams are treated as base animals (the usual default when groups are
  ignored).

For each the script reports, against the true breeding values: correlation
and regression slope ``b(TBV | EBV)`` for lambs, and mean bias
``mean(EBV - TBV)`` for purchased rams and the latest lamb crop.  Group
solutions are compared with the realised mean true breeding value of the
purchased rams in each group.  One replicate shows the direction of an
effect, not its size; see the user manual for how to study this properly.

Run from the repository root:  python examples/11_sheep_upg/compare.py
"""

from __future__ import annotations

import csv
import sys
import tempfile
import tomllib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from abp.workflows.evaluate import run_evaluation  # noqa: E402

HERE = Path(__file__).resolve().parent


def _read(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def no_group_spec(tmp: Path, folder: Path = HERE) -> Path:
    """Copy of the random-group example with groups replaced by unknown parents."""
    (tmp / "data").mkdir(parents=True, exist_ok=True)
    rows = _read(folder / "data" / "pedigree.csv")
    with open(tmp / "data" / "pedigree.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        for r in rows:
            for c in ("sire", "dam"):
                if r[c].startswith("UPG:"):
                    r[c] = "0"
            w.writerow(r)
    text = (folder / "analysis.toml").read_text(encoding="utf-8")
    start = text.index("[upg]")
    end = text.index("\n\n", start)
    text = text[:start] + text[end + 2:]                     # drop the [upg] table
    text = text.replace('name = "sheep_wwt_upg_random"', 'name = "sheep_wwt_no_groups"')
    weaning = (folder / "data" / "weaning.csv").as_posix()
    text = text.replace('phenotypes = "data/weaning.csv"', f'phenotypes = "{weaning}"')
    tomllib.loads(text)                                      # still a valid spec file
    path = tmp / "analysis_no_groups.toml"
    path.write_text(text, encoding="utf-8")
    return path


def metrics(ebv: dict[str, float], truth: dict[str, dict]) -> dict:
    lambs = [a for a, t in truth.items() if t["origin"] == "flock"]
    rams = [a for a, t in truth.items() if t["origin"] in ("A", "B")]
    last = [a for a in lambs if truth[a]["birth_year"] == "2024"]
    e = np.array([ebv[a] for a in lambs])
    u = np.array([float(truth[a]["tbv"]) for a in lambs])
    c = np.cov(e, u)
    return {"corr_lambs": float(np.corrcoef(e, u)[0, 1]),
            "slope_lambs": float(c[0, 1] / c[0, 0]),
            "bias_rams": float(np.mean([ebv[a] - float(truth[a]["tbv"]) for a in rams])),
            "bias_2024": float(np.mean([ebv[a] - float(truth[a]["tbv"]) for a in last]))}


def evaluate_all(folder: Path = HERE, out: Path | None = None) -> dict:
    """Run the three evaluations on the data in ``folder`` and score them."""
    truth = {r["id"]: r for r in _read(folder / "truth" / "tbv.csv")}
    ped = {r["id"]: r for r in _read(folder / "data" / "pedigree.csv")}
    realised = {}
    for a, r in ped.items():
        if r["sire"].startswith("UPG:"):
            realised.setdefault(r["sire"], []).append(float(truth[a]["tbv"]))
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(out) if out else Path(tmp)
        specs = {"random": folder / "analysis.toml", "fixed": folder / "analysis_fixed.toml",
                 "none": no_group_spec(Path(tmp) / "nogroups", folder)}
        res, groups = {}, {}
        for name, spec in specs.items():
            o = run_evaluation(spec, base / name, force=True, console=False)
            ebv = {r["animal"]: float(r["ebv"]) for r in _read(o.out_dir / "ebv_wwt.csv")}
            res[name] = metrics(ebv, truth)
            f = o.results["traits"]["wwt"]["files"].get("upg_solutions")
            if f:
                groups[name] = {r["group"]: float(r["solution"])
                                for r in _read(o.out_dir / f)}
    return {"metrics": res, "groups": groups,
            "realised": {g: float(np.mean(v)) for g, v in realised.items()},
            "n_rams": {g: len(v) for g, v in realised.items()}}


def main(out: Path | None = None) -> dict:
    r = evaluate_all(HERE, out)
    res, groups = r["metrics"], r["groups"]
    print("| model | corr(EBV,TBV) lambs | b(TBV|EBV) lambs | bias purchased rams | "
          "bias 2024 lambs |")
    print("|---|---:|---:|---:|---:|")
    for name, m in res.items():
        print(f"| {name} | {m['corr_lambs']:.3f} | {m['slope_lambs']:.3f} | "
              f"{m['bias_rams']:+.3f} | {m['bias_2024']:+.3f} |")
    print()
    print("| group | rams | realised mean TBV | random | fixed |")
    print("|---|---:|---:|---:|---:|")
    for g in sorted(r["realised"]):
        print(f"| {g} | {r['n_rams'][g]} | {r['realised'][g]:+.3f} | "
              f"{groups['random'][g]:+.3f} | {groups['fixed'][g]:+.3f} |")
    return r


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
