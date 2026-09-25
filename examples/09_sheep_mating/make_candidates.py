#!/usr/bin/env python3
"""Build candidates.csv for example 09 from example 06 (deterministic).

* merit: the economic index of example 06 (four-trait BLUP, SYNTHETIC weights);
* rams: the 40 highest-index male lambs born in 2025, capacity 25 matings each;
* ewes: all females born in 2024 or 2025, capacity 1 mating each;
* carrier: SYNTHETIC probability of carrying a recessive defect - 0 or 1 for
  genotyped candidates ("tested"), 0.10 (population prior) for the others;
  drawn with a fixed seed, for demonstration only.

Run from the repository root:  python examples/09_sheep_mating/make_candidates.py
"""

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from abp.workflows.evaluate import run_evaluation  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = ROOT / "examples" / "sheep_data" / "data"


def main():
    with tempfile.TemporaryDirectory() as tmp:
        out = run_evaluation(ROOT / "examples" / "06_sheep_multitrait_index" / "analysis.toml",
                             Path(tmp) / "ex06", console=False)
        with open(out.out_dir / "index.csv", encoding="utf-8") as fh:
            index = {r["animal"]: float(r["index"]) for r in csv.DictReader(fh)}
    with open(DATA / "pedigree.csv", encoding="utf-8") as fh:
        ped = {r["id"]: r for r in csv.DictReader(fh)}
    with open(DATA / "genotypes.csv", encoding="utf-8") as fh:
        genotyped = {line.split(",", 1)[0] for line in list(fh)[1:]}
    rams = sorted((a for a, r in ped.items() if r["sex"] == "M" and r["birth_date"].startswith("2025")),
                  key=lambda a: (-index[a], a))[:40]
    ewes = sorted(a for a, r in ped.items() if r["sex"] == "F" and r["birth_date"][:4] in ("2024", "2025"))
    rng = np.random.default_rng(20260925)
    rows = []
    for a in rams + ewes:
        if a in genotyped:
            carrier = 1.0 if rng.random() < 0.10 else 0.0
        else:
            carrier = 0.10
        rows.append([a, ped[a]["sex"], repr(round(index[a], 6)), 25 if a in rams else 1, carrier])
    with open(HERE / "candidates.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["id", "sex", "merit", "capacity", "carrier"])
        w.writerows(rows)
    print(f"wrote {HERE / 'candidates.csv'}: {len(rams)} rams, {len(ewes)} ewes")


if __name__ == "__main__":
    main()
