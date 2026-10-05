#!/usr/bin/env python3
"""Generate the SYNTHETIC data of example 17 (deterministic, seed 20261001).

Two repeated ewe traits recorded at every lambing of the flock in
``examples/sheep_data`` (same ewes, years, parities and flocks as ``lambing.csv``):

* ``lwt``  litter weight at weaning (kg);
* ``bcs``  ewe body condition score at weaning (continuous score).

Model: ``y = flock-year (column flock_year) + parity + u + pe + e`` per trait, with
``u ~ N(0, A (x) G0)`` by gene dropping through the flock pedigree (founders
unrelated), ``pe ~ N(0, I (x) P0)`` per ewe, ``e ~ N(0, R0)`` per record; about 10%
of the bcs scores are missing.  The true values are written to ``truth/`` (for
validation only; analyses must not read them).

Usage: python examples/17_sheep_ewe_repeated_bayes/make_data.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import spsolve_triangular

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "sheep_data" / "data"
G0 = np.array([[4.0, 0.3], [0.3, 0.06]])         # h2 about 0.2 / 0.15
P0 = np.array([[3.0, 0.2], [0.2, 0.05]])
R0 = np.array([[13.0, 0.6], [0.6, 0.29]])
MEANS = np.array([28.0, 3.0])
PARITY = {"1": np.array([-3.0, -0.10]), "2": np.array([0.0, 0.0]),
          "3": np.array([1.5, 0.05]), "4": np.array([1.0, 0.0])}
SEED = 20261001


def main():
    import sys
    sys.path.insert(0, str(HERE.parents[1] / "src"))
    from abp.core.pedigree import Pedigree
    rng = np.random.default_rng(SEED)
    rows = list(csv.DictReader(open(SRC / "pedigree.csv", encoding="utf-8")))
    ped = Pedigree.from_parent_ids([r["id"] for r in rows],
                                   [None if r["sire"] == "0" else r["sire"] for r in rows],
                                   [None if r["dam"] == "0" else r["dam"] for r in rows])
    F = spsolve_triangular(ped._l_matrix(), np.sqrt(ped.mendelian_d())[:, None]
                           * rng.standard_normal((ped.n, 2)), lower=True, unit_diagonal=True)
    U = F @ np.linalg.cholesky(G0).T
    lam = list(csv.DictReader(open(SRC / "lambing.csv", encoding="utf-8")))
    ewes = sorted({r["id"] for r in lam})
    PE = {e: rng.standard_normal(2) @ np.linalg.cholesky(P0).T for e in ewes}
    fy = {}
    out = []
    Lr = np.linalg.cholesky(R0)
    for r in lam:
        key = (r["flock"], r["year"])
        if key not in fy:
            fy[key] = rng.normal(0, 1, 2) * np.array([2.0, 0.15])
        i = ped.index_of([r["id"]])[0]
        y = MEANS + fy[key] + PARITY[r["parity"]] + U[i] + PE[r["id"]] + Lr @ rng.standard_normal(2)
        bcs = "NA" if rng.random() < 0.1 else f"{y[1]:.2f}"
        out.append([r["record_id"], r["id"], r["year"], r["parity"], r["flock"],
                    f"{r['flock']}-{r['year']}", f"{y[0]:.1f}", bcs])
    with open(HERE / "data" / "ewe_records.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["record_id", "id", "year", "parity", "flock", "flock_year", "lwt", "bcs"])
        w.writerows(out)
    with open(HERE / "truth" / "tbv.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["id", "tbv_lwt", "tbv_bcs"])
        for k, a in enumerate(ped.ids):
            w.writerow([a, f"{U[k, 0]:.6f}", f"{U[k, 1]:.6f}"])
    (HERE / "truth" / "parameters.json").write_text(json.dumps(
        {"seed": SEED, "G0": G0.tolist(), "P0": P0.tolist(), "R0": R0.tolist(),
         "note": "SYNTHETIC; for validation only"}, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(out)} records of {len(ewes)} ewes")


if __name__ == "__main__":
    main()
