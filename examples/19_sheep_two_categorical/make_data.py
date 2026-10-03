#!/usr/bin/env python3
"""Generate the SYNTHETIC data of example 19 (deterministic, seed 20261003).

The lambs of ``examples/sheep_data`` (same animals, contemporary groups, sexes and birth
types as ``lambs.csv``) with three traits re-simulated on the liability scale:

* ``wwt``: weaning weight (kg), continuous;
* ``vigour``: lamb vigour score at birth, 3 ordered categories (1 weak, 2 medium,
  3 vigorous), thresholds 0 and 1.2 on the liability scale;
* ``surv``: survival to weaning, binary (1 died, 2 survived), threshold 0.

``l = mean + cg + sex + birth type + a + e`` per trait, ``a ~ N(0, A (x) G0)`` by gene
dropping through the flock pedigree (founders unrelated), ``e ~ N(0, R0)`` with
``R0 = diag(8, 1, 1)`` (independent residuals: the model of example 19 puts every trait
in its own residual group).  Lambs that died have no weaning weight.

True values go to ``truth/`` (validation only; analyses must not read them).

Usage: python examples/19_sheep_two_categorical/make_data.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import spsolve_triangular

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "sheep_data" / "data"
# genetic covariance matrix: wwt (kg^2), vigour and surv (liability scale)
SD = np.array([2.0, np.sqrt(0.3), np.sqrt(0.2)])
CORR = np.array([[1.0, 0.4, 0.3], [0.4, 1.0, 0.5], [0.3, 0.5, 1.0]])
G0 = CORR * np.outer(SD, SD)
R0 = np.diag([8.0, 1.0, 1.0])
MEAN = np.array([30.0, 0.6, 1.0])                  # liability means (vigour, surv)
SEX = {"M": np.array([1.5, -0.1, -0.1]), "F": np.array([-1.5, 0.1, 0.1])}
BIRTH_TYPE = {"1": np.array([2.0, 0.3, 0.3]), "2": np.zeros(3),
              "3": np.array([-2.5, -0.4, -0.5])}
THRESH_VIGOUR = (0.0, 1.2)
SEED = 20261003


def main():
    import sys
    sys.path.insert(0, str(HERE.parents[1] / "src"))
    from abp.core.pedigree import Pedigree
    rng = np.random.default_rng(SEED)
    prow = list(csv.DictReader(open(SRC / "pedigree.csv", encoding="utf-8")))
    ped = Pedigree.from_parent_ids([r["id"] for r in prow],
                                   [None if r["sire"] == "0" else r["sire"] for r in prow],
                                   [None if r["dam"] == "0" else r["dam"] for r in prow])
    F = spsolve_triangular(ped._l_matrix(), np.sqrt(ped.mendelian_d())[:, None]
                           * rng.standard_normal((ped.n, 3)), lower=True, unit_diagonal=True)
    U = F @ np.linalg.cholesky(G0).T
    lambs = list(csv.DictReader(open(SRC / "lambs.csv", encoding="utf-8")))
    cg_eff, out = {}, []
    for r in lambs:
        k = ped.index_of([r["id"]])[0]
        if r["cg"] not in cg_eff:
            cg_eff[r["cg"]] = rng.normal(0, 1.0, 3) * np.array([2.0, 0.3, 0.3])
        bt = r["birth_type"] if r["birth_type"] in BIRTH_TYPE else "2"
        liab = (MEAN + cg_eff[r["cg"]] + SEX[r["sex"]] + BIRTH_TYPE[bt] + U[k]
                + rng.standard_normal(3) * np.sqrt(np.diag(R0)))
        vigour = 1 + int(liab[1] > THRESH_VIGOUR[0]) + int(liab[1] > THRESH_VIGOUR[1])
        surv = 2 if liab[2] > 0.0 else 1
        wwt = f"{liab[0]:.1f}" if surv == 2 else "NA"
        out.append([r["id"], r["cg"], r["sex"], bt, wwt, vigour, surv])
    with open(HERE / "data" / "lambs_two_categorical.csv", "w", encoding="utf-8",
              newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["id", "cg", "sex", "birth_type", "wwt", "vigour", "surv"])
        w.writerows(out)
    with open(HERE / "truth" / "tbv.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["id", "tbv_wwt", "tbv_vigour", "tbv_surv"])
        for k, a in enumerate(ped.ids):
            w.writerow([a] + [f"{x:.6f}" for x in U[k]])
    (HERE / "truth" / "parameters.json").write_text(json.dumps(
        {"seed": SEED, "traits": ["wwt", "vigour", "surv"], "G0": G0.tolist(),
         "genetic_correlations": CORR.tolist(), "R0": R0.tolist(),
         "thresholds_vigour_liability": list(THRESH_VIGOUR),
         "note": "SYNTHETIC; for validation only"}, indent=2) + "\n", encoding="utf-8")
    vs = np.bincount([o[5] for o in out], minlength=4)[1:]
    ss = np.bincount([o[6] for o in out], minlength=3)[1:]
    print(f"wrote {len(out)} lambs; vigour 1/2/3: {vs.tolist()}; surv died/survived: "
          f"{ss.tolist()}")


if __name__ == "__main__":
    main()
