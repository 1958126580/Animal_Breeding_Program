#!/usr/bin/env python3
"""Generate the SYNTHETIC data of example 18 (deterministic, seed 20261002).

Weaning weight of the lambs of ``examples/sheep_data`` (same lambs, dams, contemporary
groups, sexes and birth types as ``lambs.csv``) re-simulated with maternal effects:

``wwt = mean + cg + sex + birth type + a_lamb + m_dam + c_dam + e``

* ``(a, m) ~ N(0, A (x) G0)``: direct and maternal genetic effects by gene dropping
  through the flock pedigree (founders unrelated), direct-maternal correlation -0.35;
* ``c ~ N(0, var_c)``: maternal permanent environment of the dam (shared by her
  lambs of all years);
* ``e ~ N(0, var_e)``.

True values go to ``truth/`` (validation only; analyses must not read them).

Usage: python examples/18_sheep_wwt_maternal_bayes/make_data.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import spsolve_triangular

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "sheep_data" / "data"
G0 = np.array([[4.0, -1.0], [-1.0, 2.0]])          # direct, maternal (kg^2)
VAR_C = 1.5
VAR_E = 8.0
SEX = {"M": 1.5, "F": -1.5}
BIRTH_TYPE = {"1": 2.0, "2": 0.0, "3": -2.5}
SEED = 20261002


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
                           * rng.standard_normal((ped.n, 2)), lower=True, unit_diagonal=True)
    AM = F @ np.linalg.cholesky(G0).T
    lambs = list(csv.DictReader(open(SRC / "lambs.csv", encoding="utf-8")))
    cg_eff, c_eff, out = {}, {}, []
    for r in lambs:
        k = ped.index_of([r["id"]])[0]
        dk = int(ped.dam[k])
        if dk < 0:
            continue
        dam = ped.ids[dk]
        if r["cg"] not in cg_eff:
            cg_eff[r["cg"]] = rng.normal(0, 2.0)
        if dam not in c_eff:
            c_eff[dam] = rng.normal(0, np.sqrt(VAR_C))
        bt = r["birth_type"] if r["birth_type"] in BIRTH_TYPE else "2"
        y = (30.0 + cg_eff[r["cg"]] + SEX[r["sex"]] + BIRTH_TYPE[bt] + AM[k, 0] + AM[dk, 1]
             + c_eff[dam] + rng.normal(0, np.sqrt(VAR_E)))
        out.append([r["id"], dam, r["cg"], r["sex"], bt, f"{y:.1f}"])
    with open(HERE / "data" / "lambs_maternal.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["id", "dam", "cg", "sex", "birth_type", "wwt"])
        w.writerows(out)
    with open(HERE / "truth" / "tbv.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["id", "tbv_direct", "tbv_maternal"])
        for k, a in enumerate(ped.ids):
            w.writerow([a, f"{AM[k, 0]:.6f}", f"{AM[k, 1]:.6f}"])
    (HERE / "truth" / "parameters.json").write_text(json.dumps(
        {"seed": SEED, "G0_direct_maternal": G0.tolist(), "var_maternal_pe": VAR_C,
         "var_residual": VAR_E, "note": "SYNTHETIC; for validation only"}, indent=2) + "\n",
        encoding="utf-8")
    print(f"wrote {len(out)} lambs of {len(c_eff)} dams")


if __name__ == "__main__":
    main()
