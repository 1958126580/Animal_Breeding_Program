#!/usr/bin/env python3
"""Inbreeding kernels on large pedigrees (round 9).

Times the round-1 Meuwissen-Luo kernel (``_native.inbreeding_ml``) against the
depth kernel (``_native.inbreeding_depth``) with the automatic choice (mode 0), the
trace forced (mode 1) and the Colleau columns forced (mode 2), and checks that all
give the same F (max |difference|).  Pedigrees:

* discrete generations (``run_benchmarks.sim_pedigree``): 20 x 10,000 with 50 or
  1,000 sires per generation, 10 x 20,000 with 2,000 sires;
* overlapping generations: 100,000 animals, parents drawn from the previous 3,000
  animals, 5% unknown parents; sires the first 300 males of that window, or any male.

Usage: python benchmarks/inbreeding_benchmark.py [--out docs/validation/inbreeding_benchmark.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from abp.core import pedigree as pmod  # noqa: E402
from run_benchmarks import sim_pedigree  # noqa: E402


def overlapping(n, n_founders, window, n_sires, seed=1):
    rng = np.random.default_rng(seed)
    sire = np.full(n, -1, dtype=np.int64)
    dam = np.full(n, -1, dtype=np.int64)
    male = rng.random(n) < 0.5
    for i in range(n_founders, n):
        lo = max(0, i - window)
        m = np.flatnonzero(male[lo:i]) + lo
        f = np.flatnonzero(~male[lo:i]) + lo
        if m.size and rng.random() > 0.05:
            sire[i] = m[rng.integers(min(n_sires, m.size))]
        if f.size and rng.random() > 0.05:
            dam[i] = f[rng.integers(f.size)]
    return sire, dam


def discrete(gens, per_gen, n_sires):
    ids, s, d = sim_pedigree(gens, per_gen, n_sires)
    ix = {a: i for i, a in enumerate(ids)}
    return (np.array([ix[x] if x else -1 for x in s], dtype=np.int64),
            np.array([ix[x] if x else -1 for x in d], dtype=np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "validation" / "inbreeding_benchmark.json"))
    a = ap.parse_args()
    nat = pmod._native
    if nat is None or not hasattr(nat, "inbreeding_depth"):
        sys.exit("native kernel with inbreeding_depth not available")
    cases = {
        "discrete_20x10000_50_sires": discrete(20, 10000, 50),
        "discrete_20x10000_1000_sires": discrete(20, 10000, 1000),
        "discrete_10x20000_2000_sires": discrete(10, 20000, 2000),
        "overlapping_100000_300_sires": overlapping(100000, 500, 3000, 300),
        "overlapping_100000_all_males": overlapping(100000, 500, 3000, 10 ** 9),
    }
    rows = []
    for name, (s, d) in cases.items():
        t = time.perf_counter()
        F = np.frombuffer(nat.inbreeding_ml(s, d))
        rec = {"case": name, "n": int(s.size), "mean_F": float(F.mean()),
               "meuwissen_luo_kernel_s": time.perf_counter() - t}
        for mode in (0, 1, 2):
            t = time.perf_counter()
            raw, depths, n_ml, n_col, n_cd = nat.inbreeding_depth(s, d, mode)
            rec[f"depth_mode{mode}"] = {
                "s": time.perf_counter() - t, "depths": depths, "meuwissen_luo_pairs": n_ml,
                "colleau_columns": n_col, "colleau_depths": n_cd,
                "max_abs_diff": float(np.max(np.abs(np.frombuffer(raw) - F)))}
        rows.append(rec)
        print(f"{name}: n {rec['n']}, round-1 kernel {rec['meuwissen_luo_kernel_s']:.2f} s, "
              + ", ".join(f"mode {m} {rec[f'depth_mode{m}']['s']:.2f} s "
                          f"(diff {rec[f'depth_mode{m}']['max_abs_diff']:.1e})" for m in (0, 1, 2)),
              flush=True)
    from abp.workflows.manifest import environment
    Path(a.out).write_text(json.dumps({"benchmark": "inbreeding kernels", "environment":
                                       environment(), "cases": rows}, indent=2) + "\n",
                           encoding="utf-8")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
