#!/usr/bin/env python3
"""Generate the SYNTHETIC data of example 11 (unknown-parent groups).

Scenario: a closed ewe flock that buys all its rams from two external
breeders whose animals have no recorded ancestry.

* base flock (parents unknown and *not* grouped): 6 rams born 2013 and
  150 ewes born 2011-2015, breeding values ``u ~ N(0, sigma_a^2)`` - they
  define the genetic base;
* purchased rams (3 per year, 2016-2024, born two years earlier): origin A
  (2 per year) runs its own selection programme, so the mean breeding value
  of its rams rises by 0.6 kg per year of purchase; origin B (1 per year)
  sells rams 1 kg below the base mean.  Their unknown parents are coded as
  groups by origin and purchase period (``UPG:A_16_19``, ``UPG:A_20_24``,
  ``UPG:B``); a purchased ram's breeding value is its group mean plus a
  base-like deviation ``N(0, sigma_a^2)``;
* matings: every year each ewe of age 1-6 produces one lamb; sires are the
  rams bought that year and the year before (in 2016 also the base rams),
  chosen at random; after lambing, ewes aged 6 and a random 15% of the
  others leave and randomly chosen female lambs refill the flock to 150;
* lamb breeding values follow the infinitesimal model with Mendelian
  sampling variance ``sigma_a^2/2 (1 - (F_s + F_d)/2)``;
* weaning weight: ``30 + sex (M +2 kg) + year + u + e`` with
  ``sigma_a^2 = 4``, ``sigma_e^2 = 12`` (h2 = 0.25).

Everything is drawn from one NumPy PCG64 stream with a fixed seed, so the
files are reproducible byte for byte.  Truth files (breeding values and
group means) are written for the comparison in ``compare.py`` and must
never be used as model input.

Run from the repository root:  python examples/11_sheep_upg/make_data.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from abp.core.pedigree import Pedigree  # noqa: E402  (inbreeding of the simulated parents)

HERE = Path(__file__).resolve().parent
SEED = 20260925
SIGMA_A2, SIGMA_E2 = 4.0, 12.0
YEARS = range(2016, 2025)
N_EWES = 150


def origin_mean(origin: str, year: int) -> float:
    """True mean breeding value of rams bought from an origin in a year (kg)."""
    return 0.6 * (year - 2014) if origin == "A" else -1.0


def group_of(origin: str, year: int) -> str:
    if origin == "B":
        return "UPG:B"
    return "UPG:A_16_19" if year <= 2019 else "UPG:A_20_24"


def simulate(seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    ids, sire, dam, sex, born, origin, grp = [], [], [], [], [], [], []

    def add(a, s, d, sx, b, o, g):
        ids.append(a)
        sire.append(s)
        dam.append(d)
        sex.append(sx)
        born.append(b)
        origin.append(o)
        grp.append(g)

    for k in range(6):
        add(f"R13{k + 1:03d}", None, None, "M", 2013, "base", None)
    for k in range(N_EWES):
        add(f"E{11 + k % 5}{k + 1:03d}", None, None, "F", 2011 + k % 5, "base", None)
    base_rams = [a for a in ids if a.startswith("R")]
    ewes = {a: b for a, b in zip(ids, born) if a.startswith("E")}   # active ewe -> birth year
    bought: dict[int, list[str]] = {}
    year_effect = {y: float(v) for y, v in zip(YEARS, rng.normal(0.0, 1.5, len(YEARS)))}
    lambs_by_year: dict[int, list[int]] = {}
    for y in YEARS:
        bought[y] = []
        for k, o in enumerate(("A", "A", "B")):
            a = f"P{y % 100:02d}{o}{k + 1}"
            add(a, None, None, "M", y - 2, o, group_of(o, y))
            bought[y].append(a)
        sires = bought[y] + bought.get(y - 1, []) + (base_rams if y == 2016 else [])
        dams = sorted(a for a, b in ewes.items() if 1 <= y - b <= 6)
        lambs_by_year[y] = []
        for k, d in enumerate(dams):
            s = sires[int(rng.integers(len(sires)))]
            sx = "M" if rng.random() < 0.5 else "F"
            add(f"L{y % 100:02d}{k + 1:04d}", s, d, sx, y, "flock", None)
            lambs_by_year[y].append(len(ids) - 1)
        # replacements: ewes aged 6 and a random 15% of the others leave
        for a in sorted(ewes):
            if y - ewes[a] >= 6 or rng.random() < 0.15:
                del ewes[a]
        females = [i for i in lambs_by_year[y] if sex[i] == "F"]
        need = N_EWES - len(ewes)
        for i in rng.permutation(females)[:max(need, 0)]:
            ewes[ids[i]] = y

    n = len(ids)
    pos = {a: i for i, a in enumerate(ids)}
    ped = Pedigree.from_parent_ids(ids, sire, dam)
    F = ped.inbreeding()[ped.index_of(ids)]
    u = np.zeros(n)
    for i in range(n):                                        # input order: parents first
        if sire[i] is None:
            mu = origin_mean(origin[i], born[i] + 2) if origin[i] in ("A", "B") else 0.0
            u[i] = mu + rng.normal(0.0, np.sqrt(SIGMA_A2))
        else:
            s, d = pos[sire[i]], pos[dam[i]]
            v = 0.5 * SIGMA_A2 * (1.0 - 0.5 * (F[s] + F[d]))
            u[i] = 0.5 * (u[s] + u[d]) + rng.normal(0.0, np.sqrt(v))
    phen = []
    for y in YEARS:
        for i in lambs_by_year[y]:
            e = rng.normal(0.0, np.sqrt(SIGMA_E2))
            phen.append((ids[i], sex[i], str(y), 30.0 + (2.0 if sex[i] == "M" else 0.0)
                         + year_effect[y] + u[i] + e))
    return {"ids": ids, "sire": sire, "dam": dam, "sex": sex, "born": born, "origin": origin,
            "group": grp, "u": u, "F": F, "phen": phen}


def write(sim: dict, out: Path = HERE) -> None:
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "truth").mkdir(parents=True, exist_ok=True)
    with open(out / "data" / "pedigree.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["id", "sire", "dam", "sex", "birth_year"])
        for a, s, d, sx, b, g in zip(sim["ids"], sim["sire"], sim["dam"], sim["sex"],
                                     sim["born"], sim["group"]):
            w.writerow([a, s or g or "0", d or g or "0", sx, b])
    with open(out / "data" / "weaning.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["id", "sex", "year", "wwt"])
        for a, sx, y, v in sim["phen"]:
            w.writerow([a, sx, y, f"{v:.2f}"])
    with open(out / "truth" / "tbv.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["id", "origin", "birth_year", "tbv", "inbreeding"])
        for a, o, b, v, f in zip(sim["ids"], sim["origin"], sim["born"], sim["u"], sim["F"]):
            w.writerow([a, o, b, f"{v:.6f}", f"{f:.6f}"])
    with open(out / "truth" / "groups.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["group", "true_mean_breeding_value"])
        for g, years in (("UPG:A_16_19", range(2016, 2020)), ("UPG:A_20_24", range(2020, 2025))):
            w.writerow([g, f"{np.mean([origin_mean('A', y) for y in years]):.4f}"])
        w.writerow(["UPG:B", f"{origin_mean('B', 2016):.4f}"])


if __name__ == "__main__":
    s = simulate()
    write(s)
    print(f"example 11: {len(s['ids'])} animals, {len(s['phen'])} weaning weights written")
