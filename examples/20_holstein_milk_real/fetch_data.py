#!/usr/bin/env python3
"""Download and convert the REAL data of example 20 (round 15).

Source: the ``milk`` and ``pedCowsR`` data sets of the R package ``pedigreemm``
(version 0.3-5, Bates, Vazquez & Perez Rodriguez; licence GPL (>= 2)), taken from the
read-only GitHub mirror of CRAN (``raw.githubusercontent.com/cran/pedigreemm/0.3-5``).
The package documentation describes ``milk`` as 3,397 lactation records (first to fifth
parity) of 1,359 Holstein cows, daughters of 38 sires in 57 herds, downloaded from the
USDA (AIPL) web site, with 305-day milk, fat and protein yield and the somatic cell
score (Vazquez, Bates, Rosa, Gianola & Weigel 2010, J Anim Sci 88:497-504).  The yields
are in pounds (mean milk 25,632; the documentation's 11,636 kg = 25,653 lb).
``pedCowsR`` is the pedigree (6,547 animals).

The data are NOT stored in this repository (their licence is the package's GPL, the
project licence is undecided); this script downloads them, checks their SHA-256 and
writes ``data/pedigree.csv`` and ``data/lactations.csv``.  Reading the R files needs the
``rdata`` package (``pip install rdata``; MIT licence), used only here.

Usage: python examples/20_holstein_milk_real/fetch_data.py
"""

from __future__ import annotations

import csv
import hashlib
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = "https://raw.githubusercontent.com/cran/pedigreemm/0.3-5/data/"
FILES = {"milk.rda": "6c3e02d21c5cd91d47ba3be990b4c8dc484c1b1dde7abde3804885d24fb84078",
         "pedCowsR.rda": "8ee04bcb5cd98bdceb81af408b78cf551dad8b2b98b4421e4fbfb87078de40ef"}


def download(name: str, dest: Path) -> bytes:
    if dest.exists():
        raw = dest.read_bytes()
    else:
        with urllib.request.urlopen(BASE + name, timeout=120) as r:
            raw = r.read()
        dest.write_bytes(raw)
    h = hashlib.sha256(raw).hexdigest()
    if h != FILES[name]:
        sys.exit(f"{name}: SHA-256 {h} differs from the recorded {FILES[name]}; not using it")
    return raw


def main():
    try:
        import rdata
    except ImportError:
        sys.exit("reading the R data files needs the 'rdata' package: pip install rdata")
    import warnings
    raw_dir = HERE / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        download(name, raw_dir / name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        milk = rdata.read_rda(str(raw_dir / "milk.rda"))["milk"]
        ped = rdata.read_rda(str(raw_dir / "pedCowsR.rda"))["pedCowsR"]
    labels = [str(x) for x in ped.label]
    sire = [None if m else int(v) for v, m in zip(ped.sire.data, ped.sire.mask)] \
        if hasattr(ped.sire, "mask") else [int(v) for v in ped.sire]
    dam = [None if m else int(v) for v, m in zip(ped.dam.data, ped.dam.mask)] \
        if hasattr(ped.dam, "mask") else [int(v) for v in ped.dam]
    if [int(x) for x in labels] != list(range(1, len(labels) + 1)):
        sys.exit("unexpected pedigree labels (expected 1..n)")
    # Sire repair (documented in README.md): pedCowsR lacks the sire of some cows whose
    # record carries a sire code ('milk$sire'); every sire code that also occurs with a
    # cow of known pedigree sire maps to exactly one pedigree sire, so the missing sire
    # is taken from that map.  data/pedigree_as_published.csv keeps the original.
    code_to_sire: dict = {}
    for r in milk.itertuples(index=False):
        s = sire[int(r.id) - 1]
        if s:
            code_to_sire.setdefault(str(r.sire), set()).add(s)
    if any(len(v) > 1 for v in code_to_sire.values()):
        sys.exit("a sire code maps to several pedigree sires; repair not possible")
    repaired = list(sire)
    n_rep = 0
    for r in milk.itertuples(index=False):
        k = int(r.id) - 1
        if repaired[k] is None and str(r.sire) in code_to_sire:
            repaired[k] = next(iter(code_to_sire[str(r.sire)]))
            n_rep += 1
    for fname, sires in (("pedigree_as_published.csv", sire), ("pedigree.csv", repaired)):
        with open(HERE / "data" / fname, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["id", "sire", "dam"])
            for i, a in enumerate(labels):
                w.writerow([f"C{a}", f"C{sires[i]}" if sires[i] else "0",
                            f"C{dam[i]}" if dam[i] else "0"])
    n_cows = len({int(r.id) for r in milk.itertuples(index=False)
                  if repaired[int(r.id) - 1] is not None and sire[int(r.id) - 1] is None})
    print(f"sire repair: {n_cows} cows ({n_rep} records) got their sire from the record's "
          f"sire code")
    with open(HERE / "data" / "lactations.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["record_id", "id", "lact", "herd", "sire_code", "dim", "milk", "fat",
                    "prot", "scs", "first_lact"])
        for k, r in enumerate(milk.itertuples(index=False)):
            w.writerow([k + 1, f"C{r.id}", int(r.lact), f"H{r.herd}", f"S{r.sire}",
                        int(r.dim), int(r.milk), int(r.fat), int(r.prot), float(r.scs),
                        int(r.lact) == 1])
    rows = list(csv.DictReader(open(HERE / "data" / "lactations.csv", encoding="utf-8")))
    with open(HERE / "data" / "lactations_first.csv", "w", encoding="utf-8",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(r for r in rows if r["lact"] == "1")
    print(f"wrote data/pedigree.csv ({len(labels)} animals), data/lactations.csv "
          f"({len(milk)} records) and data/lactations_first.csv (first lactations)")


if __name__ == "__main__":
    main()
