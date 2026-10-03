#!/usr/bin/env python3
"""Download and convert the REAL data of example 21 (round 15).

Source: the ``mice`` data of the R package ``BGLR`` (version 1.1.4, de los Campos &
Perez Rodriguez; licence GPL-3), from the read-only GitHub mirror of CRAN
(``raw.githubusercontent.com/cran/BGLR/1.1.4/data/mice.RData``).  The package
documents 1,814 heterogeneous-stock mice genotyped for 10,346 SNPs, with phenotypes and
covariates from the Wellcome Trust experiment of Valdar et al. (2006, Nat Genet 38:879;
Genetics 174:959), analysed for genomic selection by Legarra et al. (2008, Genetics
180:611).  Genotypes are coded 0/1/2; the SNP name carries the counted allele
(``rs3683945_G``: copies of G).  The map gives the two alleles and a position column
``mbp``; the genome assembly is not stated in the source.

The data are NOT stored in this repository (GPL-3; project licence undecided).  This
script checks the SHA-256 and writes ``data/genotypes.csv`` (ABP dosage matrix),
``data/markers.csv``, ``data/phenotypes.csv`` and ``data/pedigree_A.npy`` (the
pedigree relationship matrix ``mice.A`` shipped with the data, same order as the mice).  Needs ``pip install rdata`` (MIT).

Usage: python examples/21_mice_bodyweight_real/fetch_data.py
"""

from __future__ import annotations

import csv
import hashlib
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
URL = "https://raw.githubusercontent.com/cran/BGLR/1.1.4/data/mice.RData"
SHA256 = "5aec95f7736460cbc98f90fbf02dc12693786d24353bea054768c0da8ae9e3a5"
ASSEMBLY = "BGLR-1.1.4-mice.map (assembly not stated by the source)"


def main():
    try:
        import rdata
    except ImportError:
        sys.exit("reading the R data file needs the 'rdata' package: pip install rdata")
    import warnings

    import numpy as np
    raw_dir = HERE / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    f = raw_dir / "mice.RData"
    if not f.exists():
        with urllib.request.urlopen(URL, timeout=300) as r:
            f.write_bytes(r.read())
    h = hashlib.sha256(f.read_bytes()).hexdigest()
    if h != SHA256:
        sys.exit(f"mice.RData: SHA-256 {h} differs from the recorded {SHA256}; not using it")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = rdata.read_rda(str(f))
    pheno, X, mp = x["mice.pheno"], x["mice.X"], x["mice.map"]
    ids = [str(v) for v in X.coords[X.dims[0]].values]
    if ids != [str(v) for v in pheno["SUBJECT.NAME"]]:
        sys.exit("genotype and phenotype rows are not in the same order")
    G = np.asarray(X, dtype=float)
    snps = [str(s) for s in mp["snp_id"]]
    with open(HERE / "data" / "markers.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["marker_id", "chrom", "pos", "ref", "alt", "counted_allele", "assembly"])
        for s, c, pos, al in zip(snps, mp["chr"], mp["mbp"], mp["alleles"]):
            counted = s.rsplit("_", 1)[1]
            a, b = str(al).split(";")
            other = a if b == counted else b
            if counted not in (a, b):
                sys.exit(f"{s}: counted allele not among {al}")
            w.writerow([s, str(c), int(round(float(pos) * 1e6)), other, counted, counted,
                        ASSEMBLY])
    with open(HERE / "data" / "genotypes.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["id"] + snps)
        for i, a in enumerate(ids):
            row = G[i]
            w.writerow([a] + ["NA" if np.isnan(v) else str(int(v)) for v in row])
    cols = ["SUBJECT.NAME", "GENDER", "Obesity.Date.Year", "Obesity.Date.Season", "Litter",
            "cage", "Obesity.EndNormalBW", "Obesity.BodyLength", "Obesity.BMI"]
    out = ["id", "sex", "year", "season", "litter", "cage", "bw", "body_length", "bmi"]
    with open(HERE / "data" / "phenotypes.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(out + ["year_season"])
        for r in pheno[cols].itertuples(index=False):
            vals = ["NA" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v)
                    for v in r]
            w.writerow(vals + [f"{vals[2]}-{vals[3]}"])
    A = np.asarray(x["mice.A"], dtype=float)          # pedigree relationship (BGLR)
    np.save(HERE / "data" / "pedigree_A.npy", A)
    print(f"wrote data/genotypes.csv ({len(ids)} mice x {len(snps)} SNPs), data/markers.csv, "
          "data/phenotypes.csv")


if __name__ == "__main__":
    main()
