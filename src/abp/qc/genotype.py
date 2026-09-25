"""Genotype import (ABP dosage-matrix format) and genotype QC (M01/M02/M06).

File contracts
--------------
*Marker map* (CSV): ``marker_id, chrom, pos, ref, alt, counted_allele,
assembly``.  ``counted_allele`` must equal ``ref`` or ``alt``; one assembly
per file.  Strand-ambiguous SNPs (A/T, C/G) are flagged for review because
their orientation cannot be checked from allele names.

*Dosage matrix* (CSV): column ``id`` then one column per marker (any order;
the header must contain exactly the markers of the map).  Cells hold the
number of copies of the counted allele (0..2; decimals allowed for imputed
dosages) or a missing-value token.  This is ABP's own documented text
format; PLINK 1 binary files are read by :mod:`abp.io.plink`; VCF/BGEN
readers are *not* provided (not_run).

QC rules (all reversible; excluded items are listed)
----------------------------------------------------
GEN-DOSAGE-RANGE     error        dosage outside [0, 2]
GEN-ID               error        duplicated / invalid animal IDs
GEN-MAP              error        map/header mismatch, bad counted allele,
                                  several assemblies
GEN-AMBIGUOUS        review       A/T or C/G markers
GEN-CALLRATE-ANIMAL  quarantine   animal call rate < min_call_rate_animal
GEN-CALLRATE-MARKER  quarantine   marker call rate < min_call_rate_marker
GEN-MONOMORPHIC      quarantine   no variation in the frequency sample
GEN-MAF              quarantine   MAF < min_maf (only if min_maf > 0)
GEN-LOW-MAF          review       0 < MAF < 0.01 (kept)
GEN-MENDEL           review/error opposing homozygotes parent-offspring:
                                  >1% review, >5% error (identity conflict)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.genomic import allele_frequencies, minor_allele_frequency
from ..io.tables import Table, check_identifier
from .report import QCReport

MAP_COLUMNS = ("marker_id", "chrom", "pos", "ref", "alt", "counted_allele", "assembly")
AMBIGUOUS = {frozenset("AT"), frozenset("CG")}
MENDEL_REVIEW, MENDEL_ERROR = 0.01, 0.05


@dataclass
class GenotypeData:
    ids: list[str]
    markers: list[str]
    dosage: np.ndarray        # n x m float64 (missing entries hold 0; see ``missing``)
    missing: np.ndarray       # n x m bool
    assembly: str
    counted_allele: list[str]
    qc: QCReport
    sha256: dict[str, str]


def load_genotypes(geno: Table, mapt: Table, missing_tokens: set[str]) -> GenotypeData:
    qc = QCReport("genotypes")
    for c in MAP_COLUMNS:
        mapt.column(c)  # raises SCHEMA_MISSING_COLUMN if absent
    mid = mapt.column("marker_id")
    dup = sorted({m for m in mid if mid.count(m) > 1}) if len(set(mid)) != len(mid) else []
    if dup:
        qc.add("GEN-MAP", "error", "duplicated marker_id in the marker map",
               [{"marker_id": m} for m in dup], "DUPLICATE_KEY")
        qc.raise_if_blocking()
    ref, alt, cnt = mapt.column("ref"), mapt.column("alt"), mapt.column("counted_allele")
    asm = set(mapt.column("assembly"))
    bad = [{"marker_id": m, "ref": r, "alt": a, "counted_allele": c}
           for m, r, a, c in zip(mid, ref, alt, cnt) if c not in (r, a) or r == a or not r or not a]
    if bad:
        qc.add("GEN-MAP", "error", "counted_allele must equal ref or alt (ref != alt)", bad,
               "GENOTYPE_ALLELE_MISMATCH")
    if len(asm) != 1 or "" in asm:
        qc.add("GEN-MAP", "error", f"exactly one non-empty assembly is required, found {sorted(asm)}",
               [{"assemblies": sorted(asm)}], "GENOTYPE_ALLELE_MISMATCH")
    qc.raise_if_blocking()
    amb = [{"marker_id": m, "alleles": f"{r}/{a}"} for m, r, a in zip(mid, ref, alt)
           if frozenset((r.upper(), a.upper())) in AMBIGUOUS]
    if amb:
        qc.add("GEN-AMBIGUOUS", "review", "strand-ambiguous (A/T, C/G) markers: orientation "
               "cannot be verified from allele names", amb)

    header = list(geno.header)
    if header[0] != "id":
        qc.add("GEN-MAP", "error", "the first column of the genotype file must be 'id'", [],
               "GENOTYPE_ID_CONFLICT")
        qc.raise_if_blocking()
    gm = header[1:]
    if set(gm) != set(mid) or len(gm) != len(mid):
        extra = sorted(set(gm) - set(mid))[:20]
        lack = sorted(set(mid) - set(gm))[:20]
        qc.add("GEN-MAP", "error", "genotype columns and marker map differ",
               [{"only_in_genotypes": extra, "only_in_map": lack}], "GENOTYPE_ID_CONFLICT")
        qc.raise_if_blocking()
    col_of = {m: k + 1 for k, m in enumerate(gm)}
    order = np.array([col_of[m] for m in mid])
    ids = [r[0] for r in geno.rows]
    for k, a in enumerate(ids):
        check_identifier(a, geno, k, "id")
    dups = sorted({a for a in ids if ids.count(a) > 1}) if len(set(ids)) != len(ids) else []
    if dups:
        qc.add("GEN-ID", "error", "duplicated animal IDs in the genotype file",
               [{"id": a} for a in dups], "DUPLICATE_KEY")
        qc.raise_if_blocking()
    n, m = len(ids), len(mid)
    raw = np.array([[r[c] for c in order] for r in geno.rows], dtype=object) if n else np.empty((0, m))
    missing = np.isin(raw, list(missing_tokens))
    M = np.zeros((n, m))
    bad_cells = []
    try:  # fast vectorized path; the loop below only runs to locate bad cells
        fast = np.where(missing, "0", raw).astype(np.float64)
        if np.all((fast >= 0.0) & (fast <= 2.0)):
            M = fast
            n_scan = 0
        else:
            n_scan = n
    except ValueError:
        n_scan = n
    for i in range(n_scan):
        for j in np.flatnonzero(~missing[i]):
            try:
                v = float(raw[i, j])
            except ValueError:
                v = float("nan")
            if not (0.0 <= v <= 2.0):
                if len(bad_cells) < 20:
                    bad_cells.append({"line": geno.lines[i], "id": ids[i], "marker": mid[j],
                                      "value": raw[i, j]})
                else:
                    bad_cells.append(None)
            else:
                M[i, j] = v
    if bad_cells:
        qc.add("GEN-DOSAGE-RANGE", "error", "dosages must be numbers in [0, 2] or missing codes",
               [b for b in bad_cells if b], "GENOTYPE_DOSAGE_RANGE")
        qc.findings[-1].count = len(bad_cells)
        qc.raise_if_blocking()
    noninteger = int(np.sum((M != np.round(M)) & ~missing))
    if noninteger:
        qc.add("GEN-FRACTIONAL", "info", f"{noninteger} fractional (imputed) dosages present")
    qc.stats = {"n_animals_in_file": n, "n_markers_in_file": m, "assembly": next(iter(asm)),
                "missing_rate": float(missing.mean()) if missing.size else 0.0}
    return GenotypeData(ids, list(mid), M, missing, next(iter(asm)), list(cnt), qc,
                        {"genotypes": geno.sha256, "marker_map": mapt.sha256})


def filter_genotypes(g: GenotypeData, cfg: dict, freq_sample: np.ndarray | None = None,
                     parents: dict[str, tuple[str | None, str | None]] | None = None) -> tuple[GenotypeData, np.ndarray]:
    """Apply call-rate, monomorphism, MAF and Mendelian QC.

    ``freq_sample`` selects the rows (after animal QC) used to compute the
    frequencies that define monomorphism/MAF.  Returns the filtered data and
    the frequencies of the kept markers in that sample.
    """
    qc = g.qc
    obs = ~g.missing
    a_cr = obs.mean(axis=1) if g.missing.shape[1] else np.ones(len(g.ids))
    keep_a = a_cr >= cfg["min_call_rate_animal"]
    if (~keep_a).any():
        items = [{"id": g.ids[i], "call_rate": float(a_cr[i])} for i in np.flatnonzero(~keep_a)]
        qc.add("GEN-CALLRATE-ANIMAL", "quarantine",
               f"animals with call rate < {cfg['min_call_rate_animal']}", items)
        qc.excluded.extend({**it, "reason": "GEN-CALLRATE-ANIMAL"} for it in items)
    m_cr = obs[keep_a].mean(axis=0)
    keep_m = m_cr >= cfg["min_call_rate_marker"]
    if (~keep_m).any():
        items = [{"marker_id": g.markers[j], "call_rate": float(m_cr[j])}
                 for j in np.flatnonzero(~keep_m)]
        qc.add("GEN-CALLRATE-MARKER", "quarantine",
               f"markers with call rate < {cfg['min_call_rate_marker']}", items)
        qc.excluded.extend({**it, "reason": "GEN-CALLRATE-MARKER"} for it in items)
    rows = np.flatnonzero(keep_a)
    sample = rows if freq_sample is None else np.intersect1d(rows, freq_sample)
    if sample.size == 0:
        from ..errors import ABPError
        raise ABPError("GENOTYPE_ZERO_SCALING", "no genotyped animal in the frequency sample")
    p = allele_frequencies(g.dosage[sample], g.missing[sample])
    maf = minor_allele_frequency(np.nan_to_num(p, nan=0.0))
    mono = keep_m & ((maf == 0) | np.isnan(p))
    if mono.any():
        items = [{"marker_id": g.markers[j]} for j in np.flatnonzero(mono)]
        qc.add("GEN-MONOMORPHIC", "quarantine", "monomorphic markers in the frequency sample "
               "(no information)", items)
        qc.excluded.extend({**it, "reason": "GEN-MONOMORPHIC"} for it in items)
    keep_m &= ~mono
    if cfg["min_maf"] > 0:
        low = keep_m & (maf < cfg["min_maf"])
        if low.any():
            items = [{"marker_id": g.markers[j], "maf": float(maf[j])} for j in np.flatnonzero(low)]
            qc.add("GEN-MAF", "quarantine", f"markers with MAF < {cfg['min_maf']}", items)
            qc.excluded.extend({**it, "reason": "GEN-MAF"} for it in items)
            keep_m &= ~low
    rare = keep_m & (maf < 0.01)
    if rare.any():
        qc.add("GEN-LOW-MAF", "review", "markers with MAF < 0.01 kept (review signal only)",
               [{"marker_id": g.markers[j], "maf": float(maf[j])} for j in np.flatnonzero(rare)])
    cols = np.flatnonzero(keep_m)
    ids = [g.ids[i] for i in rows]
    M = g.dosage[np.ix_(rows, cols)]
    miss = g.missing[np.ix_(rows, cols)]
    if parents:
        _mendel(qc, ids, M, miss, parents)
    qc.stats.update({"n_animals_used": len(ids), "n_markers_used": int(cols.size),
                     "n_markers_excluded": int(len(g.markers) - cols.size),
                     "n_animals_excluded": int(len(g.ids) - len(ids)),
                     "mean_maf_used": float(maf[cols].mean()) if cols.size else None})
    qc.raise_if_blocking()
    out = GenotypeData(ids, [g.markers[j] for j in cols], M, miss, g.assembly,
                       [g.counted_allele[j] for j in cols], qc, g.sha256)
    return out, p[cols]


def _mendel(qc: QCReport, ids: list[str], M: np.ndarray, miss: np.ndarray,
            parents: dict[str, tuple[str | None, str | None]]) -> None:
    """Opposing homozygotes between genotyped parent-offspring pairs."""
    pos = {a: i for i, a in enumerate(ids)}
    review, error = [], []
    for a, (s, d) in parents.items():
        if a not in pos:
            continue
        for role, p in (("sire", s), ("dam", d)):
            if p is None or p not in pos:
                continue
            i, j = pos[a], pos[p]
            both = ~miss[i] & ~miss[j]
            if both.sum() == 0:
                continue
            opp = np.sum(both & (np.abs(M[i] - M[j]) >= 1.5))
            rate = float(opp / both.sum())
            item = {"offspring": a, "parent": p, "role": role, "conflict_rate": rate,
                    "n_compared": int(both.sum())}
            if rate > MENDEL_ERROR:
                error.append(item)
            elif rate > MENDEL_REVIEW:
                review.append(item)
    if review:
        qc.add("GEN-MENDEL", "review", f"parent-offspring pairs with {MENDEL_REVIEW:.0%}-"
               f"{MENDEL_ERROR:.0%} opposing homozygotes", review)
    if error:
        qc.add("GEN-MENDEL", "error", f"parent-offspring pairs with > {MENDEL_ERROR:.0%} opposing "
               "homozygotes (pedigree error or sample swap); fix pedigree or samples", error,
               "GENOTYPE_ID_CONFLICT")
