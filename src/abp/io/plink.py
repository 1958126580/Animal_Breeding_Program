"""PLINK 1 binary genotype files (.bed/.bim/.fam), SNP-major mode.

Format (PLINK 1.9 documentation, "PLINK 1 binary biallelic genotype table"):

* ``.bed``: bytes ``0x6c 0x1b`` (magic), then ``0x01`` (SNP-major; the
  obsolete individual-major mode ``0x00`` is rejected).  Each variant uses
  ``ceil(n_samples / 4)`` bytes; within a byte, samples are packed two bits
  each starting at the **lowest-order** bits.  Codes: ``00`` homozygous for
  the first .bim allele (A1), ``01`` missing, ``10`` heterozygous, ``11``
  homozygous for the second allele (A2).
* ``.bim``: chromosome, variant id, genetic position, bp position, A1, A2
  (whitespace separated).
* ``.fam``: family id, individual id, father, mother, sex, phenotype.

ABP reads dosages of **A1** (the allele PLINK's ``--recode A`` counts), so the
counted allele is A1 by construction.  A .bim file does not say which allele
is the reference, and has no assembly; ABP therefore requires the assembly to
be declared and records ref/alt as unknown - it never guesses them.  The
individual id (IID) is the animal id; family ids are ignored but recorded.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ..errors import ABPError
from ..qc.genotype import AMBIGUOUS, GenotypeData
from ..qc.report import QCReport

MAGIC = b"\x6c\x1b"
_LUT_CODES = np.array([[(b >> (2 * k)) & 3 for k in range(4)] for b in range(256)], dtype=np.uint8)
_CODE_TO_DOSAGE = np.array([2.0, np.nan, 1.0, 0.0])   # 00, 01, 10, 11 -> copies of A1
_BYTE_TO_DOSAGES = _CODE_TO_DOSAGE[_LUT_CODES]         # 256 x 4: one byte -> 4 samples
DECODE_BLOCK = 2048                                    # variants decoded per block


def _read_text(path: Path, n_cols: int, what: str) -> list[list[str]]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        raise ABPError("INPUT_NOT_FOUND", f"PLINK {what} file not found: {path}",
                       file=str(path)) from None
    except UnicodeDecodeError:
        raise ABPError("INPUT_ENCODING", f"{path.name} is not valid UTF-8", file=str(path)) from None
    rows = []
    for k, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        f = line.split()
        if len(f) != n_cols:
            raise ABPError("SCHEMA_TYPE", f"{path.name} line {k}: expected {n_cols} fields, found "
                           f"{len(f)}", file=str(path), line=k)
        rows.append(f)
    if not rows:
        raise ABPError("EMPTY_INPUT", f"{path.name} is empty", file=str(path))
    return rows


def decode_bed(raw: bytes, n_samples: int, n_variants: int) -> np.ndarray:
    """Decode SNP-major .bed bytes into an ``n_samples x n_variants`` A1-dosage
    matrix with NaN for missing genotypes."""
    if raw[:2] != MAGIC:
        raise ABPError("SCHEMA_TYPE", "not a PLINK .bed file (magic number mismatch)")
    if len(raw) < 3 or raw[2] != 1:
        raise ABPError("UNSUPPORTED_COMBINATION",
                       "individual-major .bed files are not supported; convert with "
                       "plink --make-bed")
    per = (n_samples + 3) // 4
    expected = 3 + per * n_variants
    if len(raw) != expected:
        raise ABPError("SCHEMA_TYPE", f".bed has {len(raw)} bytes, expected {expected} for "
                       f"{n_samples} samples x {n_variants} variants (mismatched .fam/.bim?)",
                       bytes=len(raw), expected=expected)
    body = np.frombuffer(raw, dtype=np.uint8, offset=3).reshape(n_variants, per)
    out = np.empty((n_samples, n_variants))
    block = min(DECODE_BLOCK, max(n_variants, 1))
    buf = np.empty((block, per, 4))                          # reused: bounded, touched once
    for j0 in range(0, n_variants, block):
        j1 = min(n_variants, j0 + block)
        b = buf[:j1 - j0]
        np.take(_BYTE_TO_DOSAGES, body[j0:j1], axis=0, out=b)
        out[:, j0:j1] = b.reshape(j1 - j0, per * 4)[:, :n_samples].T
    return out


def load_plink(prefix: str | Path, assembly: str) -> GenotypeData:
    """Read ``prefix.bed/.bim/.fam`` into :class:`GenotypeData` (A1 dosages)."""
    prefix = Path(prefix)
    if not assembly:
        raise ABPError("GENOTYPE_ALLELE_MISMATCH", "PLINK input needs data.genotype_assembly")
    fam = _read_text(prefix.with_suffix(".fam"), 6, ".fam")
    bim = _read_text(prefix.with_suffix(".bim"), 6, ".bim")
    bed_path = prefix.with_suffix(".bed")
    try:
        raw = bed_path.read_bytes()
    except FileNotFoundError:
        raise ABPError("INPUT_NOT_FOUND", f"PLINK .bed file not found: {bed_path}",
                       file=str(bed_path)) from None
    qc = QCReport("genotypes")
    ids = [r[1] for r in fam]
    dup = sorted({a for a in ids if ids.count(a) > 1}) if len(set(ids)) != len(ids) else []
    if dup:
        qc.add("GEN-ID", "error", "duplicated individual IDs (IID) in the .fam file",
               [{"id": a} for a in dup], "DUPLICATE_KEY")
        qc.raise_if_blocking()
    markers = [r[1] for r in bim]
    mdup = sorted({m for m in markers if markers.count(m) > 1}) if len(set(markers)) != len(markers) else []
    if mdup:
        qc.add("GEN-MAP", "error", "duplicated variant IDs in the .bim file",
               [{"marker_id": m} for m in mdup], "DUPLICATE_KEY")
        qc.raise_if_blocking()
    a1 = [r[4] for r in bim]
    a2 = [r[5] for r in bim]
    same = [{"marker_id": m, "a1": x, "a2": y} for m, x, y in zip(markers, a1, a2) if x == y]
    if same:
        qc.add("GEN-MAP", "error", "variants with identical A1 and A2", same,
               "GENOTYPE_ALLELE_MISMATCH")
        qc.raise_if_blocking()
    amb = [{"marker_id": m, "alleles": f"{x}/{y}"} for m, x, y in zip(markers, a1, a2)
           if frozenset((x.upper(), y.upper())) in AMBIGUOUS]
    if amb:
        qc.add("GEN-AMBIGUOUS", "review", "strand-ambiguous (A/T, C/G) variants", amb)
    M = decode_bed(raw, len(ids), len(markers))
    missing = np.isnan(M)
    M = np.where(missing, 0.0, M)
    qc.add("GEN-PLINK", "info", "PLINK input: counted allele = A1 of the .bim file; reference/"
           "alternative alleles are not stated by the format and are recorded as unknown")
    qc.stats = {"n_animals_in_file": len(ids), "n_markers_in_file": len(markers),
                "assembly": assembly, "missing_rate": float(missing.mean()),
                "source_format": "PLINK 1 binary (SNP-major)",
                "family_ids": sorted(set(r[0] for r in fam))[:20]}
    sha = {"genotypes": hashlib.sha256(raw).hexdigest(),
           "marker_map": hashlib.sha256(prefix.with_suffix(".bim").read_bytes()).hexdigest(),
           "fam": hashlib.sha256(prefix.with_suffix(".fam").read_bytes()).hexdigest()}
    return GenotypeData(ids, markers, M, missing, assembly, a1, qc, sha)


def write_bed(prefix: str | Path, ids: list[str], markers: list[tuple[str, str, int, str, str]],
              dosage_a1: np.ndarray) -> None:
    """Write a SNP-major PLINK fileset (used by tests and the example converter).

    ``markers``: (variant id, chromosome, bp, A1, A2); ``dosage_a1``: samples x
    variants with 0/1/2 or NaN.
    """
    prefix = Path(prefix)
    n, m = dosage_a1.shape
    code = np.full(dosage_a1.shape, 1, dtype=np.uint8)            # missing
    code[dosage_a1 == 2] = 0
    code[dosage_a1 == 1] = 2
    code[dosage_a1 == 0] = 3
    per = (n + 3) // 4
    padded = np.zeros((m, per * 4), dtype=np.uint8)
    padded[:, :n] = code.T
    q = padded.reshape(m, per, 4)
    body = q[..., 0] | (q[..., 1] << 2) | (q[..., 2] << 4) | (q[..., 3] << 6)
    prefix.with_suffix(".bed").write_bytes(MAGIC + b"\x01" + body.astype(np.uint8).tobytes())
    prefix.with_suffix(".fam").write_text(
        "".join(f"F1 {a} 0 0 0 -9\n" for a in ids), encoding="utf-8")
    prefix.with_suffix(".bim").write_text(
        "".join(f"{c} {v} 0 {bp} {x} {y}\n" for v, c, bp, x, y in markers), encoding="utf-8")
