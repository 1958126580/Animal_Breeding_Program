"""Delimited-text tables with provenance (hash, line numbers) and strict parsing.

Contract
--------
* Files are UTF-8 (a UTF-8 byte-order mark, as written by Excel, is
  accepted).  Any other encoding is rejected rather than guessed.
* The first line is the header; header names must be unique.
* Every data row must have exactly as many fields as the header.  Blank
  lines are skipped and counted.
* Cells are kept as raw strings; typed parsing happens per column so that
  errors can name the file, line, column and offending value.
* The SHA-256 of the raw bytes is recorded for the run manifest.
"""

from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ..errors import ABPError


@dataclass(frozen=True)
class Table:
    path: Path
    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    lines: tuple[int, ...]  #: 1-based file line number of every data row
    sha256: str
    n_blank_lines: int

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    def has(self, column: str) -> bool:
        return column in self.header

    def column(self, name: str) -> list[str]:
        """Raw string values of a column (``SCHEMA_MISSING_COLUMN`` if absent)."""
        try:
            k = self.header.index(name)
        except ValueError:
            raise ABPError("SCHEMA_MISSING_COLUMN",
                           f"column {name!r} not found in {self.path.name}; available: "
                           + ", ".join(self.header), file=str(self.path), column=name,
                           available=list(self.header)) from None
        return [r[k] for r in self.rows]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_table(path: str | Path, delimiter: str = ",") -> Table:
    """Read a delimited UTF-8 table, validating its shape."""
    path = Path(path)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        raise ABPError("INPUT_NOT_FOUND", f"file not found: {path}", file=str(path)) from None
    except OSError as exc:
        raise ABPError("INPUT_NOT_FOUND", f"cannot read {path}: {exc.strerror}",
                       file=str(path)) from None
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ABPError("INPUT_ENCODING",
                       f"{path.name} is not valid UTF-8 (byte offset {exc.start})",
                       file=str(path), byte_offset=exc.start) from None
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter)
    header: tuple[str, ...] | None = None
    rows: list[tuple[str, ...]] = []
    lines: list[int] = []
    blanks = 0
    for rec in reader:
        line = reader.line_num
        if not rec or all(c.strip() == "" for c in rec):
            blanks += 1
            continue
        if header is None:
            header = tuple(c.strip() for c in rec)
            dup = sorted({h for h in header if header.count(h) > 1})
            if dup:
                raise ABPError("DUPLICATE_KEY", f"duplicate column names in {path.name}: {dup}",
                               file=str(path), columns=dup)
            if any(h == "" for h in header):
                raise ABPError("SCHEMA_MISSING_COLUMN", f"empty column name in header of {path.name}",
                               file=str(path))
            continue
        if len(rec) != len(header):
            raise ABPError("SCHEMA_TYPE",
                           f"{path.name} line {line}: expected {len(header)} fields, found {len(rec)}",
                           file=str(path), line=line, expected=len(header), found=len(rec))
        rows.append(tuple(rec))
        lines.append(line)
    if header is None:
        raise ABPError("EMPTY_INPUT", f"{path.name} is empty (no header)", file=str(path))
    if not rows:
        raise ABPError("EMPTY_INPUT", f"{path.name} has a header but no data rows", file=str(path))
    return Table(path, header, tuple(rows), tuple(lines),
                 hashlib.sha256(raw).hexdigest(), blanks)


def parse_float_column(table: Table, column: str, missing: Iterable[str]) -> np.ndarray:
    """Parse a numeric column; missing-value tokens become NaN.

    Raises ``SCHEMA_TYPE`` naming the first unparsable cell (file, line,
    column, value).  ``inf``/``nan`` spelled in the file are rejected: use a
    declared missing-value token instead.
    """
    missing = set(missing)
    values = table.column(column)
    out = np.empty(len(values), dtype=np.float64)
    for k, v in enumerate(values):
        s = v.strip()
        if s in missing or v in missing:
            out[k] = np.nan
            continue
        try:
            x = float(s)
        except ValueError:
            x = None
        if x is None or not np.isfinite(x):
            raise ABPError("SCHEMA_TYPE",
                           f"{table.path.name} line {table.lines[k]}, column {column!r}: "
                           f"{v!r} is not a finite number or a declared missing-value code",
                           file=str(table.path), line=table.lines[k], column=column, value=v)
        out[k] = x
    return out


def check_identifier(value: str, table: Table, row: int, column: str) -> None:
    """Identifiers must be non-empty and free of leading/trailing whitespace."""
    if value == "" or value != value.strip():
        raise ABPError("ID_INVALID",
                       f"{table.path.name} line {table.lines[row]}, column {column!r}: identifier "
                       f"{value!r} is empty or has leading/trailing whitespace",
                       file=str(table.path), line=table.lines[row], column=column, value=value)


def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence]) -> None:
    """Write a UTF-8 CSV (LF line endings) with full float precision."""
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow([_fmt(v) for v in r])


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return ""
        return repr(float(v))
    return str(v)
