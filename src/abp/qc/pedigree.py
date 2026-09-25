"""Pedigree import and QC (modules M02/M03).

Checks, in order (all problems of a kind are collected before stopping):

=========================  ========  =========================================
check                      severity  rule
=========================  ========  =========================================
PED-ID                     error     empty IDs / IDs with surrounding spaces
PED-DUP-CONFLICT           error     same animal, different parents/sex/birth
PED-DUP-IDENTICAL          review    exact duplicate rows (kept once, listed)
PED-SELF-PARENT            error     animal is its own sire or dam
PED-SAME-SIRE-DAM          error     one animal recorded as both parents
PED-ROLE-CONFLICT          error     animal used both as sire and as dam
PED-SEX-CONFLICT           error     declared sex contradicts parental role
PED-BIRTH-ORDER            error     parent born on/after offspring
PED-FOUNDER-ADDED          info      parents without own row -> founders
PED-CYCLE                  error     directed cycle in the parent graph
PED-SINGLE-PARENT          info      animals with exactly one known parent
=========================  ========  =========================================

Nothing is silently corrected: identical duplicate rows are the only rows
merged, and they are listed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..core.pedigree import Pedigree
from ..errors import ABPError
from ..io.tables import Table
from .report import QCReport

_DATE = re.compile(r"^(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?$")
SEX_CODES = {"M": "M", "F": "F", "U": "U", "m": "M", "f": "F", "u": "U"}


@dataclass(frozen=True)
class PedigreeColumns:
    id: str = "id"
    sire: str = "sire"
    dam: str = "dam"
    sex: str | None = None
    birth_date: str | None = None


@dataclass
class PedigreeData:
    pedigree: Pedigree
    sex: dict[str, str]
    birth: dict[str, tuple[int, ...]]
    added_founders: list[str]
    qc: QCReport


def parse_date(value: str) -> tuple[int, ...] | None:
    """Parse ``YYYY``, ``YYYY-MM`` or ``YYYY-MM-DD`` into a comparable tuple."""
    m = _DATE.match(value.strip())
    if not m:
        return None
    parts = tuple(int(g) for g in m.groups() if g is not None)
    if len(parts) >= 2 and not 1 <= parts[1] <= 12:
        return None
    if len(parts) == 3 and not 1 <= parts[2] <= 31:
        return None
    return parts


def _born_not_before(parent: tuple[int, ...], child: tuple[int, ...]) -> bool:
    """True if the parent is provably born on/after the child."""
    k = min(len(parent), len(child))
    if parent[:k] > child[:k]:
        return True
    return parent[:k] == child[:k] and k == 3  # same day


def load_pedigree(table: Table, cols: PedigreeColumns, unknown_parent: set[str],
                  missing: set[str], extra_founders: Sequence[str] = ()) -> PedigreeData:
    """Validate a pedigree table and build an ordered :class:`Pedigree`.

    ``extra_founders`` are animals with records but no pedigree row that the
    spec explicitly allows to be added as founders (listed in the QC report).
    """
    qc = QCReport("pedigree")
    ids = table.column(cols.id)
    sires = table.column(cols.sire)
    dams = table.column(cols.dam)
    sexes = table.column(cols.sex) if cols.sex else [""] * table.n_rows
    births = table.column(cols.birth_date) if cols.birth_date else [""] * table.n_rows

    bad_ids = []
    for k, a in enumerate(ids):
        if a == "" or a != a.strip():
            bad_ids.append({"line": table.lines[k], "column": cols.id, "value": a})
        for col, v in ((cols.sire, sires[k]), (cols.dam, dams[k])):
            if v not in unknown_parent and v != v.strip():
                bad_ids.append({"line": table.lines[k], "column": col, "value": v})
    if bad_ids:
        qc.add("PED-ID", "error", "identifiers must be non-empty without surrounding spaces",
               bad_ids, "ID_INVALID")
        qc.raise_if_blocking()

    def norm_parent(v: str) -> str | None:
        return None if v in unknown_parent else v

    rows: dict[str, tuple] = {}
    first_line: dict[str, int] = {}
    order: list[str] = []
    conflicts, identical = [], []
    bad_sex, bad_date = [], []
    for k, a in enumerate(ids):
        sx = sexes[k].strip()
        if sx in missing or sx == "":
            sx = "U"
        elif sx in SEX_CODES:
            sx = SEX_CODES[sx]
        else:
            bad_sex.append({"line": table.lines[k], "animal": a, "value": sexes[k]})
            sx = "U"
        bd_raw = births[k].strip()
        bd = None
        if bd_raw and bd_raw not in missing:
            bd = parse_date(bd_raw)
            if bd is None:
                bad_date.append({"line": table.lines[k], "animal": a, "value": bd_raw})
        rec = (norm_parent(sires[k]), norm_parent(dams[k]), sx, bd)
        if a in rows:
            if rows[a] == rec:
                identical.append({"line": table.lines[k], "animal": a,
                                  "first_line": first_line[a]})
            else:
                conflicts.append({"line": table.lines[k], "animal": a,
                                  "first_line": first_line[a]})
            continue
        rows[a] = rec
        first_line[a] = table.lines[k]
        order.append(a)
    if bad_sex:
        qc.add("PED-SEX-CODE", "error", "sex must be M, F or U (or a missing-value code)",
               bad_sex, "SCHEMA_TYPE")
    if bad_date:
        qc.add("PED-DATE", "error", "birth dates must be YYYY, YYYY-MM or YYYY-MM-DD",
               bad_date, "SCHEMA_TYPE")
    if conflicts:
        qc.add("PED-DUP-CONFLICT", "error", "animal listed more than once with different data",
               conflicts, "PEDIGREE_CONFLICTING_DUPLICATE")
    if identical:
        qc.add("PED-DUP-IDENTICAL", "review", "identical duplicate rows were kept once",
               identical)

    self_par, same_par = [], []
    for a in order:
        s, d, _, _ = rows[a]
        if a in (s, d):
            self_par.append({"animal": a, "line": first_line[a]})
        if s is not None and s == d:
            same_par.append({"animal": a, "parent": s, "line": first_line[a]})
    if self_par:
        qc.add("PED-SELF-PARENT", "error", "animal recorded as its own parent", self_par,
               "PEDIGREE_SELF_PARENT")
    if same_par:
        qc.add("PED-SAME-SIRE-DAM", "error", "the same animal is recorded as sire and dam",
               same_par, "PEDIGREE_SEX_CONFLICT")

    as_sire = {rows[a][0] for a in order} - {None}
    as_dam = {rows[a][1] for a in order} - {None}
    both = sorted(as_sire & as_dam)
    if both:
        qc.add("PED-ROLE-CONFLICT", "error", "animal used both as sire and as dam",
               [{"animal": a} for a in both], "PEDIGREE_SEX_CONFLICT")
    sex_conf = [{"animal": a, "declared_sex": rows[a][2], "role": "sire"}
                for a in sorted(as_sire) if a in rows and rows[a][2] == "F"]
    sex_conf += [{"animal": a, "declared_sex": rows[a][2], "role": "dam"}
                 for a in sorted(as_dam) if a in rows and rows[a][2] == "M"]
    if sex_conf:
        qc.add("PED-SEX-CONFLICT", "error", "declared sex contradicts parental role", sex_conf,
               "PEDIGREE_SEX_CONFLICT")

    late = []
    for a in order:
        s, d, _, bd = rows[a]
        if bd is None:
            continue
        for p in (s, d):
            if p is not None and p in rows and rows[p][3] is not None \
                    and _born_not_before(rows[p][3], bd):
                late.append({"animal": a, "parent": p,
                             "animal_birth": "-".join(map(str, bd)),
                             "parent_birth": "-".join(map(str, rows[p][3]))})
    if late:
        qc.add("PED-BIRTH-ORDER", "error", "parent born on or after its offspring", late,
               "PEDIGREE_BIRTH_ORDER")
    qc.raise_if_blocking()

    missing_parents = []
    seen = set(rows)
    for a in order:
        for p in rows[a][:2]:
            if p is not None and p not in seen:
                seen.add(p)
                missing_parents.append(p)
    if missing_parents:
        qc.add("PED-FOUNDER-ADDED", "info",
               "parents without their own pedigree row were added as founders "
               "(unknown parents, base population)",
               [{"animal": p} for p in missing_parents])
    record_only = [a for a in extra_founders if a not in seen]
    if record_only:
        qc.add("PED-RECORD-ANIMAL-ADDED", "info",
               "animals with records but no pedigree row were added as founders "
               "(qc.unknown_animals = add_as_founder)", [{"animal": a} for a in record_only])
        missing_parents = missing_parents + record_only
    all_ids = missing_parents + order
    sire_l = [None] * len(missing_parents) + [rows[a][0] for a in order]
    dam_l = [None] * len(missing_parents) + [rows[a][1] for a in order]
    try:
        ped = Pedigree.from_parent_ids(all_ids, sire_l, dam_l)
    except ABPError as exc:
        qc.add("PED-CYCLE", "error", exc.message,
               [{"animal": a} for a in exc.details.get("animals", [])], exc.spec.name)
        qc.raise_if_blocking()
        raise
    single = [a for a in order if (rows[a][0] is None) != (rows[a][1] is None)]
    if single:
        qc.add("PED-SINGLE-PARENT", "info", "animals with exactly one known parent",
               [{"animal": a} for a in single])
    F = ped.inbreeding()
    known_both = (ped.sire >= 0) & (ped.dam >= 0)
    qc.stats = {
        "n_rows": table.n_rows,
        "n_animals": ped.n,
        "n_founders_added": len(missing_parents),
        "n_both_parents_known": int(known_both.sum()),
        "n_one_parent_known": len(single),
        "n_no_parent_known": int(((ped.sire < 0) & (ped.dam < 0)).sum()),
        "n_sires": len(as_sire),
        "n_dams": len(as_dam),
        "max_generation": int(ped.generation.max()),
        "n_inbred": int((F > 0).sum()),
        "mean_inbreeding": float(F.mean()),
        "max_inbreeding": float(F.max()),
        "inbreeding_kernel": ped.inbreeding_kernel,
    }
    sex = {a: rows[a][2] for a in order}
    sex.update({p: ("M" if p in as_sire else "F" if p in as_dam else "U")
                for p in missing_parents})
    birth = {a: rows[a][3] for a in order if rows[a][3] is not None}
    return PedigreeData(ped, sex, birth, missing_parents, qc)


def mean_inbreeding_by_generation(ped: Pedigree) -> list[dict]:
    """Descriptive inbreeding trajectory by pedigree generation depth."""
    F = ped.inbreeding()
    out = []
    for g in np.unique(ped.generation):
        m = ped.generation == g
        out.append({"generation": int(g), "n": int(m.sum()), "mean_F": float(F[m].mean()),
                    "max_F": float(F[m].max())})
    return out
