"""Phenotype import and QC (modules M01/M02).

A *record* is one row of the phenotype file: one animal, one measurement
occasion, one or more trait values plus the fixed/random classification
variables.  Missing-value tokens mean "not measured"; ABP does not impute
phenotypes.

Checks
------
=====================  ==========================  =============================
check                  severity                    rule
=====================  ==========================  =============================
PHE-ID                 error                       invalid animal/record IDs
PHE-RECORD-DUP         error                       duplicated record_id
PHE-UNKNOWN-ANIMAL     error (or info if allowed)  animal not in the pedigree
PHE-RANGE              error or quarantine         outside [min, max] of trait
PHE-MISSING-CLASS      error or quarantine         used record lacks a fixed or
                                                   random classification value
PHE-REPEATED           error                       repeated records without a
                                                   permanent-environment term
PHE-OUTLIER            review                      |z| > qc.outlier_sd (kept)
PHE-SINGLETON-CLASS    review                      fixed-factor level with one
                                                   record (no within-level contrast)
=====================  ==========================  =============================
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from ..io.tables import Table, parse_float_column
from .report import QCReport


@dataclass
class RecordSet:
    """Validated phenotype records (all arrays aligned with ``animal``)."""

    record_id: list[str]
    animal: list[str]
    line: list[int]
    traits: dict[str, np.ndarray]           # NaN = not measured / quarantined
    factors: dict[str, list[str | None]]    # None = missing
    covariates: dict[str, np.ndarray]       # NaN = missing
    units: dict[str, str] = field(default_factory=dict)
    dates: list[str] | None = None          # raw record dates (data.phenotype_columns.date)

    @property
    def n(self) -> int:
        return len(self.animal)

    def used_mask(self, traits: list[str]) -> np.ndarray:
        """Records with at least one of ``traits`` measured."""
        m = np.zeros(self.n, dtype=bool)
        for t in traits:
            m |= ~np.isnan(self.traits[t])
        return m


def load_phenotypes(table: Table, spec: dict, pedigree_ids: set[str] | None) -> tuple[RecordSet, QCReport, list[str]]:
    """Validate the phenotype table against the spec.

    Returns ``(records, qc, animals_to_add)`` where ``animals_to_add`` lists
    record animals missing from the pedigree that the spec allows to be
    added as founders (``qc.unknown_animals = "add_as_founder"``).
    """
    qc = QCReport("phenotypes")
    data, model, qcs = spec["data"], spec["model"], spec["qc"]
    missing = set(data["missing_values"])
    id_col = data["phenotype_columns"]["id"]
    animals = table.column(id_col)
    bad = [{"line": table.lines[k], "column": id_col, "value": a}
           for k, a in enumerate(animals) if a == "" or a != a.strip() or a in missing]
    rec_col = data["phenotype_columns"]["record_id"]
    if rec_col:
        rids = table.column(rec_col)
        bad += [{"line": table.lines[k], "column": rec_col, "value": r}
                for k, r in enumerate(rids) if r == "" or r != r.strip()]
    else:
        rids = [f"line{ln}" for ln in table.lines]
    if bad:
        qc.add("PHE-ID", "error", "animal/record identifiers must be non-empty without "
               "surrounding spaces", bad, "ID_INVALID")
        qc.raise_if_blocking()
    dups = [r for r, c in Counter(rids).items() if c > 1]
    if dups:
        qc.add("PHE-RECORD-DUP", "error", "record_id values must be unique",
               [{"record_id": r} for r in dups], "DUPLICATE_KEY")
        qc.raise_if_blocking()

    to_add: list[str] = []
    if pedigree_ids is not None:
        unknown = sorted({a for a in animals if a not in pedigree_ids})
        if unknown:
            if qcs["unknown_animals"] == "add_as_founder":
                qc.add("PHE-UNKNOWN-ANIMAL", "info",
                       "record animals absent from the pedigree are added as founders",
                       [{"animal": a} for a in unknown])
                to_add = unknown
            else:
                qc.add("PHE-UNKNOWN-ANIMAL", "error", "record animals absent from the pedigree",
                       [{"animal": a} for a in unknown], "PHENOTYPE_UNKNOWN_ANIMAL")
                qc.raise_if_blocking()

    trait_decl = {t["name"]: t for t in spec["traits"]}
    traits: dict[str, np.ndarray] = {}
    units: dict[str, str] = {}
    for tname in model["traits"]:
        decl = trait_decl[tname]
        vals = parse_float_column(table, decl["column"], missing)
        lo = -np.inf if decl["min"] is None else decl["min"]
        hi = np.inf if decl["max"] is None else decl["max"]
        out = (~np.isnan(vals)) & ((vals < lo) | (vals > hi))
        if out.any():
            items = [{"line": table.lines[k], "record_id": rids[k], "animal": animals[k],
                      "trait": tname, "value": float(vals[k]), "valid_min": decl["min"],
                      "valid_max": decl["max"]} for k in np.flatnonzero(out)]
            if qcs["out_of_range"] == "quarantine":
                qc.add("PHE-RANGE", "quarantine",
                       f"{tname}: values outside the declared range were excluded", items)
                qc.excluded.extend({**it, "reason": "PHE-RANGE"} for it in items)
                vals = vals.copy()
                vals[out] = np.nan
            else:
                qc.add("PHE-RANGE", "error", f"{tname}: values outside the declared range",
                       items, "PHENOTYPE_OUT_OF_RANGE")
        traits[tname] = vals
        units[tname] = decl["unit"]
    qc.raise_if_blocking()

    factors: dict[str, list[str | None]] = {}
    covariates: dict[str, np.ndarray] = {}
    class_cols: list[str] = []
    for f in model["fixed"]:
        if f["type"] == "factor":
            factors[f["column"]] = [None if (v in missing or v.strip() in missing) else v
                                    for v in table.column(f["column"])]
            class_cols.append(f["column"])
        else:
            covariates[f["column"]] = parse_float_column(table, f["column"], missing)
            class_cols.append(f["column"])
    for r in model["random"]:
        if r["kind"] == "iid" and r["column"] not in factors:
            factors[r["column"]] = [None if (v in missing or v.strip() in missing) else v
                                    for v in table.column(r["column"])]
            class_cols.append(r["column"])
    date_col = data["phenotype_columns"].get("date")
    dates = [v.strip() for v in table.column(date_col)] if date_col else None
    rs = RecordSet(list(rids), list(animals), list(table.lines), traits, factors, covariates,
                   units, dates)

    # a classification value is required only for records measured on a trait
    # the term applies to (fixed terms may be trait-specific in multi-trait models)
    applies = {f["column"]: f.get("traits") or model["traits"] for f in model["fixed"]}
    for r in model["random"]:
        if r["kind"] == "iid":
            applies.setdefault(r["column"], model["traits"])
    miss_items = []
    for col in class_cols:
        used_c = rs.used_mask(list(applies[col]))
        col_missing = (np.isnan(covariates[col]) if col in covariates
                       else np.array([v is None for v in factors[col]]))
        for k in np.flatnonzero(used_c & col_missing):
            miss_items.append({"line": rs.line[k], "record_id": rs.record_id[k], "column": col})
    if miss_items:
        if qcs["missing_fixed"] == "exclude":
            qc.add("PHE-MISSING-CLASS", "quarantine",
                   "records lacking a classification value were excluded", miss_items)
            qc.excluded.extend({**it, "reason": "PHE-MISSING-CLASS"} for it in miss_items)
            pos = {r: k for k, r in enumerate(rs.record_id)}
            for it in miss_items:
                for t in applies[it["column"]]:
                    traits[t][pos[it["record_id"]]] = np.nan
        else:
            qc.add("PHE-MISSING-CLASS", "error",
                   "records with a trait value lack a fixed/random classification value",
                   miss_items, "SCHEMA_TYPE")
    qc.raise_if_blocking()

    used = rs.used_mask(model["traits"])
    pe_on_animal = any(r["kind"] == "iid" and r["column"] == id_col for r in model["random"])
    for tname in model["traits"]:
        has = ~np.isnan(traits[tname])
        counts = Counter(a for a, h in zip(animals, has) if h)
        rep = sorted(a for a, c in counts.items() if c > 1)
        if rep and not pe_on_animal:
            qc.add("PHE-REPEATED", "error",
                   f"{tname}: animals have repeated records but the model has no iid "
                   f"(permanent-environment) term on column {id_col!r}; repeated records "
                   "must not be treated as independent",
                   [{"animal": a, "n_records": counts[a]} for a in rep], "UNSUPPORTED_COMBINATION")
        vals = traits[tname][has]
        if vals.size >= 3 and np.std(vals) > 0:
            z = (traits[tname] - vals.mean()) / vals.std(ddof=1)
            flag = has & (np.abs(z) > qcs["outlier_sd"])
            if flag.any():
                qc.add("PHE-OUTLIER", "review",
                       f"{tname}: values beyond {qcs['outlier_sd']} SD (kept; please verify)",
                       [{"line": rs.line[k], "record_id": rs.record_id[k], "animal": animals[k],
                         "value": float(traits[tname][k]), "z": float(z[k])}
                        for k in np.flatnonzero(flag)])
    qc.raise_if_blocking()
    for f in model["fixed"]:
        if f["type"] != "factor":
            continue
        cnt = Counter(v for v, u in zip(factors[f["column"]], used) if u)
        single = sorted(k for k, c in cnt.items() if c == 1)
        if single:
            qc.add("PHE-SINGLETON-CLASS", "review",
                   f"levels of {f['column']!r} with a single record carry no within-level "
                   "contrast for genetic comparison", [{"level": s} for s in single])
    if not used.any():
        qc.add("PHE-NONE", "error", "no record has a value for the analysed trait(s)", [],
               "NO_USABLE_RECORDS")
        qc.raise_if_blocking()
    qc.stats = {
        "n_rows": table.n_rows,
        "n_records_used": int(used.sum()),
        "n_animals_with_records": len({a for a, u in zip(animals, used) if u}),
        "traits": {t: _describe(traits[t]) for t in model["traits"]},
        "n_excluded": len(qc.excluded),
    }
    return rs, qc, to_add


def _describe(v: np.ndarray) -> dict:
    x = v[~np.isnan(v)]
    if x.size == 0:
        return {"n": 0}
    return {"n": int(x.size), "mean": float(x.mean()),
            "sd": float(x.std(ddof=1)) if x.size > 1 else None,
            "min": float(x.min()), "max": float(x.max())}
