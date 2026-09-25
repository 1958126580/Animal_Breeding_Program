"""Analysis specification (TOML): declarative schema, validation, JSON Schema.

The analysis spec is the single, versioned, human-readable description of a
run: what is estimated (the *estimand*), for which population and
information time, from which files, with which model and settings.  It is
validated **before any computation**:

* unknown keys are errors (typos never silently fall back to defaults);
* required keys must be present; types and allowed values are checked;
* defaults are filled in explicitly so that the run manifest records the
  complete, effective configuration.

The Python table :data:`SCHEMA` is the single source of truth; the JSON
Schema in ``contracts/analysis_spec.schema.json`` is generated from it by
``python -m abp.core.spec`` and a test checks that the file is up to date.
"""

from __future__ import annotations

import json
import math
import re
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..errors import ABPError

TASKS = ("phenotype_prediction", "additive_ebv", "total_genetic_value", "mating_utility")
IMPLEMENTED_TASKS = ("additive_ebv",)
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ---------------------------------------------------------------- schema DSL
@dataclass(frozen=True)
class Field:
    type: str  # "str" | "bool" | "int" | "float" | "str_list" | "number_or_matrix_map" | "float_map"
    required: bool = False
    default: Any = None
    choices: tuple | None = None
    check: Callable[[Any], str | None] | None = None  # returns an error text or None
    doc: str = ""


@dataclass(frozen=True)
class Section:
    fields: dict[str, "Field | Section | TableArray"]
    required: bool = False
    doc: str = ""


@dataclass(frozen=True)
class TableArray:
    item: Section
    required: bool = False
    min_items: int = 0
    doc: str = ""


def _positive(x):
    return None if x > 0 else "must be > 0"


def _nonneg(x):
    return None if x >= 0 else "must be >= 0"


def _unit_interval(x):
    return None if 0 <= x <= 1 else "must lie in [0, 1]"


def _maf_range(x):
    return None if 0 <= x < 0.5 else "must lie in [0, 0.5)"


def _name(x):
    return None if _NAME.match(x) else "must start with a letter and contain only letters, digits, _"


def _date(x):
    return None if _DATE.match(x) else "must be an ISO date YYYY-MM-DD"


SCHEMA = Section({
    "schema_version": Field("str", required=True, choices=("1",),
                            doc="Version of this spec format."),
    "project": Section({
        "name": Field("str", required=True, doc="Short project name."),
        "species": Field("str", required=True, doc="Species (e.g. sheep, cattle, pig, chicken)."),
        "synthetic_data": Field("bool", required=True,
                                doc="True if the data are synthetic (must be declared)."),
        "description": Field("str", default=""),
    }, required=True),
    "analysis": Section({
        "task": Field("str", required=True, choices=TASKS,
                      doc="Estimand class; only additive_ebv is implemented in 0.1."),
        "target_population": Field("str", required=True,
                                   doc="Population to which results apply."),
        "information_cutoff": Field("str", required=True, check=_date,
                                    doc="Latest date of information used for training."),
        "genetic_base": Field("str", required=True,
                              doc="Declared genetic base (e.g. pedigree founders)."),
        "applicability": Field("str", default="", doc="Scope/limits of use of the results."),
    }, required=True),
    "data": Section({
        "pedigree": Field("str", doc="Pedigree CSV (relative to the spec file)."),
        "phenotypes": Field("str", required=True, doc="Phenotype CSV."),
        "genotypes": Field("str", doc="Genotype dosage CSV (ABP dosage-matrix format)."),
        "marker_map": Field("str", doc="Marker map CSV (required with genotypes)."),
        "allele_frequencies": Field("str", doc="Frequency CSV for genomic.frequency_source='file'."),
        "delimiter": Field("str", default=",", check=lambda x: None if len(x) == 1 else "one character"),
        "missing_values": Field("str_list", default=["", "NA", "."]),
        "unknown_parent_values": Field("str_list", default=["0", "", "NA", "."]),
        "pedigree_columns": Section({
            "id": Field("str", default="id"),
            "sire": Field("str", default="sire"),
            "dam": Field("str", default="dam"),
            "sex": Field("str"),
            "birth_date": Field("str"),
        }),
        "phenotype_columns": Section({
            "id": Field("str", default="id"),
            "record_id": Field("str"),
        }),
    }, required=True),
    "traits": TableArray(Section({
        "name": Field("str", required=True, check=_name),
        "column": Field("str", doc="Column in the phenotype file (default: name)."),
        "unit": Field("str", required=True, doc="Measurement unit, e.g. kg."),
        "min": Field("float", doc="Smallest valid value (inclusive)."),
        "max": Field("float", doc="Largest valid value (inclusive)."),
        "description": Field("str", default=""),
    }), required=True, min_items=1),
    "model": Section({
        "traits": Field("str_list", required=True, doc="Traits analysed jointly (1 = single-trait)."),
        "intercept": Field("bool", default=True),
        "fixed": TableArray(Section({
            "column": Field("str", required=True),
            "type": Field("str", required=True, choices=("factor", "covariate")),
        })),
        "random": TableArray(Section({
            "name": Field("str", required=True, check=_name),
            "kind": Field("str", required=True, choices=("additive", "iid")),
            "relationship": Field("str", choices=("pedigree", "genomic", "single_step"),
                                  doc="Covariance structure of an additive term."),
            "column": Field("str", doc="Grouping column of an iid term (default: animal id)."),
        }), min_items=1),
    }, required=True),
    "variances": Section({
        "mode": Field("str", required=True, choices=("known", "reml")),
        "values": Field("number_or_matrix_map",
                        doc="Variance (single-trait) or covariance matrix (multi-trait) per "
                            "random term and 'residual'. Required for mode='known'."),
    }, required=True),
    "reml": Section({
        "algorithm": Field("str", default="ai", choices=("ai", "em")),
        "max_iter": Field("int", default=200, check=_positive),
        "tol": Field("float", default=1e-8, check=_positive),
        "start": Field("float_map", doc="Starting variances (default: data-based heuristic)."),
    }),
    "solver": Section({
        "method": Field("str", default="auto", choices=("auto", "dense", "sparse_direct", "pcg")),
        "tol": Field("float", default=1e-10, check=_positive),
        "max_iter": Field("int", default=10000, check=_positive),
        "pev": Field("str", default="exact", choices=("exact", "none")),
    }),
    "qc": Section({
        "unknown_animals": Field("str", default="error", choices=("error", "add_as_founder")),
        "out_of_range": Field("str", default="error", choices=("error", "quarantine")),
        "missing_fixed": Field("str", default="error", choices=("error", "exclude")),
        "outlier_sd": Field("float", default=4.0, check=_positive,
                            doc="Review signal only; values are never removed by this rule."),
        "ungenotyped_records": Field("str", default="error", choices=("error", "exclude"),
                                     doc="GBLUP only: records of animals without genotypes."),
    }),
    "genomic": Section({
        "ploidy": Field("int", default=2, choices=(2,)),
        "frequency_source": Field("str", default="genotyped_all",
                                  choices=("genotyped_all", "training_genotyped", "file", "fixed_0.5")),
        "min_call_rate_marker": Field("float", default=0.90, check=_unit_interval),
        "min_call_rate_animal": Field("float", default=0.90, check=_unit_interval),
        "min_maf": Field("float", default=0.0, check=_maf_range),
        "singular_policy": Field("str", default="error", choices=("error", "blend", "ridge")),
        "blend_alpha": Field("float", default=0.05, check=_unit_interval),
        "ridge": Field("float", default=0.01, check=_positive),
        "tuning": Field("str", default="none", choices=("none", "match_a22")),
    }),
    "index": Section({
        "weights": Field("float_map", required=True, doc="Economic weight per unit of each trait."),
        "weight_units": Field("str", required=True, doc="e.g. 'EUR per kg'."),
        "synthetic_weights": Field("bool", required=True,
                                   doc="True if the weights are placeholders, not industry values."),
    }),
    "output": Section({
        "top_n": Field("int", default=20, check=_positive),
    }),
    "resources": Section({
        "max_memory_gb": Field("float", default=4.0, check=_positive),
        "min_free_disk_mb": Field("float", default=200.0, check=_nonneg),
    }),
    "backend": Section({
        "device": Field("str", default="cpu", choices=("cpu", "cuda")),
        "on_unavailable": Field("str", default="error", choices=("error", "fallback_cpu")),
    }),
}, required=True)


# ---------------------------------------------------------------- validation
def _err(path: str, msg: str, **kw) -> ABPError:
    return ABPError("SPEC_INVALID", f"{path}: {msg}", key=path, **kw)


def _check_field(f: Field, value: Any, path: str) -> Any:
    t = f.type
    if t == "str":
        if not isinstance(value, str):
            raise _err(path, f"expected a string, got {type(value).__name__}")
    elif t == "bool":
        if not isinstance(value, bool):
            raise _err(path, "expected true or false")
    elif t == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise _err(path, "expected an integer")
    elif t == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _err(path, "expected a number")
        value = float(value)
        if not math.isfinite(value):
            raise _err(path, "must be finite")
    elif t == "str_list":
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise _err(path, "expected a list of strings")
    elif t == "float_map":
        if not isinstance(value, dict) or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                for v in value.values()):
            raise _err(path, "expected a table of finite numbers")
        value = {k: float(v) for k, v in value.items()}
    elif t == "number_or_matrix_map":
        if not isinstance(value, dict):
            raise _err(path, "expected a table")
        out = {}
        for k, v in value.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out[k] = float(v)
            elif (isinstance(v, list) and v and all(isinstance(r, list) and len(r) == len(v)
                                                     for r in v)
                  and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                          for r in v for x in r)):
                out[k] = [[float(x) for x in r] for r in v]
            else:
                raise _err(f"{path}.{k}", "expected a number or a square matrix of numbers")
        value = out
    else:  # pragma: no cover - schema bug
        raise AssertionError(t)
    if f.choices is not None and value not in f.choices:
        raise _err(path, f"must be one of {list(f.choices)}, got {value!r}")
    if f.check is not None:
        msg = f.check(value)
        if msg:
            raise _err(path, msg)
    return value


def _validate_section(sec: Section, data: Any, path: str) -> dict:
    if not isinstance(data, dict):
        raise _err(path or "<root>", "expected a table")
    unknown = sorted(set(data) - set(sec.fields))
    if unknown:
        raise _err(f"{path}.{unknown[0]}" if path else unknown[0],
                   f"unknown key (allowed: {sorted(sec.fields)})")
    out: dict = {}
    for key, f in sec.fields.items():
        p = f"{path}.{key}" if path else key
        present = key in data
        if isinstance(f, Section):
            if present:
                out[key] = _validate_section(f, data[key], p)
            elif f.required:
                raise _err(p, "required section is missing")
            else:
                out[key] = None if _has_required(f) else _validate_section(f, {}, p)
        elif isinstance(f, TableArray):
            if present:
                items = data[key]
                if not isinstance(items, list):
                    raise _err(p, "expected an array of tables ([[...]])")
                if len(items) < f.min_items:
                    raise _err(p, f"needs at least {f.min_items} entr(y/ies)")
                out[key] = [_validate_section(f.item, it, f"{p}[{i}]") for i, it in enumerate(items)]
            elif f.required:
                raise _err(p, "required array is missing")
            else:
                out[key] = []
        else:
            if present:
                out[key] = _check_field(f, data[key], p)
            elif f.required:
                raise _err(p, "required key is missing")
            else:
                out[key] = f.default.copy() if isinstance(f.default, list) else f.default
    return out


def _has_required(sec: Section) -> bool:
    return any(getattr(f, "required", False) for f in sec.fields.values())


@dataclass
class AnalysisSpec:
    """A validated analysis spec plus its location and raw bytes hash."""

    data: dict
    path: Path
    sha256: str
    extra: dict = field(default_factory=dict)

    @property
    def base_dir(self) -> Path:
        return self.path.parent

    def resolve(self, rel: str | None) -> Path | None:
        """Resolve a data path relative to the spec file's folder."""
        if rel is None:
            return None
        p = Path(rel)
        return p if p.is_absolute() else (self.base_dir / p)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]


def validate_spec_dict(raw: dict) -> dict:
    """Validate a parsed spec and apply cross-field rules; returns filled dict."""
    d = _validate_section(SCHEMA, raw, "")
    trait_names = [t["name"] for t in d["traits"]]
    if len(set(trait_names)) != len(trait_names):
        raise _err("traits", "trait names must be unique")
    for t in d["traits"]:
        if t["column"] is None:
            t["column"] = t["name"]
        if t["min"] is not None and t["max"] is not None and t["min"] > t["max"]:
            raise _err(f"traits.{t['name']}", "min must not exceed max")
    m = d["model"]
    if not m["traits"]:
        raise _err("model.traits", "at least one trait is required")
    for tr in m["traits"]:
        if tr not in trait_names:
            raise _err("model.traits", f"trait {tr!r} is not declared in [[traits]]")
    if len(set(m["traits"])) != len(m["traits"]):
        raise _err("model.traits", "duplicate trait")
    names = [r["name"] for r in m["random"]]
    if len(set(names)) != len(names) or "residual" in names:
        raise _err("model.random", "random term names must be unique and not 'residual'")
    additive = [r for r in m["random"] if r["kind"] == "additive"]
    if len(additive) != 1:
        raise _err("model.random", "exactly one additive genetic term is required in 0.1")
    for r in m["random"]:
        if r["kind"] == "additive":
            if r["relationship"] is None:
                raise _err(f"model.random.{r['name']}", "additive terms need 'relationship'")
            if r["column"] is not None:
                raise _err(f"model.random.{r['name']}",
                           "additive terms use the animal id column; remove 'column'")
        else:
            if r["relationship"] is not None:
                raise _err(f"model.random.{r['name']}", "'relationship' applies to additive terms only")
            if r["column"] is None:
                r["column"] = d["data"]["phenotype_columns"]["id"]
    rel = additive[0]["relationship"]
    if rel in ("pedigree", "single_step") and d["data"]["pedigree"] is None:
        raise _err("data.pedigree", f"relationship {rel!r} needs a pedigree file")
    if rel in ("genomic", "single_step"):
        if d["data"]["genotypes"] is None or d["data"]["marker_map"] is None:
            raise _err("data.genotypes", f"relationship {rel!r} needs genotypes and marker_map")
    if d["genomic"]["frequency_source"] == "file" and d["data"]["allele_frequencies"] is None:
        raise _err("data.allele_frequencies", "required when genomic.frequency_source = 'file'")
    v = d["variances"]
    expected = set(names) | {"residual"}
    t = len(m["traits"])
    if v["mode"] == "known":
        if v["values"] is None:
            raise _err("variances.values", "required when mode = 'known'")
        if set(v["values"]) != expected:
            raise _err("variances.values",
                       f"must give exactly {sorted(expected)}, got {sorted(v['values'])}")
        for k, val in v["values"].items():
            if t == 1 and not isinstance(val, float):
                raise _err(f"variances.values.{k}", "single-trait models need a number")
            if t > 1 and not (isinstance(val, list) and len(val) == t):
                raise _err(f"variances.values.{k}", f"multi-trait models need a {t}x{t} matrix")
    else:
        if t > 1:
            raise _err("variances.mode", "multi-trait REML is not implemented in 0.1; "
                                         "use mode = 'known'")
        if v["values"] is not None:
            raise _err("variances.values", "not used with mode = 'reml' (use reml.start)")
        start = d["reml"]["start"]
        if start is not None and set(start) != expected:
            raise _err("reml.start", f"must give exactly {sorted(expected)}")
    if d["analysis"]["task"] not in IMPLEMENTED_TASKS:
        raise ABPError("UNSUPPORTED_COMBINATION",
                       f"analysis.task = {d['analysis']['task']!r} is not implemented in 0.1 "
                       f"(implemented: {list(IMPLEMENTED_TASKS)})")
    idx = raw.get("index")
    if idx is None:
        d["index"] = None
    else:
        unknown = set(d["index"]["weights"]) - set(m["traits"])
        if unknown:
            raise _err("index.weights", f"weights for traits not in the model: {sorted(unknown)}")
    return d


def load_spec(path: str | Path) -> AnalysisSpec:
    """Read and validate a TOML analysis spec."""
    import hashlib
    path = Path(path)
    try:
        raw_bytes = path.read_bytes()
    except FileNotFoundError:
        raise ABPError("INPUT_NOT_FOUND", f"spec file not found: {path}", file=str(path)) from None
    try:
        raw = tomllib.loads(raw_bytes.decode("utf-8-sig"))
    except UnicodeDecodeError:
        raise ABPError("INPUT_ENCODING", f"{path.name} is not valid UTF-8", file=str(path)) from None
    except tomllib.TOMLDecodeError as exc:
        raise ABPError("SPEC_INVALID", f"{path.name} is not valid TOML: {exc}",
                       file=str(path)) from None
    return AnalysisSpec(validate_spec_dict(raw), path.resolve(),
                        hashlib.sha256(raw_bytes).hexdigest())


# -------------------------------------------------------------- JSON Schema
def _json_field(f: Field) -> dict:
    t = {"str": {"type": "string"}, "bool": {"type": "boolean"}, "int": {"type": "integer"},
         "float": {"type": "number"},
         "str_list": {"type": "array", "items": {"type": "string"}},
         "float_map": {"type": "object", "additionalProperties": {"type": "number"}},
         "number_or_matrix_map": {"type": "object", "additionalProperties": {
             "oneOf": [{"type": "number"},
                       {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}]}},
         }[f.type].copy()
    if f.choices is not None:
        t["enum"] = list(f.choices)
    if f.default is not None:
        t["default"] = f.default
    if f.doc:
        t["description"] = f.doc
    return t


def _json_section(sec: Section) -> dict:
    props, req = {}, []
    for k, f in sec.fields.items():
        if isinstance(f, Section):
            props[k] = _json_section(f)
        elif isinstance(f, TableArray):
            props[k] = {"type": "array", "items": _json_section(f.item)}
            if f.min_items:
                props[k]["minItems"] = f.min_items
        else:
            props[k] = _json_field(f)
        if getattr(f, "required", False):
            req.append(k)
    out = {"type": "object", "additionalProperties": False, "properties": props}
    if req:
        out["required"] = req
    if sec.doc:
        out["description"] = sec.doc
    return out


def json_schema() -> dict:
    s = _json_section(SCHEMA)
    s["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    s["title"] = "ABP analysis spec (schema_version 1)"
    s["description"] = ("Generated from abp.core.spec.SCHEMA; cross-field rules are enforced "
                        "by abp.core.spec.validate_spec_dict.")
    return s


if __name__ == "__main__":  # regenerate contracts/analysis_spec.schema.json
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("contracts/analysis_spec.schema.json")
    target.write_text(json.dumps(json_schema(), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {target}")
