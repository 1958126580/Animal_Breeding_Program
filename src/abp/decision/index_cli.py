"""``abp index``: Smith-Hazel / restricted selection index from a TOML file.

File format (all matrices in the order of the name lists)::

    schema_version = "1"
    [index]
    information = ["own_wwt", "own_fat"]        # names of the information sources x
    objective_traits = ["g_wwt", "g_fat"]       # traits in H = a'g
    weights = [2.0, -3.0]                       # a, per unit of each objective trait
    weight_units = "SCU per unit"
    synthetic = true                            # declare placeholder parameters
    P = [[...], [...]]                          # Var(x)            n_x x n_x
    C = [[...], [...]]                          # Cov(x, g')        n_x x n_g
    G = [[...], [...]]                          # Var(g)            n_g x n_g
    restrict = ["g_fat"]                        # optional: zero expected change
    proportion_selected = 0.1                   # optional: response to selection
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np

from ..errors import ABPError
from .selection_index import smith_hazel

REQUIRED = ("information", "objective_traits", "weights", "weight_units", "synthetic", "P", "C", "G")
OPTIONAL = ("restrict", "proportion_selected")


def _check_matrix(v, path):
    if not (isinstance(v, list) and v and all(isinstance(r, list) and r for r in v)
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) for r in v for x in r)
            and len({len(r) for r in v}) == 1):
        raise ABPError("SPEC_INVALID", f"{path}: expected a rectangular numeric matrix", key=path)
    return np.array(v, dtype=np.float64)


def run_index_spec(path: str | Path) -> dict:
    path = Path(path)
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise ABPError("INPUT_NOT_FOUND", f"file not found: {path}", file=str(path)) from None
    except tomllib.TOMLDecodeError as exc:
        raise ABPError("SPEC_INVALID", f"{path.name} is not valid TOML: {exc}") from None
    unknown = set(raw) - {"schema_version", "index"}
    if unknown or raw.get("schema_version") != "1" or "index" not in raw:
        raise ABPError("SPEC_INVALID", "expected schema_version = \"1\" and an [index] table only",
                       unknown=sorted(unknown))
    ix = raw["index"]
    allowed = set(REQUIRED) | set(OPTIONAL)
    extra = set(ix) - allowed
    if extra:
        raise ABPError("SPEC_INVALID", f"index.{sorted(extra)[0]}: unknown key "
                       f"(allowed: {sorted(allowed)})")
    for k in REQUIRED:
        if k not in ix:
            raise ABPError("SPEC_INVALID", f"index.{k}: required key is missing")
    info, traits = list(ix["information"]), list(ix["objective_traits"])
    a = np.array(ix["weights"], dtype=np.float64)
    P = _check_matrix(ix["P"], "index.P")
    C = _check_matrix(ix["C"], "index.C")
    G = _check_matrix(ix["G"], "index.G")
    if not isinstance(ix["synthetic"], bool):
        raise ABPError("SPEC_INVALID", "index.synthetic: expected true or false")
    res = smith_hazel(P, C, a, G, info, traits, restrict=ix.get("restrict", []),
                      proportion_selected=ix.get("proportion_selected"))
    out = {"file": str(path), "weight_units": ix["weight_units"],
           "synthetic_parameters": ix["synthetic"],
           "objective": dict(zip(traits, a.tolist())), **res.to_dict(info, traits)}
    if ix["synthetic"]:
        out["warning"] = "parameters declared synthetic: illustration only"
    return out
