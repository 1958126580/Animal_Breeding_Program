"""Aggregate-objective index on EBVs (written after the genetic evaluation)."""

from __future__ import annotations

import numpy as np

from ..core.spec import AnalysisSpec
from ..decision.selection_index import ebv_index
from ..io.tables import write_csv
from .multitrait import EvalState
from .outputs import OutputStage


def write_index(spec: AnalysisSpec, state: EvalState, stage: OutputStage) -> dict:
    ix = spec["index"]
    w = np.array([ix["weights"].get(t, 0.0) for t in state.traits])
    idx, rel = ebv_index(state.ebv, state.pev_blocks, w, state.G0, state.k_diag)
    order = np.argsort(-idx, kind="stable")
    write_csv(stage.path("index.csv"), ["rank", "animal", "sex", "index", "reliability"],
              [[r + 1, state.labels[k], state.sex.get(state.labels[k], "U"), float(idx[k]),
                None if rel is None else float(rel[k])] for r, k in enumerate(order)])
    zero = [t for t in state.traits if t not in ix["weights"]]
    if state.multi_trait:
        note = ("Index I = sum_j a_j EBV_j computed from the joint multi-trait BLUP; it is the "
                "BLUP of the aggregate genotype H = sum_j a_j g_j, and its reliability is "
                "1 - a'PEV a / (a'G0 a K_ii).")
    else:
        note = "Single-trait model: the index is the EBV multiplied by its economic weight."
    if zero:
        note += " Traits without a weight (weight 0): " + ", ".join(zero) + "."
    top_n = spec["output"]["top_n"]
    return {"weights": {t: float(v) for t, v in zip(state.traits, w)},
            "weight_units": ix["weight_units"], "synthetic_weights": ix["synthetic_weights"],
            "method_note": note, "file": "index.csv",
            "top": [{"rank": r + 1, "animal": state.labels[k],
                     "sex": state.sex.get(state.labels[k], "U"), "index": float(idx[k]),
                     "reliability": None if rel is None else float(rel[k])}
                    for r, k in enumerate(order[:top_n])]}
