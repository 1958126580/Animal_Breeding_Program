"""Workflow step for multi-trait BLUP with known (co)variances."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..core.design import FixedTerm, build_fixed_design
from ..core.model import GeneticStructure
from ..core.spec import AnalysisSpec
from ..errors import ABPError
from ..io.tables import write_csv
from ..qc.pedigree import PedigreeData
from ..qc.phenotype import RecordSet
from ..solvers.multitrait import STACKING_ORDER, MTData, build_and_solve
from .outputs import OutputStage

log = logging.getLogger("abp")


@dataclass
class EvalState:
    """What the index step needs from an evaluation (all in structure order)."""

    labels: tuple[str, ...]
    traits: list[str]
    ebv: np.ndarray                  # q x t
    pev_blocks: np.ndarray | None    # q x t x t
    G0: np.ndarray                   # t x t
    k_diag: np.ndarray               # q
    multi_trait: bool
    sex: dict[str, str]


def _corr(S: np.ndarray) -> list[list[float]]:
    d = np.sqrt(np.diag(S))
    return (S / np.outer(d, d)).tolist()


def run_multitrait(spec: AnalysisSpec, records: RecordSet, structure: GeneticStructure,
                   ped_data: PedigreeData | None, stage: OutputStage, manifest: dict,
                   budget: int) -> tuple[dict, EvalState]:
    d = spec.data
    m = d["model"]
    traits = list(m["traits"])
    t = len(traits)
    vals = d["variances"]["values"]
    add_name = next(r["name"] for r in m["random"] if r["kind"] == "additive")
    G0 = np.array(vals[add_name], dtype=np.float64)
    R0 = np.array(vals["residual"], dtype=np.float64)
    Y = np.column_stack([records.traits[tr] for tr in traits])
    keep = ~np.all(np.isnan(Y), axis=1)
    rec = np.flatnonzero(keep)
    Y = Y[rec]
    animals = [records.animal[k] for k in rec]
    missing = sorted({a for a in animals if a not in structure.index})
    if missing:
        raise ABPError("PHENOTYPE_UNKNOWN_ANIMAL",
                       f"{len(missing)} animal(s) with records are not in the {structure.kind} "
                       f"relationship structure (e.g. {missing[:5]})", animals=missing[:50])
    X_blocks, fixed_labels = [], []
    for j, tr in enumerate(traits):
        rows = rec[~np.isnan(Y[:, j])]
        cols, terms = {}, []
        for f in m["fixed"]:
            if tr not in f["traits"]:
                continue
            c = f["column"]
            cols[c] = ([records.factors[c][k] for k in rows] if f["type"] == "factor"
                       else records.covariates[c][rows])
            terms.append(FixedTerm(c, f["type"]))
        fd = build_fixed_design(cols, terms, intercept=m["intercept"], n=rows.size)
        X_blocks.append(fd.X)
        fixed_labels.append(fd)
    animal_col = np.array([structure.index[a] for a in animals], dtype=np.int64)
    sol = d["solver"]
    res = build_and_solve(MTData(Y, X_blocks, animal_col), structure.k_inv, structure.k_diag,
                          G0, R0, method=sol["method"], compute_pev=(sol["pev"] == "exact"),
                          tol=sol["tol"], max_iter=sol["max_iter"], memory_budget_bytes=budget)
    s = res.solve
    log.info("multi-trait (%d traits, %d observations): solved %d equations with %s (%s); "
             "relative residual %.2e; %.2f s", t, res.n_obs, s.solution.size, s.method,
             s.selection_reason, s.rel_residual, s.wall_seconds)
    ped = ped_data.pedigree if ped_data else None
    sex = ped_data.sex if ped_data else {}
    n_rec = {tr: {} for tr in traits}
    for j, tr in enumerate(traits):
        for a, obs in zip(animals, ~np.isnan(Y[:, j])):
            if obs:
                n_rec[tr][a] = n_rec[tr].get(a, 0) + 1
    header = ["animal", "sire", "dam", "sex", "generation", "inbreeding"]
    for tr in traits:
        header += [f"ebv_{tr}", f"reliability_{tr}", f"sep_{tr}", f"n_records_{tr}"]
    rows_out = []
    F = ped.inbreeding() if ped is not None else None
    for i, a in enumerate(structure.labels):
        if ped is not None and ped.contains(a):
            k = ped.index_of([a])[0]
            base = [a, ped.ids[ped.sire[k]] if ped.sire[k] >= 0 else "",
                    ped.ids[ped.dam[k]] if ped.dam[k] >= 0 else "", sex.get(a, "U"),
                    int(ped.generation[k]), float(F[k])]
        else:
            base = [a, "", "", "", None, None]
        for j, tr in enumerate(traits):
            base += [float(res.ebv[i, j]),
                     None if res.reliability is None else float(res.reliability[i, j]),
                     None if res.pev_blocks is None else float(np.sqrt(res.pev_blocks[i, j, j])),
                     n_rec[tr].get(a, 0)]
        rows_out.append(base)
    write_csv(stage.path("ebv_multitrait.csv"), header, rows_out)
    out: dict = {}
    top_n = d["output"]["top_n"]
    for j, tr in enumerate(traits):
        fd = fixed_labels[j]
        fname = f"fixed_effects_{tr}.csv"
        fr, k = [], 0
        for lab, kept in zip(fd.labels, fd.kept):
            if kept:
                fr.append([lab[0], lab[1], float(res.fixed[j][k]), "estimated_under_constraints"])
                k += 1
            else:
                fr.append([lab[0], lab[1], 0.0, "constrained_to_zero"])
        write_csv(stage.path(fname), ["term", "level", "solution", "status"], fr)
        e = res.ebv[:, j]
        order = np.argsort(-e, kind="stable")[:top_n]
        rel = None if res.reliability is None else res.reliability[:, j]
        out[tr] = {
            "unit": records.units.get(tr, ""),
            "n_records": int((~np.isnan(Y[:, j])).sum()),
            "n_animals_evaluated": len(structure.labels),
            "variance_source": "known (multi-trait covariance matrices)",
            "variance_components": {add_name: float(G0[j, j]), "residual": float(R0[j, j])},
            "heritability": float(G0[j, j] / (G0[j, j] + R0[j, j])),
            "reml": None,
            "genetic_term": add_name,
            "solver": {"method": s.method, "selection_reason": s.selection_reason,
                       "n_equations": int(s.solution.size), "relative_residual": s.rel_residual,
                       "iterations": s.iterations, "wall_seconds": s.wall_seconds,
                       "pev": sol["pev"], "joint_multi_trait_system": True},
            "fixed_effects": None,
            "n_fixed_constrained": len(fd.constrained_labels),
            "top": [{"rank": r + 1, "animal": structure.labels[k], "sex": sex.get(structure.labels[k], "U"),
                     "ebv": float(e[k]), "reliability": None if rel is None else float(rel[k]),
                     "sep": None if res.pev_blocks is None else float(np.sqrt(res.pev_blocks[k, j, j])),
                     "n_records": n_rec[tr].get(structure.labels[k], 0)}
                    for r, k in enumerate(order)],
            "reliability_summary": None if rel is None else {
                "mean": float(rel.mean()), "min": float(rel.min()), "max": float(rel.max()),
                "n_rounding_clamped": 0},
            "ebv_summary": {"mean": float(e.mean()), "sd": float(e.std()),
                            "min": float(e.min()), "max": float(e.max())},
            "files": {"ebv": "ebv_multitrait.csv", "fixed_effects": fname},
        }
    manifest["diagnostics"]["multi_trait"] = {
        "traits": traits, "stacking_order": STACKING_ORDER, "n_observations": res.n_obs,
        "genetic_covariance": G0.tolist(), "genetic_correlation": _corr(G0),
        "residual_covariance": R0.tolist(), "residual_correlation": _corr(R0),
        "solver": out[traits[0]]["solver"]}
    state = EvalState(structure.labels, traits, res.ebv, res.pev_blocks, G0, structure.k_diag,
                      True, sex)
    return out, state
