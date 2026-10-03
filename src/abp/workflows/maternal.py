"""Workflow step for the maternal animal model with REML or known (co)variances (round 13).

One trait; an additive term (pedigree), a ``kind = "maternal"`` term (dam from the
pedigree) and optional independent terms (e.g. a maternal permanent environment on a
dam column).  ``variances.mode = "reml"``: :func:`abp.solvers.maternal_reml.maternal_reml_fit`;
``"known"``: the additive term's value is the 2 x 2 matrix ``[[direct, covariance],
[covariance, maternal]]``.  BLUP then gives direct and maternal EBVs with their exact
prediction-error variances (``C^-1``; dense or sparse selected inversion).
"""

from __future__ import annotations

import logging

import numpy as np

from ..core.design import FixedTerm, build_fixed_design
from ..core.model import GeneticStructure
from ..core.spec import AnalysisSpec
from ..errors import ABPError
from ..io.tables import write_csv
from ..qc.pedigree import PedigreeData
from ..qc.phenotype import RecordSet
from .outputs import OutputStage

log = logging.getLogger("abp")


def run_maternal(spec: AnalysisSpec, records: RecordSet, trait: str,
                 structure: GeneticStructure, ped_data: PedigreeData | None,
                 stage: OutputStage, manifest: dict, budget: int):
    from ..solvers.maternal_reml import MaternalData, maternal_blup, maternal_reml_fit
    from .multitrait import EvalState
    d = spec.data
    m = d["model"]
    add = next(r for r in m["random"] if r["kind"] == "additive")
    mat = next(r for r in m["random"] if r["kind"] == "maternal")
    iid_terms = [r for r in m["random"] if r["kind"] == "iid"]
    if structure.kind != "pedigree" or ped_data is None:
        raise ABPError("UNSUPPORTED_COMBINATION", "the maternal animal model needs a pedigree "
                       "relationship matrix (no genetic groups or metafounders)")
    y_all = records.traits[trait]
    rec = np.flatnonzero(~np.isnan(y_all))
    y = y_all[rec]
    animals = [records.animal[k] for k in rec]
    missing = sorted({a for a in animals if a not in structure.index})
    if missing:
        raise ABPError("PHENOTYPE_UNKNOWN_ANIMAL",
                       f"{len(missing)} animal(s) with records are not in the pedigree "
                       f"(e.g. {missing[:5]})", animals=missing[:50])
    cols, terms = {}, []
    for f in m["fixed"]:
        if trait not in f["traits"]:
            continue
        c = f["column"]
        cols[c] = ([records.factors[c][k] for k in rec] if f["type"] == "factor"
                   else records.covariates[c][rec])
        terms.append(FixedTerm(c, f["type"]))
    fd = build_fixed_design(cols, terms, intercept=m["intercept"], n=rec.size)
    animal_col = np.array([structure.index[a] for a in animals], dtype=np.int64)
    ped = ped_data.pedigree
    dams = []
    for a in animals:
        k = ped.index_of([a])[0]
        dk = int(ped.dam[k])
        dams.append(structure.index.get(ped.ids[dk], -1) if dk >= 0 else -1)
    dam_col = np.array(dams, dtype=np.int64)
    log.info("maternal genetic effect %r: %d of %d records have a known dam", mat["name"],
             int((dam_col >= 0).sum()), dam_col.size)
    if not (dam_col >= 0).any():
        raise ABPError("MODEL_NOT_IDENTIFIABLE", "no record has a known dam: the maternal "
                       "genetic effect cannot be estimated")
    iid, iid_levels = [], []
    for r in iid_terms:
        vals = [records.factors[r["column"]][k] for k in rec]
        lv = sorted(set(vals))
        idx = {v: i for i, v in enumerate(lv)}
        iid.append((r["name"], np.array([idx[v] for v in vals], dtype=np.int64), len(lv)))
        iid_levels.append(lv)
    data = MaternalData(y, fd.X.tocsr(), animal_col, dam_col, iid)
    reml_info = None
    if d["variances"]["mode"] == "reml":
        fit = maternal_reml_fit(data, structure.k_inv, structure.logdet_k, d["reml"],
                                memory_budget_bytes=budget)
        G0 = fit.G0
        iid_vars = [fit.iid[nm] for nm, _, _ in iid]
        resid = fit.residual
        reml_info = fit.to_dict()
        log.info("maternal-model REML %s after %d iterations (%s); logL %.6f; G0 %s",
                 fit.status, fit.iterations, fit.trace_method, fit.loglik,
                 np.round(G0, 4).tolist())
        variance_source = "reml (maternal animal model, AI-REML with EM fallback)"
        boundary = fit.start.get("boundary", [])
    else:
        vals = d["variances"]["values"]
        G0 = np.array(vals[add["name"]], dtype=np.float64)
        iid_vars = [float(vals[r["name"]]) for r in iid_terms]
        resid = float(vals["residual"])
        variance_source = "known (maternal animal model)"
        boundary = []
        ev_ = np.linalg.eigvalsh(G0) if G0.shape == (2, 2) else None
        if ev_ is None or not np.allclose(G0, G0.T) or ev_[0] <= 0:
            raise ABPError("COVARIANCE_NOT_PD", f"variances.values.{add['name']} must be a "
                           "symmetric positive-definite 2 x 2 matrix (direct, maternal)")
    # BLUP; a term estimated at zero is dropped from the BLUP model
    keep = [i for i, (nm, _, _) in enumerate(iid) if nm not in boundary]
    data_b = MaternalData(y, data.X, animal_col, dam_col, [iid[i] for i in keep])
    beta, U, pev, csol, sinfo = maternal_blup(data_b, structure.k_inv, G0,
                                       [iid_vars[i] for i in keep], resid,
                                       memory_budget_bytes=budget)
    k_diag = np.asarray(structure.k_diag, dtype=np.float64)
    kh = None                      # PEV including the uncertainty of the REML estimates
    if reml_info is not None and fit.cov is not None:
        from ..solvers.vc_uncertainty import kackar_harville_delta_maternal
        th = np.concatenate([[G0[0, 0], G0[0, 1], G0[1, 1]], iid_vars, [resid]])
        D = kackar_harville_delta_maternal(data, structure.k_inv, th, fit.cov,
                                           memory_budget_bytes=budget)
        kh = pev + D
        log.info("maternal model: PEV including REML uncertainty (Kackar-Harville): mean "
                 "relative increase direct %.4f, maternal %.4f",
                 float(np.mean(D[:, 0, 0] / pev[:, 0, 0])), float(np.mean(D[:, 1, 1] / pev[:, 1, 1])))
    rel = np.clip(1.0 - pev[:, 0, 0] / (k_diag * G0[0, 0]), 0.0, 1.0)
    mrel = np.clip(1.0 - pev[:, 1, 1] / (k_diag * G0[1, 1]), 0.0, 1.0)
    sep, msep = np.sqrt(pev[:, 0, 0]), np.sqrt(pev[:, 1, 1])
    sex = ped_data.sex
    F = ped.inbreeding()
    n_rec: dict[str, int] = {}
    for a in animals:
        n_rec[a] = n_rec.get(a, 0) + 1
    header = ["animal", "sire", "dam", "sex", "generation", "inbreeding", "ebv", "reliability",
              "sep", "mebv", "mreliability", "msep", "n_records"]
    if kh is not None:
        header += ["pev_incl_vc_uncertainty", "reliability_incl_vc_uncertainty",
                   "mpev_incl_vc_uncertainty", "mreliability_incl_vc_uncertainty"]
        rel_kh = np.clip(1.0 - kh[:, 0, 0] / (k_diag * G0[0, 0]), 0.0, 1.0)
        mrel_kh = np.clip(1.0 - kh[:, 1, 1] / (k_diag * G0[1, 1]), 0.0, 1.0)
    rows = []
    for i, a in enumerate(structure.labels):
        k = ped.index_of([a])[0]
        rows.append([a, ped.ids[ped.sire[k]] if ped.sire[k] >= 0 else "",
                     ped.ids[ped.dam[k]] if ped.dam[k] >= 0 else "", sex.get(a, "U"),
                     int(ped.generation[k]), float(F[k]), float(U[i, 0]), float(rel[i]),
                     float(sep[i]), float(U[i, 1]), float(mrel[i]), float(msep[i]),
                     n_rec.get(a, 0)]
                    + ([float(kh[i, 0, 0]), float(rel_kh[i]), float(kh[i, 1, 1]),
                        float(mrel_kh[i])] if kh is not None else []))
    files = {"ebv": f"ebv_{trait}.csv", "fixed_effects": f"fixed_effects_{trait}.csv"}
    write_csv(stage.path(files["ebv"]), header, rows)
    fr, k = [], 0
    for lab, kept in zip(fd.labels, fd.kept):
        if kept:
            fr.append([lab[0], lab[1], float(beta[k]), "estimated_under_constraints"])
            k += 1
        else:
            fr.append([lab[0], lab[1], 0.0, "constrained_to_zero"])
    write_csv(stage.path(files["fixed_effects"]), ["term", "level", "solution", "status"], fr)
    for j, i in enumerate(keep):
        r = iid_terms[i]
        fname = f"{r['name']}_{trait}.csv"
        write_csv(stage.path(fname), [r["column"], "solution"],
                  [[lvl, float(v)] for lvl, v in zip(iid_levels[i], csol[j])])
        files[r["name"]] = fname
    vp = {r["name"]: float(v) for r, v in zip(iid_terms, iid_vars)}
    sP = float(G0[0, 0] + G0[1, 1] + G0[0, 1] + sum(vp.values()) + resid)
    vc = {add["name"]: float(G0[0, 0]), mat["name"]: float(G0[1, 1]),
          f"{add['name']}-{mat['name']} covariance": float(G0[0, 1]), **vp,
          "residual": float(resid)}
    if reml_info is not None:
        reml_info["heritability_se"] = reml_info["derived"].get("h2_se")
        reml_info["boundary"] = boundary
    top_n = d["output"]["top_n"]
    order = np.argsort(-U[:, 0], kind="stable")[:top_n]
    out = {
        "unit": records.units.get(trait, ""),
        "n_records": int(rec.size),
        "n_animals_evaluated": len(structure.labels),
        "variance_source": variance_source,
        "variance_components": vc,
        "heritability": float(G0[0, 0] / sP),
        "maternal": {"term": mat["name"], "m2": float(G0[1, 1] / sP),
                     "direct_maternal_correlation": float(G0[0, 1]
                                                          / np.sqrt(G0[0, 0] * G0[1, 1])),
                     "phenotypic_variance": sP,
                     "reliability_summary": {"mean": float(mrel.mean()),
                                             "max": float(mrel.max())},
                     "records_with_known_dam": int((dam_col >= 0).sum())},
        "reml": reml_info,
        "genetic_term": add["name"],
        "solver": {**sinfo, "pev": "exact"},
        "fixed_effects": None,
        "n_fixed_constrained": len(fd.constrained_labels),
        "top": [{"rank": r_ + 1, "animal": structure.labels[k_], "sex": sex.get(structure.labels[k_], "U"),
                 "ebv": float(U[k_, 0]), "reliability": float(rel[k_]), "sep": float(sep[k_]),
                 "mebv": float(U[k_, 1]), "n_records": n_rec.get(structure.labels[k_], 0)}
                for r_, k_ in enumerate(order)],
        "reliability_summary": {"mean": float(rel.mean()), "min": float(rel.min()),
                                "max": float(rel.max()), "n_rounding_clamped": 0},
        "ebv_summary": {"mean": float(U[:, 0].mean()), "sd": float(U[:, 0].std()),
                        "min": float(U[:, 0].min()), "max": float(U[:, 0].max())},
        "files": files,
    }
    manifest["diagnostics"].setdefault(trait, {}).update(
        {"maternal_model": {"variance_source": variance_source, "reml": reml_info,
                            "G0": G0.tolist(), "fixed_constrained":
                                [list(x) for x in fd.constrained_labels]}})
    state = EvalState(tuple(structure.labels), [trait], U[:, :1].copy(),
                      pev[:, :1, :1].copy(), G0[:1, :1].copy(), k_diag, False, sex)
    return out, state
