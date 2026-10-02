"""Workflow step for the Bayesian multi-trait models, (co)variances sampled by Gibbs
(``variances.mode = "bayes"``, more than one model trait): the multi-trait threshold
model (``bayes.method = "threshold"``: one ordered categorical trait and continuous
traits) and the multi-trait linear model (``bayes.method = "multitrait"``: continuous
traits only).  ``bayes.residual_groups`` fixes residual covariances between groups at 0.

Diagnostics and traces are written before the convergence decision (a withheld
result keeps its evidence); unconverged chains stop the run with ``ABP-E405``.
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
from .outputs import OutputStage, atomic_write_json

log = logging.getLogger("abp")


def run_mt_threshold(spec: AnalysisSpec, records: RecordSet, structure: GeneticStructure,
                     ped_data: PedigreeData | None, stage: OutputStage, manifest: dict):
    from ..solvers.mt_threshold_gibbs import MTThresholdGibbsConfig, mt_threshold_gibbs
    from .multitrait import EvalState
    d = spec.data
    m = d["model"]
    b = d["bayes"]
    traits = list(m["traits"])
    t = len(traits)
    add_name = next(r["name"] for r in m["random"] if r["kind"] == "additive")
    pe_term = next((r for r in m["random"] if r["kind"] == "iid"), None)
    mat_term = next((r for r in m["random"] if r["kind"] == "maternal"), None)
    types = {tr["name"]: tr["type"] for tr in d["traits"]}
    cat = next((j for j, tr in enumerate(traits) if types[tr] == "categorical"), None)
    model_name = ("multi-trait threshold model" if cat is not None
                  else "Bayesian multi-trait linear model")
    if any(r["kind"] == "maternal" for r in m["random"]):
        if t == 1:
            model_name = ("threshold model" if cat is not None
                          else "Bayesian linear animal model")
        model_name += " with maternal genetic effects"
    groups = None
    if b["residual_groups"]:
        gid = {}
        for j, tr in enumerate(traits):
            gid.setdefault(int(b["residual_groups"][tr]), []).append(j)
        groups = [gid[k] for k in sorted(gid)]
    Y = np.column_stack([records.traits[tr] for tr in traits])
    rec = np.flatnonzero(~np.all(np.isnan(Y), axis=1))
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
        fd = build_fixed_design(cols, terms, intercept=True, n=rows.size)
        X_blocks.append(fd.X)
        fixed_labels.append(fd)
    q = len(structure.labels)
    animal_col = np.array([structure.index[a] for a in animals], dtype=np.int64)
    dam_col = None
    if mat_term is not None:
        ped = ped_data.pedigree
        dams = []
        for a in animals:
            k = ped.index_of([a])[0] if ped.contains(a) else -1
            dk = int(ped.dam[k]) if k >= 0 else -1
            dams.append(structure.index.get(ped.ids[dk], -1) if dk >= 0 else -1)
        dam_col = np.array(dams, dtype=np.int64)
        log.info("maternal genetic effect %r: %d of %d records have a known dam",
                 mat_term["name"], int((dam_col >= 0).sum()), dam_col.size)
    pe_col, pe_levels = None, []
    if pe_term is not None:
        vals = [records.factors[pe_term["column"]][k] for k in rec]
        pe_levels = sorted(set(vals))
        lv = {v: i for i, v in enumerate(pe_levels)}
        pe_col = np.array([lv[v] for v in vals], dtype=np.int64)
    iw = b["variance_prior"] == "inverse_wishart"
    pc = (b["prior_covariance"] or {}) if iw else {}
    pe_prior = pc.get(pe_term["name"]) if pe_term is not None else None
    r_prior = pc.get("residual")
    cfg = MTThresholdGibbsConfig(
        chains=b["chains"], iterations=b["iterations"], burn_in=b["burn_in"], thin=b["thin"],
        seed=b["seed"], rhat_max=b["rhat_max"], ess_min=b["ess_min"],
        max_iterations=b["max_iterations"],
        prior_nu=float(b["nu"]) if iw else None,
        prior_G0=np.array(b["prior_covariance"][add_name], dtype=np.float64) if iw else None,
        residual_groups=groups,
        prior_nu_pe=float(b["nu"]) if pe_prior is not None else None,
        prior_P0=np.array(pe_prior, dtype=np.float64) if pe_prior is not None else None,
        prior_nu_r=float(b["nu"]) if r_prior is not None else None,
        prior_R0=np.array(r_prior, dtype=np.float64) if r_prior is not None else None)
    log.info("%s (%d traits%s%s%s): Gibbs sampler, %d chains, %d iterations (up to %d)",
             model_name, t, "" if cat is None else f", categorical {traits[cat]!r}",
             "" if groups is None else f", residual groups {groups}",
             "" if pe_term is None else f", {pe_term['name']!r}: {len(pe_levels)} levels",
             cfg.chains, cfg.iterations, cfg.max_iterations)
    g = mt_threshold_gibbs(Y, cat, X_blocks, animal_col, structure.k_inv, cfg, pe_col=pe_col,
                           dam_col=dam_col)
    prior_txt = (f"inverse Wishart IW(nu = {cfg.prior_nu:g}, nu G_prior), G_prior = "
                 f"{np.round(cfg.prior_G0, 6).tolist()}" if iw else
                 "flat on the positive definite matrices (IW with nu = -(t + 1), scale 0)")
    if cat is not None:
        method_txt = ("multi-trait threshold model (one ordered categorical trait, continuous "
                      "traits), Gibbs sampler: data augmentation of missing values, Cowles "
                      "threshold step, block draws of the location effects, inverse-Wishart "
                      "G0, R0 with R0[c,c] = 1 (Korsgaard et al. 2003)")
        r0_txt = "flat on (b, S) of R0 = [[1, b'], [b, S + b b']]"
    else:
        method_txt = ("Bayesian multi-trait linear model (continuous traits), Gibbs sampler: "
                      "data augmentation of missing values, block draws of the location "
                      "effects, inverse-Wishart G0 and R0")
        r0_txt = "flat on the positive definite matrices: R0 | E ~ IW(E'E, n - t - 1)"
    if r_prior is not None:
        r0_txt = (f"inverse Wishart IW(nu = {cfg.prior_nu_r:g}, nu R_prior) per block of "
                  f"continuous traits, R_prior = {np.round(cfg.prior_R0, 6).tolist()}"
                  + ("" if cat is None else "; categorical trait alone, variance fixed at 1")
                  + ("" if groups is None else "; block diagonal, residual covariances "
                     f"between the groups {[[traits[j] for j in B] for B in groups]} fixed at 0"))
    if groups is not None and r_prior is None:
        parts = []
        for B in groups:
            names = [traits[j] for j in B]
            if cat in B and len(B) == 1:
                parts.append(f"{names}: residual variance fixed at 1 (liability scale)")
            elif cat in B:
                parts.append(f"{names}: flat on (b, S) of [[1, b'], [b, S + b b']]")
            else:
                parts.append(f"{names}: flat, R0_BB | E_B ~ IW(E_B'E_B, n - |B| - 1)")
        r0_txt = ("block diagonal, residual covariances between the groups fixed at 0, "
                  "each block sampled from its own conditional; " + "; ".join(parts))
    diag = {"method": method_txt, "model": model_name,
            "genetic_effects": [f"{add_name}:{tr}" for tr in traits]
            + ([f"{mat_term['name']}:{tr}" for tr in traits] if mat_term is not None else []),
            "traits": traits, "categorical_trait": None if cat is None else traits[cat],
            "residual_groups": (None if groups is None
                                else [[traits[j] for j in B] for B in groups]),
            "converged": g.converged, "iterations": g.iterations,
            "draws_per_chain": g.draws_per_chain, "chains": cfg.chains, "thin": cfg.thin,
            "burn_in": cfg.burn_in, "chain_seeds": g.seeds, "master_seed": cfg.seed,
            "wall_seconds": g.wall_seconds,
            "criteria": {"rhat_max": cfg.rhat_max, "ess_min": cfg.ess_min},
            "summaries": g.summaries, "ebv_diagnostics": g.ebv_diagnostics,
            "G0": g.G0, "R0": g.R0, "derived": g.derived,
            "threshold_step": ({"acceptance_after_burn_in": g.acceptance}
                               if cat is not None else None),
            "permanent_environment": (None if pe_term is None else {
                "term": pe_term["name"], "column": pe_term["column"],
                "levels": len(pe_levels), "P0": g.P0,
                "prior": (f"inverse Wishart IW(nu = {cfg.prior_nu_pe:g}, nu P_prior), P_prior "
                          f"= {np.round(cfg.prior_P0, 6).tolist()}" if pe_prior is not None
                          else "flat on the positive definite matrices")}),
            "priors": {"G0": prior_txt, "R0": r0_txt, "fixed_effects": "flat",
                       **({"thresholds": "flat subject to ordering"} if cat is not None
                          else {})},
            "trace_file": "mcmc_trace_multitrait.csv"}
    atomic_write_json(stage.path("mcmc_diagnostics_multitrait.json"), diag)
    names = list(g.traces)
    write_csv(stage.path("mcmc_trace_multitrait.csv"), ["chain", "iteration"] + names,
              [[c + 1, cfg.burn_in + (k + 1) * cfg.thin] + [float(g.traces[x][c, k])
                                                            for x in names]
               for c in range(cfg.chains) for k in range(g.draws_per_chain)])
    manifest["diagnostics"]["multi_trait"] = {
        "traits": traits, "mcmc": {k: diag[k] for k in ("method", "converged", "iterations",
                                                        "chains", "chain_seeds", "criteria")}}
    manifest["randomness"] = {"seed": cfg.seed, "generator": "numpy PCG64 via SeedSequence.spawn",
                              "chain_seeds": g.seeds}
    if not g.converged:
        raise ABPError("MCMC_NOT_CONVERGED",
                       f"{model_name}: Gibbs diagnostics not met after "
                       f"{g.iterations} iterations (criteria R-hat < {cfg.rhat_max}, "
                       f"ESS >= {cfg.ess_min})",
                       summaries={k: {x: v[x] for x in ("rhat", "ess_bulk", "ess_tail")}
                                  for k, v in g.summaries.items()}, ebv=g.ebv_diagnostics)
    G0 = np.array(g.G0["mean"])
    R0 = np.array(g.R0["mean"])
    log.info("%s converged after %d iterations; posterior mean G0 %s", model_name,
             g.iterations, np.round(G0, 4).tolist())
    k_diag = np.asarray(structure.k_diag, dtype=np.float64)
    rel = np.clip(1.0 - g.pev / (k_diag[:, None] * np.diag(G0)[None, :t]), 0.0, 1.0)
    sep = np.sqrt(g.pev)
    if mat_term is not None:          # maternal genetic effects: columns t..2t-1 of G0
        mrel = np.clip(1.0 - g.maternal_pev / (k_diag[:, None] * np.diag(G0)[None, t:]),
                       0.0, 1.0)
        msep = np.sqrt(g.maternal_pev)
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
        if mat_term is not None:
            header += [f"mebv_{tr}", f"mreliability_{tr}", f"msep_{tr}"]
    F = ped.inbreeding() if ped is not None else None
    rows_out = []
    for i, a in enumerate(structure.labels):
        if ped is not None and ped.contains(a):
            k = ped.index_of([a])[0]
            base = [a, ped.ids[ped.sire[k]] if ped.sire[k] >= 0 else "",
                    ped.ids[ped.dam[k]] if ped.dam[k] >= 0 else "", sex.get(a, "U"),
                    int(ped.generation[k]), float(F[k])]
        else:
            base = [a, "", "", "", None, None]
        for j, tr in enumerate(traits):
            base += [float(g.ebv[i, j]), float(rel[i, j]), float(sep[i, j]),
                     n_rec[tr].get(a, 0)]
            if mat_term is not None:
                base += [float(g.maternal_ebv[i, j]), float(mrel[i, j]), float(msep[i, j])]
        rows_out.append(base)
    write_csv(stage.path("ebv_multitrait.csv"), header, rows_out)
    if pe_term is not None:
        write_csv(stage.path("pe_multitrait.csv"), [pe_term["column"]]
                  + [f"pe_{tr}" for tr in traits],
                  [[lvl] + [float(v) for v in g.pe_mean[i]] for i, lvl in enumerate(pe_levels)])
    P0m = np.array(g.P0["mean"]) if pe_term is not None else None
    bayes_out = {"method": b["method"], "chains": cfg.chains, "iterations": g.iterations,
                 "converged": g.converged, "variance_prior": prior_txt,
                 "summaries": g.summaries, "ebv_diagnostics": g.ebv_diagnostics,
                 "thresholds": g.thresholds_mean.tolist() if cat is not None else None,
                 "categories": ([float(x) for x in g.categories] if cat is not None
                                else None),
                 "diagnostics_file": "mcmc_diagnostics_multitrait.json",
                 "trace_file": "mcmc_trace_multitrait.csv"}
    out: dict = {}
    top_n = d["output"]["top_n"]
    for j, tr in enumerate(traits):
        fname = f"fixed_effects_{tr}.csv"
        fd = fixed_labels[j]
        fr, k = [], 0
        for lab, kept in zip(fd.labels, fd.kept):
            if kept:
                fr.append([lab[0], lab[1], float(g.fixed_mean[j][k]),
                           "posterior_mean_under_constraints"])
                k += 1
            else:
                fr.append([lab[0], lab[1], 0.0, "constrained_to_zero"])
        write_csv(stage.path(fname), ["term", "level", "solution", "status"], fr)
        e = g.ebv[:, j]
        order = np.argsort(-e, kind="stable")[:top_n]
        h2 = g.derived[f"h2_{j}"]["mean"]
        out[tr] = {
            "unit": "liability" if cat is not None and j == cat else records.units.get(tr, ""),
            "n_records": int((~np.isnan(Y[:, j])).sum()),
            "n_animals_evaluated": q,
            "variance_source": f"bayes ({model_name}, Gibbs sampler; posterior means)",
            "variance_components": {add_name: float(G0[j, j]), "residual": float(R0[j, j]),
                                    **({pe_term["name"]: float(P0m[j, j])}
                                       if pe_term is not None else {}),
                                    **({mat_term["name"]: float(G0[t + j, t + j]),
                                        f"{add_name}-{mat_term['name']} covariance":
                                        float(G0[j, t + j])}
                                       if mat_term is not None else {})},
            **({"maternal": {"term": mat_term["name"], "m2": g.derived[f"m2_{j}"],
                             "direct_maternal_correlation": g.derived[f"rG_{j}_{t + j}"],
                             "reliability_summary": {"mean": float(mrel[:, j].mean()),
                                                     "max": float(mrel[:, j].max())}}}
               if mat_term is not None else {}),
            "heritability": float(h2),
            "reml": None,
            "genetic_term": add_name,
            "solver": {"method": "gibbs", "selection_reason": f"{model_name}, variances.mode = bayes",
                       "n_equations": int(g.n_equations), "relative_residual": None,
                       "iterations": g.iterations, "wall_seconds": g.wall_seconds,
                       "pev": "posterior variance", "joint_multi_trait_system": True},
            "fixed_effects": None,
            "n_fixed_constrained": len(fd.constrained_labels),
            "top": [{"rank": r + 1, "animal": structure.labels[k],
                     "sex": sex.get(structure.labels[k], "U"), "ebv": float(e[k]),
                     "reliability": float(rel[k, j]), "sep": float(sep[k, j]),
                     "n_records": n_rec[tr].get(structure.labels[k], 0)}
                    for r, k in enumerate(order)],
            "reliability_summary": {"mean": float(rel[:, j].mean()),
                                    "min": float(rel[:, j].min()),
                                    "max": float(rel[:, j].max()), "n_rounding_clamped": 0},
            "ebv_summary": {"mean": float(e.mean()), "sd": float(e.std()),
                            "min": float(e.min()), "max": float(e.max())},
            "files": {"ebv": "ebv_multitrait.csv", "fixed_effects": fname,
                      **({"permanent_environment": "pe_multitrait.csv"}
                         if pe_term is not None else {})},
            "bayes": bayes_out,
            "genetic_correlations": {f"{traits[a]}-{traits[c]}":
                                     g.derived[f"rG_{a}_{c}"] for a in range(t)
                                     for c in range(a + 1, t)},
        }
    state = EvalState(structure.labels, traits, g.ebv, g.pev_blocks, G0[:t, :t],
                      structure.k_diag, True, sex)
    return out, state
