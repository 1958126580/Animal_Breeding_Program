"""Workflow step for Bayesian marker regression (``variances.mode = "bayes"``).

Estimand: the posterior mean of the genomic breeding value ``w_i' beta`` of
every QC-passed genotyped animal, relative to the genomic base defined by
the declared allele frequencies.  Posterior standard deviations are reported
instead of PEV-based reliabilities (they are not the same quantity).  If the
convergence diagnostics fail within ``bayes.max_iterations``, no result is
published (``ABP-E405``); the diagnostics are kept in the failed run folder.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np

from ..core.design import FixedTerm, build_fixed_design
from ..core.genomic import centered
from ..core.spec import AnalysisSpec
from ..errors import ABPError
from ..io.tables import write_csv
from ..qc.pedigree import PedigreeData
from ..qc.phenotype import RecordSet
from ..solvers.bayes import BayesConfig, run_bayes
from .genomic_inputs import prepare_genotypes
from .outputs import OutputStage, atomic_write_json

log = logging.getLogger("abp")


def run_bayes_trait(spec: AnalysisSpec, records: RecordSet, trait: str,
                    ped_data: PedigreeData | None, stage: OutputStage, manifest: dict,
                    phe_qc) -> tuple[dict, object]:
    from .evaluate import _handle_ungenotyped
    from .multitrait import EvalState
    d = spec.data
    used = records.used_mask([trait])
    with_rec = {a for a, u in zip(records.animal, used) if u}
    prep = prepare_genotypes(spec, ped_data, manifest, with_rec)
    geno = prep.geno
    index = {a: i for i, a in enumerate(geno.ids)}
    _handle_ungenotyped(records, SimpleNamespace(index=index), phe_qc, d)
    y_all = records.traits[trait]
    rows = np.flatnonzero(~np.isnan(y_all))
    if rows.size == 0:
        raise ABPError("NO_USABLE_RECORDS", f"no records of genotyped animals for {trait!r}")
    animals = [records.animal[k] for k in rows]
    if len(set(animals)) != len(animals):
        raise ABPError("UNSUPPORTED_COMBINATION", "Bayesian marker models in this version take "
                       "one record per animal (no permanent-environment term)")
    cols, terms = {}, []
    for f in d["model"]["fixed"]:
        c = f["column"]
        cols[c] = ([records.factors[c][k] for k in rows] if f["type"] == "factor"
                   else records.covariates[c][rows])
        terms.append(FixedTerm(c, f["type"]))
    fd = build_fixed_design(cols, terms, intercept=d["model"]["intercept"], n=rows.size)
    W_all = centered(geno.dosage, prep.p, geno.missing)
    W_train = W_all[[index[a] for a in animals]]
    sum2pq = float(2.0 * np.sum(prep.p * (1.0 - prep.p)))
    b = d["bayes"]
    cfg = BayesConfig(method=b["method"], chains=b["chains"], iterations=b["iterations"],
                      burn_in=b["burn_in"], thin=b["thin"], seed=b["seed"],
                      prior_r2=b["prior_r2"], pi0=b["pi0"], nu=b["nu"], nu_e=b["nu_e"],
                      rhat_max=b["rhat_max"], ess_min=b["ess_min"],
                      max_iterations=b["max_iterations"])
    log.info("%s: Bayesian %s on %d records, %d markers, %d chains (kernel will be reported)",
             trait, cfg.method, rows.size, W_all.shape[1], cfg.chains)
    res = run_bayes(records.traits[trait][rows], fd.X.toarray(), W_train, W_all, cfg, sum2pq)
    diag = {"method": res.method, "converged": res.converged, "iterations": res.iterations,
            "draws_per_chain": res.draws_per_chain, "chains": cfg.chains,
            "thin": cfg.thin, "burn_in": cfg.burn_in, "chain_seeds": res.seeds,
            "master_seed": cfg.seed, "kernel": res.kernel, "wall_seconds": res.wall_seconds,
            "criteria": {"rhat_max": cfg.rhat_max, "ess_min": cfg.ess_min},
            "summaries": res.summaries, "gebv_diagnostics": res.gebv_diagnostics,
            "priors": {"nu": res.priors.nu, "s2": res.priors.s2, "nu_e": res.priors.nu_e,
                       "s2_e": res.priors.s2_e, "pi0": res.priors.pi0,
                       "gamma": list(res.priors.gamma), "dirichlet": list(res.priors.dirichlet),
                       "derivation": res.priors.derivation},
            "frequencies": prep.freq_note,
            "posterior_predictive": res.ppc,
            "trace_file": f"mcmc_trace_{trait}.csv"}
    atomic_write_json(stage.path(f"mcmc_diagnostics_{trait}.json"), diag)
    extreme = {k: v["p_value"] for k, v in res.ppc["statistics"].items()
               if not 0.01 <= v["p_value"] <= 0.99}
    if extreme:
        log.warning("%s: posterior predictive check flags %s (p outside [0.01, 0.99]); the model "
                    "does not reproduce these features of the data - review before use",
                    trait, extreme)
    names = list(res.traces)
    trace_rows = [[c + 1, cfg.burn_in + (k + 1) * cfg.thin]
                  + [float(res.traces[q][c, k]) for q in names]
                  for c in range(cfg.chains) for k in range(res.draws_per_chain)]
    write_csv(stage.path(f"mcmc_trace_{trait}.csv"), ["chain", "iteration"] + names, trace_rows)
    manifest["diagnostics"][trait] = {"mcmc": {k: diag[k] for k in (
        "method", "converged", "iterations", "chains", "chain_seeds", "kernel", "criteria")}}
    manifest["randomness"] = {"seed": cfg.seed, "generator": "numpy PCG64 via SeedSequence.spawn",
                              "chain_seeds": res.seeds}
    if not res.converged:
        raise ABPError("MCMC_NOT_CONVERGED",
                       f"{trait}: diagnostics not met after {res.iterations} iterations "
                       f"(criteria R-hat < {cfg.rhat_max}, ESS >= {cfg.ess_min})",
                       summaries={k: {m: v[m] for m in ("rhat", "ess_bulk", "ess_tail")}
                                  for k, v in res.summaries.items()},
                       gebv=res.gebv_diagnostics)
    n_rec = {a: 1 for a in animals}
    ped = ped_data.pedigree if ped_data else None
    sex = ped_data.sex if ped_data else {}
    out_rows = []
    for i, a in enumerate(geno.ids):
        s = dm = ""
        if ped is not None and ped.contains(a):
            k = ped.index_of([a])[0]
            s = ped.ids[ped.sire[k]] if ped.sire[k] >= 0 else ""
            dm = ped.ids[ped.dam[k]] if ped.dam[k] >= 0 else ""
        out_rows.append([a, s, dm, sex.get(a, "U"), n_rec.get(a, 0), float(res.gebv_mean[i]),
                         float(res.gebv_sd[i])])
    write_csv(stage.path(f"ebv_{trait}.csv"), ["animal", "sire", "dam", "sex", "n_records",
                                               "gebv_posterior_mean", "gebv_posterior_sd"], out_rows)
    write_csv(stage.path(f"marker_effects_{trait}.csv"),
              ["marker_id", "counted_allele", "frequency", "effect_posterior_mean",
               "inclusion_probability"],
              [[mk, ca, float(p), float(bm), float(ip)] for mk, ca, p, bm, ip in zip(
                  geno.markers, geno.counted_allele, prep.p, res.beta_mean, res.inclusion_prob)])
    fixed_rows, k = [], 0
    for lab, kept in zip(fd.labels, fd.kept):
        if kept:
            fixed_rows.append([lab[0], lab[1], float(res.fixed_mean[k]), "posterior_mean"])
            k += 1
        else:
            fixed_rows.append([lab[0], lab[1], 0.0, "constrained_to_zero"])
    write_csv(stage.path(f"fixed_effects_{trait}.csv"), ["term", "level", "solution", "status"],
              fixed_rows)
    top_n = d["output"]["top_n"]
    order = np.argsort(-res.gebv_mean, kind="stable")[:top_n]
    sm = res.summaries
    out = {
        "unit": records.units.get(trait, ""),
        "n_records": int(rows.size),
        "n_animals_evaluated": len(geno.ids),
        "variance_source": f"bayes ({res.method}; posterior means)",
        "variance_components": {"genetic_variance": sm["genetic_variance"]["mean"],
                                "residual": sm["sigma_e2"]["mean"]},
        "heritability": sm["h2"]["mean"],
        "reml": None,
        "bayes": {"method": res.method, "converged": res.converged,
                  "iterations": res.iterations, "chains": cfg.chains,
                  "summaries": {k: {m: v[m] for m in ("mean", "sd", "q05", "q95", "rhat",
                                                      "ess_bulk", "ess_tail", "mcse_mean")}
                                for k, v in sm.items()},
                  "gebv_diagnostics": res.gebv_diagnostics, "kernel": res.kernel,
                  "posterior_predictive": res.ppc},
        "genetic_term": "animal",
        "solver": {"method": f"gibbs ({res.kernel})", "selection_reason": "variances.mode = bayes",
                   "n_equations": int(W_all.shape[1] + fd.rank), "relative_residual": None,
                   "iterations": res.iterations, "wall_seconds": res.wall_seconds, "pev": "none"},
        "fixed_effects": None,
        "n_fixed_constrained": len(fd.constrained_labels),
        "top": [{"rank": r + 1, "animal": geno.ids[i], "sex": sex.get(geno.ids[i], "U"),
                 "ebv": float(res.gebv_mean[i]), "reliability": None,
                 "sep": float(res.gebv_sd[i]), "n_records": n_rec.get(geno.ids[i], 0)}
                for r, i in enumerate(order)],
        "reliability_summary": None,
        "ebv_summary": {"mean": float(res.gebv_mean.mean()), "sd": float(res.gebv_mean.std()),
                        "min": float(res.gebv_mean.min()), "max": float(res.gebv_mean.max())},
        "files": {"ebv": f"ebv_{trait}.csv", "fixed_effects": f"fixed_effects_{trait}.csv",
                  "marker_effects": f"marker_effects_{trait}.csv",
                  "diagnostics": f"mcmc_diagnostics_{trait}.json",
                  "trace": f"mcmc_trace_{trait}.csv"},
    }
    state = EvalState(tuple(geno.ids), [trait], res.gebv_mean[:, None], None,
                      np.array([[sm["genetic_variance"]["mean"]]]), np.ones(len(geno.ids)),
                      False, sex)
    return out, state
