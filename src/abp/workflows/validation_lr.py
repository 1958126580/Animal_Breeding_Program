"""Forward-in-time validation by the LR method (module M13).

Design (method registry id ``val.lr``)
--------------------------------------
Two evaluations are compared on a set of *focal* animals:

* **partial**: records dated after ``validation.cutoff`` are hidden (their
  values are never read by the partial evaluation);
* **whole**: all records.

Both use the **same** variance components.  With ``variances.mode = "known"``
they are the spec values; with ``"reml"`` they are estimated on the partial
data only, so that no hidden phenotype influences the partial evaluation
(training boundary).  The published main evaluation of the run is not
affected by this step.

Statistics on the focal animals (definitions of the project specification,
§5.8; Legarra & Reverter 2018):

    Delta_p = mean(u_p) - mean(u_w)                          (bias; expected 0)
    b_w|p   = Cov(u_w, u_p) / Var(u_p)                       (dispersion; expected 1)
    rho_wp  = Cov(u_w, u_p) / sqrt(Var(u_w) Var(u_p))        (relates to acc_p / acc_w)

Uncertainty: cluster bootstrap over sire families (or over animals), with a
recorded seed; 95% percentile intervals.  The LR *population accuracy*
estimator is deliberately **not** computed: its published form has a 2019
erratum that could not be verified in this environment (see the method
registry).  The assumptions of LR (same model and parameters in both
evaluations; focal animals without own records in the partial data) are
stated in the output.  A correlation near 1 only shows that the two
evaluations agree; it does not prove that either is accurate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..core.model import GeneticStructure, build_single_trait
from ..core.spec import AnalysisSpec
from ..errors import ABPError
from ..io.tables import write_csv
from ..qc.pedigree import PedigreeData, parse_date
from ..qc.phenotype import RecordSet
from ..solvers.blup import blup
from .outputs import OutputStage

log = logging.getLogger("abp")


def _date_range(value: str) -> tuple[tuple[int, int, int], tuple[int, int, int]] | None:
    """Earliest and latest calendar day covered by YYYY, YYYY-MM or YYYY-MM-DD."""
    p = parse_date(value)
    if p is None:
        return None
    if len(p) == 3:
        return p, p
    if len(p) == 2:
        return (p[0], p[1], 1), (p[0], p[1], 31)
    return (p[0], 1, 1), (p[0], 12, 31)


def _earliest(parts: tuple[int, ...]) -> tuple[int, int, int]:
    """Earliest calendar day of a (year[, month[, day]]) tuple."""
    return tuple(list(parts) + [1] * (3 - len(parts)))  # type: ignore[return-value]


def split_by_cutoff(records: RecordSet, cutoff: str) -> np.ndarray:
    """Boolean mask of records hidden in the partial evaluation (dated after cutoff).

    A record whose date range straddles the cutoff (e.g. year 2024 against a
    cutoff 2024-06-30) cannot be classified and is an error.
    """
    if records.dates is None:
        raise ABPError("SPEC_INVALID", "validation needs data.phenotype_columns.date")
    c = parse_date(cutoff)
    hidden = np.zeros(records.n, dtype=bool)
    bad, ambiguous = [], []
    for k, v in enumerate(records.dates):
        rng = _date_range(v)
        if rng is None:
            bad.append({"line": records.line[k], "value": v})
            continue
        lo, hi = rng
        if lo > c:
            hidden[k] = True
        elif hi > c:
            ambiguous.append({"line": records.line[k], "value": v})
    if bad:
        raise ABPError("SCHEMA_TYPE", f"{len(bad)} record date(s) are not YYYY, YYYY-MM or "
                       "YYYY-MM-DD", examples=bad[:10])
    if ambiguous:
        raise ABPError("SPEC_INVALID", f"{len(ambiguous)} record date(s) straddle the cutoff "
                       f"{cutoff}; use full dates or a cutoff at a period boundary",
                       examples=ambiguous[:10])
    return hidden


def lr_statistics(u_p: np.ndarray, u_w: np.ndarray) -> dict:
    """Delta_p, b_w|p and rho_wp for aligned partial/whole predictions."""
    u_p = np.asarray(u_p, dtype=np.float64)
    u_w = np.asarray(u_w, dtype=np.float64)
    if u_p.size < 3:
        raise ABPError("MODEL_NOT_IDENTIFIABLE", "LR statistics need at least 3 focal animals")
    cov = float(np.cov(u_w, u_p)[0, 1])
    var_p = float(np.var(u_p, ddof=1))
    var_w = float(np.var(u_w, ddof=1))
    if var_p <= 0 or var_w <= 0:
        raise ABPError("MODEL_NOT_IDENTIFIABLE", "focal EBVs have zero variance")
    return {"bias_delta_p": float(u_p.mean() - u_w.mean()),
            "dispersion_b_w_p": cov / var_p,
            "rho_wp": cov / np.sqrt(var_w * var_p)}


def cluster_bootstrap(u_p: np.ndarray, u_w: np.ndarray, clusters: list[str], n_boot: int,
                      seed: int) -> dict:
    """Percentile intervals and SEs by resampling whole clusters with replacement."""
    if n_boot == 0:
        return {}
    labels, inv = np.unique(np.asarray(clusters, dtype=object), return_inverse=True)
    members = [np.flatnonzero(inv == c) for c in range(labels.size)]
    rng = np.random.default_rng(seed)
    stats = {"bias_delta_p": [], "dispersion_b_w_p": [], "rho_wp": []}
    failed = 0
    for _ in range(n_boot):
        pick = rng.integers(0, labels.size, labels.size)
        idx = np.concatenate([members[c] for c in pick])
        try:
            s = lr_statistics(u_p[idx], u_w[idx])
        except ABPError:
            failed += 1
            continue
        for k in stats:
            stats[k].append(s[k])
    out = {"replicates": n_boot, "failed_replicates": failed, "n_clusters": int(labels.size),
           "seed": seed, "generator": "numpy PCG64"}
    for k, v in stats.items():
        v = np.array(v)
        out[k] = {"se": float(v.std(ddof=1)), "ci95": [float(np.quantile(v, 0.025)),
                                                       float(np.quantile(v, 0.975))]}
    return out


@dataclass
class _Eval:
    ebv: dict[str, float]
    rel: dict[str, float]


def _evaluate(spec: AnalysisSpec, records: RecordSet, trait: str, structure: GeneticStructure,
              variances: dict[str, float], budget: int) -> _Eval:
    model = build_single_trait(records, trait, spec.data, structure)
    active = [t for t in model.terms if variances.get(t.name, 0.0) > 0.0]
    vc = {k: v for k, v in variances.items() if v > 0.0}
    sol = spec["solver"]
    res = blup(model.y, model.fixed.X, active, vc, method=sol["method"],
               compute_pev=(sol["pev"] == "exact"), tol=sol["tol"], max_iter=sol["max_iter"],
               memory_budget_bytes=budget)
    g = res.terms[model.genetic_term]
    rel = {} if g.reliability is None else dict(zip(g.labels, g.reliability.tolist()))
    return _Eval(dict(zip(g.labels, g.solution.tolist())), rel)


def run_lr(spec: AnalysisSpec, records: RecordSet, trait: str, structure: GeneticStructure,
           ped_data: PedigreeData | None, stage: OutputStage, budget: int,
           structure_for=None) -> dict:
    """Run the partial and whole evaluations and write ``lr_validation.*`` outputs.

    ``structure_for(records)`` rebuilds the relationship structure from a
    record set; it is used when the structure depends on which animals have
    records (``genomic.frequency_source = "training_genotyped"``), so that the
    partial data alone define it.  The same structure is then used for both
    evaluations (LR requires identical models).
    """
    d = spec.data
    v = d["validation"]
    hidden = split_by_cutoff(records, v["cutoff"])
    y = records.traits[trait]
    measured = ~np.isnan(y)
    n_hidden = int((hidden & measured).sum())
    if n_hidden == 0:
        raise ABPError("SPEC_INVALID", f"no {trait} record is dated after {v['cutoff']}")
    partial = RecordSet(records.record_id, records.animal, records.line,
                        {t: np.where(hidden, np.nan, a) for t, a in records.traits.items()},
                        records.factors, records.covariates, records.units, records.dates)
    structure_note = "same structure as the main evaluation"
    if (structure_for is not None and structure.kind in ("genomic", "single_step")
            and d["genomic"]["frequency_source"] == "training_genotyped"):
        structure = structure_for(partial)
        structure_note = "rebuilt from the partial records (training-genotyped frequencies)"
    # variance components: spec values, or REML on the partial data only
    if d["variances"]["mode"] == "known":
        vc = {k: float(x) for k, x in d["variances"]["values"].items()}
        vc_source = "known (spec)"
    else:
        from ..solvers.reml import reml_fit
        pm = build_single_trait(partial, trait, d, structure)
        fit = reml_fit(pm.y, pm.fixed.X, pm.terms, d["reml"], memory_budget_bytes=budget)
        vc = fit.variances
        vc_source = f"REML on the partial data only ({fit.status}, {fit.iterations} iterations)"
    ev_p = _evaluate(spec, partial, trait, structure, vc, budget)
    ev_w = _evaluate(spec, records, trait, structure, vc, budget)
    # focal animals
    animals_partial = {a for a, m, h in zip(records.animal, measured, hidden) if m and not h}
    animals_hidden = {a for a, m, h in zip(records.animal, measured, hidden) if m and h}
    if v["focal"] == "new_records_only":
        focal = sorted(animals_hidden - animals_partial)
        focal_rule = "animals whose only records of the trait are dated after the cutoff"
    else:
        if ped_data is None or not ped_data.birth:
            raise ABPError("SPEC_INVALID", "'born_after_cutoff' needs pedigree birth dates")
        c = parse_date(v["cutoff"])
        focal = sorted(a for a in animals_hidden - animals_partial
                       if a in ped_data.birth and _earliest(ped_data.birth[a]) > c)
        focal_rule = "animals born after the cutoff with records only after it"
    focal = [a for a in focal if a in ev_p.ebv]
    u_p = np.array([ev_p.ebv[a] for a in focal])
    u_w = np.array([ev_w.ebv[a] for a in focal])
    stats = lr_statistics(u_p, u_w)
    if v["bootstrap_cluster"] == "sire" and ped_data is not None:
        P = ped_data.pedigree
        clusters = []
        for a in focal:
            i = P.index_of([a])[0] if P.contains(a) else -1
            clusters.append(P.ids[P.sire[i]] if i >= 0 and P.sire[i] >= 0 else f"self:{a}")
        cluster_rule = "sire family (animals with unknown sire form their own cluster)"
    else:
        clusters = list(focal)
        cluster_rule = "individual animal"
    boot = cluster_bootstrap(u_p, u_w, clusters, v["bootstrap_replicates"], v["seed"])
    rel_p = [ev_p.rel[a] for a in focal if a in ev_p.rel]
    rel_w = [ev_w.rel[a] for a in focal if a in ev_w.rel]
    write_csv(stage.path(f"lr_focal_{trait}.csv"),
              ["animal", "ebv_partial", "ebv_whole", "reliability_partial", "reliability_whole"],
              [[a, ev_p.ebv[a], ev_w.ebv[a], ev_p.rel.get(a), ev_w.rel.get(a)] for a in focal])
    out = {
        "method": "LR (partial vs whole)",
        "trait": trait,
        "cutoff": v["cutoff"],
        "n_records_hidden": n_hidden,
        "n_records_partial": int((measured & ~hidden).sum()),
        "focal_rule": focal_rule,
        "n_focal": len(focal),
        "variance_components": vc,
        "variance_source": vc_source,
        "relationship_structure": structure_note,
        "statistics": stats,
        "expected_under_correct_model": {"bias_delta_p": 0.0, "dispersion_b_w_p": 1.0,
                                         "rho_wp": "acc_p / acc_w (< 1)"},
        "bootstrap": boot,
        "bootstrap_cluster": cluster_rule,
        "model_based_reliability_focal": {
            "mean_partial": float(np.mean(rel_p)) if rel_p else None,
            "mean_whole": float(np.mean(rel_w)) if rel_w else None,
            "sqrt_ratio_partial_over_whole": float(np.sqrt(np.mean(rel_p) / np.mean(rel_w)))
            if rel_p and rel_w else None},
        "not_computed": ["LR population accuracy estimator (published form has a 2019 erratum "
                         "not verifiable in this environment)"],
        "interpretation": ("bias and dispersion compare partial predictions with the whole-data "
                           "predictions of the same animals; rho_wp near 1 shows agreement, not "
                           "accuracy; results are specific to this cutoff and population"),
        "file": f"lr_focal_{trait}.csv",
    }
    log.info("LR validation (%s, cutoff %s): %d focal animals, Delta_p %.4f, b_w|p %.4f, "
             "rho_wp %.4f", trait, v["cutoff"], len(focal), stats["bias_delta_p"],
             stats["dispersion_b_w_p"], stats["rho_wp"])
    return out
