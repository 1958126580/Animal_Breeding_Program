"""End-to-end genetic evaluation workflow (the first vertical slice).

    spec -> inputs (hashed) -> QC -> relationship structure -> model
         -> variance components (known | REML) -> BLUP + PEV + reliability
         -> CSV/JSON outputs -> report -> manifest -> atomic publish

Every step logs to ``run.log``; the manifest records inputs, settings,
environment, diagnostics, output hashes and the final status (one of
``passed``, ``failed``, ``cancelled``).  Outputs are published only when
the whole run succeeds (see :mod:`abp.workflows.outputs`).
"""

from __future__ import annotations

import logging
import time
import traceback
from types import SimpleNamespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..core.model import GeneticStructure, SingleTraitModel, build_single_trait, pedigree_structure
from ..core.spec import AnalysisSpec, load_spec, parent_code_prefix
from ..errors import ABPError
from ..io.tables import read_table, write_csv
from ..qc.pedigree import PedigreeColumns, PedigreeData, load_pedigree, mean_inbreeding_by_generation
from ..qc.phenotype import RecordSet, load_phenotypes
from ..solvers.blup import BLUPResult, blup
from . import manifest as mf
from .outputs import OutputStage, atomic_write_json, atomic_write_text
from .report import render_report

log = logging.getLogger("abp")


@dataclass
class RunOutcome:
    status: str
    out_dir: Path
    manifest: dict
    results: dict | None


# ------------------------------------------------------------------ helpers
def _setup_logging(logfile: Path, console: bool) -> list[logging.Handler]:
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S")
    handlers: list[logging.Handler] = []
    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setFormatter(fmt)
    handlers.append(fh)
    if console:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        handlers.append(ch)
    for h in handlers:
        log.addHandler(h)
    return handlers


def _teardown_logging(handlers: list[logging.Handler]) -> None:
    for h in handlers:
        log.removeHandler(h)
        h.close()


def _check_backend(spec: dict, manifest: dict) -> None:
    b = spec["backend"]
    manifest["backend"] = {"requested": b["device"], "used": "cpu", "fallback": False}
    if b["device"] == "cuda":
        # No CUDA kernels exist in this ABP version (see method registry); never pretend.
        if b["on_unavailable"] == "fallback_cpu":
            log.warning("backend.device = 'cuda' requested but this ABP version has no CUDA kernels; "
                        "falling back to CPU as configured (backend.on_unavailable)")
            manifest["backend"]["fallback"] = True
        else:
            raise ABPError("BACKEND_UNAVAILABLE",
                           "backend.device = 'cuda' requested but this ABP version has no CUDA kernels")


def _heritability(vc: dict[str, float], genetic: str) -> float:
    total = sum(vc.values())
    return vc[genetic] / total


def _structure(spec: AnalysisSpec, ped: PedigreeData | None, manifest: dict,
               records: RecordSet) -> GeneticStructure:
    rel = next(r for r in spec["model"]["random"] if r["kind"] == "additive")["relationship"]
    if spec["metafounders"] is not None:
        from .metafounder_inputs import metafounder_structure
        used = records.used_mask(list(spec["model"]["traits"]))
        with_records = {a for a, u in zip(records.animal, used) if u}
        return metafounder_structure(spec, ped, rel, manifest, with_records)
    if rel == "pedigree" and spec["upg"] is not None:
        from ..core.upg import upg_structure
        u = spec["upg"]
        if not ped.groups.labels:
            raise ABPError("SPEC_INVALID", f"[upg] is declared but no parent code starts with "
                                           f"{u['prefix']!r}")
        return upg_structure(ped.pedigree, ped.groups, u["effect"], u["variance_ratio"])
    if rel == "pedigree":
        return pedigree_structure(ped.pedigree)
    from .genomic_inputs import genomic_structure  # genomic / single-step
    used = records.used_mask(list(spec["model"]["traits"]))
    with_records = {a for a, u in zip(records.animal, used) if u}
    return genomic_structure(spec, ped, rel, manifest, with_records)


# ------------------------------------------------------------ main pipeline
def run_evaluation(spec_path: str | Path, out_dir: str | Path, force: bool = False,
                   resume: bool = False, console: bool = True) -> RunOutcome:
    """Run a complete evaluation described by an analysis spec.

    Raises :class:`ABPError` on any documented failure *after* writing a
    failed manifest into ``<out>.failed-<run_id>``.
    """
    t_start = time.perf_counter()
    run_id = mf.new_run_id()
    spec = load_spec(spec_path)
    manifest: dict[str, Any] = {
        "schema_version": mf.MANIFEST_SCHEMA_VERSION,
        "document_type": "run_manifest",
        "run_id": run_id,
        "status": "running",
        "started_at": mf.utc_now(),
        "finished_at": None,
        "command": None,
        "code": mf.code_provenance(),
        "environment": mf.environment(),
        "spec": {"path": str(spec.path), "sha256": spec.sha256, "effective": spec.data},
        "inputs": [],
        "estimand": spec["analysis"]["task"],
        "genetic_base": spec["analysis"]["genetic_base"],
        "information_cutoff": spec["analysis"]["information_cutoff"],
        "synthetic_data": spec["project"]["synthetic_data"],
        "randomness": {"seed": None, "note": "deterministic computation; no random numbers used"},
        "validation": {"design": "none (full-data evaluation)", "test_phenotypes_used": False},
        "diagnostics": {},
        "outputs": [],
        "limitations": [],
        "error": None,
    }
    stage = OutputStage(Path(out_dir), run_id, force, spec["resources"]["min_free_disk_mb"])
    handlers = _setup_logging(stage.path("run.log"), console)
    try:
        log.info("ABP run %s started; spec %s (sha256 %s)", run_id, spec.path, spec.sha256[:12])
        _check_backend(spec.data, manifest)
        results = _run(spec, stage, manifest, resume)
        manifest["status"] = "passed"
        results["status"] = "passed"
        manifest["execution"] = {"wall_seconds": time.perf_counter() - t_start,
                                 "peak_rss_bytes": mf.peak_rss_bytes(), "exit_code": 0}
        atomic_write_json(stage.path("results.json"), results)
        atomic_write_text(stage.path("report.md"), render_report(results, manifest))
        manifest["outputs"] = _output_hashes(stage.dir, exclude={"manifest.json", "run.log"})
        manifest["finished_at"] = mf.utc_now()
        log.info("run %s passed in %.2f s", run_id, manifest["execution"]["wall_seconds"])
        atomic_write_json(stage.path("manifest.json"), manifest)
        _teardown_logging(handlers)
        final = stage.publish()
        return RunOutcome("passed", final, manifest, results)
    except KeyboardInterrupt:
        err = ABPError("CANCELLED", "run cancelled by the user")
        _fail(stage, manifest, err, "cancelled", handlers, t_start)
        raise err from None
    except ABPError as err:
        _fail(stage, manifest, err, "failed", handlers, t_start)
        raise
    except Exception as exc:  # unexpected: keep the traceback in the log
        log.error("internal error:\n%s", traceback.format_exc())
        err = ABPError("INTERNAL", f"{exc.__class__.__name__}: {exc}")
        _fail(stage, manifest, err, "failed", handlers, t_start)
        raise err from exc


def _fail(stage: OutputStage, manifest: dict, err: ABPError, status: str,
          handlers: list[logging.Handler], t_start: float) -> None:
    log.error("%s", err)
    manifest["status"] = status
    manifest["error"] = err.to_dict()
    manifest["finished_at"] = mf.utc_now()
    manifest["execution"] = {"wall_seconds": time.perf_counter() - t_start,
                             "peak_rss_bytes": mf.peak_rss_bytes(), "exit_code": err.exit_status}
    try:
        atomic_write_json(stage.path("manifest.json"), manifest)
    except Exception:  # pragma: no cover - never mask the original error
        pass
    _teardown_logging(handlers)
    stage.abandon(status)


def _output_hashes(folder: Path, exclude: set[str]) -> list[dict]:
    out = []
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.name not in exclude:
            out.append({"file": p.name, "bytes": p.stat().st_size, "sha256": mf.sha256_path(p)})
    return out


def _run(spec: AnalysisSpec, stage: OutputStage, manifest: dict, resume: bool) -> dict:
    d = spec.data
    data = d["data"]
    missing = set(data["missing_values"])
    unknown_parent = set(data["unknown_parent_values"])
    budget = int(d["resources"]["max_memory_gb"] * 2**30)

    # -- inputs -------------------------------------------------------------
    phe_table = read_table(spec.resolve(data["phenotypes"]), data["delimiter"])
    manifest["inputs"].append({"role": "phenotypes", "path": str(phe_table.path),
                               "sha256": phe_table.sha256, "rows": phe_table.n_rows})
    ped_data: PedigreeData | None = None
    ped_ids: set[str] | None = None
    ped_table = None
    if data["pedigree"] is not None:
        ped_table = read_table(spec.resolve(data["pedigree"]), data["delimiter"])
        manifest["inputs"].append({"role": "pedigree", "path": str(ped_table.path),
                                   "sha256": ped_table.sha256, "rows": ped_table.n_rows})
        pc = data["pedigree_columns"]
        ped_ids = set(ped_table.column(pc["id"])) | (
            (set(ped_table.column(pc["sire"])) | set(ped_table.column(pc["dam"]))) - unknown_parent)
        if parent_code_prefix(d) is not None:
            ped_ids = {a for a in ped_ids if not a.startswith(parent_code_prefix(d))}
    log.info("read %d phenotype rows%s", phe_table.n_rows,
             f" and {ped_table.n_rows} pedigree rows" if ped_table else "")

    # -- QC -----------------------------------------------------------------
    records, phe_qc, to_add = load_phenotypes(phe_table, d, ped_ids)
    atomic_write_json(stage.path("qc_phenotypes.json"), phe_qc.to_dict())
    if ped_table is not None:
        pc = data["pedigree_columns"]
        ped_data = load_pedigree(ped_table, PedigreeColumns(pc["id"], pc["sire"], pc["dam"],
                                                            pc["sex"], pc["birth_date"]),
                                 unknown_parent, missing, extra_founders=to_add,
                                 group_prefix=parent_code_prefix(d))
        ped_data.qc.stats["inbreeding_by_generation"] = mean_inbreeding_by_generation(
            ped_data.pedigree)
        atomic_write_json(stage.path("qc_pedigree.json"), ped_data.qc.to_dict())
        manifest["sample_mapping_sha256"] = mf.sha256_array(list(ped_data.pedigree.ids))
        log.info("pedigree QC passed: %d animals, max generation %d, mean F %.4f (%s kernel)",
                 ped_data.pedigree.n, ped_data.qc.stats["max_generation"],
                 ped_data.qc.stats["mean_inbreeding"], ped_data.pedigree.inbreeding_kernel)
    excluded = phe_qc.excluded
    write_csv(stage.path("qc_excluded_records.csv"),
              ["reason", "line", "record_id", "animal", "trait", "value", "column"],
              [[e.get("reason"), e.get("line"), e.get("record_id"), e.get("animal"),
                e.get("trait"), e.get("value"), e.get("column")] for e in excluded])
    log.info("phenotype QC passed: %d records used, %d excluded (listed in qc_excluded_records.csv)",
             phe_qc.stats["n_records_used"], len(excluded))

    if d["variances"]["mode"] == "bayes" and d["bayes"]["method"] != "threshold":
        return _run_bayes(spec, records, phe_qc, ped_data, stage, manifest)

    # -- relationship structure --------------------------------------------
    structure = _structure(spec, ped_data, manifest, records)
    manifest["relationship"] = {"kind": structure.kind, "n": len(structure.labels),
                                **{k: v for k, v in structure.meta.items() if _is_small(v)}}

    if structure.kind == "genomic":
        _handle_ungenotyped(records, structure, phe_qc, d)
        atomic_write_json(stage.path("qc_phenotypes.json"), phe_qc.to_dict())
        write_csv(stage.path("qc_excluded_records.csv"),
                  ["reason", "line", "record_id", "animal", "trait", "value", "column"],
                  [[e.get("reason"), e.get("line"), e.get("record_id"), e.get("animal"),
                    e.get("trait"), e.get("value"), e.get("column")] for e in phe_qc.excluded])
    geno_qc = manifest.pop("_qc_sections", {}).get("genotypes")
    if geno_qc is not None:
        atomic_write_json(stage.path("qc_genotypes.json"), geno_qc)
    results: dict[str, Any] = {
        "abp_version": manifest["code"]["abp_version"],
        "run_id": manifest["run_id"],
        "project": d["project"],
        "analysis": d["analysis"],
        "relationship": manifest["relationship"],
        "qc": {"phenotypes": phe_qc.to_dict(),
               "pedigree": ped_data.qc.to_dict() if ped_data else None,
               "genotypes": geno_qc},
        "traits": {},
        "limitations": _limitations(d, structure),
    }
    traits = d["model"]["traits"]
    if len(traits) > 1:
        from .multitrait import run_multitrait
        results["traits"], state = run_multitrait(spec, records, structure, ped_data, stage,
                                                  manifest, budget)
    else:
        out, state = _run_single_trait(spec, records, traits[0], structure, ped_data, stage,
                                       manifest, budget, resume)
        results["traits"][traits[0]] = out
        if d.get("validation"):
            from .validation_lr import run_lr
            lr = run_lr(spec, records, traits[0], structure, ped_data, stage, budget,
                        structure_for=lambda recs: _structure(spec, ped_data, {"inputs": []},
                                                              recs))
            atomic_write_json(stage.path("lr_validation.json"), lr)
            results["validation"] = lr
            manifest["validation"] = {"design": "LR forward-in-time", "cutoff": lr["cutoff"],
                                      "test_phenotypes_used": False,
                                      "note": "hidden records are never read by the partial "
                                              "evaluation; variances from spec or partial REML"}
    if d.get("index"):
        from .index_outputs import write_index
        results["index"] = write_index(spec, state, stage)
    manifest["limitations"] = results["limitations"]
    return results


def _handle_ungenotyped(records: RecordSet, structure: GeneticStructure, phe_qc, d: dict) -> None:
    """GBLUP: records of animals absent from G are an error unless explicitly excluded."""
    items = []
    for t in d["model"]["traits"]:
        vals = records.traits[t]
        for k in np.flatnonzero(~np.isnan(vals)):
            if records.animal[k] not in structure.index:
                items.append({"line": records.line[k], "record_id": records.record_id[k],
                              "animal": records.animal[k], "trait": t, "value": float(vals[k])})
    if not items:
        return
    if d["qc"]["ungenotyped_records"] == "exclude":
        phe_qc.add("PHE-UNGENOTYPED", "quarantine",
                   "records of animals without (QC-passed) genotypes were excluded from GBLUP",
                   items)
        for it in items:
            records.traits[it["trait"]][records.record_id.index(it["record_id"])] = np.nan
        phe_qc.excluded.extend({**it, "reason": "PHE-UNGENOTYPED"} for it in items)
        phe_qc.stats["n_excluded"] = len(phe_qc.excluded)
        phe_qc.stats["n_records_used"] = int(records.used_mask(d["model"]["traits"]).sum())
        log.info("GBLUP: %d records of non-genotyped animals excluded (listed)", len(items))
    else:
        raise ABPError("PHENOTYPE_UNKNOWN_ANIMAL",
                       f"{len(items)} record(s) belong to animals without QC-passed genotypes; "
                       "use relationship = 'single_step' to include them, or set "
                       "qc.ungenotyped_records = 'exclude' to analyse genotyped animals only",
                       n=len(items), examples=items[:10])


def _run_bayes(spec: AnalysisSpec, records: RecordSet, phe_qc, ped_data: PedigreeData | None,
               stage: OutputStage, manifest: dict) -> dict:
    """Bayesian marker-regression path (no relationship matrix is built)."""
    from .bayes_eval import run_bayes_trait
    d = spec.data
    trait = d["model"]["traits"][0]
    out, state = run_bayes_trait(spec, records, trait, ped_data, stage, manifest, phe_qc)
    atomic_write_json(stage.path("qc_phenotypes.json"), phe_qc.to_dict())
    write_csv(stage.path("qc_excluded_records.csv"),
              ["reason", "line", "record_id", "animal", "trait", "value", "column"],
              [[e.get("reason"), e.get("line"), e.get("record_id"), e.get("animal"),
                e.get("trait"), e.get("value"), e.get("column")] for e in phe_qc.excluded])
    geno_qc = manifest.pop("_qc_sections", {}).get("genotypes")
    if geno_qc is not None:
        atomic_write_json(stage.path("qc_genotypes.json"), geno_qc)
    manifest["relationship"] = {"kind": "marker_model", "n": out["n_animals_evaluated"],
                                "method": d["bayes"]["method"],
                                "frequency_source": d["genomic"]["frequency_source"]}
    lim = _limitations(d, SimpleNamespace(kind="genomic"))
    lim.append("Bayesian results: posterior means and SDs conditional on the stated priors; "
               "posterior SDs are not PEV-based reliabilities.")
    results: dict[str, Any] = {
        "abp_version": manifest["code"]["abp_version"], "run_id": manifest["run_id"],
        "project": d["project"], "analysis": d["analysis"],
        "relationship": manifest["relationship"],
        "qc": {"phenotypes": phe_qc.to_dict(),
               "pedigree": ped_data.qc.to_dict() if ped_data else None, "genotypes": geno_qc},
        "traits": {trait: out}, "limitations": lim,
    }
    if d.get("index"):
        from .index_outputs import write_index
        results["index"] = write_index(spec, state, stage)
    manifest["limitations"] = lim
    return results


def _is_small(v: Any) -> bool:
    return not isinstance(v, np.ndarray)


def _limitations(d: dict, structure: GeneticStructure) -> list[str]:
    lim = [
        "Estimand: additive breeding values relative to the declared genetic base; they are "
        "not total genetic values and not phenotype predictions.",
        "PEV and reliabilities are conditional on the variance components (their estimation "
        "error is not propagated).",
        "Model-based reliabilities assume the model is correct; they are not validated "
        "prediction accuracies.",
    ]
    if structure.kind == "pedigree_upg":
        m = structure.meta
        if m["upg_effect"] == "random":
            lim.append(f"Unknown-parent groups ({m['n_groups']}) are random with the declared "
                       f"ratio sigma_g^2/sigma_a^2 = {m['upg_variance_ratio']} (not estimated); "
                       "EBVs include the group contributions (u* = u + Qg).")
        else:
            lim.append(f"Unknown-parent groups ({m['n_groups']}) are fixed effects: EBVs "
                       "include the estimated group contributions, PEV includes their "
                       "estimation error, and reliabilities are not defined (not reported).")
        lim.append("Unknown parents without a group code, and all groups, carry no "
                   "relationships or inbreeding among themselves (no metafounders).")
    elif structure.kind in ("pedigree_mf", "single_step_mf"):
        m = structure.meta
        lim.append(f"Genetic base: {m['n_groups']} metafounder(s) {m['metafounders']} with "
                   f"relationships Gamma = {np.round(np.array(m['gamma']), 4).tolist()} "
                   f"({m['gamma_provenance']}). EBVs, variances and reliabilities refer to this "
                   "base and are not directly comparable with an evaluation that treats unknown "
                   "parents as unrelated; variance components should be estimated under the "
                   "same model.")
        lim.append("Gamma is treated as known: its estimation error is not propagated into "
                   "EBVs, PEV or reliabilities.")
    else:
        lim.append("No unknown-parent groups or metafounders: all unknown parents are treated "
                   "as unrelated, non-inbred base animals.")
    if any(t["type"] == "categorical" and t["name"] in d["model"]["traits"] for t in d["traits"]):
        lim.append("Categorical traits: EBVs are on the liability scale of a threshold (probit) "
                   "model with residual variance 1; PEV and reliabilities are Laplace "
                   "approximations at the posterior mode."
                   + (" Liability variances were estimated by Laplace-approximate REML, which "
                      "is biased when there is little information per animal (see the "
                      "validation report); their estimation error is not propagated."
                      if d["variances"]["mode"] == "reml" else ""))
    if d["project"]["synthetic_data"]:
        lim.insert(0, "SYNTHETIC DATA: results illustrate the method only and carry no "
                      "information about any real population.")
    if structure.kind in ("genomic", "single_step"):
        lim.append("Genomic relationships depend on the declared allele-frequency source and "
                   "marker coding (see relationship section).")
    lim.append("No external validation (G5) has been run for this analysis.")
    return lim


def _run_single_trait(spec: AnalysisSpec, records: RecordSet, trait: str,
                      structure: GeneticStructure, ped_data: PedigreeData | None,
                      stage: OutputStage, manifest: dict, budget: int, resume: bool) -> dict:
    d = spec.data
    model = build_single_trait(records, trait, d, structure)
    if next(t["type"] for t in d["traits"] if t["name"] == trait) == "categorical":
        return _run_threshold_trait(spec, model, structure, ped_data, stage, manifest, budget)
    log.info("%s: %d records, %d fixed columns kept (%d constrained), random terms %s",
             trait, model.y.size, model.fixed.rank, len(model.fixed.constrained_labels),
             [t.name for t in model.terms])
    upg = _upg_check(model, structure)
    vmode = d["variances"]["mode"]
    reml_info = None
    fit = None
    if vmode == "known":
        vc = {k: float(v) for k, v in d["variances"]["values"].items()}
    else:
        from ..solvers.reml import reml_fit
        checkpoint = stage.final.with_name(stage.final.name + f".reml-checkpoint-{trait}.json")
        fit = reml_fit(model.y, model.fixed.X, model.terms, d["reml"],
                       memory_budget_bytes=budget, checkpoint_path=checkpoint, resume=resume,
                       fingerprint=_fingerprint(spec, manifest, trait))
        vc = fit.variances
        reml_info = fit.to_dict()
        log.info("%s: REML %s after %d iterations; logL %.6f; variances %s", trait,
                 fit.status, fit.iterations, fit.loglik, {k: round(v, 6) for k, v in vc.items()})
        if checkpoint.exists():
            checkpoint.unlink()
        if model.genetic_term in fit.boundary:
            raise ABPError("MODEL_NOT_IDENTIFIABLE",
                           f"{trait}: the additive genetic variance is estimated at zero "
                           "(boundary); no genetic ranking can be issued from these data",
                           trait=trait, reml=reml_info)
    sol = d["solver"]
    # Terms whose variance is exactly zero (REML boundary) are removed from the MME.
    active_terms = [t for t in model.terms if vc.get(t.name, 0.0) > 0.0]
    vc_active = {k: v for k, v in vc.items() if v > 0.0}
    model.terms = active_terms
    res = blup(model.y, model.fixed.X, active_terms, vc_active, method=sol["method"],
               compute_pev=(sol["pev"] == "exact"), tol=sol["tol"], max_iter=sol["max_iter"],
               memory_budget_bytes=budget, factorization=sol["factorization"])
    s = res.solve
    log.info("%s: solved %d equations with %s (%s); relative residual %.2e; %.2f s", trait,
             s.solution.size, s.method, s.selection_reason, s.rel_residual, s.wall_seconds)
    vc_extra = None
    if (fit is not None and fit.cov is not None and gen_pev_available(res, model.genetic_term)
            and model.genetic_term in fit.cov_names):
        from ..solvers.vc_uncertainty import kackar_harville_delta
        delta = kackar_harville_delta(model.y, model.fixed.X, active_terms, vc_active, fit.cov,
                                      fit.cov_names, model.genetic_term, method=sol["method"],
                                      memory_budget_bytes=budget,
                                      factorization=sol["factorization"])
        g = res.terms[model.genetic_term]
        prior = vc_active[model.genetic_term] * np.asarray(model.terms[
            [t.name for t in model.terms].index(model.genetic_term)].k_diag, dtype=np.float64)
        pev_t = g.pev + delta
        vc_extra = {"pev_incl_vc_uncertainty": pev_t,
                    "reliability_incl_vc_uncertainty": np.clip(1.0 - pev_t / prior, 0.0, 1.0)}
        log.info("%s: PEV including variance-estimation uncertainty (Kackar-Harville): mean "
                 "increase %.4g (%.1f%%)", trait, float(delta.mean()),
                 100.0 * float(delta.mean() / g.pev.mean()))
    groups = None
    mf_info = None
    contrast = None
    if structure.kind in ("pedigree_mf", "single_step_mf"):
        from .metafounder_inputs import base_contrast
        contrast = base_contrast(res, model.genetic_term, structure,
                                 vc_active[model.genetic_term], d["metafounders"])
    if upg is not None or structure.kind in ("pedigree_mf", "single_step_mf"):
        res.terms[model.genetic_term], groups = _split_upg(res.terms[model.genetic_term],
                                                           structure)
    if vc_extra is not None and groups is not None:          # keep animal equations only
        n_an = structure.meta["n_animals"]
        vc_extra = {k: v[:n_an] for k, v in vc_extra.items()}
    files = _write_single_trait_outputs(stage, model, res, ped_data, contrast, vc_extra)
    if upg is not None:
        files["upg_solutions"] = _write_upg(stage, trait, groups, structure)
        upg["file"] = files["upg_solutions"]
    elif groups is not None:
        files["metafounder_solutions"] = _write_metafounders(stage, trait, groups, structure)
        m = structure.meta
        mf_info = {"metafounders": m["metafounders"], "gamma": m["gamma"],
                   "reference": contrast["reference"],
                   "reliability_vs_base_summary": contrast["summary"],
                   "gamma_source": m["gamma_source"], "gamma_provenance": m["gamma_provenance"],
                   "file": files["metafounder_solutions"]}
    gen = res.terms[model.genetic_term]
    top_n = d["output"]["top_n"]
    order = np.argsort(-gen.solution, kind="stable")[:top_n]
    sex = ped_data.sex if ped_data else {}
    top = [{"rank": r + 1, "animal": gen.labels[k], "sex": sex.get(gen.labels[k], "U"),
            "ebv": float(gen.solution[k]),
            "reliability": None if gen.reliability is None else float(gen.reliability[k]),
            "sep": None if gen.pev is None else float(np.sqrt(gen.pev[k])),
            "n_records": model.n_records_per_animal.get(gen.labels[k], 0)}
           for r, k in enumerate(order)]
    fixed_rows = _fixed_rows(model, res)
    out = {
        "unit": model.unit,
        "n_records": int(model.y.size),
        "n_animals_evaluated": len(gen.labels),
        "variance_source": vmode,
        "variance_components": vc,
        "heritability": _heritability(vc, model.genetic_term),
        "reml": reml_info,
        "genetic_term": model.genetic_term,
        "solver": {"method": s.method, "selection_reason": s.selection_reason,
                   "n_equations": int(s.solution.size), "relative_residual": s.rel_residual,
                   "iterations": s.iterations, "wall_seconds": s.wall_seconds,
                   "pev": d["solver"]["pev"]},
        "fixed_effects": fixed_rows,
        "n_fixed_constrained": len(model.fixed.constrained_labels),
        "upg": upg,
        "metafounders": mf_info,
        "top": top,
        "reliability_summary": None if gen.reliability is None else {
            "mean": float(gen.reliability.mean()), "min": float(gen.reliability.min()),
            "max": float(gen.reliability.max()), "n_rounding_clamped": gen.n_reliability_clamped},
        "ebv_summary": {"mean": float(gen.solution.mean()), "sd": float(gen.solution.std()),
                        "min": float(gen.solution.min()), "max": float(gen.solution.max())},
        "files": files,
    }
    manifest["diagnostics"][trait] = {"solver": out["solver"], "reml": reml_info,
                                      "fixed_constrained": [list(x) for x in
                                                            model.fixed.constrained_labels]}
    if upg is not None:
        manifest["diagnostics"][trait]["upg"] = upg
    from .multitrait import EvalState
    n_a = len(gen.labels)
    k_diag = np.asarray(structure.k_diag)[:n_a]
    pev_idx = None if gen.pev is None or not np.all(np.isfinite(k_diag)) else gen.pev[:, None, None]
    if contrast is not None:          # metafounders: index on EBVs relative to the base
        pev_c = None if contrast["pev"] is None else contrast["pev"][:, None, None]
        state = EvalState(tuple(gen.labels), [trait], contrast["ebv"][:, None], pev_c,
                          np.array([[vc[model.genetic_term]]]), contrast["k_factor"], False, sex)
        return out, state
    state = EvalState(tuple(gen.labels), [trait], gen.solution[:, None], pev_idx,
                      np.array([[vc[model.genetic_term]]]), k_diag, False, sex)
    return out, state


def _run_threshold_trait(spec: AnalysisSpec, model: SingleTraitModel,
                         structure: GeneticStructure, ped_data: PedigreeData | None,
                         stage: OutputStage, manifest: dict, budget: int) -> tuple[dict, Any]:
    """Ordered categorical trait: threshold (probit) model on the liability scale."""
    from types import SimpleNamespace

    from ..solvers.threshold import threshold_blup, threshold_laplace_reml
    from .multitrait import EvalState
    d = spec.data
    trait = model.trait
    sol = d["solver"]
    reml_info = None
    if d["variances"]["mode"] == "reml":
        rc = d["reml"]
        fit = threshold_laplace_reml(model.y, model.fixed.X, model.terms,
                                     intercept=d["model"]["intercept"], start=rc["start"],
                                     tol=max(float(rc["tol"]), 1e-5),
                                     max_eval=max(int(rc["max_iter"]), 200),
                                     memory_budget_bytes=budget)
        vc = dict(fit.variances)
        reml_info = {"status": fit.status, "iterations": fit.evaluations, "loglik": fit.loglik,
                     "se": None, "heritability_se": None,
                     "method": "laplace_approximate_reml", "note": fit.note}
        variance_source = ("reml (Laplace-approximate, liability scale; residual variance "
                           "fixed at 1)")
        log.info("%s: Laplace-approximate REML on the liability scale: %s (%d evaluations)",
                 trait, {k: round(v, 5) for k, v in vc.items()}, fit.evaluations)
    elif d["variances"]["mode"] == "bayes":
        gibbs = _threshold_gibbs_step(spec, model, stage, manifest)
        vc = {k: v["mean"] for k, v in gibbs.variances.items() if k != "h2_liability"}
        vc["residual"] = 1.0
        variance_source = ("bayes (threshold-model Gibbs sampler; posterior means, liability "
                           "scale; residual variance fixed at 1)")
    else:
        vc = {k: float(v) for k, v in d["variances"]["values"].items()}
        variance_source = "known (liability scale; residual variance fixed at 1)"
    if d["variances"]["mode"] == "bayes":
        res = SimpleNamespace(terms=gibbs.terms, fixed_solution=gibbs.fixed_mean,
                              thresholds=gibbs.thresholds_mean, categories=gibbs.categories,
                              iterations=gibbs.iterations, n_equations=gibbs.n_equations,
                              log_posterior=None, solver="gibbs, sparse LDL' block draws")
    else:
        res = threshold_blup(model.y, model.fixed.X, model.terms, vc,
                             intercept=d["model"]["intercept"],
                             compute_pev=(sol["pev"] == "exact"), memory_budget_bytes=budget)
        log.info("%s: threshold model, %d categories, converged in %d Newton iterations "
                 "(largest last step %.1e); thresholds %s", trait, res.categories.size,
                 res.iterations, res.max_step, np.round(res.thresholds, 4).tolist())
    files = _write_single_trait_outputs(stage, model, SimpleNamespace(
        terms=res.terms, fixed_solution=res.fixed_solution), ped_data)
    name = f"thresholds_{trait}.csv"
    write_csv(stage.path(name), ["threshold", "between_category", "and_category", "value", "status"],
              [[k + 1, float(res.categories[k]), float(res.categories[k + 1]), float(tv),
                "fixed_at_zero_for_identifiability" if (k == 0 and d["model"]["intercept"])
                else "estimated"] for k, tv in enumerate(res.thresholds)])
    files["thresholds"] = name
    gen = res.terms[model.genetic_term]
    order = np.argsort(-gen.solution, kind="stable")[:d["output"]["top_n"]]
    sex = ped_data.sex if ped_data else {}
    total = sum(vc.values())
    out = {
        "unit": "liability (residual SD = 1)",
        "n_records": int(model.y.size),
        "n_animals_evaluated": len(gen.labels),
        "variance_source": variance_source,
        "variance_components": vc,
        "heritability": (gibbs.variances["h2_liability"]["mean"]
                         if d["variances"]["mode"] == "bayes" else vc[model.genetic_term] / total),
        "reml": reml_info,
        "genetic_term": model.genetic_term,
        "solver": ({"method": f"threshold model, Newton-Raphson ({res.solver})",
                    "selection_reason": "categorical trait", "n_equations": res.n_equations,
                    "relative_residual": None, "iterations": res.iterations,
                    "wall_seconds": None,
                    "pev": ("laplace_approximation" if sol["pev"] == "exact" else "none")}
                   if d["variances"]["mode"] != "bayes" else
                   {"method": f"threshold model, {res.solver}",
                    "selection_reason": "categorical trait, variances.mode = bayes",
                    "n_equations": res.n_equations, "relative_residual": None,
                    "iterations": res.iterations, "wall_seconds": gibbs.wall_seconds,
                    "pev": "posterior_variance"}),
        "fixed_effects": _fixed_rows(model, SimpleNamespace(fixed_solution=res.fixed_solution)),
        "n_fixed_constrained": len(model.fixed.constrained_labels),
        "upg": None,
        "metafounders": None,
        "threshold_model": {"categories": res.categories.tolist(),
                            "thresholds": res.thresholds.tolist(),
                            "log_posterior": res.log_posterior, "file": name,
                            "note": "EBVs, PEV and reliabilities are on the liability scale; "
                                    "PEV is a Laplace approximation (inverse Hessian at the "
                                    "posterior mode)"},
        "top": [{"rank": r + 1, "animal": gen.labels[k], "sex": sex.get(gen.labels[k], "U"),
                 "ebv": float(gen.solution[k]),
                 "reliability": None if gen.reliability is None else float(gen.reliability[k]),
                 "sep": None if gen.pev is None else float(np.sqrt(gen.pev[k])),
                 "n_records": model.n_records_per_animal.get(gen.labels[k], 0)}
                for r, k in enumerate(order)],
        "reliability_summary": None if gen.reliability is None else {
            "mean": float(gen.reliability.mean()), "min": float(gen.reliability.min()),
            "max": float(gen.reliability.max()), "n_rounding_clamped": gen.n_reliability_clamped},
        "ebv_summary": {"mean": float(gen.solution.mean()), "sd": float(gen.solution.std()),
                        "min": float(gen.solution.min()), "max": float(gen.solution.max())},
        "files": files,
    }
    manifest["diagnostics"].setdefault(trait, {}).update(
        {"solver": out["solver"], "reml": reml_info, "threshold_model": out["threshold_model"]})
    if d["variances"]["mode"] == "bayes":
        out["bayes"] = {"method": "threshold", "converged": gibbs.converged,
                        "iterations": gibbs.iterations, "chains": d["bayes"]["chains"],
                        "summaries": {k: {m: v[m] for m in ("mean", "sd", "q05", "q95", "rhat",
                                                            "ess_bulk", "ess_tail", "mcse_mean")}
                                      for k, v in gibbs.summaries.items()},
                        "variances": gibbs.variances, "ebv_diagnostics": gibbs.ebv_diagnostics}
        out["files"]["diagnostics"] = f"mcmc_diagnostics_{trait}.json"
        out["files"]["trace"] = f"mcmc_trace_{trait}.csv"
        out["threshold_model"]["note"] = (
            "EBVs are posterior means and PEV posterior variances on the liability scale "
            "(they include the uncertainty of the variances); reliabilities use the "
            "posterior mean of the genetic variance")
    k_diag = np.asarray(structure.k_diag)[:len(gen.labels)]
    state = EvalState(tuple(gen.labels), [trait], gen.solution[:, None],
                      None if gen.pev is None else gen.pev[:, None, None],
                      np.array([[vc[model.genetic_term]]]), k_diag, False, sex)
    return out, state


def _threshold_gibbs_step(spec: AnalysisSpec, model: SingleTraitModel, stage: OutputStage,
                          manifest: dict):
    """Gibbs sampler for a categorical trait: diagnostics and traces are written
    before the convergence decision (a withheld result keeps its evidence)."""
    from ..solvers.threshold_gibbs import ThresholdGibbsConfig, threshold_gibbs
    d = spec.data
    trait = model.trait
    b = d["bayes"]
    cfg = ThresholdGibbsConfig(chains=b["chains"], iterations=b["iterations"],
                               burn_in=b["burn_in"], thin=b["thin"], seed=b["seed"],
                               rhat_max=b["rhat_max"], ess_min=b["ess_min"],
                               max_iterations=b["max_iterations"])
    log.info("%s: threshold-model Gibbs sampler, %d chains, %d iterations (up to %d)", trait,
             cfg.chains, cfg.iterations, cfg.max_iterations)
    g = threshold_gibbs(model.y, model.fixed.X, model.terms, d["model"]["intercept"], cfg,
                        genetic_term=model.genetic_term)
    diag = {"method": "threshold model, Gibbs sampler (Sorensen et al. 1995; Cowles 1996 "
                      "threshold step; block draws of the location effects)",
            "converged": g.converged, "iterations": g.iterations,
            "draws_per_chain": g.draws_per_chain, "chains": cfg.chains, "thin": cfg.thin,
            "burn_in": cfg.burn_in, "chain_seeds": g.seeds, "master_seed": cfg.seed,
            "wall_seconds": g.wall_seconds,
            "criteria": {"rhat_max": cfg.rhat_max, "ess_min": cfg.ess_min},
            "summaries": g.summaries, "ebv_diagnostics": g.ebv_diagnostics,
            "variances": g.variances,
            "threshold_step": {"acceptance_after_burn_in": g.acceptance,
                               "proposal_sd": g.proposal_sd},
            "priors": {"variances": "uniform on (0, inf) (scaled inverse chi-square, nu = -2, "
                                    "s2 = 0)", "fixed_effects": "flat",
                       "thresholds": "flat subject to ordering"},
            "trace_file": f"mcmc_trace_{trait}.csv"}
    atomic_write_json(stage.path(f"mcmc_diagnostics_{trait}.json"), diag)
    names = list(g.traces)
    write_csv(stage.path(f"mcmc_trace_{trait}.csv"), ["chain", "iteration"] + names,
              [[c + 1, cfg.burn_in + (k + 1) * cfg.thin] + [float(g.traces[q][c, k])
                                                            for q in names]
               for c in range(cfg.chains) for k in range(g.draws_per_chain)])
    manifest["diagnostics"].setdefault(trait, {})["mcmc"] = {
        k: diag[k] for k in ("method", "converged", "iterations", "chains", "chain_seeds",
                             "criteria")}
    manifest["randomness"] = {"seed": cfg.seed, "generator": "numpy PCG64 via SeedSequence.spawn",
                              "chain_seeds": g.seeds}
    if not g.converged:
        raise ABPError("MCMC_NOT_CONVERGED",
                       f"{trait}: threshold-model Gibbs diagnostics not met after "
                       f"{g.iterations} iterations (criteria R-hat < {cfg.rhat_max}, "
                       f"ESS >= {cfg.ess_min})",
                       summaries={k: {m: v[m] for m in ("rhat", "ess_bulk", "ess_tail")}
                                  for k, v in g.summaries.items()}, ebv=g.ebv_diagnostics)
    log.info("%s: Gibbs sampler converged after %d iterations; liability variances %s", trait,
             g.iterations, {k: round(v["mean"], 4) for k, v in g.variances.items()})
    return g


def _upg_check(model: SingleTraitModel, structure: GeneticStructure) -> dict | None:
    """UPG bookkeeping; fixed groups must be estimable jointly with the fixed effects."""
    if structure.kind != "pedigree_upg":
        return None
    m = structure.meta
    info = {"effect": m["upg_effect"], "variance_ratio": m["upg_variance_ratio"],
            "n_groups": m["n_groups"], "estimability": None}
    if m["upg_effect"] == "fixed":
        from ..core.upg import check_fixed_groups_estimable
        Z = next(t for t in model.terms if t.genetic).Z[:, :m["n_animals"]]
        info["estimability"] = check_fixed_groups_estimable(
            model.fixed.X, np.asarray(Z @ m["group_fractions"]), tuple(m["groups"]))
    return info


def _split_upg(tr, structure: GeneticStructure):
    """Separate animal equations (u*) from group equations (g)."""
    from ..solvers.blup import TermResult
    n = structure.meta["n_animals"]

    def part(sl):
        rel = None if tr.reliability is None else tr.reliability[sl]
        if rel is not None and not np.any(np.isfinite(rel)):
            rel = None                              # fixed groups: undefined
        return TermResult(tr.name, tuple(tr.labels[sl]), tr.solution[sl],
                          None if tr.pev is None else tr.pev[sl], rel,
                          tr.n_reliability_clamped if sl.start == 0 else 0)
    return part(slice(0, n)), part(slice(n, None))


def _write_upg(stage: OutputStage, trait: str, groups, structure: GeneticStructure) -> str:
    Q = structure.meta["group_fractions"]
    name = f"upg_solutions_{trait}.csv"
    rows = []
    for k, g in enumerate(groups.labels):
        pev = None if groups.pev is None else float(groups.pev[k])
        rows.append([g, structure.meta["upg_effect"], float(groups.solution[k]), pev,
                     None if pev is None else float(np.sqrt(pev)),
                     None if groups.reliability is None else float(groups.reliability[k]),
                     int(np.count_nonzero(Q[:, k] > 0)), float(Q[:, k].sum())])
    write_csv(stage.path(name), ["group", "effect", "solution", "pev", "sep", "reliability",
                                 "n_animals_with_contribution", "sum_gene_fraction"], rows)
    return name


def _write_metafounders(stage: OutputStage, trait: str, groups,
                        structure: GeneticStructure) -> str:
    """Metafounder solutions (genetic level of each base population) and their PEV."""
    Q = structure.meta["group_fractions"]
    gamma = np.array(structure.meta["gamma"])
    name = f"metafounder_solutions_{trait}.csv"
    rows = []
    for k, g in enumerate(groups.labels):
        pev = None if groups.pev is None else float(groups.pev[k])
        rows.append([g, float(gamma[k, k]), float(groups.solution[k]), pev,
                     None if pev is None else float(np.sqrt(pev)),
                     None if groups.reliability is None else float(groups.reliability[k]),
                     int(np.count_nonzero(Q[:, k] > 0)), float(Q[:, k].sum())])
    write_csv(stage.path(name), ["metafounder", "gamma_self", "solution", "pev", "sep",
                                 "reliability", "n_animals_with_contribution",
                                 "sum_gene_fraction"], rows)
    return name


def _fingerprint(spec: AnalysisSpec, manifest: dict, trait: str) -> str:
    return mf.sha256_array([spec.sha256, trait] + [i["sha256"] for i in manifest["inputs"]])


def _fixed_rows(model: SingleTraitModel, res: BLUPResult) -> list[dict]:
    rows, k = [], 0
    for (term, level), kept in zip(model.fixed.labels, model.fixed.kept):
        if kept:
            rows.append({"term": term, "level": level, "solution": float(res.fixed_solution[k]),
                         "status": "estimated_under_constraints"})
            k += 1
        else:
            rows.append({"term": term, "level": level, "solution": 0.0,
                         "status": "constrained_to_zero"})
    return rows


def gen_pev_available(res, term: str) -> bool:
    return res.terms[term].pev is not None


def _write_single_trait_outputs(stage: OutputStage, model: SingleTraitModel, res: BLUPResult,
                                ped_data: PedigreeData | None, contrast: dict | None = None,
                                extra: dict | None = None) -> dict:
    trait = model.trait
    gen = res.terms[model.genetic_term]
    files = {}
    ped = ped_data.pedigree if ped_data else None
    F = ped.inbreeding() if ped is not None else None
    if "inbreeding_mf_base" in model.structure.meta:      # F relative to the metafounder base
        F = model.structure.meta["inbreeding_mf_base"]
    name = f"ebv_{trait}.csv"
    rows = []
    for k, a in enumerate(gen.labels):
        if ped is not None and ped.contains(a):
            i = ped.index_of([a])[0]
            s = ped.ids[ped.sire[i]] if ped.sire[i] >= 0 else ""
            dm = ped.ids[ped.dam[i]] if ped.dam[i] >= 0 else ""
            f_i, g_i, sx = float(F[i]), int(ped.generation[i]), ped_data.sex.get(a, "U")
        else:
            s = dm = sx = ""
            f_i = g_i = None
        pev = None if gen.pev is None else float(gen.pev[k])
        rel = None if gen.reliability is None else float(gen.reliability[k])
        rows.append([a, s, dm, sx, g_i, f_i, model.n_records_per_animal.get(a, 0),
                     float(gen.solution[k]), pev, None if pev is None else float(np.sqrt(pev)),
                     rel, None if rel is None else float(np.sqrt(rel))])
        if contrast is not None:
            rows[-1] += [float(contrast["ebv"][k]),
                         None if contrast["pev"] is None else float(contrast["pev"][k]),
                         None if contrast["reliability"] is None
                         else float(contrast["reliability"][k])]
        if extra is not None:
            rows[-1] += [float(v[k]) for v in extra.values()]
    header = ["animal", "sire", "dam", "sex", "generation", "inbreeding",
              "n_records", "ebv", "pev", "sep", "reliability", "accuracy"]
    if contrast is not None:
        header += ["ebv_vs_base", "pev_vs_base", "reliability_vs_base"]
    if extra is not None:
        header += list(extra)
    write_csv(stage.path(name), header, rows)
    files["ebv"] = name
    name = f"fixed_effects_{trait}.csv"
    write_csv(stage.path(name), ["term", "level", "solution", "status"],
              [[r["term"], r["level"], r["solution"], r["status"]] for r in _fixed_rows(model, res)])
    files["fixed_effects"] = name
    for t in model.terms:
        if t.genetic:
            continue
        name = f"random_{t.name}_{trait}.csv"
        tr = res.terms[t.name]
        write_csv(stage.path(name), ["level", "solution", "pev"],
                  [[lv, float(tr.solution[k]), None if tr.pev is None else float(tr.pev[k])]
                   for k, lv in enumerate(tr.labels)])
        files[f"random_{t.name}"] = name
    return files
