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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..core.model import GeneticStructure, SingleTraitModel, build_single_trait, pedigree_structure
from ..core.spec import AnalysisSpec, load_spec
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
        # No CUDA kernels exist in ABP 0.1 (see method registry); never pretend.
        if b["on_unavailable"] == "fallback_cpu":
            log.warning("backend.device = 'cuda' requested but ABP 0.1 has no CUDA kernels; "
                        "falling back to CPU as configured (backend.on_unavailable)")
            manifest["backend"]["fallback"] = True
        else:
            raise ABPError("BACKEND_UNAVAILABLE",
                           "backend.device = 'cuda' requested but ABP 0.1 has no CUDA kernels")


def _heritability(vc: dict[str, float], genetic: str) -> float:
    total = sum(vc.values())
    return vc[genetic] / total


def _structure(spec: AnalysisSpec, ped: PedigreeData | None, manifest: dict) -> GeneticStructure:
    rel = next(r for r in spec["model"]["random"] if r["kind"] == "additive")["relationship"]
    if rel == "pedigree":
        return pedigree_structure(ped.pedigree)
    from .genomic_inputs import genomic_structure  # genomic / single-step
    return genomic_structure(spec, ped, rel, manifest)


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
    log.info("read %d phenotype rows%s", phe_table.n_rows,
             f" and {ped_table.n_rows} pedigree rows" if ped_table else "")

    # -- QC -----------------------------------------------------------------
    records, phe_qc, to_add = load_phenotypes(phe_table, d, ped_ids)
    atomic_write_json(stage.path("qc_phenotypes.json"), phe_qc.to_dict())
    if ped_table is not None:
        pc = data["pedigree_columns"]
        ped_data = load_pedigree(ped_table, PedigreeColumns(pc["id"], pc["sire"], pc["dam"],
                                                            pc["sex"], pc["birth_date"]),
                                 unknown_parent, missing, extra_founders=to_add)
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

    # -- relationship structure --------------------------------------------
    structure = _structure(spec, ped_data, manifest)
    manifest["relationship"] = {"kind": structure.kind, "n": len(structure.labels),
                                **{k: v for k, v in structure.meta.items() if _is_small(v)}}

    results: dict[str, Any] = {
        "abp_version": manifest["code"]["abp_version"],
        "run_id": manifest["run_id"],
        "project": d["project"],
        "analysis": d["analysis"],
        "relationship": manifest["relationship"],
        "qc": {"phenotypes": phe_qc.to_dict(),
               "pedigree": ped_data.qc.to_dict() if ped_data else None},
        "traits": {},
        "limitations": _limitations(d, structure),
    }
    traits = d["model"]["traits"]
    if len(traits) > 1:
        from .multitrait import run_multitrait
        results["traits"] = run_multitrait(spec, records, structure, ped_data, stage, manifest,
                                           budget)
    else:
        results["traits"][traits[0]] = _run_single_trait(spec, records, traits[0], structure,
                                                         ped_data, stage, manifest, budget, resume)
    if d.get("index"):
        from .index_outputs import write_index
        results["index"] = write_index(spec, results, structure, stage, manifest)
    manifest["limitations"] = results["limitations"]
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
        "No unknown-parent groups or metafounders: all unknown parents are treated as "
        "unrelated, non-inbred base animals.",
    ]
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
    log.info("%s: %d records, %d fixed columns kept (%d constrained), random terms %s",
             trait, model.y.size, model.fixed.rank, len(model.fixed.constrained_labels),
             [t.name for t in model.terms])
    vmode = d["variances"]["mode"]
    reml_info = None
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
    sol = d["solver"]
    res = blup(model.y, model.fixed.X, model.terms, vc, method=sol["method"],
               compute_pev=(sol["pev"] == "exact"), tol=sol["tol"], max_iter=sol["max_iter"],
               memory_budget_bytes=budget)
    s = res.solve
    log.info("%s: solved %d equations with %s (%s); relative residual %.2e; %.2f s", trait,
             s.solution.size, s.method, s.selection_reason, s.rel_residual, s.wall_seconds)
    files = _write_single_trait_outputs(stage, model, res, ped_data)
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
    return out


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


def _write_single_trait_outputs(stage: OutputStage, model: SingleTraitModel, res: BLUPResult,
                                ped_data: PedigreeData | None) -> dict:
    trait = model.trait
    gen = res.terms[model.genetic_term]
    files = {}
    ped = ped_data.pedigree if ped_data else None
    F = ped.inbreeding() if ped is not None else None
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
    write_csv(stage.path(name), ["animal", "sire", "dam", "sex", "generation", "inbreeding",
                                 "n_records", "ebv", "pev", "sep", "reliability", "accuracy"], rows)
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
