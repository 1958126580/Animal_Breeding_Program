"""Breeder-facing Markdown report generated only from recorded results.

The report is a *view* of ``results.json`` and ``manifest.json``: every
number printed here is read from those records; nothing is recomputed,
decorated or invented.  Missing values print as "n/a".
"""

from __future__ import annotations

from typing import Any


def _f(x: Any, nd: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _e(x: Any) -> str:
    return "n/a" if x is None else f"{x:.2e}"


def render_report(results: dict, manifest: dict) -> str:
    p, a = results["project"], results["analysis"]
    L: list[str] = []
    num = iter(range(1, 100))
    L.append(f"# Genetic evaluation report: {p['name']}")
    L.append("")
    L.append(f"Status: **{results.get('status', manifest.get('status'))}** | "
             f"run `{manifest['run_id']}` | ABP {results['abp_version']}")
    L.append("")
    if p["synthetic_data"]:
        L.append("> **SYNTHETIC DATA.** These results demonstrate the software only. They say "
                 "nothing about any real flock or herd.")
        L.append("")

    L.append(f"## {next(num)}. What was estimated")
    L.append("")
    L.append("| Item | Value |")
    L.append("|---|---|")
    L.append(f"| Species | {p['species']} |")
    L.append(f"| Estimand | `{a['task']}`: additive breeding value (EBV). An EBV predicts the "
             "genetic merit an animal passes to its offspring; on average an offspring receives "
             "half of each parent's EBV. It is not a prediction of the animal's own phenotype. |")
    L.append(f"| Genetic base | {a['genetic_base']} |")
    L.append(f"| Target population | {a['target_population']} |")
    L.append(f"| Information cut-off | {a['information_cutoff']} |")
    if a.get("applicability"):
        L.append(f"| Applicability | {a['applicability']} |")
    rel = results["relationship"]
    L.append(f"| Relationship matrix | {rel['kind']} ({rel['n']} animals) |")
    L.append("")

    for trait, t in results["traits"].items():
        L.append(f"## {next(num)}. Results for trait `{trait}` ({t['unit']})")
        L.append("")
        vc = t["variance_components"]
        L.append(f"Variance components ({t['variance_source']}): "
                 + ", ".join(f"{k} = {_f(v, 4)}" for k, v in vc.items())
                 + f"; heritability h2 = {_f(t['heritability'])}"
                 + (f" (approx. SE {_f(t['reml']['heritability_se'])})"
                    if t.get("reml") and t["reml"].get("heritability_se") is not None else "")
                 + ".")
        if t.get("reml"):
            r = t["reml"]
            L.append("")
            L.append(f"REML: status **{r['status']}**, {r['iterations']} iterations, "
                     f"log-likelihood {_f(r['loglik'], 6)}. "
                     + ("Approximate standard errors (from the inverse average-information "
                        "matrix; not valid at a boundary): "
                        + ", ".join(f"{k} = {_f(v, 4)}" for k, v in (r.get('se') or {}).items())
                        if r.get("se") else "Standard errors not reported (see status)."))
            if r.get("boundary"):
                L.append("")
                L.append(f"Boundary: {', '.join(r['boundary'])} estimated at zero (boundary "
                         "optimum; the term was removed from the model for BLUP).")
        L.append("")
        L.append(f"### Top {len(t['top'])} animals by EBV")
        L.append("")
        L.append("| Rank | Animal | Sex | EBV | Reliability | SEP | Own records |")
        L.append("|---:|---|---|---:|---:|---:|---:|")
        for r in t["top"]:
            L.append(f"| {r['rank']} | {r['animal']} | {r['sex']} | {_f(r['ebv'])} | "
                     f"{_f(r['reliability'])} | {_f(r['sep'])} | {r['n_records']} |")
        L.append("")
        L.append("Reliability is the model-based squared correlation between EBV and true "
                 "breeding value (1 - PEV / genetic variance of the animal). SEP is the standard "
                 "error of prediction (square root of PEV). Animals with low reliability can "
                 "change rank substantially when more information arrives.")
        L.append("")
        rs = t.get("reliability_summary")
        if rs:
            L.append(f"Reliability across {t['n_animals_evaluated']} evaluated animals: mean "
                     f"{_f(rs['mean'])}, min {_f(rs['min'])}, max {_f(rs['max'])}.")
            L.append("")
        L.append(f"Full list: `{t['files']['ebv']}`. Fixed effects: `{t['files']['fixed_effects']}` "
                 f"({t['n_fixed_constrained']} level(s) constrained to zero for identifiability; "
                 "individual fixed-effect solutions are not estimable functions and should not "
                 "be interpreted on their own).")
        L.append("")

    if results.get("index"):
        ix = results["index"]
        L.append(f"## {next(num)}. Aggregate breeding objective (index on EBVs)")
        L.append("")
        L.append(f"Weights ({ix['weight_units']}): "
                 + ", ".join(f"{k} = {v}" for k, v in ix["weights"].items()) + ".")
        if ix["synthetic_weights"]:
            L.append("")
            L.append("> The economic weights are **synthetic placeholders**, not industry values.")
        L.append("")
        L.append(ix["method_note"])
        L.append("")
        L.append("| Rank | Animal | Sex | Index | Reliability of index |")
        L.append("|---:|---|---|---:|---:|")
        for r in ix["top"]:
            L.append(f"| {r['rank']} | {r['animal']} | {r['sex']} | {_f(r['index'])} | "
                     f"{_f(r['reliability'])} |")
        L.append("")

    L.append(f"## {next(num)}. Data and quality control")
    L.append("")
    qp = results["qc"]["phenotypes"]
    L.append(f"Phenotype rows: {qp['stats'].get('n_rows')}; records used: "
             f"{qp['stats'].get('n_records_used')}; excluded by approved QC rules: "
             f"{qp['stats'].get('n_excluded')} (listed in `qc_excluded_records.csv`).")
    qped = results["qc"].get("pedigree")
    if qped:
        s = qped["stats"]
        L.append("")
        L.append(f"Pedigree: {s['n_animals']} animals ({s['n_founders_added']} added as founders), "
                 f"{s['n_sires']} sires, {s['n_dams']} dams, deepest generation {s['max_generation']}, "
                 f"mean inbreeding {_f(s['mean_inbreeding'], 4)}, max {_f(s['max_inbreeding'], 4)}.")
    L.append("")
    L.append("| Section | Check | Severity | Count | Message |")
    L.append("|---|---|---|---:|---|")
    for sec in ("pedigree", "phenotypes", "genotypes"):
        q = results["qc"].get(sec)
        if not q:
            continue
        for f in q["findings"]:
            L.append(f"| {sec} | {f['check']} | {f['severity']} | {f['count']} | {f['message']} |")
    L.append("")

    L.append(f"## {next(num)}. Computation and numerical verification")
    L.append("")
    L.append("| Trait | Solver | Why | Equations | Relative residual | Seconds |")
    L.append("|---|---|---|---:|---:|---:|")
    for trait, t in results["traits"].items():
        s = t["solver"]
        L.append(f"| {trait} | {s['method']} | {s['selection_reason']} | {s['n_equations']} | "
                 f"{_e(s['relative_residual'])} | {_f(s['wall_seconds'], 2)} |")
    L.append("")
    L.append("The relative residual ||C s - r|| / ||r|| is recomputed on the original "
             "mixed-model equations after solving; runs above the limit are rejected.")
    L.append("")

    L.append(f"## {next(num)}. Limitations")
    L.append("")
    for x in results["limitations"]:
        L.append(f"- {x}")
    L.append("")

    L.append(f"## {next(num)}. Provenance")
    L.append("")
    code = manifest["code"]
    L.append(f"- Spec: `{manifest['spec']['path']}` (sha256 `{manifest['spec']['sha256']}`)")
    for i in manifest["inputs"]:
        L.append(f"- Input {i['role']}: `{i['path']}` (sha256 `{i['sha256']}`, {i['rows']} rows)")
    L.append(f"- Code: ABP {code['abp_version']}, commit `{code.get('commit') or 'n/a'}`"
             + ("" if code.get("working_tree_clean", True) else " (uncommitted changes present)"))
    env = manifest["environment"]
    L.append(f"- Environment: Python {env['python']}, NumPy {env['numpy']}, SciPy {env['scipy']}, "
             f"{env['os']}, native kernel: {env['native_kernel']}")
    L.append(f"- Backend: requested {manifest['backend']['requested']}, used "
             f"{manifest['backend']['used']}")
    L.append("- Full machine-readable record: `manifest.json`, `results.json`, `run.log`.")
    L.append("")
    return "\n".join(L)
