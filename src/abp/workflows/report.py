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
        if t.get("bayes"):
            bz = t["bayes"]
            L.append("")
            L.append(f"Bayesian marker regression ({bz['method']}, {bz['chains']} chains, "
                     f"{bz['iterations']} iterations, kernel {bz['kernel']}); all monitored "
                     f"quantities passed the convergence criteria: **{bz['converged']}**.")
            L.append("")
            L.append("| Quantity | Posterior mean | 90% interval | R-hat | Bulk ESS | Tail ESS |")
            L.append("|---|---:|---|---:|---:|---:|")
            for k, v in bz["summaries"].items():
                L.append(f"| {k} | {v['mean']:.4g} | [{v['q05']:.4g}, {v['q95']:.4g}] | "
                         f"{v['rhat']:.4f} | {v['ess_bulk']:.0f} | {v['ess_tail']:.0f} |")
            g = bz.get("gebv_diagnostics") or {}
            pp = bz.get("posterior_predictive") or {}
            if pp:
                L.append("")
                L.append("Posterior predictive check (p = P(T(y_rep) >= T(y)); values below 0.01 or "
                         "above 0.99 indicate that the model does not reproduce this feature of the "
                         "data): " + ", ".join(f"{k} {v['p_value']:.3f}"
                                              for k, v in pp["statistics"].items()) + ".")
            if "not_computed" in g:
                L.append("")
                L.append(f"GEBVs of {g['n_animals']} animals: {g['not_computed']}. The SEP column "
                         "below is the posterior standard deviation.")
            elif g:
                L.append("")
                L.append(f"GEBVs of {g['n_animals']} animals: worst R-hat {g['max_rhat']:.4f}, "
                         f"smallest bulk ESS {g['min_ess_bulk']:.0f}. The SEP column below is the "
                         "posterior standard deviation.")
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
        if t.get("upg"):
            u = t["upg"]
            kind = (f"random, sigma_g^2/sigma_a^2 = {_f(u['variance_ratio'])} (declared)"
                    if u["effect"] == "random" else
                    "fixed (estimability verified: rank "
                    f"{u['estimability']['rank']} of {u['estimability']['columns']})")
            L.append(f"Unknown-parent groups: {u['n_groups']} group(s), {kind}. EBVs above "
                     "include the group contributions (u* = u + Qg); group solutions and the "
                     f"gene fractions they apply to are in `{u['file']}`.")
            if u["effect"] == "fixed":
                L.append("")
                L.append("With fixed groups the prior variance of u* is not defined, so no "
                         "reliabilities are reported; SEP includes the estimation error of the "
                         "group effects.")
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

    if results.get("validation"):
        v = results["validation"]
        st, bt = v["statistics"], v.get("bootstrap") or {}

        def ci(k):
            c = (bt.get(k) or {}).get("ci95")
            return f"[{c[0]:.3f}, {c[1]:.3f}]" if c else "n/a"
        L.append(f"## {next(num)}. Forward-in-time validation (LR method)")
        L.append("")
        L.append(f"Records of `{v['trait']}` dated after {v['cutoff']} were hidden in a partial "
                 f"evaluation ({v['n_records_hidden']} hidden, {v['n_records_partial']} kept). "
                 f"Focal animals: {v['n_focal']} ({v['focal_rule']}). Both evaluations used "
                 f"the same variances ({v['variance_source']}).")
        L.append("")
        L.append("| Statistic | Estimate | 95% bootstrap interval | Expected if unbiased |")
        L.append("|---|---:|---|---|")
        L.append(f"| Bias Delta_p = mean(EBV_partial) - mean(EBV_whole) | "
                 f"{st['bias_delta_p']:.4f} | {ci('bias_delta_p')} | 0 |")
        L.append(f"| Dispersion b_w\\|p | {st['dispersion_b_w_p']:.4f} | "
                 f"{ci('dispersion_b_w_p')} | 1 |")
        L.append(f"| Correlation rho_wp | {st['rho_wp']:.4f} | {ci('rho_wp')} | acc_p/acc_w |")
        L.append("")
        L.append(f"Bootstrap: {bt.get('replicates', 0)} replicates over {v['bootstrap_cluster']} "
                 f"clusters (seed {bt.get('seed')}). {v['interpretation'].capitalize()}.")
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
