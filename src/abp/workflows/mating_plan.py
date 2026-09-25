"""``abp mate``: optimal contributions + mating plan from a TOML mating spec.

The output is a **proposal** for the breeder: contributions per candidate,
integer mating numbers, a sire-dam plan, and the evidence that every hard
constraint holds.  ABP never executes matings.

Mating spec (TOML)::

    schema_version = "1"
    [project]
    name = "..."; species = "..."; synthetic_data = true
    [mating]
    candidates = "candidates.csv"      # id, sex (M/F), merit, capacity [, carrier]
    pedigree = "pedigree.csv"          # defines A for the candidates
    merit_units = "SCU (index units)"
    n_matings = 120
    delta_f = 0.01                     # or max_coancestry = 0.05 (exactly one)
    max_pair_relationship = 0.25       # optional hard limit on A_sd
    max_affected_risk = 0.0            # optional; needs a carrier column
    forbidden_pairs = "forbidden.csv"  # optional: columns sire,dam
    group_prefix = "UPG:"              # optional: parent codes naming unknown-parent
                                       # groups (unknown parents for A; see abp.core.upg)
"""

from __future__ import annotations

import logging
import time
import tomllib
from pathlib import Path

import numpy as np

from ..core.spec import Field, Section, _validate_section
from ..decision.mating import allocate, forbidden_mask
from ..decision.ocs import (coancestry_target_from_delta_f, expected_response, integer_matings,
                            repair_integer_plan, solve_ocs)
from ..errors import ABPError
from ..io.tables import parse_float_column, read_table, write_csv
from ..qc.pedigree import PedigreeColumns, load_pedigree
from . import manifest as mf
from .outputs import OutputStage, atomic_write_json, atomic_write_text

log = logging.getLogger("abp")

MATING_SCHEMA = Section({
    "schema_version": Field("str", required=True, choices=("1",)),
    "project": Section({
        "name": Field("str", required=True),
        "species": Field("str", required=True),
        "synthetic_data": Field("bool", required=True),
        "description": Field("str", default=""),
    }, required=True),
    "mating": Section({
        "candidates": Field("str", required=True),
        "pedigree": Field("str", required=True),
        "delimiter": Field("str", default=","),
        "id_column": Field("str", default="id"),
        "sex_column": Field("str", default="sex"),
        "merit_column": Field("str", default="merit"),
        "capacity_column": Field("str", default="capacity"),
        "carrier_column": Field("str"),
        "merit_units": Field("str", required=True),
        "n_matings": Field("int", required=True, check=lambda x: None if x > 0 else "must be > 0"),
        "delta_f": Field("float"),
        "max_coancestry": Field("float"),
        "max_pair_relationship": Field("float"),
        "max_affected_risk": Field("float"),
        "forbidden_pairs": Field("str"),
        "group_prefix": Field("str"),
    }, required=True),
}, required=True)


def load_mating_spec(path: Path) -> dict:
    try:
        raw = tomllib.loads(Path(path).read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise ABPError("INPUT_NOT_FOUND", f"mating spec not found: {path}", file=str(path)) from None
    except tomllib.TOMLDecodeError as exc:
        raise ABPError("SPEC_INVALID", f"{Path(path).name} is not valid TOML: {exc}") from None
    d = _validate_section(MATING_SCHEMA, raw, "")
    m = d["mating"]
    if (m["delta_f"] is None) == (m["max_coancestry"] is None):
        raise ABPError("SPEC_INVALID", "mating: give exactly one of delta_f or max_coancestry",
                       key="mating")
    if m["max_affected_risk"] is not None and m["carrier_column"] is None:
        raise ABPError("SPEC_INVALID", "mating.max_affected_risk needs mating.carrier_column",
                       key="mating.max_affected_risk")
    return d


def plan_matings(spec_path: str | Path, out_dir: str | Path, force: bool = False) -> Path:
    t0 = time.perf_counter()
    spec_path = Path(spec_path).resolve()
    d = load_mating_spec(spec_path)
    m = d["mating"]
    base = spec_path.parent
    run_id = mf.new_run_id()
    stage = OutputStage(Path(out_dir), run_id, force, 50.0)
    manifest = {"schema_version": mf.MANIFEST_SCHEMA_VERSION, "document_type": "mating_manifest",
                "run_id": run_id, "status": "running", "started_at": mf.utc_now(),
                "code": mf.code_provenance(), "environment": mf.environment(),
                "spec": {"path": str(spec_path), "sha256": mf.sha256_path(spec_path),
                         "effective": d},
                "inputs": [], "note": "a proposal for breeder review; no action is executed"}
    try:
        cand = read_table(base / m["candidates"], m["delimiter"])
        pedt = read_table(base / m["pedigree"], m["delimiter"])
        for role, t in (("candidates", cand), ("pedigree", pedt)):
            manifest["inputs"].append({"role": role, "path": str(t.path), "sha256": t.sha256,
                                       "rows": t.n_rows})
        ped = load_pedigree(pedt, PedigreeColumns(), {"0", "", "NA", "."}, {"", "NA", "."},
                            group_prefix=m["group_prefix"]).pedigree
        ids = cand.column(m["id_column"])
        if len(set(ids)) != len(ids):
            raise ABPError("DUPLICATE_KEY", "candidate IDs must be unique")
        missing = [a for a in ids if not ped.contains(a)]
        if missing:
            raise ABPError("PHENOTYPE_UNKNOWN_ANIMAL", f"{len(missing)} candidate(s) not in the "
                           "pedigree", animals=missing[:20])
        sex = [s.strip().upper() for s in cand.column(m["sex_column"])]
        bad = [a for a, s in zip(ids, sex) if s not in ("M", "F")]
        if bad:
            raise ABPError("SCHEMA_TYPE", "candidate sex must be M or F", animals=bad[:20])
        male = np.array([s == "M" for s in sex])
        g = parse_float_column(cand, m["merit_column"], set())
        cap = parse_float_column(cand, m["capacity_column"], set())
        if np.any(cap < 0) or np.any(cap != np.round(cap)):
            raise ABPError("SCHEMA_TYPE", "capacity must be a non-negative integer")
        cap = cap.astype(np.int64)
        carrier = None
        if m["carrier_column"]:
            carrier = parse_float_column(cand, m["carrier_column"], set())
            if np.any((carrier < 0) | (carrier > 1)):
                raise ABPError("SCHEMA_TYPE", "carrier probabilities must lie in [0, 1]")
        idx = ped.index_of(ids)
        A = ped.a_submatrix(idx)
        N = m["n_matings"]
        hi = cap / (2.0 * N)
        if m["delta_f"] is not None:
            cmax, ct = coancestry_target_from_delta_f(A, m["delta_f"])
            target = {"rule": "C_max = C_t + delta_F (1 - C_t)", "delta_f": m["delta_f"],
                      "C_t_mean_coancestry_of_candidates": ct, "C_max": cmax}
        else:
            cmax = m["max_coancestry"]
            target = {"rule": "explicit", "C_max": cmax}
        log.info("OCS: %d candidates (%d M, %d F), %d matings, C_max %.6f", len(ids), male.sum(),
                 (~male).sum(), N, cmax)
        ocs = solve_ocs(g, A, male, np.zeros(len(ids)), hi, cmax)
        rounded = integer_matings(ocs.c, cap, male, N)
        # the ceiling is a hard constraint for the whole-number plan as well
        counts, repair = repair_integer_plan(rounded, g, A, male, cap, N, cmax)
        c_real = counts / (2.0 * N)
        sires = np.flatnonzero(male & (counts > 0))
        dams = np.flatnonzero(~male & (counts > 0))
        A_sd = A[np.ix_(sires, dams)]
        explicit = set()
        if m["forbidden_pairs"]:
            ft = read_table(base / m["forbidden_pairs"], m["delimiter"])
            pos_s = {ids[i]: k for k, i in enumerate(sires)}
            pos_d = {ids[j]: k for k, j in enumerate(dams)}
            for s, dd in zip(ft.column("sire"), ft.column("dam")):
                if s in pos_s and dd in pos_d:
                    explicit.add((pos_s[s], pos_d[dd]))
        cs = carrier[sires] if carrier is not None else None
        cd = carrier[dams] if carrier is not None else None
        forb, reasons = forbidden_mask(A_sd, m["max_pair_relationship"], cs, cd,
                                       m["max_affected_risk"], explicit)
        plan = allocate(counts[sires], counts[dams], A_sd, forb, cs, cd)
        coan_real = float(c_real @ A @ c_real) / 2.0
        write_csv(stage.path("contributions.csv"),
                  ["animal", "sex", "merit", "capacity", "contribution_optimal", "n_matings",
                   "contribution_realized"],
                  [[ids[i], sex[i], float(g[i]), int(cap[i]), float(ocs.c[i]), int(counts[i]),
                    float(c_real[i])] for i in np.argsort(-ocs.c, kind="stable")])
        rows = []
        for (s, dd), f, r in zip(plan.pairs, plan.offspring_inbreeding, plan.affected_risk):
            i, j = sires[s], dams[dd]
            rows.append([ids[i], ids[j], float(A[i, j]), float(f), float((g[i] + g[j]) / 2),
                         float(r)])
        rows.sort()
        write_csv(stage.path("mating_plan.csv"), ["sire", "dam", "relationship_A_sd",
                                                  "offspring_inbreeding", "expected_offspring_merit",
                                                  "affected_risk"], rows)
        summary = {
            "n_candidates": len(ids), "n_male": int(male.sum()), "n_female": int((~male).sum()),
            "n_matings": N, "merit_units": m["merit_units"], "coancestry_target": target,
            "ocs": {"status": ocs.status, "merit_c_g": ocs.merit,
                    "expected_response_vs_candidate_mean": expected_response(ocs.c, g),
                    "coancestry": ocs.coancestry, "max_merit_coancestry": ocs.max_merit_coancestry,
                    "min_achievable_coancestry": ocs.min_coancestry, "penalty_mu": ocs.penalty_mu,
                    "bisection_steps": ocs.bisection_steps, "kkt_certificate": ocs.kkt,
                    "n_selected_male": int((male & (ocs.c > 1e-12)).sum()),
                    "n_selected_female": int((~male & (ocs.c > 1e-12)).sum())},
            "integer_plan": {"coancestry_realized": coan_real,
                             "ceiling_satisfied": bool(coan_real <= cmax + 1e-15),
                             "coancestry_after_rounding_only": float(
                                 (rounded / (2.0 * N)) @ A @ (rounded / (2.0 * N))) / 2.0,
                             "local_search_moves": repair["moves"],
                             "merit_gap_to_continuous_optimum": ocs.merit - float(c_real @ g),
                             "coancestry_change_from_rounding": coan_real - ocs.coancestry,
                             "merit_realized": float(c_real @ g),
                             "n_sires_used": int(sires.size), "n_dams_used": int(dams.size)},
            "mating": {"mean_offspring_inbreeding": plan.mean_inbreeding,
                       "max_offspring_inbreeding": float(plan.offspring_inbreeding.max()),
                       "max_affected_risk": float(plan.affected_risk.max()) if plan.pairs else 0.0,
                       "forbidden_pairs_by_reason": reasons, "checks": plan.checks},
            "definitions": {
                "coancestry": "c'Ac/2 with A the pedigree relationship matrix of the candidates "
                              "(founder base); expected progeny inbreeding under random union "
                              "of gametes incl. selfing, not of this plan",
                "offspring_inbreeding": "A_sd / 2 of each planned pair",
                "affected_risk": "p_s p_d / 4 for a declared autosomal recessive"},
            "note": "proposal for breeder review; no mating is executed by ABP",
        }
        atomic_write_json(stage.path("mating_summary.json"), summary)
        atomic_write_text(stage.path("mating_report.md"), _report(d, summary))
        manifest["status"] = "passed"
        manifest["finished_at"] = mf.utc_now()
        manifest["execution"] = {"wall_seconds": time.perf_counter() - t0,
                                 "peak_rss_bytes": mf.peak_rss_bytes(), "exit_code": 0}
        manifest["outputs"] = [{"file": p.name, "sha256": mf.sha256_path(p)}
                               for p in sorted(stage.dir.iterdir()) if p.is_file()]
        atomic_write_json(stage.path("manifest.json"), manifest)
        return stage.publish()
    except ABPError as err:
        manifest.update(status="failed", error=err.to_dict(), finished_at=mf.utc_now())
        atomic_write_json(stage.path("manifest.json"), manifest)
        stage.abandon("failed")
        raise


def _report(d: dict, s: dict) -> str:
    p, o, ip, mt = d["project"], s["ocs"], s["integer_plan"], s["mating"]
    L = [f"# Mating plan proposal: {p['name']}", ""]
    if p["synthetic_data"]:
        L += ["> **SYNTHETIC DATA.** Demonstration only.", ""]
    L += ["This is a **proposal for breeder review**. ABP does not execute matings.", "",
          "## Targets and result", "",
          "| Item | Value |", "|---|---|",
          f"| Candidates | {s['n_candidates']} ({s['n_male']} M, {s['n_female']} F) |",
          f"| Matings | {s['n_matings']} |",
          f"| Coancestry ceiling | {s['coancestry_target']['C_max']:.6f} ({s['coancestry_target']['rule']}) |",
          f"| Optimal contributions | {o['status']}; coancestry {o['coancestry']:.6f}; "
          f"merit c'g {o['merit_c_g']:.4f} {s['merit_units']} |",
          f"| Without the ceiling | coancestry {o['max_merit_coancestry']:.6f} |",
          f"| Minimum achievable coancestry | {o['min_achievable_coancestry']:.6f} |",
          f"| After rounding to whole matings | coancestry {ip['coancestry_realized']:.6f}; merit "
          f"{ip['merit_realized']:.4f}; {ip['n_sires_used']} sires, {ip['n_dams_used']} dams |",
          f"| Planned progeny inbreeding | mean {mt['mean_offspring_inbreeding']:.5f}, max "
          f"{mt['max_offspring_inbreeding']:.5f} |",
          f"| Largest recessive-disease risk of a pair | {mt['max_affected_risk']:.5f} |",
          "", "## Evidence", "",
          f"KKT certificate of the contribution optimum: stationarity "
          f"{o['kkt_certificate']['stationarity_max_abs']:.1e}, sex-sum residual "
          f"{o['kkt_certificate']['sex_sum_residual']:.1e}, coancestry slack "
          f"{o['kkt_certificate']['coancestry_slack']:.1e}, smallest bound multiplier "
          f"{o['kkt_certificate']['min_bound_multiplier']:.1e}.", "",
          "Mating-plan checks: " + ", ".join(f"{k} = {v}" for k, v in mt["checks"].items()) + ".",
          "", "Forbidden pairs by reason: " + (", ".join(
              f"{k}: {v}" for k, v in mt["forbidden_pairs_by_reason"].items()) or "none") + ".",
          "", "## Definitions", ""]
    L += [f"- **{k}**: {v}" for k, v in s["definitions"].items()]
    L += ["", "Files: `contributions.csv`, `mating_plan.csv`, `mating_summary.json`, "
          "`manifest.json`.", ""]
    return "\n".join(L)
