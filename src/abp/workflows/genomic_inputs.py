"""Genomic inputs for the workflow: genotype QC, frequencies, G/H structures."""

from __future__ import annotations

import csv

import numpy as np

from ..core.genomic import apply_g_policy, single_step, spd_inverse_and_logdet, vanraden_g
from ..core.model import GeneticStructure
from ..core.spec import AnalysisSpec
from ..errors import ABPError
from ..io.tables import read_table
from ..qc.genotype import GenotypeData, filter_genotypes, load_genotypes
from ..qc.pedigree import PedigreeData


def load_genotype_inputs(spec: AnalysisSpec) -> GenotypeData:
    data = spec["data"]
    geno = read_table(spec.resolve(data["genotypes"]), data["delimiter"])
    mapt = read_table(spec.resolve(data["marker_map"]), data["delimiter"])
    return load_genotypes(geno, mapt, set(data["missing_values"]))


def _training_rows(spec: AnalysisSpec, ids: list[str]) -> np.ndarray:
    """Genotyped animals with at least one record for the analysed traits."""
    data = spec["data"]
    phe = read_table(spec.resolve(data["phenotypes"]), data["delimiter"])
    id_col = data["phenotype_columns"]["id"]
    trait_cols = {t["name"]: t["column"] for t in spec["traits"]}
    missing = set(data["missing_values"])
    have = set()
    cols = [phe.column(trait_cols[t]) for t in spec["model"]["traits"]]
    for k, a in enumerate(phe.column(id_col)):
        if any(c[k].strip() not in missing and c[k] not in missing for c in cols):
            have.add(a)
    return np.array([i for i, a in enumerate(ids) if a in have], dtype=np.int64)


def _file_frequencies(spec: AnalysisSpec, markers: list[str]) -> np.ndarray:
    path = spec.resolve(spec["data"]["allele_frequencies"])
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = {r["marker_id"]: float(r["frequency"]) for r in csv.DictReader(fh)}
    missing = [m for m in markers if m not in rows]
    if missing:
        raise ABPError("GENOTYPE_ID_CONFLICT", f"{len(missing)} marker(s) lack a frequency in "
                       f"{path.name}", markers=missing[:20])
    return np.array([rows[m] for m in markers])


def genomic_structure(spec: AnalysisSpec, ped: PedigreeData | None, relationship: str,
                      manifest: dict) -> GeneticStructure:
    """Build the G (GBLUP) or H (single-step) structure with full provenance."""
    cfg = spec["genomic"]
    raw = load_genotype_inputs(spec)
    manifest["inputs"].append({"role": "genotypes", "path": str(spec.resolve(spec["data"]["genotypes"])),
                               "sha256": raw.sha256["genotypes"], "rows": len(raw.ids)})
    manifest["inputs"].append({"role": "marker_map", "path": str(spec.resolve(spec["data"]["marker_map"])),
                               "sha256": raw.sha256["marker_map"], "rows": len(raw.markers)})
    parents = None
    if ped is not None:
        P = ped.pedigree
        parents = {a: (P.ids[P.sire[i]] if P.sire[i] >= 0 else None,
                       P.ids[P.dam[i]] if P.dam[i] >= 0 else None) for i, a in enumerate(P.ids)}
    src = cfg["frequency_source"]
    sample = _training_rows(spec, raw.ids) if src == "training_genotyped" else None
    geno, p_sample = filter_genotypes(raw, cfg, sample, parents)
    if src == "genotyped_all":
        p = p_sample
        freq_note = ("all QC-passed genotyped animals, including selection candidates without "
                     "phenotypes (transductive use of candidate genotypes; no phenotypes of "
                     "candidates are used)")
        transductive = True
    elif src == "training_genotyped":
        p = p_sample
        freq_note = "genotyped animals with records for the analysed trait(s) (inductive)"
        transductive = False
    elif src == "file":
        p = _file_frequencies(spec, geno.markers)
        freq_note = f"user file {spec['data']['allele_frequencies']}"
        transductive = False
    else:
        p = np.full(len(geno.markers), 0.5)
        freq_note = "fixed p = 0.5 for every marker"
        transductive = False
    G, d = vanraden_g(geno.dosage, p, geno.missing)
    meta = {"method": "VanRaden (2008) method 1: G = WW'/(2 sum p(1-p))",
            "frequency_source": src, "frequency_note": freq_note,
            "candidate_genotype_use": "transductive_unsupervised" if transductive else "none",
            "missing_dosage_policy": "set to 2p (centred value 0) at the reference frequency",
            "n_genotyped": len(geno.ids), "n_markers": len(geno.markers), "scaling_d": d,
            "assembly": geno.assembly,
            "counted_allele": "per-marker counted_allele column of the marker map",
            "mean_diag_G": float(np.mean(np.diag(G)))}
    manifest.setdefault("_qc_sections", {})["genotypes"] = geno.qc.to_dict()
    A22 = None
    g_index = None
    if relationship == "single_step" or cfg["singular_policy"] == "blend" or cfg["tuning"] != "none":
        if ped is None:
            raise ABPError("UNSUPPORTED_COMBINATION", "this genomic configuration needs a pedigree")
        P = ped.pedigree
        absent = [a for a in geno.ids if not P.contains(a)]
        if absent:
            raise ABPError("GENOTYPE_ID_CONFLICT", f"{len(absent)} genotyped animal(s) are not in "
                           "the pedigree", animals=absent[:20])
        g_index = P.index_of(geno.ids)
        A22 = P.a_submatrix(g_index)
    Gs, rec = apply_g_policy(G, cfg["singular_policy"], cfg["tuning"], A22,
                             cfg["blend_alpha"], cfg["ridge"])
    meta["g_policy"] = rec.__dict__
    if A22 is not None:
        off = ~np.eye(A22.shape[0], dtype=bool)
        meta["compatibility"] = {"mean_diag_A22": float(np.mean(np.diag(A22))),
                                 "mean_offdiag_A22": float(np.mean(A22[off])),
                                 "mean_diag_Gstar": float(np.mean(np.diag(Gs))),
                                 "mean_offdiag_Gstar": float(np.mean(Gs[off]))}
    if relationship == "genomic":
        g_inv, logdet = spd_inverse_and_logdet(Gs, "G*")
        return GeneticStructure("genomic", tuple(geno.ids), g_inv, np.diag(Gs).copy(), logdet,
                                meta)
    ss = single_step(ped.pedigree, g_index, Gs, a22=A22)
    meta["single_step"] = "H^-1 = A^-1 + embed(G*^-1 - A22^-1); A22^-1 from A22 itself"
    return GeneticStructure("single_step", ped.pedigree.ids, ss.h_inv, ss.h_diag, ss.logdet_h, meta)
