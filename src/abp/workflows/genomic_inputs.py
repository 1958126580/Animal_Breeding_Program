"""Genomic inputs for the workflow: genotype QC, frequencies, G/H structures."""

from __future__ import annotations

import csv
from dataclasses import dataclass

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
    storage = spec["genomic"]["genotype_storage"]
    if data["plink"] is not None:
        from ..io.plink import load_plink
        return load_plink(spec.resolve(data["plink"]), data["genotype_assembly"], storage)
    geno = read_table(spec.resolve(data["genotypes"]), data["delimiter"])
    mapt = read_table(spec.resolve(data["marker_map"]), data["delimiter"])
    g = load_genotypes(geno, mapt, set(data["missing_values"]))
    if storage == "int8":
        from ..qc.genotype import to_int8
        g = to_int8(g)
    return g


def _training_rows(ids: list[str], animals_with_records: set[str]) -> np.ndarray:
    """Genotyped animals with at least one QC-passed record for the analysed traits."""
    return np.array([i for i, a in enumerate(ids) if a in animals_with_records], dtype=np.int64)


def _file_frequencies(spec: AnalysisSpec, markers: list[str]) -> np.ndarray:
    path = spec.resolve(spec["data"]["allele_frequencies"])
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = {r["marker_id"]: float(r["frequency"]) for r in csv.DictReader(fh)}
    missing = [m for m in markers if m not in rows]
    if missing:
        raise ABPError("GENOTYPE_ID_CONFLICT", f"{len(missing)} marker(s) lack a frequency in "
                       f"{path.name}", markers=missing[:20])
    return np.array([rows[m] for m in markers])


@dataclass
class PreparedGenotypes:
    """QC-passed genotypes with the reference allele frequencies and provenance."""

    geno: GenotypeData
    p: np.ndarray
    freq_note: str
    transductive: bool


def prepare_genotypes(spec: AnalysisSpec, ped: PedigreeData | None, manifest: dict,
                      animals_with_records: set[str]) -> PreparedGenotypes:
    """Load, QC and choose frequencies (shared by GBLUP, single step and Bayes)."""
    cfg = spec["genomic"]
    raw = load_genotype_inputs(spec)
    dat = spec["data"]
    if dat["plink"] is not None:
        prefix = spec.resolve(dat["plink"])
        paths = {"genotypes": prefix.with_suffix(".bed"), "marker_map": prefix.with_suffix(".bim"),
                 "sample_list": prefix.with_suffix(".fam")}
        rows = {"genotypes": len(raw.ids), "marker_map": len(raw.markers),
                "sample_list": len(raw.ids)}
        for role, key in (("genotypes", "genotypes"), ("marker_map", "marker_map"),
                          ("sample_list", "fam")):
            manifest["inputs"].append({"role": role, "path": str(paths[role]),
                                       "sha256": raw.sha256[key], "rows": rows[role]})
    else:
        manifest["inputs"].append({"role": "genotypes", "path": str(spec.resolve(dat["genotypes"])),
                                   "sha256": raw.sha256["genotypes"], "rows": len(raw.ids)})
        manifest["inputs"].append({"role": "marker_map", "path": str(spec.resolve(dat["marker_map"])),
                                   "sha256": raw.sha256["marker_map"], "rows": len(raw.markers)})
    parents = None
    if ped is not None:
        P = ped.pedigree
        parents = {a: (P.ids[P.sire[i]] if P.sire[i] >= 0 else None,
                       P.ids[P.dam[i]] if P.dam[i] >= 0 else None) for i, a in enumerate(P.ids)}
    src = cfg["frequency_source"]
    sample = _training_rows(raw.ids, animals_with_records) if src == "training_genotyped" else None
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
    manifest.setdefault("_qc_sections", {})["genotypes"] = geno.qc.to_dict()
    return PreparedGenotypes(geno, p, freq_note, transductive)


def genomic_structure(spec: AnalysisSpec, ped: PedigreeData | None, relationship: str,
                      manifest: dict, animals_with_records: set[str]) -> GeneticStructure:
    """Build the G (GBLUP) or H (single-step) structure with full provenance."""
    cfg = spec["genomic"]
    dat = spec["data"]
    prep = prepare_genotypes(spec, ped, manifest, animals_with_records)
    geno, p, freq_note, transductive = prep.geno, prep.p, prep.freq_note, prep.transductive
    src = cfg["frequency_source"]
    if relationship == "single_step" and cfg["single_step_mode"] == "matrix_free":
        return _matrix_free_single_step(spec, ped, prep)
    G, d = vanraden_g(geno.dosage, p, geno.missing)
    meta = {"method": "VanRaden (2008) method 1: G = WW'/(2 sum p(1-p))",
            "frequency_source": src, "frequency_note": freq_note,
            "candidate_genotype_use": "transductive_unsupervised" if transductive else "none",
            "missing_dosage_policy": "set to 2p (centred value 0) at the reference frequency",
            "n_genotyped": len(geno.ids), "n_markers": len(geno.markers), "scaling_d": d,
            "genotype_storage": str(geno.dosage.dtype),
            "assembly": geno.assembly,
            "counted_allele": ("A1 of the PLINK .bim file" if dat["plink"] is not None
                               else "per-marker counted_allele column of the marker map"),
            "mean_diag_G": float(np.mean(np.diag(G)))}
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
    n_core = int(cfg["apy_core_size"])
    Gs, rec = apply_g_policy(G, cfg["singular_policy"], cfg["tuning"], A22,
                             cfg["blend_alpha"], cfg["ridge"], check_pd=(n_core == 0))
    meta["g_policy"] = rec.__dict__
    apy = None
    if n_core:
        from ..core.genomic import apy_inverse
        from .manifest import sha256_array
        n2 = len(geno.ids)
        if n_core >= n2:
            raise ABPError("SPEC_INVALID", f"genomic.apy_core_size ({n_core}) must be smaller "
                           f"than the number of genotyped animals ({n2})")
        core = np.sort(np.random.default_rng(cfg["apy_seed"]).choice(n2, n_core, replace=False))
        apy = apy_inverse(Gs, core)
        ids = [geno.ids[k] for k in core]
        meta["apy"] = {**apy.record, "core_selection": f"random, seed {cfg['apy_seed']}",
                       "core_ids_sha256": sha256_array(ids), "core_ids": ids,
                       "note": "APY replaces G* by G_APY (a different model): core-core and "
                               "core-noncore relationships and non-core diagonals are kept, "
                               "non-core off-diagonals become G_nc G_cc^-1 G_cn"}
        Gs = apy.g_apy
    if A22 is not None:
        off = ~np.eye(A22.shape[0], dtype=bool)
        meta["compatibility"] = {"mean_diag_A22": float(np.mean(np.diag(A22))),
                                 "mean_offdiag_A22": float(np.mean(A22[off])),
                                 "mean_diag_Gstar": float(np.mean(np.diag(Gs))),
                                 "mean_offdiag_Gstar": float(np.mean(Gs[off]))}
    if relationship == "genomic":
        g_inv, logdet = ((apy.g_inv, apy.logdet) if apy is not None
                         else spd_inverse_and_logdet(Gs, "G*"))
        return GeneticStructure("genomic", tuple(geno.ids), g_inv, np.diag(Gs).copy(), logdet,
                                meta)
    ss = single_step(ped.pedigree, g_index, Gs, a22=A22,
                     g_inverse=None if apy is None else (apy.g_inv, apy.logdet))
    meta["single_step"] = "H^-1 = A^-1 + embed(G*^-1 - A22^-1); A22^-1 from A22 itself"
    return GeneticStructure("single_step", ped.pedigree.ids, ss.h_inv, ss.h_diag, ss.logdet_h, meta)


def _matrix_free_single_step(spec: AnalysisSpec, ped: PedigreeData | None,
                             prep: PreparedGenotypes) -> GeneticStructure:
    """Single step with ``H^{-1}`` as an operator (:mod:`abp.core.ssop`); no dense
    ``A22``, ``A22^{-1}`` or ``n x n2`` block is formed, and with APY no ``G``."""
    from ..core.genomic import centered, scaling_d
    from ..core.ssop import (A22InverseOperator, APYOperator, DenseInverseOperator,
                             SingleStepHInverse, a_block, apy_blocks_from_dosage,
                             apy_blocks_from_genotypes)
    from .manifest import sha256_array
    cfg = spec["genomic"]
    if ped is None:
        raise ABPError("UNSUPPORTED_COMBINATION", "single step needs a pedigree")
    geno, p = prep.geno, prep.p
    P = ped.pedigree
    absent = [a for a in geno.ids if not P.contains(a)]
    if absent:
        raise ABPError("GENOTYPE_ID_CONFLICT", f"{len(absent)} genotyped animal(s) are not in "
                       "the pedigree", animals=absent[:20])
    g_index = P.index_of(geno.ids)
    n2 = len(geno.ids)
    d = scaling_d(p)
    compact = geno.dosage.dtype == np.int8
    Wc = None if compact else centered(geno.dosage, p, geno.missing)
    meta = {"method": "VanRaden (2008) method 1: G = WW'/(2 sum p(1-p))",
            "frequency_source": cfg["frequency_source"], "frequency_note": prep.freq_note,
            "candidate_genotype_use": ("transductive_unsupervised" if prep.transductive
                                       else "none"),
            "missing_dosage_policy": "set to 2p (centred value 0) at the reference frequency",
            "n_genotyped": n2, "n_markers": len(geno.markers), "scaling_d": d,
            "genotype_storage": str(geno.dosage.dtype),
            "assembly": geno.assembly,
            "g_policy": {"singular_policy": cfg["singular_policy"], "tuning": "none",
                         "blend_alpha": (cfg["blend_alpha"] if cfg["singular_policy"] == "blend"
                                         else None),
                         "ridge": cfg["ridge"] if cfg["singular_policy"] == "ridge" else None},
            "single_step": "matrix-free: H^-1 v = A^-1 v + embed(G*^-1 v2 - A22^-1 v2); "
                           "A22^-1 v2 = A^22 v2 - A^21 (A^11)^-1 A^12 v2 (sparse LDL' of A^11)",
            "single_step_mode": "matrix_free"}
    n_core = int(cfg["apy_core_size"])
    pol = cfg["singular_policy"]
    if n_core:
        if n_core >= n2:
            raise ABPError("SPEC_INVALID", f"genomic.apy_core_size ({n_core}) must be smaller "
                           f"than the number of genotyped animals ({n2})")
        core = np.sort(np.random.default_rng(cfg["apy_seed"]).choice(n2, n_core, replace=False))
        a22_cols = a22_diag = None
        if pol == "blend":
            a22_cols = a_block(P, g_index, g_index[core])             # n2 x c, in blocks
            a22_diag = 1.0 + P.inbreeding()[g_index]
        if compact:
            gcc, gcn, gnn = apy_blocks_from_dosage(geno.dosage, p, core, pol, cfg["blend_alpha"],
                                                   cfg["ridge"], a22_cols, a22_diag)
        else:
            gcc, gcn, gnn = apy_blocks_from_genotypes(Wc, d, core, pol, cfg["blend_alpha"],
                                                      cfg["ridge"], a22_cols, a22_diag)
        g_op = APYOperator(gcc, gcn, gnn, core, n2)
        ids = [geno.ids[k] for k in core]
        meta["apy"] = {"method": "APY (Misztal, Legarra & Aguilar 2014), applied as an operator",
                       "n_core": int(n_core), "n_noncore": int(n2 - n_core),
                       "min_m": float(g_op.m.min()), "mean_m": float(g_op.m.mean()),
                       "core_selection": f"random, seed {cfg['apy_seed']}",
                       "core_ids_sha256": sha256_array(ids), "core_ids": ids}
    else:
        G = vanraden_g(geno.dosage, p, geno.missing)[0]
        A22 = P.a_submatrix(g_index) if pol == "blend" else None
        Gs, rec = apply_g_policy(G, pol, "none", A22, cfg["blend_alpha"], cfg["ridge"])
        meta["g_policy"] = rec.__dict__
        chol = None
        if spec["solver"]["pev"] == "sampled":
            import scipy.linalg as sla
            chol = sla.cholesky(Gs, lower=True)
        g_op = DenseInverseOperator(spd_inverse_and_logdet(Gs, "G*")[0], chol)
    a22_op = A22InverseOperator(P.ainv(), g_index)
    h = SingleStepHInverse(P.ainv().tocsr(), g_index, g_op, a22_op, meta, ped=P)
    # diag(H) is not computed on this path; reliabilities are refused by the spec rules
    return GeneticStructure("single_step", P.ids, h, np.full(P.n, np.nan), None, meta)
