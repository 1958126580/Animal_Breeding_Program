"""Workflow glue for metafounders: assignment, Gamma provenance, A^Gamma / H structures.

The genetic structure has equations for the animals (pedigree order) followed
by the metafounders (``meta['groups']`` order), like unknown-parent groups, so
the workflow reports the two parts separately.  Everything needed to audit
Gamma - source, provenance text or estimate with its diagnostics - is written
to ``meta`` and from there to the run manifest and report.
"""

from __future__ import annotations

import numpy as np

from ..core.genomic import apply_g_policy, vanraden_g
from ..core.metafounders import (MetafounderPedigree, estimate_gamma_gls, read_gamma_file,
                                 single_step_mf)
from ..core.model import GeneticStructure
from ..core.spec import AnalysisSpec
from ..core.upg import GroupAssignment
from ..errors import ABPError
from ..qc.pedigree import PedigreeData
from ..solvers.blup import RELIABILITY_ROUNDING_BAND


def assign_metafounders(ped_data: PedigreeData, default: str | None) -> GroupAssignment:
    """Metafounder of every unknown parent: its prefix code, else ``default``.

    Unknown parents that are neither coded nor covered by ``default`` stay
    unassigned; :class:`MetafounderPedigree` then refuses them (ABP-E205).
    """
    ped = ped_data.pedigree
    g = ped_data.groups
    labels = list(g.labels) if g is not None else []
    sg = g.sire_group.copy() if g is not None else np.full(ped.n, -1, dtype=np.int64)
    dg = g.dam_group.copy() if g is not None else np.full(ped.n, -1, dtype=np.int64)
    if default is not None:
        if default not in labels:
            labels.append(default)
        k = labels.index(default)
        sg[(ped.sire < 0) & (sg < 0)] = k
        dg[(ped.dam < 0) & (dg < 0)] = k
    used = sorted(set(sg[sg >= 0].tolist()) | set(dg[dg >= 0].tolist()))
    if len(used) != len(labels):                      # e.g. default never needed
        remap = np.full(len(labels), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        labels = [labels[j] for j in used]
        sg = np.where(sg >= 0, remap[np.maximum(sg, 0)], -1)
        dg = np.where(dg >= 0, remap[np.maximum(dg, 0)], -1)
    return GroupAssignment(tuple(labels), sg, dg)


def metafounder_structure(spec: AnalysisSpec, ped_data: PedigreeData, relationship: str,
                          manifest: dict, animals_with_records: set[str]) -> GeneticStructure:
    """``A^Gamma`` (relationship 'pedigree') or ``H`` on the metafounder base ('single_step')."""
    cfg = spec["metafounders"]
    ped = ped_data.pedigree
    groups = assign_metafounders(ped_data, cfg["default"])
    if not groups.labels:
        raise ABPError("SPEC_INVALID", f"[metafounders] is declared but no parent code starts "
                       f"with {cfg['prefix']!r} and metafounders.default is not set")
    n_coded = int((groups.sire_group >= 0).sum() + (groups.dam_group >= 0).sum())
    meta: dict = {"metafounders": list(groups.labels), "groups": list(groups.labels),
                  "n_animals": ped.n, "n_groups": len(groups.labels),
                  "n_unknown_parents_assigned": n_coded, "default_metafounder": cfg["default"],
                  "gamma_source": cfg["gamma_source"]}
    prep = None
    g_index = None
    if relationship == "single_step" or cfg["gamma_source"] == "genotypes_gls":
        from .genomic_inputs import prepare_genotypes
        prep = prepare_genotypes(spec, ped_data, manifest, animals_with_records)
        absent = [a for a in prep.geno.ids if not ped.contains(a)]
        if absent:
            raise ABPError("GENOTYPE_ID_CONFLICT", f"{len(absent)} genotyped animal(s) are not "
                           "in the pedigree", animals=absent[:20])
        g_index = ped.index_of(prep.geno.ids)
    if cfg["gamma_source"] == "file":
        path = spec.resolve(cfg["gamma_file"])
        gamma = read_gamma_file(path, groups.labels)
        from .manifest import sha256_path
        manifest["inputs"].append({"role": "metafounder_gamma", "path": str(path),
                                   "sha256": sha256_path(path), "rows": len(groups.labels)})
        meta["gamma_provenance"] = cfg["gamma_provenance"]
    else:
        est = estimate_gamma_gls(ped, groups, g_index, prep.geno.dosage, prep.geno.missing,
                                 sampling_correction=cfg["sampling_correction"])
        gamma = est.gamma
        meta["gamma_estimate"] = est.to_dict()
        meta["gamma_provenance"] = ("estimated in this run from the QC-passed genotypes "
                                    "(GLS base allele frequencies; Garcia-Baccino et al. 2017)")
    mfp = MetafounderPedigree(ped, groups, gamma)
    meta.update({"gamma": gamma.tolist(), "ml_kernel": mfp.kernel,
                 "group_fractions": mfp.Q,
                 "base": "metafounders: unknown parents descend from base populations with "
                         "relationships Gamma (Legarra et al. 2015)",
                 "inbreeding_mf_base": mfp.inbreeding(),
                 "mean_inbreeding_mf_base": float(np.mean(mfp.inbreeding()))})
    if relationship == "pedigree":
        return GeneticStructure("pedigree_mf", mfp.labels, mfp.ainv_ext(), mfp.diag_ext(),
                                mfp.logdet_ext(), meta)
    gcfg = spec["genomic"]
    G, d = vanraden_g(prep.geno.dosage, prep.p, prep.geno.missing)
    A22 = mfp.a_submatrix(g_index)
    Gs, rec = apply_g_policy(G, gcfg["singular_policy"], gcfg["tuning"], A22,
                             gcfg["blend_alpha"], gcfg["ridge"])
    off = ~np.eye(A22.shape[0], dtype=bool)
    meta.update({"method": "G05 = (M - 1)(M - 1)' / (m/2) (VanRaden method 1 with p = 0.5)",
                 "frequency_note": prep.freq_note, "n_genotyped": len(prep.geno.ids),
                 "n_markers": len(prep.geno.markers), "scaling_d": d,
                 "g_policy": rec.__dict__,
                 "compatibility": {"mean_diag_A22_gamma": float(np.mean(np.diag(A22))),
                                   "mean_offdiag_A22_gamma": float(np.mean(A22[off])),
                                   "mean_diag_Gstar": float(np.mean(np.diag(Gs))),
                                   "mean_offdiag_Gstar": float(np.mean(Gs[off]))},
                 "single_step": "H^-1 = A_Gamma^-1 + embed(G*^-1 - A22_Gamma^-1) over "
                                "(animals, metafounders)"})
    ss = single_step_mf(mfp, g_index, Gs, a22=A22)
    return GeneticStructure("single_step_mf", mfp.labels, ss.h_inv, ss.h_diag, ss.logdet_h, meta)


def reference_metafounder(cfg: dict, labels: list[str]) -> str:
    """The metafounder whose genetic level defines the reported base."""
    ref = cfg["reference"] or cfg["default"] or (labels[0] if len(labels) == 1 else None)
    if ref is None:
        raise ABPError("SPEC_INVALID", "several metafounders are used; declare "
                       "metafounders.reference (the base population EBVs are expressed against)",
                       metafounders=labels)
    if ref not in labels:
        raise ABPError("SPEC_INVALID", f"metafounders.reference {ref!r} is not used in the "
                       f"pedigree (metafounders: {labels})")
    return ref


def base_contrast(res, term: str, structure: GeneticStructure, sigma2: float,
                  cfg: dict) -> dict:
    """EBVs relative to the reference metafounder, with their PEV and reliability.

    Under metafounders ``u_i`` includes the unknown genetic level of the base
    population, a shift common to its descendants.  The estimable quantity a
    breeder uses is the contrast ``c_i = u_i - u_r`` with the reference
    metafounder ``r``:

    * ``PEV(c_i) = C^ii + C^rr - 2 C^ir`` (``C^{..}`` blocks of the MME
      inverse; one extra solve per reference);
    * prior ``Var(c_i) = sigma^2 (A_ii - 2 (Q Gamma)_ir + Gamma_rr)``, which is
      ``sigma^2 (1 - gamma/2)`` for a founder of ``r``;
    * reliability ``1 - PEV(c_i) / Var(c_i)`` (range-checked like BLUP).
    """
    m = structure.meta
    n = m["n_animals"]
    labels = list(m["metafounders"])
    ref = reference_metafounder(cfg, labels)
    r = labels.index(ref)
    tr = res.terms[term]
    ebv = tr.solution[:n] - tr.solution[n + r]
    out = {"reference": ref, "ebv": ebv, "pev": None, "reliability": None, "summary": None}
    if tr.pev is None or res.solve.factor is None:
        return out
    a, _ = res.system.offsets[term]
    e = np.zeros(res.system.n_equations)
    e[a + n + r] = 1.0
    col = res.solve.factor.solve(e)[a:a + n]
    pev = tr.pev[:n] + tr.pev[n + r] - 2.0 * col
    gamma = np.asarray(m["gamma"])
    prior = sigma2 * (np.asarray(structure.k_diag)[:n] - 2.0 * (m["group_fractions"] @ gamma)[:, r]
                      + gamma[r, r])
    rel_raw = 1.0 - pev / prior
    bad = (rel_raw < -RELIABILITY_ROUNDING_BAND) | (rel_raw > 1 + RELIABILITY_ROUNDING_BAND)
    if np.any(bad):
        k = int(np.flatnonzero(bad)[0])
        raise ABPError("RELIABILITY_OUT_OF_RANGE",
                       f"reliability relative to {ref} is {rel_raw[k]:.6g} for "
                       f"{tr.labels[k]!r}, outside [0, 1]", value=float(rel_raw[k]),
                       n_bad=int(bad.sum()))
    rel = np.clip(rel_raw, 0.0, 1.0)
    out.update(pev=pev, reliability=rel,
               summary={"mean": float(rel.mean()), "min": float(rel.min()),
                        "max": float(rel.max()),
                        "n_rounding_clamped": int(np.sum(rel != rel_raw))})
    return out
