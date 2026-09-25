"""Translate validated records + relationship structures into model matrices.

This module is the "model compiler" of the platform: from the analysis spec
it produces the response vector, the full-rank fixed design and the random
terms with their covariance structures, and checks dimensions and index
mappings on the way.  It never estimates anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import scipy.sparse as sp

from ..errors import ABPError
from ..qc.phenotype import RecordSet
from ..solvers.blup import RandomTerm
from .design import FixedDesign, FixedTerm, build_fixed_design
from .pedigree import Pedigree


@dataclass
class GeneticStructure:
    """Covariance structure ``K`` of the additive genetic term.

    ``labels`` are the animal IDs in equation order; ``k_inv`` is ``K^{-1}``;
    ``k_diag`` is ``diag(K)`` (for reliabilities) and ``logdet_k`` is
    ``log|K|`` (for REML).  ``meta`` records how ``K`` was built.
    """

    kind: str
    labels: tuple[str, ...]
    k_inv: sp.spmatrix | np.ndarray
    k_diag: np.ndarray
    logdet_k: float | None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.index = {a: i for i, a in enumerate(self.labels)}
        q = len(self.labels)
        if self.k_inv.shape != (q, q) or self.k_diag.shape != (q,):
            raise ValueError("inconsistent genetic structure dimensions")


def pedigree_structure(ped: Pedigree) -> GeneticStructure:
    """``K = A`` with sparse ``A^{-1}`` (Henderson/Quaas rules incl. inbreeding)."""
    return GeneticStructure("pedigree", ped.ids, ped.ainv(), 1.0 + ped.inbreeding(),
                            ped.logdet_a(),
                            {"inbreeding_kernel": ped.inbreeding_kernel,
                             "base": "unknown parents = unrelated, non-inbred base animals"})


@dataclass
class SingleTraitModel:
    trait: str
    unit: str
    y: np.ndarray
    records: np.ndarray            #: indices of used records in the RecordSet
    fixed: FixedDesign
    terms: list[RandomTerm]
    genetic_term: str
    structure: GeneticStructure
    n_records_per_animal: dict[str, int]


def _incidence(values: Sequence[str], levels_index: dict[str, int], q: int) -> sp.csr_matrix:
    n = len(values)
    cols = np.fromiter((levels_index[v] for v in values), dtype=np.int64, count=n)
    return sp.csr_matrix((np.ones(n), (np.arange(n), cols)), shape=(n, q))


def build_single_trait(records: RecordSet, trait: str, spec: dict,
                       structure: GeneticStructure) -> SingleTraitModel:
    """Build ``y``, the fixed design and all random terms for one trait."""
    model = spec["model"]
    y_all = records.traits[trait]
    used = np.flatnonzero(~np.isnan(y_all))
    if used.size == 0:
        raise ABPError("NO_USABLE_RECORDS", f"no records for trait {trait!r}", trait=trait)
    y = y_all[used].astype(np.float64)
    n = used.size
    cols: dict[str, Sequence] = {}
    terms_f: list[FixedTerm] = []
    for f in model["fixed"]:
        c = f["column"]
        if f["type"] == "factor":
            cols[c] = [records.factors[c][k] for k in used]
        else:
            cols[c] = records.covariates[c][used]
        terms_f.append(FixedTerm(c, f["type"]))
    fixed = build_fixed_design(cols, terms_f, intercept=model["intercept"], n=n)

    animals = [records.animal[k] for k in used]
    missing = sorted({a for a in animals if a not in structure.index})
    if missing:
        raise ABPError("PHENOTYPE_UNKNOWN_ANIMAL",
                       f"{len(missing)} animal(s) with records for {trait!r} are not in the "
                       f"{structure.kind} relationship structure (e.g. {missing[:5]})",
                       animals=missing[:50], structure=structure.kind)
    terms: list[RandomTerm] = []
    genetic_name = ""
    for r in model["random"]:
        if r["kind"] == "additive":
            genetic_name = r["name"]
            Z = _incidence(animals, structure.index, len(structure.labels))
            terms.append(RandomTerm(r["name"], Z, structure.k_inv, structure.labels, genetic=True,
                                    k_diag=structure.k_diag, logdet_k=structure.logdet_k))
        else:
            vals = [records.factors[r["column"]][k] for k in used]
            levels = sorted(set(vals))
            idx = {lv: i for i, lv in enumerate(levels)}
            Z = _incidence(vals, idx, len(levels))
            terms.append(RandomTerm(r["name"], Z, sp.identity(len(levels), format="csr"),
                                    tuple(levels), genetic=False, k_diag=np.ones(len(levels)),
                                    logdet_k=0.0))
    counts: dict[str, int] = {}
    for a in animals:
        counts[a] = counts.get(a, 0) + 1
    unit = records.units.get(trait, "")
    return SingleTraitModel(trait, unit, y, used, fixed, terms, genetic_name, structure, counts)
