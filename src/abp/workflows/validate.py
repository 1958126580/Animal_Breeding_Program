"""``abp validate``: spec validation and full QC without fitting a model."""

from __future__ import annotations

from ..core.spec import load_spec
from ..io.tables import read_table
from ..qc.pedigree import PedigreeColumns, load_pedigree
from ..qc.phenotype import load_phenotypes


def validate_inputs(spec_path) -> dict:
    """Return a QC summary; raises ABPError on the first blocking problem class."""
    spec = load_spec(spec_path)
    d = spec.data
    data = d["data"]
    unknown_parent = set(data["unknown_parent_values"])
    phe = read_table(spec.resolve(data["phenotypes"]), data["delimiter"])
    ped_ids = None
    ped_table = None
    if data["pedigree"]:
        pc = data["pedigree_columns"]
        ped_table = read_table(spec.resolve(data["pedigree"]), data["delimiter"])
        ped_ids = set(ped_table.column(pc["id"])) | (
            (set(ped_table.column(pc["sire"])) | set(ped_table.column(pc["dam"]))) - unknown_parent)
        if d["upg"] is not None:
            ped_ids = {a for a in ped_ids if not a.startswith(d["upg"]["prefix"])}
    records, phe_qc, to_add = load_phenotypes(phe, d, ped_ids)
    summary = {"spec": str(spec.path), "spec_sha256": spec.sha256, "status": "passed",
               "phenotypes": phe_qc.to_dict()}
    if ped_table is not None:
        pc = data["pedigree_columns"]
        ped = load_pedigree(ped_table, PedigreeColumns(pc["id"], pc["sire"], pc["dam"], pc["sex"],
                                                       pc["birth_date"]),
                            unknown_parent, set(data["missing_values"]), extra_founders=to_add,
                            group_prefix=d["upg"]["prefix"] if d["upg"] else None)
        summary["pedigree"] = ped.qc.to_dict()
    if data["genotypes"] or data["plink"]:
        from .genomic_inputs import load_genotype_inputs
        geno = load_genotype_inputs(spec)
        summary["genotypes"] = geno.qc.to_dict()
    return summary
