"""``abp pedigree``: stand-alone pedigree QC, inbreeding and A-inverse export."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..io.tables import read_table, write_csv
from ..qc.pedigree import PedigreeColumns, load_pedigree, mean_inbreeding_by_generation
from . import manifest as mf
from .outputs import OutputStage, atomic_write_json


def pedigree_report(pedigree: str | Path, out: str | Path, id_col: str = "id",
                    sire_col: str = "sire", dam_col: str = "dam", sex_col: str | None = None,
                    birth_col: str | None = None, delimiter: str = ",",
                    unknown=("0", "", "NA", "."), force: bool = False) -> Path:
    """Write ``inbreeding.csv``, ``ainv_triplets.csv`` (1-based, lower triangle)
    and ``qc_pedigree.json`` for a pedigree file."""
    run_id = mf.new_run_id()
    table = read_table(pedigree, delimiter)
    data = load_pedigree(table, PedigreeColumns(id_col, sire_col, dam_col, sex_col, birth_col),
                         set(unknown), {"", "NA", "."})
    ped = data.pedigree
    stage = OutputStage(Path(out), run_id, force, 50.0)
    try:
        F = ped.inbreeding()
        D = ped.mendelian_d()
        write_csv(stage.path("inbreeding.csv"),
                  ["order", "animal", "sire", "dam", "generation", "inbreeding", "mendelian_d"],
                  [[i + 1, a, ped.ids[ped.sire[i]] if ped.sire[i] >= 0 else "",
                    ped.ids[ped.dam[i]] if ped.dam[i] >= 0 else "", int(ped.generation[i]),
                    float(F[i]), float(D[i])] for i, a in enumerate(ped.ids)])
        Ai = ped.ainv().tocoo()
        low = Ai.row >= Ai.col
        order = np.lexsort((Ai.col[low], Ai.row[low]))
        write_csv(stage.path("ainv_triplets.csv"), ["row", "col", "value"],
                  zip((Ai.row[low][order] + 1).tolist(), (Ai.col[low][order] + 1).tolist(),
                      Ai.data[low][order].tolist()))
        qc = data.qc.to_dict()
        qc["stats"]["inbreeding_by_generation"] = mean_inbreeding_by_generation(ped)
        qc["input"] = {"path": str(table.path), "sha256": table.sha256}
        qc["note"] = ("ainv_triplets.csv uses the 1-based 'order' of inbreeding.csv; "
                      "log|A| = %.12g" % ped.logdet_a())
        atomic_write_json(stage.path("qc_pedigree.json"), qc)
        return stage.publish()
    except BaseException:
        stage.abandon("failed")
        raise
