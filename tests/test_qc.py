"""QC: each rule detects its seeded error, and clean data raise no false alarm."""

from pathlib import Path

import numpy as np
import pytest

from abp.errors import ABPError
from abp.io.tables import read_table
from abp.qc.genotype import filter_genotypes, load_genotypes
from abp.qc.pedigree import PedigreeColumns, load_pedigree

UNKNOWN = {"0", "", "NA"}
COLS = PedigreeColumns("id", "sire", "dam", "sex", "birth_date")


def _ped(tmp_path: Path, text: str):
    p = tmp_path / "ped.csv"
    p.write_text(text, encoding="utf-8")
    return load_pedigree(read_table(p), COLS, UNKNOWN, {"", "NA"})


CLEAN = ("id,sire,dam,sex,birth_date\nS,0,0,M,2019\nD,0,0,F,2019\n"
         "A,S,D,F,2021-03-01\nB,S,D,M,2021-03-01\n")


def test_clean_pedigree_has_no_blocking_or_review_findings(tmp_path):
    pd = _ped(tmp_path, CLEAN)
    assert not [f for f in pd.qc.findings if f.severity in ("error", "review")]
    assert pd.qc.stats["n_animals"] == 4


@pytest.mark.parametrize("text,code,check", [
    (CLEAN + "A,S,B,F,2021-03-01\n", "ABP-E204", "PED-DUP-CONFLICT"),
    (CLEAN + "C,C,D,F,2022\n", "ABP-E201", "PED-SELF-PARENT"),
    (CLEAN + "C,A,D,F,2022\n", "ABP-E202", "PED-SEX-CONFLICT"),          # A is F, used as sire
    (CLEAN + "C,S,S,F,2022\n", "ABP-E202", "PED-SAME-SIRE-DAM"),
    (CLEAN + "C,B,A,M,2020\n", "ABP-E203", "PED-BIRTH-ORDER"),           # parents born 2021
    (CLEAN + "C,B,A,X,2022\n", "ABP-E103", "PED-SEX-CODE"),
    (CLEAN + "C,B,A,M,22/03/2022\n", "ABP-E103", "PED-DATE"),
    (CLEAN + " C,B,A,M,2022\n", "ABP-E105", "PED-ID"),
])
def test_seeded_pedigree_errors(tmp_path, text, code, check):
    with pytest.raises(ABPError) as exc:
        _ped(tmp_path, text)
    assert exc.value.code == code
    assert check in {f["check"] for f in exc.value.details["findings"]}


def test_identical_duplicates_and_missing_parents_are_reported_not_hidden(tmp_path):
    pd = _ped(tmp_path, CLEAN + "B,S,D,M,2021-03-01\nC,X9,D,M,2022\n")
    checks = {f.check: f for f in pd.qc.findings}
    assert checks["PED-DUP-IDENTICAL"].severity == "review"
    assert checks["PED-FOUNDER-ADDED"].examples == [{"animal": "X9"}]
    assert pd.pedigree.contains("X9") and pd.sex["X9"] == "M"


def test_all_problems_reported_together(tmp_path):
    with pytest.raises(ABPError) as exc:
        _ped(tmp_path, CLEAN + "C,C,D,F,2022\nE,A,D,F,2022\n")
    checks = {f["check"] for f in exc.value.details["findings"]}
    assert {"PED-SELF-PARENT", "PED-SEX-CONFLICT"} <= checks


def _geno(tmp_path, dos_rows, map_rows=None, header=None):
    mp = tmp_path / "map.csv"
    map_rows = map_rows or ["m1,1,100,A,G,G,ASM1", "m2,1,200,C,T,T,ASM1", "m3,2,50,A,C,C,ASM1"]
    mp.write_text("marker_id,chrom,pos,ref,alt,counted_allele,assembly\n" + "\n".join(map_rows) + "\n",
                  encoding="utf-8")
    gp = tmp_path / "geno.csv"
    gp.write_text((header or "id,m1,m2,m3") + "\n" + "\n".join(dos_rows) + "\n", encoding="utf-8")
    return load_genotypes(read_table(gp), read_table(mp), {"NA"})


CFG = {"min_call_rate_animal": 0.5, "min_call_rate_marker": 0.5, "min_maf": 0.0}


def test_genotype_contract_errors(tmp_path):
    with pytest.raises(ABPError) as e:
        _geno(tmp_path, ["a,0,1,3"])
    assert e.value.code == "ABP-E220"
    with pytest.raises(ABPError) as e:
        _geno(tmp_path, ["a,0,1,2"], header="id,m1,m2,m9")
    assert e.value.code == "ABP-E222"
    with pytest.raises(ABPError) as e:
        _geno(tmp_path, ["a,0,1,2"], map_rows=["m1,1,100,A,G,T,ASM1", "m2,1,200,C,T,T,ASM1",
                                               "m3,2,50,A,C,C,ASM1"])
    assert e.value.code == "ABP-E223"
    with pytest.raises(ABPError) as e:
        _geno(tmp_path, ["a,0,1,2"], map_rows=["m1,1,100,A,G,G,ASM1", "m2,1,200,C,T,T,ASM2",
                                               "m3,2,50,A,C,C,ASM1"])
    assert e.value.code == "ABP-E223"
    with pytest.raises(ABPError) as e:
        _geno(tmp_path, ["a,0,1,2", "a,1,1,1"])
    assert e.value.code == "ABP-E106"


def test_genotype_filters_and_ambiguous_flag(tmp_path):
    g = _geno(tmp_path, ["a,0,1,2", "b,NA,1,2", "c,NA,2,2", "d,NA,0,2"],
              map_rows=["m1,1,100,A,T,T,ASM1", "m2,1,200,C,T,T,ASM1", "m3,2,50,A,C,C,ASM1"])
    assert "GEN-AMBIGUOUS" in {f.check for f in g.qc.findings}
    out, p = filter_genotypes(g, CFG)
    checks = {f.check for f in out.qc.findings}
    assert "GEN-CALLRATE-MARKER" in checks          # m1 has call rate 0.25
    assert "GEN-MONOMORPHIC" in checks              # m3 all 2
    assert out.markers == ["m2"] and np.allclose(p, [0.5])
    assert {e["reason"] for e in out.qc.excluded} == {"GEN-CALLRATE-MARKER", "GEN-MONOMORPHIC"}


def test_mendelian_conflict_detects_sample_swap(tmp_path):
    rng = np.random.default_rng(0)
    m = 200
    sire = rng.integers(0, 3, m)
    dam = rng.integers(0, 3, m)
    kid = np.array([int(rng.random() < s / 2) + int(rng.random() < d / 2)
                    for s, d in zip(sire, dam)])
    other = 2 - sire  # an unrelated animal with many opposing homozygotes vs the sire
    rows = [",".join(["S"] + [str(v) for v in sire]), ",".join(["D"] + [str(v) for v in dam]),
            ",".join(["K"] + [str(v) for v in kid]), ",".join(["X"] + [str(v) for v in other])]
    names = [f"m{j}" for j in range(m)]
    maps = [f"{n},1,{j + 1},A,G,G,ASM1" for j, n in enumerate(names)]
    g = _geno(tmp_path, rows, map_rows=maps, header="id," + ",".join(names))
    out, _ = filter_genotypes(g, CFG, parents={"K": ("S", "D")})
    assert not [f for f in out.qc.findings if f.check == "GEN-MENDEL"]
    with pytest.raises(ABPError) as e:
        filter_genotypes(g, CFG, parents={"K": ("S", "D"), "X": ("S", None)})
    assert e.value.code == "ABP-E222"
