"""PLINK 1 .bed reader: hand-derived bytes, round trip, errors, and identical
GBLUP results from PLINK and ABP dosage-matrix inputs."""

import csv
import shutil

import numpy as np
import pytest

from abp.errors import ABPError
from abp.examples.sheep import SheepSimConfig, write_sheep_example
from abp.io.plink import decode_bed, load_plink, write_bed
from abp.workflows.evaluate import run_evaluation


def test_hand_derived_bytes():
    """5 samples, 2 variants. Codes per the PLINK spec (low bits first):
    SNP1 A1-dosages [2,1,0,NA,2] -> codes 00,10,11,01 | 00 -> bytes 0x78, 0x00
    SNP2 A1-dosages [0,0,1,2,NA] -> codes 11,11,10,00 | 01 -> bytes 0x2F, 0x01"""
    raw = bytes([0x6C, 0x1B, 0x01, 0x78, 0x00, 0x2F, 0x01])
    M = decode_bed(raw, 5, 2)
    expected = np.array([[2, 0], [1, 0], [0, 1], [np.nan, 2], [2, np.nan]])
    np.testing.assert_array_equal(np.isnan(M), np.isnan(expected))
    np.testing.assert_array_equal(np.nan_to_num(M, nan=-1), np.nan_to_num(expected, nan=-1))


def test_round_trip_and_loader(tmp_path):
    rng = np.random.default_rng(0)
    D = rng.integers(0, 3, (13, 7)).astype(float)
    D[rng.random(D.shape) < 0.1] = np.nan
    markers = [(f"v{j}", "1", 100 * (j + 1), "G", "A") for j in range(7)]
    write_bed(tmp_path / "x", [f"s{i}" for i in range(13)], markers, D)
    g = load_plink(tmp_path / "x", "ASM-TEST")
    np.testing.assert_array_equal(g.missing, np.isnan(D))
    np.testing.assert_array_equal(g.dosage, np.nan_to_num(D, nan=0.0))
    assert g.counted_allele == ["G"] * 7 and g.ids[0] == "s0" and g.assembly == "ASM-TEST"
    assert {f.check for f in g.qc.findings} >= {"GEN-PLINK"}


def test_bad_files(tmp_path):
    with pytest.raises(ABPError) as e:
        decode_bed(bytes([0x00, 0x1B, 0x01, 0x00]), 4, 1)
    assert e.value.code == "ABP-E103"
    with pytest.raises(ABPError) as e:
        decode_bed(bytes([0x6C, 0x1B, 0x00, 0x00]), 4, 1)
    assert e.value.code == "ABP-E303"
    with pytest.raises(ABPError) as e:
        decode_bed(bytes([0x6C, 0x1B, 0x01, 0x00, 0x00]), 4, 1)       # one byte too many
    assert "expected 4" in e.value.message
    write_bed(tmp_path / "y", ["a", "b"], [("v", "1", 1, "C", "C")], np.array([[0.0], [1.0]]))
    with pytest.raises(ABPError) as e:
        load_plink(tmp_path / "y", "ASM")
    assert e.value.code == "ABP-E223"


def _read(p):
    with open(p, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_plink_and_dosage_inputs_give_identical_gblup(tmp_path):
    flock = tmp_path / "sheep"
    write_sheep_example(flock, seed=4, cfg=SheepSimConfig(
        n_founder_rams=4, n_founder_ewes=40, rams_per_year=4, years=(2021, 2022, 2023),
        snp_per_chrom=40, qtl_per_chrom=20, n_chrom=5, n_wwt_typos=0))
    geno = _read(flock / "data" / "genotypes.csv")
    mk = _read(flock / "data" / "markers.csv")
    ids = [r["id"] for r in geno]
    D = np.array([[np.nan if r[m["marker_id"]] == "NA" else float(r[m["marker_id"]]) for m in mk]
                  for r in geno])
    # counted allele of the ABP map = A1 in PLINK
    write_bed(flock / "data" / "geno", ids,
              [(m["marker_id"], m["chrom"], int(m["pos"]), m["counted_allele"],
                m["ref"] if m["counted_allele"] == m["alt"] else m["alt"]) for m in mk], D)
    common = '''schema_version = "1"
[project]
name = "p"
species = "sheep"
synthetic_data = true
[analysis]
task = "additive_ebv"
target_population = "t"
information_cutoff = "2025-12-31"
genetic_base = "genotyped"
[data]
phenotypes = "sheep/data/lambs.csv"
{geno}
[[traits]]
name = "wwt"
unit = "kg"
[model]
traits = ["wwt"]
fixed = [{{ column = "cg", type = "factor" }}]
random = [{{ name = "animal", kind = "additive", relationship = "genomic" }}]
[variances]
mode = "known"
values = {{ animal = 3.0, residual = 10.0 }}
[genomic]
singular_policy = "ridge"
ridge = 0.01
[qc]
ungenotyped_records = "exclude"
'''
    a = tmp_path / "a.toml"
    a.write_text(common.format(geno='genotypes = "sheep/data/genotypes.csv"\n'
                                    'marker_map = "sheep/data/markers.csv"'), encoding="utf-8")
    b = tmp_path / "b.toml"
    b.write_text(common.format(geno='plink = "sheep/data/geno"\n'
                                    'genotype_assembly = "SYNTHETIC-OVINE-ASSEMBLY-0"'),
                 encoding="utf-8")
    ra = run_evaluation(a, tmp_path / "ra", console=False)
    rb = run_evaluation(b, tmp_path / "rb", console=False)
    assert (ra.out_dir / "ebv_wwt.csv").read_bytes() == (rb.out_dir / "ebv_wwt.csv").read_bytes()
    roles = {i["role"] for i in rb.manifest["inputs"]}
    assert {"genotypes", "marker_map", "sample_list"} <= roles
    assert rb.manifest["relationship"]["counted_allele"] == "A1 of the PLINK .bim file"
    # abp validate reads the PLINK fileset as well (genotype QC without fitting)
    from abp.workflows.validate import validate_inputs
    summary = validate_inputs(b)
    assert any(f["check"] == "GEN-PLINK" for f in summary["genotypes"]["findings"])
