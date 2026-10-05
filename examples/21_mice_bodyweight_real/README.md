# Example 21: real mouse data (BGLR `mice`), GBLUP

**Real data, not stored in this repository.** Fetch and convert them first:

```bash
pip install rdata                                   # MIT licence; used only by fetch_data.py
python examples/21_mice_bodyweight_real/fetch_data.py
abp run examples/21_mice_bodyweight_real/analysis_bw_gblup.toml --out out21
```

## Source and licence

* `mice.X`, `mice.pheno`, `mice.map` and `mice.A` from the R package **BGLR 1.1.4**
  (de los Campos, Perez Rodriguez; licence GPL-3), downloaded from the read-only GitHub
  mirror of CRAN (`raw.githubusercontent.com/cran/BGLR/1.1.4/data/mice.RData`); SHA-256
  checked by the script.
* 1,814 heterogeneous-stock mice of the Wellcome Trust experiment (Valdar et al. 2006,
  Nat Genet 38:879; Genetics 174:959), genotyped for 10,346 SNPs; used for genomic
  prediction by Legarra et al. (2008, Genetics 180:611).
* Because the project licence is not chosen yet and the data are distributed under the
  package's GPL-3, the data are downloaded on demand and not committed.

## Conversion (`fetch_data.py`)

* `data/genotypes.csv`: dosage 0/1/2 of the counted allele, which is the suffix of the
  SNP name (`rs3683945_G` counts G). `data/markers.csv`: the source map (`mbp` positions);
  the genome assembly is not stated by the source and is recorded as such.
* `data/phenotypes.csv`: `id, sex, year, season, litter, cage, bw, body_length, bmi,
  year_season, test_date` (body weight in g, body length in cm). `test_date` is the last
  day of the test season, for forward-in-time validation. It rests on an **assumption the
  source does not state**: a year's "winter" is its first quarter (winter < spring <
  summer < autumn).
* `data/pedigree_A.npy`: the pedigree relationship matrix shipped with the data (used by
  `benchmarks/real_mice_validation.py` for the PBLUP comparison; the pedigree itself is
  not in the package).

## Analysis

`analysis_bw_gblup.toml`: body weight, sex and test year × season fixed, cage as an
independent random effect, genomic additive effect (VanRaden G, frequencies from all
genotyped mice, `G + 0.01 I` because there is no pedigree to blend with), REML. About
30 s. REML estimates (g²): additive 2.13, cage 2.17, residual 3.50 (h² 0.27).

`analysis_bw_lr.toml` (round 16): forward-in-time validation by the LR method. The 677
mice tested in 2004 are hidden in a partial evaluation (cutoff 2003-12-31; variances by
REML on the 1,137 mice tested in 2003), and their partial EBVs are compared with the
whole evaluation. About 4 minutes. Result: bias −0.010 ± 0.026 g, dispersion 0.90 ± 0.02,
ρ 0.869 ± 0.009 (validation report §7.20).

Cross-validated predictive ability (GBLUP vs PBLUP) is in the validation report §7.19.
Results are for method validation only (gate G5).
