# Examples

All data here are **synthetic** or taken from a textbook teaching example.
Each example is run end to end by the test suite (`tests/test_examples.py`,
`tests/test_workflow.py`, `tests/test_ocs_mating.py`,
`tests/test_bayes_workflow.py`, `tests/test_upg_workflow.py`,
`tests/test_metafounder_workflow.py`, `tests/test_multitrait_reml.py`,
`tests/test_threshold.py`), so the
commands below are known to work.

| # | Folder | Analysis | Command |
|---|---|---|---|
| 01 | `01_textbook_mrode_3_1` | single-trait animal model, known variances (Mrode 2005, Ex. 3.1: 8 animals, 5 records) | `abp run examples/01_textbook_mrode_3_1/analysis.toml --out runs/ex01` |
| 02 | `02_sheep_wwt_reml` | weaning weight, pedigree animal model, REML; QC quarantines 3 typing errors | `abp run examples/02_sheep_wwt_reml/analysis.toml --out runs/ex02` |
| 03 | `03_sheep_nlb_repeatability` | litter size at every lambing, repeatability model (animal + permanent environment), REML | `abp run examples/03_sheep_nlb_repeatability/analysis.toml --out runs/ex03` |
| 04 | `04_sheep_fec_gblup` | faecal egg count, GBLUP with explicit G blending, genotyped animals only | `abp run examples/04_sheep_fec_gblup/analysis.toml --out runs/ex04` |
| 05 | `05_sheep_wwt_single_step` | weaning weight, single-step GBLUP with G tuned to A22, REML | `abp run examples/05_sheep_wwt_single_step/analysis.toml --out runs/ex05` |
| 06 | `06_sheep_multitrait_index` | growth, carcass, health and reproduction in one four-trait model + economic index (synthetic weights) | `abp run examples/06_sheep_multitrait_index/analysis.toml --out runs/ex06` |
| 07 | `07_selection_index` | stand-alone Smith-Hazel index (spec gold standard T02) | `abp index examples/07_selection_index/index.toml` |
| 08 | `08_sheep_wwt_lr_validation` | weaning weight with forward-in-time LR validation (cutoff end of 2024, REML on the partial data) | `abp run examples/08_sheep_wwt_lr_validation/analysis.toml --out runs/ex08` |
| 09 | `09_sheep_mating` | optimal contributions and a mating plan for 150 ewes from the index of example 06 (ΔF 1%, no half-sib matings, recessive-risk limit); `make_candidates.py` rebuilds the candidate file | `abp mate examples/09_sheep_mating/mating.toml --out runs/mating` |
| 10 | `10_sheep_fec_bayesc` | faecal egg count, BayesC (π₀ = 0.95), 4 chains, convergence-gated | `abp run examples/10_sheep_fec_bayesc/analysis.toml --out runs/ex10` |
| 11 | `11_sheep_upg` | a flock buying rams from two breeders without ancestry: random (`analysis.toml`, REML) and fixed (`analysis_fixed.toml`) genetic groups; `make_data.py` regenerates the data, `compare.py` compares with and without groups against the truth | `abp run examples/11_sheep_upg/analysis.toml --out runs/ex11` |
| 12 | `12_sheep_wwt_single_step_metafounder` | weaning weight, single step on a metafounder base: all unknown parents from base population `MF:BASE`, γ estimated from the genotypes, `G05` without rescaling, REML; EBVs also against the base (`ebv_vs_base`) | `abp run examples/12_sheep_wwt_single_step_metafounder/analysis.toml --out runs/ex12` |
| 13 | `13_sheep_multitrait_reml` | weaning weight, fat depth and faecal egg count: genetic and residual covariance matrices by multi-trait REML, then multi-trait BLUP | `abp run examples/13_sheep_multitrait_reml/analysis.toml --out runs/ex13` |
| 14 | `14_sheep_nlb_threshold` | litter size (1/2/3 lambs) with a threshold (probit) repeatability model on the liability scale | `abp run examples/14_sheep_nlb_threshold/analysis.toml --out runs/ex14` |
| 15 | `15_sheep_wwt_nlb1_threshold` | weaning weight (continuous) and litter size at first lambing (1/2/3, ewes only) in one multi-trait threshold model: (co)variances, thresholds and EBVs by Gibbs sampling, inverse-Wishart prior (about 8-15 minutes) | `abp run examples/15_sheep_wwt_nlb1_threshold/analysis.toml --out runs/ex15` |
| - | `api_example.py` | the Python API, step by step | `python examples/api_example.py` |

### Round-5 variants (edit one line of an existing example)

* **Example 05, matrix-free single step:** set `[variances] mode = "known"`
  with `values = { animal = 4.0, residual = 12.25 }`, `[genomic] tuning = "none"`,
  `single_step_mode = "matrix_free"` (optionally `apy_core_size = 150`) and
  `[solver] pev = "none"`. The EBVs equal those of the explicit single step
  with the same settings (`tests/test_single_step_matrix_free.py`).
* **Example 13, reduced rank:** add `boundary = "reduced_rank"` to `[reml]`;
  it only acts when the full-rank fit stops at the boundary (`ABP-E300`).
* **Example 14, estimated liability variances:** replace the `[variances]`
  block by `mode = "reml"` (Laplace approximation; biased when animals have few
  records, see the manual §7.14).
* **Example 14, Gibbs sampler (round 6):** replace the `[variances]` block by
  `mode = "bayes"` and add `[bayes]` with `method = "threshold"`,
  `iterations = 4000`, `burn_in = 1000`, `thin = 2`, `max_iterations = 32000`
  (about 1.5 minutes; see the manual §7.14).
* **Example 05, sampled reliabilities (round 6):** in the matrix-free variant
  above use `pev = "sampled"` and e.g. `pev_samples = 300`.
* **Example 13 (round 6):** its REML run writes
  `pev_incl_vc_uncertainty_<trait>` and `reliability_incl_vc_uncertainty_<trait>`.
* **Example 02:** its REML run already writes `pev_incl_vc_uncertainty` and
  `reliability_incl_vc_uncertainty` to `ebv_wwt.csv`.

## The synthetic sheep flock (`sheep_data/`)

Generated by `abp simulate-sheep --out examples/sheep_data --seed 20260925`
(re-running the command reproduces the files byte for byte on the same
software versions):

* two flocks, founders born 2018–2019, lamb crops 2021–2025, 2,108 animals;
* `data/pedigree.csv`, `data/lambs.csv` (1,896 lambs: weaning weight, fat
  depth, log faecal egg count, first-parity litter size of ewes),
  `data/lambing.csv` (1,208 lambing records), `data/genotypes.csv` (476
  animals × 2,000 SNPs), `data/markers.csv`;
* `truth/`: simulated true breeding values and parameters. **Analyses never
  read this folder.** Only `benchmarks/simulation_check.py` uses it, to compare
  EBVs with the truth.

The generator uses gene dropping with recombination and QTL effects,
independently of ABP's relationship code (see `src/abp/examples/sheep.py`).
None of its parameters describes a real breed.

## The flock with purchased rams (`11_sheep_upg/`)

Generated by `python examples/11_sheep_upg/make_data.py` (seed 20260925;
the test suite checks that it reproduces the shipped files byte for byte):
a closed flock of 150 ewes (base animals born 2011–2015) buying three rams a
year from two breeders, lamb crops 2016–2024, 1,533 animals and 1,350
weaning weights. `truth/` holds the simulated breeding values and group
means; only `compare.py` and `benchmarks/upg_study.py` read it.
