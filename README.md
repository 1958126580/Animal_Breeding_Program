# ABP: auditable genetic evaluation for animal breeding

ABP is a command-line tool and Python library for estimating breeding values
and planning selection. It covers pedigree and genomic BLUP, single-step
GBLUP (explicit or matrix-free), APY, REML variance components (single- and
multi-trait, with reduced-rank genetic covariance matrices at the boundary),
PEV that includes the uncertainty of REML variances, multi-trait BLUP,
threshold models for categorical traits (alone or together with continuous
traits), unknown-parent groups,
metafounders,
Bayesian marker models (BayesA/B/C/Cπ/R, Bayesian ridge regression),
forward-in-time validation, selection indices, optimal contribution selection
and mating plans. Every result can be traced back to its inputs, model and
code. It is built to the project's research and
development specification (`docs/` and the uploaded instruction set).

> **Status: 0.12.0, research-grade.** Every method listed below has passed
> analytical and independent-reference tests on Linux; simulation studies
> cover EBV calibration, genetic groups and the Bayesian samplers. It has
> **not** been validated on real breeding data, compared with
> BLUPF90/MiXBLUP/ASReml/JWAS/BGLR, or run on CUDA. Single-step GBLUP with
> G tuned to A22 was biased in the calibration scenario (finding F6); on a
> metafounder base it is unbiased, and calibrated with a dense marker panel
> (10,000 SNPs); with 2,000 SNPs its PEV is about 8% too small.
> Liability variances of the threshold model are best estimated with the
> Gibbs sampler (unbiased for the genetic variance in the simulation study);
> the Laplace approximation was biased; with 2–3 records per animal the
> data identify these variances weakly, so a proper prior largely decides them.
> CI runs the test suite on Linux and Windows (Python 3.11–3.13).
> ABP makes no claim of superiority over any other software. See
> [`docs/validation_report.md`](docs/validation_report.md) for exactly what has
> and has not been verified.

## What it does

| Area | Capability (0.11) | Evidence |
|---|---|---|
| Data contracts and QC | CSV import with strict schemas; pedigree QC (cycles, duplicates, sex and birth-order conflicts, missing parents); phenotype QC (ranges, repeated records, outliers); genotype QC (allele and assembly contract, call rates, MAF, Mendelian conflicts). Excluded records are always listed. | `tests/test_qc.py`, `tests/test_workflow.py` |
| Pedigree relationships | Ordering, inbreeding (Meuwissen-Luo; optional C++20 kernel that switches per pedigree depth to Colleau columns: 200,000 animals in 1.8 s), sparse A-inverse, `A x` products without forming A (Colleau); unknown-parent groups (random or estimable fixed) by the QP transformation; metafounders (related base populations; Γ from a documented file or estimated from genotypes) for pedigree BLUP, REML and single step | spec gold standard T03, independent tabular method; explicit-model references for groups and metafounders; 20-replicate group study; 50-replicate single-step study |
| BLUP | Single-trait animal and repeatability models; rank-deficient fixed effects; dense, sparse-direct and PCG solvers; exact PEV, SEP and reliability at scale by ABP's sparse LDL' and selected inversion (100,500 equations in 3.2 s); threshold (probit) model for ordered categorical traits, alone or with continuous traits in a multi-trait Gibbs sampler; PEV and reliability including the uncertainty of REML variances (Kackar–Harville) | T04, Mrode (2005) Ex. 3.1, independent V-form reference, 50-replicate calibration study |
| REML | Average-information REML with EM fallback, boundary (zero-variance) handling, standard errors, checkpoint and resume; dense or sparse (selected-inversion) traces; multi-trait AI-REML for G0 and R0 with missing traits; reduced-rank G0 = ΛΛ′ at the boundary (analytic gradient) and rank selection by AIC with a conservative margin; PEV including REML uncertainty for full-rank and reduced-rank multi-trait models; threshold-model variances by Gibbs sampling (uniform or proper priors) or Laplace-approximate REML | T05, independent optimizer, 40-replicate simulation, V-form references with singular G0, independent Laplace computation |
| Genomics | Dosage CSV or PLINK 1 binary input, optionally held as int8 (2 bytes per call instead of 9); VanRaden G with recorded frequency source; explicit singular-G policy; GBLUP; GBLUP/SNP-BLUP equivalence; single-step H-inverse; APY; matrix-free single step (H⁻¹ as an operator: 1.7 GB instead of 10.9 GB for 50,000 animals / 6,000 genotyped; 200,000 animals / 30,000 genotyped in 6.75 GB) with sampled reliabilities (variance-reduced estimator) | T01, T06, T15, explicit-H reference, hand-decoded PLINK bytes |
| Bayesian marker models | BRR, BayesA, BayesB, BayesC, BayesCπ, BayesR (Gibbs, 4+ chains, optional C++ sweep); R-hat, bulk/tail ESS and MCSE for every scalar and GEBV; results withheld unless converged; traces and posterior predictive checks | exact Gaussian posterior (BRR), exact inclusion probability, simulation-based calibration of all six samplers, diagnostics equal to ArviZ to 1e-15 |
| Validation | Forward-in-time LR method: bias, dispersion, correlation of partial vs whole EBVs with bootstrap intervals; hidden records cannot leak | hand-computed statistics, leakage test |
| Multi-trait | Multi-trait BLUP with known covariances, trait-specific fixed effects, missing traits, per-animal PEV blocks; Bayesian multi-trait linear model (covariance matrices sampled by Gibbs, posterior PEV), also with repeated records and a permanent-environment term and with maternal genetic effects (maternal animal model); residual covariances fixed at zero between groups of traits in the Gibbs samplers | 2×2 Kronecker-order gold standard, V-form reference |
| Decisions | Smith-Hazel and restricted indices; economic index on multi-trait EBVs with reliability; optimal contribution selection with a coancestry ceiling; mating plans minimising progeny inbreeding under relationship and recessive-risk limits (`abp mate`, proposals only) | T02, KKT certificates, independent optimizer, enumeration of small mating problems |
| Engineering | Atomic outputs, run manifests with hashes, report generated from recorded results, stable error codes and exit statuses, UTF-8 and Chinese/space paths, cancellation, disk check, memory budget | `tests/test_workflow.py`, `tests/test_manifest_contract.py` |

Not implemented yet (tracked as `not_run` in
[`docs/method_registry.toml`](docs/method_registry.toml)): maternal models with
REML or known variances (the Bayesian maternal animal model exists), random-regression models, threshold models with several categorical traits, survival models,
single-step Bayesian models, genomic OCS,
VCF/BGEN readers, CUDA.

## Install

Requires Python 3.11 or newer on Linux or Windows.

```bash
git clone https://github.com/1958126580/Animal_Breeding_Program.git
cd Animal_Breeding_Program
python -m pip install -e ".[test]"   # builds the optional C++20 kernel if a compiler is present
abp selftest                         # analytical installation check; must print RESULT: PASS
```

For hash-pinned dependencies (reproducible or offline installs) see
`requirements-lock.txt` and the [user manual](docs/user_manual.md#2-installation).

## Quick start

```bash
# 1. the textbook example (5 records, known variances)
abp run examples/01_textbook_mrode_3_1/analysis.toml --out runs/ex01

# 2. a synthetic sheep flock: REML + BLUP for weaning weight
abp run examples/02_sheep_wwt_reml/analysis.toml --out runs/ex02

# 3. four-trait evaluation (growth, carcass, health, reproduction) + economic index
abp run examples/06_sheep_multitrait_index/analysis.toml --out runs/ex06

# 4. genetic groups for purchased rams; single step on a metafounder base;
#    BayesC; a mating plan for next season
abp run examples/11_sheep_upg/analysis.toml --out runs/ex11
abp run examples/12_sheep_wwt_single_step_metafounder/analysis.toml --out runs/ex12

# 5. multi-trait REML; litter size with a threshold model
abp run examples/13_sheep_multitrait_reml/analysis.toml --out runs/ex13
abp run examples/14_sheep_nlb_threshold/analysis.toml --out runs/ex14
# the traits of example 13 with sampled covariance matrices (about 6 minutes)
abp run examples/16_sheep_multitrait_bayes/analysis.toml --out runs/ex16
# two repeated ewe traits with a permanent environment (about 6 minutes)
abp run examples/17_sheep_ewe_repeated_bayes/analysis.toml --out runs/ex17
# weaning weight with maternal genetic effects (about 3 minutes)
abp run examples/18_sheep_wwt_maternal_bayes/analysis.toml --out runs/ex18
abp run examples/10_sheep_fec_bayesc/analysis.toml --out runs/ex10
abp mate examples/09_sheep_mating/mating.toml --out runs/mating
```

Each run writes `report.md` (for breeders), EBV tables, QC reports,
`results.json`, `manifest.json` and `run.log` into the output folder. On
Windows use `packaging\windows\abp.ps1` (or `abp.cmd`), which keeps a UTF-8
log and returns ABP's exit status.

## Documentation

| Document | Contents |
|---|---|
| [User manual](docs/user_manual.md) | concepts, installation, data preparation, every spec key, outputs, QC rules, genomic and multi-trait analyses, troubleshooting |
| [Methods reference](docs/methods.md) | the equations and algorithms, the code implementing them, sources and errata |
| [Python API](docs/api.md) | using ABP as a library |
| [Validation report](docs/validation_report.md) | what was tested, how, results, and what was not run |
| [Benchmarks](docs/benchmarks.md) | measured run times and scaling |
| [Examples](examples/README.md) | fourteen runnable examples |
| [Error codes](docs/error_codes.md) | every error code, exit status and remedy |
| [Requirements trace](docs/requirements.md) | requirement → code → test → status |
| [Architecture decisions](docs/adr/) | language choice, native kernels, atomic outputs |
| [Handoff](docs/HANDOFF.md) | project state and the next concrete tasks |

## Scientific ground rules (enforced in code)

* Every run declares its estimand, genetic base, target population and
  information cut-off. Additive breeding values are never presented as total
  genetic values or phenotype predictions.
* Unknown parents are distinct base animals, never one common ancestor.
  `A22⁻¹` is the inverse of `A22` (explicitly, or exactly as the Schur
  complement of blocks of `A⁻¹` in the matrix-free path), never the
  genotyped block of `A⁻¹`.
* A singular `G` is never repaired silently. Blending, ridge and tuning are
  explicit, and their magnitudes are recorded.
* Non-convergence (REML or MCMC), reliabilities outside [0, 1], confounded
  fixed genetic groups, an unexpected `cuda` request, or a missing
  permanent-environment term for repeated records all stop the run with a
  documented error. They never produce a quiet warning.
* Decisions (contributions, mating plans) are proposals with verified hard
  constraints. Infeasibility is reported, never relaxed. ABP never triggers
  real-world actions.

## License

The repository owner has not yet chosen a project license. Third-party
dependencies and their licenses are listed in
[`docs/license_inventory.md`](docs/license_inventory.md).
