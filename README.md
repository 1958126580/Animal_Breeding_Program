# ABP: auditable genetic evaluation for animal breeding

ABP is a command-line tool and Python library for estimating breeding values
and planning selection. It covers pedigree and genomic BLUP, single-step
GBLUP, REML variance components, multi-trait BLUP, unknown-parent groups,
Bayesian marker models (BayesA/B/C/Cπ/R, Bayesian ridge regression),
forward-in-time validation, selection indices, optimal contribution selection
and mating plans. Every result can be traced back to its inputs, model and
code. It is built to the project's research and
development specification (`docs/` and the uploaded instruction set).

> **Status: 0.2.0, research-grade.** Every method listed below has passed
> analytical and independent-reference tests on Linux; simulation studies
> cover EBV calibration, genetic groups and the Bayesian samplers. It has
> **not** been validated on real breeding data, compared with
> BLUPF90/MiXBLUP/ASReml/JWAS/BGLR, or run on CUDA. Single-step GBLUP was
> biased in the calibration scenario (finding F6 in the validation report).
> CI runs the test suite on Linux and Windows (Python 3.11–3.13).
> ABP makes no claim of superiority over any other software. See
> [`docs/validation_report.md`](docs/validation_report.md) for exactly what has
> and has not been verified.

## What it does

| Area | Capability (0.2) | Evidence |
|---|---|---|
| Data contracts and QC | CSV import with strict schemas; pedigree QC (cycles, duplicates, sex and birth-order conflicts, missing parents); phenotype QC (ranges, repeated records, outliers); genotype QC (allele and assembly contract, call rates, MAF, Mendelian conflicts). Excluded records are always listed. | `tests/test_qc.py`, `tests/test_workflow.py` |
| Pedigree relationships | Ordering, inbreeding (Meuwissen-Luo; optional C++20 kernel), sparse A-inverse, `A x` products without forming A (Colleau); unknown-parent groups (random or estimable fixed) by the QP transformation | spec gold standard T03, independent tabular method; explicit-model references for groups; 20-replicate group study |
| BLUP | Single-trait animal and repeatability models; rank-deficient fixed effects; dense, sparse-direct and PCG solvers; PEV, SEP, reliability | T04, Mrode (2005) Ex. 3.1, independent V-form reference, 50-replicate calibration study |
| REML | Average-information REML with EM fallback, boundary (zero-variance) handling, standard errors, checkpoint and resume | T05, independent optimizer, 40-replicate simulation |
| Genomics | Dosage CSV or PLINK 1 binary input; VanRaden G with recorded frequency source; explicit singular-G policy; GBLUP; GBLUP/SNP-BLUP equivalence; single-step H-inverse | T01, T06, explicit-H reference, hand-decoded PLINK bytes |
| Bayesian marker models | BRR, BayesA, BayesB, BayesC, BayesCπ, BayesR (Gibbs, 4+ chains, optional C++ sweep); R-hat, bulk/tail ESS and MCSE for every scalar and GEBV; results withheld unless converged; traces and posterior predictive checks | exact Gaussian posterior (BRR), exact inclusion probability, simulation-based calibration of all six samplers, diagnostics equal to ArviZ to 1e-15 |
| Validation | Forward-in-time LR method: bias, dispersion, correlation of partial vs whole EBVs with bootstrap intervals; hidden records cannot leak | hand-computed statistics, leakage test |
| Multi-trait | Multi-trait BLUP with known covariances, trait-specific fixed effects, missing traits, per-animal PEV blocks | 2×2 Kronecker-order gold standard, V-form reference |
| Decisions | Smith-Hazel and restricted indices; economic index on multi-trait EBVs with reliability; optimal contribution selection with a coancestry ceiling; mating plans minimising progeny inbreeding under relationship and recessive-risk limits (`abp mate`, proposals only) | T02, KKT certificates, independent optimizer, enumeration of small mating problems |
| Engineering | Atomic outputs, run manifests with hashes, report generated from recorded results, stable error codes and exit statuses, UTF-8 and Chinese/space paths, cancellation, disk check, memory budget | `tests/test_workflow.py`, `tests/test_manifest_contract.py` |

Not implemented yet (tracked as `not_run` in
[`docs/method_registry.toml`](docs/method_registry.toml)): metafounders, APY,
maternal and random-regression models, threshold and survival models,
multi-trait REML, multi-trait and single-step Bayesian models, genomic OCS,
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

# 4. genetic groups for purchased rams; BayesC; a mating plan for next season
abp run examples/11_sheep_upg/analysis.toml --out runs/ex11
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
| [Examples](examples/README.md) | eleven runnable examples |
| [Error codes](docs/error_codes.md) | every error code, exit status and remedy |
| [Requirements trace](docs/requirements.md) | requirement → code → test → status |
| [Architecture decisions](docs/adr/) | language choice, native kernels, atomic outputs |
| [Handoff](docs/HANDOFF.md) | project state and the next concrete tasks |

## Scientific ground rules (enforced in code)

* Every run declares its estimand, genetic base, target population and
  information cut-off. Additive breeding values are never presented as total
  genetic values or phenotype predictions.
* Unknown parents are distinct base animals, never one common ancestor.
  `A22⁻¹` is computed from `A22`, not taken from `A⁻¹`.
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
