# Changelog

All notable changes. Scientific-result changes are marked **[results]**.
Versioning: 0.x is pre-release; any change in the algorithm or the genetic
base that alters results is listed here, whatever the size of the version bump.

## [0.2.0] - 2026-09-25

Second development round. Evidence and gaps: `docs/validation_report.md`.

### Added
- Unknown-parent groups (`[upg]`): group codes in the pedigree, QP transformation (Quaas 1988), random groups with a declared variance ratio (REML available) or fixed groups with an SVD estimability test that stops confounded models (`ABP-E300`); `upg_solutions_<trait>.csv`; example 11 with a deterministic generator, a comparison script and a 20-replicate study.
- Forward-in-time LR validation (`[validation]`): partial vs whole evaluations with the same variances (REML on partial data only), bias, dispersion and `ρ_wp`, sire-cluster bootstrap; example 08.
- Optimal contribution selection and mating plans (`abp mate`): exact QP with KKT certificate, ceiling-preserving integer plans, minimum-inbreeding allocation with relationship and recessive-risk limits, infeasibility reports; example 09. Proposals only.
- PLINK 1 binary input (`data.plink`, counted allele A1, declared assembly).
- Bayesian marker models (`variances.mode = "bayes"`): BRR, BayesA, BayesB, BayesC, BayesCπ, BayesR by Gibbs sampling with an optional C++ sweep; rank-normalized split R-hat, bulk/tail ESS and MCSE for every scalar and every GEBV; automatic extension and withholding of unconverged results (`ABP-E405`); per-draw traces; posterior predictive checks; example 10.
- Evidence: 50-replicate EBV calibration study, UPG study (three scenarios), prior simulation-based calibration of all six samplers, cross-check of the MCMC diagnostics against ArviZ.
- `abp selftest` checks T07–T11 (PLINK bytes, group A*⁻¹, OCS closed form, MCMC diagnostics reference values, native sweep).
- Python API sections for the new modules (`examples/api_example.py`).

### Changed
- **[results]** Effective sample sizes now follow Stan's reference truncation exactly (previously up to 0.9% different); the MCSE of the mean uses the SD of all draws. MCMC gating decisions can change marginally.
- **[results]** `blup()` accepts `NaN` in `diag(K)` for equations without a defined prior variance (fixed groups); their reliabilities are `NaN`/empty instead of an error.
- Per-GEBV diagnostics that are skipped for memory reasons are now reported as `not_computed` instead of being silently absent.
- Messages and documentation no longer hard-code version 0.1.
- The spec schema (`contracts/analysis_spec.schema.json`) gains `[upg]`, `[bayes]`, `[validation]`, `data.plink`, `data.genotype_assembly` and `data.phenotype_columns.date`.

## [0.1.0] - 2026-09-25

First vertical slice. See `docs/validation_report.md` for evidence and gaps.

### Added
- Data contracts: TOML analysis spec with a strict validator and generated JSON Schema; data dictionary; run-manifest schema; stable error codes and exit statuses.
- QC: pedigree (cycles, duplicates, self-parenthood, sex and role conflicts, birth order, missing parents), phenotype (ranges with error/quarantine, missing classifications, repeated records, outliers, singletons), genotype (map/allele/assembly contract, dosage range, call rates, monomorphic markers, MAF, Mendelian conflicts). Every exclusion is listed.
- Pedigree kernel: deterministic ordering, Meuwissen-Luo inbreeding (Python reference plus optional C++20 kernel), sparse A⁻¹, `log|A|`, Colleau `A x` products.
- Mixed-model equations: fixed-effect constraints with a documented rule; dense Cholesky, SuperLU and Jacobi-PCG solvers with residual verification; PEV, SEP, reliability with range checks.
- REML: average-information REML with EM fallback, active-set boundary handling with a Kuhn-Tucker check, standard errors, checkpoint and resume.
- Genomics: VanRaden G with recorded frequency source, explicit singular policy (blend/ridge), `match_a22` tuning, GBLUP, single-step H⁻¹ with `diag(H)` and `log|H|`.
- Multi-trait BLUP with known covariances, trait-specific fixed effects and per-animal PEV blocks.
- Decisions: Smith-Hazel and restricted indices; EBV aggregate index with reliability; `abp index` command.
- Workflow: atomic output publication, manifests, reports rendered from recorded results, cancellation, disk and memory checks, CPU-only backend with explicit CUDA refusal or fallback.
- Synthetic sheep flock generator (gene dropping) and seven examples; `abp selftest`.
- Documentation: user manual, methods reference, API, validation report, benchmarks, ADRs, requirements trace, method registry, license inventory, handoff.
- CI for Linux and Windows (Python 3.11–3.13) and a pure-Python-kernel job; Windows and Linux launchers.

### Changed during development (before release)
- **[results]** None relative to a previous release (first release).
- Automatic solver choice uses PCG for large systems without PEV (measured 0.26 s vs 15 s for sparse direct at 100k animals), with a fallback to sparse direct.
- PEV diagonal blocks are taken from `L⁻¹` instead of the full inverse (four-trait example: 15.3 s → 8.3 s; identical results).
