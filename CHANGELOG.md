# Changelog

All notable changes. Scientific-result changes are marked **[results]**.
Versioning: 0.x is pre-release; any change in the algorithm or the genetic
base that alters results is listed here, whatever the size of the version bump.

## [0.4.0] - 2026-09-29

Fourth development round. Evidence and gaps: `docs/validation_report.md`.

### Added
- Multi-trait REML (`variances.mode = "reml"` with several traits): AI-REML with EM fallback for G0 and R0, missing traits handled exactly, dense or sparse selected-inversion traces; boundary optima stop with a diagnosis (`ABP-E300`); example 13.
- Threshold (probit) model for ordered categorical traits (`[[traits]] type = "categorical"`): posterior mode by Newton-Raphson, liability-scale EBVs with Laplace PEV, thresholds file; example 14.
- Sparse LDL' factorization with minimum-degree ordering (C++ kernels `mindegree_order`, `ldl_numeric`, `ldl_solve`); `solver.factorization = "auto" | "ldl" | "superlu"`.
- APY inverse of G for GBLUP and single step (`genomic.apy_core_size`, `genomic.apy_seed`).
- Metafounders for multi-trait models (contrasts per trait, index on contrasts) and for LR validation.
- Studies: F6 factor study (`calibration_study.py --design`), multi-trait calibration, two-metafounder composite, linear vs threshold model; simulator options for SNP/QTL frequency ranges and random ram selection (defaults byte-identical).
- Self-test T14 (sparse LDL'); benchmark groups `ldl` and `apy`; API sections 15-18; methods §19-22.

### Changed
- **[results]** The sparse direct path uses ABP's LDL' by default when the compiled kernel is present (same results as SuperLU to rounding; 5.5x faster at 100,500 equations).
- **[results]** Under metafounders, single-trait and multi-trait economic indices use the EBVs relative to the reference metafounder.
- Metafounders are no longer restricted to single-trait models or runs without LR validation.

### Fixed
- Stored zeros of `A⁻¹` (exact cancellation, e.g. a son mated to his own dam) were requested from the selected inverse and refused, which stopped sparse REML (single- and multi-trait) on such pedigrees.

## [0.3.0] - 2026-09-28

Third development round. Evidence and gaps: `docs/validation_report.md`.

### Added
- Metafounders (`[metafounders]`; Legarra et al. 2015): related, possibly inbred base populations for pedigree BLUP, REML and single step. Extended relationship matrix `A^Γ = T D T' + QΓQ'` with a sparse inverse, log-determinant and Colleau products; generalised Meuwissen–Luo kernel in Python and C++ (`ml_general`); Γ from a file with required provenance, or estimated from genotypes by GLS base allele frequencies with a sampling correction; single step on the metafounder base (`G05`, no tuning). EBV files gain `ebv_vs_base`, `pev_vs_base`, `reliability_vs_base` against a reference metafounder; `metafounder_solutions_<trait>.csv`; new error `ABP-E205 PEDIGREE_UNASSIGNED_BASE`; example 12.
- Sparse selected inversion (Takahashi equations) with a symbolic Cholesky pattern that closes over numeric cancellation; Python reference and C++ kernels (`symbolic_cholesky`, `takahashi`). Exact PEV is no longer limited to 30,000 equations (limit: factor memory, `ABP-E500`); REML above 12,000 equations uses the sparse factor and selected inversion (`trace_method` in the REML output); multi-trait PEV blocks use it too.
- Calibration study scenarios `single_step_reml`, `single_step_mf_reml`, `single_step_mf_true`; parallel replicates (`--workers`).
- Benchmark groups `selinv` and `metafounders`.
- Python API sections 13 (metafounders) and 14 (selected inversion); methods reference §17 and §18.

### Changed
- **[results]** `solver.method = "auto"` with `pev = "exact"` above the dense limit now selects sparse direct with selected inversion instead of refusing above 30,000 equations. For systems that were already solved on the sparse path, PEV values are the same up to rounding.
- **[results]** REML above 12,000 equations uses sparse selected inversion instead of the dense inverse (same likelihood, score and AI to rounding); systems beyond the dense memory budget are no longer refused.
- `ABPError` is picklable (errors raised in worker processes reach the parent).
- The `PED-UPG` QC message covers both genetic groups and metafounders.

### Fixed
- The prior variance of the base contrast in single step used `Cov(u_i, u_ref)` from `A^Γ` instead of `H` (found during development by the reliability range check; never released).
- The calibration study set its one-thread BLAS limit after importing NumPy, which oversubscribed the CPU when run with several workers.

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
