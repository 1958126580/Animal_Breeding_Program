# Changelog

All notable changes. Scientific-result changes are marked **[results]**.
Versioning: 0.x is pre-release; any change in the algorithm or the genetic
base that alters results is listed here, whatever the size of the version bump.

## [0.6.0] - 2026-09-30

Sixth development round. Evidence and gaps: `docs/validation_report.md`.

### Added
- Gibbs sampler for the threshold model with unknown liability variances (`variances.mode = "bayes"`, `bayes.method = "threshold"`): Cowles threshold step, exact block draws of the location effects by perturbation with a reused sparse LDL' pattern, parameter-expanded scale moves, uniform variance priors; R-hat/ESS gating, traces and diagnostics as for the marker models (`abp.solvers.threshold_gibbs`).
- Kackar–Harville PEV for multi-trait REML: `pev_incl_vc_uncertainty_<trait>` and `reliability_incl_vc_uncertainty_<trait>` in `ebv_multitrait.csv`; `MTREMLFit.cov`.
- Sampled PEV and reliabilities for the matrix-free single step (`solver.pev = "sampled"`, `solver.pev_samples`, `solver.pev_seed`; `reliability_mc_se` column); exact draws from N(0, H), N(0, G_APY) without forming them (`abp.solvers.pev_sampling`, samplers in `abp.core.ssop`).
- Analytic gradient for reduced-rank multi-trait REML (`ReducedRankEvaluator.value_and_gradient`); Newton-decrement convergence criterion (`ReducedRankFit.newton_decrement`).
- `apy_blocks_from_dosage` (int8 genotypes), `a_block`; `SparseLDL.refactor`, `SparseLDL.l_times`.
- Studies: threshold study scenario `threshold_gibbs`; `benchmarks/rr_calibration_study.py`; multi-trait study metrics with the corrected PEV; `benchmarks/ssmf_large.py` (200,000 animals). Self-test T16. API sections 23–26, methods §27–29.

### Changed
- **[results]** Reduced-rank REML optimises with the analytic gradient and stops on the Newton decrement (≤ 1e-6) instead of a gradient bound of 1e-3; estimates agree with round 5 to the optimiser tolerance, and fits that previously stopped with `ABP-E403` near the optimum now converge.
- **[results]** Full-rank multi-trait REML runs write two additional columns per trait (existing columns unchanged).
- `bayes.method` accepts `threshold`; categorical traits with `mode = "bayes"` require it, marker models refuse categorical traits.

### Fixed
- `SparseLDL` failed ("entry outside the symbolic pattern") on matrices that are symmetric only to rounding but have an asymmetric sparsity pattern (a product that cancels to ~1e-17 on one side and to 0 on the other). The matrix is now symmetrised before ordering. Found by the rank-2 reduced-rank fit; regression test and self-test T16.

## [0.5.0] - 2026-09-29

Fifth development round. Evidence and gaps: `docs/validation_report.md`.

### Added
- PEV and reliability including the uncertainty of REML variance estimates (Kackar–Harville, first order): `pev_incl_vc_uncertainty` and `reliability_incl_vc_uncertainty` columns for single-trait REML runs; `REMLFit.cov`, `REMLFit.cov_names`; `abp.solvers.vc_uncertainty`.
- Laplace-approximate REML for the liability variances of categorical (threshold) traits (`variances.mode = "reml"` with `type = "categorical"`); warm-started Newton evaluations; documented bias in sparse data.
- Reduced-rank genetic covariance matrix for multi-trait REML (`G0 = ΛΛ'`): `reml.boundary = "reduced_rank"` (refit after a boundary stop) and `reml.rank`; BLUP and PEV in the latent-factor space (`loadings` in `assemble_multitrait` / `build_and_solve`).
- Matrix-free single step (`genomic.single_step_mode = "matrix_free"`): `H⁻¹` applied as an operator inside PCG, `A22⁻¹` as the Schur complement of sparse blocks of `A⁻¹`, APY inverse as an operator built from genotypes (`abp.core.ssop`); solutions only.
- Self-test T15; benchmark `benchmarks/ssmf_benchmark.py`; calibration-study scenarios `single_step_apy150_true`, `single_step_apy300_true` and the Kackar–Harville metrics; threshold-study scenario `threshold_laplace`; API sections 19–22; methods §23–26.
- Evidence: F9 repeated with 200 replicates (resolved as sampling variation).

### Changed
- **[results]** Single-trait REML runs write two additional EBV-file columns (existing columns unchanged).
- The spec rule "categorical traits need known variances" now also accepts `mode = "reml"` (Laplace); `reml.start.residual` must then be 1.
- Report: reduced-rank fits are explained; likelihood evaluations are counted as such.

### Fixed
- None in released code this round (test-construction corrections are listed in the validation report).

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
