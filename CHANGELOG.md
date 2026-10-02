# Changelog

All notable changes. Scientific-result changes are marked **[results]**.
Versioning: 0.x is pre-release; any change in the algorithm or the genetic
base that alters results is listed here, whatever the size of the version bump.

## [0.12.0] - 2026-10-02

Twelfth development round. Evidence and gaps: `docs/validation_report.md`.

### Added
- Sparse LDL': dense trailing block. When the last columns of the factor are nearly dense (fill-reducing orders leave a separator there), the up-looking kernel stops at the block and returns its Schur complement, which is factorized by LAPACK Cholesky and stored on the symbolic pattern (`dense_tail_split`, C++ kernel `ldl_numeric_split` with a Python reference). Block size by a work-saving rule, capped by the memory budget; `SparseLDL(..., dense_tail=False)` keeps the previous factor. Maternal model with 1,500 animals: 61 → 23 ms per factorization, maternal-study replicate 1,235 → 408 s.
- Posterior medians in the G0, R0 and P0 summaries of the multi-trait Gibbs samplers.
- `maternal_study.py --prior equal` (weak inverse-Wishart priors centred on equal shares of the phenotypic variance) and a variant of example 18 with these priors (`analysis_equal_prior.toml`). Finding F19 reduced: maternal variance 2.37 instead of 2.54 (true 2.0), RMSE 1.01 → 0.59, covariance unbiased, 16 of 20 fits converged instead of 1 (validation report §7.16).
- Methods §21 (dense trailing block), manual §7.16 (priors for maternal models), validation report §7.16.

### Changed
- **[results]** Every model factorized by the sparse LDL' whose factor ends in a dense enough block now gets `d` and `L` from LAPACK for that block: equal to the previous factor to rounding (relative 1e-14 in the tests); Gibbs chains may follow different rounding paths, within Monte-Carlo error (the flat-prior maternal study reproduced its round-11 summaries to three decimals).

## [0.11.0] - 2026-10-02

Eleventh development round. Evidence and gaps: `docs/validation_report.md`.

### Added
- Maternal genetic effects (maternal animal model) in the multi-trait Gibbs samplers: random term `kind = "maternal"` (dam from the pedigree), `2t × 2t` genetic covariance matrix with direct-maternal covariances, maternal EBVs (`mebv_`, `mreliability_`, `msep_` columns), maternal heritability `m2`, direct-maternal correlation; also for a single trait (`bayes.method = "multitrait"` with one trait) and with a maternal permanent environment (iid term on a dam column) (`mt_threshold_gibbs(..., dam_col=)`, `MTProblem.incidence`). Example 18 (weaning weight, synthetic data from `make_data.py`).
- `benchmarks/maternal_study.py` (20 replicates): maternal and direct EBVs calibrated (MSE/PEV 0.98 / 1.03, coverage 0.95), maternal variance biased upwards under flat priors (2.54 vs 2.0; finding F19).
- Methods §32 (maternal effects), API section 32, manual §5.6/§7.16, validation report §7.15.

### Changed
- Scale and shear moves run over all genetic columns (direct and maternal) through one incidence function; results without maternal effects are unchanged in distribution.

## [0.10.0] - 2026-10-01

Tenth development round. Evidence and gaps: `docs/validation_report.md`.

### Added
- Permanent-environment (iid) term in the multi-trait Gibbs samplers (`bayes.method = "multitrait"` or multi-trait `"threshold"`): repeated records per animal, `pe ~ N(0, I ⊗ P0)` with P0 sampled (flat or inverse-Wishart prior); outputs P0, `c2_<trait>`, `pe_multitrait.csv`; heritabilities on G + P + R (`mt_threshold_gibbs(..., pe_col=)`). Example 17 (two repeated ewe traits, synthetic data from `make_data.py`).
- Proper inverse-Wishart priors for the permanent-environment and residual matrices: `bayes.prior_covariance` accepts the iid term's name and `residual` (with a categorical trait the R0 prior needs that trait in its own residual group).
- `bayes_vs_reml_study.py --prior pheno` (weak data-centred priors) and `--seeds`. F16 follow-up (30 replicates, three traits): weak priors brought the low-heritability traits to MSE/PEV 1.03 / 1.11 (flat 1.10 / 1.20, Kackar–Harville 1.22 / 1.44 on the same seeds) at +0.025 for the best-determined trait.
- Methods §32 (permanent environment, R0 prior), API section 31, manual §7.16 (repeated records), validation report §7.14.

### Changed
- **[results]** Scale moves (parameter expansion) for every trait on (u_j, G0) and (pe_j, P0) in both multi-trait Gibbs samplers (round 9: the categorical trait only). Same posterior; different random streams, so multi-trait Gibbs results change within Monte-Carlo error; mixing of the variances improves (example 17: not converged → converged).
- Multi-trait models with known or REML variances still refuse extra random terms; the error message names the Gibbs alternative.

### Fixed
- The sparsity pattern of the multi-trait Gibbs coefficient matrix is built from absolute values: a numerical surrogate cancelled exactly with repeated records and dropped needed entries (found by the new exact test before release; no released analysis had repeated records in a multi-trait Gibbs model).

## [0.9.0] - 2026-10-01

Ninth development round. Evidence and gaps: `docs/validation_report.md`.

### Added
- Bayesian multi-trait linear model: `variances.mode = "bayes"`, `bayes.method = "multitrait"` (two or more continuous traits, one additive term, own fixed effects per trait, missing traits); `G0` and `R0` sampled with the breeding values (`R0 | E ~ IW(E'E, n - t - 1)`, flat; `G0` flat or inverse Wishart); EBVs are posterior means, PEVs posterior variances. Same outputs and convergence gating as the multi-trait threshold model (`mt_threshold_gibbs(Y, None, ...)`). Example 16 (the traits of example 13).
- `bayes.residual_groups`: residual covariances fixed at exactly 0 between groups of traits, in both multi-trait Gibbs samplers (`MTThresholdGibbsConfig.residual_groups`, `draw_R0(..., groups=)`); each block drawn from its own exact conditional. Example 15 now puts weaning weight and first-parity litter size in different groups.
- Inbreeding by pedigree depth (`_native.inbreeding_depth`, used by `Pedigree.inbreeding`; kernel name `native_cpp_depth_ml_colleau`): per depth the cheaper of Meuwissen–Luo traces and Colleau columns of A, the same F. 200,000 animals in 20 discrete generations: 1.8 s instead of 16.4 s (50 sires per generation) and 17.4 s instead of 116.4 s (1,000 sires); no slower where the traces are cheap.
- `benchmarks/bayes_vs_reml_study.py` (finding F16: posterior PEV vs REML + Kackar–Harville; 2 × 50 replicates, 1,000 animals: MSE/PEV of the low-heritability traits 1.000 ± 0.040 with two traits against 1.142 with Kackar–Harville, 1.062 / 1.142 with three traits against 1.157 / 1.310 — F16 partly resolved), `benchmarks/inbreeding_benchmark.py`.
- Methods §32–33, API section 30, manual §7.16 and a note on BLAS threads on shared machines (§8), validation report §7.13.

### Changed
- **[results]** Example 15 fixes the residual covariance between weaning weight and nlb1 at 0 (previously estimated, correlation 0.28).
- The report and `results.json` name the Bayesian multi-trait linear model (`bayes.method = "multitrait"`); `mcmc_diagnostics_multitrait.json` gains `model` and `residual_groups`, and `categorical_trait`/`thresholds` are `null` for the linear model.
- Pedigree QC logs and manifests name the inbreeding kernel `native_cpp_depth_ml_colleau` (results unchanged: identical F in all tests and benchmarks).

### Fixed
- A round-9 test expected the native kernel name under `ABP_DISABLE_NATIVE=1` (CI pure-Python job failed once; test defect, production code unaffected).

## [0.8.0] - 2026-09-30

Eighth development round. Evidence and gaps: `docs/validation_report.md`.

### Added
- Multi-trait threshold model: one ordered categorical trait together with continuous traits (`variances.mode = "bayes"`, `bayes.method = "threshold"`, more than one model trait), own fixed effects per trait, missing traits; Gibbs sampler with conditional categorical liabilities, exact block draws, scale and shear parameter-expansion moves, `G0` flat or inverse Wishart (`bayes.variance_prior = "inverse_wishart"`, `bayes.nu`, `bayes.prior_covariance`), `R0` with the categorical residual variance fixed at 1 (Korsgaard et al. 2003); outputs `ebv_multitrait.csv`, `mcmc_diagnostics_multitrait.json`, `mcmc_trace_multitrait.csv`; unconverged chains withheld (`abp.solvers.mt_threshold_gibbs`, `abp.workflows.mt_threshold`). Example 15.
- `benchmarks/rank_selection_study.py` (BLUP of the model chosen by the AIC rule, 2 and 3 traits), `benchmarks/mt_threshold_study.py`, `benchmarks/pev_estimator_study.py`.
- Native Colleau product `A x` (`_native.colleau_times`), used by `Pedigree.a_times`; `SparseLDL.refactor_values`; self-test T18.
- API section 29, methods §31, validation report §7.12.

### Changed
- **[results]** Sampled reliabilities (`solver.pev = "sampled"`) use the estimator `mean h² / (mean h² + mean d²)` with its second-order bias removed instead of `1 − mean d² / mean u*²`: about three times fewer simulations for the same precision at a reliability of 0.32; `sampled_pev(..., estimator="ratio")` keeps the old one. The manifest records the estimator.
- `SparseLDL.refactor` accepts a matrix whose entries lie on the stored pattern (exact zeros this time).
- Genotype QC makes no full copies of the dosage and missing matrices when nothing is excluded; APY centring works in place.
- Categorical traits in multi-trait models are accepted with the threshold Gibbs sampler (previously refused).

## [0.7.0] - 2026-09-30

Seventh development round. Evidence and gaps: `docs/validation_report.md`.

### Added
- Proper variance priors for the threshold Gibbs sampler in the spec: `bayes.variance_prior = "scaled_inv_chi2"`, `bayes.nu`, per-term `bayes.prior_variances` (`ThresholdGibbsConfig.s2` accepts a dict). Uniform priors stay the default. Prior-sensitivity study `docs/validation/threshold_prior_study.json`.
- Kackar–Harville PEV for reduced-rank multi-trait REML fits (`kackar_harville_delta_reduced_rank`; `ReducedRankFit.x`, `.cov_x = 2 H⁻¹`); the `*_incl_vc_uncertainty_<trait>` columns are now written for reduced-rank runs too. In the 100-replicate rank-1 study MSE/PEV went from 1.021/1.076 to 1.007/1.038.
- Rank selection for multi-trait REML: `reml.rank_selection = "aic"` fits every rank and chooses one by AIC with a conservative margin (a lower rank only if its AIC is smaller by at least 2); table in the report and `results.json` (`select_rank`, `n_parameters`, `AIC_MARGIN`).
- Compact genotype storage: `genomic.genotype_storage = "int8"` (PLINK decoded directly to int8, −1 = missing; `qc.genotype.to_int8`; `load_plink(..., storage=)`); the matrix-free APY path builds its blocks from the int8 dosages.
- `benchmarks/ssmf_workflow_large.py`: `abp run` on generated files at scale (int8, APY, matrix-free, sampled PEV) with peak memory of the child process. `rr_calibration_study.py --g0 full` (false rank reductions) and choice counts per rule.
- API sections 27–28; methods §25 (rank selection), §27 (priors), §28 (reduced-rank Kackar–Harville), §30 (compact storage).

### Changed
- **[results]** Allele frequencies and VanRaden G are computed in blocks of 4,096 markers for both storages (no full float copy of the genotypes). Results change only in the order of floating-point summation (EBVs equal to 1e-8 relative in the tests).
- **[results]** Reduced-rank multi-trait REML runs write two additional columns per trait (existing columns unchanged).

### Fixed
- The manifest lost `diagnostics.<trait>.pev_sampling` (number of simulations, seed, mean Monte-Carlo SE) because a later step replaced the trait's diagnostics; now updated in place. Results were unaffected. Found by the workflow-level benchmark.

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
- `SparseLDL` failed ("entry outside the symbolic pattern") on matrices that are symmetric only to rounding but have an asymmetric sparsity pattern (a product that cancels to ~1e-17 on one side and to 0 on the other). The matrix is now symmetrised before ordering, in ABP's LDL' and in the SuperLU path used without the native kernel (whose selected inversion failed the same way). Found by the rank-2 reduced-rank fit (native) and by the local pure-Python-kernel test run (SuperLU; the CI runner's rounding did not produce the one-sided entry); regression tests for both paths and self-test T16.

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
