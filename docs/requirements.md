# Requirements trace

Source: the project control instruction (`01_总控指令.md`) and the full
specification (`02_完整研发指令与技术规范.md`), 2026-09-25. Status values:
**passed** (implemented, and the required checks ran and passed),
**partial** (part implemented or verified; the gap is stated), **not_run**
(not implemented or not executed). "Linux" means the checks ran on the Linux
build machine recorded in `docs/validation_report.md`.

## Scientific constraints (control instruction, §"不可违反的科学约束")

| ID | Requirement | Implementation | Evidence | Status |
|---|---|---|---|---|
| SCI-1 | Model, likelihood, parameterization, dimensions and outputs written before kernels; formulas linked to sources | `docs/methods.md`, module docstrings, `docs/method_registry.toml` | `tests/test_registry.py` | passed |
| SCI-2 | Only identifiable quantities are output; additive, non-additive and total genetic values named separately; base versioned | `analysis.task` (only `additive_ebv` accepted), `genetic_base` required, fixed-effect constraints reported as non-estimable | `tests/test_blup.py::test_invariance_to_constraint_choice_and_record_order` | passed (non-additive models not implemented) |
| SCI-3 | Training boundary: test phenotypes never enter training; transductive candidate-genotype use declared | frequency source and `candidate_genotype_use` recorded; excluded records cannot influence results; LR validation hides post-cutoff records from the partial evaluation, including its REML | `tests/test_genomic_workflow.py::test_excluded_records_cannot_influence_results`, `tests/test_validation_lr.py::test_hidden_phenotypes_cannot_leak_into_partial_evaluation`, `::test_partial_reml_ignores_hidden_records` | passed (cutoff-based design; no k-fold or across-population designs) |
| SCI-4 | Unknown parents are not one ancestor; A/G/H order consistent; A22⁻¹ is not (A⁻¹)₂₂; tuning and blending recorded | `abp.core.pedigree`, `abp.core.genomic` | T03, T06, `test_unknown_parents_are_distinct_base_animals`, counterexample in `test_t06_single_step_identities` | passed |
| SCI-5 | Factorizations or iterations, no explicit inverse on large production paths; Float64 reference | `abp.solvers.mme` (explicit inverse only for dense PEV/REML traces, under a memory budget) | `tests/test_blup.py` | passed |
| SCI-6 | Correlation not equated with accuracy; report bias, slope, PEV | PEV/SEP/reliability per animal; LR bias, dispersion and `ρ_wp` with bootstrap intervals (and the statement that `ρ_wp` is not accuracy); calibration studies report slope, bias, PEV ratio and coverage | `docs/validation/calibration_study.json`, `docs/validation/upg_study.json`, `tests/test_validation_lr.py` | passed (LR population-accuracy estimator deliberately not computed) |
| SCI-7 | Bayesian computation records priors, chains, seeds, MCSE, ESS and diagnostics | `abp.solvers.bayes`, `abp.solvers.mcmc_diagnostics`: priors and their derivation, chain seeds, R-hat, bulk/tail ESS, MCSE for every scalar and every GEBV; results withheld (`ABP-E405`) unless the criteria pass | `tests/test_bayes.py`, `tests/test_bayes_workflow.py::test_nonconvergence_withholds_results`, ArviZ cross-check | passed |
| SCI-8 | Decisions respect inbreeding, diversity, carriers, capacity; infeasibility reported; no automatic real-world actions | selection indices; OCS with a coancestry ceiling and capacities; mating plans with pair-relationship and recessive-risk limits; infeasible ceilings and plans reported (minimum coancestry, unmatchable parents), never relaxed; outputs are proposals, ABP triggers no real-world actions | `tests/test_selection_index.py`, `tests/test_ocs_mating.py` | passed (pedigree coancestry only; no genomic OCS) |

## Engineering constraints

| ID | Requirement | Implementation | Evidence | Status |
|---|---|---|---|---|
| ENG-1 | Modular monolith: core, numerics, workflows, CLI/API layers | `src/abp/{core,io,qc,solvers,decision,workflows}`, `cli.py` | `docs/adr/0001-primary-language.md` | passed |
| ENG-2 | One scientific core; a single primary language, with an ADR if not Julia | Python + optional C++20 kernel | ADR 0001, ADR 0002 | passed |
| ENG-3 | Native Windows and Linux; launcher keeps logs and exit codes; Chinese/space paths; explicit GPU fallback | `packaging/windows/abp.ps1`, `abp.cmd`, `packaging/linux/abp.sh`; `backend.on_unavailable` | Linux: `tests/test_workflow.py::test_unicode_and_space_paths`, `::test_cuda_request_is_never_faked`, manual launcher run; Windows: CI run 36130441504 (tests, self-test, launcher with Chinese/space path) | partial: CUDA not implemented; no Windows performance data |
| ENG-4 | Explicit schemas for all inputs | `contracts/`, `abp.core.spec`, `abp.io.tables`, `abp.qc.*` | `tests/test_qc.py`, `tests/test_workflow.py` | passed |
| ENG-5 | Provenance: hashes, model spec, dependency lock, environment, logs, output checks | `manifest.json`, `requirements-lock.txt`, `run.log` | `tests/test_manifest_contract.py`, `tests/test_workflow.py::test_example01_end_to_end_matches_independent_reference` | passed |
| ENG-6 | Resume, cancel, atomic writes, disk check, resource budget, error codes | ADR 0003; REML checkpoints; SIGINT/SIGTERM → status 130; `resources.*`; `abp.errors` | `tests/test_reml.py::test_non_convergence_is_an_error_and_checkpoint_resume`, `tests/test_workflow.py::test_output_exists_requires_force` | partial: thread-count budget not exposed (BLAS threads follow `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`) |
| ENG-7 | CI records compile, unit tests, platform runs and GPU runs separately | `.github/workflows/ci.yml` (Linux and Windows × Python 3.11–3.13, pure-Python-kernel job) | run 36130441504: all 7 jobs success | partial: no GPU job (no CUDA code) |
| ENG-8 | Licenses checked; no unauthorized data transfer | `docs/license_inventory.md`; ABP makes no network calls | code review | passed |

## Work method

| ID | Requirement | Status |
|---|---|---|
| WM-1 | README, requirements, method registry, ADRs, data contracts, gold standards, environment lock and CI before or alongside implementation | passed |
| WM-2 | Independent references first; production code never generates its own expected values | passed (`tests/reference/dense_reference.py` imports nothing from `abp`) |
| WM-3 | Each module delivers sources, code, minimal example, reference, failure cases, logs, performance, user notes and limits | passed for implemented modules (see method registry) |
| WM-4 | State written to the repository for loss-free continuation | passed (`docs/HANDOFF.md`, `CHANGELOG.md`) |

## Module status (spec §6)

| Module | Status | Notes |
|---|---|---|
| M01 data contracts and estimands | passed | TOML spec + JSON Schema, data dictionary, error codes |
| M02 QC | passed | pedigree, phenotype and genotype rules; PLINK 1 binary input; batch and sex-chromosome checks not implemented |
| M03 relationships and base | passed | unknown-parent groups (random or fixed, QP transformation); no metafounders |
| M04 LMM and BLUP | passed | dense, sparse and PCG |
| M05 REML and reliability | passed | single-trait; dense path |
| M06 GBLUP | passed | VanRaden G with policies; SNP-BLUP equivalence tested |
| M07 ssGBLUP | partial | exact H⁻¹; no APY |
| M08 Bayesian | partial | BRR, BayesA, BayesB, BayesC, BayesCπ, BayesR with MCMC diagnostics (M20 part); single-trait genotyped-only; no Bayesian LASSO/horseshoe, single-step or multi-trait Bayes |
| M09 multi-trait / repeatability / random regression | partial | multi-trait BLUP with known covariances; repeatability model; no random regression; no multi-trait REML |
| M10 threshold / survival / G×E | not_run | |
| M11 selection index | passed | Smith-Hazel, restricted, EBV index |
| M12 OCS and mating | passed | pedigree-based OCS with KKT certificate; integer plans; minimum-inbreeding mating with hard constraints; no genomic OCS or multi-generation planning |
| M13 validation, simulation, benchmarking | partial | LR forward validation; 50-replicate calibration study; 20-replicate UPG study; no comparison with other software |
| M14 Windows/Linux/CUDA delivery | partial | Linux and Windows functional tests passed (CI); no CUDA; no installers |
| M15–M19, M21–M23 | not_run | research and extension modules |
| M20 Bayesian computation credibility | passed (core) | ≥ 4 chains from independent streams; rank-normalized folded split R-hat, bulk/tail ESS, MCSE for every scalar and GEBV (equal to ArviZ to ≤ 8·10⁻¹⁶); traces; extension or failure (`ABP-E405`); prior SBC of all six samplers (300 replicates each, passed); posterior predictive checks. Not run: posterior SBC near target data |
| M24 usability | partial | CLI, reports, bilingual paths; no GUI; English reports only |
| M25 evidence management | passed | manifests, registry, handoff |
| M26 release and maintenance | partial | source release, lock, changelog; no binary wheels or installers; project license not chosen |
