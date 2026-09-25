# Validation report: ABP 0.2.0

Date: 2026-09-25 · Platforms executed: **Linux x86_64** (build machine, full
evidence below). Round 1 (0.1.0) was also run on **Windows Server 2025 and
Ubuntu** in GitHub Actions run
[36130441504](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/36130441504);
the CI result for this round is recorded in §3a.
Raw logs: `docs/validation/`. Status vocabulary: passed, failed, blocked,
not_run.

## 1. Summary by acceptance gate

| Gate | Meaning | Status | Evidence |
|---|---|---|---|
| G0 source and target | versions, licenses, estimands, contracts, base | passed | `docs/method_registry.toml`, `contracts/`, `docs/license_inventory.md` (project license: open owner decision) |
| G1 mathematics | independent derivations, analytical cases, dimension, limit and equivalence tests | passed | §4, §5 |
| G2 numerics | residuals, convergence, boundaries, exact references | passed | §4, §5; MCMC diagnostics equal ArviZ to ≤ 8·10⁻¹⁶ |
| G3 software | ID mapping, bad inputs, recovery, interface consistency | passed (Linux; Windows CI) | §3, §5, §6 |
| G4 statistical calibration | simulation bias and coverage | **partial** | pedigree BLUP calibrated over 50 replicates; **single-step failed** (F6); REML 40 replicates; genetic groups 20 replicates × 3 scenarios; SBC of all six Bayesian samplers (§7) |
| G5 external validity | real data, time or population hold-out | **not_run** | no real data were available or authorized (the LR workflow exists and was run on synthetic data) |
| G6 scale and platform | measured resources; Windows, Linux, GPU | **partial** | Linux measured (`docs/benchmarks.md`); Windows functional tests in CI, no Windows timings; no CUDA path |
| G7 decision and release | feasible plans, installation reproduction, evidence package | **partial** | mating plans satisfy every hard constraint, verified per plan (§5, §6); no binary release; project license not chosen |

**No claim of leadership or superiority over any software is made.** No
comparison with BLUPF90, MiXBLUP, ASReml, DMU, JWAS, BGLR, AlphaSimR or
AlphaMate has been run.

## 2. Environment of the executed checks

| Item | Value |
|---|---|
| OS | Linux 6.18.44 x86_64, glibc 2.39 (cloud VM) |
| CPU / RAM | Intel Xeon @ 2.10 GHz, 4 vCPU / 15 GiB |
| Python / NumPy / SciPy | 3.11.15 / 2.4.6 / 1.17.1 |
| BLAS / LAPACK | scipy-openblas 0.3.31 |
| Compiler (native kernel) | GCC 13.3, `-O3 -std=c++20` |
| Dependency lock | `requirements-lock.txt` (hash-pinned) |
| Test oracle (not a dependency) | ArviZ 0.23.4, installed separately |

## 3. Commands executed and results

| Command | Result | Log |
|---|---|---|
| `python -m pytest -v` (native C++ kernel) | **173 passed**, 0 failed, 0 skipped, 96.7 s | `docs/validation/pytest-linux-py311-native.log`, `junit-linux-py311-native.xml` |
| `ABP_DISABLE_NATIVE=1 python -m pytest -v` (pure-Python kernels) | **173 passed**, 0 failed, 0 skipped, 281.2 s | `docs/validation/pytest-linux-py311-python-kernels.log`, `junit-linux-py311-python-kernels.xml` |
| `abp selftest` with and without the native kernel | RESULT: PASS (both; T11 skipped without the kernel) | `docs/validation/selftest-linux.log` |
| `abp run` examples 01–06, 08, 10, 11 (+ `analysis_fixed.toml`), `abp index` 07, `abp mate` 09, `compare.py`, `api_example.py` | all exit status 0 | `docs/validation/examples-linux.log` |
| `python benchmarks/calibration_study.py --replicates 50` | completed | `docs/validation/calibration_study.json`, `.log` |
| `python benchmarks/upg_study.py` (linear trend; `--trend step`; `--trend step --ratio 25`), 20 replicates each | completed | `docs/validation/upg_study*.json`, `.log` |
| `python benchmarks/sbc_bayes.py --replicates 300` | all 32 rank-uniformity tests passed (smallest p = 0.029) | `docs/validation/sbc_bayes.json`, `.log` |
| `ARVIZ_PATH=… python benchmarks/mcmc_diagnostics_crosscheck.py` | max relative difference 2.2e-16 (R-hat), 7.8e-16 (bulk ESS), 2.5e-16 (tail ESS), 2.3e-16 (MCSE) | `docs/validation/mcmc_diagnostics_crosscheck.json` |
| `python benchmarks/run_benchmarks.py --only bayes ocs upg plink` | completed | `benchmarks/results/2026-09-25-linux-x86_64-round2.json` |
| `python benchmarks/run_benchmarks.py --full` (round 1), `benchmarks/simulation_check.py` | completed (round 1) | `benchmarks/results/2026-09-25-linux-x86_64.json`, `docs/validation/simulation_check.*` |

### 3a. Continuous integration

Round 1: run 36130441504 (commit 2cd2922): all 7 jobs succeeded —
Windows Server 2025 and Ubuntu × Python 3.11/3.12/3.13 (C++20 kernel built
with MSVC on Windows; `abp selftest` PASS; launcher runs into
`%RUNNER_TEMP%\结果 输出`) and a pure-Python-kernel job. Round 2: the CI run triggered by the push of this round is recorded in the follow-up commit on this branch.

## 4. Analytical gold standards (spec §8.2 and round-2 additions)

Expected values were derived by hand, come from the spec or a cited source,
or (marked) from an independent implementation. None was produced by ABP.
Tolerance: `|x − ref| ≤ 1e−10 + 1e−8 |ref|` unless stated.

| ID | Case | Expected | Result | Test |
|---|---|---|---|---|
| T01 | MAF for p = 0.1, 0, 0.5, 0.9, 1 | 0.1, 0, 0.5, 0.1, 0 | passed | `test_genomic.py::test_t01_maf_is_minimum` |
| T02 | Smith-Hazel, P = [[4,1],[1,9]], q = [2,3], Var(H) = 5 | b = [3/7, 2/7], r² = 12/35, r = 0.585540043769 | passed | `test_selection_index.py::test_t02_gold_standard` |
| T03 | full-sib mating pedigree | F₅ = 0.25; A row 5 = [.5,.5,.75,.75,1.25]; exact A⁻¹ | passed | `test_pedigree.py::test_t03_hand_derived_gold_standard` |
| T04 | two animals, unknown intercept, variances 1 | μ = 3, EBV ∓0.5, PEV 0.75, rel 0.25 | passed | `test_blup.py::test_t04_hand_derived` |
| T05 | REML genetic score at (0.7, 1.3) | −0.01463020355; wrong derivative (I for A) +1.03497570401 detected | passed | `test_reml.py::test_t05_score_matches_v_form_and_detects_wrong_derivative` |
| T06 | single step, G* = A22 | H⁻¹ = A⁻¹; and H⁻¹ = inv(H) for G* = 0.8 A22 + 0.2 I | passed | `test_genomic.py::test_t06_single_step_identities` |
| Mrode (2005) Ex. 3.1 | printed solutions (3 decimals) | sex 4.358/3.404; animals 0.098 … 0.183 | passed (±6e-4) | `test_blup.py::test_mrode_example_3_1` |
| 2×2 Kronecker | 2 animals × 2 traits, trait-major reference | EBV and PEV blocks | passed (1e-12) | `test_multitrait.py::test_two_animals_two_traits_kronecker_order` |
| T07 | PLINK bytes `78 00 2F 01`, 5 samples × 2 variants | A1 dosages [2,1,0,NA,2], [0,0,1,2,NA] | passed | `test_plink.py::test_hand_derived_bytes`, self-test |
| T08 | animal with both parents in group G, offspring with one unknown parent | A*⁻¹ = [[4/3,−2/3,−1],[−2/3,4/3,0],[−1,0,1]], Q = (1, ½) | passed | self-test; `test_upg.py::test_qp_identity` (general case vs block formula) |
| T09 | OCS, four unrelated founders, ceiling C = 0.14 | top male `a = (1 + √(16C − 2))/4`, females ¼ each | passed (1e-9) | self-test; `test_ocs_mating.py::test_closed_form_for_unrelated_candidates` |
| T10 | R-hat, bulk/tail ESS, MCSE of RNG-free chains | values from ArviZ 0.23.4 (independent implementation) | passed (1e-10) | `test_mcmc_diagnostics.py::test_agreement_with_arviz_reference_values`, self-test |
| T11 | BayesC inclusion probability for one marker | ratio of exact Gaussian marginal likelihoods | passed (decision flips exactly at the exact probability) | `test_bayes.py::test_inclusion_probability_matches_exact_marginal_likelihood` |

## 5. Independent-reference, property and failure tests

`tests/reference/dense_reference.py` uses different formulations (tabular A,
marginal V-form GLS/BLUP, V-form REML likelihood and score) and imports
nothing from `abp`.

| Area | Checks (all passed) |
|---|---|
| Pedigree | A, F, A⁻¹A = I and log\|A\| vs the tabular method on random inbred pedigrees; C++ vs Python kernel (1e-13); input-order invariance; unknown parents as distinct base animals; cycle, self-parent and same-sire-and-dam rejection |
| Genetic groups | `A*⁻¹` = explicit block formula; `Q` = independent recursion; random groups: solutions and PEV = V-form BLUP with the explicit covariance (1e-9), `log det K` = explicit log-determinant; fixed groups: solutions and PEV = explicit MME in `(b, g, u)` transformed to `(u + Qg, g)` (1e-10); confounding with the intercept and groups without descendants refused (unit and workflow); group codes parsed as groups, animal IDs with the prefix rejected; spec rules |
| BLUP | V-form with a generalized inverse for rank-deficient X plus a permanent-environment term; dense, sparse and PCG agreement; invariance to the constraint choice and record order; kg→g scaling; invalid variances rejected; PCG non-convergence is an error; automatic PCG → direct fallback recorded |
| REML | log L, score and AI matrix vs V-form (three components); optimum vs independent Nelder-Mead (rtol 2e-5), including the dense-G path; EM = AI; engineered boundary gives exactly σ²e = var(y); non-convergence is an error; checkpoint resume = fresh fit; mismatched checkpoint refused |
| Genomic | hand G; allele-flip invariance; GBLUP = SNP-BLUP (including unphenotyped animals and with a ridge); singular G needs an explicit policy; A22 tuning; workflow GBLUP and single-step vs explicit-H V-form (1e-8); training frequency sample uses QC-passed records only; PLINK and dosage inputs give byte-identical EBV files |
| Bayesian models | BRR with fixed variances: posterior means within 5 MCSE and SDs within 10% of the exact Gaussian posterior; sparse architecture recovered by BayesCπ; all six methods run and report diagnostics; seed determinism and independent chain streams; C++ sweep = Python reference (1e-12); explicit priors used as given; traces and draws consistent with the reported means; non-converged runs withheld with `ABP-E405`; skipped GEBV diagnostics reported |
| MCMC diagnostics | iid draws; AR(1) ESS vs `n(1 − φ)/(1 + φ)`; stuck chains flagged; agreement with ArviZ on eleven chain types (≤ 8·10⁻¹⁶) and on three RNG-free reference cases (1e-10) |
| LR validation | statistics vs hand computation; cutoff rules (straddling dates refused); seeded cluster bootstrap; adding 25 kg to every hidden record leaves partial EBVs bit-identical; partial REML ignores hidden records; spec rules |
| OCS and mating | OCS vs SciPy trust-constr and an independent KKT check; closed form; LP limit; infeasible ceiling reports the minimum; monotonicity in the ceiling; integer repair meets the ceiling and is bounded by the continuous optimum; mating plans optimal by enumeration of all assignments of small problems; carrier risk forbids only risky pairs; infeasible plans name the unmatchable dam; example 09 plan satisfies every hard constraint pair by pair; group-coded pedigrees |
| Multi-trait | missing patterns, trait-specific fixed effects and a structural residual zero vs V-form (dense and sparse); decomposition into single-trait models; one-trait degeneration; unit change; non-PD covariance rejected |
| Index | unit invariance; restricted index vs independent KKT solution; collinear information reported; selection intensity; EBV-index reliability |
| QC | each seeded pedigree error detected with the right code; all problems reported together; clean data produce no false alarm; genotype contract errors; call-rate, monomorphic and ambiguous-SNP handling; Mendelian sample-swap detection; PLINK magic number, layout, size and allele errors |
| Workflow | end-to-end example 01 vs independent reference; byte-identical reruns; `--force` semantics; failure leaves only `<out>.failed-*` with a failed manifest; quarantine lists; unknown spec keys rejected; repeated records without PE refused; Chinese and space paths; bad UTF-8 and ragged rows located by line; CUDA request refused or recorded as a CPU fallback; manifest contract for passed and failed runs; `abp validate` and `abp pedigree` with group codes and PLINK input |
| Leakage | changing phenotypes excluded by an approved rule leaves every output byte unchanged; LR hidden records cannot reach the partial evaluation |
| Documentation | every registry evidence test exists; JSON Schema and the error-code reference match the code; the API example script runs |

## 6. Examples (Linux, CLI)

| Example | Result |
|---|---|
| 01 textbook | EBVs equal the V-form reference to 1e-10; textbook values reproduced |
| 02 wwt REML | converged in 7 AI iterations; σ²a 3.552, σ²e 12.360, h² 0.223 (SE 0.047); 3 typing errors quarantined |
| 03 nlb repeatability | converged in 15 iterations; σ²a 0.030, σ²pe 0.025, σ²e 0.335, h² 0.078 (SE 0.048) |
| 04 fec GBLUP | converged in 15 iterations on the 247 records of genotyped animals; h² 0.55 (SE 0.16), see F3 |
| 05 wwt single step | converged in 8 iterations; σ²a 3.220, σ²e 12.668, h² 0.203 (SE 0.044); G* matched to A22 means |
| 06 four traits + index | 8,506 equations, dense solve, residual 3.3e-15; index equals Σ aⱼ EBVⱼ (tested) |
| 07 selection index | T02 values |
| 08 wwt LR validation | 403 focal lambs; partial REML converged (σ²a 3.92, σ²e 12.32); Δp = 0.057 kg (95% bootstrap interval −0.25, 0.21), b_w\|p = 0.99 (0.90, 1.24), ρ_wp = 0.67 (0.58, 0.78); 1,000 sire-cluster replicates, none failed |
| 09 mating plan | C_t 0.0410, ceiling 0.0506 (ΔF 1%) active; maximum-merit coancestry would be 0.120; KKT stationarity 9.2e-11; integer plan 20 sires × 150 ewes, coancestry 0.05057 ≤ ceiling after 1 repair and 2 improving moves, merit gap 0.0019 SCU; 331 pairs forbidden by relationship, 258 by recessive risk; mean progeny F 0.00003; all plan checks true |
| 10 fec BayesC | converged after one extension (8,000 iterations × 4 chains, native kernel, 25 s); h² 0.31 (SD 0.12); worst scalar R-hat 1.005, worst GEBV R-hat 1.003; smallest bulk ESS 813; predictive p-values 0.30–0.74. **BayesCπ on the same data did not pass within 16,000 iterations (π₀ mixing) and was withheld (exit status 6)** — intended behaviour |
| 11 wwt genetic groups | random groups: REML converged in 6 iterations (σ²a 4.23, σ²e 11.51); fixed groups: estimability rank 13/13, no reliabilities reported; group solutions A_16_19 / A_20_24 / B = +2.57 / +3.32 / −1.63 (random), +3.80 / +4.98 / −0.80 (fixed); realised group means +2.79 / +5.14 / −1.22 (single replicate, see §7) |

## 7. Statistical calibration studies (G4)

### 7.1 EBV calibration, 50 replicates (`benchmarks/calibration_study.py`)

Sheep generator, seeds 1–50, weaning weight, selection candidates of the
last season. Means ± Monte-Carlo SE across replicates. Bias = mean(TBV − EBV).

| Scenario | Slope b(TBV \| EBV) | Bias (kg) | Realized accuracy | Model accuracy | MSE / mean PEV | Coverage of 95% intervals |
|---|---:|---:|---:|---:|---:|---:|
| pedigree BLUP, true variances | 0.978 ± 0.018 | 0.016 ± 0.058 | 0.612 ± 0.010 | 0.623 ± 0.002 | 1.027 ± 0.023 | 0.948 ± 0.003 |
| pedigree BLUP, REML variances | 0.964 ± 0.023 | −0.064 ± 0.077 | 0.610 ± 0.010 | 0.629 ± 0.005 | 1.070 ± 0.033 | 0.942 ± 0.004 |
| single step (match_a22, 5% blend), true variances | 0.977 ± 0.019 | **0.660 ± 0.067** | 0.607 ± 0.011 | 0.638 ± 0.002 | **1.283 ± 0.042** | **0.917 ± 0.005** |

Pedigree BLUP is calibrated within Monte-Carlo error (F1 of round 1 was a
single-replicate artefact). Single step is under-predicting by 0.66 kg and
over-confident (finding F6).

### 7.2 Genetic groups, 20 replicates per scenario (`benchmarks/upg_study.py`)

Example-11 design. Bias = mean(EBV − TBV); lambs: all recorded lambs.

| Scenario | Model | corr(EBV, TBV), lambs | Slope, lambs | Bias purchased rams (kg) | Bias 2024 lambs (kg) |
|---|---|---:|---:|---:|---:|
| linear trend of breeder A (default) | random groups, ratio 1 | 0.753 ± 0.008 | 1.021 ± 0.020 | −1.088 ± 0.115 | −1.346 ± 0.144 |
| | fixed groups | 0.760 ± 0.007 | 1.007 ± 0.012 | −0.420 ± 0.187 | −0.681 ± 0.224 |
| | no groups | 0.714 ± 0.007 | 0.944 ± 0.021 | −1.825 ± 0.076 | −2.189 ± 0.077 |
| step trend (control: groups correctly specified) | random groups, ratio 1 | 0.757 ± 0.007 | 1.014 ± 0.019 | −0.948 ± 0.114 | −1.001 ± 0.141 |
| | fixed groups | 0.761 ± 0.006 | 0.981 ± 0.013 | −0.151 ± 0.187 | −0.199 ± 0.224 |
| | no groups | 0.717 ± 0.007 | 0.940 ± 0.021 | −1.797 ± 0.076 | −1.998 ± 0.076 |
| step trend, random ratio 25 | random groups, ratio 25 | 0.758 ± 0.007 | 1.000 ± 0.017 | −0.236 ± 0.177 | −0.276 ± 0.206 |

Groups raise the EBV–TBV correlation by about 0.04 and remove most of the
bias of purchased rams. With correctly specified groups (step scenario)
fixed-group biases are within about one Monte-Carlo SE of zero (group
effects −0.06 ± 0.15, −0.18 ± 0.25, −0.21 ± 0.21 kg from the realised group
means). Random groups with ratio 1 are shrunk towards zero and, through the
year effects, bias all groups downwards; with ratio 25 they approach the
fixed-group result. These are properties of the declared models, not
computational errors (the computation is verified against explicit models
in §5); see F7.

### 7.3 Simulation-based calibration of the Bayesian samplers (`benchmarks/sbc_bayes.py`)

300 replicates per method; prior draws of all parameters, 150 records × 40
markers, 2 chains × 2,200 iterations (200 posterior draws per replicate);
chi-square uniformity p-values of the ranks (10 bins; failure if p < 0.001).

| Method | σ²e | marker variance | var(Wβ) | β₁ | GEBV₁ | π₀ |
|---|---:|---:|---:|---:|---:|---:|
| BRR | 0.328 | 0.590 | 0.189 | 0.052 | 0.043 | – |
| BayesA | 0.271 | 0.954 | 0.649 | 1.000 | 0.042 | – |
| BayesB | 0.474 | 0.776 | 0.953 | 0.915 | 0.457 | – |
| BayesC | 0.732 | 0.494 | 0.029 | 0.252 | 0.676 | – |
| BayesCpi | 0.823 | 0.624 | 0.077 | 0.249 | 0.963 | 0.886 |
| BayesR | 0.833 | 0.654 | 0.638 | 0.072 | 0.988 | 0.634 |

All 32 tests passed (smallest p = 0.029; 3 of 32 below 0.05, where 1.6 are expected by chance). SBC has limited power against small errors: passing supports, but does not prove, that each sampler computes the posterior of its stated model. The mean normalised rank of every quantity lies between 0.47 and 0.56 (0.5 expected).

### 7.4 Other

REML calibration: across 40 replicates simulated from the model (σ²a = 2,
σ²e = 3, 250 animals), mean estimates lay within 3 Monte-Carlo SE of the
truth (`test_reml.py::test_reml_calibration_by_simulation`). The round-1
single-replicate check (`benchmarks/simulation_check.py`) remains available
for the four-trait example.

## 8. Findings and risks

| ID | Finding | Status | Assessment and next action |
|---|---|---|---|
| F1 | Under-dispersion and bias of candidate EBVs in one replicate (round 1) | **resolved** | 50 replicates (§7.1): slope 0.978 ± 0.018, bias 0.02 ± 0.06 kg, coverage 0.948 for pedigree BLUP |
| F2 | For nlb (a thresholded count) the model reliability (√ = 0.21) understates realized accuracy (0.38) | open | expected scale mismatch of a linear model on a liability trait; threshold model (M10); do not use nlb reliabilities for decisions |
| F3 | GBLUP h² for fec from 247 records is 0.55 ± 0.16 (pedigree 0.26, single step 0.30, simulated ≈ 0.25) | open (explained) | small selected subset; REML with dense G matches the independent optimum; the manual advises single step over GBLUP subsets |
| F4 | fec model reliability (0.48) understates realized accuracy (0.58) in the four-trait run | open | the calibration study covered weaning weight only; extend it to the multi-trait model |
| F5 | Exact PEV for > 30,000 equations and REML beyond the dense memory limit are refused | open (by design) | sparse selected inversion; approximate reliabilities |
| F6 | Single step (match_a22, 5% blend, 2,000 SNPs) biased by +0.66 kg (TBV − EBV) with PEV understated by 28% and coverage 0.917 | open | mathematics verified (G1–G3); model misspecification between the genomic and pedigree bases; next: metafounders or other base alignment, then repeat §7.1 |
| F7 | Genetic groups leave residual bias when the group level drifts within a period; random groups with a small ratio are strongly shrunk | open (documented) | define groups by shorter periods where data allow; choose the ratio deliberately (§7.9 of the manual); fixed groups where estimable |
| F8 | BayesCπ on example 10 data mixes slowly in π₀ and was withheld after 16,000 iterations | open (behaves as designed) | fix π₀ (BayesC) or run longer; a reparameterized π₀ update is a candidate improvement |

## 9. Defects found and fixed

| Round | Defect | How found | Fix | Regression test |
|---|---|---|---|---|
| 1 | `frequency_source = "training_genotyped"` counted animals whose only record had been quarantined | self-review | frequency sample built from QC-validated records | `test_training_frequency_sample_uses_qc_passed_records` |
| 1 | REML crawled with EM towards an exact zero and hit the iteration limit | engineered boundary test | active-set boundary detection with a Kuhn-Tucker check | `test_forced_boundary_case_is_detected` |
| 1 | `auto` chose sparse direct (15 s) where PCG takes 0.26 s | benchmark | PCG for large systems without PEV, with a recorded fallback | `test_auto_selection_and_pcg_fallback` |
| 1 | Dense PEV formed the full inverse and symmetrized it with extra copies (15.3 s) | profiling | diagonal blocks from L⁻¹; in-place symmetrization | multi-trait and BLUP reference tests |
| 1 | Pure-Python inbreeding took 90 s on a deep 100k pedigree | profiling | optional C++20 kernel (ADR 0002) | `test_native_kernel_matches_python_reference` |
| 2 | OCS multiplier sign and LP KKT signs wrong | independent KKT check | multipliers recomputed from the active set | `test_ocs_matches_independent_optimizer_and_kkt` |
| 2 | Rounding contributions to whole matings could exceed the coancestry ceiling | property test | ceiling-preserving local search; gap reported | `test_integer_repair_meets_ceiling_and_is_bounded_by_continuous_optimum` |
| 2 | ESS truncation differed from the reference algorithm (up to 0.9%); MCSE used the SD of split draws | cross-check against ArviZ | Stan's initial positive/monotone sequence reproduced exactly; SD of all draws | `test_agreement_with_arviz_reference_values` |
| 2 | Per-GEBV diagnostics silently absent when draws exceed the storage limit | documentation review | reported as `not_computed` | `test_skipped_gebv_diagnostics_are_reported_not_silent` |
| 2 | Confounded fixed groups were not detected before solving (Cholesky succeeded on a numerically singular system) | unit test | SVD estimability test of `[X, ZQ]` before solving | `test_fixed_groups_confounded_with_intercept_are_refused` |
| 2 | `abp validate` ignored `[upg]` group codes and PLINK input | review | both passed through | `test_validate_and_pedigree_commands_understand_group_codes`, `test_plink_and_dosage_inputs_give_identical_gblup` |

Errors in *test or benchmark construction* that were caught and corrected
(production code unaffected): a counterexample built with the wrong V (T05),
a NumPy boolean sum in a Mendelian fixture, a reference solve under a
monkeypatch, an SLSQP reference that failed on its own (replaced by
trust-constr), a closed-form OCS test with a negative contribution, a
fixed-group test whose design was in fact estimable, and a benchmark that
gave the two sweep kernels different starting residuals. Each fix kept the
original acceptance threshold.

## 10. Not run (and why)

| Item | Reason |
|---|---|
| Windows performance measurements and interactive use | only CI execution (§3a); no timings or desktop testing on Windows |
| CUDA / GPU | no CUDA implementation exists (requests are refused or fall back explicitly) |
| Real-data validation (G5) | no authorized real data; the LR workflow is ready for it |
| LR population-accuracy estimator | its published form has a 2019 erratum that could not be verified here (the source was not reachable) |
| Posterior SBC near the target data | the prior SBC (§7.3) checks the computation over the prior; a posterior SBC is a further step |
| Comparison with BLUPF90, MiXBLUP, ASReml, DMU, JWAS, BGLR | not installed or licensed in this environment; must be run under a pre-registered protocol |
| Independent mature simulator (AlphaSimR, QMSim, XSim) | not installed; ABP's generators are independent of its solver code but are not mature external simulators |
| Multi-trait calibration study | F4; the study covered weaning weight only |
| Installation from a built wheel or installer | no binary packaging yet (source install only) |
