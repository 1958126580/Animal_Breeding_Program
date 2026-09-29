# Validation report: ABP 0.5.0

Date: 2026-09-29 (rounds 4 and 5; round 3: 2026-09-28; rounds 1–2: 2026-09-25) · Platforms executed: **Linux x86_64** (build machine, full
evidence below) and **Windows Server 2025 + Ubuntu** in GitHub Actions
(round 1: run
[36130441504](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/36130441504);
round 2: run
[36151891410](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/36151891410);
round 3: run
[36447242937](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/36447242937);
round 5: run
[36524938324](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/36524938324); §3a).
Raw logs: `docs/validation/`. Status vocabulary: passed, failed, blocked,
not_run.

## 1. Summary by acceptance gate

| Gate | Meaning | Status | Evidence |
|---|---|---|---|
| G0 source and target | versions, licenses, estimands, contracts, base | passed | `docs/method_registry.toml`, `contracts/`, `docs/license_inventory.md` (project license: open owner decision) |
| G1 mathematics | independent derivations, analytical cases, dimension, limit and equivalence tests | passed | §4, §5 |
| G2 numerics | residuals, convergence, boundaries, exact references | passed | §4, §5; MCMC diagnostics equal ArviZ to ≤ 8·10⁻¹⁶ |
| G3 software | ID mapping, bad inputs, recovery, interface consistency | passed (Linux; Windows CI) | §3, §5, §6 |
| G4 statistical calibration | simulation bias and coverage | **partial** | pedigree BLUP calibrated over 50 replicates; single step with `match_a22` biased (F6); single step on a metafounder base unbiased and calibrated with 10,000 SNPs (§7.4); multi-trait BLUP calibrated (§7.5); two metafounders reduce base bias (§7.6); threshold-model reliabilities calibrated (§7.7); REML 40 replicates; PEV including REML uncertainty calibrated for pedigree BLUP (§7.9); APY with 300 of 476 genotyped as core equivalent to the exact G (§7.9); Laplace-REML liability variances biased (F11, §7.9); genetic groups 20 replicates × 3 scenarios; SBC of all six Bayesian samplers (§7) |
| G5 external validity | real data, time or population hold-out | **not_run** | no real data were available or authorized (the LR workflow exists and was run on synthetic data) |
| G6 scale and platform | measured resources; Windows, Linux, GPU | **partial** | Linux measured (`docs/benchmarks.md`), including exact PEV for 100,500 equations and the matrix-free single step (50,000 animals, 6,000 genotyped: 1.7 GB instead of 10.9 GB); Windows functional tests in CI (round 3: 211 tests passed on Windows and Linux, §3a), no Windows timings; no CUDA path |
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
| `python -m pytest -v` (native C++ kernel; round 5) | **259 passed**, 0 failed, 0 skipped, 214.2 s | `docs/validation/pytest-linux-py311-native.log`, `junit-linux-py311-native.xml` |
| `ABP_DISABLE_NATIVE=1 python -m pytest -v` (pure-Python kernels; round 5) | PYKERNEL_RESULT | `docs/validation/pytest-linux-py311-python-kernels.log`, `junit-linux-py311-python-kernels.xml` |
| `abp selftest` with and without the native kernel (round 5) | RESULT: PASS (both; T01–T15; T11 skipped without the kernel) | `docs/validation/selftest-linux.log` |
| `abp run` examples 01–06, 08, 10, 11 (+ `analysis_fixed.toml`), 12–14, `abp index` 07, `abp mate` 09, `compare.py`, `api_example.py` (round 5, incl. API sections 18–21) | EXAMPLES_RESULT | `docs/validation/examples-linux.log` |
| `python benchmarks/calibration_study.py --design random_selection --scenarios pedigree_true --replicates 200` and `--design default` (round 5, F9) | completed | `docs/validation/f9_random_selection.json`, `f9_default.json`, `.log` |
| `python benchmarks/calibration_study.py --scenarios pedigree_reml single_step_reml --replicates 50 --workers 4` (round 5, Kackar–Harville PEV) | completed; round-3 values reproduced | `docs/validation/calibration_kh.json`, `.log` |
| `python benchmarks/calibration_study.py --scenarios single_step_true single_step_apy150_true single_step_apy300_true --replicates 50 --workers 4` (round 5, APY) | completed; `single_step_true` reproduced round 3 exactly | `docs/validation/calibration_apy.json`, `.log` |
| `python benchmarks/threshold_study.py --replicates 30 --workers 4` (round 5, with `threshold_laplace`) | completed; round-4 scenarios reproduced exactly; Laplace REML withheld in 6 of 30 | `docs/validation/threshold_study.json`, `.log` |
| `python benchmarks/ssmf_benchmark.py` (round 5) | completed; matrix-free EBVs equal the explicit ones to 6.4e-10 | `benchmarks/results/ssmf.json`, `.log` |
| `python benchmarks/calibration_study.py --replicates 50 --workers 4` (round 3, six scenarios) | completed; round-2 scenarios reproduced exactly | `docs/validation/calibration_study.json`, `.log` |
| `python benchmarks/run_benchmarks.py --only selinv metafounders` (round 3) | completed | `benchmarks/results/2026-09-28-linux-x86_64-round3.json` |
| `python benchmarks/calibration_study.py --design {qtl_like_snp, no_blend, dense_markers, random_selection}` (round 4, 50 replicates each) | completed | `docs/validation/f6_factor_*.json`, `.log` |
| `python benchmarks/mt_calibration_study.py --replicates 50 --workers 4` (round 4) | completed (after the stored-zero fix, §9) | `docs/validation/mt_calibration_study.json`, `.log` |
| `python benchmarks/two_metafounder_study.py --replicates 30 --workers 4` (round 4) | completed | `docs/validation/two_metafounder_study.json`, `.log` |
| `python benchmarks/threshold_study.py --replicates 30 --workers 4` (round 4) | completed (linear model withheld in 3 replicates) | `docs/validation/threshold_study.json`, `.log` |
| `python benchmarks/run_benchmarks.py --only ldl apy` (round 4) | completed | `benchmarks/results/2026-09-29-linux-x86_64-round4.json` |
| `python benchmarks/upg_study.py` (linear trend; `--trend step`; `--trend step --ratio 25`), 20 replicates each | completed | `docs/validation/upg_study*.json`, `.log` |
| `python benchmarks/sbc_bayes.py --replicates 300` | all 32 rank-uniformity tests passed (smallest p = 0.029) | `docs/validation/sbc_bayes.json`, `.log` |
| `ARVIZ_PATH=… python benchmarks/mcmc_diagnostics_crosscheck.py` | max relative difference 2.2e-16 (R-hat), 7.8e-16 (bulk ESS), 2.5e-16 (tail ESS), 2.3e-16 (MCSE) | `docs/validation/mcmc_diagnostics_crosscheck.json` |
| `python benchmarks/run_benchmarks.py --only bayes ocs upg plink` | completed | `benchmarks/results/2026-09-25-linux-x86_64-round2.json` |
| `python benchmarks/run_benchmarks.py --full` (round 1), `benchmarks/simulation_check.py` | completed (round 1) | `benchmarks/results/2026-09-25-linux-x86_64.json`, `docs/validation/simulation_check.*` |

### 3a. Continuous integration

Round 1: run 36130441504 (commit 2cd2922): all 7 jobs succeeded —
Windows Server 2025 and Ubuntu × Python 3.11/3.12/3.13 (C++20 kernel built
with MSVC on Windows; `abp selftest` PASS; launcher runs into
`%RUNNER_TEMP%\结果 输出`) and a pure-Python-kernel job. Round 2: run
[36151891410](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/36151891410)
(commit c6f72df), all 7 jobs succeeded:

| Job | Result | Notes from the job log |
|---|---|---|
| windows-latest / Python 3.11, 3.12, 3.13 | success | Windows Server 2025; C++20 kernel built with MSVC; `abp selftest` PASS including T07–T11; **173 passed** (3.13: NumPy 2.5.3, SciPy 1.18.1, 157.7 s); launcher run into `%RUNNER_TEMP%\结果 输出` passed and the repeat without `--force` returned exit status 2 |
| ubuntu-latest / Python 3.11, 3.12, 3.13 | success | **173 passed**; `abp.sh` launcher run with a Chinese/space path |
| ubuntu / pure-Python kernels (`ABP_DISABLE_NATIVE=1`, Python 3.12) | success | self-test PASS with T11 skipped (no native kernel); **173 passed** in 439.6 s |

Round 3: every push of the round ran the full matrix; all four runs
succeeded (runs 36443416871, 36443864697, 36446319558 and
[36447242937](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/36447242937),
commit 77c47c2, 7 of 7 jobs). The Windows Server / Python 3.13 job built the
C++20 kernels (including `ml_general`, `symbolic_cholesky`, `takahashi`) with
MSVC, passed the self-test and reported **211 passed** in 145.9 s, the same
count as Linux; the launcher run into `%RUNNER_TEMP%\结果 输出` passed and the
repeat returned exit status 2. That commit predates the T12/T13 self-test
checks and the documentation of this round. The final round-3 commit 5bb3db1
(ABP 0.3.0, self-test T01–T13) ran as
[36451049420](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/36451049420):
7 of 7 jobs succeeded; Windows / Python 3.13 **211 passed** in 138.3 s.

Round 4: runs for commits 1e2e1b7 (success), 2e328f9 and 81eb6ba (Linux
jobs green, the three Windows jobs failed on two tests that wrote
backslash paths into TOML — a test defect, see §9), 0957b92 (superseded)
and the fix 1184384:
[36507918397](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/36507918397),
7 of 7 jobs succeeded (Windows and Ubuntu × Python 3.11–3.13, pure-Python
kernels); the Windows / Python 3.13 job ran ABP 0.4.0 including the launcher
run into `%RUNNER_TEMP%\结果 输出`.

Round 5: runs [36522550336](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/36522550336)
(commit 07b9532), [36523269114](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/36523269114)
(ddefc85) and [36524938324](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/36524938324)
(ba094c1, version 0.5.0 with the matrix-free single step): 7 of 7 jobs
succeeded in each; the Windows / Python 3.13 job of the last run reported
ABP 0.5.0 from the launcher run into `%RUNNER_TEMP%\结果 输出` and the repeat
returned exit status 2 (`ABP-E503`). CI_FINAL

The byte-reproducibility tests of examples 09 and 11 passed on every job,
including Windows (line endings fixed by `.gitattributes`) and the newer
NumPy of the Python 3.12/3.13 jobs.

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
| T12 | two founders from one metafounder (γ = 0.4) and their offspring | A = [[1.2,.4,.8,.4],[.4,1.2,.8,.4],[.8,.8,1.2,.4],[.4,.4,.4,.4]]; inverse [[1.875,.625,−1.25,−1.25],[.625,1.875,−1.25,−1.25],[−1.25,−1.25,2.5,0],[−1.25,−1.25,0,5]]; log det = log(0.4·0.8²·0.4) | passed (both kernels) | self-test; `test_metafounders.py::test_extended_matrix_inverse_and_logdet_match_tabular_reference` (general case) |
| T13 | selected inversion of tridiag(−1, 2, −1) | diagonal (¾, 1, ¾), off-diagonal ½ | passed (both kernels) | self-test; `test_selinv.py::test_selected_inverse_equals_dense_inverse_on_pattern` (general case) |
| T11 | BayesC inclusion probability for one marker | ratio of exact Gaussian marginal likelihoods | passed (decision flips exactly at the exact probability) | `test_bayes.py::test_inclusion_probability_matches_exact_marginal_likelihood` |

## 5. Independent-reference, property and failure tests

`tests/reference/dense_reference.py` uses different formulations (tabular A,
marginal V-form GLS/BLUP, V-form REML likelihood and score) and imports
nothing from `abp`.

| Area | Checks (all passed) |
|---|---|
| Pedigree | A, F, A⁻¹A = I and log\|A\| vs the tabular method on random inbred pedigrees; C++ vs Python kernel (1e-13); input-order invariance; unknown parents as distinct base animals; cycle, self-parent and same-sire-and-dam rejection |
| Metafounders | `A_ext`, its inverse and log-determinant = independent implementation of the recursive definition (1e-11; a mutation of the dᵢ formula is caught); zero inputs reduce to ordinary Meuwissen–Luo; C++ = Python kernel on 4,000 animals × 3 metafounders; BLUP solutions, PEV and reliability = V-form (1e-9); REML log L and score = V-form (1e-10); single-step `H⁻¹`, diag(H), log det and covariances with the metafounders = dense `H`; base contrast (EBV, PEV, reliability) = V-form PEV matrix; Γ estimation by gene drop from two correlated base populations (4 seeds, error < 0.03, correction reduces the diagonal bias); missing dosages; non-estimable metafounders, non-PD Γ, too-large Γ, unassigned unknown parents and invalid Γ files refused; spec rules; example 12 |
| Multi-trait REML | Henderson log L = marginal V-form log L exactly (2 and 3 traits, missing patterns); scores = central differences of the V-form (1e-6); EM increases log L and stays PD; optimum = independent Nelder-Mead on a Cholesky parametrisation (2e-3); dense = sparse trace paths (log L, score, EM, AI); stored zeros of K⁻¹ ignored; boundary optimum reported as `ABP-E300`; example 13 |
| Threshold model | posterior mode = independent BFGS on a `scipy.stats.norm` objective (2e-5); Laplace PEV = inverse of a numerical Hessian (0.2%); category relabelling invariance and order reversal; refusals (residual ≠ 1, one category, non-integer codes); spec rules; example 14 |
| Sparse LDL' | solve, log-determinant and selected inverse = dense algebra (native and Python, with a dense row); native ordering and numeric = Python reference (identical order); fill < ½ of natural order and ≤ 1.2 × SuperLU's MMD; non-PD refused; BLUP LDL' = SuperLU = dense |
| APY | inverse = dense inverse of the implied G_APY, log-determinant; exact with one non-core animal; singular G handled below its rank, refused above; single step with APY = explicit dense H; workflow (EBV correlation > 0.98 with exact G on example 05) |
| Selected inversion | every stored entry of the selected inverse = `numpy.linalg.inv` (1e-15, three random matrices); C++ = Python kernels; symbolic pattern closes over an exact numeric cancellation that SuperLU drops; out-of-pattern requests refused; memory guard; sparse PEV and reliability = dense path (1e-12); sparse REML log L, score, EM update and AI = dense path; multi-trait PEV blocks = dense path for coupled and uncoupled traits; 100,500-equation PEV = unit-vector solves (6.7e-16, benchmark) |
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
| 12 wwt single step, metafounder base | REML converged in 8 iterations (σ²_MF 4.64, σ²e 12.54); γ̂ 0.552 (0.563 uncorrected); without rescaling, mean diagonal/off-diagonal of G05 1.284/0.588 vs A22^Γ 1.292/0.601 |
| 13 multi-trait REML (wwt, fat, fec) | converged in 15 iterations (3 EM + 12 AI), sparse trace path, 7.9 s; G0 diagonal 3.96 / 0.263 / 0.401, genetic correlations 0.29 (wwt-fat), −0.37 (wwt-fec), −0.13 (fat-fec) (single replicate) |
| 14 nlb threshold model | converged in a few Newton iterations; thresholds 0 (fixed) and 1.587; liability-scale EBVs with Laplace reliabilities in [0, 1] |
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
| single step (match_a22, 5% blend), REML variances | 1.052 ± 0.024 | **0.817 ± 0.072** | 0.604 ± 0.011 | 0.616 ± 0.005 | **1.568 ± 0.076** | **0.881 ± 0.009** |
| single step on a metafounder base (G05, 5% blend with A22^Γ), true variances¹ | 0.957 ± 0.017 | 0.057 ± 0.058 | 0.612 ± 0.010 | 0.645 ± 0.002 | 1.077 ± 0.024 | 0.941 ± 0.003 |
| single step on a metafounder base, REML variances | 0.984 ± 0.020 | 0.104 ± 0.070 | 0.610 ± 0.011 | 0.638 ± 0.004 | 1.165 ± 0.037 | 0.930 ± 0.004 |

¹ σ²_MF = 4.0/(1 − γ̂/2) with each replicate's γ̂ (estimated from genotypes
and pedigree only; mean γ̂ 0.540 ± 0.002), σ²e = 12.25. Metafounder
scenarios are scored on `ebv_vs_base`/`pev_vs_base` (EBV relative to the
metafounder), because the simulated TBVs are deviations from the founders'
mean; on the absolute scale MSE/mean PEV is 0.52 (true) and 0.56 (REML):
absolute PEVs contain the uncertainty of the base level, which is common to
all animals. REML under metafounders estimated σ²_MF = 5.32 ± 0.14,
equivalent to 3.88 ± 0.10 on the conventional scale (true 4.0).

Pedigree BLUP is calibrated within Monte-Carlo error (F1 of round 1 was a
single-replicate artefact). Single step with `match_a22` under-predicts by
0.66 kg (0.82 kg with REML) and is over-confident (finding F6). On a
metafounder base the bias is not distinguishable from zero and the PEV
understatement falls from 28% to 8% (true variances) and from 57% to 17%
(REML), but it is still about three Monte-Carlo SE from calibration (F6
remains partly open). The three round-2 scenarios were re-run with the same
seeds in round 3 and reproduced all 96 printed replicate values
(slope, bias and coverage to three decimals) exactly.

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

### 7.4 F6 factor study (round 4; `benchmarks/calibration_study.py --design`)

Metafounder single step with the converted true variances; each design
changes one factor of the default (50 replicates each; raw results
`docs/validation/f6_factor_*.json`).

| Design | Bias (kg) | Slope | MSE / mean PEV | Coverage |
|---|---:|---:|---:|---:|
| default (2,000 SNPs; §7.1) | 0.057 ± 0.058 | 0.957 ± 0.017 | 1.077 ± 0.024 | 0.941 ± 0.003 |
| QTL frequencies drawn like the SNPs (U(0.05, 0.95)) | 0.095 ± 0.068 | 0.958 ± 0.019 | 1.086 ± 0.040 | 0.940 ± 0.005 |
| no 5% blend (G05 alone) | 0.066 ± 0.058 | 0.954 ± 0.017 | 1.086 ± 0.025 | 0.940 ± 0.003 |
| **10,000 SNPs** | **0.002 ± 0.051** | **0.996 ± 0.015** | **0.978 ± 0.017** | **0.952 ± 0.002** |
| rams chosen at random (no selection) | 0.128 ± 0.043 | 0.990 ± 0.017 | 1.053 ± 0.027 | 0.942 ± 0.003 |

The residual over-confidence of §7.1 is explained by marker density: with
2,000 SNPs G measures the relationships at the QTL with error that the model
does not represent; with 10,000 SNPs single step on the metafounder base is
calibrated. The γ-mismatch and blend hypotheses of round 3 are rejected.
Observation (F9): under random selection pedigree BLUP with true variances
also shows a bias of 0.135 ± 0.044 kg (≈ 3 MC SE) — not explained yet.

### 7.5 Multi-trait calibration (round 4; `benchmarks/mt_calibration_study.py`)

Weaning weight, fat depth and faecal egg count jointly; 50 replicates;
candidates of the last season.

| Model | Trait | Slope | Bias | Realized acc. | Model acc. | MSE / PEV | Coverage |
|---|---|---:|---:|---:|---:|---:|---:|
| true covariances | wwt | 0.978 ± 0.018 | 0.014 ± 0.058 | 0.613 | 0.624 | 1.027 ± 0.023 | 0.948 |
| | fat | 0.993 ± 0.020 | 0.004 ± 0.013 | 0.592 | 0.607 | 0.984 ± 0.017 | 0.953 |
| | fec | 1.034 ± 0.026 | −0.005 ± 0.014 | 0.535 | 0.525 | 0.982 ± 0.021 | 0.952 |
| multi-trait REML | wwt | 0.959 ± 0.023 | −0.069 ± 0.078 | 0.611 | 0.634 | 1.087 ± 0.036 | 0.940 |
| | fat | 1.008 ± 0.031 | 0.002 ± 0.014 | 0.584 | 0.608 | 1.094 ± 0.038 | 0.939 |
| | fec | 1.032 ± 0.033 | 0.011 ± 0.016 | 0.522 | 0.531 | 1.129 ± 0.050 | 0.935 |

Mean REML estimates of G0 (± MC SE): diagonal 4.21 ± 0.12, 0.245 ± 0.009,
0.298 ± 0.012 (true 4.0, 0.25, 0.3025); covariances 0.315 ± 0.022, −0.040 ±
0.027, −0.011 ± 0.007 (true 0.3, 0, 0). Multi-trait BLUP is calibrated with
the true covariances (F4 was a single-replicate artefact); with estimated
covariances PEV is 9–13% too small (estimation error not propagated), as
for single-trait REML (7%).

### 7.6 Two base populations (round 4; `benchmarks/two_metafounder_study.py`)

Independent gene drop of two breeds with differentiated frequencies, F1 and
composite generations, genotyped unrecorded candidates; 30 replicates;
truth = TBV relative to breed A's base. All models REML.

| Model | Realized acc. | Slope | RMS bias across replicates | MSE / PEV | Coverage |
|---|---:|---:|---:|---:|---:|
| conventional single step (match_a22 + 5% blend) | 0.426 ± 0.014 | 0.935 ± 0.044 | 0.326 | 1.211 ± 0.068 | 0.923 |
| one metafounder | 0.426 ± 0.014 | 0.936 ± 0.044 | 0.320 | 1.204 ± 0.065 | 0.923 |
| two metafounders (by breed), reference A | 0.426 ± 0.014 | 0.939 ± 0.044 | **0.219** | **1.112 ± 0.059** | **0.934** |

Estimated Γ minus the SNP-based truth: −0.001 (AA), −0.007 (AB), −0.015 (BB).

### 7.7 Linear vs threshold model for litter size (round 4; `benchmarks/threshold_study.py`)

30 replicates; ewes with records; truth = liability TBV.

| Model | Withheld | Realized accuracy | Model accuracy | Model / realized |
|---|---:|---:|---:|---:|
| linear repeatability, REML | 3 of 30 (σ²a at the boundary) | 0.447 ± 0.011 | 0.374 ± 0.027 | 0.827 ± 0.054 |
| threshold, known liability variances | 0 | 0.451 ± 0.011 | 0.443 ± 0.001 | 1.001 ± 0.031 |

Paired difference of realized accuracy (threshold − linear, 27 replicates):
0.0036 ± 0.0008. The threshold model's reliabilities are calibrated; the
linear model's understate its accuracy (F2). The threshold model used the
true liability variances, the linear model estimated its own, so the
calibration comparison is partly confounded by that difference.

### 7.9 Round-5 studies

**F9 repeated (200 replicates, `pedigree_true`).** Random ram selection:
bias 0.049 ± 0.024 kg, slope 1.008 ± 0.007, MSE/PEV 0.993 ± 0.010, coverage
0.950; seeds 1–50 reproduce the round-4 value 0.135 ± 0.044, seeds 51–200
give 0.020 ± 0.029. Default design: bias 0.031 ± 0.031. The simulated TBVs
are centred on the founders' sample mean, the estimand of the model (checked
in `abp/examples/sheep.py`). Conclusion: the round-4 observation was sampling
variation; a residual bias of at most ≈ 0.05 kg (0.025 σa) cannot be
excluded with this number of replicates.

**PEV including REML uncertainty (50 replicates, `calibration_kh.json`).**

| Scenario | MSE/PEV plug-in | MSE/PEV incl. VC uncertainty | Coverage plug-in | Coverage incl. |
|---|---:|---:|---:|---:|
| pedigree_reml | 1.070 ± 0.033 | 1.017 ± 0.031 | 0.942 ± 0.004 | 0.949 ± 0.004 |
| single_step_reml (match_a22) | 1.568 ± 0.076 | 1.509 ± 0.074 | 0.881 ± 0.009 | 0.888 ± 0.009 |

The correction calibrates pedigree BLUP with REML variances. It cannot
repair the single-step configuration whose genetic base is mismatched (F6);
its bias (0.82 ± 0.07 kg) is unchanged.

**APY (50 replicates, `calibration_apy.json`; 476 genotyped animals, 2,000 SNPs).**

| Scenario | Bias | Realized accuracy | Model accuracy | MSE/PEV | Coverage |
|---|---:|---:|---:|---:|---:|
| single_step_true (exact G⁻¹) | 0.660 ± 0.067 | 0.607 ± 0.011 | 0.638 | 1.283 ± 0.042 | 0.917 |
| APY, 300 core | 0.665 ± 0.067 | 0.606 ± 0.011 | 0.634 | 1.280 ± 0.043 | 0.916 |
| APY, 150 core | 0.718 ± 0.066 | 0.597 ± 0.011 | 0.618 | 1.287 ± 0.045 | 0.916 |

Paired differences from the exact G⁻¹: realized accuracy −0.0013 ± 0.0006
(300 core) and −0.0103 ± 0.0018 (150 core); bias +0.005 ± 0.002 and
+0.058 ± 0.007 kg. APY with 300 core animals is practically equivalent; 150
core animals cost about 0.01 in accuracy. APY was studied on the biased
`match_a22` configuration (the question was APY vs exact G, not the base).

**Threshold model with Laplace-approximate REML (30 replicates).** Genetic
liability variance 0.085 ± 0.013 (true 0.111), permanent environment
0.130 ± 0.012 (true 0.111); 6 of 30 withheld at the search bound; model /
realized accuracy 0.802 ± 0.057 (known variances: 1.001 ± 0.031). Finding F11.

### 7.8 Other

REML calibration: across 40 replicates simulated from the model (σ²a = 2,
σ²e = 3, 250 animals), mean estimates lay within 3 Monte-Carlo SE of the
truth (`test_reml.py::test_reml_calibration_by_simulation`). The round-1
single-replicate check (`benchmarks/simulation_check.py`) remains available
for the four-trait example.

## 8. Findings and risks

| ID | Finding | Status | Assessment and next action |
|---|---|---|---|
| F1 | Under-dispersion and bias of candidate EBVs in one replicate (round 1) | **resolved** | 50 replicates (§7.1): slope 0.978 ± 0.018, bias 0.02 ± 0.06 kg, coverage 0.948 for pedigree BLUP |
| F2 | For nlb (a thresholded count) the model reliability (√ = 0.21) understates realized accuracy (0.38) | **resolved** (round 4) | threshold model: model/realized accuracy 1.001 ± 0.031 over 30 replicates (§7.7); the linear model understates it (0.827) — use `type = "categorical"` for such traits |
| F3 | GBLUP h² for fec from 247 records is 0.55 ± 0.16 (pedigree 0.26, single step 0.30, simulated ≈ 0.25) | open (explained) | small selected subset; REML with dense G matches the independent optimum; the manual advises single step over GBLUP subsets |
| F4 | fec model reliability (0.48) understates realized accuracy (0.58) in the four-trait run | **resolved** (round 4) | 50-replicate multi-trait study (§7.5): model accuracy 0.525 vs realized 0.535 ± 0.012, MSE/PEV 0.98, coverage 0.952 — a single-replicate artefact |
| F5 | Exact PEV for > 30,000 equations and REML beyond the dense memory limit are refused | **resolved** (round 3) | sparse selected inversion: exact PEV for 100,500 equations in 16.9 s (3.2 s with ABP's LDL' in round 4), agreement with unit-vector solves 6.7e-16; REML on the sparse path; the remaining limit is factor memory (checked, `ABP-E500`). Approximate reliabilities for systems whose factor does not fit remain a roadmap item |
| F6 | Single step (match_a22, 5% blend, 2,000 SNPs) biased by +0.66 kg (TBV − EBV) with PEV understated by 28% and coverage 0.917 | **resolved; cause of the remainder identified** (rounds 3–4) | on a metafounder base: bias 0.06 ± 0.06 kg, MSE/PEV 1.08 ± 0.02 (§7.1); the remaining 8% over-confidence is due to marker density — with 10,000 SNPs MSE/PEV 0.978 ± 0.017, coverage 0.952 (§7.4); not due to the QTL frequency distribution or the blend. With sparse panels single-step reliabilities stay slightly optimistic |
| F7 | Genetic groups leave residual bias when the group level drifts within a period; random groups with a small ratio are strongly shrunk | open (documented) | define groups by shorter periods where data allow; choose the ratio deliberately (§7.9 of the manual); fixed groups where estimable |
| F8 | BayesCπ on example 10 data mixes slowly in π₀ and was withheld after 16,000 iterations | open (behaves as designed) | fix π₀ (BayesC) or run longer; a reparameterized π₀ update is a candidate improvement |
| F9 | Under random selection (F6 factor design) pedigree BLUP with true variances shows a candidate bias of 0.135 ± 0.044 kg (≈ 3 MC SE) | **resolved** (round 5) | 200 replicates: 0.049 ± 0.024 kg; the 150 new replicates 0.020 ± 0.029; TBV centring checked (§7.9). Sampling variation |
| F10 | Multi-trait REML does not handle boundary optima | **addressed** (round 5; opt-in) | default still stops with `ABP-E300`; `reml.boundary = "reduced_rank"` or `reml.rank` fits G0 = ΛΛ′ (tested against the V-form and an independent optimizer). Singular R0 still stops. No calibration study of the reduced-rank fit yet |
| F11 | Laplace-approximate REML of liability variances is biased with 2–3 categorical records per animal (genetic variance −23%, 20% of runs at the search bound) | open (documented) | known liability variances recommended (manual, report text, registry gate G4 failed); a better estimator (MCMC or higher-order Laplace) is a next task |
| F12 | REML-based PEV was 7–13% optimistic in every study | **resolved for single-trait** (round 5) | Kackar–Harville PEV: pedigree REML MSE/PEV 1.017 ± 0.031; multi-trait REML PEV remains conditional |

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
| 3 | Absolute EBVs under metafounders looked badly over-dispersed in PEV (MSE/PEV 0.37) against founder-centred TBVs | first calibration replicate | not a computational error: the absolute PEV contains the uncertainty of the base level; EBVs relative to the reference metafounder with their own PEV and reliability added (`ebv_vs_base`) | `test_pedigree_metafounder_run_matches_v_form_reference` |
| 3 | Single-step base contrast used `Cov(u_i, u_ref)` from `A^Γ` instead of `H` (reliability −0.0073 for one founder) | reliability range check in replicate 42 of the calibration study | covariances with the metafounders taken from `H` | `test_single_step_with_metafounders_matches_dense_h` |
| 3 | `ABPError` raised in a worker process could not be unpickled, which hung the study's process pool | calibration study | picklable by name, message and details | `test_abp_error_survives_pickling` |
| 3 | Calibration study set its one-thread BLAS limit after importing NumPy (load average 19 on 4 cores, no replicate finished in 20 minutes) | monitoring the study | limit set before the import | — (benchmark script) |
| 4 | Stored zeros in `A⁻¹` (exact cancellation when a son is mated to his own dam) were requested from the selected inverse and refused; affected multi-trait REML (seed 2 of the study) and, latently, single-trait sparse REML | multi-trait calibration study | zero-weight entries dropped before the lookup | `test_stored_zeros_in_k_inverse_are_ignored`, `test_sparse_reml_ignores_stored_zeros_in_k_inverse` |
| 4 | Multi-trait REML reported "not converged" when the optimum was on the boundary | API example data | boundary diagnosis (`ABP-E300`) when full AI steps keep leaving the positive-definite region | `test_boundary_optimum_is_reported_as_not_identifiable` |
| 4 | Two new tests pasted absolute paths into TOML strings; on Windows the backslashes are TOML escapes, so CI runs 36503107523 and 36505883867 failed on the three Windows jobs (Linux green) | CI | paths written in POSIX form, as the existing Bayes workflow test already did | `test_workflow_example05_with_apy`, `test_lr_validation_on_a_metafounder_base` (CI Windows) |
| 4 | Minimum-degree ordering took 83 s because an intercept row absorbed every elimination | profiling | dense rows eliminated last (as in AMD): about 2 s | `test_ldl_solve_logdet_inverse_equal_dense` (dense-row case) |
| 5 | none in released code this round (see the construction errors below) | — | — | — |

Round-5 errors in the construction of tests, studies and benchmarks
(production code unaffected): the independent Laplace reference's BFGS
optimizer stepped into infeasible thresholds (log of a non-positive
probability) — it now rejects such steps; two reduced-rank workflow test
data sets were refused by QC (repeated records without a permanent-environment
term, then animals used as both sire and dam) — the test now builds a
sex-consistent pedigree with one record per animal; the threshold-study log
was lost because repository files were stashed while the study was running —
the study was re-run (identical JSON); the first threshold-study summary
restricted the round-4 comparison to replicates where all three scenarios
issued results (24 instead of 27) — the round-4 definition was restored and
reproduces the round-4 numbers exactly; the matrix-free benchmark first
compared the reference mode with itself (reported 0.0) and was first run
while another job shared the CPU — both corrected, and the table in
`docs/benchmarks.md` is from the undisturbed rerun.

Errors in *test or benchmark construction* that were caught and corrected
(production code unaffected): a counterexample built with the wrong V (T05),
a NumPy boolean sum in a Mendelian fixture, a reference solve under a
monkeypatch, an SLSQP reference that failed on its own (replaced by
trust-constr), a closed-form OCS test with a negative contribution, a
fixed-group test whose design was in fact estimable, and a benchmark that
gave the two sweep kernels different starting residuals; in round 3, a
workflow test whose data contained a repeated record (correctly refused by
QC), a REML workflow test on 7 records (correctly stopped at the boundary;
replaced by a likelihood-level test), and a native-vs-Python comparison with
a relative tolerance on values near cancellation (replaced by an absolute
tolerance at the value scale; both kernels are separately checked against
the dense inverse to 1e-15); in round 4, an LR test that compared the
LR "whole" evaluation (variances from the partial data by design) with the
main evaluation (switched to known variances), a regression test that
picked an A⁻¹ pair that was not empty, a native-kernel test that did not
honour `ABP_DISABLE_NATIVE`, and a threshold study that stopped when ABP
correctly withheld a linear-model result (now counted as withheld). Each fix
kept the original acceptance threshold.

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
| Kackar–Harville PEV for multi-trait REML | not implemented (multi-trait PEV remains conditional on the covariances) |
| Calibration study of reduced-rank multi-trait REML | not run (only exactness tests and one-data-set optimum checks) |
| Matrix-free single step above 50,000 animals / 6,000 genotyped, or with approximate reliabilities | not run; reliabilities are not available on that path |
| Installation from a built wheel or installer | no binary packaging yet (source install only) |
