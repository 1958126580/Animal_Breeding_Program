# Validation report: ABP 0.1.0

Date: 2026-09-25 · Platform actually executed: **Linux x86_64 only**.
Raw logs: `docs/validation/`. Status vocabulary: passed, failed, blocked,
not_run.

## 1. Summary by acceptance gate

| Gate | Meaning | Status | Evidence |
|---|---|---|---|
| G0 source and target | versions, licenses, estimands, contracts, base | passed | `docs/method_registry.toml`, `contracts/`, `docs/license_inventory.md` (project license: open owner decision) |
| G1 mathematics | independent derivations, analytical cases, dimension, limit and equivalence tests | passed | §3, §4 |
| G2 numerics | residuals, convergence, boundaries, exact references | passed | §4, §5 |
| G3 software | ID mapping, bad inputs, recovery, interface consistency | passed on Linux | §5, §6 |
| G4 statistical calibration | simulation bias and coverage | **partial** | REML: 40-replicate calibration passed. EBVs: one replicate of one scenario (§7). |
| G5 external validity | real data, time or population hold-out | **not_run** | no real data were available or authorized |
| G6 scale and platform | measured resources; Windows, Linux, GPU | **partial** | Linux measured (`docs/benchmarks.md`). Windows: CI job defined, result on GitHub. GPU: no CUDA path exists. |
| G7 decision and release | feasible plans, installation reproduction, evidence package | **not_run** | OCS and mating not implemented; no binary release; project license not chosen |

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

## 3. Commands executed and results

| Command | Result | Log |
|---|---|---|
| `python -m pytest -v` (native C++ kernel) | **96 passed**, 0 failed, 0 skipped, 37.3 s | `docs/validation/pytest-linux-py311-native.log`, `junit-linux-py311-native.xml` |
| `ABP_DISABLE_NATIVE=1 python -m pytest -v` (pure-Python kernels) | **96 passed**, 0 failed, 0 skipped, 37.6 s | `docs/validation/pytest-linux-py311-python-kernels.log`, `junit-linux-py311-python-kernels.xml` |
| `abp selftest` with and without the native kernel | RESULT: PASS (both) | `docs/validation/selftest-linux.log` |
| `abp run` examples 01–06, `abp index` example 07 | all exit status 0 | `docs/validation/examples-linux.log` |
| `python benchmarks/simulation_check.py` | completed | `docs/validation/simulation_check.json`, `.log` |
| `python benchmarks/run_benchmarks.py --full` | completed | `benchmarks/results/2026-09-25-linux-x86_64.json` |
| `bash packaging/linux/abp.sh run …` (Chinese + space output path, then a repeat without `--force`) | exit 0, then exit 2 (`OUTPUT_EXISTS`) with launcher log | manual run, recorded here |

## 4. Analytical gold standards (spec §8.2)

Expected values were derived by hand or come from the spec. None was
produced by ABP. Tolerance: `|x − ref| ≤ 1e−10 + 1e−8 |ref|`.

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

## 5. Independent-reference, property and failure tests

`tests/reference/dense_reference.py` uses different formulations (tabular A,
marginal V-form GLS/BLUP, V-form REML likelihood and score) and imports
nothing from `abp`.

| Area | Checks (all passed) |
|---|---|
| Pedigree | A, F, A⁻¹A = I and log\|A\| vs the tabular method on random inbred pedigrees; C++ vs Python kernel (1e-13); input-order invariance; unknown parents as distinct base animals; cycle, self-parent and same-sire-and-dam rejection |
| BLUP | V-form with a generalized inverse for rank-deficient X plus a permanent-environment term; dense, sparse and PCG agreement; invariance to the constraint choice and record order; kg→g scaling; invalid variances rejected; PCG non-convergence is an error; automatic PCG → direct fallback recorded |
| REML | log L, score and AI matrix vs V-form (three components); optimum vs independent Nelder-Mead (rtol 2e-5), including the dense-G path; EM = AI; engineered boundary gives exactly σ²e = var(y); non-convergence is an error; checkpoint resume = fresh fit; mismatched checkpoint refused |
| Genomic | hand G; allele-flip invariance; GBLUP = SNP-BLUP (including unphenotyped animals and with a ridge); singular G needs an explicit policy; A22 tuning; workflow GBLUP and single-step vs explicit-H V-form (1e-8); training frequency sample uses QC-passed records only |
| Multi-trait | missing patterns, trait-specific fixed effects and a structural residual zero vs V-form (dense and sparse); decomposition into single-trait models; one-trait degeneration; unit change; non-PD covariance rejected |
| Index | unit invariance; restricted index vs independent KKT solution; collinear information reported; selection intensity; EBV-index reliability |
| QC | each seeded pedigree error detected with the right code; all problems reported together; clean data produce no false alarm; genotype contract errors; call-rate, monomorphic and ambiguous-SNP handling; Mendelian sample-swap detection |
| Workflow | end-to-end example 01 vs independent reference; byte-identical reruns; `--force` semantics; failure leaves only `<out>.failed-*` with a failed manifest; quarantine lists; unknown spec keys rejected; repeated records without PE refused; Chinese and space paths; bad UTF-8 and ragged rows located by line; CUDA request refused or recorded as a CPU fallback; manifest contract for passed and failed runs |
| Leakage | changing phenotypes excluded by an approved rule leaves every output byte unchanged |
| Documentation | every registry evidence test exists; JSON Schema matches the validator; the API example script runs |

## 6. Examples (Linux, CLI)

| Example | Result |
|---|---|
| 01 textbook | EBVs equal the V-form reference to 1e-10; textbook values reproduced |
| 02 wwt REML | converged in 7 AI iterations; σ²a 3.552, σ²e 12.360, h² 0.223 (SE 0.047); 3 typing errors quarantined; residual 1.9e-15 |
| 03 nlb repeatability | converged in 15 iterations; σ²a 0.030, σ²pe 0.025, σ²e 0.335, h² 0.078 (SE 0.048) |
| 04 fec GBLUP | converged in 15 iterations on the 247 records of genotyped animals; h² 0.55 (SE 0.16), see finding F3 |
| 05 wwt single step | converged in 8 iterations; σ²a 3.220, σ²e 12.668, h² 0.203 (SE 0.044); G* matched to A22 means |
| 06 four traits + index | 8,506 equations, dense solve, residual 3.3e-15; index equals Σ aⱼ EBVⱼ (tested) |
| 07 selection index | T02 values |

## 7. Simulation-based check (G4, single replicate)

`benchmarks/simulation_check.py` compares EBVs with the hidden true breeding
values of the synthetic flock (seed 20260925). Selection candidates
(lambs born 2025, n = 403):

| Example / trait | Realized accuracy | Model accuracy √(mean rel) | Slope TBV on EBV | Bias (TBV − EBV) |
|---|---:|---:|---:|---:|
| 02 wwt (pedigree, REML) | 0.563 | 0.557 | 1.160 | −0.358 kg |
| 05 wwt (single step, REML), genotyped lambs | 0.624 | 0.588 | 1.366 | +0.201 kg |
| 02 wwt (pedigree), same genotyped lambs | 0.585 | 0.561 | 1.237 | −0.372 kg |
| 06 wwt (multi-trait) | 0.562 | 0.577 | 1.073 | −0.579 kg |
| 06 fat | 0.546 | 0.569 | 1.055 | −0.038 mm |
| 06 fec | 0.583 | 0.484 | 1.306 | +0.199 |
| 06 nlb1 (TBV on the liability scale) | 0.383 | 0.214 | not comparable | not comparable |

REML calibration: across 40 replicates simulated from the model
(σ²a = 2, σ²e = 3, 250 animals), mean estimates lay within 3 Monte-Carlo SE
of the truth (`test_reml.py::test_reml_calibration_by_simulation`).

## 8. Findings and risks (open)

| ID | Finding | Assessment | Next action |
|---|---|---|---|
| F1 | Candidate EBVs are under-dispersed in one replicate (slopes 1.07–1.37) and biased by up to ±0.6 kg (≈0.3 σa) | Plausible causes: one replicate (slope SE ≈ 0.1), REML σ²a below the base value, mass selection of rams, differing pedigree/genomic bases. Not yet distinguishable from a defect. | multi-replicate calibration study (≥ 50 replicates) and LR validation (M13) |
| F2 | For nlb (a thresholded count) the model reliability (√ = 0.21) understates realized accuracy (0.38) | expected scale mismatch of a linear model on a liability trait | threshold model (M10); do not use nlb reliabilities for decisions |
| F3 | GBLUP h² for fec from 247 records is 0.55 ± 0.16. Pedigree REML on all 955 records gives 0.26 ± 0.07 and single step gives 0.30 ± 0.07 (simulated ≈ 0.25). | small, selected subset; REML with a dense G matches the independent V-form optimum (test added) | none for correctness; the manual advises single step over GBLUP subsets |
| F4 | fec model reliability (0.48) understates realized accuracy (0.58) in the four-trait run | one replicate; covariances fixed at generator values | include in the calibration study |
| F5 | Exact PEV for > 30,000 equations and REML beyond the dense memory limit are refused | a deliberate limit, documented | sparse selected inversion; approximate reliabilities |

## 9. Defects found and fixed during this round

| Defect | How found | Fix | Regression test |
|---|---|---|---|
| `frequency_source = "training_genotyped"` counted animals whose only record had been quarantined | self-review | frequency sample built from QC-validated records | `test_training_frequency_sample_uses_qc_passed_records` |
| REML crawled with EM towards an exact zero and hit the iteration limit | engineered boundary test | active-set boundary detection with a Kuhn-Tucker check | `test_forced_boundary_case_is_detected` |
| `auto` chose sparse direct (15 s) where PCG takes 0.26 s | benchmark | PCG for large systems without PEV, with a recorded fallback | `test_auto_selection_and_pcg_fallback` |
| Dense PEV formed the full inverse and symmetrized it with extra copies (15.3 s) | profiling | diagonal blocks from L⁻¹; in-place symmetrization | multi-trait and BLUP reference tests (identical results) |
| Pure-Python inbreeding took 90 s on a deep 100k pedigree | profiling | optional C++20 kernel (ADR 0002) | `test_native_kernel_matches_python_reference` |

Errors in *test construction* that were caught and corrected (production
code unaffected): a counterexample built with the wrong V (T05), a NumPy
boolean sum in a Mendelian test fixture, and a reference solve that ran under
a monkeypatch. Each fix kept the original acceptance threshold.

## 10. Not run (and why)

| Item | Reason |
|---|---|
| Windows execution | no Windows machine in this session. The CI workflow runs the suite, self-test and launcher on `windows-latest`; see the Actions tab for the actual result. |
| CUDA / GPU | no CUDA implementation exists (requests are refused or fall back explicitly) |
| Real-data validation (G5), forward-in-time validation, LR statistics | no authorized real data; LR not implemented |
| Comparison with BLUPF90, MiXBLUP, ASReml, DMU, JWAS, BGLR | not installed or licensed in this environment; must be run under a pre-registered protocol |
| Independent mature simulator (AlphaSimR, QMSim, XSim) | not installed; ABP's generator is independent of its solver code but is not a mature external simulator |
| Multi-replicate EBV calibration (G4) | planned (F1) |
| Installation from a built wheel or installer | no binary packaging yet (source install only) |
