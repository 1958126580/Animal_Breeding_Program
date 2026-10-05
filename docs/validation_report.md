# Validation report: ABP 0.16.0

Date: 2026-10-05 (round 16; rounds 13–15: 2026-10-03; rounds 11–12: 2026-10-02; rounds 9–10: 2026-10-01; rounds 6–8: 2026-09-30; rounds 4 and 5: 2026-09-29; round 3: 2026-09-28; rounds 1–2: 2026-09-25) · Platforms executed: **Linux x86_64** (build machine, full
evidence below) and **Windows Server 2025 + Ubuntu** in GitHub Actions
(round 1: run
[36130441504](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/36130441504);
round 2: run
[36151891410](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/36151891410);
round 3: run
[36447242937](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/36447242937);
round 5: run
[36524938324](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/36524938324);
round 6: run
[36657626436](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/36657626436);
round 7: run
[36673772284](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/36673772284);
round 8: run
[36727601564](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/36727601564);
round 9: run
[36832541614](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/36832541614);
round 10: run
[36883815725](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/36883815725);
round 11: run
[37008139663](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/37008139663);
round 13: run
[37089590066](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/37089590066);
round 14: run
[37100055863](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/37100055863);
round 15: run
[37214291766](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/37214291766); §3a).
Raw logs: `docs/validation/`. Status vocabulary: passed, failed, blocked,
not_run.

## 1. Summary by acceptance gate

| Gate | Meaning | Status | Evidence |
|---|---|---|---|
| G0 source and target | versions, licenses, estimands, contracts, base | passed | `docs/method_registry.toml`, `contracts/`, `docs/license_inventory.md` (project license: open owner decision) |
| G1 mathematics | independent derivations, analytical cases, dimension, limit and equivalence tests | passed | §4, §5 |
| G2 numerics | residuals, convergence, boundaries, exact references | passed | §4, §5; MCMC diagnostics equal ArviZ to ≤ 8·10⁻¹⁶ |
| G3 software | ID mapping, bad inputs, recovery, interface consistency | passed (Linux; Windows CI) | §3, §5, §6 |
| G4 statistical calibration | simulation bias and coverage | **partial** | pedigree BLUP calibrated over 50 replicates; single step with `match_a22` biased (F6); single step on a metafounder base unbiased and calibrated with 10,000 SNPs (§7.4); multi-trait BLUP calibrated (§7.5); two metafounders reduce base bias (§7.6); threshold-model reliabilities calibrated (§7.7); REML 40 replicates; PEV including REML uncertainty calibrated for pedigree BLUP (§7.9); APY with 300 of 476 genotyped as core equivalent to the exact G (§7.9); Laplace-REML liability variances biased (F11, §7.9); threshold Gibbs sampler unbiased for the genetic liability variance (§7.10); multi-trait PEV including REML uncertainty calibrated (§7.10); reduced-rank REML 100 replicates (§7.10); threshold priors, rank selection and reduced-rank Kackar–Harville PEV (§7.11); sampled-reliability estimator, EBVs of rank-selected models, multi-trait threshold model (prior-dependent genetic correlation, F18) (§7.12); posterior PEV of the Bayesian multi-trait linear model vs REML + Kackar–Harville (F16 partly resolved) (§7.13); several categorical traits: EBVs more accurate than single-trait models, liability variance of a binary trait and genetic correlations prior-dependent (F21, §7.18); maternal animal model: Bayesian EBVs calibrated, posterior mean of the maternal variance ~7% above REML (F19 revised); REML estimates unbiased over 200 replicates, plug-in maternal PEV optimistic when the variance is underestimated (F20) (§7.15–7.17); genetic groups 20 replicates × 3 scenarios; SBC of all six Bayesian samplers (§7) |
| G5 external validity | real data, time or population hold-out | **partial** (rounds 15–16) | two public real data sets (Holstein lactations, pedigreemm; genotyped mice, BGLR): REML equal to an independent V-form implementation to 10⁻⁸ logL (after fixing the defect this comparison found) and to the established packages pedigreemm, rrBLUP and sommer (variances ≤ 5.5·10⁻⁵ relative, EBV correlation 1.0); forward-in-time LR validation on the mice tested in 2004 (bias −0.010 ± 0.026, dispersion 0.90 ± 0.02: over-dispersed, F22); cross-validated predictive ability (§7.19–7.20). Not the target populations; no comparison with BLUPF90/ASReml/MiXBLUP/DMU |
| G6 scale and platform | measured resources; Windows, Linux, GPU | **partial** | Linux measured (`docs/benchmarks.md`), including exact PEV for 100,500 equations and the matrix-free single step (50,000 animals, 6,000 genotyped: 1.7 GB instead of 10.9 GB; 200,000 animals, 30,000 genotyped, int8 genotypes: 6.75 GB; complete `abp run` with 120,000 animals / 30,000 genotyped from PLINK as int8: 261 s, 7.38 GB; 206 s after the round-8 speed-ups); Windows functional tests in CI (round 3: 211 tests passed on Windows and Linux, §3a), no Windows timings; no CUDA path |
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
| `python -m pytest -v` (native C++ kernel; round 15, with the Kuhn–Tucker fix) | **365 passed**, 0 failed, 1,557 s (real-data example runs included: data fetched locally) | `docs/validation/pytest-linux-py311-native.log`, `junit-linux-py311-native.xml` |
| `ABP_DISABLE_NATIVE=1 python -m pytest -v` (pure-Python kernels; round 15) | **359 passed**, 0 failed, 6 skipped (native-only comparisons), 5,098 s (alongside the milk benchmark); started before the test-data fix, so `tests/test_reml.py` was re-run on the final code with both kernels: 11 passed each | `docs/validation/pytest-linux-py311-python-kernels.log`, `junit-linux-py311-python-kernels.xml`, `pytest-reml-rerun-round15.log` |
| `abp selftest` with and without the native kernel (round 15) | RESULT: PASS (both) | `docs/validation/selftest-linux.log` |
| `abp run` examples 01–06, 08, 10–19 (+ variants), the real-data examples 20 (two specs) and 21 (data fetched locally), `abp index` 07, `abp mate` 09, `compare.py`, `api_example.py` (round 15) | all 27 exit status 0; example 20 SCS 21 s, three-trait 10 s, example 21 20 s | `docs/validation/examples-linux.log` |
| `python benchmarks/real_milk_validation.py` (round 15, real data, after the fix) | REML equal to the independent V-form REML for 3 models (logL differences ≤ 5.4·10⁻⁹); CV over cows r = 0.087 / 0.154 / 0.074; 3,793 s; before the fix: milk repeatability at the boundary, −2.36 logL (§7.19) | `docs/validation/real_milk_validation.json`, `.log`, `real_milk_validation_before_fix.json` |
| `python benchmarks/real_mice_validation.py` (round 15, real data) | GBLUP vs PBLUP: random folds equal for body weight; across families only GBLUP predicts (r = 0.27 / 0.11); 327 s | `docs/validation/real_mice_validation.json`, `.log` |
| `python -m pytest -v` (native C++ kernel; round 14) | **357 passed**, 0 failed, 0 skipped, 1493.6 s (with the examples and the pure-Python suite alongside) | `docs/validation/pytest-linux-py311-native.log`, `junit-linux-py311-native.xml` |
| `ABP_DISABLE_NATIVE=1 python -m pytest -v` (pure-Python kernels; round 14) | **351 passed**, 0 failed, 6 skipped (native-only comparisons), 6240.0 s (with the examples and the native suite alongside) | `docs/validation/pytest-linux-py311-python-kernels.log`, `junit-linux-py311-python-kernels.xml` |
| `abp selftest` with and without the native kernel (round 14) | RESULT: PASS (both; T01–T18; T11 skipped without the kernel) | `docs/validation/selftest-linux.log` |
| `python benchmarks/two_categorical_study.py --replicates 30 --workers 3` (round 14, F21) | completed, 2,560 s; 24 of 30 met the strict criteria; see §7.18 | `docs/validation/two_categorical_study.json`, `.log` |
| `python benchmarks/maternal_study.py --method reml --replicates 200 --workers 3` (round 13, F19/F20) | completed, 373 s (5.5 s per replicate); 192 converged, 3 boundary, 5 `ABP-E300`; see §7.17 | `docs/validation/maternal_study_reml.json`, `.log` |
| `python benchmarks/maternal_study.py --replicates 20 --workers 3 --animals 1500 --iterations 4000 --prior equal` and `--prior flat --out …_flat_r12.json` (round 12, F19) | completed, 2,835 s and 2,866 s (402 / 408 s per replicate); 16 of 20 and 1 of 20 met the strict criteria; see §7.16 | `docs/validation/maternal_study_equal_prior.json`, `maternal_study_flat_r12.json`, `.log` |
| `python benchmarks/maternal_study.py --replicates 20 --workers 3 --animals 1500 --iterations 4000` (round 11) | completed, 8,672 s; 1 of 20 passed the strict R-hat/ESS criteria (median worst R-hat 1.015); see §7.15 and F19 | `docs/validation/maternal_study.json`, `.log` |
| `python benchmarks/bayes_vs_reml_study.py --replicates 50 --workers 3 --iterations 4000` (round 9, F16; 2 scenarios) | completed (see §7.13) | `docs/validation/bayes_vs_reml_study.json`, `.log` |
| `python benchmarks/bayes_vs_reml_study.py --scenarios t2_full --seeds 1 2 3 4 5 6 --iterations 16000` (round 9, convergence check) | see §7.13 | `docs/validation/bayes_vs_reml_study_16000it.json`, `.log` |
| `python benchmarks/inbreeding_benchmark.py` (round 9) | completed; identical F, 5 pedigrees | `docs/validation/inbreeding_benchmark.json`, `.log` |
| `abp run` examples 01–06, 08, 10, 11 (+ `analysis_fixed.toml`), 12–19 (+ 18 `analysis_equal_prior.toml` and `analysis_reml.toml`), `abp index` 07, `abp mate` 09, `compare.py`, `api_example.py` (round 14, with both test suites alongside) | all 24 exit status 0; examples 15–18 gave the earlier posterior means; example 19 (two categorical traits): 16,000 iterations, 1,145 s | `docs/validation/examples-linux.log` |
| `python benchmarks/bayes_vs_reml_study.py --scenarios t3_full --seeds 1..30 --prior pheno` (round 10, F16 follow-up) | completed (§7.14) | `docs/validation/bayes_vs_reml_study_pheno_prior.json`, `.log` |
| `python benchmarks/pev_estimator_study.py` (round 8) | completed; SD ratio 0.575 | `docs/validation/pev_estimator_study.json`, `.log` |
| `python benchmarks/rank_selection_study.py --replicates 100 --workers 4` (round 8, 4 scenarios) | completed | `docs/validation/rank_selection_study.json`, `.log` |
| `python benchmarks/mt_threshold_study.py --replicates 30 --workers 4 --iterations 12000` (round 8; also `--iterations 3000` and `--g-prior true --replicates 12 --iterations 6000`) | completed; 30/30 converged | `docs/validation/mt_threshold_study.json`, `mt_threshold_study_3000it.json`, `mt_threshold_study_trueprior.json`, `.log` |
| `python benchmarks/mt_threshold_crosscheck.py` (round 8) | max \|z\| 1.47 | `docs/validation/mt_threshold_crosscheck.json`, `.log` |
| `python benchmarks/ssmf_workflow_large.py` and `ssmf_large.py` (round 8, after the speed-ups) | 205.9 s / 7.37 GB; A22 core columns 30.6 s (119.6) | `benchmarks/results/ssmf_workflow_large_r8.json`, `ssmf_large_r8.json`, `.log` |
| `python benchmarks/threshold_study.py --scenarios threshold_gibbs_prior005 threshold_gibbs_prior020 --replicates 30 --workers 4 --out docs/validation/threshold_prior_study.json` (round 7) | completed; 0 of 60 withheld | `docs/validation/threshold_prior_study.json`, `.log` |
| `python benchmarks/rr_calibration_study.py --replicates 100 --workers 4` (round 7: AIC rules, Kackar–Harville) | completed; round-6 values reproduced | `docs/validation/rr_calibration_study.json`, `.log` |
| `python benchmarks/rr_calibration_study.py --replicates 100 --workers 4 --g0 full --out docs/validation/rr_calibration_full.json` (round 7) | completed | `docs/validation/rr_calibration_full.json`, `.log` |
| `python benchmarks/ssmf_workflow_large.py` (round 7) | exit status 0, 261 s, 7.38 GB | `benchmarks/results/ssmf_workflow_large.json`, `.log` |
| `python benchmarks/calibration_study.py --design random_selection --scenarios pedigree_true --replicates 200` and `--design default` (round 5, F9) | completed | `docs/validation/f9_random_selection.json`, `f9_default.json`, `.log` |
| `python benchmarks/calibration_study.py --scenarios pedigree_reml single_step_reml --replicates 50 --workers 4` (round 5, Kackar–Harville PEV) | completed; round-3 values reproduced | `docs/validation/calibration_kh.json`, `.log` |
| `python benchmarks/calibration_study.py --scenarios single_step_true single_step_apy150_true single_step_apy300_true --replicates 50 --workers 4` (round 5, APY) | completed; `single_step_true` reproduced round 3 exactly | `docs/validation/calibration_apy.json`, `.log` |
| `python benchmarks/threshold_study.py --replicates 30 --workers 4` (round 5, with `threshold_laplace`) | completed; round-4 scenarios reproduced exactly; Laplace REML withheld in 6 of 30 | `docs/validation/threshold_study.json`, `.log` |
| `python benchmarks/threshold_study.py --replicates 30 --workers 4` (round 6, with `threshold_gibbs`) | completed; earlier scenarios reproduced exactly | `docs/validation/threshold_study.json`, `.log` |
| `python benchmarks/rr_calibration_study.py --replicates 100 --workers 4` (round 6) | completed (after the convergence-criterion fix, §9) | `docs/validation/rr_calibration_study.json`, `.log` |
| `python benchmarks/mt_calibration_study.py --replicates 50 --workers 4 --out docs/validation/mt_calibration_kh.json` (round 6) | completed; round-4 values reproduced | `docs/validation/mt_calibration_kh.json`, `.log` |
| `python benchmarks/ssmf_large.py` (round 6) | completed, 6.75 GB peak | `benchmarks/results/ssmf_large.json`, `.log` |
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

**Round 15** (ABP 0.15.0): commit c844eb0 (final code: Kuhn–Tucker fix, deterministic
test data), runs
[37214291766](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/37214291766)
and 37214287723, 7 of 7 jobs succeeded in each (Windows and Ubuntu, Python 3.11–3.13,
pure-Python kernels); the real-data example runs are skipped in CI (data not
redistributed). Earlier round-15 commits 8f1102a, c0b3c00 and 919d4bb failed only
`test_boundary_decision_is_invariant_to_the_unit` on 1–5 jobs: the test data changed with
the per-process hash seed (§9, round 15); fixed in 7960af5, green since.

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
returned exit status 2 (`ABP-E503`). The final documentation commit is listed in the HANDOFF.

Round 6: the pushes from 5c3ead5 to b569515 (11 runs, 36580112523 to
36651500257) failed on all 7 jobs because the published JSON schema was stale
(`test_published_json_schema_is_current`; in the pure-Python job 1 failed, 278
passed) — a process error: CI was not checked after each push this round (§9).
From the release commit 82936e3 on, CI is green:
[36652918117](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/36652918117),
36654338381 and, after the SuperLU fix,
[36657626436](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/36657626436)
(commit 398ed2c): 7 of 7 jobs succeeded (Windows and Ubuntu × Python
3.11–3.13, pure-Python kernels).

Round 7: CI was checked after every push. The runs for commits f5a2835,
9c244f1, b347451, 4f45c7a (run 36670006301), eb1a424 (36670220980), a9ffce4
([36672749790](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/36672749790))
and 4c06e84
([36673772284](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/36673772284))
succeeded on all 7 jobs (Windows and Ubuntu × Python 3.11–3.13, pure-Python
kernels), as did the release commit 329f8d5 (ABP 0.7.0):
[36681792730](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/36681792730).

Round 8: CI checked after every push; the runs for commits b0d3242
([36702726781](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/36702726781)),
d388116 (36706870919), d05b358 (36709822191) and 104e1f1
([36710892351](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/36710892351))
succeeded on all 7 jobs. The release commit fa2894b (ABP 0.8.0): run
[36727601564](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/36727601564),
7 of 7 jobs succeeded.

Round 9: commit a2fcdb2 (run
[36807935977](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/36807935977)):
6 of 7 jobs succeeded — the C++ depth kernel built with MSVC and GCC and its tests
passed on Windows and Linux — but the pure-Python job failed on a test defect (§9:
the new test expected the native kernel name under `ABP_DISABLE_NATIVE=1`). The fix,
commit d3e2a71 (run
[36812933794](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/36812933794)),
succeeded on all 7 jobs. The release commit 15d7d38 (ABP 0.9.0): run
[36832541614](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/36832541614),
7 of 7 jobs succeeded (pure-Python job 67 min).

Round 10: commit a569d8f (run
[36865978675](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/36865978675)
and the pull-request run 36865986803) succeeded on all 7 jobs, but the pure-Python job
took 2 h 17 min because two new statistical tests ran the Python sparse LDL for tens of
thousands of iterations; they were made smaller. The release commit 8a32efe (ABP
0.10.0): run
[36883815725](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/36883815725)
and pull-request run 36883826721, 7 of 7 jobs succeeded (pure-Python job 84 min).

Round 11: commit 7f30e96 (maternal effects; run
[36990166293](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/36990166293)
and pull-request run 36990173448) succeeded on all 7 jobs (pure-Python job 2 h 16 min,
after which the two maternal statistical tests were made smaller: with pure-Python
kernels locally 6 passed in 920.9 s). Commit 0c25b62 (maternal study, F19, evidence):
run [37008139663](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/37008139663)
and pull-request run 37008145319, 7 of 7 jobs succeeded (pure-Python job 1 h 56 min).

Round 12: commit 728dbe0 (dense trailing block in the sparse LDL'): run
[37024748732](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/37024748732)
and pull-request run 37024755087, 7 of 7 jobs succeeded — the new C++ split kernel built
with MSVC and GCC and its tests passed on Windows and Linux; the pure-Python job took
40 min (0c25b62: 1 h 56 min), because the Gibbs tests now factorize the dense part of the
coefficient matrix with LAPACK. Later commits of the round: see HANDOFF §2.

Round 13: commits 2b86935 (maternal REML), b520d82 (Kackar–Harville) and 62b2a7f
(docs, ABP 0.13.0): runs 37088933310, 37089408924 and
[37089590066](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/37089590066),
7 of 7 jobs succeeded on each (Windows and Linux, Python 3.11–3.13, pure-Python kernels).

Round 14: commits 683fe36 (several categorical traits), dd7ed61 and d80fc5c (docs, ABP
0.14.0): runs 37097684360, 37098406846 and
[37100055863](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/37100055863),
7 of 7 jobs succeeded on each.

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

### 7.10 Round-6 studies

**Threshold model, Gibbs sampler (30 replicates, `threshold_study.json`).**

| Scenario | Withheld | Genetic liability variance (true 0.111) | pe variance (true 0.111) | Model / realized accuracy |
|---|---:|---:|---:|---:|
| known liability variances | 0 | — | — | 1.001 ± 0.031 |
| Laplace-approximate REML | 6 | 0.085 ± 0.013 | 0.130 ± 0.012 | 0.802 ± 0.057 |
| Gibbs sampler (posterior mean) | 0 | 0.108 ± 0.009 | 0.138 ± 0.010 | 0.948 ± 0.029 |
| Gibbs sampler (posterior median) | 0 | 0.100 ± 0.009 | 0.132 ± 0.011 | — |

The Gibbs estimate of the genetic variance is unbiased; the
permanent-environment variance is 2.7 Monte-Carlo SE high (uniform priors pull
posterior means of small variances up). Realized accuracy 0.450 ± 0.010
(known variances 0.451).

**Multi-trait PEV including REML uncertainty (50 replicates, `mt_calibration_kh.json`).**
MSE/PEV with REML covariances, plug-in → corrected: wwt 1.087 ± 0.036 → 1.025 ±
0.033, fat 1.094 ± 0.038 → 1.048 ± 0.035, fec 1.129 ± 0.050 → 1.058 ± 0.044;
coverage 0.940/0.939/0.935 → 0.948/0.944/0.942.

**Reduced-rank REML (100 replicates, 1,000 animals, true G0 of rank 1,
`rr_calibration_study.json`).** Full-rank REML stopped at the boundary in 58
(converged in 42 with r_G 0.942 ± 0.008). Rank-1 REML converged in 100;
estimates (true): G0 1.988 ± 0.031 (1.96), 0.851 ± 0.015 (0.84), 0.370 ± 0.010
(0.36); R0 3.005 ± 0.026 (3.0), 0.881 ± 0.015 (0.9), 1.973 ± 0.013 (2.0). BLUP
with the estimates: MSE/PEV 1.021 ± 0.011 and 1.076 ± 0.031, coverage 0.947
and 0.940; with the true parameters 0.997 ± 0.007, coverage 0.950.

**Sampled reliabilities (matrix-free single step, example 05).** Against the
exact reliabilities of the explicit single step, with 300 simulations and
four different seeds, the mean error was between −0.0024 and +0.0007
(per-animal SE ≈ 0.04) and the spread of `(r̂ − r)/SE` 1.02–1.03.

### 7.11 Round-7 studies

**Threshold Gibbs sampler with proper priors (30 replicates, `ν = 4`,
`threshold_prior_study.json`).** Same data design as §7.10 (true genetic and
permanent-environment liability variances 0.111 each); prior scales
deliberately below and above the truth.

| Prior scale | Withheld | Genetic variance (posterior mean) | pe variance | Model / realized accuracy |
|---|---:|---:|---:|---:|
| uniform (§7.10) | 0 | 0.108 ± 0.009 | 0.138 ± 0.010 | 0.948 ± 0.029 |
| 0.05 | 0 | 0.090 ± 0.007 | 0.104 ± 0.007 | 0.906 ± 0.025 |
| 0.2 | 0 | 0.125 ± 0.005 | 0.140 ± 0.005 | 1.012 ± 0.026 |

The estimates follow the prior centre: with 2–3 records per animal the data
identify these variances weakly. Realized accuracy is unchanged (0.450 and
0.451). A proper prior therefore moves the bias rather than removing it; the
uniform prior stays the default (F13 stays open).

**Rank selection (100 replicates each, 1,000 animals;
`rr_calibration_study.json` true rank 1, `rr_calibration_full.json` true full
rank with r_G 0.595).**

| Rule | Rank 1 chosen, true rank 1 | Rank 1 chosen, true full rank |
|---|---:|---:|
| smallest AIC | 97 / 100 | 19 / 100 |
| AIC, lower rank only if smaller by ≥ 2 (ABP) | 58 / 100 | 0 / 100 |

With a full-rank truth the full-rank fit converged in 100/100 (r_G 0.598 ±
0.016). In the 19 wrong reductions of plain AIC the rank-1 BLUP understated
trait-2 PEV about fourfold (median MSE/PEV 4.3) and realized accuracy fell to
0.43 (0.54 with the true parameters) — hence the margin (F15). With a rank-1
truth the margin rule picks rank 1 exactly in the 58 replicates where the
full-rank fit stops at the boundary.

**Kackar–Harville PEV for reduced-rank fits** (rank-1 study, same 100
replicates): BLUP with the REML estimates, MSE/PEV
plug-in → corrected 1.021 ± 0.011 → 1.007 ± 0.010 (trait 1) and 1.076 ± 0.031 →
1.038 ± 0.029 (trait 2); coverage of 95% intervals 0.947 → 0.949 and 0.940 →
0.944 (true parameters: 0.997, coverage 0.950). The correction was available
in all 100 fits. Trait 2 remains 1.3 Monte-Carlo SE above 1.

**Workflow-level matrix-free single step** (`benchmarks/ssmf_workflow_large.py`):
`abp run` on generated files (pedigree
CSV with 120,000 animals, 110,000 records in 400 contemporary groups, PLINK
file with 30,000 genotyped animals × 20,000 SNPs), `genotype_storage = "int8"`,
APY with 5,000 core animals, matrix-free single step, 10 PEV simulations:
exit status 0, 261 s wall time, peak resident memory 7.38 GB (child process;
`benchmarks/results/ssmf_workflow_large.json`, `.log`). Genotype loading, QC,
frequencies and the APY/A22 operators took 182 s, PCG 5.5 s (69 iterations,
relative residual 9.8e-9), sampling 65 s (92 PCG iterations per simulation).
Mean reliability 0.32, mean Monte-Carlo SE 0.21 with 10 simulations. Held as
float64 with a missing mask, the genotypes alone would take 5.4 GB instead of
0.6 GB (arithmetic, not a measured run).

### 7.12 Round-8 studies

**Sampled reliabilities: orthogonal vs ratio estimator (`pev_estimator_study.json`).**
Small single step (80 animals, 25 genotyped, 50 records) with exact reliabilities
(mean 0.32, range 0.02–0.50); 300 seeds × 40 simulations, both estimators on the same
draws.

| Estimator | Mean error | SD of the error across seeds | Mean reported SE |
|---|---:|---:|---:|
| ratio `1 − mean d²/mean u*²` (round 6) | −0.0088 ± 0.0007 | 0.111 | 0.109 |
| orthogonal `mean h²/(mean h² + mean d²)`, bias-corrected (round 8) | −0.0002 ± 0.0004 | 0.064 | 0.059 |

SD ratio 0.575 (theory √r: 0.574), i.e. the precision of about three times as many
simulations. Without the second-order correction the orthogonal estimator had a mean
error of +0.0022 (5 MC SE); with it −0.0002.

**Rank selection: EBVs of the chosen model (`rank_selection_study.json`,**
100 replicates per scenario, 1,000 animals). MSE/PEV with the Kackar–Harville PEV of
the chosen model (true parameters: 0.99–1.01 throughout):

| Truth | Chosen rank | Trait 1 | Trait 2 | Trait 3 |
|---|---|---:|---:|---:|
| 2 traits, rank 1 | 1: 58, 2: 42 | 1.006 ± 0.010 | 0.960 ± 0.029 | – |
| 2 traits, full | 2: 100 | 1.020 ± 0.013 | 1.136 ± 0.045 | – |
| 3 traits, rank 2 | 2: 65, 3: 35 | 1.035 ± 0.014 | 1.012 ± 0.023 | 1.016 ± 0.027 |
| 3 traits, full | 3: 100 | 1.043 ± 0.013 | 1.135 ± 0.040 | 1.174 ± 0.069 |

The kept full-rank fits under a rank-1 truth (42 replicates) are conservative
(trait 2: 0.844 ± 0.029) with a small accuracy loss (0.737 vs 0.746). The low-h²
traits of the full-rank scenarios stay optimistic even with the correction (F16).

**Multi-trait threshold model (`mt_threshold_study.json`).** 30 replicates, 800 animals (a
three-category trait on the 360 females, one record each; a continuous trait on all
non-founders; true G0 = [[0.25, 0.5], [0.5, 4.0]], genetic correlation 0.5),
4 chains × 12,000 iterations; all 30 multi-trait and 30 single-trait runs converged.
IW prior ν = 5, G_prior = diag(0.2, 3.0) (no covariance):

| Quantity | Multi-trait | Single-trait (categorical only) | True |
|---|---:|---:|---:|
| liability genetic variance | 0.290 ± 0.022 | 0.259 ± 0.020 | 0.25 |
| genetic covariance | 0.316 ± 0.049 | – | 0.5 |
| genetic correlation | 0.275 ± 0.040 | – | 0.5 |
| continuous genetic variance | 3.81 ± 0.16 | – | 4.0 |
| realized accuracy, categorical | 0.414 ± 0.016 | 0.375 ± 0.014 | – |
| MSE / PEV, categorical | 1.025 ± 0.058 | 1.121 ± 0.059 | 1 |
| coverage 95%, categorical | 0.945 | 0.934 | 0.95 |

Accuracy gain of the joint model for the categorical trait: +0.039 ± 0.010 (paired).
The genetic correlation is pulled towards the prior's centre (0): with
`G_prior = G0` (12 replicates, 6,000 iterations,
`mt_threshold_study_trueprior.json`) it was 0.506 ± 0.038 and the accuracy gain
+0.075 ± 0.010. The data (single categorical records) identify the genetic
covariance weakly; the result depends on the prior (F18). A first run with 3,000
iterations (`mt_threshold_study_3000it.json`, 3/30 converged) gave the same
means (rG 0.274), so the pull is not a burn-in effect.

*Cross-check of the missing-data handling* (`mt_threshold_crosscheck.json`): the
current sampler and the first implementation (commit b0d3242, which imputes the
missing observations instead of integrating them out) agree on one data set with
50% of the categorical and 20% of the continuous values missing: all posterior
means within 1.5 Monte-Carlo SE.

**Example 15** (`abp run`, 2,108 animals, 1,894 weaning weights, 254 first-parity
litter sizes, IW prior ν = 5): converged after 20,000 iterations (4 chains, thin 10;
all R-hat ≤ 1.004, minimum EBV bulk ESS 1,530), 899 s on a loaded machine; 454 s
after the coefficient maps (examples log, with the test suite running alongside; the
same posterior means to four digits). Posterior
means (95% intervals): liability genetic variance of nlb1 0.35 (0.06–1.24),
weaning-weight genetic variance 3.94 (2.48–5.69), genetic correlation −0.36
(−0.73 to 0.18); the simulation's values (about 0.1, 4.0 and +0.1) lie inside the
intervals. The residual covariance, zero by design (lamb and ewe traits), was
estimated at a correlation of 0.28 (it could not be fixed at zero in version 0.8; since 0.9 it can, §7.13).

**Workflow-level single step after the round-8 speed-ups.** the same benchmark as §7.11
(`benchmarks/ssmf_workflow_large.py`, 120,000 animals, 30,000 genotyped from PLINK as
int8, APY with 5,000 core animals, 10 PEV simulations; `ssmf_workflow_large_r8.json`):
205.9 s instead of 261 s, peak memory 7.37 GB (7.38). The genotype and APY/A22 phase
took 130 s instead of 182 s (native Colleau product, fewer genotype copies); the mean
Monte-Carlo SE of the sampled reliabilities fell from 0.21 to 0.10 with the same 10
simulations (orthogonal estimator).

### 7.13 Round-9 studies

**F16: Bayesian posterior PEV vs REML + Kackar–Harville (`bayes_vs_reml_study.json`,
`.log`).** The `t2_full` and `t3_full` scenarios of the rank-selection study (1,000
animals, random pedigree, one record per animal, traits after the first 30% missing,
full-rank truth), 50 replicates each, the same data fitted four ways. MSE/PEV (1 =
calibrated; > 1 optimistic) and coverage of nominal 95% intervals, means ± MC SE:

| Scenario, trait (h²) | REML plug-in | REML + Kackar–Harville | Bayesian (posterior PEV) | true parameters |
|---|---:|---:|---:|---:|
| 2 traits, trait 1 (0.40) | 1.054 ± 0.020 | 1.039 ± 0.020 | 1.027 ± 0.019 | 1.013 ± 0.010 |
| 2 traits, trait 2 (0.15) | 1.212 ± 0.071 | 1.142 ± 0.062 | **1.000 ± 0.040** | 1.002 ± 0.010 |
| 3 traits, trait 1 (0.40) | 1.079 ± 0.022 | 1.058 ± 0.021 | 1.046 ± 0.020 | 1.022 ± 0.011 |
| 3 traits, trait 2 (0.23) | 1.218 ± 0.065 | 1.157 ± 0.056 | 1.062 ± 0.036 | 1.002 ± 0.011 |
| 3 traits, trait 3 (0.25) | 1.419 ± 0.181 | 1.310 ± 0.133 | 1.142 ± 0.055 | 1.014 ± 0.009 |

Paired differences Bayesian − Kackar–Harville in MSE/PEV: −0.143 ± 0.025 (2 traits,
trait 2), −0.095 ± 0.027 and −0.168 ± 0.083 (3 traits, traits 2 and 3); in realized
accuracy +0.003 to +0.004 (± 0.0013–0.0020) for the low-heritability traits, 0.000 for
trait 1. Posterior means of the genetic variances lay above the truth (2 traits:
0.445 against 0.36; REML 0.392), as expected for posterior means under flat priors.

Coverage of trait 2 in the two-trait scenario: 0.925 (plug-in), 0.932 (Kackar–Harville),
0.948 (Bayesian), 0.951 (true parameters). Realized accuracies were the same within
0.003 for all fits with estimated parameters (trait 2: 0.549 REML, 0.552 Bayesian; true
parameters 0.565).
Three traits, coverage of traits 2 and 3: 0.932 / 0.920 (Kackar–Harville), 0.942 / 0.933
(Bayesian), 0.950 / 0.948 (true parameters).

*Convergence.* With 4 chains × 4,000 iterations none of the 100 Bayesian fits met
ABP's criteria (R-hat < 1.01 and bulk/tail ESS ≥ 400 for every (co)variance; the
workflow would have extended or withheld them); the study scores them anyway. Check
(`bayes_vs_reml_study_16000it.json`, `.log`): seeds 1–6 of the two-trait scenario
again with 16,000 iterations. 4 of 6 met the criteria (worst R-hat 1.003–1.007, minimum
bulk ESS 414–937); 2 did not (R-hat 1.034 and 1.027, ESS 138 and 169). Per seed,
MSE/PEV changed by −0.011 ± 0.017 (trait 2) and −0.002 ± 0.001 (trait 1) from 4,000 to
16,000 iterations, and the posterior mean of the trait-2 genetic variance by at most
0.018: the conclusions do not depend on the short chains. The same six seeds averaged
0.91 (Bayesian) and 0.97 (Kackar–Harville) for trait 2 — a small subset; the
50-replicate means above are the comparison.

**Example 16** (`abp run`, Bayesian multi-trait linear model on the data of example 13:
2,108 animals, three traits, flat priors): converged after 16,000 iterations (4 chains,
the 4,000-iteration chains doubled twice), 937 s on a shared machine; REML (example 13)
19 s. Posterior means of G0 against the REML estimates: variances 4.36 / 0.290 / 0.450
against 3.96 / 0.263 / 0.401 (posterior means of variances lie above the REML mode),
covariances 0.315 / −0.473 / −0.037 against 0.291 / −0.460 / −0.042; residual matrices
within 3%. EBVs correlated 0.9993–0.9995 with REML BLUP. Mean PEV: wwt 2.645 (REML
2.422; with Kackar–Harville 2.516), fat 0.184 (0.170; 0.175), fec 0.301 (0.272; 0.282) —
the posterior PEVs are 5–7% larger than the first-order corrected REML PEVs, as in
the study, where the corrected PEVs were still too small.

**Example 15 with residual groups** (wwt and nlb1 in different groups, residual
covariance fixed at 0): converged after 10,000 iterations (20,000 with the covariance
estimated, round 8), 297 s; all R-hat ≤ 1.006, minimum (co)variance bulk ESS 432, worst
EBV R-hat 1.004. Posterior means (95% intervals): liability genetic variance of nlb1
0.20 (0.05–0.62), wwt genetic variance 3.57 (2.21–5.24), genetic correlation −0.07
(−0.55 to 0.45); the simulation's values (about 0.1, 4.0, +0.1) lie inside. Round 8,
with the residual covariance estimated: 0.35 (0.06–1.24), 3.94, −0.36 (−0.73 to 0.18)
and a residual correlation of 0.28 that the simulation does not contain.

**Inbreeding by pedigree depth (`inbreeding_benchmark.json`, `.log`;
`docs/benchmarks.md`).** Identical F (to 8e−14) with the round-1 kernel on five
pedigrees of 100,000–200,000 animals; 7–9 times faster where few sires per depth make
Colleau columns cheap (16.4 → 1.8 s, 116.4 → 17.4 s, 189 → 22.7 s, 233 → 32.9 s), the
same where traces are cheap (3.2 s both). Pedigree of the 200,000-animal single-step
benchmark: 4.5 s instead of 45.4 s. The automatic choice was up to 1.6 times slower
than forcing the columns on overlapping generations (conservative cost model).

### 7.14 Round-10 studies

**Example 17: repeated records with a permanent environment** (`abp run`; synthetic,
`make_data.py`; 1,208 records of 454 ewes, 1–4 lambings each, two traits). With the
round-9 moves (scale move for a categorical trait only) the chains had not met the
criteria after 16,000 iterations: R0 mixed well (bulk ESS > 6,000) but G0 and P0
traded off (bulk ESS 98–186, R-hat up to 1.035); the result was withheld (`ABP-E405`)
as designed. With the round-10 scale moves on (u_j, G0) and (pe_j, P0) for every trait:
converged after 16,000 iterations (367 s), worst R-hat 1.007, smallest bulk ESS 572
(G0[lwt] 186 → 2,022, P0[lwt] 120 → 1,532 in the first comparison run, which fitted
flock and year as main effects). After the contemporary group was corrected to
flock × year, posterior means (95% intervals) against the simulation's values:
G0 lwt 4.82 (1.93–7.90; true 4.0), bcs 0.067 (0.018–0.125; 0.06), covariance 0.41
(0.09–0.75; 0.3); P0 lwt 3.01 (0.61–6.01; 3.0), bcs 0.076 (0.025–0.129; 0.05); R0 lwt
13.48 (12.20–14.90; 13.0), bcs 0.282 (0.254–0.313; 0.29); every true value inside its
interval. The intervals for G0 and P0 are wide: 2–3 records per ewe separate genetic
and permanent-environment variance weakly. *Example definition defect, found by
comparison with the simulation:* the first version fitted flock and year as main
effects although the data contain flock × year effects; R0[lwt] was then 15.7 against
13 (the interaction went into the residual). The example now fits `flock_year`.

**F16 follow-up: weak proper priors** (`bayes_vs_reml_study_pheno_prior.json`, `.log`;
three-trait scenario, seeds 1–30, the same data as the round-9 flat-prior fits).
Priors IW(ν = t + 2) for G0 and R0 centred on 0.3 and 0.7 of the observed phenotypic
covariance (data only, not the truth). MSE/PEV (paired difference to the flat prior on
the same seeds):

| Trait (h²) | flat prior | weak prior | paired difference | REML + Kackar–Harville | true parameters |
|---|---:|---:|---:|---:|---:|
| 1 (0.40) | 1.050 ± 0.025 | 1.076 ± 0.026 | +0.025 ± 0.002 | 1.064 | 1.028 |
| 2 (0.23) | 1.101 ± 0.049 | 1.028 ± 0.030 | −0.073 ± 0.022 | 1.221 | 1.007 |
| 3 (0.25) | 1.195 ± 0.086 | 1.106 ± 0.039 | −0.088 ± 0.052 | 1.441 | 1.032 |

Realized accuracy +0.006 ± 0.001 and +0.003 ± 0.001 for traits 2 and 3. The weak prior
calibrates trait 2 and reduces the optimism of trait 3 (to 7% above the true-parameter
value on these seeds), at a small but clear cost for the well-determined trait 1 (the
prior pulls its genetic variance from 2.05 to 1.94; true 1.96). Posterior means of the
genetic variances: 1.94 / 0.629 / 0.505 (true 1.96 / 0.6 / 0.5). Chains: 4 × 4,000
iterations, none met the strict criteria (median worst R-hat 1.015), as in round 9.
These fits ran the round-9 moves (scale move for a categorical trait only; this
study has none), i.e. the same target as the released sampler. Conclusion for F16: a
weak proper prior is preferable to the flat one for low-heritability traits on small
data; neither removes the residual optimism of the least informed trait completely.

**Examples 15 and 16 with the round-10 moves** (examples log): example 15 converged
after 10,000 iterations as before (236 s), posterior means within Monte-Carlo error of
round 9 (wwt genetic variance 3.60 against 3.57, nlb1 liability variance 0.201
against 0.204); example 16 converged after 8,000 instead of 16,000 iterations (378 s
instead of 999 s), posterior genetic variances 4.30 / 0.286 / 0.441 (round 9: 4.36 /
0.290 / 0.450).

**Scale moves: invariance and mixing** (`test_scale_moves_with_permanent_environment_leave_the_posterior_unchanged`):
posterior means with and without the moves within 4 MC SE for G0, P0 and R0 entries;
mean ESS gain of the four variances > 1.3 (asserted).

### 7.15 Round-11 studies

**Example 18: maternal animal model** (`abp run`; synthetic, `make_data.py`; weaning
weight of 1,896 lambs of 454 dams, all dams known; direct, maternal genetic and
maternal permanent-environment effects): converged after 8,000 iterations (152 s standalone; 172 s in the examples log), worst
R-hat 1.008, smallest bulk ESS 411. Posterior means (95% intervals) against the
simulation: direct variance 5.07 (2.73–8.58; true 4.0), maternal 3.89 (1.96–6.19; 2.0),
direct-maternal covariance −2.36 (−4.70 to −0.60; −1.0), correlation −0.52 (−0.76 to
−0.20; −0.35), maternal permanent environment 1.13 (0.14–2.41; 1.5), residual 7.03
(5.21–8.45; 8.0). Every true value lies inside its interval; the maternal variance is
estimated high and the covariance more negative (the maternal genetic, maternal
permanent-environment and direct effects trade off with one record per lamb and about
four lambs per dam). Correlation of EBVs with the true values over all 2,108 animals:
direct 0.44, maternal 0.47 (dams: 0.54); mean √reliability: 0.52 and 0.50 (not directly
comparable: a correlation across animals against an average of individual accuracies;
the maternal reliabilities also divide by the overestimated maternal variance).

**Maternal model calibration** (`maternal_study.json`, `.log`; 20 replicates, 1,500
animals in random pedigrees, one record per non-founder, the parameters of example 18,
flat priors, 4 chains × 4,000 iterations). EBVs (all animals; MSE/PEV, coverage of
nominal 95% intervals, realized accuracy): direct 1.033 ± 0.044, 0.946, 0.577; maternal
0.984 ± 0.104, 0.949, 0.391 — the posterior PEVs of both effects are calibrated.
Posterior means of the (co)variances (mean ± MC SE over replicates, true value): direct
4.43 ± 0.25 (4.0), maternal 2.54 ± 0.19 (2.0), direct-maternal covariance −1.29 ± 0.13
(−1.0), maternal permanent environment 1.44 ± 0.13 (1.5), residual 7.71 ± 0.19 (8.0).
The maternal variance is biased upwards by 27% (2.8 MC SE) and the covariance is too
negative (2.2 MC SE): posterior means of weakly identified variances under flat priors
(as for the genetic variances of F16), consistent with example 18. Only 1 of the 20
fits met the strict R-hat/ESS criteria at 4,000 iterations (the workflow would extend
them); every fit was scored (finding F19).

### 7.16 Round-12 studies

**Dense trailing block of the sparse LDL'** (methods §21, `cholesky.py`). On the
coefficient matrix of one maternal-study replicate (1,500 animals; 3,505 equations;
317,091 factor entries; the last 700 columns carry 85% of the up-looking work): the
rule chose a dense block of 773 columns; one numeric refactorization took 61 ms with
the up-looking kernel and 23 ms with the split (native up-looking part 3 ms, LAPACK
`potrf` and the scatter 14 ms); with 3,000 animals (6,981 equations, block 1,515)
493 ms → 159 ms; with the Python kernels and 300 animals 319 ms → 40 ms. `d` and `L`
equal the up-looking factor to a relative 5e-15 / absolute 1e-15, solve residual
8e-16. Gibbs sampler of the maternal model (2 chains × 200 iterations, 1,500 animals):
35.7 s → 14.6 s. Machine: 4 cores, one BLAS thread per process; LAPACK `potrf` measured
at 16–21 GFlop/s and the up-looking kernel at about 2 GFlop/s (the ratio 8 in the
block-size rule). Not benchmarked on other machines.

**Maternal model with weak data-only priors (F19)** (`maternal_study_equal_prior.json`;
`maternal_study_flat_r12.json` re-runs the flat-prior study with the round-12 code and
records posterior medians and interval coverage; same 20 seeds, 1,500 animals, 4 chains
× 4,000 iterations). Prior: each matrix centred on an equal quarter of the observed
phenotypic variance Vp (G0 = diag(Vp/4, Vp/4), maternal pe Vp/4, residual Vp/4;
nu = dimension + 2); the true maternal variance (2.0) is below the prior guess
(about 3.6), so the prior does not pull towards the truth.

| Quantity (true) | flat: mean | flat: median | equal prior: mean | equal prior: median | paired difference (prior − flat) | RMSE flat → prior |
|---|---|---|---|---|---|---|
| direct (4.0) | 4.43 ± 0.25 | 4.38 | 4.16 ± 0.23 | 4.10 | −0.27 ± 0.03 | 1.18 → 1.01 |
| maternal (2.0) | 2.54 ± 0.20 | 2.48 | 2.37 ± 0.10 | 2.31 | −0.17 ± 0.10 | 1.01 → 0.59 |
| direct-maternal covariance (−1.0) | −1.29 ± 0.13 | −1.26 | −1.07 ± 0.10 | −1.04 | +0.21 ± 0.04 | 0.63 → 0.45 |
| maternal pe (1.5) | 1.44 ± 0.13 | 1.40 | 1.66 ± 0.06 | 1.62 | +0.22 ± 0.08 | 0.56 → 0.30 |
| residual (8.0) | 7.71 ± 0.19 | 7.70 | 7.73 ± 0.18 | 7.73 | +0.02 ± 0.02 | 0.88 → 0.81 |

Mean ± MC SE over the 20 replicates; RMSE of the posterior mean against the truth.
Fits meeting the strict R-hat/ESS criteria at 4,000 iterations: flat 1 of 20 (median
worst R-hat 1.015), equal prior 16 of 20 (1.007). 95% interval coverage of the true
value (20 replicates): direct 0.90 / 0.95, maternal 1.00 / 1.00, covariance 0.95 / 1.00
(flat / prior). EBVs (MSE/PEV, coverage, realized accuracy): flat — direct 1.033 ±
0.044, 0.946, 0.577; maternal 0.984 ± 0.104, 0.949, 0.391 (the round-11 values to three
decimals, so the new factorization did not change these results beyond Monte-Carlo
noise); equal prior — direct 1.072 ± 0.042, 0.941, 0.578; maternal 0.910 ± 0.039, 0.959,
0.397. Wall time per replicate (three replicates in parallel): flat 408 s, prior
402 s (round 11: 1,235 s with the up-looking factor while the pure-Python test suite
ran alongside).

Reading: the weak prior shrinks the spread of the variance estimates (maternal RMSE
−42%, pe −46%) and the covariance is unbiased, but the maternal variance is still 18%
high (2.37; 3.5 MC SE) and the maternal pe 11% high; posterior medians are 0.04–0.06
lower than the means, so the posterior skewness explains only a small part of the bias.
The maternal posterior PEV becomes slightly conservative (0.91) and the direct one
slightly optimistic (1.07). Posterior medians remain biased too; F19 is reduced, not
resolved.

**Example 18 with the weak prior** (`analysis_equal_prior.toml`; prior guesses from a
least-squares fit of the fixed effects, residual variance 13.9 → 3.5 each, nu = 4):
converged after 8,000 iterations (197 s, standalone run). Posterior mean (90% interval):
direct 4.71 (2.69–7.23; true 4.0), maternal 3.13 (1.87–4.67; 2.0), covariance −1.89
(−3.54 to −0.54; −1.0), maternal pe 1.61 (0.98–2.35; 1.5), residual 7.13 (5.75–8.32;
8.0); flat priors: 5.07, 3.89, −2.36, 1.13, 7.03.

### 7.17 Round-13 studies

**Maternal model by REML** (`maternal_study_reml.json`, `.log`;
`python benchmarks/maternal_study.py --method reml --replicates 200 --workers 3`; the
simulation of §7.15: 1,500 animals in random pedigrees, one record per non-founder,
direct 4.0, maternal 2.0, covariance −1.0, maternal pe 1.5, residual 8.0; seeds 1–200,
of which 1–20 are the seeds of the Bayesian studies). 192 fits converged, 3 ended with
the maternal pe variance at zero (Kuhn–Tucker check passed), 5 were refused because
the likelihood increased towards a singular G0 (`ABP-E300`). 5.5 s per replicate.

| Quantity (true) | mean ± MC SE (192 fits) | RMSE | ±1.96 SE interval coverage |
|---|---|---|---|
| direct (4.0) | 4.00 ± 0.07 | 0.94 | 0.964 |
| direct-maternal covariance (−1.0) | −1.045 ± 0.047 | 0.66 | 0.969 |
| maternal (2.0) | 2.06 ± 0.07 | 0.96 | 0.958 |
| maternal pe (1.5) | 1.47 ± 0.05 | 0.67 | 0.974 |
| residual (8.0) | 8.02 ± 0.05 | 0.71 | 0.943 |

REML estimates are unbiased within 1 MC SE and their asymptotic intervals have close to
nominal coverage. **F19 revised**: on the 19 of seeds 1–20 that converged, REML gives a
maternal variance of 2.46 — those 20 seeds were a high draw (2.46 against 2.06 ± 0.07
over 192 fits). Paired with REML on the same data, the Bayesian posterior mean exceeds
REML by +0.17 ± 0.05 (flat priors; direct +0.22 ± 0.01, covariance −0.11 ± 0.02, maternal
pe +0.08 ± 0.04) and by −0.05 with the weak data-only priors of §7.16. The bias reported
in rounds 11–12 (2.54 vs 2.0) was therefore mostly the sampling of 20 replicates; the
method-specific part is the posterior-mean excess of about 7% (flat priors).

**EBV calibration with REML estimates (finding F20).** Plug-in PEV (`C⁻¹` at the
estimates): direct MSE/PEV 1.075 ± 0.017 (median 1.009), coverage 0.940; maternal 1.56 ±
0.19 (median 1.04), coverage 0.911. The maternal mean is driven by the 24 fits that
estimated the maternal variance below 1.0 (MSE/PEV 5.19 there, 1.05 elsewhere): a
small estimate gives small PEVs and wide errors. With the Kackar–Harville correction
(`kackar_harville_delta_maternal`): direct 1.047 ± 0.016, coverage 0.944; maternal 1.35 ±
0.10 (median 1.00), coverage 0.919; 3.72 for the 24 low fits. The Bayesian posterior PEV,
which integrates over the variances, was calibrated (maternal 0.98, §7.15).

**Example 18 with REML** (`analysis_reml.toml`): converged after 11 iterations
(sparse selected inversion), 0.6 s. Estimates (SE): direct 4.44 (1.36), maternal 3.52
(1.13), covariance −1.98 (0.98), maternal pe 1.12 (0.64), residual 7.31 (0.77); h2 0.31
(0.09), m2 0.24 (0.08), r_AM −0.50 (0.15) — like the Bayesian fits of this data set
(posterior means 5.07, 3.89, −2.36, 1.13, 7.03), the data point to a high maternal
variance (true 2.0, 1.3 SE away).

### 7.18 Round-14 studies

**Two categorical traits in one threshold model** (`two_categorical_study.json`,
`.log`; `python benchmarks/two_categorical_study.py --replicates 30 --workers 3`;
800 animals in random pedigrees, on every non-founder a three-category trait, a binary
trait and a continuous trait; G0 with liability variances 0.25 and 0.16, genetic
correlations 0.5 among all three traits; independent residuals; prior IW(5, 5·diag(0.2,
0.2, 3.0)) — zero covariances; 4 chains × 3,000 iterations; every fit scored, 24 of 30
met the strict R-hat/ESS criteria, median worst R-hat 1.006; 2,560 s). Single-trait
threshold models with the same marginal priors on the same data as the reference.

| | three-category trait | binary trait |
|---|---|---|
| realized accuracy, joint model | 0.582 ± 0.010 | 0.497 ± 0.009 |
| realized accuracy, single-trait | 0.533 ± 0.011 | 0.417 ± 0.009 |
| gain (paired) | +0.049 ± 0.003 | +0.080 ± 0.006 |
| MSE/PEV, coverage (joint) | 0.945 ± 0.034, 0.954 | 0.743 ± 0.019, 0.977 |
| MSE/PEV, coverage (single-trait) | 1.10 ± 0.05, 0.936 | 0.87 ± 0.03, 0.965 |

Posterior means (true): liability variances 0.285 ± 0.013 (0.25) and 0.252 ± 0.018
(0.16); continuous 3.96 ± 0.13 (4.0); genetic covariances 0.089 (0.10), 0.440 (0.50),
0.374 (0.40); genetic correlations 0.347 ± 0.026 (0.5, between the two categorical
traits), 0.424 ± 0.023 and 0.375 ± 0.024 (0.5); 95% interval coverage of the
correlations 0.93, 0.93, 0.90. Joint modelling makes the EBVs of both categorical
traits clearly more accurate and their reliabilities calibrated or conservative, but
the binary trait's liability variance is overestimated by 58% (the prior's centre 0.2
is above the truth and one binary record per animal carries little information) and
the genetic correlations are pulled towards the prior's zero centre (finding F21; as
F18 for one categorical trait). **Example 19** (`examples/19_sheep_two_categorical`,
1,896 lambs: weaning weight, vigour score, survival): converged after 16,000
iterations (19 min with the study alongside); posterior means (90% interval) of the
liability variances 0.41 (0.26–0.61; simulated 0.30) and 0.56 (0.30–0.95; 0.20 — a
single-trait threshold fit of survival on the same data gives 0.58, so this is the
data and prior, not the joint sampler), genetic correlations 0.35 (wwt–vigour; 0.4),
0.54 (wwt–survival; 0.3), 0.24 (vigour–survival; 0.5); realized accuracies 0.58, 0.58,
0.53 against mean model accuracies 0.59, 0.60, 0.58.

### 7.19 Round-15 studies: real data (gate G5)

These are the first analyses of **real data**. Two public data sets were downloaded on
demand from the read-only GitHub mirror of CRAN, with SHA-256 checks; neither is
stored in the repository (licence inventory). The data were not sent anywhere. NCBI,
CRAN, github.com and figshare were blocked by the network policy (HTTP 403), and that
block was not worked around.

**Holstein cows, REML against an independent implementation**
(`real_milk_validation.json`, `.log`: run on the fixed code, 3,793 s, almost all of it in the dense reference; the run before the fix: `real_milk_validation_before_fix.json`; `python benchmarks/real_milk_validation.py`).
Data of example 20 (pedigreemm 0.3-5: 3,397 lactations of 1,359 cows, 57 herds,
pedigree of 6,547 animals after the sire repair; yields in lb). The reference is a
dense marginal (V-form) REML written in the benchmark from the model definition:
`A` by the tabular method, `−2 logL = log|V| + log|X′V⁻¹X| + y′Py`, maximised by
Nelder–Mead over square-root variances from two starts (half and twice ABP's estimates).
Both starts reached the same optimum in every model.

| model | records | ABP (sparse AI-REML) | V-form reference | logL ABP − reference |
|---|---|---|---|---|
| first-lactation milk; herd fixed; animal | 1,314 | animal 1,840,737, residual 11,337,848 | 1,840,736, 11,337,850 | +3.9·10⁻⁹ |
| SCS, all lactations; herd + lactation fixed; animal + pe | 3,397 | animal 0.08526, pe 0.27639, residual 1.16220 | 0.08525, 0.27639, 1.16220 | −6.3·10⁻¹⁰ |
| milk, all lactations; same repeatability model, **before the fix** | 3,397 | animal **0** (boundary), pe 5,498,779, residual 10,400,042 | 925,610, 4,638,961, 10,398,543 | **−2.363** |
| the same, **after the fix** | 3,397 | animal 925,611, pe 4,638,960, residual 10,398,543 (`converged`) | as above | −5.4·10⁻⁹ |

The third row is a **defect in ABP** (§9). The boundary check compared the score at zero
with a tolerance that did not depend on the scale of the variances. The score at zero
was +7.3·10⁻⁶ per lb², which is +76 logL units per step of one residual variance, yet
with variances near 10⁷ it passed as zero. Along the line from ABP's point to the
reference optimum the logL rises monotonically. In simulation with the trait in units
× 3,000, the old check accepted a false zero in 8 of 30 data sets. After the fix
(`kt_rejects_zero`) ABP rejects the boundary, reinstates the term and converges to the
reference optimum (h² = 0.058). Before this round the examples' and studies' traits
had variances of order 1–100, where the old tolerance was tight enough. Their fits are
unchanged by the fix (round-15 test suites, §3).

The sparse-trace rule (methods §35) reduced the SCS example from 196 s to 27 s with the
same logL to 10⁻¹⁰. The three-trait first-lactation run (example 20,
`analysis_first_lactation_multitrait.toml`, 27 s) gives h² 0.13 (milk), 0.29 (fat),
0.11 (protein), genetic correlations 0.62, 0.73, 0.53 and residual correlations 0.72,
0.93, 0.78.

**Holstein cows, predictive ability over cows.** The 1,359 cows were split into 5 folds
at random (seed 2026). In each fold all records of the fold's cows are hidden, and the
animal model (herd fixed, REML) is fitted to the first lactations of the other cows.
The hidden cow's EBV therefore comes from relatives only. The target is her mean yield
over all her lactations, adjusted for herd and lactation by least squares on all
records. Bootstrap SE over cows (2,000 resamples).

| trait | r(EBV, target) ± SE | regression of target on EBV ± SE | h² by fold |
|---|---|---|---|
| milk | 0.087 ± 0.026 | 0.76 ± 0.23 | 0.06–0.17 |
| fat | 0.154 ± 0.026 | 0.72 ± 0.12 | 0.23–0.35 |
| protein | 0.074 ± 0.026 | 0.82 ± 0.28 | 0.04–0.12 |

The predictive abilities are small but clearly positive (3–6 SE). That is expected when
a cow is predicted only from relatives' first lactations: most sires have few
daughters, and the target carries her permanent-environment and residual effects. The
regression coefficients are below 1, so the EBVs are somewhat over-dispersed. The
difference is significant for fat (2.3 SE) and within 1–1.1 SE of 1 for milk and
protein. Fat's fold heritabilities (0.23–0.35) are the highest, and its slope is the
furthest below 1.

**Mice: GBLUP against pedigree BLUP** (`real_mice_validation.json`, `.log`;
`python benchmarks/real_mice_validation.py`; 327 s). Data of example 21 (1,814 mice,
10,346 SNPs, the pedigree relationship matrix shipped with the data). Models: sex and
test year × season fixed, cage (523 levels) independent random, additive effect with
`G + 0.01 I` (VanRaden, frequencies of all mice; mean diag 1.037), with the pedigree
`A`, or absent; REML.

| | body weight: GBLUP | PBLUP | no additive | body length: GBLUP | PBLUP | no additive |
|---|---|---|---|---|---|---|
| REML logL | −2527.7 | −2572.7 | −2616.3 | 393.75 | 392.36 | 373.44 |
| h² (cage share) | 0.27 (0.28) | 0.51 (0.18) | — (0.41) | 0.16 (0.24) | 0.28 (0.18) | — (0.31) |
| 5-fold CV: r(EBV, target) ± SE | 0.430 ± 0.020 | 0.436 ± 0.019 | | 0.271 ± 0.022 | 0.311 ± 0.021 | |
| regression of target on EBV | 1.08 | 1.12 | | 1.22 | 1.17 | |
| across-family CV: r ± SE | 0.275 ± 0.022 | not defined (EBVs 0) | | 0.109 ± 0.024 | not defined | |
| regression of target on EBV | 0.66 | | | 0.52 | | |

Folds at random over mice (seed 2026); a hidden mouse's EBV comes from relatives only;
the target is its phenotype minus the training fixed-effect solution (it still contains
the cage effect, which is shared by relatives because littermates share cages). GBLUP −
PBLUP: −0.006 ± 0.017 (body weight), −0.040 ± 0.020 (body length; bootstrap over mice,
2,000 resamples). For body weight G fits the data much better than A (logL +45 with the
same number of parameters), but the predictive ability under random folds is the same. A
random split leaves full sibs in the training set, so the pedigree already captures most
of the family information; the higher pedigree h² (0.51) probably absorbs common
litter and cage environment, which the target also contains, and favours PBLUP in this
design. The pedigree matrix of these data
links only full sibs (169 families of up to 48 mice; no relationship between families),
so when whole families are hidden (folds by family, seed 2028) pedigree EBVs of the
hidden mice are exactly 0, while GBLUP still predicts body weight with r = 0.27 from
marker similarity across families (the situation Legarra et al. 2008 studied with these data).
Across families the regression of the target on the GBLUP EBV is 0.66 and 0.52:
these EBVs are over-dispersed for mice without close relatives in the training data
(G + 0.01 I and G's frequency base are a likely cause; not investigated further). With
random folds the regression coefficients above 1 come with the cage effect in the
target and are not a calibration test. **No claim that one model is better in general
is made from these numbers.**

**What G5 now covers, and what it does not.** On two real data sets ABP's REML equals an
independent implementation to 10⁻⁸ logL (after the fix that this comparison prompted).
Cross-validated predictive abilities are positive. GBLUP predicts across families where
the pedigree cannot. These are retrospective checks on public data. They are not a
forward-in-time validation (the data carry no dates for an LR split), not
a comparison with other software, and not a test on the populations ABP is meant for.

### 7.20 Round-16 studies: established software, forward-in-time validation, F22

**Comparison with established software on the real data.** R 4.3.3 and lme4 1.1-35.1
(Ubuntu packages) were installed, and pedigreemm 0.3-5, rrBLUP 4.6.3 and sommer 4.3.6
were built from their CRAN sources. The sources came from the read-only GitHub mirror of
CRAN; CRAN itself is blocked by the network policy. These packages are test oracles
only: they are not ABP dependencies and are not redistributed (licence inventory).

*Holstein data, pedigreemm* (`real_milk_pedigreemm.json`, `.log`;
`benchmarks/real_milk_pedigreemm_comparison.py`, R side `benchmarks/r/pedigreemm_fit.R`).
pedigreemm (Vazquez et al. 2010) fits animal models through lme4, using a Cholesky
factor of A; it was published with these data. The same three models as §7.19 were
fitted by REML:

| model | relative difference of variances (ABP − pedigreemm) | logL difference¹ | EBV correlation, max \|diff\| / SD² |
|---|---|---|---|
| first-lactation milk (1,314 cows) | animal 1.4·10⁻⁵, residual −1.9·10⁻⁶ | 2.2·10⁻⁹ | 1.00000000, 3.8·10⁻⁵ |
| SCS repeatability (3,397 records) | animal −4.5·10⁻⁵, pe 1.9·10⁻⁵, residual −1.1·10⁻⁶ | 4.0·10⁻⁹ | 1.00000000, 1.4·10⁻⁴ |
| milk repeatability (3,397 records) | animal 5.5·10⁻⁵, pe −5.5·10⁻⁶, residual −8.7·10⁻⁷ | 1.1·10⁻⁸ | 1.00000000, 1.3·10⁻⁴ |

¹ lme4's REML criterion includes the constant `(n − rank X) log(2π)`; after removing it
the log-likelihoods agree. ² For the 1,314 or 1,359 cows with records. The remaining EBV
differences follow from the 10⁻⁵ differences in the variances. ABP's EBVs also equal a
dense V-form BLUP (`σ²_a A Z′Py`) to 1.6·10⁻⁸.

pedigreemm also puts the milk repeatability model at an interior optimum (additive
925,561). This independently confirms the round-15 fix of ABP's boundary check.

**Observation about pedigreemm 0.3-5.** Its `ranef()` method returns `relfac %*% b`. The
model is fitted with `Z* = Z relfac′` and `A = relfac′ relfac` (`getA` is
`crossprod(relfactor)`), so the BLUP of the additive effect is `t(relfac) %*% b`. On these
data `t(relfac) %*% b` equals ABP's EBVs and the V-form BLUP. The values returned by
`ranef()` correlate only 0.71–0.75 with them, because `relfac` is upper triangular, not
symmetric. The comparison above therefore uses `t(relfac) %*% b`, and both versions are
written to the JSON. We read this as a defect in that version's `ranef()` method; it does
not affect pedigreemm's variance estimates. It was not reported upstream from this
environment.

*Mouse data, rrBLUP and sommer* (`real_mice_software.json`, `.log`;
`benchmarks/real_mice_software_comparison.py`, R side `benchmarks/r/mice_gblup_fit.R`).
The genomic relationship matrix was computed by `rrBLUP::A.mat` (VanRaden method 1),
independently of ABP's code, and equals ABP's `vanraden_g` to 1.4·10⁻¹⁴. Results:

| model | package | relative difference of variances | GEBV correlation, max \|diff\| / SD |
|---|---|---|---|
| body weight: sex + year×season, animal (G + 0.01 I) | `rrBLUP::mixed.solve` | ≤ 9.9·10⁻⁷ | 1.0000000000, 1.6·10⁻⁶ |
| body weight: + cage (example 21) | `sommer::mmer` | ≤ 3.2·10⁻⁶ | 1.0000000000, 4.6·10⁻⁶ |
| body length: animal | `rrBLUP::mixed.solve` | ≤ 6.5·10⁻⁷ | 1.0000000000, 9.7·10⁻⁷ |
| body length: + cage | `sommer::mmer` | ≤ 3.2·10⁻⁵ | 1.0000000000, 5.7·10⁻⁵ |

rrBLUP's log-likelihood differs from ABP's by the same constant for both traits
(1637.21075, equal to 10⁻⁸). The two likelihood functions are therefore identical up to a
data-independent constant. sommer's reported log-likelihood uses its own constants and
was not compared.

These are agreement checks on four models, not a performance comparison. Run times were
measured under different loads and are not reported as benchmarks. **No claim of
superiority is made.**

**Forward-in-time validation on real data** (`real_mice_lr_validation.json`, `.log`;
`abp run examples/21_mice_bodyweight_real/analysis_bw_lr.toml`, 222 s). This is the LR
method with GBLUP for body weight. The 677 mice tested in 2004 were hidden: partial
evaluation with cutoff 2003-12-31, variances by REML on the 1,137 mice tested in 2003.
`test_date` rests on an assumption the source does not state: a year's "winter" is its
first quarter (example README). Results (bootstrap over mice, 1,000 replicates):

| statistic | estimate ± SE (95% interval) | expected under a correct model |
|---|---|---|
| bias Δ̂_p | −0.010 ± 0.026 g (−0.059, 0.041) | 0 |
| dispersion b̂_w,p | 0.899 ± 0.021 (0.858, 0.942) | 1 |
| consistency ρ̂_w,p | 0.869 ± 0.009 | √(rel_p / rel_w) = 0.848 (model-based, focal mice) |

The partial EBVs show no bias but are over-dispersed by about 10%. The dispersion differs
from 1 by about 5 SE. The partial-data REML gave a larger additive variance (2.91 against
2.13 on all data). This forward-in-time result agrees in direction with the cross-family
regressions of §7.19 (F22).

**F22 factor study** (`f22_dispersion_study.json`, `.log`;
`python benchmarks/f22_dispersion_study.py`, 1,671 s). Same across-family folds as §7.19.
Each variant changes one factor; the table gives the regression of the target on the
EBV ± SE.

| variant | body weight | body length | full-data REML (body weight) |
|---|---|---|---|
| baseline | 0.658 ± 0.056 | 0.518 ± 0.114 | animal 2.13, cage 2.17, residual 3.50; logL −2527.74 |
| + full-sib family effect | 0.731 ± 0.061 | 0.560 ± 0.151 | animal 1.99, cage 1.61, family 1.02, residual 3.48; logL −2516.94 |
| variances fixed at full-data REML | 0.690 ± 0.057 | 0.571 ± 0.116 | — |
| allele frequencies 0.5 | 0.647 ± 0.052 | 0.490 ± 0.108 | same logL as baseline |
| G + 0.05 I | 0.658 ± 0.056 | 0.518 ± 0.114 | same logL |
| family effect, half of the SNPs | 0.732 ± 0.060 | 0.561 ± 0.152 | logL −2514.40 |
| family effect, quarter of the SNPs | 0.785 ± 0.064 | 0.507 ± 0.208 | logL −2520.78 |

What the study shows:

* **Common family environment is part of the cause.** The family effect is strongly
  supported: likelihood-ratio statistic 21.6 for body weight and 14.8 for body length. It
  lowers the additive variance and raises the slope by about 0.07 and 0.04.
* **Variance estimation per fold and the G frequency base or ridge are not the cause.**
  Fixing the variances adds about 0.03–0.05. With frequencies of 0.5, G differs from the
  baseline G only by a change of base, which the fixed effects absorb, so the logL is
  identical. The ridge changes nothing.
* **The marker-linkage hypothesis was not supported.** If linkage between markers and
  causal loci across families were the main cause, fewer markers should have lowered the
  slope. For body weight it rose (0.73 → 0.79), and for body length the change is within
  its SE.
* **F22 remains open.** It is partly explained, and the slope stays below 1 (0.73 for
  body weight). Anything full sibs share equally, including their share of dominance
  variance, is already absorbed by the family effect, so it cannot explain the rest.
  Remaining candidates:
  * the genomic covariance assumed between distantly related mice (σ²_a G_ij) may be
    larger than the realised one, while the variance itself is estimated mainly from
    close relatives;
  * genetic variance that differs between families or generations;
  * sampling, with only 169 families.

  None was tested here.

**Several iid terms in the multi-trait Gibbs samplers** (methods §36; tests below). This
is a feature, not a study. With known covariances, the posterior means of u and of both
terms' effects, and the PEV blocks of u, equal the dense mixed-model solution from the
model definition within Monte-Carlo error (`test_two_iid_terms_with_fixed_covariances_equal_dense_mme`).
From one simulated data set with a permanent environment and a litter effect, both
covariance matrices, G0 and R0 are recovered within 3.5 posterior SD
(`test_two_iid_terms_recover_their_covariances`). A workflow test runs two terms
end to end.

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
| F10 | Multi-trait REML does not handle boundary optima | **addressed** (rounds 5–6; opt-in) | default still stops with `ABP-E300`; `reml.boundary = "reduced_rank"` or `reml.rank` fits G0 = ΛΛ′; 100-replicate study: 100/100 converged, unbiased, EBVs close to calibrated (§7.10). Singular R0 still stops |
| F11 | Laplace-approximate REML of liability variances is biased with 2–3 categorical records per animal (genetic variance −23%, 20% of runs at the search bound) | **resolved by an alternative** (round 6) | Gibbs sampler: genetic variance 0.108 ± 0.009 (true 0.111), 0/30 withheld (§7.10); the Laplace estimator stays available and documented as biased. Remaining: pe variance +24% with uniform priors (F13) |
| F12 | REML-based PEV was 7–13% optimistic in every study | **resolved** (rounds 5–7) | Kackar–Harville PEV: pedigree REML MSE/PEV 1.017 ± 0.031; multi-trait 1.025/1.048/1.058 (§7.10); reduced-rank fits (round 7, §7.11) |
| F13 | Threshold Gibbs sampler with uniform variance priors: permanent-environment variance 0.138 ± 0.010 (true 0.111) | open (documented; round 7 study) | proper priors are in the spec (`bayes.variance_prior`); with prior scales 0.05 / 0.2 the estimates follow the prior (§7.11): with 2–3 records per animal these variances are weakly identified, and no prior choice removes that. More records per animal or known variances are the remedies |
| F14 | Sampled reliabilities carry Monte-Carlo error (SE ≈ 0.04 with 300 simulations; 0.15 with 20 at 200,000 animals) | open (by design) | reported per animal (`reliability_mc_se`); choose `solver.pev_samples` for the precision needed (0.21 with 10 at 120,000 animals, §7.11; 0.10 with the round-8 estimator, §7.12) |
| F16 | With full-rank multi-trait REML on 1,000 animals, the EBVs of low-heritability traits (h² 0.15–0.25, 30% missing) are optimistic even with the Kackar–Harville PEV (MSE/PEV 1.14–1.17, 3–3.5 MC SE above 1; round 9: 1.14–1.31) | **partly resolved** (rounds 9–10) | Bayesian multi-trait linear model: calibrated with two traits (1.000 ± 0.040); with three traits 1.06 / 1.14 under flat priors and 1.03 / 1.11 under weak data-centred IW priors (round 10, §7.14) against 1.22 / 1.44 with Kackar–Harville on the same seeds; the least informed trait stays 7% above the true-parameter benchmark; a second-order Kackar–Harville correction remains a candidate |
| F17 | With one categorical record per animal, a flat prior on the liability genetic variance gives an improper posterior (single- and multi-trait threshold samplers: chains drift, results withheld) | open (documented; remedy available) | proper priors (`variance_prior = "scaled_inv_chi2"` / `"inverse_wishart"`) make the posterior proper; their influence must be reported (F13) |
| F18 | Multi-trait threshold model with one categorical record per animal: the genetic correlation follows the prior's centre (0.27 ± 0.04 with a prior at zero covariance, 0.51 ± 0.04 with one at the true 0.5) | open (documented) | the data identify the covariance weakly; take `prior_covariance` from published estimates and report it; repeated or more records improve identification |
| F15 | Rank selection by the smallest AIC wrongly reduced a full-rank G0 (r_G 0.6) to rank 1 in 19 of 100 data sets, making trait-2 PEV about four times too small | **resolved** (round 7) | ABP moves to a lower rank only if its AIC is smaller by ≥ 2: 0 of 100 wrong reductions; the cost is that a true rank 1 is found in 58/100 instead of 97/100 (the full-rank model is kept in the rest) (§7.11) |
| F19 | Bayesian maternal animal model (one record per animal, flat priors): the posterior mean of the maternal variance looked biased upwards (2.54 ± 0.19 vs 2.0 over 20 replicates; example 18: 3.9) while the EBV posterior PEVs were calibrated (direct 1.03, maternal 0.98) | revised (round 13) | REML on the same seeds gives 2.46 and is unbiased over 200 replicates (2.06 ± 0.07, §7.17): the 20 seeds were a high draw; the method-specific part is the posterior-mean excess over REML, +0.17 ± 0.05 (≈7%) with flat priors and −0.05 with weak data-only priors (§7.16). Use weak priors or REML for point estimates; report intervals |
| F20 | Maternal animal model by REML: the plug-in maternal PEV is optimistic when the maternal variance is underestimated (MSE/PEV 1.56 ± 0.19 over 192 fits, median 1.04; 5.19 in the 24 fits with an estimate below 1.0) | open (round 13) | Kackar–Harville columns reduce it (1.35, median 1.00) but not for the low estimates (3.72); the Bayesian posterior PEV is calibrated (0.98); prefer the Bayesian model, or read maternal reliabilities from the Kackar–Harville columns, when the maternal variance is poorly determined |
| F21 | Several categorical traits (one record per animal and trait): the binary trait's liability variance is overestimated (0.25 vs 0.16, prior centred at 0.2) and genetic correlations are pulled towards the prior's centre (0.35 vs 0.5 with a prior at zero covariance); residual covariances between categorical traits are fixed at 0 by the model | open (round 14) | joint modelling still improves the categorical EBVs (+0.05, +0.08 accuracy) with calibrated or conservative reliabilities (§7.18); take prior variances and covariances from published estimates and report them; repeated records or progeny-tested sires identify the covariances better |
| F22 | Real data (mice, GBLUP): EBVs of animals without close relatives in the training data are over-dispersed (across families: regression 0.66 body weight, 0.52 body length; forward in time, mice tested in 2004: dispersion 0.90 ± 0.02); Holstein EBVs from relatives only: 0.72–0.82 | **partly explained** (round 16) | factor study (§7.20): a full-sib family (common environment) effect is strongly supported (LR statistics 21.6, 14.8) and raises the slope to 0.73 / 0.56; fold-wise variance estimation adds 0.03–0.05; the G frequency base and ridge do not matter; fewer markers did not lower the slope (linkage hypothesis not supported). Fit a common-environment (litter/family) term where full sibs share an environment (REML: any number of iid terms; Gibbs: round 16); remaining candidates: genomic covariance between distant relatives larger than realised, heterogeneous variance, sampling (169 families) |
| E1 | Engineering: with every core busy, multi-threaded OpenBLAS made the 12 s API example's multi-trait REML exceed 600 s (thread oversubscription); with `OPENBLAS_NUM_THREADS=1` it took 20 s under the same load (round 9) | open (documented) | set `OPENBLAS_NUM_THREADS` (or the BLAS thread count) when ABP shares a machine; the benchmarks and studies already set one thread per worker |

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
| 6 | `SparseLDL` and the SuperLU + selected-inversion path failed on matrices symmetric only to rounding with an asymmetric pattern (a 1e-17 entry on one side after cancellation) | reduced-rank analytic-gradient fit with rank 2 (native); local pure-Python-kernel run (SuperLU; CI's rounding did not trigger it) | matrix symmetrised before factorization in both | `test_numerically_symmetric_matrix_with_asymmetric_pattern`, `test_superlu_selected_inverse_with_asymmetric_pattern`, self-test T16 |
| 6 | 11 pushes failed CI (all 7 jobs) because the published JSON schema was not regenerated after new spec keys | CI (noticed late: the round-6 pushes were not checked until the release) | schema regenerated (`python -m abp.core.spec`); CI green from run 36652918117 | `test_published_json_schema_is_current` |
| 6 | Reduced-rank REML stopped with `ABP-E403` in 2 of 100 study fits although at the optimum (gradient bound 1e-3 unattainable at the precision of −2 log L ≈ 3600) | calibration study | Newton-decrement criterion (≤ 1e-6) | study rerun: 100/100 converged |
| 7 | The manifest lost the sampled-PEV record (`diagnostics.<trait>.pev_sampling`: samples, seed, mean Monte-Carlo SE): a later step replaced the trait's diagnostics dictionary instead of updating it. Results and the `reliability_mc_se` column were unaffected | the workflow-level 120,000-animal benchmark read `null` from the manifest | the dictionary is updated in place | `test_sampled_reliabilities_agree_with_exact_ones` (manifest assertion; fails before the fix) |
| 7 | Rank selection by the smallest AIC reduced a full-rank G0 in 19% of data sets (a method-choice defect, F15) | rank study with a full-rank truth | conservative margin of 2 | `test_rank_selection_margin_rule` |
| 8 | `SparseLDL.refactor` refused a matrix whose pattern lost exact zeros (a diagonal start value of R0 or G0 makes entries of the multi-trait equations exactly 0) | first runs of the multi-trait threshold sampler | values placed on the stored pattern (`_on_pattern`); the sampler builds its factor on the full pattern | `test_coefficient_maps_equal_direct_assembly` |
| 8 | The sampled-reliability estimator carried a zero-mean cross term (noise) and an O(1/N) ratio bias (−0.009 with 40 simulations) | analysis of the estimator; `pev_estimator_study.py` | orthogonal estimator with the bias removed | `test_orthogonal_reliability_estimator_is_unbiased_and_less_noisy` |
| 9 | With `residual_groups`, the R0 prior text in `mcmc_diagnostics_multitrait.json` described the Korsgaard parameterisation of the whole matrix although the categorical trait was alone in its group (its residual variance simply fixed at 1); the sampler itself was right | reading the diagnostics of the rerun of example 15 | prior text built per block | `test_workflow_bayesian_multitrait_linear_model` (block text) |
| 9 | Under a CPU fully used by other processes, the API example's multi-trait REML (multi-threaded OpenBLAS dense inverse) exceeded its 600 s test timeout although it takes 12–20 s alone (thread oversubscription; no code change involved) | local test run during the 3-worker study | not a code defect: tests rerun without the competing load; with `OPENBLAS_NUM_THREADS=1` the script took 20 s under the same load; recorded as an engineering risk (§8) | `test_api_example_script_runs` (unchanged) |
| 9 | The new depth-kernel test asserted the native kernel name even with `ABP_DISABLE_NATIVE=1` (the module is importable, so the test was not skipped; `Pedigree.inbreeding` correctly used the Python reference): CI run 36807935977 failed on the pure-Python job, 6 of 7 jobs green | CI (checked after the push) | the expected name follows `native_kernel_available()` (test defect; production code unaffected) | `test_native_depth_kernel_matches_reference_on_every_path` (both kernel settings) |
| 10 | The sparsity pattern of the multi-trait Gibbs coefficient matrix came from a numerical surrogate (`R0 = G0 = I + 0.5`); with repeated records an entry of W′R⁻¹W (−0.75) cancelled the prior entry (+0.75) exactly, so the pattern lacked entries the actual C needs (`ValueError: entry outside the pattern of C`) | the new exact test with repeated records (before release) | pattern from absolute values, SPD diagonally dominant values for the initial factor | `test_permanent_environment_with_fixed_covariances_equals_dense_mme` |
| 15 | REML Kuhn–Tucker check (single-trait and maternal): the score at zero (units 1/variance) was compared with `1e-6 |logL|` unscaled, so with variances near 10⁷ a positive score passed as zero; a zero additive variance was accepted 2.36 logL units below the interior optimum (real milk yields in lb; in simulation with y × 3,000: 8 of 30 data sets) | real-data comparison with an independent V-form REML (§7.19) | the score is scaled by the residual variance (`kt_rejects_zero`) | `test_boundary_decision_is_invariant_to_the_unit` (fails with the old rule) |
| 15 | Single-trait REML on sparse pedigree structures used the dense inverse up to 12,000 equations, in the score at zero and in the Kackar–Harville solves (196 s for 8,000 equations) | profiling on real data | sparse traces above 2,000 equations when every structure is sparse; record columns only for the score at zero (methods §35) | `test_score_at_zero_equals_v_form_derivative` |
| 15 | Test helper `_simulate` (REML tests) drew permanent-environment and group effects in `set()` order, which depends on Python's per-process string-hash seed, so these tests used different data in every run (the new unit-invariance test failed on 3 of 14 CI jobs) | CI failure of the new test | effects drawn in sorted order; data now identical under any `PYTHONHASHSEED` (checked with two seeds over 30 data sets) | `test_boundary_decision_is_invariant_to_the_unit` (seed 3; fails with the old Kuhn–Tucker rule at the fit level) |

Round-8 errors in the construction of tests and scripts: the location-draw test first
required the pooled SD of the covariance z-scores to be at least 0.85 (the entries are
correlated, so the pooled SD can legitimately be lower; the bound on max |z| is the
test); the first invariance test of the scale move used one binary record per animal
with a flat prior, whose posterior is improper (F17; now a proper prior); the linear
coefficient maps first included stored zeros of `K⁻¹` (caught by the tests before
commit); a shell command meant to stop leftover wait loops also stopped itself.

Round-7 errors in the construction of tests and scripts: a new unit test of
the margin rule had wrong expected AIC values (fixed before commit); the
full-rank rank study first wrote the label "true G0 of rank 1" into its JSON
(label fixed in the script and the study re-run; numbers identical).

Round-6 errors in the construction of tests, studies and scripts
(production code unaffected): a sampled-reliability test first required a
mean |error| below 0.03 (a precision wish, not a correctness criterion) and
then a mean z-score below 0.1 (the z-scores share the same simulations and
each SE comes from the same samples as its estimate; the mean is positive by
0.02–0.13 across seeds while the mean error changes sign) — replaced by the
spread of z and a bias bound on the reliability scale; a seed-comparison
script ran six evaluations in one process and the container ran out of
memory — rerun with one evaluation per process; the benchmarks page first
gave example 14 as 3,338 equations (it has 2,571 location equations) —
corrected before release.

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
| Real-data validation (G5), remaining parts | rounds 15–16 ran retrospective, cross-validated, forward-in-time (mice) and software-comparison checks on two public data sets (§7.19–7.20); validation on the populations ABP is meant for needs their data |
| LR population-accuracy estimator | its published form has a 2019 erratum that could not be verified here (the source was not reachable) |
| Posterior SBC near the target data | the prior SBC (§7.3) checks the computation over the prior; a posterior SBC is a further step |
| Comparison with BLUPF90, MiXBLUP, ASReml, DMU, JWAS | not installed or licensed in this environment (round 16 compared with the open R packages pedigreemm, rrBLUP and sommer instead, §7.20) |
| Independent mature simulator (AlphaSimR, QMSim, XSim) | not installed; ABP's generators are independent of its solver code but are not mature external simulators |
| Workflow-level run at 200,000 animals | the workflow runs used 120,000 animals (30,000 genotyped); the 200,000-animal runs are library-level |
| Residual covariances between two categorical traits in the multi-trait threshold model | not implemented (several categorical traits: round 14; several iid terms: round 16) |
| Multi-trait maternal REML; a correction of the plug-in maternal PEV under REML (F20) | not implemented (single-trait maternal REML: round 13) |
| Second-order Kackar–Harville correction (F16) | not implemented |
| Installation from a built wheel or installer | no binary packaging yet (source install only) |
