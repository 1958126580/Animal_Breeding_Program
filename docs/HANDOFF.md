# Handoff (read this first in the next session)

State as of 2026-10-03, ABP 0.15.0, branch `claude/ecstatic-archimedes-qs0idf`
(round 2 lives on `claude/festive-newton-2elqfq`; round 3 continues from it).
Trust files and tests, not this summary: re-run `python -m pytest -q` and
`abp selftest` before continuing.

**Lesson from the interrupted round-3 attempt:** work that is not committed
and pushed is lost when a session ends. Commit and push after every piece
that passes its tests.

## 1. What is implemented

Phases S0–S1 and parts of S2–S3. Every item passed its tests on Linux and in
CI on Windows and Linux (details in `docs/validation_report.md`, status per
method in `docs/method_registry.toml`):

* data contracts (TOML spec + JSON Schema, data dictionary, manifest schema,
  error codes); QC for pedigree, phenotypes and genotypes; PLINK 1 binary
  input;
* pedigree A/A⁻¹/F (C++20 kernel optional), Colleau products; unknown-parent
  groups (random with a declared ratio, or estimable fixed groups);
* **metafounders** (round 3): `A^Γ = T D T' + QΓQ'`, sparse inverse,
  log-determinant, generalised Meuwissen–Luo kernel (Python + C++), Γ from a
  documented file or estimated from genotypes (GLS base allele frequencies,
  sampling correction), single step on the metafounder base (G05, no tuning),
  EBVs against a reference metafounder with their own PEV/reliability
  (`ebv_vs_base`), example 12;
* **round 4**: multi-trait REML (AI + EM, missing traits, boundary diagnosis);
  threshold (probit) model for categorical traits; ABP's sparse LDL' with
  minimum-degree ordering (default sparse factor; SuperLU kept as
  reference); APY; metafounders for multi-trait models and LR validation;
* **round 5**: PEV/reliability including REML uncertainty (Kackar–Harville,
  single-trait); Laplace-approximate REML for threshold-model liability
  variances (biased in sparse data, F11); reduced-rank G0 = ΛΛ′ for
  multi-trait REML at the boundary (`reml.boundary`, `reml.rank`);
  matrix-free single step (`genomic.single_step_mode = "matrix_free"`,
  `abp/core/ssop.py`); self-test T15; F9 resolved; APY calibration study;
* **round 6**: Gibbs sampler for threshold-model liability variances
  (`bayes.method = "threshold"`, unbiased for the genetic variance, F11);
  Kackar–Harville PEV for multi-trait REML (F12 resolved); sampled PEV and
  reliabilities for the matrix-free single step (`solver.pev = "sampled"`);
  analytic gradient and Newton-decrement convergence for reduced-rank REML
  plus a 100-replicate study; int8 APY blocks and a 200,000-animal run;
  fix of the sparse LDL' for asymmetric patterns; self-test T16;
* **round 7**: proper variance priors for the threshold Gibbs sampler in the
  spec (`bayes.variance_prior`, `bayes.prior_variances`) with a sensitivity
  study (F13: the data identify these variances weakly); Kackar–Harville PEV
  for reduced-rank fits; rank selection (`reml.rank_selection = "aic"`,
  conservative margin of 2, F15); int8 genotype storage in the workflow
  (`genomic.genotype_storage`, blockwise G and frequencies); workflow-level
  run with 120,000 animals / 30,000 genotyped (261 s, 7.38 GB); manifest fix
  for the sampled-PEV record; self-test T17;
* **round 8**: multi-trait threshold model (one categorical + continuous traits,
  Gibbs sampler with scale and shear moves, IW prior, example 15); variance-reduced
  sampled reliabilities (about 3x fewer simulations); rank-selection study scoring
  the EBVs of the chosen model (2 and 3 traits, F16); native Colleau product and
  direct coefficient maps (speed); fewer genotype copies; self-test T18;
* **round 9**: Bayesian multi-trait linear model (`bayes.method = "multitrait"`:
  G0 and R0 sampled with the EBVs, posterior PEV; example 16); residual
  covariances fixed at exactly 0 between groups of traits (`bayes.residual_groups`)
  in both multi-trait Gibbs samplers (example 15 now uses it: converges in half the
  iterations); native inbreeding kernel by pedigree depth (Meuwissen–Luo traces or
  Colleau columns per depth, identical F; 200,000 animals 4.5 s instead of 45 s);
  F16 study (Bayesian posterior PEV vs REML + Kackar–Harville);
* **round 10**: permanent-environment (iid) term in both multi-trait Gibbs samplers
  (repeated records; P0 sampled; `pe_multitrait.csv`, `c2`; example 17); proper
  inverse-Wishart priors for P0 and R0 (`prior_covariance.pe`, `.residual`); scale
  moves for every trait on (u, G0) and (pe, P0) (example 17: converged instead of
  withheld); F16 follow-up with weak data-centred priors;
* **round 11**: maternal genetic effects (maternal animal model) in both multi-trait
  Gibbs samplers: `kind = "maternal"` (dam from the pedigree), 2t × 2t G0 with
  direct-maternal covariances, maternal EBVs and reliabilities, m2; one trait allowed;
  maternal permanent environment as an iid term on a dam column; example 18;
* **round 12**: sparse LDL' with a dense trailing block factorized by LAPACK (Schur
  complement from the up-looking kernel; C++ `ldl_numeric_split` + Python reference;
  maternal model 2.6–3.1 times faster per factorization); posterior medians of G0, R0,
  P0; F19 prior study (weak data-only priors) and example 18 variant
  `analysis_equal_prior.toml`;
* **round 15**: first real-data validation (G5 partial): examples 20 (Holstein
  lactations, pedigreemm) and 21 (genotyped mice, BGLR), data fetched on demand
  (`fetch_data.py`, SHA-256, GPL data not stored; needs `pip install rdata`);
  `benchmarks/real_milk_validation.py` (ABP REML vs independent V-form REML; 5-fold CV
  over cows) and `benchmarks/real_mice_validation.py` (GBLUP vs PBLUP, random and
  across-family CV); **defect fixed**: the REML Kuhn–Tucker check was not scale-invariant
  and accepted a false zero variance for large-variance traits (`kt_rejects_zero`, single
  trait and maternal; [results]); sparse REML traces above 2,000 equations for sparse
  structures (SCS example 196 s → 27 s);
* **round 14**: several categorical traits in the multi-trait threshold model (each in
  its own residual group; residual covariances between them 0); example 19;
  30-replicate study (EBV accuracy +0.05/+0.08 over single-trait models; F21);
* **round 13**: maternal animal model by REML or known variances (one trait;
  `abp.solvers.maternal_reml`, `workflows/maternal.py`; boundary of independent terms,
  singular G0 refused; Kackar–Harville PEV columns); example 18 `analysis_reml.toml`;
  200-replicate REML study: F19 revised (the 20 Bayesian seeds were a high draw;
  posterior mean ~7% above REML), new F20 (plug-in maternal PEV optimistic);
* single-trait BLUP (animal, repeatability), dense/sparse/PCG solvers, PEV,
  reliability; **exact PEV at any size whose factor fits in memory by sparse
  selected inversion** (round 3; Takahashi equations on the symbolic Cholesky
  pattern, Python + C++); REML (AI + EM, boundary, SE, checkpoint and
  resume), with sparse selected-inversion traces above 12,000 equations;
* GBLUP (VanRaden G, explicit policies), single-step H⁻¹;
* Bayesian marker models (BRR, BayesA/B/C/Cπ/R) with convergence gating,
  traces, predictive checks; MCMC diagnostics equal to ArviZ;
* multi-trait BLUP with known covariances; Smith-Hazel and restricted
  indices; EBV index;
* forward-in-time LR validation;
* optimal contribution selection and mating plans (`abp mate`, proposals
  only);
* workflow with atomic outputs, manifests and reports; CLI; launchers;
  synthetic sheep generator; 18 examples; documentation; self-test T01–T18.

## 2. Commands that were run (Linux) and their results

Round 15: full test suites (native 365 passed; pure-Python kernels 359 passed, 6
native-only skipped; `tests/test_reml.py` re-run on the final code with both kernels),
self-test with both kernels, all 27 example runs (real-data examples with locally fetched
data), the real-data benchmarks (milk 3,793 s after the fix; mice 327 s). CI: the commits
before the test-data fix failed only the new unit-invariance test on some jobs
(non-reproducible test data, fixed); see the validation report §3a for the final run.

Round 14: full test suites (native 357 passed; pure-Python kernels 351 passed, 6
native-only skipped), self-test T01–T18 with both kernels, all 24 example runs (example
19: 1,145 s), the 30-replicate two-categorical study (2,560 s). CI: 683fe36, dd7ed61 and
d80fc5c (ABP 0.14.0), runs 37097684360, 37098406846 and
[37100055863](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/37100055863),
7 of 7 jobs succeeded each.

Round 13: full test suites (native 351 passed; pure-Python kernels 345 passed, 6
native-only skipped), self-test T01–T18 with both kernels, all 23 example runs (example
18 REML: 1.3 s), the 200-replicate maternal REML study (373 s). CI: 2b86935, b520d82 and
62b2a7f (ABP 0.13.0), runs 37088933310, 37089408924 and
[37089590066](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/37089590066),
7 of 7 jobs succeeded each.

Round 12: full test suites (native 338 passed; pure-Python kernels 332 passed, 6
native-only skipped; 4,649 s instead of 7,254 s), self-test T01–T18 with both kernels,
all 22 example runs (examples 15–18 gave the round-11 posterior means; example 18 with
weak priors), the maternal study twice (flat and weak priors, 20 replicates each,
2,866 s and 2,835 s), factorization timings (validation report §7.16). CI: 728dbe0 run
[37024748732](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/37024748732),
7 of 7 jobs succeeded (pure-Python job 40 min).

Round 11: full test suites (native 332 passed; pure-Python kernels 327 passed, 5
native-only skipped, run alongside the maternal study), self-test T01–T18 with both
kernels, all 21 example runs (example 18, maternal model: converged after 8,000
iterations), the maternal calibration study (20 replicates × 1,500 animals, 8,672 s;
finding F19). The two maternal statistical tests were then made smaller for the
pure-Python CI job (pure-Python kernels: 6 passed, 920.9 s). CI: 7f30e96 run
36990166293 and 0c25b62 run
[37008139663](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/37008139663),
7 of 7 jobs succeeded each.

Round 10: full test suites (native 326 passed; pure-Python kernels 321 passed, 5
native-only skipped), self-test T01–T18 with both kernels, all 20 example runs (examples 15–17 with the
new moves), the F16 weak-prior study (30 replicates, three traits); CI after every
push: a569d8f green (pure-Python job 2 h 17 min, tests then made smaller), release
commit 8a32efe (ABP 0.10.0): run
[36883815725](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/36883815725),
7 of 7 jobs succeeded.

Round 9: full test suites (native 317 passed; pure-Python kernels 312 passed,
5 native-only skipped; both with one BLAS thread while a study used the other
cores), self-test T01–T18 with both kernels, all 19 example runs (including
examples 15 and 16) and the API example, the F16 study (2 scenarios × 50
replicates, 4,000 iterations; plus a 16,000-iteration check on 6 seeds), the
inbreeding benchmark (5 pedigrees); CI after every push: run 36807935977 failed
on the pure-Python job (a test defect, fixed in d3e2a71: run 36812933794, 7 of 7
green); release commit 15d7d38 (ABP 0.9.0): run
[36832541614](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/36832541614),
7 of 7 jobs succeeded (validation report §3, §3a, §7.13).

Round 8: full test suites (native 311 passed; pure-Python kernels 306 passed,
5 native-only skipped), self-test T01–T18 with both kernels, all 18 example runs
(including example 15) and the API example (sections 18–27), the estimator study
(300 seeds), the rank-selection study (4 × 100), the multi-trait threshold study
(30 replicates × 12,000 iterations, plus 3,000-iteration and true-prior runs), the
cross-check against the first implementation, and the workflow-level (120,000
animals) and library-level (200,000 animals) benchmarks after the speed-ups; CI
after every push, all green; release commit fa2894b (ABP 0.8.0): run
[36727601564](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/36727601564),
7 of 7 jobs succeeded (validation report §3, §3a, §7.12).

Round 7: full test suites (native 297 passed; pure-Python kernels
292 passed, 5 native-only skipped), self-test T01–T17 with both kernels, all examples and the API example
(sections 18–26), threshold prior study (30 replicates × 2 priors), rank
studies with rank-1 and full-rank truths (100 each; the rank-1 study also
scores the reduced-rank Kackar–Harville PEV), workflow-level large run; CI
after every push, all green; release commit 329f8d5 (ABP 0.7.0): run
[36681792730](https://github.com/1958126580/Animal_Breeding_Program/actions/runs/36681792730),
7 of 7 jobs succeeded (validation report §3, §3a, §7.11).

Round 6: full test suites (native 285 passed; pure-Python kernels 280 passed,
5 native-only skipped), self-test
T01–T16, all examples and API sections 18–25, threshold study with the Gibbs
scenario (30), reduced-rank study (100), multi-trait study with the corrected
PEV (50), 200,000-animal matrix-free run; CI (validation report §3, §3a, §7.10).

Round 5: full test suite with the native kernel (259 passed) and with
`ABP_DISABLE_NATIVE=1` (254 passed, 5 native-only skipped), all 17 examples exit 0, `abp selftest` (T01–T15), all examples and the API
example, F9 study (2 × 200 replicates), Kackar–Harville calibration (50),
APY calibration (50), threshold study with Laplace REML (30), matrix-free
benchmark (50,000 animals / 6,000 genotyped); CI runs 36522550336,
36523269114, 36524938324 green on 7 jobs (validation report §3, §3a, §7.9).

Round 4: full test suite with both kernels, `abp selftest` (T01–T14), all
examples incl. 13 and 14, the F6 factor study (4 × 50 replicates), the
multi-trait study (50), the two-metafounder study (30), the threshold study
(30) and benchmarks `--only ldl apy` (details in `docs/validation_report.md`
§3 and §7.4–7.7).


See `docs/validation_report.md` §3 for the full list with logs. Round 3:
the full test suite with the native kernel (211 passed) and with
`ABP_DISABLE_NATIVE=1`, `abp selftest` (both kernels, T01–T13 PASS), every
example (exit status 0), the 50-replicate calibration study with six
scenarios (the three round-2 scenarios reproduced exactly), and the round-3
benchmarks (`--only selinv metafounders`). CI: all four round-3 pushes green
on 7 jobs (run 36447242937: 211 passed on Windows and Linux).

## 3. Not run, and why

Windows performance measurements (functional tests only, in CI), CUDA (no
implementation), comparison software (no licenses), forward-in-time real-data
validation (round-15 real data have no dates; NCBI, CRAN, github.com and figshare are
blocked by the network policy, raw.githubusercontent.com is allowed), LR population accuracy (erratum not verifiable), posterior SBC,
threshold models with several categorical traits or extra random terms,
a second-order Kackar–Harville correction (F16), extra random terms or proper R0
priors in the Bayesian multi-trait linear model, a workflow-level run at 200,000 animals (the 200,000-animal runs are
library-level), and a study of the multi-trait threshold model with repeated
categorical records.

## 4. Open scientific and engineering risks

See `docs/validation_report.md` §8. The most important:

* **F6 (resolved; cause of the remainder known)**: metafounder single step is
  unbiased; its residual 8% PEV over-confidence comes from marker density
  (calibrated with 10,000 SNPs). Single step with `match_a22` remains biased.
* **F9 resolved**: 200 replicates, bias 0.049 ± 0.024 kg (sampling variation).
* **F10 addressed (opt-in)**: reduced-rank G0; default still stops with
  ABP-E300; singular R0 still stops.
* **F11 resolved by the Gibbs sampler** (genetic liability variance 0.108 ±
  0.009, true 0.111); Laplace REML stays available and documented as biased.
* **F12 resolved**: single-trait, multi-trait and reduced-rank
  Kackar–Harville PEV calibrated (reduced rank 1.007/1.038).
* **F13 (open)**: with 2–3 categorical records per animal the liability
  variances are weakly identified: uniform priors give pe +24%; proper priors
  move the estimates towards their centre (round-7 study). Remedy: more
  records per animal or known variances. **F14**: sampled reliabilities carry
  Monte-Carlo error (SE 0.21 with 10 simulations at 120,000 animals).
* **F16 (partly resolved, rounds 9–10)**: full-rank multi-trait REML on small data:
  low-h² traits stay 14–44% optimistic even with the Kackar–Harville PEV. The
  Bayesian multi-trait linear model is calibrated with two traits (MSE/PEV 1.000 ±
  0.040); with three traits 1.06 / 1.14 (flat priors) and 1.03 / 1.11 with weak
  data-centred priors (round 10), at +0.025 for the best-determined trait; §7.13–7.14.
* **F19 (round 11; revised in round 13)**: Bayesian maternal model, posterior mean of
  the maternal variance 2.54 vs 2.0 over 20 replicates. REML on the same seeds gives
  2.46 and is unbiased over 200 replicates (2.06), so the seeds were a high draw; the
  posterior mean exceeds REML by +0.17 ± 0.05 (flat) and −0.05 (weak priors, round 12)
  on the same data (validation report §7.16–7.17).
* **Round-15 defect (fixed)**: the REML zero-variance check accepted a false zero
  when variances are large (milk in lb: additive 0 instead of 925,611; 8 of 30
  simulated data sets with y × 3,000). Any earlier fit of a large-variance trait that
  ended on the boundary should be re-run with 0.15.0.
* **F22 (new, round 15)**: real data, EBVs of animals without close relatives in the
  training data are over-dispersed (mice across families: regression 0.52–0.66;
  Holstein from relatives only: 0.72–0.82); targets contain shared environment, so
  not a pure calibration test; next: LR validation on data with dates.
* **F21 (new, round 14)**: several categorical traits with one record per animal and
  trait: binary liability variance overestimated (0.25 vs 0.16) and genetic
  correlations pulled to the prior centre (0.35 vs 0.5); EBVs still better than
  single-trait (validation report §7.18).
* **F20 (new, round 13)**: maternal REML, plug-in maternal PEV optimistic when the
  maternal variance is underestimated (MSE/PEV 1.56, median 1.04; 5.2 for estimates
  below 1.0); Kackar–Harville columns 1.35; Bayesian posterior PEV calibrated (0.98).
* **Engineering (round 9)**: multi-threaded OpenBLAS under a fully loaded CPU slowed
  a 12 s REML example beyond 600 s (thread oversubscription); with one BLAS thread
  it took 20 s. Set `OPENBLAS_NUM_THREADS` when ABP shares a machine.
* **F17 (new)**: one categorical record per animal + flat variance prior = improper
  posterior (chains drift, results withheld); use a proper prior and report it.
* **F18 (new)**: the genetic correlation between a single-record categorical
  trait and a continuous trait follows the prior's centre (0.27 vs 0.51 for
  prior 0 vs 0.5; truth 0.5): use published estimates for `prior_covariance`.
* **F15 resolved**: plain minimum AIC reduced a full-rank G0 in 19% of data
  sets; the margin of 2 prevents it (0/100) at the cost of recognising a true
  rank 1 less often (58/100 instead of 97/100).
* **F7**: genetic groups defined by long periods leave bias when the level
  drifts within a period.
* **F8**: BayesCπ π₀ mixing can be slow; ABP withholds such results.
* The Γ sampling correction is pooled over metafounders; it slightly
  over-corrects a metafounder with extreme base frequencies (unit test:
  −0.008 vs +0.018 uncorrected).
* Large exact-PEV runs are dominated by the single-threaded SuperLU
  factorization (15.2 of 16.9 s at 100,500 equations).

## 5. Information gaps that block the next phase (at most five)

1. **Real data and authorization scope.** Which species and populations,
   which files, and whether data may leave the local machine (currently
   never). Round 15 used two public data sets; the target populations and a
   data set with dates (for forward-in-time validation) are still needed.
2. **Breeding objectives and economic weights** per species, with units and
   sources. The examples use synthetic placeholders only.
3. **Deployment targets.** Typical data sizes (animals, genotyped animals,
   markers) and hardware. Exact PEV now scales to ~10⁵ equations on 4 cores;
   larger genotyped populations need APY and a faster factorization.
4. **Access to comparison software** (BLUPF90, MiXBLUP, ASReml, DMU, JWAS,
   BGLR), with license terms for benchmarking.
5. **Project license and distribution model.** This is the owner's decision;
   no license file has been added.

## 6. Next concrete tasks (in order)

Round 15 did the first part of task 7 below (retrospective real-data checks on public
data; it found and fixed a REML boundary defect). Round 14 did the first part of task 2
(several categorical traits). Round 13 did task 1 of the round-12 list. Next:

1. Maternal models: multi-trait maternal REML; a second-order or bootstrap correction
   for the maternal PEV under REML (F20); a singular-G0 (reduced-rank) maternal fit
   instead of stopping with ABP-E300.
2. Multi-trait Gibbs samplers: residual correlations between categorical traits (a
   correlation-matrix step for the liabilities); more than one iid term.
3. Sparse LDL': general supernodes (dense updates for every front, not only the
   trailing block); measure the block-size rule's flop-rate ratio on other machines
   (Windows CI timings).
4. F16 remainder: the least informed trait of three stays 7% optimistic even with
   weak priors; test a second-order Kackar–Harville correction and longer chains on
   the three-trait scenario.
5. APY construction at scale: the dense `G_cn` product (float32 option with a
   documented error bound) and parallel Colleau products.
6. Sampled PEV: combine the orthogonal estimator with control variates from an
   approximate reliability.
7. Real-data validation (G5): a forward-in-time LR validation on public data with
   birth or test dates (search the CRAN mirror on raw.githubusercontent.com, e.g. data
   sets of other animal-breeding packages); F22 (over-dispersion across families:
   compare G frequency bases and the ridge); the target populations and comparison
   software remain blocked on the gaps in §5.

## 7. Where things are

| Path | Content |
|---|---|
| `src/abp/` | package (see `docs/api.md` for layers) |
| `src/abp/core/metafounders.py`, `src/abp/workflows/metafounder_inputs.py` | metafounders (round 3) |
| `src/abp/solvers/selinv.py` | sparse selected inversion (round 3) |
| `src/abp/solvers/cholesky.py`, `multitrait_reml.py`, `threshold.py` | sparse LDL', multi-trait REML, threshold model (round 4); reduced-rank REML and Laplace REML (round 5) |
| `src/abp/solvers/vc_uncertainty.py`, `src/abp/core/ssop.py` | Kackar–Harville PEV; matrix-free single-step operators (round 5) |
| `src/abp/solvers/threshold_gibbs.py`, `src/abp/solvers/pev_sampling.py` | threshold Gibbs sampler; sampled PEV (round 6) |
| `src/abp/solvers/mt_threshold_gibbs.py`, `src/abp/workflows/mt_threshold.py` | multi-trait threshold model (round 8); Bayesian multi-trait linear model and residual groups (round 9); permanent environment, P0/R0 priors, scale moves for every trait (round 10) |
| `src/abp/_native.cpp` | C++20 kernels: inbreeding (`inbreeding_ml`, depth kernel `inbreeding_depth`, round 9), Bayesian sweep, `ml_general`, `symbolic_cholesky`, `takahashi`, `mindegree_order`, `ldl_numeric`, `ldl_solve`, `colleau_times` |
| `examples/20_holstein_milk_real/`, `examples/21_mice_bodyweight_real/`, `benchmarks/real_*_validation.py` | real-data examples (fetched on demand) and validation scripts (round 15) |
| `tests/` | test suite; `tests/reference/` holds the independent dense references (incl. `tabular_a_metafounders`) |
| `examples/` | runnable examples and synthetic data (`*/truth` folders are for validation only) |
| `docs/` | manual, methods, API, validation, benchmarks, ADRs, requirements, registry, license inventory, error codes |
| `docs/validation/` | raw evidence: test logs, JUnit XML, study results |
| `contracts/` | JSON Schemas and data dictionary |
| `benchmarks/` | benchmark, calibration, UPG, SBC and cross-check scripts and results |
| `packaging/` | Windows and Linux launchers |
| `.github/workflows/ci.yml` | CI matrix |
