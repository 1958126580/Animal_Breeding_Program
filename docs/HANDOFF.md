# Handoff (read this first in the next session)

State as of 2026-09-30, ABP 0.7.0, branch `claude/ecstatic-archimedes-qs0idf`
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
  synthetic sheep generator; 14 examples; documentation; self-test T01–T17.

## 2. Commands that were run (Linux) and their results

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
implementation), real-data validation and comparison software (no data or
licenses), LR population accuracy (erratum not verifiable), posterior SBC,
calibration of the full-rank EBVs kept by the rank-selection rule (not scored
in the rank studies), a workflow-level run at 200,000 animals (the workflow
run used 120,000).

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
   never). Needed for gate G5.
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

Round 7 completed tasks 1–4 listed here in round 6 (task 1 without
multi-trait threshold models). Next:

1. Multi-trait threshold/linear models (categorical and continuous traits
   together), with a Gibbs sampler extending `threshold_gibbs`.
2. Rank selection: score the calibration of the full-rank EBVs kept by the
   margin rule, and study the rule with three or more traits.
3. Sampled PEV: variance reduction (e.g. control variates from a
   block-diagonal approximation) so that fewer simulations reach SE ≤ 0.05.
4. Workflow-level run at 200,000+ animals and a profile of the genotype
   loading / APY construction phase (182 s of 261 s at 120,000 animals).
5. Real-data validation (G5) and comparison software — blocked on the gaps
   in §5.

## 7. Where things are

| Path | Content |
|---|---|
| `src/abp/` | package (see `docs/api.md` for layers) |
| `src/abp/core/metafounders.py`, `src/abp/workflows/metafounder_inputs.py` | metafounders (round 3) |
| `src/abp/solvers/selinv.py` | sparse selected inversion (round 3) |
| `src/abp/solvers/cholesky.py`, `multitrait_reml.py`, `threshold.py` | sparse LDL', multi-trait REML, threshold model (round 4); reduced-rank REML and Laplace REML (round 5) |
| `src/abp/solvers/vc_uncertainty.py`, `src/abp/core/ssop.py` | Kackar–Harville PEV; matrix-free single-step operators (round 5) |
| `src/abp/solvers/threshold_gibbs.py`, `src/abp/solvers/pev_sampling.py` | threshold Gibbs sampler; sampled PEV (round 6) |
| `src/abp/_native.cpp` | C++20 kernels: inbreeding, Bayesian sweep, `ml_general`, `symbolic_cholesky`, `takahashi`, `mindegree_order`, `ldl_numeric`, `ldl_solve` |
| `tests/` | test suite; `tests/reference/` holds the independent dense references (incl. `tabular_a_metafounders`) |
| `examples/` | runnable examples and synthetic data (`*/truth` folders are for validation only) |
| `docs/` | manual, methods, API, validation, benchmarks, ADRs, requirements, registry, license inventory, error codes |
| `docs/validation/` | raw evidence: test logs, JUnit XML, study results |
| `contracts/` | JSON Schemas and data dictionary |
| `benchmarks/` | benchmark, calibration, UPG, SBC and cross-check scripts and results |
| `packaging/` | Windows and Linux launchers |
| `.github/workflows/ci.yml` | CI matrix |
