# Handoff (read this first in the next session)

State as of 2026-09-25, ABP 0.2.0, branch `claude/festive-newton-2elqfq`.
Per the handoff instruction, trust files and tests, not this summary: re-run
`python -m pytest -q` and `abp selftest` before continuing.

## 1. What is implemented

Phases S0–S1 and parts of S2–S3. Every item passed its tests on Linux
(details in `docs/validation_report.md`, status per method in
`docs/method_registry.toml`):

* data contracts (TOML spec + JSON Schema, data dictionary, manifest schema,
  error codes); QC for pedigree, phenotypes and genotypes; PLINK 1 binary
  input;
* pedigree A/A⁻¹/F (C++20 kernel optional), Colleau products; unknown-parent
  groups (random with a declared ratio, or estimable fixed groups);
* single-trait BLUP (animal, repeatability), dense/sparse/PCG solvers, PEV,
  reliability; REML (AI + EM, boundary, SE, checkpoint and resume);
* GBLUP (VanRaden G, explicit policies), single-step H⁻¹;
* Bayesian marker models (BRR, BayesA/B/C/Cπ/R) with convergence gating,
  traces, predictive checks; MCMC diagnostics equal to ArviZ;
* multi-trait BLUP with known covariances; Smith-Hazel and restricted
  indices; EBV index;
* forward-in-time LR validation;
* optimal contribution selection and mating plans (`abp mate`, proposals
  only);
* workflow with atomic outputs, manifests and reports; CLI; launchers;
  synthetic sheep generator; 11 examples; documentation.

## 2. Commands that were run (Linux) and their results

See `docs/validation_report.md` §3 for the full list with logs. In short:
the full test suite with the native kernel and with `ABP_DISABLE_NATIVE=1`,
`abp selftest` (both kernels), every example, the 50-replicate calibration
study, the UPG study (three scenarios), the SBC of all six samplers, the
ArviZ cross-check, and the round-2 benchmarks. CI run 36151891410 (commit
c6f72df): all 7 jobs green (Windows and Ubuntu × Python 3.11–3.13, and the
pure-Python-kernel job), 173 tests passed in each.

## 3. Not run, and why

Windows performance measurements (functional tests only, in CI), CUDA (no
implementation), real-data validation and comparison software (no data or
licenses), LR population accuracy (erratum not verifiable), posterior SBC,
multi-trait calibration study.

## 4. Open scientific and engineering risks

See `docs/validation_report.md` §8. The most important:

* **F6**: single step (match_a22 + 5% blend) under-predicts candidates by
  0.66 kg with PEV understated by 28% in the calibration scenario. Do not
  use single-step reliabilities for decisions until base alignment
  (metafounders) is implemented and §7.1 is repeated.
* **F7**: genetic groups defined by long periods leave bias when the level
  drifts within a period; random groups with small ratios are strongly
  shrunk.
* **F8**: BayesCπ π₀ mixing can be slow; ABP withholds such results.

## 5. Information gaps that block the next phase (at most five)

1. **Real data and authorization scope.** Which species and populations,
   which files, and whether data may leave the local machine (currently
   never). Needed for gate G5.
2. **Breeding objectives and economic weights** per species, with units and
   sources. The examples use synthetic placeholders only.
3. **Deployment targets.** Windows versions, typical data sizes (animals,
   genotyped animals, markers), and whether GPUs exist. This decides which
   scale work (APY, sparse selected inversion, CUDA) comes first.
4. **Access to comparison software** (BLUPF90, MiXBLUP, ASReml, DMU, JWAS,
   BGLR), with license terms for benchmarking.
5. **Project license and distribution model.** This is the owner's decision;
   no license file has been added.

## 6. Next concrete tasks (in order)

1. Metafounders (Legarra et al. 2015) for pedigree and single step, then
   repeat the calibration study to address F6.
2. Sparse selected inversion (Takahashi) for exact PEV and REML traces
   beyond the dense limit (F5).
3. Multi-trait calibration study (F4) and multi-trait REML.
4. Threshold model for categorical traits (F2).
5. Posterior SBC near the example data; genomic coancestry for OCS.
6. APY for large genotyped populations, after the deployment sizes are
   known.

## 7. Where things are

| Path | Content |
|---|---|
| `src/abp/` | package (see `docs/api.md` for layers) |
| `tests/` | test suite; `tests/reference/` holds the independent dense references |
| `examples/` | runnable examples and synthetic data (`*/truth` folders are for validation only) |
| `docs/` | manual, methods, API, validation, benchmarks, ADRs, requirements, registry, license inventory, error codes |
| `docs/validation/` | raw evidence: test logs, JUnit XML, study results |
| `contracts/` | JSON Schemas and data dictionary |
| `benchmarks/` | benchmark, calibration, UPG, SBC and cross-check scripts and results |
| `packaging/` | Windows and Linux launchers |
| `.github/workflows/ci.yml` | CI matrix |
