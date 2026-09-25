# Handoff (read this first in the next session)

State as of 2026-09-25, ABP 0.1.0, branch `claude/festive-newton-2elqfq`.
Per the handoff instruction, trust files and tests, not this summary: re-run
`python -m pytest -q` and `abp selftest` before continuing.

## 1. What is implemented

Phase S0 (specification freeze) and most of S1 (first runnable version), plus
parts of S2 and S3. Every item passed its tests on Linux (details in
`docs/validation_report.md`, status per method in
`docs/method_registry.toml`):

* data contracts (TOML spec + JSON Schema, data dictionary, manifest schema,
  error codes); QC for pedigree, phenotypes and genotypes;
* pedigree A/A⁻¹/F (C++20 kernel optional), Colleau products;
* single-trait BLUP (animal, repeatability), dense/sparse/PCG solvers, PEV,
  reliability;
* REML (AI + EM, boundary, SE, checkpoint and resume);
* GBLUP (VanRaden G, explicit policies), single-step H⁻¹;
* multi-trait BLUP with known covariances; Smith-Hazel and restricted indices;
  EBV index;
* workflow with atomic outputs, manifests and reports; CLI; launchers;
  synthetic sheep generator; 7 examples; documentation.

## 2. Commands that were run (Linux) and their results

* `python -m pytest -v` → 96 passed (native kernel and `ABP_DISABLE_NATIVE=1`)
* `abp selftest` → PASS (both kernels)
* examples 01–07 → exit status 0
* `benchmarks/run_benchmarks.py --full`, `benchmarks/simulation_check.py` → results committed

## 3. Not run, and why

Windows performance measurements (functional tests passed in CI run 36130441504 on Windows Server 2025, Python 3.11–3.13), CUDA
(no implementation), real-data validation and comparison software (no data or
licenses), multi-replicate EBV calibration (planned).

## 4. Open scientific and engineering risks

See `docs/validation_report.md` §8 (F1–F5). The most important is F1:
under-dispersion and bias of candidate EBVs in the single simulated
replicate. It must be resolved by a multi-replicate study before any claim
about calibration.

## 5. Information gaps that block the next phase (at most five)

1. **Real data and authorization scope.** Which species and populations
   (sheep, cattle, pig, chicken), which files, and whether data may leave
   the local machine (currently never). This is needed for gate G5.
2. **Breeding objectives and economic weights** per species, with units and
   sources. The examples use synthetic placeholders only.
2. **Deployment targets.** Windows versions, typical data sizes (animals,
   genotyped animals, markers), and whether GPUs exist. This decides which
   scale work (APY, sparse selected inversion, CUDA) comes first.
3. **Access to comparison software** (BLUPF90, MiXBLUP, ASReml, DMU, JWAS,
   BGLR), with license terms for benchmarking.
4. **Project license and distribution model.** This is the owner's decision;
   no license file has been added.

## 6. Next concrete tasks (in order)

1. M13: forward-in-time validation workflow with LR statistics (bias,
   dispersion, ρ_wp, using the corrected SD-product denominator), plus a
   ≥ 50-replicate calibration study with the sheep generator (resolves F1/F4).
2. Sparse selected inversion (Takahashi) for exact PEV and REML traces
   beyond the dense limit. Consider a C++ kernel only after profiling.
3. M03 extension: unknown-parent groups and metafounders, with their own
   contracts and tests.
4. M08/M20: BayesC/BayesR with multi-chain diagnostics (R-hat, ESS, MCSE)
   and conjugate reference tests.
5. M12: OCS (quadratic cone program with KKT certificate) and integer mating
   allocation with enumeration tests on small cases.
6. PLINK `.bed/.bim/.fam` reader (M01), tested against a byte-level
   hand-constructed file.

## 7. Where things are

| Path | Content |
|---|---|
| `src/abp/` | package (see `docs/api.md` for layers) |
| `tests/` | test suite; `tests/reference/` holds the independent dense references |
| `examples/` | runnable examples and synthetic data (`sheep_data/truth` is for validation only) |
| `docs/` | manual, methods, API, validation, benchmarks, ADRs, requirements, registry, license inventory |
| `contracts/` | JSON Schemas and data dictionary |
| `benchmarks/` | benchmark and simulation-check scripts and results |
| `packaging/` | Windows and Linux launchers |
| `.github/workflows/ci.yml` | CI matrix |
