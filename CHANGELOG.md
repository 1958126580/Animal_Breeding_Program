# Changelog

All notable changes. Scientific-result changes are marked **[results]**.
Versioning: 0.x is pre-release; any change in the algorithm or the genetic
base that alters results is listed here, whatever the size of the version bump.

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
