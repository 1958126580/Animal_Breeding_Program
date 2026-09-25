# Benchmarks

Measured 2026-09-25 with `python benchmarks/run_benchmarks.py --full`
(round 1) and `--only bayes ocs upg plink` (round 2). Raw results:
`benchmarks/results/2026-09-25-linux-x86_64.json` and
`benchmarks/results/2026-09-25-linux-x86_64-round2.json`.

**Machine:** cloud Linux VM, Intel Xeon @ 2.10 GHz, 4 vCPUs (1 thread per
core), 15 GiB RAM, no GPU. **Software:** Python 3.11.15, NumPy 2.4.6,
SciPy 1.17.1, OpenBLAS 0.3.31 (scipy-openblas), ABP 0.1.0/0.2.0, native kernel
built with GCC 13.3 (`-O3 -std=c++20`). **Method:** synthetic inputs from
fixed seeds; wall-clock time of one run per case (`time.perf_counter`), with
warm imports but no warm-up run. Peak RSS of the whole benchmark process was
1.82 GB.

These numbers describe this machine only. They are **not** a comparison with
other software: no comparison with BLUPF90, MiXBLUP, DMU, ASReml, JWAS or
BGLR has been run (`not_run`).

## Pedigree kernels

| Case | Animals | Ordering | Inbreeding (C++) | Inbreeding (Python ref.) | A⁻¹ build | max &#124;ΔF&#124; C++ vs Python |
|---|---:|---:|---:|---:|---:|---:|
| 10 generations × 1,000 | 10,000 | 0.06 s | 0.06 s | 0.74 s | 0.006 s | 0.0 |
| 10 generations × 10,000 | 100,000 | 0.42 s | 0.77 s | 8.33 s | 0.11 s | 0.0 |
| 20 generations × 5,000 | 100,000 | 0.38 s | 8.69 s | 89.62 s | 0.13 s | 0.0 |

The deep-pedigree Python time (89.6 s) is the profiling evidence behind ADR
0002. Inbreeding cost grows with the number of (animal, ancestor) pairs, so
deep pedigrees remain the most expensive case. A faster algorithm for very
large, deep pedigrees is a roadmap item.

## Mixed-model equations (single trait, known variances)

Animal model with a 500-level contemporary-group factor; 10 discrete
generations of 50 sires.

| Case | Equations | Solver | Time | Relative residual | Iterations |
|---|---:|---|---:|---:|---:|
| 5,000 animals, 4,000 records, with PEV | 5,500 | dense Cholesky + L⁻¹ | 4.78 s | 7.0e-15 | – |
| 100,000 animals, 80,000 records, no PEV | 100,500 | sparse direct (SuperLU) | 15.08 s | 1.1e-12 | – |
| 100,000 animals, 80,000 records, no PEV | 100,500 | Jacobi-PCG, tol 1e-10 | 0.26 s | 9.2e-11 | 111 |

These results changed the automatic solver choice. Large systems without PEV
now go to PCG, with a fallback to sparse direct if PCG does not converge.

## REML

| Case | Equations | Iterations | Time | Estimates (simulated) |
|---|---:|---:|---:|---|
| 3,000 animals, 2,500 records, AI-REML (dense) | 3,050 | 6 | 3.19 s | σ²a 1.78 (2.0), σ²e 4.36 (4.0) |

## Genomic relationship matrix

| Case | Time |
|---|---:|
| G, 2,000 animals × 50,000 markers | 1.25 s |

## End-to-end examples (same machine)

| Example | Equations | Wall time |
|---|---:|---:|
| 02 wwt, REML (7 AI iterations) + BLUP + PEV, 2,108 animals | 2,132 | 3.0 s |
| 04 fec GBLUP, REML | 490 | 0.9 s |
| 06 four-trait BLUP with PEV blocks + index | 8,506 | 8.3 s (was 15.3 s before PEV blocks were taken from L⁻¹) |

## Round 2: genetic groups, Bayesian sweep, OCS and mating, PLINK

| Case | Size | Time | Check |
|---|---|---:|---|
| A⁻¹ with genetic groups (Henderson rules, groups as parents) | 100,000 animals, 50 groups | 0.052 s (plain A⁻¹: 0.036 s; inbreeding, shared: 0.73 s) | — |
| gene fractions `Q` (vectorised by generation) | 100,000 × 50 | 0.050 s (Python loop before optimisation: 0.41 s) | rows sum to 1 when every founder is grouped |
| one Gibbs marker sweep, BayesC path, C++ kernel | 1,000 records × 10,000 markers | 0.012 s (Python reference 0.032 s) | max \|Δβ\| 1.5e-16 |
| one Gibbs marker sweep, C++ kernel | 5,000 records × 50,000 markers | 0.34 s (Python reference 0.39 s) | max \|Δβ\| 2.8e-16; memory-bound (W is 2 GB) |
| optimal contributions (ceiling active) | 500 candidates | 0.15 s | KKT stationarity 3.7e-10 |
| mating allocation LP (HiGHS) | 76 sires × 400 dams (30,400 pairs) | 0.56 s | all plan checks true |
| optimal contributions (ceiling active) | 1,500 candidates | 1.00 s | KKT stationarity 7.6e-11 |
| mating allocation LP (HiGHS) | 97,200 candidate pairs | 12.6 s | all plan checks true |
| PLINK `.bed` decoding to float64 | 1,000 × 10,000 (80 MB output) | 0.09 s | — |
| PLINK `.bed` decoding to float64 | 5,000 × 50,000 (2 GB output) | 10.3 s | on this VM, allocating and first-touching 2 GB alone takes 8.0 s (measured separately) |
| example 10, BayesC, 4 chains × 8,000 iterations | 247 records × 2,000 markers, 476 GEBVs | 25 s end to end | converged |

The native Bayesian sweep gains most on small and medium problems; at
5,000 × 50,000 both kernels are limited by memory bandwidth (every sweep
reads the whole 2 GB genotype matrix). Storing genotypes as float64 costs
8 bytes per genotype; a compact storage type is a roadmap item for large
marker panels. The mating LP grows with the number of sire × dam pairs.

## Scale limits

| Operation | Limit | Reason |
|---|---|---|
| dense solver (default choice) | ≤ 12,000 equations and within `resources.max_memory_gb` | memory 16–24 × N² bytes |
| exact PEV, sparse path | ≤ 30,000 equations | one sparse solve per equation |
| REML | dense only | needs selected elements of C⁻¹; sparse selected inversion is on the roadmap |
| G and H | dense `n_g × n_g` | APY is on the roadmap |
| fixed-effect rank check | ≤ 20,000 columns | dense X'X scan |
| OCS | a few thousand candidates | dense candidate A and active-set QP |
| mating LP | ~10⁵ sire × dam pairs in seconds | LP size = number of pairs |
| Bayesian models | genotypes held as dense float64 in memory | 8 bytes per genotype |

## Not measured

Windows timings (the CI job records test results, not benchmarks), GPU/CUDA
(no CUDA path exists), NUMA and multi-socket effects, energy use (no meter),
and data sets above 100,000 animals.
