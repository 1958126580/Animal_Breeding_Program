# Benchmarks

Measured 2026-09-25 with `python benchmarks/run_benchmarks.py --full`
(round 1) and `--only bayes ocs upg plink` (round 2), and 2026-09-28 with
`--only selinv metafounders` (round 3), and 2026-09-29 with `--only ldl apy`
(round 4). Raw results:
`benchmarks/results/2026-09-25-linux-x86_64.json`,
`benchmarks/results/2026-09-25-linux-x86_64-round2.json` and
`benchmarks/results/2026-09-28-linux-x86_64-round3.json` and
`benchmarks/results/2026-09-29-linux-x86_64-round4.json`.

**Machine:** cloud Linux VM, Intel Xeon @ 2.10 GHz, 4 vCPUs (1 thread per
core), 15 GiB RAM, no GPU. **Software:** Python 3.11.15, NumPy 2.4.6,
SciPy 1.17.1, OpenBLAS 0.3.31 (scipy-openblas), ABP 0.1.0–0.4.0, native kernel
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

## Round 3: selected inversion and metafounders

Same machine type (a new VM of the same configuration), ABP 0.3.0.

| Case | Size | Time | Check |
|---|---|---:|---|
| BLUP with **exact PEV**, sparse direct + selected inversion (previously refused above 30,000 equations) | 100,000 animals, 80,000 records, 100,500 equations; 1,274,782 factor entries | 16.9 s end to end: SuperLU factorization 15.2 s, symbolic analysis + Takahashi recurrence 2.1 s | PEV of 300 random equations equal unit-vector solves to 6.7e-16; mean reliability 0.433 |
| AI-REML, sparse factor + selected inversion (previously dense only) | 20,000 animals, 16,000 records, 20,050 equations | 7.3 s, 6 iterations | σ²a 2.19 (simulated 2.0), σ²e 3.91 (4.0) |
| metafounder relationships `diag(A^Γ)` and `d`, 5 metafounders, C++ `ml_general` | 100,000 animals, 20 generations | 11.9 s (Python reference 167.6 s) | identical mean diagonal (1.04826) |
| inverse of the extended matrix | 100,005 equations | 0.03 s | — |

With selected inversion in place, the cost of exact PEV is dominated by the
sparse factorization (SuperLU, one thread), not by the inversion. A
supernodal Cholesky factorization is the next lever for large systems. The
metafounder trace costs the same as the ordinary Meuwissen–Luo trace on a
deep pedigree (8.7 s in round 1); founders and animals with a metafounder
parent need no tracing.

## Round 4: sparse LDL', APY, multi-trait REML

| Case | Size | Time | Check |
|---|---|---:|---|
| exact PEV, ABP LDL' (minimum degree) + selected inversion | 100,000 animals, 100,500 equations, 1,276,361 factor entries | 3.2 s end to end (ordering ≈ 2 s, numeric 0.3 s, selected inversion 0.6 s) | EBV and PEV equal the SuperLU path to 1.8e-13 and 5.7e-14 |
| same with SuperLU | 1,274,782 factor entries | 17.7 s | reference |
| G⁻¹ exact (Cholesky) | 8,000 genotyped × 10,000 SNPs | 14.6 s | — |
| G_APY⁻¹, 2,000 core animals | same | 2.4 s | smallest m_i 0.74 (all positive) |
| multi-trait REML, 3 traits (example 13) | 6,390 equations, 15 iterations | 7.9 s sparse / 44.2 s dense trace path | identical log L |

The first version of the minimum-degree ordering took 83 s on the
100,500-equation system because the intercept equation (linked to all
80,000 recorded animals) absorbed every elimination; setting rows of degree
> 10√n aside and eliminating them last (as AMD does) reduced it to about
2 s at the same fill.

## Round 5: matrix-free single step

`python benchmarks/ssmf_benchmark.py` (results `benchmarks/results/ssmf.json`,
`.log`): 50,000 animals (10 generations), 6,000 genotyped (last generations),
5,000 random SNPs, 45,000 records, 50,400 equations, G* = 0.95 G + 0.05 A22,
PCG to a relative residual of 10⁻¹⁰. Each mode ran alone in its own process
(peak resident memory of that process).

| Mode | Build H⁻¹ / operator | PCG solve | Iterations | Peak memory | Check |
|---|---:|---:|---:|---:|---|
| explicit (dense G*⁻¹, A22, A22⁻¹, n × n₂ block) | 188.1 s | 9.6 s | 143 | 10.9 GB | reference |
| explicit with APY (2,000 core) | 157.4 s | 7.7 s | 112 | 10.9 GB | reference for APY |
| matrix-free, dense G*⁻¹ | 62.7 s | 2.0 s | 145 | 3.6 GB | max \|ΔEBV\| 3.4e-10 vs explicit |
| matrix-free, APY operator (2,000 core) | 19.1 s | 1.4 s | 118 | 1.7 GB | max \|ΔEBV\| 6.4e-10 vs explicit APY |

The explicit build is dominated by the dense `n × n₂` block used for
`diag(H)` (reliabilities), which the matrix-free path does not compute. A
first measurement in which another job shared the CPU gave the same memory
figures (build 179 / 229 / 58 / 20 s); the table is the undisturbed rerun.

## Round 6: large matrix-free single step, Gibbs sampler, sampled PEV

`python benchmarks/ssmf_large.py` (`benchmarks/results/ssmf_large.json`, `.log`;
200,000 animals in 20 generations, the last 30,000 genotyped with 20,000
random SNPs stored as `int8`, 5,000 random core animals, 5% A22 blend,
190,000 records, 202,000 equations):

| Step | Wall time | Peak memory (process) |
|---|---:|---:|
| pedigree and inbreeding (native kernel) | 44.8 s | 0.13 GB |
| int8 genotypes | 12.3 s | 0.84 GB |
| A22 core columns (Colleau, blocks of 256) | 119.6 s | 3.7 GB |
| APY blocks from int8 + APY operator | 76.3 s | 6.75 GB |
| A22⁻¹ operator (sparse LDL' of A¹¹, 170,000 animals) | 48.2 s | 6.75 GB |
| PCG solve (101 iterations, relative residual 9e-9) | 14.9 s | 6.75 GB |
| sampled PEV, 20 simulations (99 PCG iterations each) | 187.9 s | 6.75 GB |

The explicit single step would need a dense 200,000 × 30,000 block alone
(44.7 GB). With 20 simulations the Monte-Carlo SE of a reliability averages
0.15; it falls with the square root of the number of simulations.

Other round-6 timings (build machine, see the logs): the threshold-model
Gibbs sampler on example 14 (1,208 lambings, 2,571 location equations, 4 chains)
converged after 8,000 iterations in about 85 s (2.6 ms per chain-iteration);
without the parameter-expansion move it had not converged after 16,000
iterations. The multi-trait Kackar–Harville correction on example 13 (24
extra sparse solves of 6,390 equations) took about 25 s with sparse solves and
262 s with dense ones. Reduced-rank REML with the analytic gradient fitted
the boundary test data in 0.4 s (19 evaluations).

## Scale limits

| Operation | Limit | Reason |
|---|---|---|
| dense solver (default choice) | ≤ 12,000 equations and within `resources.max_memory_gb` | memory 16–24 × N² bytes |
| exact PEV, sparse path | memory of the symbolic factor (24 bytes per entry) within `resources.max_memory_gb` | LDL' or SuperLU + selected inversion; checked before numeric work (`ABP-E500`) |
| REML | dense ≤ 12,000 equations; above, sparse factor + selected inversion within the memory budget | factor fill |
| metafounder Γ estimation | dense in genotyped animals (`A22`, n₂ × m dosages) | GLS base allele frequencies |
| G and H (explicit) | dense `n_g × n_g` and `n × n_g` storage | APY reduces the inversion to `O(c³ + n c²)`; `single_step_mode = "matrix_free"` avoids the dense blocks (solutions only; memory `O(nnz(A⁻¹) + c² + c n_g + n_g m)` with APY) |
| fixed-effect rank check | ≤ 20,000 columns | dense X'X scan |
| OCS | a few thousand candidates | dense candidate A and active-set QP |
| mating LP | ~10⁵ sire × dam pairs in seconds | LP size = number of pairs |
| Bayesian models | genotypes held as dense float64 in memory | 8 bytes per genotype |

## Not measured

Windows timings (the CI job records test results, not benchmarks), GPU/CUDA
(no CUDA path exists), NUMA and multi-socket effects, energy use (no meter),
and data sets above 100,000 animals.

## Round 7: workflow-level matrix-free single step with int8 genotypes

`python benchmarks/ssmf_workflow_large.py` (`benchmarks/results/ssmf_workflow_large.json`, `.log`).
The script writes a pedigree CSV (12 discrete generations × 10,000 animals),
a phenotype CSV (110,000 records, 400 contemporary groups) and a PLINK file
(the last 30,000 animals, 20,000 random SNPs, 150 MB) and runs the complete
`abp run` workflow in a child process: QC, allele frequencies, APY (5,000
random core animals, 5% A22 blend) built from int8 dosages, matrix-free single
step, PCG, 10 PEV simulations, outputs and manifest.

| Phase (from the run log) | Wall time |
|---|---:|
| reading, pedigree and phenotype QC (native inbreeding kernel) | 5 s |
| PLINK (int8), genotype QC, frequencies, A22 core columns, APY blocks, A22⁻¹ operator | 182 s |
| PCG solve, 120,400 equations (69 iterations, relative residual 9.8e-9) | 5.5 s |
| sampled PEV, 10 simulations (92 PCG iterations each) | 65 s |
| **total** | **261 s**, peak resident memory **7.38 GB** |

Generating the data took 38 s (not included). The mean Monte-Carlo SE of a
reliability was 0.21 with 10 simulations; more simulations are needed for
decisions on individual animals. The first run of this benchmark found a
manifest defect (validation report §9); the numbers above are from the rerun
after the fix (first run: 255 s, 7.39 GB).

