# ADR 0002: Optional C++20 kernels behind a Python reference

Status: accepted (2026-09-25)

## Context

Profiling (`benchmarks/results/2026-09-25-linux-x86_64.json`) showed that
the pure-Python Meuwissen-Luo (1992) inbreeding tracer took 89.6 s for a
100,000-animal, 20-generation pedigree, while every other pedigree step took
under 0.5 s. The algorithm is sequential pointer chasing that NumPy cannot
vectorise.

## Decision

* Implement the tracer once in C++20 (`src/abp/_native.cpp`) using only the
  CPython C API (buffer protocol in, `bytes` out; GIL released while
  computing). Build it as an **optional** setuptools extension.
* Keep `abp.core.pedigree.inbreeding_meuwissen_luo` (Python) as the
  scientific reference. Tests require agreement to 1e-13 on inbred
  pedigrees, and agreement with the independent tabular method.
* Selection at run time: the native kernel when importable, unless
  `ABP_DISABLE_NATIVE=1`. The manifest and QC report name the kernel.

## Consequences

* Measured speed-up: 89.6 s → 8.7 s (100k animals, 20 generations) and
  8.3 s → 0.77 s (100k, 10 generations). Results were bit-identical in the
  benchmark (maximum absolute difference 0.0).
* The C++ kernel memoises complete (sire, dam) families instead of only
  consecutive full sibs; the traced quantity is identical.
* Remaining cost is proportional to the number of (animal, ancestor) pairs.
  Faster algorithms for very deep, large pedigrees (e.g. Sargolzaei et al.
  2005) are a roadmap item, to be added only with the same reference tests.
