# ADR 0001: Python 3.11+ with NumPy/SciPy as the single primary language

Status: accepted (2026-09-25) · Supersedes: none

## Context

The project control instruction recommends Julia for model development and a
reference implementation, with C++20 for profiled hot kernels, and allows a
single primary language when existing code or team conditions do not fit,
provided the decision is recorded. It also demands one scientific core, not
two diverging full implementations, and native Windows and Linux use.

Facts at decision time:

* The repository was empty; there was no existing Julia or C++ code.
* The build environment has Python 3.11, GCC 13 and CMake, but no Julia.
* NumPy/SciPy ship pre-built, BLAS/LAPACK-backed wheels for Windows and Linux
  (OpenBLAS; dense Cholesky, triangular inverse, SuperLU and sparse
  triangular solves are all compiled code).
* The uploaded JWAS (Julia) snapshot is reference material only; the spec
  says it must not be treated as production quality assurance.

## Decision

* **Primary language: Python >= 3.11** for the model compiler, workflows,
  CLI and the scientific reference implementation. Heavy linear algebra is
  delegated to LAPACK/SuperLU through NumPy/SciPy.
* **C++20 only for kernels that profiling shows to be bottlenecks**, exposed
  through the CPython C API without extra dependencies, each with the Python
  implementation kept as the reference and the automatic fallback (ADR 0002).
* **No Julia code** in 0.1. The model specification (TOML + JSON Schema) is
  language-neutral, so a Julia or C++ back end can be added behind the same
  contract later if profiling justifies it.
* Float64 everywhere on the scientific path.

## Consequences

* One scientific core; tests compare it to independent dense references.
* Pure-Python loops are slow; they are acceptable only outside hot paths,
  and every hot path must be measured (`benchmarks/run_benchmarks.py`).
* Installation from source needs a C++20 compiler for the optional kernel;
  without one ABP still works (slower inbreeding computation), and the run
  manifest records which kernel ran.
* GPU/CUDA is not implemented; requesting it fails or falls back to CPU
  explicitly, as configured.
