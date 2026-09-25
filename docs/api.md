# ABP Python API

Version 0.1.0. Every snippet below is taken from `examples/api_example.py`,
which the test suite runs (`tests/test_examples.py::test_api_example_script_runs`).
The mathematics behind each function is in [`methods.md`](methods.md).

The public surface in 0.1 is the set of modules and names listed here. Other
names (leading underscore, or not listed) may change without notice. All
arrays are NumPy `float64`, and sparse matrices are SciPy CSR.

## Layers

| Package | Role |
|---|---|
| `abp.core` | pedigree (`pedigree`), fixed-effect design (`design`), genomic relationships (`genomic`), model compiler (`model`), analysis spec (`spec`) |
| `abp.io` | delimited-table reader with provenance (`tables`) |
| `abp.qc` | pedigree, phenotype and genotype QC with structured findings |
| `abp.solvers` | MME assembly and solvers (`mme`), single-trait BLUP (`blup`), REML (`reml`), multi-trait BLUP (`multitrait`) |
| `abp.decision` | selection indices (`selection_index`) |
| `abp.workflows` | end-to-end evaluation, outputs, manifests, reports |
| `abp.errors` | `ABPError` and the error-code table |

## Errors

Every detected failure raises `abp.errors.ABPError`:

```python
from abp.errors import ABPError
try:
    ...
except ABPError as err:
    err.code          # "ABP-E200"
    err.spec.name     # "PEDIGREE_CYCLE"
    err.exit_status   # 4
    err.message       # occurrence-specific text
    err.details       # machine-readable context (animals, lines, values ...)
    err.to_dict()     # JSON-ready form, as written to manifest.json
```

## 1. Pedigree: `abp.core.pedigree.Pedigree`

```python
from abp.core.pedigree import Pedigree

ped = Pedigree.from_parent_ids(
    ids=["1", "2", "3", "4", "5", "6", "7", "8"],
    sires=[None, None, None, "1", "3", "1", "4", "3"],
    dams=[None, None, None, None, "2", "2", "5", "6"])
F = ped.inbreeding()          # internal order = ped.ids (parents before offspring)
Ainv = ped.ainv()             # scipy.sparse CSR, exact Henderson/Quaas rules
ped.relationship(["7"], ["8"])  # -> array([[0.25]])
```

| Member | Returns |
|---|---|
| `Pedigree.from_parent_ids(ids, sires, dams)` | ordered, validated pedigree. `None` = unknown parent. Every named parent must be listed (the QC layer adds missing parents). Raises on duplicates, cycles, self-parenthood. |
| `.ids`, `.sire`, `.dam`, `.generation`, `.n` | internal order; parent indices, with `-1` for unknown |
| `.index_of(ids)` | internal indices of IDs |
| `.inbreeding()` | `F` (C++ kernel if available); `.inbreeding_kernel` names the kernel that ran |
| `.mendelian_d()` | `d_i` of `A = T D T'` |
| `.ainv()` | sparse `A⁻¹` |
| `.logdet_a()` | `log|A|` |
| `.a_times(x)` | `A @ x` for a vector or matrix, without forming `A` |
| `.a_columns(idx)`, `.a_submatrix(idx)`, `.a_dense()` | dense parts of `A` (`a_dense` refuses more than 20,000 animals) |

## 2. Fixed effects: `abp.core.design`

```python
from abp.core.design import FixedTerm, build_fixed_design
fixed = build_fixed_design({"sex": ["M", "F", "F", "M", "M"]},
                           [FixedTerm("sex", "factor")], intercept=False, n=5)
fixed.X                    # full-column-rank sparse design
fixed.labels, fixed.kept   # all candidate columns and which were kept
fixed.constrained_labels   # columns set to zero for identifiability
```

## 3. Single-trait BLUP: `abp.solvers.blup`

```python
import numpy as np, scipy.sparse as sp
from abp.solvers.blup import RandomTerm, blup

rec_ids = ["4", "5", "6", "7", "8"]
y = np.array([4.5, 2.9, 3.9, 3.5, 5.0])
Z = sp.csr_matrix((np.ones(5), (np.arange(5), ped.index_of(rec_ids))), shape=(5, ped.n))
animal = RandomTerm("animal", Z, Ainv, ped.ids, genetic=True,
                    k_diag=1.0 + F, logdet_k=ped.logdet_a())
res = blup(y, fixed.X, [animal], {"animal": 20.0, "residual": 40.0})
t = res.terms["animal"]      # .solution, .pev, .reliability, .labels
res.fixed_solution           # kept fixed-effect columns
res.solve.method, res.solve.selection_reason, res.solve.rel_residual
```

`RandomTerm(name, Z, k_inv, labels, genetic, k_diag=None, logdet_k=None)`
takes the inverse structure `K⁻¹` *without* variance scaling. `blup(y, X,
terms, variances, method="auto", compute_pev=True, tol=1e-10,
max_iter=10000, memory_budget_bytes=4 GiB)` returns a `BLUPResult`.
`method` is one of `auto`, `dense`, `sparse_direct` or `pcg` (see
`abp.solvers.mme.choose_method`).

## 4. REML: `abp.solvers.reml`

```python
from abp.solvers.reml import reml_fit
fit = reml_fit(y2, X2, [term2], {"algorithm": "ai", "max_iter": 100, "tol": 1e-8, "start": None})
fit.status          # "converged" or "converged_boundary"
fit.variances       # {"animal": ..., "residual": ...}; boundary components are 0.0
fit.se, fit.heritability, fit.heritability_se, fit.history
```

Optional arguments: `checkpoint_path`, `resume`, `fingerprint` (checkpoint
and resume) and `memory_budget_bytes`. Raises `ABPError("REML_NOT_CONVERGED")`
when the budget is exhausted. `REMLEvaluator(y, X, terms, budget).evaluate(theta)`
exposes log L, the score, the AI matrix and the EM update at any `theta`.

## 5. Genomic: `abp.core.genomic`

```python
from abp.core.genomic import allele_frequencies, apply_g_policy, single_step, vanraden_g
p = allele_frequencies(M)                  # M: n x m dosages of the counted allele
G, d = vanraden_g(M, p)                    # G = WW'/d
Gstar, record = apply_g_policy(G, policy="blend", tuning="match_a22", A22=A22,
                               alpha=0.05, ridge=0.0)
ss = single_step(ped2, ped2.index_of(geno_ids), Gstar, a22=A22)
ss.h_inv, ss.h_diag, ss.logdet_h
```

`apply_g_policy` raises `RELATIONSHIP_SINGULAR` when the result is not
positive definite. `record` holds every adjustment and the eigenvalues before
and after. `minor_allele_frequency(p)` returns `min(p, 1 − p)`.

## 6. Multi-trait BLUP: `abp.solvers.multitrait`

```python
from abp.solvers.multitrait import MTData, build_and_solve
mt = build_and_solve(MTData(Y, Xb, np.arange(ped2.n)), ped2.ainv(), 1 + ped2.inbreeding(), G0, R0)
mt.ebv          # q x t
mt.pev_blocks   # q x t x t
mt.reliability  # q x t
```

In `MTData(Y, X_per_trait, animal_col)`, `Y` is `n_records × t` with `NaN`
for missing values. `X_per_trait[j]` has one row per record in which trait
`j` is observed. `animal_col` maps records to columns of `K`.

## 7. Selection indices: `abp.decision.selection_index`

```python
from abp.decision.selection_index import ebv_index, smith_hazel
r = smith_hazel(P=np.array([[4.0, 1.0], [1.0, 9.0]]), C=np.array([[2.0], [3.0]]),
                a=np.array([1.0]), G_H=np.array([[5.0]]), info_names=["own", "sibs"],
                trait_names=["H"], proportion_selected=0.1)
r.b, r.reliability, r.accuracy, r.response_objective
idx, idx_rel = ebv_index(mt.ebv, mt.pev_blocks, np.array([1.0, 2.0]), G0, 1 + ped2.inbreeding())
```

`smith_hazel(..., restrict=["trait"])` computes the restricted index.
`selection_intensity(p)` returns the truncation-selection intensity.

## 8. Whole workflow: `abp.workflows.evaluate.run_evaluation`

```python
from abp.workflows.evaluate import run_evaluation
out = run_evaluation("examples/01_textbook_mrode_3_1/analysis.toml", "runs/ex01",
                     force=False, resume=False, console=False)
out.status      # "passed"
out.out_dir     # Path of the published folder
out.results     # dict, identical to results.json
out.manifest    # dict, identical to manifest.json
```

On failure `run_evaluation` raises `ABPError` after writing
`<out>.failed-<run id>/manifest.json`. Related helpers:
`abp.workflows.validate.validate_inputs(spec)` (QC only),
`abp.core.spec.load_spec(path)` (validated spec object) and
`abp.examples.sheep.write_sheep_example(out, seed)` (synthetic data).
