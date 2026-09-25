#!/usr/bin/env python3
"""Using ABP as a Python library (every snippet of docs/api.md, runnable).

Run from the repository root:  python examples/api_example.py
"""

from pathlib import Path
import tempfile

import numpy as np
import scipy.sparse as sp

from abp.core.design import FixedTerm, build_fixed_design
from abp.core.genomic import allele_frequencies, apply_g_policy, single_step, vanraden_g
from abp.core.pedigree import Pedigree
from abp.decision.selection_index import ebv_index, smith_hazel
from abp.solvers.blup import RandomTerm, blup
from abp.solvers.multitrait import MTData, build_and_solve
from abp.solvers.reml import reml_fit
from abp.workflows.evaluate import run_evaluation

# --- 1. Pedigree: order, inbreeding, A-inverse, relationships ---------------
ped = Pedigree.from_parent_ids(
    ids=["1", "2", "3", "4", "5", "6", "7", "8"],
    sires=[None, None, None, "1", "3", "1", "4", "3"],
    dams=[None, None, None, None, "2", "2", "5", "6"])
F = ped.inbreeding()                         # internal order = ped.ids
Ainv = ped.ainv()                            # scipy.sparse CSR
print("F:", {a: round(float(f), 4) for a, f in zip(ped.ids, F)})
print("A[7, 8] =", ped.relationship(["7"], ["8"])[0, 0])

# --- 2. Single-trait BLUP with known variances (Mrode 2005, Example 3.1) ----
rec_ids = ["4", "5", "6", "7", "8"]
y = np.array([4.5, 2.9, 3.9, 3.5, 5.0])
fixed = build_fixed_design({"sex": ["M", "F", "F", "M", "M"]},
                           [FixedTerm("sex", "factor")], intercept=False, n=5)
Z = sp.csr_matrix((np.ones(5), (np.arange(5), ped.index_of(rec_ids))), shape=(5, ped.n))
animal = RandomTerm("animal", Z, Ainv, ped.ids, genetic=True,
                    k_diag=1.0 + F, logdet_k=ped.logdet_a())
res = blup(y, fixed.X, [animal], {"animal": 20.0, "residual": 40.0})
t = res.terms["animal"]
for a, e, r in zip(t.labels, t.solution, t.reliability):
    print(f"animal {a}: EBV {e:+.3f}  reliability {r:.3f}")
print("solver:", res.solve.method, "| relative residual", f"{res.solve.rel_residual:.1e}")

# --- 3. REML ---------------------------------------------------------------
rng = np.random.default_rng(1)
n_anim = 300
ids = [f"a{i}" for i in range(n_anim)]
sires = [None] * 30 + [ids[k] for k in rng.integers(0, 15, n_anim - 30)]
dams = [None] * 30 + [ids[k] for k in rng.integers(15, 30, n_anim - 30)]
ped2 = Pedigree.from_parent_ids(ids, sires, dams)
A = ped2.a_dense()
u_true = np.linalg.cholesky(A) @ rng.standard_normal(ped2.n) * np.sqrt(2.0)
y2 = 10 + u_true + rng.normal(0, 2.0, ped2.n)            # one record per animal
X2 = build_fixed_design({}, [], intercept=True, n=ped2.n).X
Z2 = sp.identity(ped2.n, format="csr")
term2 = RandomTerm("animal", Z2, ped2.ainv(), ped2.ids, True,
                   k_diag=1 + ped2.inbreeding(), logdet_k=ped2.logdet_a())
fit = reml_fit(y2, X2, [term2], {"algorithm": "ai", "max_iter": 100, "tol": 1e-8, "start": None})
print("REML:", fit.status, {k: round(v, 3) for k, v in fit.variances.items()},
      "h2 =", round(fit.heritability, 3), "+-", round(fit.heritability_se, 3))

# --- 4. Genomic relationships and single step --------------------------------
geno_ids = ped2.ids[-100:]
M = rng.integers(0, 3, (100, 1500)).astype(float)
p = allele_frequencies(M)
G, d = vanraden_g(M, p)
A22 = ped2.a_submatrix(ped2.index_of(geno_ids))
Gstar, record = apply_g_policy(G, policy="blend", tuning="match_a22", A22=A22,
                               alpha=0.05, ridge=0.0)
ss = single_step(ped2, ped2.index_of(geno_ids), Gstar, a22=A22)
print("G* policy:", record.singular_policy, "tuning b =", round(record.tuning_b, 4),
      "| H-inverse nnz:", ss.h_inv.nnz)

# --- 5. Multi-trait BLUP -----------------------------------------------------
Y = np.column_stack([y2, 0.5 * y2 + rng.normal(0, 1, ped2.n)])
Y[rng.random(ped2.n) < 0.4, 1] = np.nan                    # trait 2 partly missing
Xb = [build_fixed_design({}, [], True, int((~np.isnan(Y[:, j])).sum())).X for j in range(2)]
G0 = np.array([[2.0, 0.8], [0.8, 1.0]])
R0 = np.array([[4.0, 0.5], [0.5, 1.5]])
mt = build_and_solve(MTData(Y, Xb, np.arange(ped2.n)), ped2.ainv(), 1 + ped2.inbreeding(), G0, R0)
print("multi-trait EBV of", ped2.ids[0], mt.ebv[0].round(3), "reliabilities", mt.reliability[0].round(3))

# --- 6. Selection indices ----------------------------------------------------
r = smith_hazel(P=np.array([[4.0, 1.0], [1.0, 9.0]]), C=np.array([[2.0], [3.0]]),
                a=np.array([1.0]), G_H=np.array([[5.0]]), info_names=["own", "sibs"],
                trait_names=["H"], proportion_selected=0.1)
print("index b =", r.b.round(4), "accuracy", round(r.accuracy, 4), "response", round(r.response_objective, 3))
idx, idx_rel = ebv_index(mt.ebv, mt.pev_blocks, np.array([1.0, 2.0]), G0, 1 + ped2.inbreeding())
print("best animal by index:", ped2.ids[int(np.argmax(idx))], "reliability", round(float(idx_rel.max()), 3))

# --- 7. Whole workflow from an analysis spec ----------------------------------
root = Path(__file__).resolve().parent
with tempfile.TemporaryDirectory() as tmp:
    out = run_evaluation(root / "01_textbook_mrode_3_1" / "analysis.toml", Path(tmp) / "run",
                         console=False)
    print("workflow:", out.status, "->", sorted(p.name for p in out.out_dir.iterdir()))
