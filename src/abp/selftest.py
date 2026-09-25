"""``abp selftest``: installation check against hand-derived analytical values.

Every expected value below was derived by hand (or is printed in the cited
source) - none is produced by ABP itself.  The checks exercise the production
kernels on the target machine (BLAS/LAPACK, optional C++ kernel), so a pass
shows that the installed build computes the documented quantities.
Tolerance: |actual - expected| <= 1e-10 + 1e-8 |expected| unless noted.
"""

from __future__ import annotations

import numpy as np

ATOL, RTOL = 1e-10, 1e-8


def _close(a, b, atol=ATOL, rtol=RTOL) -> bool:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return bool(np.all(np.abs(a - b) <= atol + rtol * np.abs(b)))


def run_selftest() -> tuple[bool, list[str]]:
    import scipy.sparse as sp

    from . import __version__
    from .core.design import FixedTerm, build_fixed_design
    from .core.genomic import minor_allele_frequency, single_step
    from .core.pedigree import Pedigree, native_kernel_available
    from .decision.selection_index import smith_hazel
    from .solvers.blup import RandomTerm, blup
    from .solvers.reml import REMLEvaluator

    lines = [f"ABP {__version__} self-test (native kernel: {native_kernel_available()})"]
    ok_all = True

    def check(name, ok, detail=""):
        nonlocal ok_all
        ok_all &= bool(ok)
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")

    # T01 minor allele frequency is the minimum of the two frequencies
    check("T01 MAF", _close(minor_allele_frequency([0.1, 0.0, 0.5, 0.9]), [0.1, 0.0, 0.5, 0.1]))

    # T02 Smith-Hazel index
    r = smith_hazel(np.array([[4.0, 1.0], [1.0, 9.0]]), np.array([[2.0], [3.0]]),
                    np.array([1.0]), np.array([[5.0]]), ["x1", "x2"], ["H"])
    check("T02 index b = [3/7, 2/7], reliability 12/35",
          _close(r.b, [3 / 7, 2 / 7]) and _close(r.reliability, 12 / 35))

    # T03 inbreeding and A-inverse (full-sib mating)
    ids = ["1", "2", "3", "4", "5"]
    ped = Pedigree.from_parent_ids(ids, [None, None, "1", "1", "3"], [None, None, "2", "2", "4"])
    idx = ped.index_of(ids)
    Ainv_exact = [[2, 1, -1, -1, 0], [1, 2, -1, -1, 0], [-1, -1, 2.5, 0.5, -1],
                  [-1, -1, 0.5, 2.5, -1], [0, 0, -1, -1, 2]]
    check("T03 F5 = 0.25, A row 5, exact A-inverse",
          _close(ped.inbreeding()[idx], [0, 0, 0, 0, 0.25])
          and _close(ped.a_dense()[np.ix_(idx, idx)][4], [0.5, 0.5, 0.75, 0.75, 1.25])
          and _close(ped.ainv().toarray()[np.ix_(idx, idx)], Ainv_exact),
          f"kernel {ped.inbreeding_kernel}")

    # T04 BLUP and PEV
    p2 = Pedigree.from_parent_ids(["A", "B"], [None, None], [None, None])
    X = build_fixed_design({}, [], True, 2).X
    Z = sp.csr_matrix(np.eye(2))
    t = RandomTerm("animal", Z, p2.ainv(), p2.ids, True, k_diag=np.ones(2), logdet_k=0.0)
    res = blup(np.array([2.0, 4.0]), X, [t], {"animal": 1.0, "residual": 1.0}, method="dense")
    tr = res.terms["animal"]
    check("T04 intercept 3, EBV -/+0.5, PEV 0.75, reliability 0.25",
          _close(res.fixed_solution, [3.0]) and _close(tr.solution[p2.index_of(["A", "B"])], [-0.5, 0.5])
          and _close(tr.pev, [0.75, 0.75]) and _close(tr.reliability, [0.25, 0.25]))

    # T05 REML score for the genetic variance (derivative uses A, not I)
    ev = REMLEvaluator(np.array([1.0, 2.0, -1.0, 0.0, 3.0]), build_fixed_design({}, [], True, 5).X,
                       [RandomTerm("animal", sp.csr_matrix(np.eye(5)[:, idx]), ped.ainv(), ped.ids,
                                   True, k_diag=1 + ped.inbreeding(), logdet_k=ped.logdet_a())],
                       2**28)
    score = ev.evaluate(np.array([0.7, 1.3])).score[0]
    check("T05 REML genetic score = -0.01463020355", _close(score, -0.01463020355, atol=1e-10))

    # T06 single step collapses to A-inverse when G* = A22
    g = ped.index_of(["3", "4", "5"])
    A22 = ped.a_submatrix(g)
    ss = single_step(ped, g, A22.copy())
    check("T06 H-inverse = A-inverse when G* = A22",
          _close(ss.h_inv.toarray(), ped.ainv().toarray(), atol=1e-12))

    # Mrode (2005) Example 3.1, printed solutions (3 decimals)
    pm = Pedigree.from_parent_ids([str(i) for i in range(1, 9)],
                                  [None, None, None, "1", "3", "1", "4", "3"],
                                  [None, None, None, None, "2", "2", "5", "6"])
    rec = ["4", "5", "6", "7", "8"]
    fd = build_fixed_design({"sex": ["M", "F", "F", "M", "M"]}, [FixedTerm("sex", "factor")],
                            False, 5)
    Zm = sp.csr_matrix((np.ones(5), (np.arange(5), pm.index_of(rec))), shape=(5, 8))
    rm = blup(np.array([4.5, 2.9, 3.9, 3.5, 5.0]), fd.X,
              [RandomTerm("animal", Zm, pm.ainv(), pm.ids, True, k_diag=1 + pm.inbreeding())],
              {"animal": 20.0, "residual": 40.0})
    u = rm.terms["animal"].solution[pm.index_of([str(i) for i in range(1, 9)])]
    check("Mrode Ex. 3.1 sex and animal solutions (+-6e-4)",
          _close(rm.fixed_solution, [3.404, 4.358], atol=6e-4, rtol=0)
          and _close(u, [0.098, -0.019, -0.041, -0.009, -0.186, 0.177, -0.249, 0.183],
                     atol=6e-4, rtol=0))
    lines.append("RESULT: " + ("PASS" if ok_all else "FAIL"))
    return ok_all, lines
