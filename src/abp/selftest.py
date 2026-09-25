"""``abp selftest``: installation check against hand-derived analytical values.

Every expected value below was derived by hand, is printed in the cited
source, or (T10) was computed by an independent implementation (ArviZ) -
none is produced by ABP itself.  The checks exercise the production
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
    # T07 PLINK .bed decoding (low bits first; 00=2, 01=missing, 10=1, 11=0 copies of A1)
    from .io.plink import decode_bed
    Mb = decode_bed(bytes([0x6C, 0x1B, 0x01, 0x78, 0x00, 0x2F, 0x01]), 5, 2)
    exp_b = np.array([[2, 0], [1, 0], [0, 1], [np.nan, 2], [2, np.nan]])
    check("T07 PLINK bytes 78 00 2F 01 -> A1 dosages",
          bool(np.array_equal(np.isnan(Mb), np.isnan(exp_b))
               and _close(np.nan_to_num(Mb, nan=-1), np.nan_to_num(exp_b, nan=-1))))

    # T08 unknown-parent groups: x has both parents in group G, y = x x unknown.
    # By hand: A*^-1 (x, y, G) = [[4/3, -2/3, -1], [-2/3, 4/3, 0], [-1, 0, 1]], Q = (1, 1/2).
    from .core.upg import GroupAssignment, ainv_with_groups, group_fractions
    pu = Pedigree.from_parent_ids(["x", "y"], [None, "x"], [None, None])
    iu = pu.index_of(["x", "y"])
    sg = np.full(2, -1)
    dg = np.full(2, -1)
    sg[iu[0]] = dg[iu[0]] = 0
    grp = GroupAssignment(("G",), sg, dg)
    order = np.concatenate([iu, [2]])
    check("T08 group A*-inverse and gene fractions (Quaas 1988)",
          _close(ainv_with_groups(pu, grp).toarray()[np.ix_(order, order)],
                 [[4 / 3, -2 / 3, -1], [-2 / 3, 4 / 3, 0], [-1, 0, 1]])
          and _close(group_fractions(pu, grp)[iu, 0], [1.0, 0.5]))

    # T09 optimal contributions, 4 unrelated founders: males g = (1, 0), females g = (0, 0),
    # ceiling C: female contributions 1/4 each, top male a = (1 + sqrt(16 C - 2)) / 4.
    from .decision.ocs import solve_ocs
    C = 0.14
    oc = solve_ocs(np.array([1.0, 0.0, 0.0, 0.0]), np.eye(4),
                   np.array([True, True, False, False]), np.zeros(4), np.full(4, 0.5), C)
    a = (1 + np.sqrt(16 * C - 2)) / 4
    check("T09 OCS closed form a = (1 + sqrt(16C - 2))/4",
          _close(oc.c, [a, 0.5 - a, 0.25, 0.25], atol=1e-9) and _close(oc.coancestry, C, atol=1e-12))

    # T10 MCMC diagnostics against reference values of an independent implementation
    # (ArviZ 0.23.4) for an RNG-free chain set (see tests/test_mcmc_diagnostics.py)
    from .solvers.mcmc_diagnostics import bulk_ess, rhat, tail_ess
    tt = np.arange(401)
    xm = np.zeros((4, 401))
    for c in range(4):
        e = ((tt * 0.6180339887498949 + (c + 1) * 0.41421356237309515) % 1.0) - 0.5
        e = e + 0.3 * np.sin(1.7 * tt + c)
        v = 0.0
        for k in range(401):
            v = 0.7 * v + e[k]
            xm[c, k] = v
    check("T10 R-hat, bulk and tail ESS = reference values",
          _close([rhat(xm), bulk_ess(xm), tail_ess(xm)],
                 [0.9977880860052293, 1507.7710457404758, 1557.687953082816], atol=0, rtol=1e-9))

    # T11 compiled Bayesian marker sweep equals the Python reference (if compiled)
    from .core import pedigree as pmod
    if native_kernel_available() and hasattr(pmod._native, "bayes_sweep"):
        from .solvers.bayes import sweep_python
        k = np.arange(1, 7, dtype=float)
        Wb = np.asfortranarray(np.sin(np.outer(np.arange(1, 9), k)))
        wtw = np.einsum("ij,ij->j", Wb, Wb)
        outs = []
        for native in (False, True):
            e0 = np.cos(np.arange(8.0))
            beta, delta = np.zeros(6), np.zeros(6, dtype=np.int64)
            args = (wtw, e0, beta, delta, np.full(6, 0.3), np.log([0.6, 0.4]), np.array([0.0, 0.3]),
                    1.1, 1, np.linspace(-1, 1, 6), np.linspace(0.1, 0.9, 6))
            if native:
                pmod._native.bayes_sweep(Wb.T, *args)
            else:
                sweep_python(Wb, *args)
            outs.append((e0, beta, delta))
        check("T11 native Bayesian sweep = Python reference",
              _close(outs[1][0], outs[0][0], atol=1e-12) and _close(outs[1][1], outs[0][1], atol=1e-12)
              and bool(np.array_equal(outs[1][2], outs[0][2])))
    else:
        lines.append("  [SKIP] T11 native Bayesian sweep (native kernel not available or disabled)")
    lines.append("RESULT: " + ("PASS" if ok_all else "FAIL"))
    return ok_all, lines
