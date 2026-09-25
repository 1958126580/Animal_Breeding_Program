"""BLUP/PEV against hand-derived values and the independent V-form reference."""

import numpy as np
import pytest
import scipy.sparse as sp

from abp.core.design import FixedTerm, build_fixed_design
from abp.core.pedigree import Pedigree
from abp.errors import ABPError
from abp.solvers.blup import RandomTerm, blup
from tests.reference.dense_reference import blup_v_form, tabular_a

ATOL, RTOL = 1e-10, 1e-8


def animal_term(ped: Pedigree, record_ids):
    n = len(record_ids)
    cols = ped.index_of(record_ids)
    Z = sp.csr_matrix((np.ones(n), (np.arange(n), cols)), shape=(n, ped.n))
    return RandomTerm("animal", Z, ped.ainv(), ped.ids, genetic=True,
                      k_diag=1.0 + ped.inbreeding(), logdet_k=ped.logdet_a())


def iid_term(name, levels_per_record):
    levels = sorted(set(levels_per_record))
    pos = {lv: i for i, lv in enumerate(levels)}
    n = len(levels_per_record)
    Z = sp.csr_matrix((np.ones(n), (np.arange(n), [pos[v] for v in levels_per_record])),
                      shape=(n, len(levels)))
    return RandomTerm(name, Z, sp.identity(len(levels), format="csr"), levels, genetic=False,
                      k_diag=np.ones(len(levels)), logdet_k=0.0)


def test_t04_hand_derived():
    """Spec T04: two unrelated animals, unknown intercept, both variances 1."""
    ped = Pedigree.from_parent_ids(["A", "B"], [None, None], [None, None])
    X = build_fixed_design({}, [], intercept=True, n=2).X
    res = blup(np.array([2.0, 4.0]), X, [animal_term(ped, ["A", "B"])],
               {"animal": 1.0, "residual": 1.0}, method="dense")
    t = res.terms["animal"]
    idx = ped.index_of(["A", "B"])
    np.testing.assert_allclose(res.fixed_solution, [3.0], atol=ATOL)
    np.testing.assert_allclose(t.solution[idx], [-0.5, 0.5], atol=ATOL)
    np.testing.assert_allclose(t.pev[idx], [0.75, 0.75], atol=ATOL)
    np.testing.assert_allclose(t.reliability[idx], [0.25, 0.25], atol=ATOL)


MRODE_31 = {  # Mrode (2005, 2nd ed.) Example 3.1: pre-weaning gain (kg)
    "ped": [("1", None, None), ("2", None, None), ("3", None, None), ("4", "1", None),
            ("5", "3", "2"), ("6", "1", "2"), ("7", "4", "5"), ("8", "3", "6")],
    "records": [("4", "M", 4.5), ("5", "F", 2.9), ("6", "F", 3.9), ("7", "M", 3.5), ("8", "M", 5.0)],
    "sigma_a2": 20.0, "sigma_e2": 40.0,
    # printed solutions (3 decimals)
    "sex": {"F": 3.404, "M": 4.358},
    "ebv": {"1": 0.098, "2": -0.019, "3": -0.041, "4": -0.009, "5": -0.186, "6": 0.177,
            "7": -0.249, "8": 0.183},
}


def _mrode_setup():
    ids, sires, dams = zip(*MRODE_31["ped"])
    ped = Pedigree.from_parent_ids(ids, sires, dams)
    rec_ids = [r[0] for r in MRODE_31["records"]]
    sex = [r[1] for r in MRODE_31["records"]]
    y = np.array([r[2] for r in MRODE_31["records"]])
    fd = build_fixed_design({"sex": sex}, [FixedTerm("sex", "factor")], intercept=False, n=5)
    return ped, ids, sires, dams, rec_ids, sex, y, fd


@pytest.mark.parametrize("method", ["dense", "sparse_direct", "pcg"])
def test_mrode_example_3_1(method):
    ped, ids, sires, dams, rec_ids, sex, y, fd = _mrode_setup()
    va, ve = MRODE_31["sigma_a2"], MRODE_31["sigma_e2"]
    res = blup(y, fd.X, [animal_term(ped, rec_ids)], {"animal": va, "residual": ve},
               method=method, compute_pev=(method != "pcg"), tol=1e-13)
    # (1) independent V-form reference, full precision
    A = tabular_a(list(ids), list(sires), list(dams))
    order = {a: k for k, a in enumerate(ids)}
    Z = np.zeros((5, 8))
    for r, a in enumerate(rec_ids):
        Z[r, order[a]] = 1
    Xd = fd.X.toarray()
    b_ref, u_ref, pev_ref = blup_v_form(y, Xd, Z, va * A, ve * np.eye(5))
    u = res.terms["animal"].solution[ped.index_of(ids)]
    np.testing.assert_allclose(res.fixed_solution, b_ref, atol=1e-9)
    np.testing.assert_allclose(u, u_ref, atol=1e-9)
    if method != "pcg":
        np.testing.assert_allclose(res.terms["animal"].pev[ped.index_of(ids)],
                                   np.diag(pev_ref), atol=1e-9)
    # (2) printed textbook values (rounded to 3 decimals)
    labels = [lab[1] for lab, k in zip(fd.labels, fd.kept) if k]
    for lvl, val in MRODE_31["sex"].items():
        assert res.fixed_solution[labels.index(lvl)] == pytest.approx(val, abs=6e-4)
    for a, val in MRODE_31["ebv"].items():
        assert u[order[a]] == pytest.approx(val, abs=6e-4)


def _random_problem(seed):
    rng = np.random.default_rng(seed)
    n_anim = 60
    ids = [f"x{i}" for i in range(n_anim)]
    sires, dams = [None] * n_anim, [None] * n_anim
    for i in range(8, n_anim):
        s, d = rng.choice(np.arange(max(0, i - 25), i), 2, replace=False)
        sires[i], dams[i] = ids[s], ids[d]
    rec_animals = [ids[k] for k in rng.integers(10, n_anim, size=90)]  # repeated records
    herd = [f"h{k}" for k in rng.integers(0, 4, size=90)]
    season = [f"s{k}" for k in rng.integers(0, 3, size=90)]
    age = rng.normal(3, 1, size=90)
    y = rng.normal(10, 3, size=90)
    return ids, sires, dams, rec_animals, herd, season, age, y


@pytest.mark.parametrize("seed", [4, 5])
def test_repeatability_model_matches_v_form(seed):
    """Rank-deficient X (intercept + 2 factors + covariate), animal + PE effects."""
    ids, sires, dams, rec_animals, herd, season, age, y = _random_problem(seed)
    ped = Pedigree.from_parent_ids(ids, sires, dams)
    cols = {"herd": herd, "season": season, "age": age}
    terms_f = [FixedTerm("herd", "factor"), FixedTerm("season", "factor"),
               FixedTerm("age", "covariate")]
    fd = build_fixed_design(cols, terms_f, intercept=True, n=90)
    assert fd.rank == 1 + 4 + 3 + 1 - 2  # intercept+herd+season+age minus 2 dependencies
    va, vpe, ve = 2.0, 0.7, 5.0
    pe = iid_term("pe", rec_animals)
    res = blup(y, fd.X, [animal_term(ped, rec_animals), pe],
               {"animal": va, "pe": vpe, "residual": ve}, method="dense")
    # Independent reference with a generalized inverse on the *full* design.
    A = tabular_a(ids, sires, dams)
    pos = {a: k for k, a in enumerate(ids)}
    Za = np.zeros((90, 60))
    Za[np.arange(90), [pos[a] for a in rec_animals]] = 1
    Zp = pe.Z.toarray()
    Xfull = np.column_stack([np.ones(90),
                             *[np.array(herd) == h for h in sorted(set(herd))],
                             *[np.array(season) == s for s in sorted(set(season))], age])
    Zall = np.hstack([Za, Zp])
    Gall = np.block([[va * A, np.zeros((60, Zp.shape[1]))],
                     [np.zeros((Zp.shape[1], 60)), vpe * np.eye(Zp.shape[1])]])
    V = Zall @ Gall @ Zall.T + ve * np.eye(90)
    Vi = np.linalg.inv(V)
    XtViX_g = np.linalg.pinv(Xfull.T @ Vi @ Xfull)
    P = Vi - Vi @ Xfull @ XtViX_g @ Xfull.T @ Vi
    u_ref = Gall @ Zall.T @ P @ y
    pev_ref = np.diag(Gall - Gall @ Zall.T @ P @ Zall @ Gall)
    idx = ped.index_of(ids)
    u = np.concatenate([res.terms["animal"].solution[idx], res.terms["pe"].solution])
    pev = np.concatenate([res.terms["animal"].pev[idx], res.terms["pe"].pev])
    np.testing.assert_allclose(u, u_ref, atol=1e-9, rtol=1e-8)
    np.testing.assert_allclose(pev, pev_ref, atol=1e-9, rtol=1e-8)
    # Estimable function X b is invariant: compare fitted fixed part.
    fitted_ref = Xfull @ XtViX_g @ Xfull.T @ Vi @ y
    np.testing.assert_allclose(fd.X @ res.fixed_solution, fitted_ref, atol=1e-9)


def test_invariance_to_constraint_choice_and_record_order():
    ids, sires, dams, rec_animals, herd, season, age, y = _random_problem(9)
    ped = Pedigree.from_parent_ids(ids, sires, dams)
    va, ve = 2.0, 5.0
    fd1 = build_fixed_design({"herd": herd, "season": season},
                             [FixedTerm("herd", "factor"), FixedTerm("season", "factor")],
                             intercept=True, n=90)
    fd2 = build_fixed_design({"herd": herd, "season": season},
                             [FixedTerm("season", "factor"), FixedTerm("herd", "factor")],
                             intercept=False, n=90)
    assert fd1.constrained_labels != fd2.constrained_labels
    r1 = blup(y, fd1.X, [animal_term(ped, rec_animals)], {"animal": va, "residual": ve})
    r2 = blup(y, fd2.X, [animal_term(ped, rec_animals)], {"animal": va, "residual": ve})
    np.testing.assert_allclose(r1.terms["animal"].solution, r2.terms["animal"].solution, atol=1e-10)
    np.testing.assert_allclose(r1.terms["animal"].pev, r2.terms["animal"].pev, atol=1e-10)
    perm = np.random.default_rng(1).permutation(90)
    fd3 = build_fixed_design({"herd": [herd[k] for k in perm], "season": [season[k] for k in perm]},
                             [FixedTerm("herd", "factor"), FixedTerm("season", "factor")],
                             intercept=True, n=90)
    r3 = blup(y[perm], fd3.X, [animal_term(ped, [rec_animals[k] for k in perm])],
              {"animal": va, "residual": ve})
    np.testing.assert_allclose(r3.terms["animal"].solution, r1.terms["animal"].solution, atol=1e-10)


def test_unit_change_kg_to_g():
    """Spec property 3: y x1000 and variances x1e6 -> EBV x1000, PEV x1e6, same reliability."""
    ids, sires, dams, rec_animals, herd, season, age, y = _random_problem(3)
    ped = Pedigree.from_parent_ids(ids, sires, dams)
    fd = build_fixed_design({"herd": herd}, [FixedTerm("herd", "factor")], True, 90)
    r_kg = blup(y, fd.X, [animal_term(ped, rec_animals)], {"animal": 2.0, "residual": 5.0})
    r_g = blup(1000 * y, fd.X, [animal_term(ped, rec_animals)], {"animal": 2e6, "residual": 5e6})
    a, b = r_kg.terms["animal"], r_g.terms["animal"]
    np.testing.assert_allclose(b.solution, 1000 * a.solution, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(b.pev, 1e6 * a.pev, rtol=1e-9)
    np.testing.assert_allclose(b.reliability, a.reliability, atol=1e-12)


def test_solvers_agree_on_larger_problem():
    rng = np.random.default_rng(12)
    n = 1500
    ids = [f"a{i}" for i in range(n)]
    sires = [None] * 50 + [ids[k] for k in rng.integers(0, 25, n - 50)]
    dams = [None] * 50 + [ids[k] for k in rng.integers(25, 50, n - 50)]
    ped = Pedigree.from_parent_ids(ids, sires, dams)
    rec = ids[50:]
    grp = [f"g{k}" for k in rng.integers(0, 30, len(rec))]
    y = rng.normal(size=len(rec))
    fd = build_fixed_design({"g": grp}, [FixedTerm("g", "factor")], True, len(rec))
    kw = dict(variances={"animal": 1.0, "residual": 3.0}, compute_pev=False)
    sols = {m: blup(y, fd.X, [animal_term(ped, rec)], method=m, tol=1e-12, **kw)
            .terms["animal"].solution for m in ("dense", "sparse_direct", "pcg")}
    np.testing.assert_allclose(sols["sparse_direct"], sols["dense"], atol=1e-9)
    np.testing.assert_allclose(sols["pcg"], sols["dense"], atol=1e-8)


def test_invalid_variances_are_rejected():
    ped = Pedigree.from_parent_ids(["A", "B"], [None, None], [None, None])
    X = build_fixed_design({}, [], intercept=True, n=2).X
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ABPError) as exc:
            blup(np.array([1.0, 2.0]), X, [animal_term(ped, ["A", "B"])],
                 {"animal": bad, "residual": 1.0})
        assert exc.value.code == "ABP-E301"


def test_pcg_non_convergence_is_an_error():
    ids, sires, dams, rec_animals, herd, season, age, y = _random_problem(2)
    ped = Pedigree.from_parent_ids(ids, sires, dams)
    fd = build_fixed_design({"herd": herd}, [FixedTerm("herd", "factor")], True, 90)
    with pytest.raises(ABPError) as exc:
        blup(y, fd.X, [animal_term(ped, rec_animals)], {"animal": 2.0, "residual": 5.0},
             method="pcg", compute_pev=False, max_iter=3, tol=1e-12)
    assert exc.value.code == "ABP-E400"
