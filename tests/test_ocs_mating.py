"""OCS against an independent optimizer and a closed form; mating plans
against exhaustive enumeration; infeasibility is reported, never relaxed."""

import itertools

import numpy as np
import pytest
from scipy.optimize import linprog, minimize

from abp.core.pedigree import Pedigree
from abp.decision.mating import allocate, forbidden_mask
from abp.decision.ocs import integer_matings, max_merit_lp, solve_ocs, solve_qp
from abp.errors import ABPError


def _candidates(seed, n_m=5, n_f=9):
    rng = np.random.default_rng(seed)
    n0 = 6
    ids = [f"p{i}" for i in range(n0)]
    sires, dams = [None] * n0, [None] * n0
    for i in range(n0, n0 + n_m + n_f):
        s, d = rng.choice(np.arange(0, i), 2, replace=False)
        ids.append(f"p{i}")
        sires.append(ids[s])
        dams.append(ids[d])
    ped = Pedigree.from_parent_ids(ids, sires, dams)
    cand = ids[n0:]
    A = ped.a_submatrix(ped.index_of(cand))
    male = np.array([True] * n_m + [False] * n_f)
    g = rng.normal(0, 1, n_m + n_f)
    return A, male, g


def _reference(g, A, male, lo, hi, cmax):
    """Independent optimum from SciPy trust-constr on the original QCQP."""
    from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint
    K = np.vstack([male, ~male]).astype(float)
    x0 = np.clip(np.where(male, 0.5 / male.sum(), 0.5 / (~male).sum()), lo, hi)
    r = minimize(lambda c: -c @ g, x0, jac=lambda c: -g, hess=lambda c: np.zeros((g.size,) * 2),
                 method="trust-constr",
                 constraints=[LinearConstraint(K, [0.5, 0.5], [0.5, 0.5]),
                              NonlinearConstraint(lambda c: c @ A @ c / 2, -np.inf, cmax,
                                                  jac=lambda c: A @ c, hess=lambda c, v: v[0] * A)],
                 bounds=Bounds(lo, hi), options={"gtol": 1e-12, "xtol": 1e-14, "maxiter": 20000})
    assert r.constr_violation < 1e-9, r.message
    return r.x


def _independent_kkt(g, A, male, lo, hi, c, mu, tol=1e-9):
    """Recompute multipliers from c alone and return the worst KKT violation."""
    grad = mu * (A @ c) - g
    worst = 0.0
    for sex in (True, False):
        idx = np.flatnonzero(male == sex)
        free = idx[(c[idx] > lo[idx] + tol) & (c[idx] < hi[idx] - tol)]
        assert free.size, "test expects at least one free candidate per sex"
        eta = -grad[free].mean()
        worst = max(worst, float(np.max(np.abs(grad[free] + eta))))
        at_lo = idx[c[idx] <= lo[idx] + tol]
        at_hi = idx[c[idx] >= hi[idx] - tol]
        if at_lo.size:
            worst = max(worst, float(max(0.0, -np.min(grad[at_lo] + eta))))
        if at_hi.size:
            worst = max(worst, float(max(0.0, np.max(grad[at_hi] + eta))))
    return worst


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_ocs_matches_independent_optimizer_and_kkt(seed):
    A, male, g = _candidates(seed)
    n = g.size
    lo, hi = np.zeros(n), np.full(n, 0.25)
    res0 = solve_ocs(g, A, male, lo, hi, 10.0)
    cmax = 0.5 * (res0.min_coancestry + res0.max_merit_coancestry)  # active constraint
    res = solve_ocs(g, A, male, lo, hi, cmax)
    assert res.status == "constraint_active"
    ref = _reference(g, A, male, lo, hi, cmax)
    assert res.merit >= ref @ g - 1e-9                 # at least as good as the reference
    assert res.merit == pytest.approx(ref @ g, abs=1e-6)
    np.testing.assert_allclose(res.c, ref, atol=1e-4)
    assert _independent_kkt(g, A, male, lo, hi, res.c, res.penalty_mu) < 1e-8
    k = res.kkt
    assert k["stationarity_max_abs"] < 1e-8 and k["sex_sum_residual"] < 1e-12
    assert k["bound_violation"] < 1e-15 and abs(k["coancestry_slack"]) < 1e-10
    assert k["min_bound_multiplier"] > -1e-9


def test_unconstrained_case_equals_linear_programme():
    A, male, g = _candidates(4)
    n = g.size
    lo, hi = np.zeros(n), np.full(n, 0.2)
    res = solve_ocs(g, A, male, lo, hi, 10.0)
    assert res.status == "constraint_inactive" and res.penalty_mu == 0.0
    K = np.vstack([male, ~male]).astype(float)
    lp = linprog(-g, A_eq=K, b_eq=[0.5, 0.5], bounds=list(zip(lo, hi)), method="highs")
    assert res.merit == pytest.approx(-lp.fun, abs=1e-12)
    assert res.kkt["min_bound_multiplier"] > -1e-12 and res.kkt["stationarity_max_abs"] < 1e-12


def test_closed_form_for_unrelated_candidates():
    """A = I, no binding bounds: c_i = 1/(2 n_s) + (g_i - mean_s g)/mu with
    mu = sqrt(sum (g - mean_s g)^2 / (2 C - sum_s 1/(4 n_s)))."""
    rng = np.random.default_rng(7)
    male = np.array([True] * 6 + [False] * 10)
    g = rng.normal(0, 0.02, 16)
    A = np.eye(16)
    C = 0.5 * (0.25 / 6 + 0.25 / 10) * 1.05       # close to the minimum: interior solution
    dev = np.where(male, g - g[male].mean(), g - g[~male].mean())
    mu = np.sqrt(np.sum(dev**2) / (2 * C - 0.25 / 6 - 0.25 / 10))
    c_ref = np.where(male, 0.5 / 6, 0.5 / 10) + dev / mu
    assert np.all(c_ref > 0)
    res = solve_ocs(g, A, male, np.zeros(16), np.ones(16), C)
    np.testing.assert_allclose(res.c, c_ref, atol=1e-10)
    assert res.penalty_mu == pytest.approx(mu, rel=1e-9)


def test_infeasible_ceiling_reports_minimum():
    A, male, g = _candidates(5)
    n = g.size
    lo, hi = np.zeros(n), np.full(n, 0.5)
    cmin = solve_qp(A, np.zeros(n), male, lo, hi).x
    cmin_val = cmin @ A @ cmin / 2
    ref = minimize(lambda c: c @ A @ c / 2, np.where(male, 0.1, 0.5 / 9), method="SLSQP",
                   constraints=[{"type": "eq", "fun": lambda c: c[male].sum() - 0.5},
                                {"type": "eq", "fun": lambda c: c[~male].sum() - 0.5}],
                   bounds=list(zip(lo, hi)), options={"ftol": 1e-15})
    assert cmin_val == pytest.approx(ref.fun, abs=1e-9)
    with pytest.raises(ABPError) as exc:
        solve_ocs(g, A, male, lo, hi, cmin_val * 0.99)
    assert exc.value.details["min_achievable"] == pytest.approx(cmin_val, rel=1e-9)


def test_tighter_ceiling_never_increases_merit():
    A, male, g = _candidates(6)
    n = g.size
    lo, hi = np.zeros(n), np.full(n, 0.5)
    base = solve_ocs(g, A, male, lo, hi, 10.0)
    merits = [solve_ocs(g, A, male, lo, hi, base.min_coancestry + f *
                        (base.max_merit_coancestry - base.min_coancestry)).merit
              for f in (0.9, 0.6, 0.3, 0.05)]
    assert all(a >= b - 1e-12 for a, b in zip(merits, merits[1:]))


def test_bounds_infeasible_for_a_sex():
    A, male, g = _candidates(8)
    n = g.size
    with pytest.raises(ABPError) as exc:
        solve_ocs(g, A, male, np.zeros(n), np.full(n, 0.05), 1.0)   # 5 males x 0.05 < 0.5
    assert exc.value.details["sex"] == "M"


def test_integer_matings_respect_capacity_and_totals():
    male = np.array([True, True, True, False, False, False, False])
    c = np.array([0.3, 0.15, 0.05, 0.125, 0.125, 0.125, 0.125])
    cap = np.array([10, 10, 10, 1, 1, 1, 1])
    n = integer_matings(c, cap, male, 4)
    assert n[male].sum() == 4 and n[~male].sum() == 4 and np.all(n <= cap)
    assert n.tolist() == [2, 1, 1, 1, 1, 1, 1] or n.tolist()[:3] == [3, 1, 0]


def _enumerate_best(n_s, m_d, F, forb):
    """Exhaustive search over all 0/1 plans with the required counts."""
    S, D = F.shape
    cells = [(s, d) for s in range(S) for d in range(D) if not forb[s, d]]
    best = None
    N = int(n_s.sum())
    for combo in itertools.combinations(cells, N):
        us = np.bincount([s for s, _ in combo], minlength=S)
        ud = np.bincount([d for _, d in combo], minlength=D)
        if np.array_equal(us, n_s) and np.array_equal(ud, m_d):
            val = sum(F[s, d] for s, d in combo)
            if best is None or val < best - 1e-15:
                best = val
    return best


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_mating_plan_is_optimal_by_enumeration(seed):
    rng = np.random.default_rng(seed)
    S, D = 3, 5
    A_sd = np.round(rng.uniform(0, 0.5, (S, D)), 3)
    n_s = np.array([2, 2, 1])
    m_d = np.ones(D, dtype=int)
    forb, reasons = forbidden_mask(A_sd, 0.45, None, None, None)
    best = _enumerate_best(n_s, m_d, A_sd / 2, forb)
    if best is None:
        with pytest.raises(ABPError):
            allocate(n_s, m_d, A_sd, forb)
        return
    plan = allocate(n_s, m_d, A_sd, forb)
    assert plan.total_inbreeding == pytest.approx(best, abs=1e-12)
    assert all(plan.checks[k] for k in ("sire_counts_match", "dam_counts_match",
                                        "no_forbidden_pair", "no_repeated_pair"))


def test_carrier_risk_forbids_only_risky_pairs():
    A_sd = np.zeros((2, 2))
    carrier_s = np.array([1.0, 0.0])      # sire 0 a known carrier
    carrier_d = np.array([0.5, 0.0])      # dam 0 possibly a carrier
    forb, reasons = forbidden_mask(A_sd, None, carrier_s, carrier_d, 0.01)
    assert forb.tolist() == [[True, False], [False, False]] and reasons["recessive_risk"] == 1
    plan = allocate(np.array([1, 1]), np.array([1, 1]), A_sd, forb, carrier_s, carrier_d)
    assert (0, 0) not in plan.pairs and plan.affected_risk.max() == 0.0


def test_infeasible_plan_names_unmatchable_dam():
    A_sd = np.array([[0.1, 0.6], [0.2, 0.7]])
    forb, _ = forbidden_mask(A_sd, 0.5, None, None, None)     # dam 1 related to both sires
    with pytest.raises(ABPError) as exc:
        allocate(np.array([1, 1]), np.array([1, 1]), A_sd, forb)
    assert exc.value.details["unmet_dam_indices"] == [1]


def test_max_merit_lp_fills_best_candidates():
    g = np.array([3.0, 1.0, 2.0, 0.5, 4.0])
    male = np.array([True, True, True, False, False])
    c = max_merit_lp(g, male, np.zeros(5), np.array([0.3, 0.3, 0.3, 0.3, 0.3]))
    np.testing.assert_allclose(c, [0.3, 0.0, 0.2, 0.2, 0.3])


def test_integer_repair_meets_ceiling_and_is_bounded_by_continuous_optimum():
    from abp.decision.ocs import repair_integer_plan
    A, male, g = _candidates(11, n_m=4, n_f=8)
    n = g.size
    N = 8
    cap = np.where(male, 4, 1)
    hi = cap / (2.0 * N)
    base = solve_ocs(g, A, male, np.zeros(n), hi, 10.0)
    cmax = base.min_coancestry + 0.4 * (base.max_merit_coancestry - base.min_coancestry)
    res = solve_ocs(g, A, male, np.zeros(n), hi, cmax)
    counts, info = repair_integer_plan(integer_matings(res.c, cap, male, N), g, A, male, cap, N, cmax)
    c = counts / (2.0 * N)
    assert c @ A @ c / 2 <= cmax + 1e-15
    assert counts[male].sum() == N and counts[~male].sum() == N and np.all(counts <= cap)
    assert c @ g <= res.merit + 1e-12            # continuous optimum is an upper bound
    # exhaustive check over all whole-number sire allocations (dams: 8 of 8 used once)
    best = -np.inf
    for alloc in itertools.product(range(5), repeat=4):
        if sum(alloc) != N:
            continue
        x = np.concatenate([alloc, np.ones(8)])
        cx = x / (2.0 * N)
        if cx @ A @ cx / 2 <= cmax + 1e-15:
            best = max(best, cx @ g)
    assert c @ g <= best + 1e-12 and c @ g >= best - 0.05 * abs(best)


def test_example09_plan_satisfies_every_hard_constraint(tmp_path):
    import csv
    from pathlib import Path
    from abp.cli import main
    root = Path(__file__).resolve().parents[1]
    assert main(["mate", str(root / "examples" / "09_sheep_mating" / "mating.toml"),
                 "--out", str(tmp_path / "m")]) == 0
    import json
    summ = json.loads((tmp_path / "m" / "mating_summary.json").read_text(encoding="utf-8"))
    with open(root / "examples" / "09_sheep_mating" / "candidates.csv", encoding="utf-8") as fh:
        cand = {r["id"]: r for r in csv.DictReader(fh)}
    with open(tmp_path / "m" / "mating_plan.csv", encoding="utf-8") as fh:
        plan = list(csv.DictReader(fh))
    assert len(plan) == 150
    use = {}
    for r in plan:
        assert float(r["relationship_A_sd"]) <= 0.25
        risk = float(cand[r["sire"]]["carrier"]) * float(cand[r["dam"]]["carrier"]) / 4
        assert risk <= 0.01 + 1e-15 and abs(risk - float(r["affected_risk"])) < 1e-12
        assert cand[r["sire"]]["sex"] == "M" and cand[r["dam"]]["sex"] == "F"
        for a in (r["sire"], r["dam"]):
            use[a] = use.get(a, 0) + 1
    assert all(use[a] <= int(cand[a]["capacity"]) for a in use)
    assert len({(r["sire"], r["dam"]) for r in plan}) == 150
    assert summ["integer_plan"]["ceiling_satisfied"]
    assert summ["integer_plan"]["coancestry_realized"] <= summ["coancestry_target"]["C_max"]
    assert summ["ocs"]["kkt_certificate"]["stationarity_max_abs"] < 1e-8
