"""Optimal contribution selection (OCS) and integer contributions (module M12).

Model contract (method registry id ``dec.ocs``)
-----------------------------------------------
Candidates ``i = 1..n`` with merit ``g_i`` (EBV or index, target scale),
sex, and a relationship matrix ``A`` (declared base; positive definite).
Genetic contributions ``c`` (to the gene pool of the next generation):

    maximize   c'g
    subject to sum_{i male} c_i = 1/2,   sum_{i female} c_i = 1/2,
               l_i <= c_i <= u_i,
               c'Ac / 2 <= C_max                (group coancestry of the parents)

(Meuwissen 1997, J Anim Sci 75:934).  ``c'Ac/2`` is the expected inbreeding
of the progeny under random union of gametes *including selfing*; it is not
the inbreeding of any particular mating plan, which is computed later from
the actual pairs (``A_sd / 2``).

Algorithm
---------
For a penalty ``mu >= 0`` the problem ``max c'g - mu c'Ac/2`` over the linear
constraints is a strictly convex QP (A positive definite), solved exactly by
a primal active-set method.  The coancestry of the QP solution is
non-increasing in ``mu``; ``mu = 0`` (the linear programme, solved by a
greedy fill per sex) gives the maximum-merit solution, and the minimum
achievable coancestry is the QP with the objective ``c'Ac/2`` alone.  If
``C_max`` lies between the two, ``mu`` is found by bisection so that the
coancestry constraint holds with equality (complementary slackness).  The
output carries a KKT certificate: primal residuals, bound multipliers and the
stationarity residual.  If ``C_max`` is below the minimum achievable
coancestry the problem is infeasible and ABP reports that minimum - the
constraint is never relaxed silently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla

from ..errors import ABPError


@dataclass
class QPSolution:
    x: np.ndarray
    eq_multipliers: np.ndarray       # for the two sex-sum equalities
    lower_multipliers: np.ndarray    # >= 0 where x = l
    upper_multipliers: np.ndarray    # >= 0 where x = u
    iterations: int


def _feasible_start(male: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """A point with the sex sums at 1/2 inside the bounds (water filling)."""
    x = lo.copy()
    for sex in (True, False):
        idx = np.flatnonzero(male == sex)
        need = 0.5 - lo[idx].sum()
        room = hi[idx] - lo[idx]
        if need < -1e-15 or need > room.sum() + 1e-15:
            raise ABPError("MODEL_NOT_IDENTIFIABLE",
                           f"{'male' if sex else 'female'} contributions cannot sum to 1/2 within "
                           f"the bounds (sum of lower bounds {lo[idx].sum():.6g}, "
                           f"sum of upper bounds {hi[idx].sum():.6g})",
                           sex="M" if sex else "F")
        x[idx] += room * (need / room.sum() if room.sum() > 0 else 0.0)
    return x


def solve_qp(H: np.ndarray, f: np.ndarray, male: np.ndarray, lo: np.ndarray, hi: np.ndarray,
             x0: np.ndarray | None = None, tol: float = 1e-12, max_iter: int = 100000) -> QPSolution:
    """Minimize ``x'Hx/2 - f'x`` s.t. sex sums = 1/2 and ``lo <= x <= hi`` (H PD).

    Primal active-set method (Nocedal & Wright 2006, Algorithm 16.3) with the
    two equality constraints always active and a working set of bounds.
    """
    n = f.size
    K = np.vstack([male.astype(float), (~male).astype(float)])
    x = _feasible_start(male, lo, hi) if x0 is None else np.clip(x0, lo, hi)
    if x0 is not None and np.max(np.abs(K @ x - 0.5)) > 1e-12:
        x = _feasible_start(male, lo, hi)
    at_lo = np.isclose(x, lo, atol=1e-15, rtol=0) & (lo < hi)
    at_hi = np.isclose(x, hi, atol=1e-15, rtol=0) & (lo < hi) & ~at_lo
    fixed_eq = lo >= hi  # degenerate bounds: always fixed
    for it in range(1, max_iter + 1):
        W = at_lo | at_hi | fixed_eq
        F = np.flatnonzero(~W)
        grad = H @ x - f
        # equality-constrained step on free variables: min p'Hp/2 + grad'p, K_F p_F = 0
        m = F.size
        Kf = K[:, F]
        rows = [r for r in range(2) if Kf[r].any()]
        Kf = Kf[rows]
        M = np.zeros((m + len(rows), m + len(rows)))
        M[:m, :m] = H[np.ix_(F, F)]
        M[:m, m:] = Kf.T
        M[m:, :m] = Kf
        rhs = np.concatenate([-grad[F], np.zeros(len(rows))])
        sol = sla.solve(M, rhs, assume_a="sym") if m + len(rows) else np.zeros(0)
        p = np.zeros(n)
        p[F] = sol[:m]
        if np.max(np.abs(p), initial=0.0) <= tol * max(1.0, np.max(np.abs(x))):
            # KKT on the working set:  grad + K'eta = nu_lo - nu_hi  (nu >= 0).
            # For a sex with free variables, eta comes from the KKT system
            # (H p + K_F'lambda = -grad_F with p = 0  =>  eta = lambda).
            eta = np.zeros(2)
            for r, v in zip(rows, sol[m:]):
                eta[r] = v
            for r in range(2):
                if r in rows:
                    continue
                # every variable of this sex is at a bound: any eta in
                # [max_lo(-grad), min_hi(-grad)] is valid; take the midpoint
                sex_mask = K[r] > 0
                a = -grad[sex_mask & at_lo]
                b = -grad[sex_mask & at_hi]
                lo_e = a.max() if a.size else (b.min() if b.size else 0.0)
                hi_e = b.min() if b.size else lo_e
                eta[r] = 0.5 * (lo_e + hi_e)
            resid = grad + K.T @ eta
            nu_lo = np.where(at_lo, resid, 0.0)
            nu_hi = np.where(at_hi, -resid, 0.0)
            worst = None
            for arr, mask, side in ((nu_lo, at_lo, "lo"), (nu_hi, at_hi, "hi")):
                if mask.any():
                    j = int(np.argmin(np.where(mask, arr, np.inf)))
                    if arr[j] < -tol and (worst is None or arr[j] < worst[0]):
                        worst = (arr[j], j, side)
            if worst is None:
                return QPSolution(x, eta, nu_lo, nu_hi, it)
            _, j, side = worst
            if side == "lo":
                at_lo[j] = False
            else:
                at_hi[j] = False
            continue
        # step length to the nearest blocking bound
        alpha, block, bside = 1.0, -1, ""
        for j in F:
            if p[j] < 0 and x[j] + p[j] < lo[j]:
                a = (lo[j] - x[j]) / p[j]
                if a < alpha:
                    alpha, block, bside = a, j, "lo"
            elif p[j] > 0 and x[j] + p[j] > hi[j]:
                a = (hi[j] - x[j]) / p[j]
                if a < alpha:
                    alpha, block, bside = a, j, "hi"
        x = x + alpha * p
        if block >= 0:
            if bside == "lo":
                x[block] = lo[block]
                at_lo[block] = True
            else:
                x[block] = hi[block]
                at_hi[block] = True
    raise ABPError("SOLVER_NOT_CONVERGED", "OCS active-set QP did not converge",
                   iterations=max_iter)


def max_merit_lp(g: np.ndarray, male: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Exact solution for mu = 0: per sex, fill the best candidates up to their bounds."""
    x = lo.copy()
    _feasible_start(male, lo, hi)  # feasibility check
    for sex in (True, False):
        idx = np.flatnonzero(male == sex)
        need = 0.5 - lo[idx].sum()
        for i in idx[np.argsort(-g[idx], kind="stable")]:
            take = min(hi[i] - lo[i], need)
            x[i] += take
            need -= take
            if need <= 1e-16:
                break
    return x


@dataclass
class OCSResult:
    c: np.ndarray
    merit: float                     # c'g
    coancestry: float                # c'Ac/2
    penalty_mu: float
    status: str                      # "constraint_active" | "constraint_inactive"
    max_merit_coancestry: float      # coancestry of the unconstrained (mu = 0) solution
    min_coancestry: float            # smallest achievable coancestry
    kkt: dict
    bisection_steps: int


def _coancestry(c, A):
    return float(c @ A @ c) / 2.0


def solve_ocs(g: np.ndarray, A: np.ndarray, male: np.ndarray, lo: np.ndarray, hi: np.ndarray,
              max_coancestry: float) -> OCSResult:
    """Maximize merit subject to a group-coancestry ceiling (see module docstring)."""
    g = np.asarray(g, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    male = np.asarray(male, dtype=bool)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    n = g.size
    if A.shape != (n, n):
        raise ValueError("A does not match the candidates")
    if not male.any() or male.all():
        raise ABPError("MODEL_NOT_IDENTIFIABLE", "OCS needs male and female candidates")
    try:
        sla.cholesky(A, lower=True)
    except np.linalg.LinAlgError:
        raise ABPError("RELATIONSHIP_SINGULAR", "the candidates' relationship matrix is not "
                       "positive definite (duplicated or clonal candidates?)") from None
    if np.any(lo < 0) or np.any(hi < lo):
        raise ABPError("SPEC_INVALID", "contribution bounds must satisfy 0 <= lower <= upper")
    c0 = max_merit_lp(g, male, lo, hi)
    C0 = _coancestry(c0, A)
    cmin_sol = solve_qp(A, np.zeros(n), male, lo, hi)
    Cmin = _coancestry(cmin_sol.x, A)
    if max_coancestry < Cmin - 1e-12:
        raise ABPError("MODEL_NOT_IDENTIFIABLE",
                       f"coancestry ceiling {max_coancestry:.6g} is below the minimum achievable "
                       f"coancestry {Cmin:.6g} for these candidates and bounds; the ceiling is "
                       "not relaxed automatically", max_coancestry=max_coancestry,
                       min_achievable=Cmin)
    steps = 0
    if C0 <= max_coancestry:
        sol_c, mu, status = c0, 0.0, "constraint_inactive"
        qp = None
    else:
        lo_mu, hi_mu = 0.0, 1.0
        qp = solve_qp(hi_mu * A, g, male, lo, hi)
        while _coancestry(qp.x, A) > max_coancestry:
            lo_mu, hi_mu = hi_mu, hi_mu * 4.0
            qp = solve_qp(hi_mu * A, g, male, lo, hi, x0=qp.x)
            steps += 1
            if hi_mu > 1e18:
                raise ABPError("SOLVER_NOT_CONVERGED", "penalty bracket for OCS not found")
        best = qp
        while hi_mu - lo_mu > 1e-13 * hi_mu and steps < 400:
            mid = 0.5 * (lo_mu + hi_mu)
            q = solve_qp(mid * A, g, male, lo, hi, x0=best.x)
            steps += 1
            if _coancestry(q.x, A) > max_coancestry:
                lo_mu = mid
            else:
                hi_mu, best = mid, q
        qp, mu, sol_c, status = best, hi_mu, best.x, "constraint_active"
    kkt = _kkt(g, A, male, lo, hi, sol_c, mu, qp, max_coancestry)
    return OCSResult(sol_c, float(sol_c @ g), _coancestry(sol_c, A), mu, status, C0, Cmin, kkt,
                     steps)


def _kkt(g, A, male, lo, hi, c, mu, qp: QPSolution | None, cmax) -> dict:
    """KKT certificate of ``min -g'c + mu c'Ac/2`` over the linear constraints.

    Stationarity:  mu A c - g + K'eta - nu_lo + nu_hi = 0,  nu >= 0, with
    nu_lo (nu_hi) non-zero only where c is at its lower (upper) bound.
    """
    K = np.vstack([male.astype(float), (~male).astype(float)])
    if qp is None:  # mu = 0: multipliers of the greedy LP solution
        eta = np.zeros(2)
        for r, sex in enumerate((True, False)):
            idx = np.flatnonzero(male == sex)
            free = idx[(c[idx] > lo[idx] + 1e-15) & (c[idx] < hi[idx] - 1e-15)]
            full = idx[np.isclose(c[idx], hi[idx], atol=1e-15) & (hi[idx] > lo[idx])]
            empty = idx[np.isclose(c[idx], lo[idx], atol=1e-15) & (hi[idx] > lo[idx])]
            if free.size:
                eta[r] = float(g[free].max())
            elif full.size:
                eta[r] = float(g[full].min())
            else:
                eta[r] = float(g[empty].max()) if empty.size else 0.0
        resid = -g + K.T @ eta
        nu_lo = np.where(np.isclose(c, lo, atol=1e-15), resid, 0.0)
        nu_hi = np.where(np.isclose(c, hi, atol=1e-15) & ~np.isclose(c, lo, atol=1e-15), -resid, 0.0)
    else:
        eta, nu_lo, nu_hi = qp.eq_multipliers, qp.lower_multipliers, qp.upper_multipliers
    stat = mu * (A @ c) - g + K.T @ eta - nu_lo + nu_hi
    coan = float(c @ A @ c) / 2.0
    return {
        "stationarity_max_abs": float(np.max(np.abs(stat))),
        "sex_sum_residual": float(np.max(np.abs(K @ c - 0.5))),
        "bound_violation": float(max(np.max(lo - c), np.max(c - hi), 0.0)),
        "coancestry_slack": float(cmax - coan),
        "complementary_slackness": float(mu * (cmax - coan)),
        "min_bound_multiplier": float(min(nu_lo.min(), nu_hi.min())),
        "penalty_mu": float(mu),
    }


def integer_matings(c: np.ndarray, capacity: np.ndarray, male: np.ndarray, n_matings: int) -> np.ndarray:
    """Round contributions to mating counts per sex (largest remainder).

    Each mating gives 1/(2N) contribution to its sire and to its dam, so the
    ideal count is ``2 N c_i``.  Counts respect ``capacity_i`` and sum to N per
    sex; the realized contributions are ``counts / (2N)``.
    """
    out = np.zeros(c.size, dtype=np.int64)
    for sex in (True, False):
        idx = np.flatnonzero(male == sex)
        ideal = 2.0 * n_matings * c[idx]
        base = np.minimum(np.floor(ideal + 1e-9).astype(np.int64), capacity[idx])
        rest = n_matings - int(base.sum())
        frac = ideal - base
        order = np.argsort(-frac, kind="stable")
        for k in order:
            if rest <= 0:
                break
            if base[k] < capacity[idx][k]:
                base[k] += 1
                rest -= 1
        if rest != 0:
            raise ABPError("MODEL_NOT_IDENTIFIABLE",
                           f"cannot allocate {n_matings} matings to {'males' if sex else 'females'} "
                           "within their capacities", sex="M" if sex else "F")
        out[idx] = base
    return out


def repair_integer_plan(counts: np.ndarray, g: np.ndarray, A: np.ndarray, male: np.ndarray,
                        capacity: np.ndarray, n_matings: int, max_coancestry: float,
                        max_moves: int = 100000) -> tuple[np.ndarray, dict]:
    """Make whole-number mating counts satisfy the coancestry ceiling, then improve merit.

    A move transfers one mating from candidate i to candidate j of the same
    sex: ``delta = (e_j - e_i)/(2N)``, with exact changes

        merit:      (g_j - g_i) / (2N)
        coancestry: ((Ac)_j - (Ac)_i)/(2N) + (A_ii + A_jj - 2 A_ij)/(8 N^2).

    Phase 1 (repair), while the ceiling is exceeded: apply the move with the
    smallest merit loss per unit of coancestry reduction.  Phase 2 (improve):
    apply the move with the largest merit gain that keeps the ceiling.  The
    result is feasible and locally optimal for single moves; it is not a
    proven integer optimum.  The continuous optimum's merit is an upper bound,
    so the gap is reported.
    """
    x = counts.astype(np.int64).copy()
    two_n = 2.0 * n_matings
    diagA = np.diag(A)
    moves = {"repair": 0, "improve": 0}

    def best_move(c, Ac, phase, C):
        best = None
        for sex in (True, False):
            idx = np.flatnonzero(male == sex)
            src = idx[x[idx] > 0]
            dst = idx[x[idx] < capacity[idx]]
            if src.size == 0 or dst.size == 0:
                continue
            dm = (g[dst][None, :] - g[src][:, None]) / two_n
            dC = (Ac[dst][None, :] - Ac[src][:, None]) / two_n + (
                diagA[src][:, None] + diagA[dst][None, :] - 2.0 * A[np.ix_(src, dst)]) / (two_n**2)
            same = src[:, None] == dst[None, :]
            if phase == "repair":
                ok = (dC < -1e-18) & ~same
                if not ok.any():
                    continue
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = dm / dC          # merit loss per unit of coancestry reduction
                score = np.where(ok, np.where(dm >= 0, -np.inf, ratio), np.inf)
                k = np.unravel_index(np.argmin(score), score.shape)
                if best is None or score[k] < best[0]:
                    best = (score[k], src[k[0]], dst[k[1]])
            else:
                ok = (dm > 1e-15) & (C + dC <= max_coancestry) & ~same
                if not ok.any():
                    continue
                score = np.where(ok, dm, -np.inf)
                k = np.unravel_index(np.argmax(score), score.shape)
                if best is None or score[k] > best[0]:
                    best = (score[k], src[k[0]], dst[k[1]])
        return best

    for phase in ("repair", "improve"):
        for _ in range(max_moves):
            c = x / two_n
            Ac = A @ c
            C = float(c @ Ac) / 2.0
            if phase == "repair" and C <= max_coancestry:
                break
            mv = best_move(c, Ac, phase, C)
            if mv is None:
                if phase == "repair":
                    raise ABPError("MODEL_NOT_IDENTIFIABLE",
                                   f"no whole-number allocation found that meets the coancestry "
                                   f"ceiling {max_coancestry:.6g} (current {C:.6g}); the ceiling is "
                                   "not relaxed", realized=C)
                break
            _, i, j = mv
            x[i] -= 1
            x[j] += 1
            moves[phase] += 1
    c = x / two_n
    return x, {"moves": moves, "coancestry": float(c @ A @ c) / 2.0, "merit": float(c @ g)}


def coancestry_target_from_delta_f(A: np.ndarray, delta_f: float) -> tuple[float, float]:
    """``C_max = C_t + delta_F (1 - C_t)`` with ``C_t`` the mean coancestry of the candidates."""
    n = A.shape[0]
    ct = float(A.sum()) / (2.0 * n * n)
    if not 0 < delta_f < 1:
        raise ABPError("SPEC_INVALID", "delta_f must lie in (0, 1)")
    return ct + delta_f * (1.0 - ct), ct


def expected_response(c: np.ndarray, g: np.ndarray) -> float:
    """Expected mean merit of the progeny relative to the candidates' mean merit: c'g - mean(g)."""
    return float(c @ g - g.mean())
