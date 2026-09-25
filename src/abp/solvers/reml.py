"""REML estimation of variance components (single trait; module M05).

Model and parameterization (method registry id ``reml.single_trait``)
---------------------------------------------------------------------
``y = X b + sum_k Z_k u_k + e``, ``u_k ~ N(0, theta_k K_k)``,
``e ~ N(0, theta_0 I)``, ``X`` of full column rank ``r_X`` (after the
constraints of :mod:`abp.core.design`), ``theta_k > 0`` in the interior.
With ``C`` the *unscaled* MME coefficient matrix (``W'W/theta_0`` plus
``K_k^{-1}/theta_k`` blocks) and ``s`` its solution,

    -2 logL = n log theta_0 + sum_k (q_k log theta_k + log|K_k|) + log|C| + y'Py,
    y'Py    = y'y/theta_0 - s'W'y/theta_0,

which equals ``log|V| + log|X'V^{-1}X| + y'Py`` exactly (the constant
``(n - r_X) log 2 pi`` is dropped).  Scores (first derivatives of logL):

    d/d theta_k = 1/2 [u_k'K_k^{-1}u_k / theta_k^2 - q_k/theta_k
                       + tr(K_k^{-1} C^{kk}) / theta_k^2]
    d/d theta_0 = 1/2 [e'e / theta_0^2 - tr(P)],
    tr(P)       = [n - r_X - sum_k (q_k - tr(K_k^{-1} C^{kk}) / theta_k)] / theta_0,

where ``C^{kk}`` is the ``u_k`` block of ``C^{-1}`` (= PEV).  The derivative
of ``V`` with respect to the genetic variance is ``Z A Z'`` (never ``Z Z'``
when ``A != I``; spec erratum T05).  The average-information matrix
(Gilmour, Thompson & Cullis 1995, Biometrics 51:1440) is
``AI = 1/2 F' P F`` with working variates ``f_k = Z_k u_k / theta_k`` and
``f_0 = e / theta_0``, computed with one extra multi-RHS solve.

Algorithm
---------
Average-information Newton steps with step-halving; an EM step (Dempster,
Laird & Rubin 1977, in its REML form) is used whenever the AI step fails to
increase the likelihood or leaves the parameter space.  EM updates:

    theta_k <- (u_k'K_k^{-1}u_k + tr(K_k^{-1}C^{kk})) / q_k
    theta_0 <- (e'e + tr(C^{-1} W'W)) / n

Convergence requires **both** a relative parameter change
``||d theta|| / ||theta|| < tol`` and a Newton decrement
``g' AI^{-1} g < tol``.  Boundary handling (active set): a component whose
estimate collapses below ``1e-6 * sum(theta)`` is fixed at zero, the
sub-model is re-fitted, and the zero is accepted only if the score at zero is
non-positive (the Kuhn-Tucker condition for a maximum on the boundary).
Non-convergence within the budget is an error; results are not issued.

Scale limit: this implementation uses the dense path (explicit ``C^{-1}``
for the traces) and is refused when the dense memory estimate exceeds the
configured budget.  Standard errors come from ``AI^{-1}`` (asymptotic,
conditional on the model) and are withheld at a boundary.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import scipy.sparse as sp

from ..errors import ABPError
from .blup import RandomTerm, build_system
from .mme import DenseCholesky, dense_bytes

BOUNDARY_REL = 1e-6


@dataclass
class REMLPoint:
    """Likelihood quantities at one parameter value (order: terms..., residual)."""

    theta: np.ndarray
    loglik: float
    score: np.ndarray
    ai: np.ndarray
    em: np.ndarray


@dataclass
class REMLFit:
    variances: dict[str, float]
    loglik: float
    iterations: int
    status: str                      # "converged" | "converged_boundary"
    boundary: list[str]
    active_terms: list[str]
    se: dict[str, float] | None
    heritability: float | None
    heritability_se: float | None
    ai_condition_number: float | None
    start: dict[str, float]
    start_source: str
    history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"status": self.status, "variances": self.variances, "loglik": self.loglik,
                "iterations": self.iterations, "boundary": self.boundary,
                "active_terms": self.active_terms, "se": self.se,
                "heritability": self.heritability, "heritability_se": self.heritability_se,
                "ai_condition_number": self.ai_condition_number, "start": self.start,
                "start_source": self.start_source, "history": self.history,
                "note": ("REML estimates; standard errors are asymptotic (inverse average "
                         "information) and not valid for components on the boundary.")}


class REMLEvaluator:
    """Evaluates logL, score, AI and EM updates for given variance components."""

    def __init__(self, y: np.ndarray, X: sp.csr_matrix, terms: Sequence[RandomTerm],
                 memory_budget_bytes: int):
        self.y = np.asarray(y, dtype=np.float64)
        self.X = X
        self.terms = list(terms)
        self.n = self.y.size
        self.rx = X.shape[1]
        for t in self.terms:
            if t.logdet_k is None:
                raise ABPError("UNSUPPORTED_COMBINATION",
                               f"REML needs log|K| for term {t.name!r}")
        n_eq = self.rx + sum(t.q for t in self.terms)
        need = dense_bytes(n_eq, True)
        if need > memory_budget_bytes:
            raise ABPError("RESOURCE_MEMORY",
                           f"REML (dense path) for {n_eq} equations needs ~{need / 2**30:.2f} GiB; "
                           f"budget is {memory_budget_bytes / 2**30:.2f} GiB",
                           n_equations=n_eq)
        if self.n - self.rx <= 0:
            raise ABPError("MODEL_NOT_IDENTIFIABLE",
                           "no residual degrees of freedom (records <= fixed-effect rank)")
        self.yy = float(self.y @ self.y)

    def names(self) -> list[str]:
        return [t.name for t in self.terms] + ["residual"]

    def evaluate(self, theta: np.ndarray, with_ai: bool = True) -> REMLPoint:
        names = self.names()
        vc = dict(zip(names, map(float, theta)))
        system = build_system(self.y, self.X, self.terms, vc)
        fac = DenseCholesky(system.C.toarray())
        s = fac.solve(system.rhs)
        W = system.W
        th0 = vc["residual"]
        e = self.y - W @ s
        ypy = self.yy / th0 - float(s @ system.rhs)
        m2ll = self.n * math.log(th0) + fac.logdet() + ypy
        Cinv = fac.inverse()
        score = np.empty(len(names))
        em = np.empty(len(names))
        sum_adj = 0.0
        for k, t in enumerate(self.terms):
            a, b = system.offsets[t.name]
            thk = vc[t.name]
            m2ll += t.q * math.log(thk) + t.logdet_k
            u = s[a:b]
            Kinv = t.k_inv
            Kinv_u = Kinv @ u
            quad = float(u @ Kinv_u)
            Ckk = Cinv[a:b, a:b]
            if sp.issparse(Kinv):
                tr = float(Kinv.multiply(Ckk).sum())
            else:
                tr = float(np.sum(Kinv * Ckk))
            score[k] = 0.5 * (quad / thk**2 - t.q / thk + tr / thk**2)
            em[k] = (quad + tr) / t.q
            sum_adj += t.q - tr / thk
        trP = (self.n - self.rx - sum_adj) / th0
        ee = float(e @ e)
        score[-1] = 0.5 * (ee / th0**2 - trP)
        WtW = (W.T @ W).tocoo()
        tr_cww = float(np.sum(Cinv[WtW.row, WtW.col] * WtW.data))
        em[-1] = (ee + tr_cww) / self.n
        ai = np.zeros((len(names), len(names)))
        if with_ai:
            Fw = np.empty((self.n, len(names)))
            for k, t in enumerate(self.terms):
                a, b = system.offsets[t.name]
                Fw[:, k] = t.Z @ s[a:b] / vc[t.name]
            Fw[:, -1] = e / th0
            S = fac.solve(W.T @ Fw / th0)
            PF = (Fw - W @ S) / th0
            ai = 0.5 * (Fw.T @ PF)
            ai = 0.5 * (ai + ai.T)
        return REMLPoint(np.array(theta, dtype=np.float64), -0.5 * m2ll, score, ai, em)


def default_start(y: np.ndarray, X: sp.csr_matrix, terms: Sequence[RandomTerm]) -> dict[str, float]:
    """Deterministic start: OLS residual variance split 1/3 genetic, 1/6 per other
    random term, remainder residual."""
    Xd = X.toarray()
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    r = y - Xd @ beta
    s2 = float(r @ r) / max(1, y.size - X.shape[1])
    if s2 <= 0:
        raise ABPError("MODEL_NOT_IDENTIFIABLE", "fixed effects fit the data exactly; no variance left")
    out = {}
    for t in terms:
        out[t.name] = s2 / 3.0 if t.genetic else s2 / 6.0
    out["residual"] = max(s2 - sum(out.values()), s2 / 6.0)
    return out


def _ai_step(p: REMLPoint) -> np.ndarray | None:
    try:
        L = np.linalg.cholesky(p.ai)
    except np.linalg.LinAlgError:
        return None
    return np.linalg.solve(L.T, np.linalg.solve(L, p.score))


def _newton_decrement(p: REMLPoint) -> float:
    step = _ai_step(p)
    if step is None:
        return float("inf")
    return float(p.score @ step)


def _optimize(ev: REMLEvaluator, theta0: np.ndarray, cfg: dict, history: list,
              it0: int, checkpoint, no_candidates: set[str]) -> tuple[REMLPoint, int, list[int]]:
    """Maximize logL; returns (point, iterations, boundary-candidate indices).

    A random-term component becomes a boundary candidate when (a) the full AI
    step would take it to/below the floor while its score is negative, or
    (b) it stays within 10x the floor for 3 iterations with a negative score.
    Terms in ``no_candidates`` (a boundary already rejected by the
    Kuhn-Tucker check) can only become candidates through rule (b).
    """
    tol, max_iter, algo = cfg["tol"], cfg["max_iter"], cfg["algorithm"]
    names = ev.names()
    theta = theta0.copy()
    p = ev.evaluate(theta)
    it = it0
    low_count = np.zeros(theta.size, dtype=int)
    while it < max_iter:
        it += 1
        floor = BOUNDARY_REL * float(theta.sum())
        used = "em"
        new = None
        if algo == "ai":
            step = _ai_step(p)
            if step is not None:
                full = theta + step
                cand_idx = [k for k in range(theta.size - 1)
                            if full[k] <= floor and p.score[k] < 0
                            and names[k] not in no_candidates]
                if cand_idx:
                    history.append({"iteration": it, "event": "AI step leaves the parameter "
                                    "space for " + ", ".join(names[k] for k in cand_idx)
                                    + "; testing the boundary sub-model"})
                    return p, it, cand_idx
                lam = 1.0
                for _ in range(12):
                    cand = theta + lam * step
                    if np.all(cand > floor):
                        pc = ev.evaluate(cand)
                        if pc.loglik >= p.loglik - 1e-10 * max(1.0, abs(p.loglik)):
                            new, used = pc, ("ai" if lam == 1.0 else f"ai(step {lam:g})")
                            break
                    lam *= 0.5
        if new is None:
            cand = np.maximum(p.em, 0.5 * floor)
            new = ev.evaluate(cand)
            used = "em"
        change = float(np.linalg.norm(new.theta - theta) / np.linalg.norm(theta))
        dec = _newton_decrement(new)
        history.append({"iteration": it, "method": used, "loglik": new.loglik,
                        "theta": new.theta.tolist(), "score": new.score.tolist(),
                        "rel_change": change, "newton_decrement": dec})
        if checkpoint is not None:
            checkpoint(it, new.theta, history)
        theta, p = new.theta, new
        low = theta[:-1] <= 10 * floor  # the residual variance is never put on the boundary
        low_count[:-1] = np.where(low, low_count[:-1] + 1, 0)
        hit = [int(k) for k in np.flatnonzero(low_count[:-1] >= 3) if p.score[k] < 0]
        if hit:
            return p, it, hit
        if change < tol and dec < tol:
            return p, it, []
    raise ABPError("REML_NOT_CONVERGED",
                   f"REML did not converge in {max_iter} iterations",
                   last_theta=dict(zip(ev.names(), theta.tolist())), last_score=p.score.tolist(),
                   history_tail=history[-3:])


def _score_at_zero(y, X, terms: Sequence[RandomTerm], active_vc: dict[str, float],
                   zero_term: RandomTerm) -> float:
    """dlogL/dtheta_k at theta_k = 0 (term absent from V), from the sub-model:
    1/2 [ (Z'Py)' K (Z'Py) - tr(Z'PZ K) ]."""
    system = build_system(y, X, terms, active_vc)
    fac = DenseCholesky(system.C.toarray())
    s = fac.solve(system.rhs)
    th0 = active_vc["residual"]
    Py = (y - system.W @ s) / th0
    Z = zero_term.Z
    zpy = Z.T @ Py
    Kinv = zero_term.k_inv
    Kd = Kinv.toarray() if sp.issparse(Kinv) else np.asarray(Kinv)
    K = np.linalg.inv(Kd)
    M = (system.W.T @ Z).toarray() / th0
    ZPZ = (Z.T @ Z).toarray() / th0 - M.T @ fac.solve(M)
    return 0.5 * (float(zpy @ K @ zpy) - float(np.sum(ZPZ * K)))


def reml_fit(y: np.ndarray, X: sp.csr_matrix, terms: Sequence[RandomTerm], cfg: dict,
             memory_budget_bytes: int = 4 * 2**30, checkpoint_path: Path | None = None,
             resume: bool = False, fingerprint: str = "") -> REMLFit:
    """Fit variance components by REML; raises ``REML_NOT_CONVERGED`` on failure."""
    y = np.asarray(y, dtype=np.float64)
    terms = list(terms)
    if cfg.get("start"):
        start = dict(cfg["start"])
        source = "spec (reml.start)"
    else:
        start = default_start(y, X, terms)
        source = "default: OLS residual variance split (genetic 1/3, other terms 1/6)"
    for k, v in start.items():
        if not v > 0:
            raise ABPError("COVARIANCE_NOT_PD", f"REML start value for {k!r} must be > 0", term=k)
    history: list[dict] = []
    theta = np.array([start[t.name] for t in terms] + [start["residual"]])
    it0 = 0
    active = list(terms)
    boundary: list[str] = []
    if resume and checkpoint_path is not None and checkpoint_path.exists():
        ck = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if ck.get("fingerprint") != fingerprint:
            raise ABPError("CHECKPOINT_MISMATCH",
                           f"checkpoint {checkpoint_path.name} belongs to different inputs/spec",
                           checkpoint=str(checkpoint_path))
        names_ck = ck["names"]
        active = [t for t in terms if t.name in names_ck]
        boundary = list(ck.get("boundary", []))
        theta = np.array(ck["theta"], dtype=np.float64)
        it0 = int(ck["iteration"])
        history = list(ck["history"])
        history.append({"event": f"resumed from checkpoint at iteration {it0}"})

    def make_checkpoint(names):
        if checkpoint_path is None:
            return None

        def write(it, th, hist):
            tmp = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
            tmp.write_text(json.dumps({"fingerprint": fingerprint, "iteration": it,
                                       "names": names, "theta": th.tolist(),
                                       "boundary": boundary, "history": hist}),
                           encoding="utf-8")
            os.replace(tmp, checkpoint_path)
        return write

    no_candidates: set[str] = set()
    while True:
        ev = REMLEvaluator(y, X, active, memory_budget_bytes)
        point, it0, hit = _optimize(ev, theta, cfg, history, it0,
                                    make_checkpoint(ev.names()), no_candidates)
        if hit:
            k = hit[0]
            name = active[k].name
            history.append({"event": f"{name} fixed at zero (boundary candidate); "
                                     "re-fitting the sub-model"})
            boundary.append(name)
            theta = np.delete(point.theta, k)
            active = [t for j, t in enumerate(active) if j != k]
            continue
        # Kuhn-Tucker check: the score at zero must be <= 0 for every
        # component held on the boundary, otherwise zero is not a maximum.
        names = ev.names()
        vc_active = dict(zip(names, point.theta.tolist()))
        rejected = None
        for name in boundary:
            zt = next(t for t in terms if t.name == name)
            g0 = _score_at_zero(y, X, active, vc_active, zt)
            history.append({"event": f"score at zero for {name}", "score": g0})
            if g0 > 1e-6 * max(1.0, abs(point.loglik)):
                rejected = (name, g0)
                break
        if rejected is None:
            break
        name, g0 = rejected
        if name in no_candidates:
            raise ABPError("REML_NOT_CONVERGED",
                           f"{name} repeatedly collapses to zero although the score there is "
                           f"positive ({g0:.3g}); the likelihood surface is ill-behaved",
                           term=name, score=g0)
        history.append({"event": f"boundary rejected for {name} (score at zero {g0:.3g} > 0); "
                                 "reinstating the term"})
        no_candidates.add(name)
        boundary.remove(name)
        active = [t for t in terms if t.name not in boundary]
        restart = 0.1 * vc_active["residual"]
        theta = np.array([vc_active.get(t.name, restart) for t in active]
                         + [vc_active["residual"]])
    variances = {t.name: vc_active.get(t.name, 0.0) for t in terms}
    variances["residual"] = vc_active["residual"]
    se = h2 = h2_se = cond = None
    genetic = [t.name for t in terms if t.genetic]
    total = sum(variances.values())
    if genetic:
        h2 = variances[genetic[0]] / total
    try:
        cond = float(np.linalg.cond(point.ai))
        cov = np.linalg.inv(point.ai)
        if not boundary:
            se = {n: float(math.sqrt(cov[i, i])) if cov[i, i] > 0 else None
                  for i, n in enumerate(names)}
            if genetic:
                g = np.full(len(names), -variances[genetic[0]] / total**2)
                g[names.index(genetic[0])] += 1.0 / total
                h2_se = float(math.sqrt(max(g @ cov @ g, 0.0)))
    except np.linalg.LinAlgError:
        cond = None  # singular AI matrix: components not separately identifiable
    status = "converged_boundary" if boundary else "converged"
    return REMLFit(variances, point.loglik, it0, status, boundary, [t.name for t in active], se,
                   h2, h2_se, cond, start, source, history)
