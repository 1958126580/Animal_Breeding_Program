"""REML for the maternal animal model (method id ``reml.maternal``; round 13).

Model
-----
One trait, one record per row::

    y = X b + Z_a a + Z_m m + sum_p Z_p c_p + e,

``(a, m)`` the direct and maternal genetic effects of every animal of the pedigree
(animal-major: equation ``animal * 2 + j``, ``j = 0`` direct, ``1`` maternal) with
``Var([a; m]) = A (x) G0`` (written ``kron(A, G0)`` in the animal-major order),
``G0 = [[s_a, s_am], [s_am, s_m]]``; ``Z_a`` links a record to its animal, ``Z_m`` to
its dam (no maternal effect for a record whose dam is unknown); ``c_p ~ N(0, s_p I)``
independent terms (e.g. the maternal permanent environment on a dam column);
``e ~ N(0, s_e I)``.  Willham (1963, Biometrics 19:18; 1972, J Anim Sci 35:1288);
mixed-model equations as in Henderson (1984, Applications of Linear Models in Animal
Breeding).

Parameters ``theta = (s_a, s_am, s_m, s_p..., s_e)``.  With ``W = [X Z_g Z_p...]``,
``C = W'W / s_e + blockdiag(0, kron(A^-1, G0^-1), I / s_p, ...)`` (the unscaled MME
coefficient matrix) and ``s`` its solution of ``C s = W'y / s_e``:

    -2 logL = n log s_e + q log|G0| + 2 log|A| + sum_p q_p log s_p + log|C| + y'Py,
    y'Py    = y'y / s_e - s'W'y / s_e,

which equals ``log|V| + log|X'V^-1 X| + y'Py`` (constant dropped; tested against the
marginal form).  With ``U`` the ``q x 2`` genetic solutions and
``T_jk = sum_ab (A^-1)_ab (C^-1)_{(a,j),(b,k)}``, ``S = U'A^-1 U + T``:

* score of a ``G0`` direction ``E`` (``E_jk = E_kj = 1``):
  ``1/2 [tr(G0^-1 E G0^-1 S) - q tr(E G0^-1)]`` (as for multi-trait REML);
* score of ``s_p``: ``1/2 [(c_p'c_p + tr C^{pp}) / s_p^2 - q_p / s_p]``;
* score of ``s_e``: ``1/2 [e'e / s_e^2 - tr(P)]``,
  ``tr(P) = (n - r_X - (2q - tr(G0^-1 T)) - sum_p (q_p - tr C^{pp} / s_p)) / s_e``.

The average-information matrix ``1/2 F'PF`` (Gilmour, Thompson & Cullis 1995) uses the
working variates ``f_E = Z_g vec(U (E G0^-1)')``, ``f_p = Z_p c_p / s_p``,
``f_e = e / s_e``, with ``P F = (F - W C^-1 W'F / s_e) / s_e`` (one multi-RHS solve).
EM updates (REML form): ``G0 <- S / q``, ``s_p <- (c_p'c_p + tr C^{pp}) / q_p``,
``s_e <- (e'e + tr(C^-1 W'W)) / n``.

Algorithm: three EM steps, then AI steps (halved while they leave the parameter space
- ``G0`` not positive definite or a variance <= 0 - or decrease logL), EM when no AI
step is accepted.  Convergence: relative parameter change and Newton decrement
``g'AI^-1 g`` both below ``tol``.  An independent-term variance that collapses is fixed
at zero and the sub-model re-fitted; zero is accepted only if the score at zero is
``<= 0`` (Kuhn-Tucker; as in single-trait REML).  A singular ``G0`` (``|r_am| -> 1`` or
a genetic variance -> 0) raises ``MODEL_NOT_IDENTIFIABLE``: this version does not
estimate that boundary.

Traces need ``C^-1`` only on the pattern of ``C``: the dense inverse up to
``DENSE_MAX`` equations, otherwise sparse selected inversion on one
:class:`~abp.solvers.cholesky.SparseLDL` whose ordering and symbolic factor are kept
for all iterations (numeric refactorization on a fixed pattern that includes every
``kron(A^-1, G0^-1)`` entry, so a zero ``s_am`` does not drop entries).
Standard errors from ``AI^-1``; ``h2``, ``m2`` and ``r_am`` by the delta method with
``s_P = s_a + s_m + s_am + sum_p s_p + s_e`` (Willham 1972).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

from ..errors import ABPError
from .mme import DenseCholesky, dense_bytes
from .reml import kt_rejects_zero

DENSE_MAX = 2000      #: dense trace path up to this many equations
BOUNDARY_REL = 1e-6   #: an iid variance below this share of s_P counts as on the boundary
MIN_EIG_REL = 1e-8    #: smallest / largest eigenvalue of G0 accepted at convergence
R = 2                 #: genetic effects per animal (direct, maternal)


@dataclass
class MaternalData:
    """Inputs of the maternal animal model (indices into the relationship matrix)."""

    y: np.ndarray
    X: sp.csr_matrix
    animal: np.ndarray                 # (n,) animal of each record
    dam: np.ndarray                    # (n,) dam of each record, -1 = unknown
    iid: list = field(default_factory=list)   # [(name, level index (n,), n_levels)]


@dataclass
class _Point:
    theta: np.ndarray
    loglik: float
    score: np.ndarray
    ai: np.ndarray
    em: np.ndarray
    s: np.ndarray


@dataclass
class MaternalREMLFit:
    G0: np.ndarray
    iid: dict
    residual: float
    loglik: float
    iterations: int
    status: str
    se: dict | None
    cov: np.ndarray | None
    names: list
    derived: dict
    trace_method: str
    start: dict
    history: list = field(default_factory=list)

    def variances(self) -> dict:
        out = {"direct": float(self.G0[0, 0]), "maternal": float(self.G0[1, 1]),
               "direct-maternal covariance": float(self.G0[0, 1])}
        out.update({k: float(v) for k, v in self.iid.items()})
        out["residual"] = float(self.residual)
        return out

    def to_dict(self) -> dict:
        return {"status": self.status, "algorithm": "AI-REML with EM fallback (maternal "
                "animal model)", "G0": self.G0.tolist(), "variances": self.variances(),
                "loglik": self.loglik, "iterations": self.iterations, "se": self.se,
                "derived": self.derived, "trace_method": self.trace_method,
                "start": self.start, "history": self.history,
                "note": "REML estimates; SE are asymptotic (inverse average information)."}


def maternal_design(data: MaternalData, q: int) -> tuple[sp.csr_matrix, list[sp.csr_matrix]]:
    """``Z_g`` (``n x 2q``, animal-major) and the ``Z_p`` of the independent terms."""
    n = data.y.size
    animal = np.asarray(data.animal, dtype=np.int64)
    dam = np.asarray(data.dam, dtype=np.int64)
    if animal.shape != (n,) or dam.shape != (n,):
        raise ABPError("SPEC_INVALID", "animal and dam indices need one entry per record")
    if np.any((animal < 0) | (animal >= q)) or np.any(dam >= q):
        raise ABPError("SPEC_INVALID", "animal or dam index outside the relationship matrix")
    known = dam >= 0
    rows = np.concatenate([np.arange(n), np.flatnonzero(known)])
    cols = np.concatenate([animal * R, dam[known] * R + 1])
    Zg = sp.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(n, q * R))
    Zp = []
    for name, lev, nl in data.iid:
        lev = np.asarray(lev, dtype=np.int64)
        if lev.shape != (n,) or np.any((lev < 0) | (lev >= nl)):
            raise ABPError("SPEC_INVALID", f"level index of term {name!r} out of range")
        Zp.append(sp.csr_matrix((np.ones(n), (np.arange(n), lev)), shape=(n, int(nl))))
    return Zg, Zp


def _unpack(theta: np.ndarray, n_iid: int):
    G0 = np.array([[theta[0], theta[1]], [theta[1], theta[2]]])
    return G0, np.asarray(theta[3:3 + n_iid], dtype=np.float64), float(theta[3 + n_iid])


def _in_space(theta: np.ndarray, n_iid: int) -> bool:
    G0, vp, se = _unpack(theta, n_iid)
    if not (se > 0 and np.all(vp > 0)):
        return False
    try:
        np.linalg.cholesky(G0)
        return True
    except np.linalg.LinAlgError:
        return False


GENETIC_DIRECTIONS = [(0, 0), (0, 1), (1, 1)]


def _direction(j: int, k: int) -> np.ndarray:
    E = np.zeros((R, R))
    E[j, k] = E[k, j] = 1.0
    return E


class MaternalREMLEvaluator:
    """logL, score, AI and EM update of the maternal animal model at ``theta``."""

    def __init__(self, data: MaternalData, k_inv, logdet_k: float,
                 memory_budget_bytes: int = 4 * 2**30, dense: bool | None = None):
        if logdet_k is None:
            raise ABPError("UNSUPPORTED_COMBINATION", "maternal REML needs log|A|")
        self.y = np.asarray(data.y, dtype=np.float64)
        self.n = self.y.size
        self.X = sp.csr_matrix(data.X)
        self.p = self.X.shape[1]
        self.k_inv = sp.csr_matrix(k_inv)
        self.k_inv.eliminate_zeros()
        self.q = self.k_inv.shape[0]
        self.logdet_k = float(logdet_k)
        self.Zg, self.Zp = maternal_design(data, self.q)
        self.iid_names = [nm for nm, _, _ in data.iid]
        self.n_iid = len(self.Zp)
        self.qp = [Z.shape[1] for Z in self.Zp]
        self.W = sp.hstack([self.X, self.Zg] + self.Zp, format="csr")
        self.g0 = self.p
        self.g1 = self.p + self.q * R
        offs, a = [], self.g1
        for qp in self.qp:
            offs.append((a, a + qp))
            a += qp
        self.p_off = offs
        self.n_eq = a
        if self.n - self.p <= 0:
            raise ABPError("MODEL_NOT_IDENTIFIABLE", "no residual degrees of freedom")
        self.WtW = (self.W.T @ self.W).tocsr()
        self.Wty = self.W.T @ self.y
        self.yy = float(self.y @ self.y)
        if dense is None:
            dense = self.n_eq <= DENSE_MAX and dense_bytes(self.n_eq, True) <= memory_budget_bytes
        self.dense = bool(dense)
        self.trace_method = "dense_inverse" if self.dense else "sparse_selected_inversion"
        # T pairs: (A^-1)_ab x (j, k)
        Kc = self.k_inv.tocoo()
        jj, kk = np.meshgrid(np.arange(R), np.arange(R), indexing="ij")
        jj, kk = jj.ravel(), kk.ravel()
        self.t_rows = (self.g0 + Kc.row[:, None] * R + jj[None, :]).ravel()
        self.t_cols = (self.g0 + Kc.col[:, None] * R + kk[None, :]).ravel()
        self.t_w = np.repeat(Kc.data, R * R)
        self.t_jk = np.tile(jj * R + kk, Kc.nnz)
        wc = self.WtW.tocoo()
        self.ww_r, self.ww_c, self.ww_v = wc.row, wc.col, wc.data
        self._fac = None
        if not self.dense:
            from .cholesky import SparseLDL
            # fixed pattern: C at a G0 with a non-zero covariance (every kron(A^-1, G0^-1)
            # entry present); later evaluations refactorize values on this pattern
            th = np.concatenate([[1.0, 0.3, 1.0], np.ones(self.n_iid), [1.0]])
            pat = self._coefficient(th)
            pat.sort_indices()
            self._pat = pat
            self._fac = SparseLDL(pat, memory_budget_bytes)
            self._pat = self._fac.C

    def names(self) -> list[str]:
        return ["direct", "direct-maternal covariance", "maternal"] + self.iid_names + ["residual"]

    def _coefficient(self, theta: np.ndarray) -> sp.csr_matrix:
        G0, vp, se = _unpack(theta, self.n_iid)
        Gi = np.linalg.inv(G0)
        blocks = [sp.csr_matrix((self.p, self.p)), sp.kron(self.k_inv, sp.csr_matrix(Gi))]
        blocks += [sp.identity(qp, format="csr") / v for qp, v in zip(self.qp, vp)]
        return (self.WtW / se + sp.block_diag(blocks, format="csr")).tocsr()

    def _factor(self, C: sp.csr_matrix):
        if self.dense:
            fac = DenseCholesky(C.toarray())
            Cinv = fac.inverse()
            return fac, (lambda r, c: Cinv[r, c])
        from .cholesky import _on_pattern
        Cp = _on_pattern(C, self._pat)
        self._fac.refactor_values(Cp.data)
        si = self._fac.selected_inverse()
        return self._fac, si.entries

    def solve(self, theta: np.ndarray) -> np.ndarray:
        """MME solution at ``theta`` (no inverse: for the Kackar-Harville differences)."""
        theta = np.asarray(theta, dtype=np.float64)
        C = self._coefficient(theta)
        se = theta[-1]
        if self.dense:
            return DenseCholesky(C.toarray()).solve(self.Wty / se)
        from .cholesky import _on_pattern
        self._fac.refactor_values(_on_pattern(C, self._pat).data)
        return self._fac.solve(self.Wty / se)

    def evaluate(self, theta: np.ndarray, with_ai: bool = True) -> _Point:
        theta = np.asarray(theta, dtype=np.float64)
        G0, vp, se = _unpack(theta, self.n_iid)
        Gi = np.linalg.inv(G0)
        C = self._coefficient(theta)
        fac, ent = self._factor(C)
        rhs = self.Wty / se
        s = fac.solve(rhs)
        e = self.y - self.W @ s
        ypy = self.yy / se - float(s @ rhs)
        q = self.q
        m2ll = (self.n * math.log(se) + q * np.linalg.slogdet(G0)[1] + R * self.logdet_k
                + sum(qp * math.log(v) for qp, v in zip(self.qp, vp)) + fac.logdet() + ypy)
        U = s[self.g0:self.g1].reshape(q, R)
        S = U.T @ (self.k_inv @ U)
        T = np.bincount(self.t_jk, weights=self.t_w * ent(self.t_rows, self.t_cols),
                        minlength=R * R).reshape(R, R)
        T = 0.5 * (T + T.T)
        S = S + T
        k = len(GENETIC_DIRECTIONS)
        npar = k + self.n_iid + 1
        score = np.zeros(npar)
        for i, (j, l) in enumerate(GENETIC_DIRECTIONS):
            E = _direction(j, l)
            score[i] = 0.5 * (np.trace(Gi @ E @ Gi @ S) - q * np.trace(E @ Gi))
        Gnew = S / q
        em = np.zeros(npar)
        em[:k] = [Gnew[0, 0], 0.5 * (Gnew[0, 1] + Gnew[1, 0]), Gnew[1, 1]]
        adj = R * q - float(np.sum(Gi * T))
        for i, ((a, b), qp, v) in enumerate(zip(self.p_off, self.qp, vp)):
            c = s[a:b]
            idx = np.arange(a, b)
            trc = float(np.sum(ent(idx, idx)))
            score[k + i] = 0.5 * ((float(c @ c) + trc) / v**2 - qp / v)
            em[k + i] = (float(c @ c) + trc) / qp
            adj += qp - trc / v
        ee = float(e @ e)
        trP = (self.n - self.p - adj) / se
        score[-1] = 0.5 * (ee / se**2 - trP)
        tr_cww = float(np.dot(self.ww_v, ent(self.ww_r, self.ww_c)))
        em[-1] = (ee + tr_cww) / self.n
        ai = np.zeros((npar, npar))
        if with_ai:
            F = np.empty((self.n, npar))
            for i, (j, l) in enumerate(GENETIC_DIRECTIONS):
                M = _direction(j, l) @ Gi
                F[:, i] = self.Zg @ (U @ M.T).ravel()
            for i, ((a, b), v) in enumerate(zip(self.p_off, vp)):
                F[:, k + i] = self.Zp[i] @ s[a:b] / v
            F[:, -1] = e / se
            X2 = fac.solve(self.W.T @ F / se)
            PF = (F - self.W @ X2) / se
            ai = 0.5 * (F.T @ PF)
            ai = 0.5 * (ai + ai.T)
        return _Point(theta, -0.5 * m2ll, score, ai, em, s)


def default_start(data: MaternalData) -> np.ndarray:
    """Deterministic start from the least-squares residual variance ``v`` of the fixed
    effects: direct ``v/4``, maternal ``v/8``, covariance 0, each independent term
    ``v/8``, residual the rest (at least ``v/4``); data only."""
    X = sp.csr_matrix(data.X)
    beta = lsqr(X, data.y, atol=1e-12, btol=1e-12)[0]
    r = data.y - X @ beta
    v = float(r @ r) / max(1, data.y.size - X.shape[1])
    if not v > 0:
        raise ABPError("MODEL_NOT_IDENTIFIABLE", "fixed effects fit the data exactly")
    n_iid = len(data.iid)
    rest = max(v - v / 4 - v / 8 - n_iid * v / 8, v / 4)
    return np.concatenate([[v / 4, 0.0, v / 8], np.full(n_iid, v / 8), [rest]])


def _derived(theta: np.ndarray, cov: np.ndarray | None, n_iid: int) -> dict:
    G0, vp, se = _unpack(theta, n_iid)
    sP = G0[0, 0] + G0[1, 1] + G0[0, 1] + float(vp.sum()) + se
    out = {"phenotypic_variance": float(sP), "h2": float(G0[0, 0] / sP),
           "m2": float(G0[1, 1] / sP),
           "direct_maternal_correlation": float(G0[0, 1] / math.sqrt(G0[0, 0] * G0[1, 1]))}
    if cov is not None:
        npar = theta.size
        dsP = np.ones(npar)                     # d sP / d theta (s_am enters once)
        grads = {}
        g = -G0[0, 0] / sP**2 * dsP
        g[0] += 1.0 / sP
        grads["h2"] = g
        g = -G0[1, 1] / sP**2 * dsP
        g[2] += 1.0 / sP
        grads["m2"] = g
        g = np.zeros(npar)
        d = math.sqrt(G0[0, 0] * G0[1, 1])
        g[1] = 1.0 / d
        g[0] = -0.5 * G0[0, 1] / d / G0[0, 0]
        g[2] = -0.5 * G0[0, 1] / d / G0[1, 1]
        grads["direct_maternal_correlation"] = g
        for key, g in grads.items():
            out[key + "_se"] = float(math.sqrt(max(g @ cov @ g, 0.0)))
    return out


def _optimize(ev: "MaternalREMLEvaluator", theta: np.ndarray, tol: float, max_iter: int,
              history: list, it0: int, no_candidates: frozenset = frozenset()):
    """AI/EM iterations; returns ``(point, iterations, boundary_candidate_index | None)``.

    An independent-term variance becomes a boundary candidate when the full AI step
    would take it to <= 0 while its score is negative, or when it stays below
    ``1e-4 s_P`` for 3 iterations with a negative score (as in single-trait REML); terms
    in ``no_candidates`` (a boundary already rejected by the Kuhn-Tucker check) only by
    the second rule."""
    n_iid = ev.n_iid
    point = ev.evaluate(theta)
    k = len(GENETIC_DIRECTIONS)
    low = np.zeros(n_iid, dtype=int)
    outside = 0                      # consecutive full AI steps outside the space (G0 not PD)
    it = it0
    while it < it0 + max_iter:
        it += 1
        kind, new = "em", None
        if it - it0 > 3:
            try:
                delta = np.linalg.solve(point.ai, point.score)
            except np.linalg.LinAlgError:
                delta = None
            if delta is not None:
                full = point.theta + delta
                cand = [i for i in range(n_iid) if full[k + i] <= 0 and point.score[k + i] < 0
                        and ev.iid_names[i] not in no_candidates]
                if cand:
                    return point, it, cand[0]
                outside = 0 if _in_space(full, n_iid) else outside + 1
                if outside >= 15:
                    G_ = _unpack(point.theta, n_iid)[0]
                    r_am = G_[0, 1] / math.sqrt(G_[0, 0] * G_[1, 1])
                    raise ABPError("MODEL_NOT_IDENTIFIABLE", "maternal REML: the likelihood keeps "
                                   "increasing towards a singular G0 (15 full AI steps in a row "
                                   f"leave the positive-definite region; r_am = {r_am:.4f}, "
                                   f"G0 = {G_.round(5).tolist()}); |r_am| -> 1 or a genetic "
                                   "variance -> 0 is not estimated in this version",
                                   G0=G_.tolist(), iteration=it)
                lam = 1.0
                for _ in range(8):
                    c = point.theta + lam * delta
                    if _in_space(c, n_iid):
                        pc = ev.evaluate(c)
                        if pc.loglik >= point.loglik - 1e-10 * max(1.0, abs(point.loglik)):
                            new, kind = pc, f"ai(step {lam:g})"
                            break
                    lam *= 0.5
        if new is None:
            if not _in_space(point.em, n_iid):
                raise ABPError("MODEL_NOT_IDENTIFIABLE", "maternal REML: the EM update left the "
                               "parameter space (|r_am| or a genetic variance at the boundary)",
                               iteration=it, theta=point.em.tolist())
            new = ev.evaluate(point.em)
            if new.loglik < point.loglik - 1e-8 * max(1.0, abs(point.loglik)):
                raise ABPError("REML_NOT_CONVERGED", "maternal REML: EM step decreased the "
                               "likelihood (numerical problem)", iteration=it)
        rel = float(np.max(np.abs(new.theta - point.theta))
                    / max(np.max(np.abs(new.theta)), 1e-300))
        try:
            dec = float(new.score @ np.linalg.solve(new.ai, new.score))
        except np.linalg.LinAlgError:
            dec = float("inf")
        history.append({"iteration": it, "step": kind, "loglik": new.loglik,
                        "theta": new.theta.tolist(), "rel_change": rel, "newton_decrement": dec})
        point = new
        G0, vp, se = _unpack(point.theta, n_iid)
        sP = G0[0, 0] + G0[1, 1] + G0[0, 1] + float(vp.sum()) + se
        low = np.where(vp < 1e-4 * sP, low + 1, 0)
        hit = [i for i in range(n_iid) if low[i] >= 3 and point.score[k + i] < 0]
        if hit:
            return point, it, hit[0]
        if rel < tol and abs(dec) < tol:
            return point, it, None
    raise ABPError("REML_NOT_CONVERGED", f"maternal REML did not converge in {max_iter} "
                   "iterations", history=history[-5:])


def _score_at_zero(ev: "MaternalREMLEvaluator", theta: np.ndarray, Zd: sp.csr_matrix) -> float:
    """``dlogL/ds_p`` at ``s_p = 0`` for a dropped independent term with design ``Zd``,
    from the fitted sub-model: ``1/2 [ |Zd'Py|^2 - tr(Zd'P Zd) ]``,
    ``Zd'P Zd = Zd'Zd / s_e - B' C^-1 B``, ``B = W'Zd / s_e``."""
    G0, vp, se = _unpack(theta, ev.n_iid)
    C = ev._coefficient(theta)
    fac, _ = ev._factor(C)
    s = fac.solve(ev.Wty / se)
    Py = (ev.y - ev.W @ s) / se
    zpy = Zd.T @ Py
    B = (ev.W.T @ Zd).toarray() / se
    ZPZ_tr = float(Zd.multiply(Zd).sum()) / se - float(np.sum(B * fac.solve(B)))
    return 0.5 * (float(zpy @ zpy) - ZPZ_tr)


def maternal_reml_fit(data: MaternalData, k_inv, logdet_k: float, cfg: dict,
                      memory_budget_bytes: int = 4 * 2**30, start: np.ndarray | None = None,
                      dense: bool | None = None) -> MaternalREMLFit:
    """Fit ``G0``, the independent-term variances and ``s_e`` by AI-REML (module notes).

    Independent-term variances may end on the boundary: a collapsing term is fixed at 0,
    the sub-model is re-fitted and zero is accepted only if the score at zero is <= 0
    (Kuhn-Tucker); standard errors are then withheld.  A singular ``G0`` is refused."""
    n_iid0 = len(data.iid)
    if start is None:
        theta = default_start(data)
        source = ("data-based heuristic (least-squares residual variance v: direct v/4, "
                  "maternal v/8, covariance 0, independent terms v/8, residual the rest)")
    else:
        theta = np.asarray(start, dtype=np.float64)
        source = "user"
    if theta.size != 4 + n_iid0 or not _in_space(theta, n_iid0):
        raise ABPError("COVARIANCE_NOT_PD", "maternal REML start values are not in the "
                       "parameter space (G0 positive definite, variances > 0)")
    theta0 = theta.copy()
    tol, max_iter = float(cfg.get("tol", 1e-8)), int(cfg.get("max_iter", 200))
    history: list[dict] = []
    all_names = [nm for nm, _, _ in data.iid]
    active = list(data.iid)
    boundary: list[str] = []
    it = 0
    no_candidates: set[str] = set()
    while True:
        sub = MaternalData(data.y, data.X, data.animal, data.dam, active)
        ev = MaternalREMLEvaluator(sub, k_inv, logdet_k, memory_budget_bytes, dense)
        point, it, hit = _optimize(ev, theta, tol, max_iter, history, it,
                                   frozenset(no_candidates))
        if hit is not None:
            name = active[hit][0]
            history.append({"event": f"{name} fixed at zero (boundary candidate); re-fitting"})
            boundary.append(name)
            theta = np.delete(point.theta, 3 + hit)
            active = [t for j, t in enumerate(active) if j != hit]
            continue
        rejected = None
        for name in boundary:                   # Kuhn-Tucker: score at zero must be <= 0
            _, lev, nl = next(t for t in data.iid if t[0] == name)
            Zd = sp.csr_matrix((np.ones(data.y.size), (np.arange(data.y.size),
                                                       np.asarray(lev, dtype=np.int64))),
                               shape=(data.y.size, int(nl)))
            g0 = _score_at_zero(ev, point.theta, Zd)
            history.append({"event": f"score at zero for {name}", "score": g0})
            if kt_rejects_zero(g0, float(point.theta[-1]), point.loglik):
                rejected = (name, g0)
                break
        if rejected is None:
            break
        name, g0 = rejected
        if name in no_candidates:
            raise ABPError("REML_NOT_CONVERGED", f"maternal REML: {name} repeatedly collapses to "
                           f"zero although the score there is positive ({g0:.3g}); the "
                           "likelihood surface is ill-behaved", term=name, score=g0)
        history.append({"event": f"boundary rejected for {name} (score at zero {g0:.3g} > 0); "
                                 "reinstating the term"})
        no_candidates.add(name)
        boundary.remove(name)
        G0_, vp_, se_ = _unpack(point.theta, ev.n_iid)
        cur = dict(zip(ev.iid_names, vp_))
        active = [t for t in data.iid if t[0] not in boundary]
        theta = np.concatenate([point.theta[:3], [cur.get(t[0], 0.1 * se_) for t in active],
                                [se_]])
    n_iid = ev.n_iid
    G0, vp, se = _unpack(point.theta, n_iid)
    ev_ = np.linalg.eigvalsh(G0)
    if ev_[0] <= MIN_EIG_REL * ev_[-1]:
        raise ABPError("MODEL_NOT_IDENTIFIABLE", "maternal REML: the estimated G0 is (nearly) "
                       f"singular (eigenvalues {ev_.tolist()}): |r_am| -> 1 or a genetic "
                       "variance -> 0; this version does not estimate that boundary",
                       G0=G0.tolist())
    iid = {nm: 0.0 for nm in all_names}
    iid.update(dict(zip(ev.iid_names, map(float, vp))))
    cov = se_out = None
    names = ev.names()
    if not boundary:
        try:
            c = np.linalg.inv(point.ai)
            c = 0.5 * (c + c.T)
            if np.linalg.eigvalsh(c)[0] > 0:
                cov = c
                se_out = {nm: float(math.sqrt(c[i, i])) for i, nm in enumerate(names)}
        except np.linalg.LinAlgError:
            pass
    full = np.concatenate([point.theta[:3], [iid[nm] for nm in all_names], [se]])
    return MaternalREMLFit(G0, iid, se, point.loglik, len([h for h in history if "iteration" in h]),
                           "converged_boundary" if boundary else "converged", se_out, cov, names,
                           _derived(full, cov, n_iid0), ev.trace_method,
                           {"source": source, "theta": theta0.tolist(), "boundary": boundary},
                           history)


def maternal_blup(data: MaternalData, k_inv, G0: np.ndarray, iid_vars, residual: float,
                  memory_budget_bytes: int = 4 * 2**30):
    """Solutions and prediction-error (co)variances at known (co)variances.

    Returns ``(beta, U, Upev, iid_solutions, info)``: ``U`` ``q x 2`` (direct, maternal),
    ``Upev`` ``q x 2 x 2`` (the genetic blocks of ``C^-1``), ``info`` the method, the
    number of equations, the relative residual ``||C s - r|| / ||r||`` and the time."""
    import time
    t0 = time.perf_counter()
    ev = MaternalREMLEvaluator(data, k_inv, 0.0, memory_budget_bytes)
    theta = np.concatenate([[G0[0, 0], G0[0, 1], G0[1, 1]], np.asarray(iid_vars, float),
                            [residual]])
    if not _in_space(theta, ev.n_iid):
        raise ABPError("COVARIANCE_NOT_PD", "maternal BLUP needs G0 positive definite and "
                       "positive variances")
    C = ev._coefficient(theta)
    fac, ent = ev._factor(C)
    r = ev.Wty / residual
    s = fac.solve(r)
    rel_res = float(np.linalg.norm(C @ s - r) / max(np.linalg.norm(r), 1e-300))
    q = ev.q
    U = s[ev.g0:ev.g1].reshape(q, R)
    base = ev.g0 + np.arange(q) * R
    pev = np.empty((q, R, R))
    for j in range(R):
        for k in range(R):
            pev[:, j, k] = ent(base + j, base + k)
    info = {"method": "dense" if ev.dense else "sparse_direct",
            "selection_reason": ("dense Cholesky with the explicit inverse (<= "
                                 f"{DENSE_MAX} equations)" if ev.dense else
                                 "sparse LDL' with selected inversion for the PEV"),
            "n_equations": int(ev.n_eq), "relative_residual": rel_res,
            "iterations": None, "wall_seconds": time.perf_counter() - t0}
    if not rel_res < 1e-8:
        raise ABPError("FACTORIZATION_FAILED", f"maternal BLUP: relative residual {rel_res:.2e} "
                       "of the mixed-model equations is above 1e-8")
    return s[:ev.p], U, pev, [s[a:b] for a, b in ev.p_off], info
