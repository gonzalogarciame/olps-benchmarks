from __future__ import annotations

import cvxpy as cp
import numpy as np

from strategies.pattern_matching.corn import CORN


class CORNSig(CORN):
    # CORN with the correlation test replaced by a distance between path
    # signatures. A window of w price-relative vectors is read as the path
    # of cumulative log relatives (plus a time coordinate, so that the
    # signature sees the order of the days), and its signature truncated
    # at level 2 is the feature vector. The L2 distance between two such
    # vectors is the truncated Sig-W1 metric of Ni et al. (2021) between
    # the two windows seen as point masses on path space; it is the only
    # place Wasserstein enters, since one window is one path and not a
    # distribution of paths. Everything else is CORN: the day after each
    # similar window is collected and b is the BCRP over those days.
    #
    # Distances have no natural threshold the way a correlation does, so
    # instead of rho the fraction q of earlier windows closest to the
    # latest one is kept, at least one.
    def __init__(self, n_assets: int, w: int = 5, q: float = 0.1):
        super().__init__(n_assets, w=w)
        self._q = q

    def update(self, history: np.ndarray) -> np.ndarray:
        t = len(history)
        w = self._w
        if t <= w:
            return self.initial_portfolio()

        # windows[j] = x_{j+1..j+w}; window j is followed by x_{j+w+1}
        sig = self.signature(self._windows(np.log(history)))
        latest, past = sig[-1], sig[:-1]
        if len(past) == 0:
            return self.initial_portfolio()

        dist = np.linalg.norm(past - latest, axis=1)
        k = max(1, int(np.ceil(self._q * len(past))))
        nearest = np.argsort(dist, kind="stable")[:k]
        return self._bcrp(history[nearest + w])

    def _bcrp(self, similar_days: np.ndarray) -> np.ndarray:
        # Clarabel gives up on some long, ill-conditioned sets of matched
        # days (the q-nearest set can be over a thousand days); SCS is less
        # exact but always returns, and keeps the run from dying hours in.
        try:
            return super()._bcrp(similar_days)
        except cp.error.SolverError:
            b = cp.Variable(self.n_assets)
            cp.Problem(cp.Maximize(cp.sum(cp.log(similar_days @ b))), [b >= 0, cp.sum(b) == 1]).solve(solver=cp.SCS)
            if b.value is None:
                return self.initial_portfolio()
            b = np.maximum(b.value, 0.0)
            return b / b.sum()

    def _windows(self, logx: np.ndarray) -> np.ndarray:
        w = self._w
        idx = np.arange(len(logx) - w + 1)[:, None] + np.arange(w)
        return logx[idx]

    @staticmethod
    def signature(d: np.ndarray) -> np.ndarray:
        # d: (M, w, n) increments of each path. Level 1 is the total
        # increment; level 2 is sum_{k<l} d_k^i d_l^j + 1/2 sum_k d_k^i d_k^j.
        M, w, n = d.shape
        d = np.concatenate([d, np.full((M, w, 1), 1.0 / w)], axis=2)
        before = np.cumsum(d, axis=1) - d
        level1 = d.sum(axis=1)
        level2 = np.einsum("mki,mkj->mij", before, d) + 0.5 * np.einsum("mki,mkj->mij", d, d)
        return np.concatenate([level1, level2.reshape(M, -1)], axis=1)
