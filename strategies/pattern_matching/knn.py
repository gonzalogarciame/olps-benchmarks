from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from strategies.pattern_matching.corn import CORN

# x = price relatives, b = portfolio, w = window length, q = share of earlier
# windows kept as neighbours. m = number of assets, G = grid points.

# Grid on which the distributions of a window's log relatives are compared.
# Returns beyond +-10% in a day are clipped onto the end points.
_LIMIT = 0.1
_EDGES = np.linspace(-_LIMIT, _LIMIT, 41)[1:]
_STEP = _EDGES[1] - _EDGES[0]


class BNN(CORN):
    # Nearest-neighbour sample selection (Gyorfi et al. 2008, book Sec. 6.1)
    # followed by the log-optimal portfolio over the matched days. The k
    # neighbours of the latest window among all earlier windows are taken by
    # Euclidean distance between the windows as w*m vectors of price
    # relatives. k is a fraction q of the earlier windows, as in CORN-Sig, so
    # that it grows with the history as the book's consistency result needs
    # and the three variants differ only in the distance.
    def __init__(self, n_assets: int, w: int = 5, q: float = 0.1):
        super().__init__(n_assets, w=w)
        self._q = q

    def update(self, history: np.ndarray) -> np.ndarray:
        t = len(history)
        if t <= self._w:
            return self.initial_portfolio()

        dist = self._distances(history)
        k = max(1, int(np.ceil(self._q * len(dist))))
        nearest = np.argsort(dist, kind="stable")[:k]
        # window j is followed by day j+w
        return self._bcrp(history[nearest + self._w])

    def _windows(self, x: np.ndarray, start: int = 0) -> np.ndarray:
        w = self._w
        idx = np.arange(start, len(x) - w + 1)[:, None] + np.arange(w)
        return x[idx]

    def _distances(self, history: np.ndarray) -> np.ndarray:
        flat = self._windows(history).reshape(len(history) - self._w + 1, -1)
        return np.linalg.norm(flat[:-1] - flat[-1], axis=1)


class BNNDist(BNN):
    # BNN with the Euclidean distance replaced by a Wasserstein-1 distance
    # between the distributions of each asset's w log relatives in the window.
    # In one dimension W1 is the area between the two cumulative distribution
    # functions, so each window is stored as its CDF on a fixed grid and the
    # distance is the summed area over the assets. The estimate of the
    # distribution is the choice:
    #   hist   the w values binned on the grid (a discrete probability)
    #   kde    a Gaussian kernel density, Silverman bandwidth
    #   gauss  a normal fitted by mean and standard deviation
    # The windows are compared asset by asset, so the order of the days inside
    # a window and the dependence between assets do not enter the distance.
    def __init__(self, n_assets: int, w: int = 20, q: float = 0.1, density: str = "hist"):
        super().__init__(n_assets, w=w, q=q)
        self._density = density
        self._cdfs = np.empty((0, n_assets, len(_EDGES)), dtype=np.float32)

    def _distances(self, history: np.ndarray) -> np.ndarray:
        n_windows = len(history) - self._w + 1
        if n_windows < len(self._cdfs):
            # called out of order: a past window's CDF only depends on its own
            # days, so only a shorter history needs the store to be rebuilt
            self._cdfs = self._cdfs[:0]
        logx = np.clip(np.log(history), -_LIMIT, _LIMIT)
        for start in range(len(self._cdfs), n_windows, 256):
            block = self._windows(logx[: min(start + 256 + self._w - 1, len(logx))], start)
            self._cdfs = np.concatenate([self._cdfs, self._cdf(block).astype(np.float32)])
        return np.abs(self._cdfs[:-1] - self._cdfs[-1]).sum(axis=(1, 2)) * _STEP

    def _cdf(self, windows: np.ndarray) -> np.ndarray:
        # windows: (B, w, m) -> (B, m, G)
        if self._density == "hist":
            return (windows[..., None] <= _EDGES).mean(axis=1)
        if self._density == "kde":
            h = np.maximum(1.06 * windows.std(axis=1) * self._w ** -0.2, 1e-3)
            return ndtr((_EDGES - windows[..., None]) / h[:, None, :, None]).mean(axis=1)
        mu, sd = windows.mean(axis=1), np.maximum(windows.std(axis=1), 1e-3)
        return ndtr((_EDGES - mu[..., None]) / sd[..., None])
