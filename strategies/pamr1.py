from __future__ import annotations

from strategies.pamr import PAMR


class PAMR1(PAMR):
    # PAMR-1 (book Eq. 9.3/9.7): caps tau at C via a linear slack penalty,
    # trading some of plain PAMR's aggressiveness for robustness against a
    # single noisy price relative -- the book's own fix for the exact
    # failure mode traced in plain PAMR during the 2008 crash (an
    # uncapped tau driving b to a single asset off one day's return).
    # C=1.0 is the book's own worked-example value (Sec. 9.4: eps=0.30,
    # C=1.00), used deliberately instead of a value chosen by sweeping C
    # on this project's own data: the book's tested range (50-5000, Fig.
    # 13.6) reports insensitivity, but a sweep here found C has to drop
    # well below that range before the cap does anything on real data --
    # C=1.0 sits in that effective region without being the specific
    # value a sweep on this data happened to score best.
    def __init__(self, n_assets: int, eps: float = 0.5, C: float = 1.0):
        super().__init__(n_assets, eps)
        self._C = C

    def _tau(self, loss: float, denom: float) -> float:
        return min(self._C, loss / denom) if denom > 0 else 0.0
