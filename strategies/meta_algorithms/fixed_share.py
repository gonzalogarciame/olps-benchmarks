from __future__ import annotations

import numpy as np

from engine import chain_segments, wealth_from_portfolios


def fixed_share(b_experts: np.ndarray, x: np.ndarray, eta: float, alpha: float) -> np.ndarray:
    # Hedge with Fixed Share (Herbster & Warmuth, 1998) over K experts'
    # portfolios, b_experts[i, t]. q is the weight on each expert: after
    # period t it is multiplied by that expert's own gross return to the
    # power eta (eta=1 is the Bayesian mixture, which is CORN-U's weighting
    # over its experts), then alpha of the mass is spread uniformly so no
    # expert's weight can fall too far to recover when the market changes.
    # alpha=0 is plain Hedge. The meta portfolio is the q-weighted mix of
    # the experts' portfolios, rebalanced every period.
    K, T, n = b_experts.shape
    q = np.full(K, 1.0 / K)
    b = np.empty((T, n))
    for t in range(T):
        b[t] = q @ b_experts[:, t]
        q = q * (b_experts[:, t] @ x[t]) ** eta
        q /= q.sum()
        q = (1 - alpha) * q + alpha / K
    return b


def fixed_share_wealth(
    experts: list[list[np.ndarray]], segments: list[np.ndarray], eta: float, alpha: float
) -> np.ndarray:
    # experts[i][j] is expert i's portfolios on segment j; the combiner
    # restarts at each segment, like every other strategy (asset columns
    # change at a reconstitution)
    return chain_segments(
        [
            wealth_from_portfolios(fixed_share(np.stack([e[j] for e in experts]), x, eta, alpha), x)
            for j, x in enumerate(segments)
        ]
    )
