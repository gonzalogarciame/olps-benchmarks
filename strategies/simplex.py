from __future__ import annotations

import numpy as np


def project_to_simplex(y: np.ndarray) -> np.ndarray:
    # Euclidean projection onto the simplex (min ||b - y||^2 s.t. b >= 0,
    # sum(b) = 1), used by PAMR/CWMR/OLMAR after their unconstrained update
    # steps. Closed-form sort-based algorithm (Duchi et al. 2008, cited by
    # the book, e.g. p.91) rather than a generic QP solve like ONS's
    # cvxpy-based _project -- PAMR/CWMR/OLMAR project every period inside
    # an O(n) history loop that itself gets recomputed from scratch at
    # every period, so an O(m log m) closed form (vs. a solver call) is
    # what keeps a multi-year backtest tractable.
    u = np.sort(y)[::-1]
    css = np.cumsum(u) - 1
    idx = np.arange(1, len(y) + 1)
    rho = idx[u - css / idx > 0][-1]
    theta = css[rho - 1] / rho
    return np.maximum(y - theta, 0.0)
