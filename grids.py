from __future__ import annotations

from strategies.follow_the_loser.anticor import Anticor
from strategies.follow_the_loser.cwmr import CWMR
from strategies.follow_the_loser.olmar2 import OLMAR2
from strategies.follow_the_loser.pamr import PAMR
from strategies.follow_the_loser.pamr2 import PAMR2
from strategies.follow_the_winner.eg import EG
from strategies.pattern_matching.corn import CORN
from strategies.pattern_matching.corn_k import CORNK
from strategies.pattern_matching.corn_u import CORNU

# name -> (factory(n, p), candidate values of p, default). Defaults are the
# book's own values where it gives one, the untuned starting point otherwise;
# each must be in its grid, since walk_forward.py uses it before the first
# refit.
GRIDS = {
    "EG": (lambda n, eta: EG(n, eta=eta), [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2], 0.05),
    "Anticor": (lambda n, w: Anticor(n, w=w), [3, 5, 8, 12, 20, 30], 5),
    "PAMR": (lambda n, eps: PAMR(n, eps=eps), [0.1, 0.3, 0.5, 0.8, 1.0], 0.5),
    "PAMR2": (lambda n, C: PAMR2(n, C=C), [0.01, 0.1, 1.0, 10.0, 100.0, 500.0], 1.0),
    "CWMR": (lambda n, phi: CWMR(n, phi=phi), [0.0, 0.25, 0.5, 1.0, 1.64, 2.33], 1.64),
    "OLMAR2": (lambda n, alpha: OLMAR2(n, alpha=alpha), [0.1, 0.3, 0.5, 0.7, 0.9], 0.5),
    "CORN": (
        lambda n, p: CORN(n, w=p[0], rho=p[1]),
        [(w, rho) for w in (3, 5, 7, 10) for rho in (0.0, 0.1, 0.3)],
        (5, 0.1),
    ),
    "CORN-U": (lambda n, rho: CORNU(n, rho=rho), [0.0, 0.1, 0.3], 0.1),
    "CORN-K": (lambda n, K: CORNK(n, K=K), [1, 3, 5], 3),
}

SLOW = ["CORN", "CORN-U", "CORN-K"]

# experts combined by Hedge with Fixed Share (strategies/meta_algorithms/
# fixed_share.py) -- every non-hindsight strategy at its default parameter;
# CORN is added only when INCLUDE_CORN is on
EXPERTS = ["BAH", "CRP", "UP", "EG", "Anticor", "PAMR2", "CWMR", "OLMAR2"]

# (eta, alpha); eta=1 is the Bayesian mixture, alpha=0 plain Hedge
FIXED_SHARE_GRID = [(eta, alpha) for eta in (0.5, 1.0, 2.0, 5.0) for alpha in (0.0, 0.001, 0.01, 0.1)]
FIXED_SHARE_DEFAULT = (1.0, 0.01)
