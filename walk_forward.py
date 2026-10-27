from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from engine import chain_segments, run_portfolios, wealth_from_portfolios
from metrics import sharpe_ratio
from strategies.base import Strategy

# b = portfolio, x = price relatives, S = cumulative wealth; "candidates"
# are the values of one strategy's parameter being compared

CACHE_DIR = Path("cache")


def _fingerprint(segments: list[np.ndarray]) -> str:
    # Cached portfolios are only valid for the same strategy code and the
    # same price data. meta_algorithms/ is left out of the code hash so
    # adding a combiner does not invalidate hours of expert runs.
    h = hashlib.sha1()
    for path in sorted(Path("strategies").rglob("*.py")):
        if "meta_algorithms" not in path.parts:
            h.update(path.read_bytes())
    h.update(Path("engine.py").read_bytes())
    for segment in segments:
        h.update(np.ascontiguousarray(segment).tobytes())
    return h.hexdigest()


def strategy_portfolios(
    universe: str,
    name: str,
    param: object,
    factory: Callable[[int, object], Strategy],
    segments: list[np.ndarray],
) -> list[np.ndarray]:
    # b_1..b_T for each segment. A run costs minutes to hours (every
    # strategy replays its history at each step), and both the parameter
    # selection and the combiners need the same portfolios, so each is
    # computed once and kept in cache/.
    key = "".join(c if c.isalnum() or c in "._-" else "_" for c in f"{name}__{param!r}")
    path = CACHE_DIR / universe / f"{key}.npz"
    fingerprint = _fingerprint(segments)
    if path.exists():
        with np.load(path) as f:
            if str(f["fingerprint"]) == fingerprint:
                return [f[f"b{j}"] for j in range(len(segments))]
    b_segments = [
        run_portfolios(factory(segment.shape[1], param), segment) for segment in segments
    ]
    os.makedirs(path.parent, exist_ok=True)
    np.savez(path, fingerprint=fingerprint, **{f"b{j}": b for j, b in enumerate(b_segments)})
    return b_segments


def segmented_wealth(b_segments: list[np.ndarray], segments: list[np.ndarray]) -> np.ndarray:
    return chain_segments([wealth_from_portfolios(b, x) for b, x in zip(b_segments, segments)])


def walk_forward(
    wealths: list[np.ndarray],
    default: int,
    dates: pd.DatetimeIndex,
    warmup_years: int = 3,
) -> tuple[np.ndarray, dict[int, int]]:
    # Each strategy's b_t depends only on x_1..x_{t-1}, so a candidate's
    # wealth path up to a date d is the same whether it was simulated up to
    # d or over the whole sample. One full run per candidate is therefore
    # enough: choosing at d by Sharpe over wealth[:k+1] uses nothing from
    # after d, and the chosen candidate's returns from d onward are exactly
    # what trading with that parameter from d would have earned.
    s = np.array([S[1:] / S[:-1] for S in wealths])
    chosen = np.full(len(dates), default)
    picks = {}
    for year in range(dates[0].year + warmup_years, dates[-1].year + 1):
        d = pd.Timestamp(year, 1, 1)
        k = dates.searchsorted(d)
        sharpes = np.nan_to_num([sharpe_ratio(S[: k + 1]) for S in wealths], nan=-np.inf)
        best = int(np.argmax(sharpes))
        chosen[(dates >= d) & (dates < pd.Timestamp(year + 1, 1, 1))] = best
        picks[year] = best
    S = np.concatenate([[1.0], np.cumprod(s[chosen, np.arange(len(dates))])])
    return S, picks


def walk_forward_all(
    universe: str, grids: dict, segments: list[np.ndarray], dates: pd.DatetimeIndex
) -> tuple[dict, dict, dict]:
    wealths, picks, default_b = {}, {}, {}
    for name, (factory, grid, default) in grids.items():
        b_candidates = [strategy_portfolios(universe, name, p, factory, segments) for p in grid]
        S, chosen = walk_forward(
            [segmented_wealth(b, segments) for b in b_candidates], grid.index(default), dates
        )
        wealths[name] = S
        picks[name] = {year: grid[i] for year, i in chosen.items()}
        default_b[name] = b_candidates[grid.index(default)]
    return wealths, picks, default_b
