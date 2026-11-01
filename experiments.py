from __future__ import annotations

import argparse
import hashlib
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from data import benchmark_wealth
from engine import run_backtest_segmented
from grids import EXPERTS, FIXED_SHARE_DEFAULT, FIXED_SHARE_GRID, GRIDS, POLICY_SEEDS, SLOW
from metrics import apy, calmar_ratio, max_drawdown, sharpe_ratio, t_test, volatility
from strategies.benchmarks.bah import BAH
from strategies.benchmarks.best_stock import BestStock
from strategies.benchmarks.crp import CRP
from strategies.follow_the_winner.ons import ONS
from strategies.follow_the_winner.up import UP
from strategies.meta_algorithms.fixed_share import fixed_share_wealth
from strategies.meta_algorithms.policy_gradient import policy_experts_wealth
from walk_forward import segmented_wealth, strategy_portfolios, walk_forward, walk_forward_all

# Each strategy's wealth curve is kept in results/<universe>/wealth.csv with
# the hash of everything it depends on in keys.csv. A run recomputes a
# strategy only when its hash has changed or it is missing, and saves it as
# soon as it finishes, so a long run is not lost if a later one fails.

RESULTS_DIR = Path("results")

FIXED = {
    "BAH": lambda n, p: BAH(n),
    "CRP": lambda n, p: CRP(n),
    "UP": lambda n, p: UP(n),
    "ONS": lambda n, p: ONS(n),
}
COMBINERS = ["FixedShare", "ExpertsEW", "PolicyExperts"]
ALL = ["BestStock", "SPY", *FIXED, *GRIDS, *COMBINERS]

SEED_PREFIX = "PolicyExperts/seed"


@dataclass
class Universe:
    name: str
    start: str
    end: str
    segments_meta: list
    load: Callable[[list], list[pd.DataFrame]]
    include_ons: bool
    include_corn: bool

    @cached_property
    def dfs(self) -> list[pd.DataFrame]:
        dfs = self.load(self.segments_meta)
        print(f"{sum(len(df) for df in dfs)} periods across {len(dfs)} segments:")
        for (s, e, tickers), df in zip(self.segments_meta, dfs):
            print(f"  {s} .. {e}: {tickers if len(tickers) <= 8 else f'{len(tickers)} assets'}")
        return dfs

    @cached_property
    def segment_arrays(self) -> list[np.ndarray]:
        return [df.values for df in self.dfs]

    @cached_property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex([]).append([df.index for df in self.dfs])

    @cached_property
    def plot_dates(self) -> pd.DatetimeIndex:
        # wealth has one more point than the periods (S_0 = 1 before the first)
        return self.dates.insert(0, self.dates[0] - pd.Timedelta(days=1))

    @property
    def expert_names(self) -> list[str]:
        return EXPERTS + (["CORN"] if self.include_corn else [])

    @cached_property
    def experts(self) -> list[list[np.ndarray]]:
        out = []
        for name in self.expert_names:
            factory, param = (FIXED[name], None) if name in FIXED else (GRIDS[name][0], GRIDS[name][2])
            out.append(strategy_portfolios(self.name, name, param, factory, self.segment_arrays))
        return out

    def default_names(self) -> list[str]:
        return [
            n for n in ALL if (n != "ONS" or self.include_ons) and (n not in SLOW or self.include_corn)
        ]


def key(u: Universe, name: str) -> str:
    # meta_algorithms/ and corn_sig.py are only hashed into the strategies
    # that use them, so editing a combiner leaves the experts' results valid.
    # The data itself is not hashed: the window and constituents are, which
    # is what lets a run with nothing to compute skip the download.
    h = hashlib.sha1()
    for path in sorted(Path("strategies").rglob("*.py")):
        if "meta_algorithms" in path.parts and name not in COMBINERS:
            continue
        if path.name == "corn_sig.py" and name != "CORN-Sig":
            continue
        h.update(path.read_bytes())
    for path in ("engine.py", "walk_forward.py"):
        h.update(Path(path).read_bytes())
    h.update(repr((u.name, u.start, u.end, u.segments_meta)).encode())
    if name in GRIDS:
        h.update(repr(GRIDS[name][1:]).encode())
    if name in COMBINERS:
        grids = [GRIDS[e][1:] for e in u.expert_names if e in GRIDS]
        h.update(repr((u.expert_names, grids, FIXED_SHARE_GRID, FIXED_SHARE_DEFAULT, POLICY_SEEDS)).encode())
    return h.hexdigest()[:16]


def compute(u: Universe, name: str) -> tuple[np.ndarray, dict | None, dict[str, np.ndarray]]:
    segments = u.segment_arrays
    if name == "BestStock":
        # hindsight upper bound, not an expert
        return run_backtest_segmented(lambda n, pr: BestStock(n, pr), segments), None, {}
    if name == "SPY":
        return benchmark_wealth("SPY", u.start, u.end, u.dates), None, {}
    if name in FIXED:
        b = strategy_portfolios(u.name, name, None, FIXED[name], segments)
        return segmented_wealth(b, segments), None, {}
    if name in GRIDS:
        wealths, picks, _ = walk_forward_all(u.name, {name: GRIDS[name]}, segments, u.dates)
        return wealths[name], picks[name], {}
    if name == "FixedShare":
        candidates = [fixed_share_wealth(u.experts, segments, eta, alpha) for eta, alpha in FIXED_SHARE_GRID]
        S, chosen = walk_forward(candidates, FIXED_SHARE_GRID.index(FIXED_SHARE_DEFAULT), u.dates)
        return S, {year: FIXED_SHARE_GRID[i] for year, i in chosen.items()}, {}
    if name == "ExpertsEW":
        return fixed_share_wealth(u.experts, segments, 0.0, 0.0), None, {}
    S, by_seed = policy_experts_wealth(u.experts, segments, u.dates, POLICY_SEEDS)
    return S, None, {f"{SEED_PREFIX}{s}": w for s, w in zip(POLICY_SEEDS, by_seed)}


@contextmanager
def _lock(directory: Path):
    # runs of different strategies can go in parallel; each one's save is a
    # read-merge-write of the same CSVs, so it must not interleave
    path = directory / ".lock"
    directory.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            os.close(os.open(path, os.O_CREAT | os.O_EXCL))
            break
        except FileExistsError:
            try:
                if time.time() - path.stat().st_mtime > 60:  # left by a run that was killed
                    path.unlink()
            except FileNotFoundError:
                pass
            time.sleep(0.1)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


class Store:
    def __init__(self, universe: str):
        self.dir = RESULTS_DIR / universe
        self.load()

    def load(self) -> None:
        wealth, keys, picks = self.dir / "wealth.csv", self.dir / "keys.csv", self.dir / "picks.csv"
        self.wealth = pd.read_csv(wealth, index_col="date", parse_dates=True) if wealth.exists() else pd.DataFrame()
        self.keys = pd.read_csv(keys, index_col="column")["key"].to_dict() if keys.exists() else {}
        self.picks = pd.read_csv(picks) if picks.exists() else pd.DataFrame(columns=["strategy", "year", "param"])

    def valid(self, name: str, k: str) -> bool:
        return name in self.wealth.columns and self.keys.get(name) == k

    def put(self, name, wealth, picks, extra, k, index: pd.DatetimeIndex) -> None:
        with _lock(self.dir):
            self.load()
            if len(self.wealth) and not self.wealth.index.equals(index):
                print("stored results cover different dates; discarding them")
                self.wealth, self.keys = pd.DataFrame(), {}
                self.picks = self.picks.iloc[0:0]
            if not len(self.wealth):
                self.wealth = pd.DataFrame(index=index)
            for column, w in {name: wealth, **extra}.items():
                self.wealth[column] = w
                self.keys[column] = k
            self.picks = self.picks[self.picks["strategy"] != name]
            if picks:
                rows = pd.DataFrame({"strategy": name, "year": list(picks), "param": [repr(p) for p in picks.values()]})
                self.picks = pd.concat([self.picks, rows], ignore_index=True)
            self.save()

    def save(self) -> None:
        # write then rename, so a run reading at the same moment never sees a half-written file
        for filename, write in (
            ("wealth.csv", lambda p: self.wealth.to_csv(p, index_label="date")),
            ("keys.csv", lambda p: pd.Series(self.keys, name="key").to_csv(p, index_label="column")),
            ("picks.csv", lambda p: self.picks.to_csv(p, index=False)),
        ):
            tmp = self.dir / (filename + ".tmp")
            write(tmp)
            os.replace(tmp, self.dir / filename)


def metrics_row(wealth: np.ndarray, spy_apy: float) -> dict:
    t_stat, p_value = t_test(wealth)
    return {
        "final_wealth": wealth[-1],
        "apy": apy(wealth),
        "volatility": volatility(wealth),
        "sharpe": sharpe_ratio(wealth),
        "max_drawdown": max_drawdown(wealth),
        "calmar": calmar_ratio(wealth),
        "t_stat": t_stat,
        "p_value": p_value,
        "apy_vs_spy": apy(wealth) - spy_apy,
    }


def report(name: str, m: dict) -> None:
    print(
        f"{name:10s} final wealth = {m['final_wealth']:.4f}  APY = {m['apy']:+.2%}  "
        f"vol = {m['volatility']:.2%}  Sharpe = {m['sharpe']:.2f}  "
        f"MDD = {m['max_drawdown']:.2%}  Calmar = {m['calmar']:.2f}  "
        f"t = {m['t_stat']:.2f} (p = {m['p_value']:.3f})  vs SPY = {m['apy_vs_spy']:+.2%}"
    )


def update(u: Universe, only: list[str] | None, force: bool, no_run: bool = False):
    store = Store(u.name)
    keys = {name: key(u, name) for name in ALL}
    targets = ["SPY"] + [n for n in (only or u.default_names()) if n != "SPY"]
    for name in targets:
        if not force and store.valid(name, keys[name]):
            continue
        print(f"computing {name} ...")
        t0 = time.time()
        wealth, picks, extra = compute(u, name)
        store.put(name, wealth, picks, extra, keys[name], u.plot_dates)
        print(f"  {name} done in {time.time() - t0:.0f}s, saved to {store.dir}")

    wealths = {n: store.wealth[n].values for n in ALL if store.valid(n, keys[n])}
    stale = [n for n in ALL if n in store.wealth.columns and n not in wealths]
    if stale:
        print(f"stored but outdated, not shown (rerun with --only): {', '.join(stale)}")

    if "SPY" not in wealths:
        raise SystemExit(f"nothing stored in {store.dir} yet; run once without --no-run")
    spy_apy = apy(wealths["SPY"])
    seeds = {
        c: store.wealth[c].values
        for c in store.wealth.columns
        if c.startswith(SEED_PREFIX) and "PolicyExperts" in wealths
    }
    rows = {name: metrics_row(w, spy_apy) for name, w in {**wealths, **seeds}.items()}
    pd.DataFrame(rows).T.to_csv(store.dir / "metrics.csv", index_label="strategy")
    for name in wealths:
        report(name, rows[name])
    if "PolicyExperts" in wealths and seeds:
        a = [rows[c]["apy"] for c in seeds]
        print(f"PolicyExperts per seed: APY {np.mean(a):+.2%} +- {np.std(a):.2%}, min {min(a):+.2%}, max {max(a):+.2%}")
    for name, group in store.picks.groupby("strategy", sort=False):
        if name in wealths:
            by_year = ", ".join(f"{y}: {p}" for y, p in zip(group["year"], group["param"]))
            print(f"{name:10s} chosen by year: {{{by_year}}}")
    return wealths, store.wealth.index


def print_status(u: Universe) -> None:
    store = Store(u.name)
    default = u.default_names()
    for name in ALL:
        if store.valid(name, key(u, name)):
            status = "stored"
        elif name in store.wealth.columns:
            status = "outdated"
        else:
            status = "missing"
        print(f"{name:14s} {status:9s} {'' if name in default else '(off by default)'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="+", choices=ALL, metavar="NAME", help="run just these strategies")
    p.add_argument("--force", action="store_true", help="recompute even if stored results are up to date")
    p.add_argument("--no-run", action="store_true", help="compute nothing, use stored results only")
    p.add_argument("--plot", nargs="+", choices=ALL, metavar="NAME", help="draw only these strategies (and SPY) in one custom plot")
    p.add_argument("--list", action="store_true", help="show which strategies are stored, then exit")
    return p.parse_args()
