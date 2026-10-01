from __future__ import annotations

import itertools
import math
import os

import matplotlib.pyplot as plt
import pandas as pd

import universes.djia_universe as djia_universe
from data import get_price_relatives
from engine import run_backtest_segmented
from metrics import apy, calmar_ratio, max_drawdown, sharpe_ratio, t_test, volatility
from strategies.benchmarks.bah import BAH
from strategies.benchmarks.best_stock import BestStock
from strategies.benchmarks.crp import CRP
from strategies.follow_the_loser.anticor import Anticor
from strategies.follow_the_loser.cwmr import CWMR
from strategies.follow_the_loser.olmar import OLMAR
from strategies.follow_the_loser.olmar2 import OLMAR2
from strategies.follow_the_loser.pamr2 import PAMR2
from strategies.follow_the_winner.eg import EG
from strategies.follow_the_winner.ons import ONS
from strategies.follow_the_winner.up import UP

# 2000-01-01 .. 2026-09-01: as broad a window as djia_universe's verified
# reconstitution history covers, chosen so the result isn't a single
# deliberately-picked regime (a bull year, a crisis) but whatever actually
# happened across the dot-com bust, 2008, the 2009-2019 bull run, COVID,
# and the recent AI-driven run, with the Dow's real membership changes
# tracked throughout -- see universes/djia_universe.py.
START = "2000-01-01"
END = "2026-09-01"

# ONS replays its full history on every call and solves a cvxpy QP at each
# step of that replay (Section ons.py / paper, Sec. 4.4-ish) -- O(n^2)
# solver calls over a backtest of n periods. That cost a few minutes at
# n~650 (the 2007-2009 window); at n~6700 (this window) it's an estimated
# 50+ hours, benchmarked directly rather than guessed. Every other
# strategy here is plain numpy and finishes in minutes regardless. Off by
# default for that reason; flip to True to include it anyway.
INCLUDE_ONS = False

OUTPUT_DIR = "outputs"
PLOT_PATH = f"{OUTPUT_DIR}/wealth_curves.png"
GROUPED_PLOT_PATH = f"{OUTPUT_DIR}/wealth_curves_grouped.png"
GRID_PLOT_PATH = f"{OUTPUT_DIR}/wealth_curves_grid.png"

# distinct dash patterns so nearly-overlapping wealth curves stay
# distinguishable even where color alone would not separate them
LINESTYLES = ["-", "--", ":", "-.", (0, (3, 1, 1, 1)), (0, (5, 1))]

# BAH/CRP/UP/EG never make a large, concentrated single-period bet, so they
# stay close to each other and close to 1; the rest make aggressive,
# short-horizon bets that paid off badly in this crash -- the two groups
# need different axes to both be readable (see paper, Current Results).
STEADY_GROUP = ["BAH", "CRP", "UP", "EG"]
AGGRESSIVE_GROUP = ["ONS", "Anticor", "PAMR2", "CWMR", "OLMAR", "OLMAR2"]


def _load_segment(seg_start: str, seg_end: str, tickers: list[str]) -> pd.DataFrame:
    fetch_symbols = [djia_universe.fetch_symbol(t) for t in tickers]
    df = get_price_relatives(fetch_symbols, seg_start, seg_end)
    df = df[fetch_symbols]  # fixed column order, undoing yfinance's alphabetical sort
    df.columns = tickers  # display names (undoes the UTX->RTX fetch alias)
    return df


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    segments_meta = djia_universe.segments(START, END)
    segment_dfs = [_load_segment(s, e, tickers) for s, e, tickers in segments_meta]
    segment_arrays = [df.values for df in segment_dfs]

    dates = pd.DatetimeIndex([]).append([df.index for df in segment_dfs])
    # wealth has one more point than the combined periods (S_0 = 1 before
    # any period), so prepend a date for that point too
    plot_dates = dates.insert(0, dates[0] - pd.Timedelta(days=1))

    strategy_factories = {
        "BAH": lambda n, pr: BAH(n),
        "CRP": lambda n, pr: CRP(n),
        "BestStock": lambda n, pr: BestStock(n, pr),
        "UP": lambda n, pr: UP(n),
        "EG": lambda n, pr: EG(n),
        "Anticor": lambda n, pr: Anticor(n),
        # plain PAMR and PAMR-1 are left out of this headline run -- both
        # collapse the same way (Sec. ftl-pamr-tuning in the paper has the
        # full comparison); PAMR2 (C=1.0, the book's own worked-example
        # value) is the one that actually works, so it's what's shown here.
        "PAMR2": lambda n, pr: PAMR2(n),
        "CWMR": lambda n, pr: CWMR(n),
        "OLMAR": lambda n, pr: OLMAR(n),
        "OLMAR2": lambda n, pr: OLMAR2(n),
    }
    if INCLUDE_ONS:
        strategy_factories["ONS"] = lambda n, pr: ONS(n)

    total_periods = sum(arr.shape[0] for arr in segment_arrays)
    print(f"{total_periods} periods across {len(segment_arrays)} DJIA reconstitution segments:")
    for seg_start, seg_end, tickers in segments_meta:
        print(f"  {seg_start} .. {seg_end}: {len(tickers)} assets")

    wealths = {}
    for name, factory in strategy_factories.items():
        wealth = run_backtest_segmented(factory, segment_arrays)
        t_stat, p_value = t_test(wealth)
        print(
            f"{name:10s} final wealth = {wealth[-1]:.4f}  APY = {apy(wealth):+.2%}  "
            f"vol = {volatility(wealth):.2%}  Sharpe = {sharpe_ratio(wealth):.2f}  "
            f"MDD = {max_drawdown(wealth):.2%}  Calmar = {calmar_ratio(wealth):.2f}  "
            f"t = {t_stat:.2f} (p = {p_value:.3f})"
        )
        wealths[name] = wealth

    _plot_combined(wealths, plot_dates)
    _plot_grouped(wealths, plot_dates)
    _plot_grid(wealths, plot_dates)


def _plot_combined(wealths: dict, plot_dates: pd.DatetimeIndex) -> None:
    fig, ax = plt.subplots()
    for (name, wealth), linestyle in zip(wealths.items(), itertools.cycle(LINESTYLES)):
        # BestStock is a hindsight-only upper bound (see strategies/best_stock.py),
        # not a real strategy, so it's left off the graph -- printed above for
        # reference, but not plotted alongside strategies that make causal decisions
        if name != "BestStock":
            ax.plot(plot_dates, wealth, label=name, linestyle=linestyle)

    ax.set_xlabel("date")
    ax.set_ylabel("cumulative wealth (log scale)")
    ax.set_yscale("log")
    ax.set_title(f"OLPS strategies on DJIA constituents, {START} to {END}")
    ax.legend()
    fig.autofmt_xdate()
    fig.savefig(PLOT_PATH)
    print(f"saved plot to {PLOT_PATH}")


def _plot_grouped(wealths: dict, plot_dates: pd.DatetimeIndex) -> None:
    fig, (ax_steady, ax_aggressive) = plt.subplots(2, 1, figsize=(6.4, 7.2), sharex=True)

    steady = [name for name in STEADY_GROUP if name in wealths]
    for name, linestyle in zip(steady, itertools.cycle(LINESTYLES)):
        ax_steady.plot(plot_dates, wealths[name], label=name, linestyle=linestyle)
    ax_steady.set_ylabel("cumulative wealth")
    ax_steady.set_title("Steady: no concentrated single-period bets")
    ax_steady.legend()

    aggressive = [name for name in AGGRESSIVE_GROUP if name in wealths]
    for name, linestyle in zip(aggressive, itertools.cycle(LINESTYLES)):
        ax_aggressive.plot(plot_dates, wealths[name], label=name, linestyle=linestyle)
    ax_aggressive.set_ylabel("cumulative wealth (log scale)")
    ax_aggressive.set_yscale("log")
    ax_aggressive.set_title("Aggressive: large single-period rebalancing bets")
    ax_aggressive.set_xlabel("date")
    ax_aggressive.legend()

    fig.suptitle(f"OLPS strategies on DJIA constituents, {START} to {END}")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(GROUPED_PLOT_PATH)
    print(f"saved plot to {GROUPED_PLOT_PATH}")


def _plot_grid(wealths: dict, plot_dates: pd.DatetimeIndex) -> None:
    names = [name for name in wealths if name != "BestStock"]
    n_cols = math.ceil(math.sqrt(len(names)))
    n_rows = math.ceil(len(names) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 3.2 * n_rows), sharex=True, sharey=True)
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]
    for ax, name in zip(axes_flat, names):
        ax.plot(plot_dates, wealths[name])
        ax.axhline(1.0, color="gray", linewidth=0.5, linestyle=":")
        ax.set_yscale("log")
        ax.set_title(name)
        ax.tick_params(axis="x", labelrotation=45)
    for ax in list(axes_flat)[len(names):]:
        ax.set_visible(False)

    fig.supxlabel("date")
    fig.supylabel("cumulative wealth (log scale)")
    fig.suptitle(f"OLPS strategies on DJIA constituents, {START} to {END}")
    fig.tight_layout()
    fig.savefig(GRID_PLOT_PATH)
    print(f"saved plot to {GRID_PLOT_PATH}")


if __name__ == "__main__":
    main()
