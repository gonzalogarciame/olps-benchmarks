from __future__ import annotations

import itertools
import math
import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

import universes.djia_universe as djia_universe
from data import benchmark_wealth, get_price_relatives
from engine import run_backtest_segmented
from grids import EXPERTS, FIXED_SHARE_DEFAULT, FIXED_SHARE_GRID, GRIDS, SLOW
from metrics import apy, calmar_ratio, max_drawdown, sharpe_ratio, t_test, volatility
from strategies.benchmarks.bah import BAH
from strategies.benchmarks.best_stock import BestStock
from strategies.benchmarks.crp import CRP
from strategies.follow_the_winner.ons import ONS
from strategies.follow_the_winner.up import UP
from strategies.meta_algorithms.fixed_share import fixed_share_wealth
from walk_forward import segmented_wealth, strategy_portfolios, walk_forward, walk_forward_all

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

# CORN (strategies/pattern_matching/corn.py) also replays its full history
# on every call, scanning for correlated past windows and solving a cvxpy
# BCRP over whichever days match -- also O(n^2)-ish, but much cheaper per
# step than ONS's QP. Measured directly, not assumed: 2.8s at n=179,
# 6.2s at n=249 (29 assets, w=5) -- consistent with roughly quadratic
# scaling, which would put the full ~6700-period window at a rough,
# extrapolated estimate of over an hour per candidate, not measured
# directly at that scale, and CORN, CORN-U and CORN-K have 12, 3 and 3
# candidates each (grids.py). Off by default for that reason; flip to
# True to include them anyway.
INCLUDE_CORN = False

OUTPUT_DIR = "outputs/djia"
PLOT_PATH = f"{OUTPUT_DIR}/djia_wealth_curves_all_strategies.png"
GROUPED_PLOT_PATH = f"{OUTPUT_DIR}/djia_wealth_curves_steady_vs_aggressive.png"
GRID_PLOT_PATH = f"{OUTPUT_DIR}/djia_wealth_curves_per_strategy_grid.png"

# distinct dash patterns so nearly-overlapping wealth curves stay
# distinguishable even where color alone would not separate them
LINESTYLES = ["-", "--", ":", "-.", (0, (3, 1, 1, 1)), (0, (5, 1))]

# BAH/CRP/UP/EG never make a large, concentrated single-period bet, so they
# stay close to each other and close to 1; the rest make aggressive,
# short-horizon bets that paid off badly in this crash -- the two groups
# need different axes to both be readable (see paper, Current Results).
STEADY_GROUP = ["BAH", "CRP", "UP", "EG", "SPY"]
AGGRESSIVE_GROUP = ["ONS", "Anticor", "PAMR2", "CWMR", "OLMAR2", "CORN", "CORN-U", "CORN-K", "FixedShare", "SPY"]

# plain PAMR is run and reported, but PAMR2 is the one plotted: showing
# whichever variant happened to win here would itself be a choice made
# with hindsight
PLOT_EXCLUDE = {"BestStock", "PAMR", "ExpertsEW"}

# SPY is the S&P 500 total return, a buy-and-hold reference every strategy
# is judged against, so it is drawn the same way everywhere
SPY_STYLE = {"color": "black", "linestyle": ":", "linewidth": 1.5}


def _style(name: str, linestyle) -> dict:
    return SPY_STYLE if name == "SPY" else {"linestyle": linestyle}


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

    fixed = {
        "BAH": lambda n, p: BAH(n),
        "CRP": lambda n, p: CRP(n),
        "UP": lambda n, p: UP(n),
    }
    if INCLUDE_ONS:
        fixed["ONS"] = lambda n, p: ONS(n)
    grids = {k: v for k, v in GRIDS.items() if INCLUDE_CORN or k not in SLOW}

    total_periods = sum(arr.shape[0] for arr in segment_arrays)
    print(f"{total_periods} periods across {len(segment_arrays)} DJIA reconstitution segments:")
    for seg_start, seg_end, tickers in segments_meta:
        print(f"  {seg_start} .. {seg_end}: {len(tickers)} assets")

    # BestStock is a hindsight-only upper bound, not an expert, so it is
    # not cached or combined
    wealths = {
        "BestStock": run_backtest_segmented(lambda n, pr: BestStock(n, pr), segment_arrays),
        "SPY": benchmark_wealth("SPY", START, END, dates),
    }
    b_experts = {}
    for name, factory in fixed.items():
        b_experts[name] = strategy_portfolios("djia", name, None, factory, segment_arrays)
        wealths[name] = segmented_wealth(b_experts[name], segment_arrays)
    tuned, picks, default_b = walk_forward_all("djia", grids, segment_arrays, dates)
    wealths.update(tuned)
    b_experts.update(default_b)

    experts = [b_experts[name] for name in EXPERTS + (["CORN"] if INCLUDE_CORN else [])]
    candidates = [fixed_share_wealth(experts, segment_arrays, eta, alpha) for eta, alpha in FIXED_SHARE_GRID]
    wealths["FixedShare"], fs_picks = walk_forward(
        candidates, FIXED_SHARE_GRID.index(FIXED_SHARE_DEFAULT), dates
    )
    picks["FixedShare"] = {year: FIXED_SHARE_GRID[i] for year, i in fs_picks.items()}
    wealths["ExpertsEW"] = fixed_share_wealth(experts, segment_arrays, 0.0, 0.0)

    for name, wealth in wealths.items():
        _report(name, wealth, apy(wealths["SPY"]))
    for name, by_year in picks.items():
        print(f"{name:10s} chosen by year: {by_year}")

    _plot_combined(wealths, plot_dates)
    _plot_grouped(wealths, plot_dates)
    _plot_grid(wealths, plot_dates)


def _report(name: str, wealth, spy_apy: float) -> None:
    t_stat, p_value = t_test(wealth)
    print(
        f"{name:10s} final wealth = {wealth[-1]:.4f}  APY = {apy(wealth):+.2%}  "
        f"vol = {volatility(wealth):.2%}  Sharpe = {sharpe_ratio(wealth):.2f}  "
        f"MDD = {max_drawdown(wealth):.2%}  Calmar = {calmar_ratio(wealth):.2f}  "
        f"t = {t_stat:.2f} (p = {p_value:.3f})  vs SPY = {apy(wealth) - spy_apy:+.2%}"
    )


def _log_axis(ax) -> None:
    # S is multiplicative, so equal vertical distances mean equal growth
    # factors and a straight line is constant compounding; S = 1 (break-even)
    # marks where a strategy is ahead of or behind its starting capital
    ax.set_yscale("log")
    ax.axhline(1.0, color="gray", linewidth=0.5, linestyle=":")
    # plain numbers (0.1, not 1e-01) at 1, 2 and 5 of each decade
    formatter = FuncFormatter(lambda v, _: f"{v:g}" if f"{v:.1e}"[0] in "125" else "")
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.set_minor_formatter(formatter)


def _plot_combined(wealths: dict, plot_dates: pd.DatetimeIndex) -> None:
    fig, ax = plt.subplots()
    for (name, wealth), linestyle in zip(wealths.items(), itertools.cycle(LINESTYLES)):
        # BestStock is a hindsight-only upper bound (see strategies/best_stock.py),
        # not a real strategy, so it's left off the graph -- printed above for
        # reference, but not plotted alongside strategies that make causal decisions
        if name not in PLOT_EXCLUDE:
            ax.plot(plot_dates, wealth, label=name, **_style(name, linestyle))

    ax.set_xlabel("date")
    ax.set_ylabel("cumulative wealth (log scale)")
    _log_axis(ax)
    ax.set_title(f"OLPS strategies on DJIA constituents, {START} to {END}")
    ax.legend()
    fig.autofmt_xdate()
    fig.savefig(PLOT_PATH)
    print(f"saved plot to {PLOT_PATH}")


def _plot_grouped(wealths: dict, plot_dates: pd.DatetimeIndex) -> None:
    fig, (ax_steady, ax_aggressive) = plt.subplots(2, 1, figsize=(6.4, 7.2), sharex=True)

    steady = [name for name in STEADY_GROUP if name in wealths]
    for name, linestyle in zip(steady, itertools.cycle(LINESTYLES)):
        ax_steady.plot(plot_dates, wealths[name], label=name, **_style(name, linestyle))
    ax_steady.set_ylabel("cumulative wealth (log scale)")
    _log_axis(ax_steady)
    ax_steady.set_title("Steady: no concentrated single-period bets")
    ax_steady.legend()

    aggressive = [name for name in AGGRESSIVE_GROUP if name in wealths]
    for name, linestyle in zip(aggressive, itertools.cycle(LINESTYLES)):
        ax_aggressive.plot(plot_dates, wealths[name], label=name, **_style(name, linestyle))
    ax_aggressive.set_ylabel("cumulative wealth (log scale)")
    _log_axis(ax_aggressive)
    ax_aggressive.set_title("Aggressive: large single-period rebalancing bets")
    ax_aggressive.set_xlabel("date")
    ax_aggressive.legend()

    fig.suptitle(f"OLPS strategies on DJIA constituents, {START} to {END}")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(GROUPED_PLOT_PATH)
    print(f"saved plot to {GROUPED_PLOT_PATH}")


def _plot_grid(wealths: dict, plot_dates: pd.DatetimeIndex) -> None:
    names = [name for name in wealths if name not in PLOT_EXCLUDE | {"SPY"}]
    n_cols = math.ceil(math.sqrt(len(names)))
    n_rows = math.ceil(len(names) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 3.2 * n_rows), sharex=True, sharey=True)
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]
    for ax, name in zip(axes_flat, names):
        ax.plot(plot_dates, wealths["SPY"], color="gray", linestyle=":", linewidth=1)
        ax.plot(plot_dates, wealths[name])
        _log_axis(ax)
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
