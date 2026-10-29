from __future__ import annotations

import itertools
import math
import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

import experiments
import universes.etf_universe as etf_universe
from data import get_price_relatives

# 2007-03-01 is UUP's inception -- the first date all five ETFs exist, so
# the backtest doesn't have to grow the universe mid-window the way
# djia_universe-driven runs do. Starting any earlier would mean fewer
# than 5 assets for part of the run; starting later would be trimming
# real, available history for no reason. Shorter than run.py's 2000-2026
# only because the universe itself doesn't go back further, not because
# the window was picked to flatter any particular strategy.
START = "2007-03-01"
END = "2026-09-01"

# same reasoning as run.py: ONS replays its full history and solves a QP
# every period. Off by default for consistency with run.py even though
# this window is short enough (~2700 periods) that it would likely finish
# in minutes rather than hours.
INCLUDE_ONS = False

# same reasoning as run.py's INCLUDE_CORN
INCLUDE_CORN = False

OUTPUT_DIR = "outputs/etf"
PLOT_PATH = f"{OUTPUT_DIR}/etf_wealth_curves_all_strategies.png"
GROUPED_PLOT_PATH = f"{OUTPUT_DIR}/etf_wealth_curves_steady_vs_aggressive.png"
GRID_PLOT_PATH = f"{OUTPUT_DIR}/etf_wealth_curves_per_strategy_grid.png"

LINESTYLES = ["-", "--", ":", "-.", (0, (3, 1, 1, 1)), (0, (5, 1))]

STEADY_GROUP = ["BAH", "CRP", "UP", "EG", "SPY"]
AGGRESSIVE_GROUP = ["ONS", "Anticor", "PAMR2", "CWMR", "OLMAR2", "CORN", "CORN-U", "CORN-K", "FixedShare", "PolicyExperts", "SPY"]

# same variants plotted as in run.py; plain PAMR is run and reported only
PLOT_EXCLUDE = {"BestStock", "PAMR", "ExpertsEW"}

# SPY is the S&P 500 total return, a buy-and-hold reference every strategy
# is judged against, so it is drawn the same way everywhere
SPY_STYLE = {"color": "black", "linestyle": ":", "linewidth": 1.5}


def _style(name: str, linestyle) -> dict:
    return SPY_STYLE if name == "SPY" else {"linestyle": linestyle}


def _load_segment(seg_start: str, seg_end: str, tickers: list[str]) -> pd.DataFrame:
    df = get_price_relatives(tickers, seg_start, seg_end)
    return df[tickers]  # fixed column order, undoing yfinance's alphabetical sort


def _load_segments(segments_meta: list) -> list[pd.DataFrame]:
    return [_load_segment(s, e, tickers) for s, e, tickers in segments_meta]


def main() -> None:
    args = experiments.parse_args()
    universe = experiments.Universe(
        "etf", START, END, etf_universe.segments(START, END), _load_segments, INCLUDE_ONS, INCLUDE_CORN
    )
    if args.list:
        experiments.print_status(universe)
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    wealths, plot_dates = experiments.update(universe, args.only, args.force, args.no_run)

    if args.plot:
        chosen = {n: w for n, w in wealths.items() if n in args.plot or n == "SPY"}
        _plot_combined(chosen, plot_dates, PLOT_PATH.replace(".png", "_custom.png"), exclude=set())
        return
    _plot_combined(wealths, plot_dates, PLOT_PATH, PLOT_EXCLUDE)
    _plot_grouped(wealths, plot_dates)
    _plot_grid(wealths, plot_dates)


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


def _plot_combined(wealths: dict, plot_dates: pd.DatetimeIndex, path: str, exclude: set) -> None:
    fig, ax = plt.subplots()
    for (name, wealth), linestyle in zip(wealths.items(), itertools.cycle(LINESTYLES)):
        if name not in exclude:
            ax.plot(plot_dates, wealth, label=name, **_style(name, linestyle))

    ax.set_xlabel("date")
    ax.set_ylabel("cumulative wealth (log scale)")
    _log_axis(ax)
    ax.set_title(f"OLPS strategies on a cross-asset ETF universe, {START} to {END}")
    ax.legend()
    fig.autofmt_xdate()
    fig.savefig(path)
    print(f"saved plot to {path}")


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

    fig.suptitle(f"OLPS strategies on a cross-asset ETF universe, {START} to {END}")
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
    fig.suptitle(f"OLPS strategies on a cross-asset ETF universe, {START} to {END}")
    fig.tight_layout()
    fig.savefig(GRID_PLOT_PATH)
    print(f"saved plot to {GRID_PLOT_PATH}")


if __name__ == "__main__":
    main()
