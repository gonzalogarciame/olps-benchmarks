from __future__ import annotations

import itertools
import math
import os

import matplotlib.pyplot as plt
import pandas as pd

import universes.etf_universe as etf_universe
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

OUTPUT_DIR = "outputs"
PLOT_PATH = f"{OUTPUT_DIR}/etf_wealth_curves.png"
GROUPED_PLOT_PATH = f"{OUTPUT_DIR}/etf_wealth_curves_grouped.png"
GRID_PLOT_PATH = f"{OUTPUT_DIR}/etf_wealth_curves_grid.png"

LINESTYLES = ["-", "--", ":", "-.", (0, (3, 1, 1, 1)), (0, (5, 1))]

STEADY_GROUP = ["BAH", "CRP", "UP", "EG"]
AGGRESSIVE_GROUP = ["ONS", "Anticor", "PAMR2", "CWMR", "OLMAR", "OLMAR2"]


def _load_segment(seg_start: str, seg_end: str, tickers: list[str]) -> pd.DataFrame:
    df = get_price_relatives(tickers, seg_start, seg_end)
    return df[tickers]  # fixed column order, undoing yfinance's alphabetical sort


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    segments_meta = etf_universe.segments(START, END)
    segment_dfs = [_load_segment(s, e, tickers) for s, e, tickers in segments_meta]
    segment_arrays = [df.values for df in segment_dfs]

    dates = pd.DatetimeIndex([]).append([df.index for df in segment_dfs])
    plot_dates = dates.insert(0, dates[0] - pd.Timedelta(days=1))

    strategy_factories = {
        "BAH": lambda n, pr: BAH(n),
        "CRP": lambda n, pr: CRP(n),
        "BestStock": lambda n, pr: BestStock(n, pr),
        "UP": lambda n, pr: UP(n),
        "EG": lambda n, pr: EG(n),
        # w=5 / alpha=0.5 pinned explicitly rather than left at the class
        # defaults: those defaults are now tuned for the DJIA universe
        # (strategies/follow_the_loser/anticor.py, olmar2.py), and that
        # tuning was confirmed directly NOT to transfer here -- w=12
        # gives +5.81% APY on this universe versus w=5's +9.89%, and
        # alpha=0.3 gives +14.26% versus alpha=0.5's +18.72% (paper, Sec.
        # anticor-olmar2-tuning). Kept at their original, untuned values
        # here deliberately: with only one ETF-universe window to test
        # on, tuning them for this universe specifically would be the
        # same in-sample search the DJIA tuning was careful to avoid.
        "Anticor": lambda n, pr: Anticor(n, w=5),
        "PAMR2": lambda n, pr: PAMR2(n),
        "CWMR": lambda n, pr: CWMR(n),
        "OLMAR": lambda n, pr: OLMAR(n),
        "OLMAR2": lambda n, pr: OLMAR2(n, alpha=0.5),
    }
    if INCLUDE_ONS:
        strategy_factories["ONS"] = lambda n, pr: ONS(n)

    total_periods = sum(arr.shape[0] for arr in segment_arrays)
    print(f"{total_periods} periods across {len(segment_arrays)} ETF-universe segments:")
    for seg_start, seg_end, tickers in segments_meta:
        print(f"  {seg_start} .. {seg_end}: {tickers}")

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
        if name != "BestStock":
            ax.plot(plot_dates, wealth, label=name, linestyle=linestyle)

    ax.set_xlabel("date")
    ax.set_ylabel("cumulative wealth (log scale)")
    ax.set_yscale("log")
    ax.set_title(f"OLPS strategies on a cross-asset ETF universe, {START} to {END}")
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

    fig.suptitle(f"OLPS strategies on a cross-asset ETF universe, {START} to {END}")
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
    fig.suptitle(f"OLPS strategies on a cross-asset ETF universe, {START} to {END}")
    fig.tight_layout()
    fig.savefig(GRID_PLOT_PATH)
    print(f"saved plot to {GRID_PLOT_PATH}")


if __name__ == "__main__":
    main()
