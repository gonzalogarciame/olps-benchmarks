from __future__ import annotations

import itertools

import matplotlib.pyplot as plt
import pandas as pd

import djia_universe
from data import get_price_relatives
from engine import run_backtest_segmented
from metrics import apy, calmar_ratio, max_drawdown, sharpe_ratio, t_test, volatility
from strategies.anticor import Anticor
from strategies.bah import BAH
from strategies.best_stock import BestStock
from strategies.crp import CRP
from strategies.cwmr import CWMR
from strategies.eg import EG
from strategies.olmar import OLMAR
from strategies.ons import ONS
from strategies.pamr import PAMR
from strategies.up import UP

# 2007-06-01 .. 2009-12-31: the financial crisis, chosen so Follow-the-Loser
# strategies have an actual fluctuating market to exploit (unlike the 2023
# snapshot's steady bull run, where BAH already wins) and because the Dow's
# membership genuinely changed mid-window -- see djia_universe.py for the
# point-in-time constituent tracking this requires.
START = "2007-06-01"
END = "2009-12-31"
PLOT_PATH = "wealth_curves.png"
GROUPED_PLOT_PATH = "wealth_curves_grouped.png"
GRID_PLOT_PATH = "wealth_curves_grid.png"

# distinct dash patterns so nearly-overlapping wealth curves stay
# distinguishable even where color alone would not separate them
LINESTYLES = ["-", "--", ":", "-.", (0, (3, 1, 1, 1)), (0, (5, 1))]

# BAH/CRP/UP/EG never make a large, concentrated single-period bet, so they
# stay close to each other and close to 1; the rest make aggressive,
# short-horizon bets that paid off badly in this crash -- the two groups
# need different axes to both be readable (see paper, Current Results).
STEADY_GROUP = ["BAH", "CRP", "UP", "EG"]
AGGRESSIVE_GROUP = ["ONS", "Anticor", "PAMR", "CWMR", "OLMAR"]


def _load_segment(seg_start: str, seg_end: str, tickers: list[str]) -> pd.DataFrame:
    fetch_symbols = [djia_universe.fetch_symbol(t) for t in tickers]
    df = get_price_relatives(fetch_symbols, seg_start, seg_end)
    df = df[fetch_symbols]  # fixed column order, undoing yfinance's alphabetical sort
    df.columns = tickers  # display names (undoes the UTX->RTX fetch alias)
    return df


def main() -> None:
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
        "ONS": lambda n, pr: ONS(n),
        "Anticor": lambda n, pr: Anticor(n),
        "PAMR": lambda n, pr: PAMR(n),
        "CWMR": lambda n, pr: CWMR(n),
        "OLMAR": lambda n, pr: OLMAR(n),
    }

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

    for name, linestyle in zip(STEADY_GROUP, itertools.cycle(LINESTYLES)):
        ax_steady.plot(plot_dates, wealths[name], label=name, linestyle=linestyle)
    ax_steady.set_ylabel("cumulative wealth")
    ax_steady.set_title("Steady: no concentrated single-period bets")
    ax_steady.legend()

    for name, linestyle in zip(AGGRESSIVE_GROUP, itertools.cycle(LINESTYLES)):
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
    fig, axes = plt.subplots(3, 3, figsize=(11, 9), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, names):
        ax.plot(plot_dates, wealths[name])
        ax.axhline(1.0, color="gray", linewidth=0.5, linestyle=":")
        ax.set_yscale("log")
        ax.set_title(name)
        ax.tick_params(axis="x", labelrotation=45)

    fig.supxlabel("date")
    fig.supylabel("cumulative wealth (log scale)")
    fig.suptitle(f"OLPS strategies on DJIA constituents, {START} to {END}")
    fig.tight_layout()
    fig.savefig(GRID_PLOT_PATH)
    print(f"saved plot to {GRID_PLOT_PATH}")


if __name__ == "__main__":
    main()
