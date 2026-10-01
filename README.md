# olps-benchmarks

Online Portfolio Selection (OLPS) algorithms and benchmarks, implemented
following *Online Portfolio Selection: Principles and Algorithms* (Li & Hoi).
Written up as a TFG in [paper/main.tex](paper/main.tex); the book itself is
in [book/](book/).

## Structure

- [data.py](data.py) — downloads adjusted close prices (via `yfinance`) and
  converts them into price relatives `x[t,i] = price[t,i] / price[t-1,i]`.
- [engine.py](engine.py) — runs a strategy over a price-relative history
  (`run_backtest`) and tracks cumulative wealth `S_t`; `run_backtest_segmented`
  chains backtests across a universe that changes composition over time
  (e.g. index reconstitution), resetting strategy state at each boundary.
- [metrics.py](metrics.py) — APY, volatility, Sharpe, max drawdown, Calmar,
  and a t-test against the null that the strategy isn't profitable.
- [strategies/](strategies/) — one file per algorithm, each implementing
  `strategies/base.py`'s `Strategy` interface (Algorithm A.1): stateless,
  `update(history)` is a pure function of the price-relative history up to
  `t-1` and returns the portfolio `b_t`. Subfolders follow the book's own
  taxonomy (Part II: Ch. 3 Benchmarks, Ch. 4 Follow the Winner, Ch. 5
  Follow the Loser); `base.py` and `simplex.py` stay at the top level
  since every category's strategies depend on one or both of them.
  - `benchmarks/` — not adaptive, just a baseline to beat
    - `bah.py` — Buy-and-Hold (Ch. 3.1)
    - `best_stock.py` — hindsight-only upper bound, not a real strategy (Sec. 3.2)
    - `crp.py` — Constant Rebalanced Portfolio (Ch. 3.3)
  - `follow_the_winner/` — shift weight toward recent outperformers
    - `up.py` — Universal Portfolio (Cover, 1991)
    - `eg.py` — Exponential Gradient (Helmbold et al., 1998)
    - `ons.py` — Online Newton Step via FTRL (Ch. 4.4; uses `cvxpy`)
  - `follow_the_loser/` — bet on mean reversion instead
    - `anticor.py` — Anticorrelation (Borodin, El-Yaniv & Gogan, 2004; Sec. 5.2)
    - `pamr.py`, `pamr1.py`, `pamr2.py` — Passive Aggressive Mean Reversion
      and its capped/quadratic-slack variants (Ch. 9)
    - `cwmr.py` — Confidence Weighted Mean Reversion, CWMR-Var (Ch. 10)
    - `olmar.py`, `olmar2.py` — Online Moving Average Reversion, simple and
      exponential moving-average variants (Ch. 11)
  - `simplex.py` — shared closed-form simplex projection used by
    PAMR/CWMR/OLMAR
- [universes/](universes/) — point-in-time asset universes, so a backtest
  tracks real membership changes instead of assuming today's constituents
  always existed.
  - `djia_universe.py` — DJIA constituents from 2000-01-01 onward, with
    every reconstitution since then and the data-availability gaps that
    cause (documented per-ticker, with the reasoning for each).
  - `etf_universe.py` — one large, continuously-traded ETF per major
    asset class (US equities, developed ex-US equities, EM equities,
    REITs, long Treasuries, gold, broad commodities), so strategies trade
    genuinely different risk factors instead of 28-30 variations on the
    same US-equity beta. The universe grows as each ETF's inception date
    is reached rather than being fixed to today's survivors.
- [run.py](run.py) — runs every strategy over the DJIA universe end to end
  and writes wealth-curve plots to `outputs/`.
- [run_etf.py](run_etf.py) — the same backtest/plotting pipeline over the
  ETF universe instead, for comparing results across two largely
  uncorrelated asset sets.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Example

```python
from data import get_price_relatives
from strategies.benchmarks.crp import CRP

price_relatives = get_price_relatives(
    tickers=["AAPL", "MSFT"], start="2023-01-01", end="2024-01-01"
)

strategy = CRP(n_assets=price_relatives.shape[1])
b_t = strategy.get_portfolio(price_relatives.values[:10])
```

## Running the benchmark

```bash
python run.py
```

Backtests every strategy over DJIA constituents from 2000-01-01 to
2026-09-01, printing final wealth/APY/Sharpe/MDD/Calmar/t-test per strategy
and saving wealth-curve plots to `outputs/` (gitignored, regenerated on
each run). `ONS` is off by default — it replays its full history and solves
a QP at every step, which is minutes at ~650 periods but tens of hours at
~6700; flip `INCLUDE_ONS` in `run.py` to include it anyway.

## Status

All of the book's core strategies are implemented and benchmarked on the
DJIA universe, with results and methodology written up in `paper/main.tex`.
The `etf-universe` branch adds a second, cross-asset-class universe (see
`universes/etf_universe.py` and `run_etf.py`) to check whether results hold
outside a basket of correlated US large caps.
