# olps-benchmarks

Online Portfolio Selection (OLPS) algorithms, implemented from
*Online Portfolio Selection: Principles and Algorithms* (Li & Hoi) and
tested on two universes: the Dow Jones constituents since 2000 and a
cross-asset ETF basket since 2007. The work is written up as a TFG in
[paper/main.tex](paper/main.tex); the book is in [book/](book/).

What sets the benchmark apart from a plain re-implementation:

- **No lookahead.** Each strategy's parameter is chosen every January from
  past data only (walk-forward), never from the period it is used on.
- **Point-in-time universes.** The DJIA backtest follows the real index
  reconstitutions instead of assuming today's constituents always existed.
- **A market reference.** Every result is compared with SPY (S&P 500 total
  return) over the same days.
- **Run one strategy at a time.** Each finished strategy is stored as CSV,
  so a change never means repeating a multi-hour run.

## Results at a glance

Walk-forward results. APY is the annualized return and Sharpe is
(APY - 4%) / volatility. DJIA is 2000-01-01 to 2026-09-01, ETF is
2007-03-01 to 2026-09-01.

| Strategy | DJIA APY | DJIA Sharpe | ETF APY | ETF Sharpe |
|---|---:|---:|---:|---:|
| BAH | +9.1% | 0.28 | +7.0% | 0.34 |
| CRP | +9.9% | 0.30 | +6.6% | 0.34 |
| UP | +9.9% | 0.30 | +6.7% | 0.35 |
| EG | +9.9% | 0.30 | +6.6% | 0.33 |
| Anticor | +17.3% | 0.36 | +7.0% | 0.17 |
| PAMR | +8.5% | 0.20 | +13.2% | 0.51 |
| PAMR-2 | +11.1% | 0.26 | +13.8% | 0.64 |
| CWMR | -13.3% | -0.40 | +13.3% | 0.51 |
| OLMAR-2 | -1.4% | -0.11 | +17.7% | 0.73 |
| Fixed Share | +8.1% | 0.18 | +14.8% | 0.70 |
| PolicyExperts | +8.5% | 0.16 | +15.9% | 0.75 |
| SPY (reference) | +8.4% | 0.23 | +11.1% | 0.36 |

The ranking depends on the universe: mean-reversion strategies that lose on
the DJIA (CWMR, OLMAR-2) are among the best on the ETFs. PolicyExperts
depends on its random seed (about one point of APY on the DJIA), so one run
says little. The CORN family and ONS have not been run yet. Full tables,
drawdowns and discussion are in the paper; the numbers above come from
`results/*/metrics.csv`.

## Strategies

| Family | Strategy | Source | File |
|---|---|---|---|
| Benchmarks | Buy-and-Hold (BAH) | Ch. 3.1 | [bah.py](strategies/benchmarks/bah.py) |
| | Best Stock (hindsight upper bound, not a real strategy) | Ch. 3.2 | [best_stock.py](strategies/benchmarks/best_stock.py) |
| | Constant Rebalanced Portfolio (CRP) | Ch. 3.3 | [crp.py](strategies/benchmarks/crp.py) |
| Follow the winner | Universal Portfolio (UP) | Ch. 4 | [up.py](strategies/follow_the_winner/up.py) |
| | Exponential Gradient (EG) | Ch. 4 | [eg.py](strategies/follow_the_winner/eg.py) |
| | Online Newton Step (ONS) | Ch. 4.4 | [ons.py](strategies/follow_the_winner/ons.py) |
| Follow the loser | Anticor | Ch. 5.2 | [anticor.py](strategies/follow_the_loser/anticor.py) |
| | PAMR, PAMR-1, PAMR-2 | Ch. 9 | [pamr.py](strategies/follow_the_loser/pamr.py), [pamr1.py](strategies/follow_the_loser/pamr1.py), [pamr2.py](strategies/follow_the_loser/pamr2.py) |
| | CWMR (Var) | Ch. 10 | [cwmr.py](strategies/follow_the_loser/cwmr.py) |
| | OLMAR, OLMAR-2 | Ch. 11 | [olmar.py](strategies/follow_the_loser/olmar.py), [olmar2.py](strategies/follow_the_loser/olmar2.py) |
| Pattern matching | CORN | Ch. 8, Alg. 8.1 | [corn.py](strategies/pattern_matching/corn.py) |
| | CORN-Sig (signature distance instead of correlation) | Ni et al. 2021 | [corn_sig.py](strategies/pattern_matching/corn_sig.py) |
| | CORN-U, CORN-K | Ch. 8, Alg. 8.2 / 8.3 | [corn_u.py](strategies/pattern_matching/corn_u.py), [corn_k.py](strategies/pattern_matching/corn_k.py) |
| Meta-algorithms | Hedge with Fixed Share over the experts | Herbster & Warmuth 1998 | [fixed_share.py](strategies/meta_algorithms/fixed_share.py) |
| | PolicyExperts (learned softmax policy over the experts) | this project | [policy_gradient.py](strategies/meta_algorithms/policy_gradient.py) |

Every strategy implements the `Strategy` interface in
[strategies/base.py](strategies/base.py): `update(history)` is a pure
function of the price relatives up to `t-1` and returns the portfolio `b_t`.
CORN-U and CORN-K are the one deliberate exception: they keep their experts'
wealth between calls, with a guard that replays from the start if called out
of order. PAMR-1 and OLMAR (version 1) are implemented but not part of the
benchmark runs.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Running the benchmark

```bash
python run.py                          # DJIA, every strategy that is on by default
python run_etf.py                      # the same on the ETF universe
python run.py --only Anticor CORN-Sig  # just these, including the slow ones
python run.py --list                   # which strategies are stored, outdated or missing
python run.py --force                  # recompute even if stored results are current
python run.py --no-run                 # compute nothing, redraw plots from stored results
python run.py --plot BAH Anticor       # one custom chart of just these (and SPY)
```

`run_etf.py` takes the same options. `--only` accepts any strategy name,
including ONS and the CORN family, which are off by default because of their
cost (ONS replays its history and solves a QP at every step: minutes at ~650
periods, tens of hours at ~6700). The combiners (`FixedShare`, `ExpertsEW`,
`PolicyExperts`) need their experts' portfolios and compute any that are
missing at the default parameter.

Several `--only` runs can go at the same time, one per terminal: saving is
locked and files are replaced atomically, so they do not overwrite each
other's results. Start them a few seconds apart, since each process needs a
few hundred MB while it imports `cvxpy`.

Plots go to `outputs/<universe>/` (gitignored): `..._all_strategies.png`,
`..._steady_vs_aggressive.png` and `..._per_strategy_grid.png`. `--plot`
writes `..._all_strategies_custom.png` so the standard figures are not
overwritten.

## How results are stored

Two layers, for two different costs:

| Folder | Holds | Used for |
|---|---|---|
| `cache/` (gitignored) | the portfolios `b_1..b_n` of one strategy with one parameter value, the slow part | skipping the minutes-to-hours simulation |
| `results/<universe>/` | the finished wealth curves and tables | reading results and plotting without recomputing |

The files in `results/<universe>/`:

- `wealth.csv`: one wealth curve per strategy (and per PolicyExperts seed)
- `metrics.csv`: final wealth, APY, volatility, Sharpe, drawdown, Calmar, t-test and APY versus SPY
- `picks.csv`: the parameter chosen each year by the walk-forward selection
- `keys.csv`: a hash of the code, grid and date window behind each column

A run recomputes a strategy only if its hash changed or it is missing, and
saves it as soon as it finishes. When nothing needs computing it reads the
CSVs and does not download prices. The price data itself is not hashed, so
use `--force` if you suspect the data provider revised adjusted prices.

## Project layout

```
data.py            prices from yfinance, converted to price relatives
engine.py          backtest of one strategy; chains segments of a changing universe
metrics.py         APY, volatility, Sharpe, drawdown, Calmar, t-test
walk_forward.py    yearly parameter selection from past data; portfolio cache
grids.py           candidate parameter values and defaults per strategy
experiments.py     computes one strategy at a time and stores it in results/
run.py             DJIA benchmark and plots
run_etf.py         ETF benchmark and plots
strategies/        one file per algorithm, grouped as in the table above
universes/         djia_universe.py (point-in-time Dow roster), etf_universe.py
results/           stored CSV results (tracked)
paper/main.tex     the written report
```

The DJIA universe (`universes/djia_universe.py`) follows every
reconstitution since 2000 and documents the data gaps each one causes. The
ETF universe (`universes/etf_universe.py`) holds one large ETF per asset
class (US equities, long Treasuries, gold, commodities, the US dollar),
chosen after checking correlations, so strategies trade different risk
factors and not variations of one US-equity beta.

## Using a strategy directly

```python
from data import get_price_relatives
from strategies.benchmarks.crp import CRP

price_relatives = get_price_relatives(
    tickers=["AAPL", "MSFT"], start="2023-01-01", end="2024-01-01"
)

strategy = CRP(n_assets=price_relatives.shape[1])
b_t = strategy.get_portfolio(price_relatives.values[:10])
```

## Status

Implemented: the benchmarks, Follow the Winner, Follow the Loser, Pattern
Matching (CORN, CORN-Sig, CORN-U, CORN-K) and two combiners over the
experts. Walk-forward results are stored and written up for the default
strategy set on both universes.

Not done yet: the walk-forward benchmark of ONS and the CORN family, the
book's other meta-algorithms, and a test suite.
