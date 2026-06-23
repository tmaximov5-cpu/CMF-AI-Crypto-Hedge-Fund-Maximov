# AI Crypto Hedge Fund

An assignment building a crypto trading research pipeline in five
levels, from a single-coin baseline to a 100+ pair book with risk controls.
The deliverable is [`notebook.ipynb`](notebook.ipynb); all reusable logic lives
in [`src/fund/`](src/fund) and is imported by the notebook.

## What it does

| Level | Section | Strategy |
|-------|---------|----------|
| 1 | Baseline | BTC 7/21 moving-average crossover vs buy-and-hold |
| 2 | Single-Coin AI | Econometric (ARIMA+GARCH), ML (RandomForest, LogisticRegression), deterministic AI agent; permutation significance test |
| 3 | Static Portfolio | Mean-variance optimization (max-Sharpe, min-variance), equal-weight, agent-based weights; efficient frontier |
| 4 | Dynamic Rebalancing | Trailing-window max-Sharpe, monthly + drift triggers, volatility-target overlay |
| 5 | 100+ Pairs | Universe filtering, cross-sectional ranking, capped allocation, fail-safe monitor |
| 7 | Final Comparison | All strategies across all metrics in one table |

## Requirements

- [uv](https://docs.astral.sh/uv/) (Python package/environment manager)
- Python ≥ 3.11 (uv installs a matching interpreter automatically)

Install uv if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup

From the project root:

```bash
uv sync
```

This creates `.venv/` and installs the exact pinned versions from `uv.lock`
(numpy, pandas, scikit-learn, scipy, statsmodels, arch, ccxt, matplotlib,
jupyter) plus the local `fund` package in editable mode.

## Run

The notebook reads **only** local `data/*.csv` and makes **no** network calls,
so it runs fully offline. The CSVs are committed, so you can go straight to:

```bash
# Execute top-to-bottom in place (regenerates all outputs)
uv run jupyter nbconvert --to notebook --execute --inplace notebook.ipynb
```

Or explore interactively:

```bash
uv run jupyter lab notebook.ipynb
```

### Regenerating the data (optional)

Data download is the **only** networked step and lives in its own script. It is
idempotent — existing CSVs are skipped:

```bash
uv run python scripts/download_data.py --top 100
```

It fetches ~18 months of daily OHLCV for the core 7 coins plus the top-N most
liquid USDT spot pairs on Binance (falling back to binanceus / kraken per
symbol), saving each to `data/<coin>_usdt.csv`.

## Project layout

```
notebook.ipynb          the deliverable (Sections 1-7)
pyproject.toml          uv project + pinned dependencies
uv.lock                 locked versions for reproducibility
CLAUDE.md               project rules
scripts/download_data.py   the only networked step
data/*.csv              local OHLCV (read by the notebook)
src/fund/
  data.py        CSV loading, panels, train/test split
  features.py    feature engineering + next-day target
  metrics.py     ROI, CAGR, Sharpe, Sortino, Calmar, VaR/CVaR, turnover,
                 permutation test, concentration
  backtest.py    position/weight backtests (.shift(1), turnover costs)
  agents.py      econometric / ML / deterministic AI agents
  portfolio.py   mean-variance optimization + dynamic rebalancing
  universe.py    filtering, ranking, capped allocation, reduced backtest
  monitor.py     fail-safe risk monitor (circuit breakers)
```

## Reproducibility & methodology

- **Pinned environment** via `uv.lock`; runs offline from local CSVs.
- **Seeds set**: `SEED = 42`, `np.random.seed`, `random_state` on every model,
  and seeded `np.random.default_rng` for sampling/permutation.
- **No look-ahead**: features use data up to `t`, the ML target uses
  `.shift(-1)`, positions/weights apply next period via `.shift(1)`, and all
  models/scalers are fit on the train split only. Universe filtering uses
  train-period data only.
- **Crypto annualization** uses 365 (and √365), never 252.
- **Costs**: 10 bps deducted on turnover throughout.

### Honest caveats

The test window is a single ~5.5-month crypto drawdown, so most absolute
returns are negative — conclusions are about *relative* risk control, not
profitability. The Level 5 universe is built from coins liquid *today*, which
bakes in survivorship bias the data cannot remove. This is a methodology
demonstration, not investment advice.

## Run with Docker (optional)

```bash
docker build -t crypto-fund .
docker run --rm crypto-fund
```

The image runs the notebook end-to-end offline (data is baked in). See
[`Dockerfile`](Dockerfile).
