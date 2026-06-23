"""Backtesting: turn positions/weights into net-of-cost returns + benchmarks.

No-look-ahead: a position decided from information at bar t is applied to the
return of bar t+1, enforced here with ``.shift(1)``. Callers pass positions
indexed by the bar on which the decision was made; this module does the shift.
"""

from __future__ import annotations

import pandas as pd

# Default round-trip transaction cost (per unit turnover), e.g. 10 bps.
DEFAULT_COST = 0.001


def backtest_positions(
    positions: pd.Series,
    asset_returns: pd.Series,
    cost: float = DEFAULT_COST,
) -> pd.Series:
    """Net daily strategy returns for a single asset.

    Parameters
    ----------
    positions : pd.Series
        Target exposure (e.g. 0/1 or -1..1) decided using data up to each bar.
        Applied with ``.shift(1)`` so we trade on the next bar.
    asset_returns : pd.Series
        Simple daily returns of the asset.
    cost : float
        Cost charged on turnover ``|position_t - position_{t-1}|``.
    """
    held = positions.shift(1).fillna(0.0)
    gross = held * asset_returns
    turnover = held.diff().abs().fillna(held.abs())
    return gross - turnover * cost


def backtest_weights(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    cost: float = DEFAULT_COST,
) -> pd.Series:
    """Net daily portfolio returns from a time-varying weight matrix.

    ``weights`` (rows=dates, cols=assets) are decided at each bar and applied
    with ``.shift(1)``. Turnover cost is the summed absolute weight change.
    """
    held = weights.shift(1).fillna(0.0)
    gross = (held * asset_returns).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1).fillna(held.abs().sum(axis=1))
    return gross - turnover * cost


def benchmark_buy_and_hold_btc(btc_returns: pd.Series) -> pd.Series:
    """Benchmark 1: 100% BTC, bought once and held (no rebalancing cost)."""
    return btc_returns.copy()


def benchmark_equal_weight(asset_returns: pd.DataFrame) -> pd.Series:
    """Benchmark 2: daily-rebalanced equal-weight basket of all assets."""
    return asset_returns.mean(axis=1)
