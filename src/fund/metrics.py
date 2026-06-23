"""Performance metrics. Crypto annualization uses 365, NOT 252.

All return inputs are simple daily returns. Risk metrics (VaR/CVaR) are reported
as positive loss magnitudes. ``turnover``/``n_trades`` take a position series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import ANNUALIZATION


def roi(returns: pd.Series) -> float:
    """Total return over the whole period: (1+r).prod() - 1."""
    return (1 + returns).prod() - 1


def cagr(returns: pd.Series) -> float:
    """Compound annual growth rate from a series of simple daily returns."""
    total = (1 + returns).prod()
    years = len(returns) / ANNUALIZATION
    return total ** (1 / years) - 1 if years > 0 else np.nan


def annual_volatility(returns: pd.Series) -> float:
    """Annualized volatility = daily std * sqrt(365)."""
    return returns.std() * np.sqrt(ANNUALIZATION)


def sharpe(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Annualized Sharpe ratio (risk_free is an annual rate)."""
    excess = returns - risk_free / ANNUALIZATION
    denom = excess.std()
    return np.sqrt(ANNUALIZATION) * excess.mean() / denom if denom else np.nan


def sortino(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Annualized Sortino ratio: excess return over downside deviation.

    Downside deviation = sqrt(mean(min(r, 0)**2)) computed on excess returns.
    """
    excess = returns - risk_free / ANNUALIZATION
    downside = np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2))
    return np.sqrt(ANNUALIZATION) * excess.mean() / downside if downside else np.nan


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown of the equity curve built from simple returns (negative)."""
    equity = (1 + returns).cumprod()
    return (equity / equity.cummax() - 1).min()


def calmar(returns: pd.Series) -> float:
    """Calmar ratio = CAGR / |max drawdown|."""
    mdd = abs(max_drawdown(returns))
    return cagr(returns) / mdd if mdd else np.nan


def value_at_risk(returns: pd.Series, level: float = 0.95) -> float:
    """Historical VaR at ``level`` as a positive loss magnitude.

    e.g. level=0.95 → the loss the daily return falls below 5% of the time.
    """
    return -np.percentile(returns.dropna(), (1 - level) * 100)


def conditional_var(returns: pd.Series, level: float = 0.95) -> float:
    """Historical CVaR / expected shortfall at ``level`` (positive loss).

    Mean of the returns at or below the VaR cutoff.
    """
    clean = returns.dropna()
    cutoff = np.percentile(clean, (1 - level) * 100)
    tail = clean[clean <= cutoff]
    return -tail.mean() if len(tail) else np.nan


def win_rate(returns: pd.Series) -> float:
    """Share of active (nonzero) days with a positive return."""
    active = returns[returns != 0]
    return (active > 0).mean() if len(active) else np.nan


def turnover(positions: pd.Series) -> float:
    """Mean daily turnover = average absolute day-over-day position change."""
    return positions.diff().abs().mean()


def n_trades(positions: pd.Series) -> int:
    """Number of days the position changed (entries + exits + flips)."""
    return int((positions.diff().fillna(positions) != 0).sum())


def effective_holdings(weights: pd.Series) -> float:
    """Effective number of positions = 1 / Σ(normalized weight²) (inverse HHI).

    Equals N for an equal N-way split and 1 for a single concentrated bet; a
    concentration gauge for the wide-universe book.
    """
    active = weights[weights > 0]
    total = active.sum()
    if total == 0:
        return 0.0
    proportions = active / total
    return float(1.0 / (proportions**2).sum())


def permutation_test(
    positions: pd.Series,
    asset_returns: pd.Series,
    cost: float = 0.001,
    n: int = 1000,
    seed: int = 0,
) -> tuple[float, float, np.ndarray]:
    """Significance test for a position series' *timing* skill.

    Shuffling the realized returns would leave Sharpe unchanged (it is
    permutation-invariant on the return series), so instead we shuffle the
    POSITION order ``n`` times — keeping the asset's actual return path but
    destroying the signal's timing — and recompute the net-of-cost Sharpe each
    time. The p-value is the share of random-timing runs that match or beat the
    real strategy.

    Returns ``(actual_sharpe, p_value, null_distribution)``.
    """
    from .backtest import backtest_positions  # local import avoids any cycle

    actual = sharpe(backtest_positions(positions, asset_returns, cost))
    rng = np.random.default_rng(seed)
    pos_values = positions.to_numpy()
    null = np.empty(n)
    for i in range(n):
        shuffled = pd.Series(rng.permutation(pos_values), index=positions.index)
        null[i] = sharpe(backtest_positions(shuffled, asset_returns, cost))
    p_value = float((null >= actual).mean())
    return actual, p_value, null


def summary(returns: pd.Series, positions: pd.Series | None = None) -> dict[str, float]:
    """Bundle the headline metrics for reporting in the notebook.

    Pass ``positions`` to also report turnover, win rate, and trade count.
    Returns an ordered dict for a clean side-by-side table.
    """
    out: dict[str, float] = {
        "roi": roi(returns),
        "cagr": cagr(returns),
        "ann_vol": annual_volatility(returns),
        "sharpe": sharpe(returns),
        "sortino": sortino(returns),
        "calmar": calmar(returns),
        "max_drawdown": max_drawdown(returns),
        "var_95": value_at_risk(returns, 0.95),
        "cvar_95": conditional_var(returns, 0.95),
        "win_rate": win_rate(returns),
    }
    if positions is not None:
        out["turnover"] = turnover(positions)
        out["n_trades"] = n_trades(positions)
    return out
