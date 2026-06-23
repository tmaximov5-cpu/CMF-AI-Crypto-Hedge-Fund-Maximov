"""Portfolio construction: weights, mean-variance optimization, rebalancing.

No-look-ahead: weights are estimated on the train slice and held over test; the
backtest applies them with ``.shift(1)``. Mean-variance moments are annualized
with 365 (crypto rule). All optimizers are long-only, fully invested (Σw=1), and
capped per coin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import ANNUALIZATION


# --- Mean-variance optimization (static allocation) ------------------------

def annualized_moments(returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Annualized expected returns and covariance (×365, crypto rule)."""
    return returns.mean() * ANNUALIZATION, returns.cov() * ANNUALIZATION


def portfolio_performance(
    weights: pd.Series, mu: pd.Series, cov: pd.DataFrame, risk_free: float = 0.0
) -> tuple[float, float, float]:
    """Annualized (return, volatility, Sharpe) of a weight vector."""
    w = np.asarray(weights)
    ret = float(w @ mu.to_numpy())
    vol = float(np.sqrt(w @ cov.to_numpy() @ w))
    sharpe = (ret - risk_free) / vol if vol else np.nan
    return ret, vol, sharpe


def max_sharpe_weights(
    returns: pd.DataFrame, max_weight: float = 0.4, risk_free: float = 0.0
) -> pd.Series:
    """Long-only max-Sharpe weights (Σw=1, 0≤w≤max_weight) via SLSQP."""
    mu, cov = annualized_moments(returns)
    mu_v, cov_v, n = mu.to_numpy(), cov.to_numpy(), len(mu)

    def neg_sharpe(w: np.ndarray) -> float:
        vol = np.sqrt(w @ cov_v @ w)
        return -(w @ mu_v - risk_free) / vol if vol else 0.0

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0.0, max_weight)] * n
    result = minimize(
        neg_sharpe, np.full(n, 1 / n), method="SLSQP", bounds=bounds, constraints=constraints
    )
    return pd.Series(result.x, index=mu.index)


def min_variance_weights(returns: pd.DataFrame, max_weight: float = 0.4) -> pd.Series:
    """Long-only minimum-variance weights (Σw=1, 0≤w≤max_weight) via SLSQP."""
    mu, cov = annualized_moments(returns)
    cov_v, n = cov.to_numpy(), len(mu)

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0.0, max_weight)] * n
    result = minimize(
        lambda w: w @ cov_v @ w,
        np.full(n, 1 / n),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return pd.Series(result.x, index=mu.index)


def efficient_frontier(
    returns: pd.DataFrame, n_points: int = 40, max_weight: float = 0.4
) -> tuple[np.ndarray, np.ndarray]:
    """Min-variance frontier under the per-coin cap.

    Returns ``(vols, rets)`` arrays. Target returns infeasible under the cap are
    skipped (the cap bounds the maximum achievable portfolio return).
    """
    mu, cov = annualized_moments(returns)
    mu_v, cov_v, n = mu.to_numpy(), cov.to_numpy(), len(mu)
    bounds = [(0.0, max_weight)] * n

    vols, rets = [], []
    for target in np.linspace(mu_v.min(), mu_v.max(), n_points):
        constraints = [
            {"type": "eq", "fun": lambda w: w.sum() - 1},
            {"type": "eq", "fun": lambda w, t=target: w @ mu_v - t},
        ]
        result = minimize(
            lambda w: w @ cov_v @ w,
            np.full(n, 1 / n),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        if result.success:
            vols.append(np.sqrt(result.fun))
            rets.append(target)
    return np.array(vols), np.array(rets)


def cap_weights(scores: pd.Series, max_weight: float = 0.4) -> pd.Series:
    """Turn nonneg scores into long-only weights summing to 1, each ≤ max_weight.

    Iterative water-filling: pin coins that exceed the cap at ``max_weight`` and
    redistribute the remainder proportionally among the rest until none exceed it.
    """
    raw = scores.clip(lower=0)
    if raw.sum() == 0:  # degenerate: fall back to equal weight
        return pd.Series(1 / len(scores), index=scores.index)

    weights = raw / raw.sum()
    capped: dict = {}
    while True:
        over = weights[weights > max_weight]
        if over.empty:
            break
        for coin in over.index:
            capped[coin] = max_weight
        remaining = 1 - max_weight * len(capped)
        free = weights.index.difference(capped.keys())
        free_raw = raw[free]
        weights = pd.Series(capped)
        weights = pd.concat([weights, free_raw / free_raw.sum() * remaining])
    return weights.reindex(scores.index)


# --- Dynamic rebalancing (Level 4) -----------------------------------------

def dynamic_rebalance(
    returns: pd.DataFrame,
    trade_index: pd.Index,
    lookback: int = 90,
    max_weight: float = 0.4,
    drift_threshold: float = 0.05,
    vol_target: float = 0.50,
    weight_fn=max_sharpe_weights,
) -> tuple[pd.DataFrame, pd.Series, list]:
    """Walk-forward dynamic allocation with three rebalance triggers + vol overlay.

    For each day ``t`` in ``trade_index``, decisions use only returns up to and
    including ``t`` (``returns`` should contain history before ``trade_index``);
    the resulting weights are meant to be applied next period — pass the output to
    ``backtest.backtest_weights``, which shifts by 1 and deducts turnover cost.

    Logic per day:
      - target = ``weight_fn`` on the trailing ``lookback`` returns (long-only,
        Σw=1, capped) — recomputed only when we rebalance;
      - rebalance if it is the first day, a new calendar month, OR the drifting
        mix has moved more than ``drift_threshold`` (per coin) from the mix set
        at the last rebalance; otherwise let the mix drift with returns;
      - volatility overlay: scale gross exposure by
        ``min(1, vol_target / est_vol)`` where ``est_vol`` is the ex-ante
        annualized portfolio vol from the trailing covariance and current mix
        (the remainder sits in cash).

    Returns ``(weights, exposures, rebalance_dates)``.
    """
    coins = returns.columns
    weights = pd.DataFrame(index=trade_index, columns=coins, dtype=float)
    exposures = pd.Series(index=trade_index, dtype=float)
    rebalance_dates: list = []

    base_w: pd.Series | None = None   # drifting asset mix (sums to 1, ex-cash)
    anchor: pd.Series | None = None   # mix as of the last rebalance
    last_period = None

    for t in trade_index:
        window = returns.loc[:t].iloc[-lookback:]
        cov = window.cov() * ANNUALIZATION

        if base_w is None:
            new_w = weight_fn(window, max_weight=max_weight)
            anchor = new_w
            last_period = t.to_period("M")
            rebalance_dates.append(t)
        else:
            drifted = base_w * (1 + returns.loc[t])
            drifted = drifted / drifted.sum()
            new_month = t.to_period("M") != last_period
            drift_breach = (drifted - anchor).abs().max() > drift_threshold
            if new_month or drift_breach:
                new_w = weight_fn(window, max_weight=max_weight)
                anchor = new_w
                last_period = t.to_period("M")
                rebalance_dates.append(t)
            else:
                new_w = drifted

        est_vol = float(np.sqrt(new_w.to_numpy() @ cov.to_numpy() @ new_w.to_numpy()))
        exposure = min(1.0, vol_target / est_vol) if est_vol > 0 else 1.0

        weights.loc[t] = (new_w * exposure).reindex(coins).to_numpy()
        exposures[t] = exposure
        base_w = new_w

    return weights, exposures, rebalance_dates
