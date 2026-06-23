"""Wide-universe construction: filtering, cross-sectional ranking, allocation.

Level 5 trades a 100+ coin book. This module turns the raw panel into a clean
tradeable universe, scores coins cross-sectionally, and allocates to the top
names under per-coin and total-exposure caps. No look-ahead: every input is a
trailing/as-of quantity computed from data up to the decision date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import ANNUALIZATION
from .monitor import RiskMonitor


def filter_universe(
    closes: pd.DataFrame,
    volumes: pd.DataFrame,
    min_history: int = 400,
    max_missing: float = 0.02,
    min_dollar_volume: float = 5e6,
) -> tuple[list[str], pd.DataFrame]:
    """Filter coins by history length, within-span missing-data ratio, and liquidity.

    - ``min_history``: minimum number of observed daily closes.
    - ``max_missing``: max fraction of NaN days *within* the coin's listed span
      (leading pre-listing NaNs don't count against it).
    - ``min_dollar_volume``: minimum average daily dollar volume (close × volume).

    Returns ``(kept_coins, diagnostics)`` where diagnostics has one row per coin.
    """
    records: dict[str, dict] = {}
    for coin in closes.columns:
        series = closes[coin]
        first = series.first_valid_index()
        if first is None:
            records[coin] = {"n_obs": 0, "missing": 1.0, "dollar_volume": 0.0, "keep": False}
            continue
        span = series.loc[first:]
        n_obs = int(span.notna().sum())
        missing = 1.0 - n_obs / len(span)
        dollar_volume = float((closes[coin] * volumes[coin]).mean())
        keep = (n_obs >= min_history) and (missing <= max_missing) and (dollar_volume >= min_dollar_volume)
        records[coin] = {
            "n_obs": n_obs,
            "missing": missing,
            "dollar_volume": dollar_volume,
            "keep": keep,
        }
    diagnostics = pd.DataFrame(records).T
    kept = [c for c in closes.columns if records[c]["keep"]]
    return kept, diagnostics


def _zscore(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score; all-zeros if no dispersion."""
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd and not np.isnan(sd) else s * 0.0


def ranking_score(
    ml_signal: pd.Series,
    momentum: pd.Series,
    volatility: pd.Series,
    weights: tuple[float, float, float] = (0.4, 0.3, -0.3),
) -> pd.Series:
    """Cross-sectional score = 0.4·ml + 0.3·momentum − 0.3·volatility.

    Each component is z-scored across coins first so the fixed weights combine
    comparable scales (raw P(up), returns and stdevs are not on the same scale).
    Higher is better; volatility enters with a negative weight (penalty).
    """
    df = pd.DataFrame({"ml": ml_signal, "mom": momentum, "vol": volatility}).dropna()
    score = (
        weights[0] * _zscore(df["ml"])
        + weights[1] * _zscore(df["mom"])
        + weights[2] * _zscore(df["vol"])
    )
    return score.reindex(ml_signal.index)


def allocate_top_ranked(
    scores: pd.Series,
    top_k: int = 15,
    max_weight: float = 0.10,
    total_exposure: float = 1.0,
) -> pd.Series:
    """Equal-weight the top-``top_k`` coins by score, under per-coin & total caps.

    Long-only. Each selected coin gets ``min(total_exposure/k, max_weight)``; the
    rest (and any unallocated budget when the per-coin cap binds) stay in cash.
    """
    weights = pd.Series(0.0, index=scores.index)
    top = scores.dropna().nlargest(top_k)
    if len(top) == 0:
        return weights
    per_coin = min(total_exposure / len(top), max_weight)
    weights[top.index] = per_coin
    return weights


def reduced_backtest(
    returns: pd.DataFrame,
    proba: pd.DataFrame,
    momentum: pd.DataFrame,
    volatility: pd.DataFrame,
    reference_price: pd.Series,
    monitor: RiskMonitor | None = None,
    rebalance_days: int = 7,
    top_k: int = 15,
    max_weight: float = 0.10,
    total_exposure: float = 1.0,
    cost: float = 0.001,
    cooldown_days: int = 5,
) -> dict:
    """Day-by-day walk-forward over the wide universe with a fail-safe overlay.

    A manual loop (rather than ``backtest.backtest_weights``) because the monitor
    reacts to *realized* strategy drawdown/daily-loss, so weights and returns are
    coupled. Each day: realize the return from weights decided yesterday (net of
    turnover cost), update the monitor, then decide weights for tomorrow — rank
    cross-sectionally on rebalance days, hold otherwise.

    When any guardrail fires the book goes flat for ``cooldown_days``; when the
    cooldown expires the drawdown reference (peak) is reset to current equity so
    the breaker re-arms and the strategy can resume — rather than latching to cash
    forever once a 20% drawdown is hit.

    All panels (``proba``/``momentum``/``volatility``) and ``returns`` must share
    the trade-date index. Returns a dict with the net-return series, the daily
    *held* weights, halt flags, and a {date: reasons} log.
    """
    monitor = monitor or RiskMonitor()
    dates, coins = returns.index, returns.columns

    strat = pd.Series(0.0, index=dates)
    held_weights = pd.DataFrame(0.0, index=dates, columns=coins)
    halts = pd.Series(False, index=dates)
    reasons_log: dict = {}

    held = pd.Series(0.0, index=coins)       # earns today's return (decided yesterday)
    prev_held = pd.Series(0.0, index=coins)
    target = pd.Series(0.0, index=coins)
    equity, peak = 1.0, 1.0
    cooldown = 0

    for i, date in enumerate(dates):
        # 1. realize today's net return from yesterday's weights.
        gross = float((held * returns.loc[date]).sum())
        turnover = float((held - prev_held).abs().sum())
        net = gross - turnover * cost
        strat[date] = net
        held_weights.loc[date] = held.to_numpy()
        equity *= 1 + net
        peak = max(peak, equity)
        drawdown = equity / peak - 1

        # 2. fail-safe check on information available up to today.
        history = strat.loc[:date]
        recent_vol = history.tail(5).std() * np.sqrt(ANNUALIZATION) if i >= 5 else None
        baseline_vol = history.tail(30).std() * np.sqrt(ANNUALIZATION) if i >= 30 else None
        halt, reasons = monitor.check(
            daily_return=net,
            drawdown=drawdown,
            reference_prices=reference_price.loc[:date].to_numpy(),
            long_fraction=float((proba.loc[date] > 0.55).mean()),
            short_fraction=float((proba.loc[date] < 0.45).mean()),
            recent_vol=recent_vol,
            baseline_vol=baseline_vol,
        )
        halts[date] = halt
        if reasons:
            reasons_log[date] = reasons
        if halt:
            cooldown = max(cooldown, cooldown_days)

        # 3. refresh the target on the rebalance schedule.
        if i % rebalance_days == 0:
            score = ranking_score(proba.loc[date], momentum.loc[date], volatility.loc[date])
            target = allocate_top_ranked(score, top_k, max_weight, total_exposure)

        # 4. decide tomorrow's weights: flat while cooling down, else the target.
        prev_held = held
        if cooldown > 0:
            held = pd.Series(0.0, index=coins)
            cooldown -= 1
            if cooldown == 0:
                peak = equity  # re-arm the drawdown breaker after cooling off
        else:
            held = target.copy()

    return {
        "returns": strat,
        "weights": held_weights,
        "halts": halts,
        "reasons": reasons_log,
    }
