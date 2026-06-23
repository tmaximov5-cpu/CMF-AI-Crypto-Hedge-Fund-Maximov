"""Feature engineering for the ML / AI agents.

CRITICAL no-look-ahead rules (see CLAUDE.md):
- Features describing bar t may use data up to and including t.
- The prediction TARGET uses ``.shift(-1)`` (next-day direction).
- When these features feed a position, the position is applied with
  ``.shift(1)`` in backtest.py so you trade on the *next* bar.
"""

from __future__ import annotations

import pandas as pd

FEATURE_COLUMNS = [
    "return_1d",
    "return_3d",
    "return_7d",
    "ma_7",
    "ma_21",
    "ma_ratio",
    "volatility_14",
    "momentum_7",
    "rsi_14",
    "volume_change",
]


def make_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Build the feature matrix from a single coin's OHLCV frame.

    All features use past/current data only. NaNs from rolling windows are left
    for the caller to drop *before* the train/test split.
    """
    close = ohlcv["close"]
    volume = ohlcv["volume"]

    feat = pd.DataFrame(index=ohlcv.index)
    feat["return_1d"] = close.pct_change(1)
    feat["return_3d"] = close.pct_change(3)
    feat["return_7d"] = close.pct_change(7)
    feat["ma_7"] = close.rolling(7).mean()
    feat["ma_21"] = close.rolling(21).mean()
    feat["ma_ratio"] = feat["ma_7"] / feat["ma_21"]
    feat["volatility_14"] = close.pct_change().rolling(14).std()
    feat["momentum_7"] = close / close.shift(7) - 1
    feat["rsi_14"] = _rsi(close, 14)
    feat["volume_change"] = volume.pct_change()
    return feat[FEATURE_COLUMNS]


def make_target(close: pd.Series) -> pd.Series:
    """Classification target: 1 if NEXT day's simple return is positive, else 0.

    Uses ``.shift(-1)`` — the only place future data is referenced, and it is the
    label, never a feature.
    """
    next_ret = close.pct_change().shift(-1)
    return (next_ret > 0).astype(int)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder-style RSI computed from past data only."""
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)
