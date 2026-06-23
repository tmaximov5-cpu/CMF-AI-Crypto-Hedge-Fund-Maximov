"""Data loading and train/test splitting.

Data is read ONLY from local ``data/*.csv`` (no network calls in the notebook).
CSVs are produced by ``scripts/download_data.py``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_coin(coin: str, data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """Load one coin's daily OHLCV CSV, indexed by timestamp (ascending).

    Parameters
    ----------
    coin : str
        Coin symbol, e.g. ``"BTC"`` (case-insensitive). Maps to
        ``data/<coin>_usdt.csv``.
    """
    path = Path(data_dir) / f"{coin.lower()}_usdt.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df.set_index("timestamp").sort_index()


def load_universe(coins: list[str], data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """Load close prices for several coins into one aligned DataFrame.

    Returns a frame whose columns are coin symbols and rows are dates, aligned
    on the common date index (outer join, then forward gaps left as NaN for the
    caller to handle explicitly — no silent fills that could leak information).
    """
    closes = {c.upper(): load_coin(c, data_dir)["close"] for c in coins}
    return pd.DataFrame(closes).sort_index()


def available_coins(data_dir: Path | str = DATA_DIR) -> list[str]:
    """All coin symbols with a CSV in ``data_dir`` (upper-case, sorted)."""
    paths = Path(data_dir).glob(f"*_{ 'usdt' }.csv")
    return sorted(p.name.split("_")[0].upper() for p in paths)


def load_panel(
    coins: list[str], field: str = "close", data_dir: Path | str = DATA_DIR
) -> pd.DataFrame:
    """Aligned panel of one OHLCV field across many coins (cols=coins, rows=dates).

    Outer-joined on the union of dates; gaps stay NaN for the caller to handle
    explicitly (the universe filter uses the missing-data ratio).
    """
    series = {c.upper(): load_coin(c, data_dir)[field] for c in coins}
    return pd.DataFrame(series).sort_index()


def train_test_split(df: pd.DataFrame, train_frac: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split — NO shuffling (would leak future into the past).

    The first ``train_frac`` of rows is train; the remainder is test. Scalers and
    models must be fit on the train slice only.
    """
    n_train = int(len(df) * train_frac)
    return df.iloc[:n_train].copy(), df.iloc[n_train:].copy()
