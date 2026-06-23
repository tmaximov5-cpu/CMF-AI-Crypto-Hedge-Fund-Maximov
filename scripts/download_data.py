"""Fetch ~18 months of daily OHLCV via ccxt for a wide USDT universe.

This is the ONLY place network access happens — the notebook reads the produced
``data/<coin>_usdt.csv`` files offline. Idempotent: a coin whose CSV already
exists is skipped.

Two stages:
  1. discover the top-N most liquid USDT spot pairs on Binance by 24h quote
     volume (excluding stablecoins and leveraged tokens);
  2. download the core 7 coins plus those, with per-symbol fallback to
     binanceus / kraken on geo-block or missing symbol.

Run with: ``uv run python scripts/download_data.py [--top N]`` (default N=100).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import ccxt
import pandas as pd

COINS = ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE"]  # core universe (always fetched)
QUOTE = "USDT"
TIMEFRAME = "1d"
LOOKBACK_DAYS = 550  # ~18 months
EXCHANGES = ["binance", "binanceus", "kraken"]

# Bases to skip: fiat/stablecoins (no signal) — leveraged tokens are filtered separately.
STABLES = {
    "USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDP", "USDD", "PYUSD", "USTC",
    "EUR", "EURI", "AEUR", "GBP", "UST", "SUSD", "GUSD", "XUSD", "BFUSD",
}

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def fetch_ohlcv(exchange: ccxt.Exchange, symbol: str, since_ms: int) -> list[list]:
    """Page through fetch_ohlcv from ``since_ms`` to now."""
    all_rows: list[list] = []
    cursor = since_ms
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, since=cursor, limit=1000)
        if not batch:
            break
        all_rows += batch
        cursor = batch[-1][0] + 1
        if len(batch) < 1000:
            break
        time.sleep((exchange.rateLimit or 200) / 1000)
    return all_rows


def _save(rows: list[list], coin: str) -> int:
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    df.to_csv(DATA_DIR / f"{coin.lower()}_{QUOTE.lower()}.csv", index=False)
    return len(df)


def rank_liquid_usdt(exchange: ccxt.Exchange, n: int, exclude: set[str]) -> list[str]:
    """Top-n USDT spot bases by 24h quote volume (no stables / leveraged tokens)."""
    tickers = exchange.fetch_tickers()
    ranked: list[tuple[str, float]] = []
    for symbol, ticker in tickers.items():
        if not symbol.endswith(f"/{QUOTE}"):
            continue
        market = exchange.markets.get(symbol, {})
        if not (market.get("spot") and market.get("active")):
            continue
        base = market["base"]
        if base in STABLES or base in exclude:
            continue
        if base.endswith(("UP", "DOWN", "BULL", "BEAR")):  # leveraged tokens
            continue
        ranked.append((base, ticker.get("quoteVolume") or 0.0))
    ranked.sort(key=lambda pair: pair[1], reverse=True)

    out: list[str] = []
    for base, _ in ranked:
        if base not in out:
            out.append(base)
        if len(out) >= n:
            break
    return out


def download_coin(coin: str, since_ms: int) -> tuple[bool, str]:
    """Multi-exchange fallback for a single coin. Returns (ok, source/reason)."""
    symbol = f"{coin}/{QUOTE}"
    for name in EXCHANGES:
        try:
            exchange = getattr(ccxt, name)({"enableRateLimit": True})
            exchange.load_markets()
            if symbol not in exchange.markets:
                continue
            rows = fetch_ohlcv(exchange, symbol, since_ms)
            if rows:
                return True, f"{name} ({_save(rows, coin)} rows)"
        except Exception as exc:  # geo-block (451), missing symbol, network, etc.
            print(f"  [{coin}] {name} failed: {type(exc).__name__}: {str(exc)[:80]}")
    return False, "all exchanges failed"


def main(top: int = 100) -> int:
    DATA_DIR.mkdir(exist_ok=True)
    since_ms = ccxt.Exchange.milliseconds() - LOOKBACK_DAYS * 24 * 60 * 60 * 1000

    # Stage 1: discover liquid pairs on a single reused Binance instance.
    primary = ccxt.binance({"enableRateLimit": True})
    try:
        primary.load_markets()
        liquid = rank_liquid_usdt(primary, top, exclude=set(COINS))
        print(f"discovered {len(liquid)} liquid USDT pairs")
    except Exception as exc:
        print(f"discovery failed ({type(exc).__name__}); downloading core only: {str(exc)[:80]}")
        primary, liquid = None, []

    universe = COINS + [c for c in liquid if c not in COINS]

    # Stage 2: download (fast path via the reused primary, else multi-exchange fallback).
    ok = skipped = missing = 0
    for coin in universe:
        out = DATA_DIR / f"{coin.lower()}_{QUOTE.lower()}.csv"
        if out.exists():
            skipped += 1
            continue
        symbol = f"{coin}/{QUOTE}"
        done = False
        if primary is not None and symbol in primary.markets:
            try:
                rows = fetch_ohlcv(primary, symbol, since_ms)
                if rows:
                    n_rows = _save(rows, coin)
                    print(f"[{coin}] binance ({n_rows} rows)")
                    ok += 1
                    done = True
            except Exception as exc:
                print(f"  [{coin}] binance failed: {type(exc).__name__}: {str(exc)[:60]}")
        if not done:
            success, info = download_coin(coin, since_ms)
            print(f"[{coin}] {info}")
            ok += int(success)
            missing += int(not success)

    print(f"\n=== Summary === universe={len(universe)} | downloaded={ok} skipped={skipped} missing={missing}")
    return 1 if (ok == 0 and skipped == 0) else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=100, help="number of liquid USDT pairs to fetch")
    args = parser.parse_args()
    sys.exit(main(args.top))
