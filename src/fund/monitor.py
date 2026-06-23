"""Risk monitoring / fail-safe overlay.

A live fund needs circuit breakers. ``RiskMonitor.check`` is called once per bar
with the current risk state and returns ``(halt, reasons)``; when ``halt`` is
True the strategy moves to cash for the next bar until the condition clears.

Guardrails:
- daily loss worse than ``max_daily_loss`` (default 5%)
- equity drawdown worse than ``max_drawdown`` (default 20%)
- stale data (a reference price unchanged for ``stale_days`` consecutive days)
- contradictory agent signals (breadth conflict: many coins bullish *and* many
  bearish at once — no consensus)
- volatility spike (recent realized vol > ``vol_spike_mult`` × its baseline)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MonitorConfig:
    max_daily_loss: float = 0.05
    max_drawdown: float = 0.20
    stale_days: int = 3
    conflict_ratio: float = 0.40
    vol_spike_mult: float = 3.0


class RiskMonitor:
    """Stateless per-bar fail-safe; pass only as-of/trailing inputs (no look-ahead)."""

    def __init__(self, config: MonitorConfig | None = None):
        self.config = config or MonitorConfig()

    def check(
        self,
        *,
        daily_return: float | None = None,
        drawdown: float | None = None,
        reference_prices: np.ndarray | None = None,
        long_fraction: float | None = None,
        short_fraction: float | None = None,
        recent_vol: float | None = None,
        baseline_vol: float | None = None,
    ) -> tuple[bool, list[str]]:
        cfg = self.config
        reasons: list[str] = []

        if daily_return is not None and daily_return < -cfg.max_daily_loss:
            reasons.append(f"daily loss {daily_return:.1%}")

        if drawdown is not None and drawdown < -cfg.max_drawdown:
            reasons.append(f"drawdown {drawdown:.1%}")

        if reference_prices is not None and len(reference_prices) >= cfg.stale_days:
            tail = np.asarray(reference_prices[-cfg.stale_days:], dtype=float)
            if np.all(np.diff(tail) == 0):
                reasons.append("stale data")

        if (
            long_fraction is not None
            and short_fraction is not None
            and long_fraction >= cfg.conflict_ratio
            and short_fraction >= cfg.conflict_ratio
        ):
            reasons.append("contradictory agent signals")

        if (
            recent_vol is not None
            and baseline_vol is not None
            and baseline_vol > 0
            and recent_vol > cfg.vol_spike_mult * baseline_vol
        ):
            reasons.append(f"volatility spike ({recent_vol / baseline_vol:.1f}x)")

        return (len(reasons) > 0, reasons)
