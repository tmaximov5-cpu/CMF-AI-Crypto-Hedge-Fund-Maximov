"""AI Crypto Hedge Fund — reusable logic imported by notebook.ipynb.

All strategy/backtest logic lives here so the notebook stays a thin, readable
orchestration layer. See CLAUDE.md for the non-negotiable rules (no look-ahead
bias, crypto annualization = 365, transaction costs, benchmarks, etc.).
"""

# Crypto trades 365 days/year — annualize with 365 (and sqrt(365)), never 252.
ANNUALIZATION = 365

__all__ = ["ANNUALIZATION"]
