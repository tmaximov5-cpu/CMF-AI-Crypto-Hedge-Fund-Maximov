"""Trading agents.

Every agent is documented as: Input -> Logic -> Output -> Risk constraint
(see CLAUDE.md). An agent maps market data available up to bar t into a desired
position for the *next* bar. Agents NEVER look ahead; any model/scaler an agent
uses is fit on training data only.

sklearn classes stay CamelCase (RandomForestClassifier, ...); our own variables
and functions are lowercase. Heavy deps (sklearn / statsmodels / arch) are
imported lazily inside the agents that need them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Agent(ABC):
    """Base agent. Subclasses document the four fields and implement ``generate``."""

    #: Short statement of the risk constraint this agent respects.
    risk_constraint: str = "TODO: state the risk constraint"

    @abstractmethod
    def generate(self, *args, **kwargs) -> pd.Series:
        """Return a position series (applied with .shift(1) downstream)."""
        raise NotImplementedError


class EconometricAgent(Agent):
    """Classical time-series agent: ARIMA for direction, GARCH for volatility.

    Input            : daily returns (train slice to fit, test slice to act on).
    Logic            : ARIMA(p,d,q) one-step-ahead return forecast (walk-forward,
                       params fit on train, state updated with realized actuals);
                       GARCH(1,1) conditional-variance recursion filtered forward
                       with train-estimated omega/alpha/beta for a vol regime.
    Output           : position 1.0 (up, calm) / 0.5 (up, high-vol) / 0.0 (down).
    Risk constraint  : long-only; flat on a non-positive return forecast; halve
                       exposure when forecast vol is in the high-vol regime.

    After ``generate`` the attributes ``direction_``, ``cond_vol_`` and
    ``high_vol_`` (bool) are available for reuse by other agents.
    """

    risk_constraint = (
        "long-only; flat on non-positive return forecast; halve exposure in high-vol regime"
    )

    def __init__(self, arima_order: tuple[int, int, int] = (1, 0, 1), vol_quantile: float = 0.66):
        self.arima_order = arima_order
        self.vol_quantile = vol_quantile

    def fit(self, train_returns: pd.Series) -> "EconometricAgent":
        from arch import arch_model
        from statsmodels.tsa.arima.model import ARIMA

        self._train_returns = train_returns.dropna()
        self._arima_res = ARIMA(self._train_returns, order=self.arima_order).fit()

        # GARCH on percent returns (zero-mean) for numerical stability.
        scaled = self._train_returns * 100
        garch_res = arch_model(scaled, mean="Zero", vol="Garch", p=1, q=1, dist="normal").fit(
            disp="off"
        )
        self._omega = garch_res.params["omega"]
        self._alpha = garch_res.params["alpha[1]"]
        self._beta = garch_res.params["beta[1]"]
        self._last_var = float(garch_res.conditional_volatility[-1] ** 2)
        self._last_eps = float(scaled.iloc[-1])
        # High-vol regime = train conditional vol above this quantile (percent units).
        self.high_vol_threshold_ = float(
            np.quantile(garch_res.conditional_volatility, self.vol_quantile)
        )
        return self

    def generate(self, test_returns: pd.Series) -> pd.Series:
        test_returns = test_returns.dropna()
        idx = test_returns.index

        # --- ARIMA walk-forward: one-step-ahead forecast, then ingest the actual.
        direction = pd.Series(index=idx, dtype=float)
        cur = self._arima_res
        for t in idx:
            direction[t] = float(cur.forecast(1).iloc[0])
            cur = cur.append(test_returns.loc[[t]], refit=False)

        # --- GARCH conditional-variance recursion, filtered forward (percent units).
        cond_vol = pd.Series(index=idx, dtype=float)
        var_prev, eps_prev = self._last_var, self._last_eps
        for t in idx:
            var_t = self._omega + self._alpha * eps_prev**2 + self._beta * var_prev
            cond_vol[t] = np.sqrt(var_t)  # one-step-ahead vol for day t (info up to t-1)
            eps_prev = float(test_returns.loc[t] * 100)
            var_prev = var_t

        self.direction_ = direction
        self.cond_vol_ = cond_vol
        self.high_vol_ = cond_vol > self.high_vol_threshold_

        up = direction > 0
        position = pd.Series(0.0, index=idx)
        position[up & ~self.high_vol_] = 1.0
        position[up & self.high_vol_] = 0.5
        return position


class MLAgent(Agent):
    """Supervised classifier agent (next-day direction).

    Input            : feature matrix; target = next-day up (1) / down (0).
    Logic            : StandardScaler fit on TRAIN only, then a sklearn
                       classifier; P(up) from predict_proba.
    Output           : position 1.0 when P(up) > threshold, else 0.0.
    Risk constraint  : long-only, no leverage; enter only above the probability
                       threshold (default 0.55).

    After ``generate`` the predicted probabilities are on ``proba_``.
    """

    risk_constraint = "long-only, no leverage; enter only when P(up) > threshold"

    def __init__(self, model, threshold: float = 0.55):
        from sklearn.preprocessing import StandardScaler

        self.model = model
        self.threshold = threshold
        self.scaler = StandardScaler()

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "MLAgent":
        self.scaler.fit(X_train)
        self.model.fit(self.scaler.transform(X_train), y_train)
        return self

    def proba(self, X: pd.DataFrame) -> pd.Series:
        scaled = self.scaler.transform(X)
        return pd.Series(self.model.predict_proba(scaled)[:, 1], index=X.index)

    def generate(self, X_test: pd.DataFrame) -> pd.Series:
        self.proba_ = self.proba(X_test)
        return (self.proba_ > self.threshold).astype(float)


class DeterministicAIAgent(Agent):
    """Rule-based ensemble combining technical, ML, volatility and drawdown views.

    Input            : features (ma_ratio, momentum_7), ML P(up), high-vol flag,
                       close price (for live drawdown) — all on the test index.
    Logic            : deterministic priority rules mapping the four views to an
                       action {exit, reduce, buy, hold}.
    Output           : exposure in {0.0, 0.5, 1.0} (hold carries prior exposure).
    Risk constraint  : exposure capped at 1.0; hard exit on drawdown breach;
                       de-risk to 0.5 in high-vol or low-conviction regimes.
    """

    risk_constraint = (
        "exposure in {0,0.5,1.0}; hard exit on drawdown breach; de-risk in high-vol/uncertain regimes"
    )

    def __init__(self, dd_exit: float = -0.20, buy_threshold: float = 0.55, exit_threshold: float = 0.45):
        self.dd_exit = dd_exit
        self.buy_threshold = buy_threshold
        self.exit_threshold = exit_threshold

    def generate(
        self,
        features: pd.DataFrame,
        proba: pd.Series,
        high_vol: pd.Series,
        close: pd.Series,
    ) -> pd.Series:
        idx = features.index
        drawdown = close / close.cummax() - 1  # running-peak DD, data <= t
        bullish = (features["ma_ratio"] > 1) & (features["momentum_7"] > 0)

        exposure = pd.Series(index=idx, dtype=float)
        actions = pd.Series(index=idx, dtype=object)
        prev = 0.0
        for t in idx:
            p = float(proba[t])
            if drawdown[t] < self.dd_exit or p < self.exit_threshold:
                act, expo = "exit", 0.0
            elif bool(high_vol[t]) or p < self.buy_threshold:
                act, expo = "reduce", 0.5
            elif bool(bullish[t]) and p > self.buy_threshold:
                act, expo = "buy", 1.0
            else:
                act, expo = "hold", prev
            actions[t] = act
            exposure[t] = expo
            prev = expo

        self.actions_ = actions
        return exposure
