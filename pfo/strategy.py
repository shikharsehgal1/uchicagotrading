"""End-to-end strategy: data → signals → regime → risk → MV → execution."""

from __future__ import annotations

import numpy as np

from pfo.constants import MIN_HISTORY_DAYS, N_ASSETS, TARGET_EMA, TRADING_DAYS
from pfo.data_pipeline import (
    daily_closes_from_ticks,
    daily_log_returns,
    intraday_realized_vol_window,
    residual_returns,
    sector_return_series,
)
from pfo.execution import apply_execution
from pfo.optimizer import solve_mean_variance
from pfo.regime import regime_weights
from pfo.risk import estimate_covariance
from pfo.signals import (
    alpha_cost_adjust,
    alpha_residual_mean_reversion,
    alpha_residual_momentum,
    alpha_sector_momentum,
    combine_alphas,
)


def safe_weights(w: np.ndarray) -> np.ndarray:
    w = np.where(np.isfinite(w), w, 0.0)
    g = float(np.sum(np.abs(w)))
    if g > 1.0:
        w = w / g
    return w


class PortfolioOptimizationStrategy:
    """
    Systematic portfolio construction (not the legacy heuristic stack).
    Implements competition ``StrategyBase``-compatible interface via duck typing.
    """

    def __init__(self) -> None:
        self.spread = np.zeros(N_ASSETS)
        self.borrow = np.zeros(N_ASSETS)
        self.sector_id = np.zeros(N_ASSETS, dtype=int)
        self.n_sectors = 1
        self.prev_weights = np.ones(N_ASSETS) / float(N_ASSETS)
        self._w_target_ema: np.ndarray | None = None
        self._initialized = False

    def fit(self, train_prices: np.ndarray, meta: object, **kwargs) -> None:
        self.spread = np.asarray(meta.spread_bps, dtype=float) / 1e4
        self.borrow = np.asarray(meta.borrow_bps_annual, dtype=float) / 1e4
        self.sector_id = np.asarray(meta.sector_id, dtype=int)
        self.n_sectors = int(np.max(self.sector_id)) + 1
        self.prev_weights = np.ones(N_ASSETS) / float(N_ASSETS)
        self._w_target_ema = None
        self._initialized = True

    def get_weights(self, price_history: np.ndarray, meta: object, day: int) -> np.ndarray:
        try:
            return self._compute(price_history)
        except Exception:
            return safe_weights(self.prev_weights.copy())

    def _compute(self, price_history: np.ndarray) -> np.ndarray:
        tick = np.asarray(price_history, dtype=float)
        closes = daily_closes_from_ticks(tick)
        if closes.shape[0] < MIN_HISTORY_DAYS:
            return safe_weights(self.prev_weights.copy())

        if np.any(closes <= 0) or not np.all(np.isfinite(closes)):
            return safe_weights(self.prev_weights.copy())

        daily_rets = daily_log_returns(closes)
        if daily_rets.shape[0] < 30:
            return safe_weights(self.prev_weights.copy())

        sec_rets = sector_return_series(daily_rets, self.sector_id, self.n_sectors)
        res_rets = residual_returns(daily_rets, self.sector_id, self.n_sectors)

        ivol = intraday_realized_vol_window(tick, n_days=21)

        a_sec = alpha_sector_momentum(sec_rets, self.n_sectors, self.sector_id)
        a_rm = alpha_residual_momentum(res_rets)
        a_rr = alpha_residual_mean_reversion(res_rets)

        w_mom, w_rev, vol_scale = regime_weights(daily_rets, sec_rets, ivol)
        blended = combine_alphas(a_sec, a_rm, a_rr, w_mom, w_rev, vol_scale)

        half_spread = float(np.mean(self.spread) / 2.0)
        mu_hat = alpha_cost_adjust(blended, self.spread, self.borrow, half_spread)

        # Map dimensionless scores → expected return scale using recent realized vols
        tail = min(60, daily_rets.shape[0])
        dvol = np.std(daily_rets[-tail:], axis=0, ddof=1) * np.sqrt(float(TRADING_DAYS))
        dvol = np.maximum(dvol, 1e-6)
        mu = mu_hat * dvol * 0.22

        sigma = estimate_covariance(daily_rets)
        w_star = solve_mean_variance(mu, sigma, w0=self.prev_weights.copy())
        # Anchor to diversification baseline (reduces fragility in short-train folds)
        ew = np.ones(N_ASSETS, dtype=float) / float(N_ASSETS)
        w_star = 0.26 * ew + 0.74 * w_star
        w_star = w_star / (float(np.sum(w_star)) + 1e-12)

        if self._w_target_ema is None:
            self._w_target_ema = w_star.copy()
        else:
            self._w_target_ema = float(TARGET_EMA) * self._w_target_ema + (1.0 - float(TARGET_EMA)) * w_star

        w_exec = apply_execution(self._w_target_ema, self.prev_weights, self.spread)
        w_exec = safe_weights(w_exec)
        self.prev_weights = w_exec.copy()
        return w_exec


def create_portfolio_strategy() -> PortfolioOptimizationStrategy:
    return PortfolioOptimizationStrategy()
