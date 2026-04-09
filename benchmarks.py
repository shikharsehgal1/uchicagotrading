"""Baseline strategies for comparison (equal-weight, vol-managed EW)."""

from __future__ import annotations

import numpy as np

from case2 import N_ASSETS, PublicMeta, StrategyBase
from pfo.data_pipeline import daily_closes_from_ticks as daily_close
from pfo.strategy import safe_weights

TRADING_DAYS_PER_YEAR = 252


class EqualWeightStrategy(StrategyBase):
    """1/N long-only, rebalanced whenever the harness asks for new weights."""

    def get_weights(self, price_history: np.ndarray, meta: PublicMeta, day: int) -> np.ndarray:
        return np.ones(N_ASSETS, dtype=float) / N_ASSETS


class VolManagedEqualWeightStrategy(StrategyBase):
    """Long-only inverse-vol weights from recent daily log returns (competition-style signal)."""

    VOL_WIN = 21

    def fit(self, train_prices: np.ndarray, meta: PublicMeta, **kwargs) -> None:
        self._prev = np.ones(N_ASSETS, dtype=float) / N_ASSETS

    def get_weights(self, price_history: np.ndarray, meta: PublicMeta, day: int) -> np.ndarray:
        dp = daily_close(price_history)
        need = self.VOL_WIN + 1
        if dp.shape[0] < need or np.any(dp <= 0) or not np.all(np.isfinite(dp)):
            return safe_weights(self._prev.copy())

        rets = np.diff(np.log(dp[-need:]), axis=0)
        if not np.all(np.isfinite(rets)):
            return safe_weights(self._prev.copy())

        vol = np.std(rets, axis=0, ddof=1) * np.sqrt(float(TRADING_DAYS_PER_YEAR))
        vol = np.maximum(vol, 1e-4)
        invv = 1.0 / vol
        w = invv / np.sum(invv)
        self._prev = w
        return safe_weights(w)
