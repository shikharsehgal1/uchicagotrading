from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

N_ASSETS = 25
TICKS_PER_DAY = 30
ASSET_COLUMNS = tuple(f"A{i:02d}" for i in range(N_ASSETS))

# Ablation toggles for research (defaults match original submission behavior).
FEATURE_SECTOR_MOMENTUM = True
FEATURE_INV_VOL_WITHIN_SECTOR = True
FEATURE_VOL_TARGETING = True
FEATURE_COST_GATE = True
USE_LEDOIT_WOLF = True


@dataclass(frozen=True)
class PublicMeta:
    sector_id: np.ndarray
    spread_bps: np.ndarray
    borrow_bps_annual: np.ndarray


def load_prices(path: str = "prices.csv") -> np.ndarray:
    df = pd.read_csv(path, index_col="tick")
    return df[list(ASSET_COLUMNS)].to_numpy(dtype=float)


def load_meta(path: str = "meta.csv") -> PublicMeta:
    df = pd.read_csv(path)
    return PublicMeta(
        sector_id=df["sector_id"].to_numpy(dtype=int),
        spread_bps=df["spread_bps"].to_numpy(dtype=float),
        borrow_bps_annual=df["borrow_bps_annual"].to_numpy(dtype=float),
    )


class StrategyBase:
    def fit(self, train_prices: np.ndarray, meta: PublicMeta, **kwargs) -> None:
        pass

    def get_weights(self, price_history: np.ndarray, meta: PublicMeta, day: int) -> np.ndarray:
        raise NotImplementedError


def daily_close(tick_prices: np.ndarray) -> np.ndarray:
    n = tick_prices.shape[0] // TICKS_PER_DAY
    if n == 0:
        return tick_prices[:1]
    return tick_prices[TICKS_PER_DAY - 1 : n * TICKS_PER_DAY : TICKS_PER_DAY]


def safe_weights(w: np.ndarray) -> np.ndarray:
    w = np.where(np.isfinite(w), w, 0.0)
    gross = np.sum(np.abs(w))
    if gross > 1.0:
        w /= gross
    return w


def intraday_realized_vol(tick_prices: np.ndarray, n_days: int = 21) -> np.ndarray:
    n_ticks = tick_prices.shape[0]
    total_days = n_ticks // TICKS_PER_DAY
    start_day = max(0, total_days - n_days)
    daily_rvars = []
    for d in range(start_day, total_days):
        t0 = d * TICKS_PER_DAY
        day_p = tick_prices[t0 : t0 + TICKS_PER_DAY]
        if day_p.shape[0] < 2:
            continue
        intra_rets = np.diff(np.log(day_p), axis=0)
        daily_rvars.append(np.sum(intra_rets**2, axis=0))
    if len(daily_rvars) == 0:
        return np.ones(tick_prices.shape[1]) * 0.15
    rvars = np.array(daily_rvars)
    lam = 0.94
    ewma = rvars[0].copy()
    for i in range(1, len(rvars)):
        ewma = lam * ewma + (1 - lam) * rvars[i]
    return np.sqrt(np.maximum(ewma, 1e-10) * 252)


class MyStrategy(StrategyBase):
    """Black-Litterman inspired: soft momentum tilts on EW prior."""

    # Defaults chosen by coarse CV on the released training CSV (tune_cv.py, robust score).
    REBAL_EVERY = 3
    SEC_MOM_FAST = 21
    SEC_MOM_SLOW = 63
    MOM_BLEND = 0.65
    VOL_TARGET = 0.12
    TAU = 0.13
    LONG_ONLY = False
    COST_GATE_THRESHOLD = 0.0004
    # Rolling length (days) for smoothing daily log-returns used only in sector momentum; 0 = off.
    MOM_SMOOTH_DAYS = 0

    def fit(self, train_prices: np.ndarray, meta: PublicMeta, **kwargs) -> None:
        self.spread = meta.spread_bps / 1e4
        self.borrow = meta.borrow_bps_annual / 1e4
        self.sectors = meta.sector_id
        self.n_sectors = int(np.max(self.sectors)) + 1
        self.prev_weights = np.ones(N_ASSETS) / N_ASSETS
        self.last_rebal_day = -999

    def get_weights(self, price_history: np.ndarray, meta: PublicMeta, day: int) -> np.ndarray:
        try:
            return self._compute_weights(price_history, meta, day)
        except Exception:
            return safe_weights(self.prev_weights.copy())

    def _estimate_cov(self, rets: np.ndarray) -> np.ndarray:
        cov_win = min(63, rets.shape[0])
        window = rets[-cov_win:]
        use_lw = getattr(self, "USE_LEDOIT_WOLF", USE_LEDOIT_WOLF)
        if use_lw:
            try:
                return LedoitWolf().fit(window).covariance_
            except Exception:
                pass
        return np.cov(window, rowvar=False)

    def _momentum_input_rets(self, rets: np.ndarray) -> np.ndarray:
        w = int(getattr(self, "MOM_SMOOTH_DAYS", self.MOM_SMOOTH_DAYS) or 0)
        if w <= 1 or rets.shape[0] < w:
            return rets
        t, n = rets.shape
        c = np.concatenate([np.zeros((1, n), dtype=rets.dtype), np.cumsum(rets, axis=0)], axis=0)
        out = np.empty_like(rets)
        out[: w - 1] = rets[: w - 1]
        out[w - 1 :] = (c[w:] - c[:-w]) / float(w)
        return out

    def _sector_momentum_view_z(self, rets: np.ndarray) -> np.ndarray:
        if not getattr(self, "FEATURE_SECTOR_MOMENTUM", FEATURE_SECTOR_MOMENTUM):
            return np.zeros(N_ASSETS)

        rets_m = self._momentum_input_rets(rets)
        sec_views: dict[int, float] = {}
        for s in range(self.n_sectors):
            mask = self.sectors == s
            fast = float(np.mean(np.sum(rets_m[-self.SEC_MOM_FAST :, mask], axis=0)))
            slow = float(np.mean(np.sum(rets_m[-self.SEC_MOM_SLOW :, mask], axis=0)))
            sec_views[s] = self.MOM_BLEND * fast + (1 - self.MOM_BLEND) * slow

        view_signal = np.zeros(N_ASSETS)
        for s in range(self.n_sectors):
            mask = self.sectors == s
            view_signal[mask] = sec_views[s]

        vs_std = np.std(view_signal)
        if vs_std > 1e-8:
            return (view_signal - np.mean(view_signal)) / vs_std
        return np.zeros(N_ASSETS)

    def _inv_vol_within_sector(self, price_history: np.ndarray) -> np.ndarray:
        if not getattr(self, "FEATURE_INV_VOL_WITHIN_SECTOR", FEATURE_INV_VOL_WITHIN_SECTOR):
            inv_rvol = np.ones(N_ASSETS)
        else:
            rvol = intraday_realized_vol(price_history, n_days=21)
            inv_rvol = 1.0 / np.maximum(rvol, 0.01)

        for s in range(self.n_sectors):
            idx = np.where(self.sectors == s)[0]
            sector_sum = np.sum(inv_rvol[idx])
            if sector_sum > 1e-10:
                inv_rvol[idx] /= sector_sum
        return inv_rvol

    def _apply_vol_target(self, weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
        if not getattr(self, "FEATURE_VOL_TARGETING", FEATURE_VOL_TARGETING):
            return weights
        port_vol = float(np.sqrt(np.abs(weights @ cov @ weights))) * np.sqrt(252)
        if port_vol > 1e-6:
            vol_scalar = np.clip(self.VOL_TARGET / port_vol, 0.3, 1.5)
            weights = weights * vol_scalar
        return safe_weights(weights)

    def _maybe_skip_rebalance_for_cost(self, weights: np.ndarray, day: int) -> np.ndarray | None:
        if not getattr(self, "FEATURE_COST_GATE", FEATURE_COST_GATE):
            return None
        delta = weights - self.prev_weights
        est_cost = float(
            np.sum(self.spread / 2 * np.abs(delta)) + np.sum(2.5 * self.spread * delta**2)
        )
        thr = float(getattr(self, "COST_GATE_THRESHOLD", self.COST_GATE_THRESHOLD))
        if est_cost < thr and day > 0:
            return self.prev_weights
        return None

    def _compute_weights(self, price_history: np.ndarray, meta: PublicMeta, day: int) -> np.ndarray:
        if day - self.last_rebal_day < self.REBAL_EVERY and day > 0:
            return self.prev_weights

        dp = daily_close(price_history)
        if dp.shape[0] < self.SEC_MOM_SLOW + 5:
            return safe_weights(self.prev_weights.copy())
        if np.any(dp <= 0) or not np.all(np.isfinite(dp)):
            return safe_weights(self.prev_weights.copy())

        rets = np.diff(np.log(dp), axis=0)
        if not np.all(np.isfinite(rets)):
            return safe_weights(self.prev_weights.copy())

        cov = self._estimate_cov(rets)

        view_z = self._sector_momentum_view_z(rets)
        tilt = self.TAU * view_z

        inv_rvol = self._inv_vol_within_sector(price_history)
        cost_adj = 1.0 / (1.0 + self.spread * 100)

        base = 1.0 / N_ASSETS
        weights = (base + tilt) * inv_rvol * cost_adj

        if getattr(self, "LONG_ONLY", False):
            weights = np.maximum(weights, 0.0)
        else:
            weights = np.where(weights < -0.02, weights, np.maximum(weights, 0.0))
        weights = safe_weights(weights)

        weights = self._apply_vol_target(weights, cov)

        skip = self._maybe_skip_rebalance_for_cost(weights, day)
        if skip is not None:
            return skip

        self.prev_weights = weights.copy()
        self.last_rebal_day = day
        return weights


def create_strategy() -> StrategyBase:
    return MyStrategy()


def create_strategy_with_params(**params: object) -> StrategyBase:
    """Instantiate ``MyStrategy`` with instance-level overrides (for CV tuning)."""
    s = MyStrategy()
    for k, v in params.items():
        setattr(s, k, v)
    return s
