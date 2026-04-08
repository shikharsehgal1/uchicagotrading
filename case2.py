from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

N_ASSETS = 25
TICKS_PER_DAY = 30
ASSET_COLUMNS = tuple(f"A{i:02d}" for i in range(N_ASSETS))

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
    """EWMA-smoothed realized vol from 30 intraday ticks per day."""
    n_ticks = tick_prices.shape[0]
    total_days = n_ticks // TICKS_PER_DAY
    start_day = max(0, total_days - n_days)
    daily_rvars = []
    for d in range(start_day, total_days):
        t0 = d * TICKS_PER_DAY
        day_p = tick_prices[t0:t0 + TICKS_PER_DAY]
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
    """
    Sector momentum + intraday realized vol + Ledoit-Wolf vol targeting.
    
    Confirmed best configuration after testing PAMR, B-L, HRP, partial
    rebalancing, and multiple signal combinations.
    """

    REBAL_EVERY = 5
    SEC_MOM_FAST = 21
    SEC_MOM_SLOW = 63
    MOM_BLEND = 0.6
    VOL_TARGET = 0.14
    SHORT_THRESHOLD = -0.02

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

        # === BLENDED SECTOR MOMENTUM ===
        sec_mom = {}
        for s in range(self.n_sectors):
            mask = self.sectors == s
            fast = float(np.mean(np.sum(rets[-self.SEC_MOM_FAST:, mask], axis=0)))
            slow = float(np.mean(np.sum(rets[-self.SEC_MOM_SLOW:, mask], axis=0)))
            sec_mom[s] = self.MOM_BLEND * fast + (1 - self.MOM_BLEND) * slow

        ranked = sorted(sec_mom.keys(), key=lambda s: sec_mom[s], reverse=True)

        worst_mom = sec_mom[ranked[-1]]
        if worst_mom < self.SHORT_THRESHOLD:
            allocs = {ranked[0]: 0.38, ranked[1]: 0.28,
                      ranked[2]: 0.16, ranked[3]: 0.10, ranked[4]: -0.08}
        else:
            allocs = {ranked[0]: 0.38, ranked[1]: 0.28,
                      ranked[2]: 0.16, ranked[3]: 0.10, ranked[4]: 0.08}

        # === WITHIN-SECTOR: inverse realized vol × cost discount ===
        rvol = intraday_realized_vol(price_history, n_days=21)

        weights = np.zeros(N_ASSETS)
        for s, alloc in allocs.items():
            idx = np.where(self.sectors == s)[0]
            inv_rvol = 1.0 / np.maximum(rvol[idx], 0.01)
            if alloc >= 0:
                cost_adj = 1.0 / (1.0 + self.spread[idx] * 100)
            else:
                cost_adj = 1.0 / (1.0 + self.borrow[idx] * 10)
            w = inv_rvol * cost_adj
            w_sum = float(np.sum(w))
            if w_sum < 1e-10:
                w = np.ones(len(idx)) / len(idx)
            else:
                w /= w_sum
            weights[idx] = alloc * w

        # === VOL TARGETING (Ledoit-Wolf) ===
        cov_win = min(63, rets.shape[0])
        try:
            cov = LedoitWolf().fit(rets[-cov_win:]).covariance_
        except Exception:
            cov = np.cov(rets[-cov_win:], rowvar=False)

        port_vol = float(np.sqrt(np.abs(weights @ cov @ weights))) * np.sqrt(252)
        if port_vol > 1e-6:
            vol_scalar = np.clip(self.VOL_TARGET / port_vol, 0.3, 1.5)
            weights *= vol_scalar

        weights = safe_weights(weights)

        # === COST GATE ===
        delta = weights - self.prev_weights
        est_cost = float(np.sum(self.spread / 2 * np.abs(delta))
                         + np.sum(2.5 * self.spread * delta ** 2))
        if est_cost < 0.0005 and day > 0:
            return self.prev_weights

        self.prev_weights = weights.copy()
        self.last_rebal_day = day
        return weights


def create_strategy() -> StrategyBase:
    return MyStrategy()
