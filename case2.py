from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

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

class MyStrategy(StrategyBase):
    """Exact reproduction of the 1.44 Sharpe version."""
    REBAL_EVERY = 5
    SEC_MOM_FAST = 21
    SEC_MOM_SLOW = 63
    MOM_BLEND = 0.6
    VOL_WIN = 42
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

        # Blended sector momentum (fast 21d + slow 63d)
        sec_mom = {}
        for s in range(self.n_sectors):
            mask = self.sectors == s
            fast = float(np.mean(np.sum(rets[-self.SEC_MOM_FAST:, mask], axis=0)))
            slow = float(np.mean(np.sum(rets[-self.SEC_MOM_SLOW:, mask], axis=0)))
            sec_mom[s] = self.MOM_BLEND * fast + (1 - self.MOM_BLEND) * slow

        ranked = sorted(sec_mom.keys(), key=lambda s: sec_mom[s], reverse=True)

        # Adaptive shorting
        worst_mom = sec_mom[ranked[-1]]
        if worst_mom < self.SHORT_THRESHOLD:
            allocs = {ranked[0]: 0.38, ranked[1]: 0.28,
                      ranked[2]: 0.16, ranked[3]: 0.10, ranked[4]: -0.08}
        else:
            allocs = {ranked[0]: 0.38, ranked[1]: 0.28,
                      ranked[2]: 0.16, ranked[3]: 0.10, ranked[4]: 0.08}

        # Within sector: inv-vol * cost discount
        weights = np.zeros(N_ASSETS)
        recent = rets[-self.VOL_WIN:]

        for s, alloc in allocs.items():
            idx = np.where(self.sectors == s)[0]
            vols = np.std(recent[:, idx], axis=0, ddof=1)
            vols = np.maximum(vols, 1e-8)
            inv_vol = 1.0 / vols
            if alloc >= 0:
                cost_adj = 1.0 / (1.0 + self.spread[idx] * 100)
            else:
                cost_adj = 1.0 / (1.0 + self.borrow[idx] * 10)
            w = inv_vol * cost_adj
            w_sum = float(np.sum(w))
            if w_sum < 1e-10:
                w = np.ones(len(idx)) / len(idx)
            else:
                w /= w_sum
            weights[idx] = alloc * w

        # Vol targeting (shrunk cov)
        cov_win = min(63, rets.shape[0])
        cov = np.cov(rets[-cov_win:], rowvar=False)
        mu = np.trace(cov) / N_ASSETS
        alpha_shrink = min(1.0, max(0.0, 2.0 / cov_win))
        cov = (1 - alpha_shrink) * cov + alpha_shrink * mu * np.eye(N_ASSETS)

        port_vol = float(np.sqrt(np.abs(weights @ cov @ weights))) * np.sqrt(252)
        if port_vol > 1e-6:
            vol_scalar = np.clip(self.VOL_TARGET / port_vol, 0.3, 1.5)
            weights *= vol_scalar

        weights = safe_weights(weights)

        # Cost gate
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
