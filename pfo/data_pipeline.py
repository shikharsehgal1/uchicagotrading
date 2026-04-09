"""Tick → daily features; no lookahead beyond observed closes."""

from __future__ import annotations

import numpy as np

from pfo.constants import N_ASSETS, TICKS_PER_DAY


def daily_closes_from_ticks(tick_prices: np.ndarray) -> np.ndarray:
    """Last tick of each full day; shape (n_days, n_assets)."""
    n_ticks = tick_prices.shape[0]
    n_days = n_ticks // TICKS_PER_DAY
    if n_days == 0:
        return tick_prices[:1]
    idx = np.arange(TICKS_PER_DAY - 1, n_days * TICKS_PER_DAY, TICKS_PER_DAY)
    return tick_prices[idx].astype(float)


def daily_log_returns(closes: np.ndarray) -> np.ndarray:
    """r[t] = log(c[t]/c[t-1]); shape (n_days-1, n_assets)."""
    if closes.shape[0] < 2:
        return np.zeros((0, closes.shape[1]))
    return np.diff(np.log(np.maximum(closes, 1e-12)), axis=0)


def intraday_realized_vol_window(
    tick_prices: np.ndarray, n_days: int = 21
) -> np.ndarray:
    """Annualized sqrt of EWMA of intraday sum of squared log returns per asset."""
    n_ticks = tick_prices.shape[0]
    total_days = n_ticks // TICKS_PER_DAY
    start = max(0, total_days - n_days)
    daily_rv: list[np.ndarray] = []
    for d in range(start, total_days):
        t0 = d * TICKS_PER_DAY
        seg = tick_prices[t0 : t0 + TICKS_PER_DAY]
        if seg.shape[0] < 2:
            continue
        lr = np.diff(np.log(np.maximum(seg, 1e-12)), axis=0)
        daily_rv.append(np.sum(lr**2, axis=0))
    if not daily_rv:
        return np.ones(N_ASSETS) * 0.15
    rvars = np.array(daily_rv)
    lam = 0.94
    ew = rvars[0].copy()
    for i in range(1, len(rvars)):
        ew = lam * ew + (1 - lam) * rvars[i]
    return np.sqrt(np.maximum(ew, 1e-12) * 252)


def cross_sectional_dispersion_series(daily_rets: np.ndarray, win: int = 20) -> float:
    """Mean over last `win` days of cross-sectional std of daily returns."""
    if daily_rets.shape[0] < win:
        win = daily_rets.shape[0]
    if win <= 0:
        return 0.0
    tail = daily_rets[-win:]
    return float(np.mean(np.std(tail, axis=1, ddof=1)))


def sector_return_series(
    daily_rets: np.ndarray, sector_id: np.ndarray, n_sectors: int
) -> np.ndarray:
    """Sector equal-weight return each day; shape (T, n_sectors)."""
    t, _ = daily_rets.shape
    out = np.zeros((t, n_sectors))
    for s in range(n_sectors):
        m = sector_id == s
        if not np.any(m):
            continue
        out[:, s] = np.mean(daily_rets[:, m], axis=1)
    return out


def residual_returns(
    daily_rets: np.ndarray, sector_id: np.ndarray, n_sectors: int
) -> np.ndarray:
    """Asset minus its sector return each day."""
    sec = sector_return_series(daily_rets, sector_id, n_sectors)
    res = np.zeros_like(daily_rets)
    for i in range(N_ASSETS):
        s = int(sector_id[i])
        res[:, i] = daily_rets[:, i] - sec[:, s]
    return res
