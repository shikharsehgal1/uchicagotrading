"""Four alpha pillars: sector mom, residual mom, residual MR, cost gate."""

from __future__ import annotations

import numpy as np

from pfo.constants import (
    H_LONG,
    H_MED,
    H_RES_REV,
    H_RES_VOL,
    H_SHORT,
    N_ASSETS,
)


def _sum_window(x: np.ndarray, h: int) -> np.ndarray:
    """Sum last h rows of x (time x cross); if shorter, use all."""
    if x.shape[0] == 0:
        return np.zeros(x.shape[1])
    hh = min(h, x.shape[0])
    return np.sum(x[-hh:], axis=0)


def _zscore_cs(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    m, s = float(np.mean(v)), float(np.std(v, ddof=1))
    if s < eps:
        return np.zeros_like(v)
    return (v - m) / s


def alpha_sector_momentum(
    sector_rets: np.ndarray, n_sectors: int, sector_id: np.ndarray
) -> np.ndarray:
    """
    (A) Multi-horizon sector momentum → per-asset score (same score within sector).
    sector_rets: (T, n_sectors)
    """
    if sector_rets.shape[0] < H_SHORT + 1:
        return np.zeros(N_ASSETS)

    c5 = _sum_window(sector_rets, H_SHORT)
    c20 = _sum_window(sector_rets, H_MED)
    c60 = _sum_window(sector_rets, H_LONG)
    sec_score = np.zeros(n_sectors)
    for s in range(n_sectors):
        raw = 0.25 * c5[s] + 0.35 * c20[s] + 0.40 * c60[s]
        sec_score[s] = raw
    sec_z = _zscore_cs(sec_score)
    out = np.zeros(N_ASSETS)
    for i in range(N_ASSETS):
        out[i] = sec_z[int(sector_id[i])]
    return out


def alpha_residual_momentum(residual_rets: np.ndarray) -> np.ndarray:
    """(B) 5d and 20d cumulative residual return, cross-sectionally z-scored."""
    if residual_rets.shape[0] < H_SHORT:
        return np.zeros(N_ASSETS)
    r5 = _sum_window(residual_rets, H_SHORT)
    r20 = _sum_window(residual_rets, H_MED)
    raw = 0.45 * r5 + 0.55 * r20
    return _zscore_cs(raw)


def alpha_residual_mean_reversion(residual_rets: np.ndarray) -> np.ndarray:
    """
    (C) Short-horizon residual reversal when move is large vs local residual vol.
    """
    if residual_rets.shape[0] < max(H_RES_VOL, H_RES_REV) + 1:
        return np.zeros(N_ASSETS)

    short = _sum_window(residual_rets, H_RES_REV)
    tail = residual_rets[-H_RES_VOL:]
    vol = np.std(tail, axis=0, ddof=1)
    vol = np.maximum(vol, 1e-6)
    z = short / (vol * np.sqrt(float(H_RES_REV)) + 1e-9)
    signal = np.zeros(N_ASSETS)
    for i in range(N_ASSETS):
        if abs(z[i]) < 1.95:
            continue
        # Fade the residual jump
        signal[i] = -np.tanh(z[i] / 2.2)
    return _zscore_cs(signal)


def alpha_cost_adjust(
    alpha: np.ndarray,
    spread: np.ndarray,
    borrow: np.ndarray,
    half_spread: float,
) -> np.ndarray:
    """
    (D) Down-weight names with wide spreads; penalize short candidates by borrow;
    zero weak names vs a cost hurdle.
    """
    a = np.asarray(alpha, dtype=float).copy()
    # Liquidity tilt (multiplicative)
    sp = np.asarray(spread, dtype=float)
    a = a / (1.0 + 35.0 * sp)
    # Short-intent penalty: shrink negative alpha where borrow is high
    bor = np.asarray(borrow, dtype=float)
    pen = 1.0 / (1.0 + 2.8 * bor * np.sqrt(252.0))
    a = np.where(a < 0, a * pen, a)

    hurdle = 0.85 * float(half_spread) * np.sqrt(252.0)
    a = np.where(np.abs(a) * 0.012 < hurdle, 0.0, a)
    return a


def combine_alphas(
    a_sec: np.ndarray,
    a_res_m: np.ndarray,
    a_res_r: np.ndarray,
    w_mom: float,
    w_rev: float,
    vol_scale: float,
) -> np.ndarray:
    """Blend momentum block vs mean-reversion block, then global vol scaling."""
    mom_block = _zscore_cs(a_sec + a_res_m)
    rev_block = _zscore_cs(a_res_r)
    w_m = float(np.clip(w_mom, 0.0, 1.0))
    w_r = float(np.clip(w_rev, 0.0, 1.0))
    s = w_m * mom_block + w_r * rev_block
    if float(np.std(s, ddof=1)) < 1e-9:
        return np.zeros(N_ASSETS)
    s = _zscore_cs(s) * float(np.clip(vol_scale, 0.45, 1.0)) * 0.52
    return s
