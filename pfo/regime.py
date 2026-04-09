"""Smooth regime weights: trend vs reversion vs high-vol de-risk."""

from __future__ import annotations

import numpy as np

from pfo.constants import H_MED, H_SHORT, TRADING_DAYS


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def regime_weights(
    daily_rets: np.ndarray,
    sector_rets: np.ndarray,
    ivol_cross_section: np.ndarray,
) -> tuple[float, float, float]:
    """
    Returns (w_momentum_blend, w_reversion_blend, vol_scale).

    - Trend: |short sector mom| / (|longer| + eps) elevated → more momentum.
    - Reversion: cross-sectional dispersion / mean vol elevated → more MR.
    - High vol: elevated mean ivol → scale down exposure and turnover appetite.
    """
    if daily_rets.shape[0] < H_MED + 5 or sector_rets.shape[0] < H_MED + 5:
        return 0.58, 0.30, 0.90

    # Market-wide realized vol (mean asset 20d stdev of daily returns, annualized)
    tail = daily_rets[-H_MED:]
    dvol = np.std(tail, axis=0, ddof=1) * np.sqrt(float(TRADING_DAYS))
    mkt_vol = float(np.mean(dvol))

    # Dispersion: avg cross-sectional std
    xdisp = float(np.mean(np.std(tail, axis=1, ddof=1))) * np.sqrt(float(TRADING_DAYS))

    # Trend strength on sectors
    s5 = np.sum(sector_rets[-H_SHORT:], axis=0)
    s20 = np.sum(sector_rets[-H_MED:], axis=0)
    ratio = float(np.mean(np.abs(s5) / (np.abs(s20) + 1e-8)))
    trend_score = float(_sigmoid(4.0 * (ratio - 0.55)))

    # Breadth: fraction of sectors with positive 20d cum return
    c20 = np.sum(sector_rets[-H_MED:], axis=0)
    breadth = float(np.mean(c20 > 0))

    trend_combined = 0.5 * trend_score + 0.5 * _sigmoid(6.0 * (breadth - 0.45))

    # Dispersion vs vol → reversion opportunity
    disp_ratio = xdisp / (mkt_vol + 1e-8)
    rev_push = float(_sigmoid(3.5 * (disp_ratio - 1.05)))

    # High-vol regime
    med_ivol = float(np.median(ivol_cross_section))
    hv = float(_sigmoid(8.0 * (mkt_vol / (med_ivol + 1e-8) - 1.12)))

    w_mom = (1.0 - hv) * (0.32 + 0.68 * trend_combined)
    w_rev = (1.0 - hv) * (0.28 + 0.55 * rev_push) * (1.0 - 0.35 * trend_combined)
    # Normalize blend mass
    s = w_mom + w_rev
    if s > 1e-8:
        w_mom /= s
        w_rev /= s
    vol_scale = max(0.52, 1.0 - 0.40 * hv)
    return float(w_mom), float(w_rev), float(vol_scale)
