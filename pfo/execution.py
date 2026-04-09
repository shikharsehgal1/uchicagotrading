"""Turnover control: no-trade band + partial move toward target."""

from __future__ import annotations

import numpy as np

from pfo.constants import EXEC_BLEND, IMPACT_MULT, NO_TRADE_FRAC_OF_HALFSPREAD


def marginal_txn_cost(spread: np.ndarray, delta: np.ndarray) -> float:
    linear = float(np.sum((spread / 2.0) * np.abs(delta)))
    quad = float(np.sum(IMPACT_MULT * spread * (delta**2)))
    return linear + quad


def apply_execution(
    w_target: np.ndarray,
    w_prev: np.ndarray,
    spread: np.ndarray,
    *,
    blend: float = EXEC_BLEND,
) -> np.ndarray:
    """
    1) If marginal benefit of full move is small vs estimated txn, hold.
    2) Else move partially: w_prev + blend * (w_target - w_prev).
    """
    w_target = np.asarray(w_target, dtype=float)
    w_prev = np.asarray(w_prev, dtype=float)
    spread = np.asarray(spread, dtype=float)

    delta_full = w_target - w_prev
    half_spread = float(np.mean(spread) / 2.0)
    band = max(1e-6, NO_TRADE_FRAC_OF_HALFSPREAD * half_spread)
    if float(np.max(np.abs(delta_full))) < band:
        return w_prev.copy()

    w_intent = w_prev + float(np.clip(blend, 0.0, 1.0)) * delta_full

    delta = w_intent - w_prev
    cost = marginal_txn_cost(spread, delta)
    # Hurdle: skip partial move if cost dominates a crude benefit proxy
    benefit_proxy = float(np.sum(np.abs(delta_full))) * half_spread * 0.4
    if cost > benefit_proxy * 2.2 and float(np.max(np.abs(delta))) < band * 3:
        return w_prev.copy()

    g = np.sum(np.abs(w_intent))
    if g > 1.0 and g > 1e-12:
        w_intent /= g
    return w_intent
