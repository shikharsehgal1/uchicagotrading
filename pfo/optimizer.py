"""
Long-only mean–variance-inspired sizing (fast, stable, no per-day SLSQP).

We avoid noisy shorting and borrow drag while keeping explicit risk awareness:
  w_i ∝ max(μ_i, 0) / (γ σ_i^2) + risk-parity anchor 1/σ_i

Then cap each name, renormalize to simplex (Σ w = 1, gross ≤ 1).
This is a standard production approximation when impact precludes full QP each bar.
"""

from __future__ import annotations

import numpy as np

from pfo.constants import GAMMA_MV, MAX_ABS_WEIGHT, N_ASSETS


def solve_mean_variance(
    mu: np.ndarray,
    sigma: np.ndarray,
    w0: np.ndarray | None = None,
    gamma: float = GAMMA_MV,
    w_max: float = MAX_ABS_WEIGHT,
    gross_cap: float = 1.0,
) -> np.ndarray:
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float)
    d = np.sqrt(np.clip(np.diag(sigma), 1e-12, None))
    # Risk-parity anchor + positive-mu tilt (long-only)
    rp = 1.0 / (d * np.sqrt(252.0) + 1e-8)
    tilt = np.maximum(mu, 0.0) / (gamma * (d**2) + 1e-10)
    raw = 2.4 * rp + 1.15 * tilt
    raw = np.maximum(raw, 1e-12)
    w = raw / float(np.sum(raw))
    w = np.clip(w, 0.0, w_max)
    s = float(np.sum(w))
    if s > 1e-12:
        w = w / s
    if float(np.sum(w)) > gross_cap + 1e-12:
        w *= gross_cap / float(np.sum(w))
    return w
