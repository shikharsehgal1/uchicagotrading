"""Covariance with Ledoit–Wolf shrinkage on rolling daily returns."""

from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf

from pfo.constants import COV_WINDOW


def estimate_covariance(daily_rets: np.ndarray) -> np.ndarray:
    """
    daily_rets: (T, n_assets). Uses last COV_WINDOW rows.
    Falls back to diagonal sample variance if LW fails.
    """
    n = daily_rets.shape[0]
    w = min(COV_WINDOW, n)
    if w < 5:
        return np.eye(daily_rets.shape[1]) * 0.04 / 252.0

    x = daily_rets[-w:]
    try:
        cov = LedoitWolf().fit(x).covariance_
    except Exception:
        cov = np.cov(x, rowvar=False)

    cov = np.asarray(cov, dtype=float)
    d = np.diag(np.diag(cov))
    cov = 0.74 * cov + 0.26 * d
    eigvals, eigvec = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-10)
    cov = (eigvec * eigvals) @ eigvec.T
    return cov
