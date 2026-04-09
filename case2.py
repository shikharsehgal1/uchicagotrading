"""
UChicago Case 2 — submission API.

Implementation lives in ``pfo/`` (data, signals, regime, risk, optimizer, execution).
See STRATEGY.md for economics and architecture.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pfo.constants import ASSET_COLUMNS, N_ASSETS
from pfo.strategy import PortfolioOptimizationStrategy


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


class MyStrategy(PortfolioOptimizationStrategy, StrategyBase):
    """Thin adapter so external code expecting ``MyStrategy`` keeps working."""

    def __init__(self) -> None:
        PortfolioOptimizationStrategy.__init__(self)


def create_strategy() -> StrategyBase:
    return MyStrategy()
