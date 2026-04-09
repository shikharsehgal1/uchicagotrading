"""Competition-aligned backtest harness (tick-level PnL, borrow, transaction costs).

Mirrors the reference ``validate.py`` mechanics so local results match the case.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from case2 import PublicMeta

N_ASSETS = 25
TICKS_PER_DAY = 30
TRADING_DAYS_PER_YEAR = 252
IMPACT_MULT = 2.5
DT_YEAR = 1.0 / (TRADING_DAYS_PER_YEAR * TICKS_PER_DAY)

TRAIN_YEARS = 4
HOLDOUT_YEARS = 1
TRAIN_TICKS = TRAIN_YEARS * TRADING_DAYS_PER_YEAR * TICKS_PER_DAY
HOLDOUT_TICKS = HOLDOUT_YEARS * TRADING_DAYS_PER_YEAR * TICKS_PER_DAY


def project_to_gross_limit(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float).copy()
    gross = float(np.sum(np.abs(w)))
    if not np.isfinite(gross):
        return w
    if gross > 1.0:
        w /= gross
    return w


def _transaction_cost(
    spread: np.ndarray, delta_weights: np.ndarray, impact_mult: float
) -> tuple[float, float]:
    linear = float(np.sum((spread / 2.0) * np.abs(delta_weights)))
    quadratic = float(np.sum((impact_mult * spread) * (delta_weights**2)))
    return linear, quadratic


def _hold_fixed_weights_one_day(
    wealth: float,
    weights: np.ndarray,
    logret: np.ndarray,
    borrow: np.ndarray,
    *,
    day: int,
) -> float:
    t0 = day * TICKS_PER_DAY
    t_begin = t0 + 1 if day == 0 else t0
    for t in range(t_begin, t0 + TICKS_PER_DAY):
        pnl = float(np.sum(weights * (np.exp(logret[t]) - 1.0)))
        borrow_cost = float(np.sum(np.maximum(-weights, 0.0) * borrow) * DT_YEAR)
        wealth *= 1.0 + pnl - borrow_cost
    return wealth


def _history_through_day(
    train_prices: np.ndarray, hold_prices: np.ndarray, day: int
) -> np.ndarray:
    cutoff = (day + 1) * TICKS_PER_DAY
    return np.vstack([train_prices, hold_prices[:cutoff]])


def annualized_sharpe(daily_returns: np.ndarray) -> float:
    x = np.asarray(daily_returns, dtype=float)
    mu, sd = float(np.mean(x)), float(np.std(x, ddof=1))
    if not np.isfinite(sd) or sd < 1e-12:
        return -math.inf if mu <= 0 else math.inf
    return math.sqrt(TRADING_DAYS_PER_YEAR) * mu / sd


def run_backtest(
    train_prices: np.ndarray,
    hold_prices: np.ndarray,
    strategy: Any,
    meta: PublicMeta,
) -> dict[str, Any]:
    spread = np.asarray(meta.spread_bps, dtype=float) / 1e4
    borrow = np.asarray(meta.borrow_bps_annual, dtype=float) / 1e4

    strategy.fit(train_prices, meta, ticks_per_day=TICKS_PER_DAY)
    weights = project_to_gross_limit(strategy.get_weights(train_prices, meta, day=0))
    if not np.all(np.isfinite(weights)):
        raise ValueError("Non-finite weights at initialization")

    wealth = 1.0
    entry_linear, entry_quadratic = _transaction_cost(spread, weights, IMPACT_MULT)
    wealth *= 1.0 - (entry_linear + entry_quadratic)

    logret = np.zeros_like(hold_prices)
    logret[1:] = np.log(hold_prices[1:] / hold_prices[:-1])

    n_days = hold_prices.shape[0] // TICKS_PER_DAY
    daily_returns = np.zeros(n_days)
    daily_costs = np.zeros(n_days + 1)
    daily_costs[0] = entry_linear + entry_quadratic
    daily_turnover = np.zeros(n_days + 1)
    daily_turnover[0] = 0.0

    for day in range(n_days):
        wealth_start = wealth
        try:
            wealth = _hold_fixed_weights_one_day(wealth, weights, logret, borrow, day=day)
        except FloatingPointError:
            wealth = float("nan")
        if wealth <= 0 or not np.isfinite(wealth):
            daily_returns[day:] = -1.0
            return {
                "daily_returns": daily_returns,
                "daily_costs": daily_costs[: day + 1],
                "daily_turnover": daily_turnover[: day + 1],
                "blown_up": True,
            }

        history = _history_through_day(train_prices, hold_prices, day)
        target = project_to_gross_limit(strategy.get_weights(history, meta, day=day + 1))
        if not np.all(np.isfinite(target)):
            raise ValueError(f"Non-finite weights on holdout day {day}")

        delta = target - weights
        linear, quadratic = _transaction_cost(spread, delta, IMPACT_MULT)
        trade_cost = linear + quadratic
        wealth *= 1.0 - trade_cost
        daily_costs[day + 1] = trade_cost
        daily_turnover[day + 1] = float(np.sum(np.abs(delta)))
        daily_returns[day] = wealth / wealth_start - 1.0
        weights = target

    return {
        "daily_returns": daily_returns,
        "daily_costs": daily_costs,
        "daily_turnover": daily_turnover,
        "blown_up": False,
    }


def summarize_result(result: dict[str, Any]) -> dict[str, float | bool]:
    dr = result["daily_returns"]
    costs = result["daily_costs"]
    turnover = result.get("daily_turnover", np.zeros(1))
    cum = np.cumprod(1.0 + dr)
    max_dd = float(np.min(np.minimum.accumulate(cum) / np.maximum.accumulate(cum) - 1.0))
    return {
        "sharpe": annualized_sharpe(dr),
        "total_return": float(np.prod(1.0 + dr) - 1.0),
        "total_cost": float(np.sum(costs)),
        "max_drawdown": max_dd,
        "blown_up": bool(result["blown_up"]),
        "mean_daily_turnover": float(np.mean(turnover[1:])) if turnover.size > 1 else 0.0,
    }


def cv_folds(prices: np.ndarray) -> list[tuple[int, int, int]]:
    ticks_per_year = TRADING_DAYS_PER_YEAR * TICKS_PER_DAY
    total_years = prices.shape[0] // ticks_per_year
    folds: list[tuple[int, int, int]] = []
    for k in range(2, total_years):
        train_end = k * ticks_per_year
        test_end = (k + 1) * ticks_per_year
        if test_end > prices.shape[0]:
            break
        folds.append((k, train_end, test_end))
    return folds
