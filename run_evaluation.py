#!/usr/bin/env python3
"""Compare strategies on competition-style splits (single 4y/1y or time-series CV).

Usage:
  python run_evaluation.py --data-dir "/path/to/participant folder"
  python run_evaluation.py --data-dir "..." --cv --strategies all
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from benchmarks import EqualWeightStrategy, VolManagedEqualWeightStrategy
from case2 import N_ASSETS, PublicMeta, create_strategy, load_meta, load_prices
from harness import (
    HOLDOUT_TICKS,
    HOLDOUT_YEARS,
    TRAIN_TICKS,
    TRAIN_YEARS,
    TICKS_PER_DAY,
    cv_folds,
    run_backtest,
    summarize_result,
)


def _strategy_factories():
    return {
        "my": create_strategy,
        "equal_weight": EqualWeightStrategy,
        "vol_ew": VolManagedEqualWeightStrategy,
    }


def _resolve_strategies(names: list[str]) -> dict[str, callable]:
    fac = _strategy_factories()
    if "all" in names:
        return {k: fac[k] for k in ("my", "equal_weight", "vol_ew")}
    out = {}
    for n in names:
        if n not in fac:
            raise SystemExit(f"Unknown strategy {n!r}. Choose from: {sorted(fac)} or all")
        out[n] = fac[n]
    return out


def _load(data_dir: str) -> tuple[np.ndarray, PublicMeta]:
    p = os.path.join(data_dir, "prices.csv")
    m = os.path.join(data_dir, "meta.csv")
    prices = load_prices(p)
    meta = load_meta(m)
    assert prices.shape[1] == N_ASSETS
    return prices, meta


def run_single(data_dir: str, strategies: dict[str, callable]) -> None:
    prices, meta = _load(data_dir)
    train_prices = prices[:TRAIN_TICKS]
    hold_prices = prices[TRAIN_TICKS : TRAIN_TICKS + HOLDOUT_TICKS]

    print(f"Data: {prices.shape[0]:,} ticks ({prices.shape[0] // TICKS_PER_DAY} days)")
    print(f"Split: train {TRAIN_YEARS}y ({train_prices.shape[0]:,} ticks) / hold {HOLDOUT_YEARS}y")
    print("=" * 60)

    rows = []
    for name, factory in strategies.items():
        strat = factory()
        res = run_backtest(train_prices, hold_prices, strat, meta)
        s = summarize_result(res)
        rows.append((name, s))
        print(f"\n[{name}]")
        if s["blown_up"]:
            print("  ** BLOWN UP **")
        print(f"  Sharpe:        {s['sharpe']:+.4f}")
        print(f"  Total return:  {s['total_return']:+.2%}")
        print(f"  Total costs:   {s['total_cost']:.4%}")
        print(f"  Max drawdown:  {s['max_drawdown']:.2%}")
        print(f"  Mean Σ|Δw|/day: {s['mean_daily_turnover']:.6f} (end-of-day rebalance)")

    print("\n" + "=" * 60)
    print("SUMMARY (single split)")
    for name, s in rows:
        print(f"  {name:12}  Sharpe {s['sharpe']:+.4f}")
    print("=" * 60)


def run_cv(data_dir: str, strategies: dict[str, callable]) -> None:
    prices, meta = _load(data_dir)
    folds = cv_folds(prices)
    print(f"Data: {prices.shape[0]:,} ticks, {len(folds)} CV folds (expanding train, 1y test)")
    print("=" * 60)

    table: dict[str, list[float]] = {n: [] for n in strategies}

    for fi, (k, train_end, test_end) in enumerate(folds):
        print(f"\n--- Fold {fi + 1}: train years 0–{k - 1}, test year {k} ---")
        train_p = prices[:train_end]
        hold_p = prices[train_end:test_end]
        for name, factory in strategies.items():
            strat = factory()
            res = run_backtest(train_p, hold_p, strat, meta)
            s = summarize_result(res)
            table[name].append(float(s["sharpe"]))
            tag = "BLAST" if s["blown_up"] else "ok"
            print(f"  {name:12}  Sharpe {s['sharpe']:+.4f}  ({tag})")

    print("\n" + "=" * 60)
    print("CV SUMMARY (annualized Sharpe, zero rf)")
    for name, sharpes in table.items():
        a = np.array(sharpes, dtype=float)
        print(
            f"  {name:12}  mean {a.mean():+.4f}  std {a.std(ddof=1):.4f}  min {a.min():+.4f}"
        )
    print("=" * 60)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Evaluate portfolio strategies (competition harness)")
    p.add_argument(
        "--data-dir",
        required=True,
        help="Folder containing prices.csv and meta.csv",
    )
    p.add_argument("--cv", action="store_true", help="Time-series CV instead of single 4y/1y split")
    p.add_argument(
        "--strategies",
        nargs="+",
        default=["my", "equal_weight", "vol_ew"],
        help="Strategy keys: my equal_weight vol_ew all",
    )
    args = p.parse_args(argv)

    if not os.path.isdir(args.data_dir):
        sys.exit(f"Not a directory: {args.data_dir}")

    strategies = _resolve_strategies(args.strategies)

    if args.cv:
        run_cv(args.data_dir, strategies)
    else:
        run_single(args.data_dir, strategies)


if __name__ == "__main__":
    main()
