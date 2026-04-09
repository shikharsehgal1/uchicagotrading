#!/usr/bin/env python3
"""Robustness metrics for the pfo strategy: single split + CV (competition harness)."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from benchmarks import EqualWeightStrategy
from case2 import MyStrategy, load_meta, load_prices
from harness import (
    HOLDOUT_TICKS,
    TRAIN_TICKS,
    cv_folds,
    run_backtest,
    summarize_result,
)

LAMBDA = 0.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    args = ap.parse_args()
    if not os.path.isdir(args.data_dir):
        sys.exit(f"Not a directory: {args.data_dir}")

    p = os.path.join(args.data_dir, "prices.csv")
    m = os.path.join(args.data_dir, "meta.csv")
    prices = load_prices(p)
    meta = load_meta(m)

    def cv_for(factory):
        folds = cv_folds(prices)
        srs = []
        costs = []
        for _k, te, tst in folds:
            r = run_backtest(prices[:te], prices[te:tst], factory(), meta)
            s = summarize_result(r)
            srs.append(float(s["sharpe"]))
            costs.append(float(s["total_cost"]))
        a = np.array(srs, dtype=float)
        return {
            "mean": float(np.mean(a)),
            "std": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
            "min": float(np.min(a)),
            "robust": float(np.mean(a) - LAMBDA * np.std(a, ddof=1)) if len(a) > 1 else float(np.mean(a)),
            "folds": a,
            "mean_cost": float(np.mean(costs)),
        }

    print("=== Cross-validation (expanding window, 1y test) ===")
    for name, fac in [
        ("equal_weight", EqualWeightStrategy),
        ("pfo_systematic", MyStrategy),
    ]:
        o = cv_for(fac)
        print(f"\n[{name}]")
        print(f"  mean Sharpe:  {o['mean']:+.4f}")
        print(f"  std Sharpe:   {o['std']:.4f}")
        print(f"  robust score: {o['robust']:+.4f}  (= mean - 0.5*std)")
        print(f"  min fold:     {o['min']:+.4f}")
        print(f"  folds:        {o['folds']}")
        print(f"  mean txn cost (fold sum): {o['mean_cost']:.4%}")

    train = prices[:TRAIN_TICKS]
    hold = prices[TRAIN_TICKS : TRAIN_TICKS + HOLDOUT_TICKS]
    r = run_backtest(train, hold, MyStrategy(), meta)
    s = summarize_result(r)
    print("\n=== Single split (4y train / 1y holdout) — pfo_systematic ===")
    print(f"  Sharpe:       {s['sharpe']:+.4f}")
    print(f"  total return: {s['total_return']:+.2%}")
    print(f"  total costs:  {s['total_cost']:.4%}")
    print(f"  max DD:       {s['max_drawdown']:.2%}")


if __name__ == "__main__":
    main()
