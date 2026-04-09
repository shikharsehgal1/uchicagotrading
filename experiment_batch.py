#!/usr/bin/env python3
"""Run a fixed ablation / stress batch on time-series CV (competition harness).

Priority encoded here (quant research view for a single hidden OOS year):
  1) Robust CV: mean(fold Sharpe) - 0.5 * std(fold Sharpe)
  2) Tail: min fold Sharpe (avoid one catastrophic regime)
  3) Guardrail: min fold Sharpe > min fold equal-weight Sharpe

Usage:
  python experiment_batch.py --data-dir /path/to/csv_folder
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable

import numpy as np

from benchmarks import EqualWeightStrategy, VolManagedEqualWeightStrategy
from case2 import MyStrategy, load_meta, load_prices
from harness import cv_folds, run_backtest, summarize_result

LAMBDA_STD = 0.5


def cv_metrics(prices: np.ndarray, meta, factory: Callable[[], object]) -> dict:
    folds = cv_folds(prices)
    srs: list[float] = []
    tos: list[float] = []
    blown = False
    for _k, train_end, test_end in folds:
        strat = factory()
        res = run_backtest(prices[:train_end], prices[train_end:test_end], strat, meta)
        s = summarize_result(res)
        srs.append(float(s["sharpe"]))
        tos.append(float(s["mean_daily_turnover"]))
        blown = blown or bool(s["blown_up"])
    a = np.array(srs, dtype=float)
    mean_sr = float(np.mean(a))
    std_sr = float(np.std(a, ddof=1)) if a.size > 1 else 0.0
    min_sr = float(np.min(a))
    robust = mean_sr - LAMBDA_STD * std_sr
    return {
        "sharpes": a,
        "mean": mean_sr,
        "std": std_sr,
        "min": min_sr,
        "robust": robust,
        "mean_turnover": float(np.mean(tos)),
        "blown_up": blown,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="CV ablation batch for MyStrategy")
    ap.add_argument("--data-dir", required=True)
    args = ap.parse_args()
    if not os.path.isdir(args.data_dir):
        sys.exit(f"Not a directory: {args.data_dir}")

    prices = load_prices(os.path.join(args.data_dir, "prices.csv"))
    meta = load_meta(os.path.join(args.data_dir, "meta.csv"))

    # pfo architecture: component ablations live in code modules, not instance flags.
    experiments: list[tuple[str, Callable[[], object]]] = [
        ("equal_weight", lambda: EqualWeightStrategy()),
        ("vol_managed_ew", lambda: VolManagedEqualWeightStrategy()),
        ("pfo_systematic", lambda: MyStrategy()),
    ]

    rows: list[tuple[str, dict]] = []
    for name, factory in experiments:
        m = cv_metrics(prices, meta, factory)
        rows.append((name, m))

    ew_min = next(m["min"] for n, m in rows if n == "equal_weight")

    print(f"Equal-weight CV min fold Sharpe (floor): {ew_min:+.4f}")
    print(f"Metric: robust = mean - {LAMBDA_STD}*std on {len(cv_folds(prices))} folds\n")

    # Sort: no blow-up first, then higher robust, then higher min fold, then lower turnover
    def sort_key(item: tuple[str, dict]) -> tuple:
        _n, m = item
        return (m["blown_up"], -m["robust"], -m["min"], m["mean_turnover"])

    rows_sorted = sorted(rows, key=sort_key)

    hdr = f"{'experiment':<28} {'robust':>8} {'mean':>8} {'std':>7} {'min':>8} {'>EWmin':>7} {'blow':>4} {'m|dw':>8}"
    print(hdr)
    print("-" * len(hdr))
    for name, m in rows_sorted:
        flag = "yes" if m["min"] >= ew_min - 1e-9 else "no"
        bl = "Y" if m["blown_up"] else "."
        print(
            f"{name:<28} {m['robust']:>+8.4f} {m['mean']:>+8.4f} {m['std']:>7.4f} {m['min']:>+8.4f} "
            f"{flag:>7} {bl:>4} {m['mean_turnover']:>8.5f}"
        )
        print(f"     folds: {np.array2string(m['sharpes'], precision=4, floatmode='fixed')}")

    print("\nInterpretation:")
    print("  - Blocks that *hurt* robust when OFF are valuable (restore them).")
    print("  - If OFF improves robust/min, consider dropping or replacing that block.")
    print("  - Prefer configs with min > equal-weight min on all folds when ties.")


if __name__ == "__main__":
    main()
