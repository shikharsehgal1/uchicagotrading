#!/usr/bin/env python3
"""Sequential disciplined checks vs A_baseline_tuned (no broad search).

See user spec: tau, cost-gate threshold, optional momentum smoothing only.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from case2 import MyStrategy, create_strategy_with_params, load_meta, load_prices
from harness import cv_folds, run_backtest, summarize_result

LAMBDA = 0.5
EPS = 1e-9


def cv_stats(prices: np.ndarray, meta, factory) -> dict:
    folds = cv_folds(prices)
    srs: list[float] = []
    for _k, train_end, test_end in folds:
        strat = factory()
        res = run_backtest(prices[:train_end], prices[train_end:test_end], strat, meta)
        s = summarize_result(res)
        srs.append(float(s["sharpe"]))
    a = np.array(srs, dtype=float)
    mean_sr = float(np.mean(a))
    std_sr = float(np.std(a, ddof=1)) if a.size > 1 else 0.0
    min_sr = float(np.min(a))
    robust = mean_sr - LAMBDA * std_sr
    return {
        "mean": mean_sr,
        "std": std_sr,
        "min": min_sr,
        "robust": robust,
        "folds": a,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    args = ap.parse_args()
    if not os.path.isdir(args.data_dir):
        sys.exit(f"Not a directory: {args.data_dir}")

    prices = load_prices(os.path.join(args.data_dir, "prices.csv"))
    meta = load_meta(os.path.join(args.data_dir, "meta.csv"))

    print("Control: A_baseline_tuned (MyStrategy class defaults)")
    print("Objective: robust = mean(fold Sharpe) - 0.5*std; improve min fold.")
    print("Reject variant if min < baseline_min OR robust < baseline_robust.\n")

    baseline = cv_stats(prices, meta, lambda: MyStrategy())
    b_rob, b_min = baseline["robust"], baseline["min"]
    print(
        f"[CONTROL] mean={baseline['mean']:+.6f}  std={baseline['std']:.6f}  "
        f"robust={b_rob:+.6f}  min={b_min:+.6f}"
    )
    print(f"           folds={np.array2string(baseline['folds'], precision=4)}\n")

    rows: list[tuple[str, dict]] = [("A_baseline_tuned (control)", baseline)]

    experiments: list[tuple[str, dict]] = []

    # 1) Tau (one at a time in log order)
    for tau in (0.11, 0.13, 0.15):
        label = f"1_tau_{tau:.2f}"
        if tau == 0.13:
            label += "_same_as_control"
        experiments.append((label, {"TAU": tau}))

    # 2) Cost gate threshold ±10%, ±20%
    base_thr = float(MyStrategy.COST_GATE_THRESHOLD)
    for pct, label in (
        (-0.20, "2_gate_m20pct"),
        (-0.10, "2_gate_m10pct"),
        (0.10, "2_gate_p10pct"),
        (0.20, "2_gate_p20pct"),
    ):
        thr = base_thr * (1.0 + pct)
        experiments.append((label, {"COST_GATE_THRESHOLD": thr}))

    # 3) Optional smoothing
    experiments.append(("3_mom_smooth_3d", {"MOM_SMOOTH_DAYS": 3}))

    for label, params in experiments:
        st = cv_stats(prices, meta, lambda p=params: create_strategy_with_params(**p))
        rows.append((label, st))
        ok = st["min"] + EPS >= b_min and st["robust"] + EPS >= b_rob
        status = "ACCEPT (vs constraints)" if ok else "REJECT"
        print(f"[{label}] {status}")
        print(
            f"  mean={st['mean']:+.6f}  std={st['std']:.6f}  "
            f"robust={st['robust']:+.6f}  min={st['min']:+.6f}"
        )
        print(f"  folds={np.array2string(st['folds'], precision=4)}")
        if params:
            print(f"  params={params}")
        print()

    # Table
    print("=" * 88)
    print(f"{'experiment':<32} {'mean':>10} {'std':>10} {'robust':>10} {'min_fold':>10} {'pass':>5}")
    print("-" * 88)
    for name, st in rows:
        ok = st["min"] + EPS >= b_min and st["robust"] + EPS >= b_rob
        ps = "yes" if ok else "no"
        print(
            f"{name:<32} {st['mean']:>+10.4f} {st['std']:>10.4f} "
            f"{st['robust']:>+10.4f} {st['min']:>+10.4f} {ps:>5}"
        )

    # Strict beaters: strictly better robust AND strictly better min
    beaters = [
        (n, s)
        for n, s in rows
        if n != "A_baseline_tuned (control)"
        and s["robust"] > b_rob + EPS
        and s["min"] > b_min + EPS
    ]

    print("=" * 88)
    print("STRICT BEATERS (robust > control AND min_fold > control)")
    if not beaters:
        print("  None. Baseline remains optimal under this definition.")
    else:
        for n, s in beaters:
            print(
                f"  {n}: robust {s['robust']:+.4f} (Δ {s['robust'] - b_rob:+.4f}), "
                f"min {s['min']:+.4f} (Δ {s['min'] - b_min:+.4f})"
            )


if __name__ == "__main__":
    main()
