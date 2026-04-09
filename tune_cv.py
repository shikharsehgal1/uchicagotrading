#!/usr/bin/env python3
"""Coarse time-series CV grid search with a robustness-penalized score.

Objective (default): maximize  mean(Sharpe_fold) - λ·std(Sharpe_fold)
with λ=0.5, plus hard filters (no blow-ups; optional floor vs equal-weight min).

This is tuned for *out-of-sample stability*, not single-split peak Sharpe.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from dataclasses import dataclass

import numpy as np

from benchmarks import EqualWeightStrategy
from case2 import MyStrategy, create_strategy_with_params, load_meta, load_prices
from harness import annualized_sharpe, cv_folds, run_backtest, summarize_result


@dataclass(frozen=True)
class EvalRow:
    params: dict
    sharpes: np.ndarray
    mean_turnover: float
    blown_up: bool

    @property
    def mean_sr(self) -> float:
        return float(np.mean(self.sharpes))

    @property
    def std_sr(self) -> float:
        return float(np.std(self.sharpes, ddof=1)) if self.sharpes.size > 1 else 0.0

    @property
    def min_sr(self) -> float:
        return float(np.min(self.sharpes))


def _cv_sharpes(
    prices: np.ndarray,
    meta,
    *,
    factory,
) -> tuple[np.ndarray, float, bool]:
    folds = cv_folds(prices)
    srs: list[float] = []
    turnovers: list[float] = []
    blown = False
    for k, train_end, test_end in folds:
        strat = factory()
        res = run_backtest(prices[:train_end], prices[train_end:test_end], strat, meta)
        s = summarize_result(res)
        srs.append(float(s["sharpe"]))
        turnovers.append(float(s["mean_daily_turnover"]))
        blown = blown or bool(s["blown_up"])
    return np.array(srs, dtype=float), float(np.mean(turnovers)), blown


def baseline_equal_weight_floor(prices: np.ndarray, meta) -> float:
    srs, _, _ = _cv_sharpes(prices, meta, factory=EqualWeightStrategy)
    return float(np.min(srs))


def default_grid() -> list[dict]:
    # ~432 combos: coarse knobs only; avoids exhaustive search / overfit.
    rebal = [3, 5, 8, 10]
    tau = [0.09, 0.11, 0.13]
    vol = [0.10, 0.12, 0.14]
    long_only = [False, True]
    mom_blend = [0.55, 0.65, 0.75]
    windows = [(18, 54), (21, 63)]
    out: list[dict] = []
    for r, t, v, lo, mb, (sf, sl) in itertools.product(
        rebal, tau, vol, long_only, mom_blend, windows
    ):
        out.append(
            {
                "REBAL_EVERY": r,
                "TAU": t,
                "VOL_TARGET": v,
                "LONG_ONLY": lo,
                "MOM_BLEND": mb,
                "SEC_MOM_FAST": sf,
                "SEC_MOM_SLOW": sl,
            }
        )
    return out


def quick_grid() -> list[dict]:
    rebal = [5, 8]
    tau = [0.10, 0.12, 0.14]
    vol = [0.11, 0.12, 0.13]
    long_only = [False, True]
    mom_blend = [0.55, 0.65]
    windows = [(21, 63), (18, 54)]
    out: list[dict] = []
    for r, t, v, lo, mb, (sf, sl) in itertools.product(
        rebal, tau, vol, long_only, mom_blend, windows
    ):
        out.append(
            {
                "REBAL_EVERY": r,
                "TAU": t,
                "VOL_TARGET": v,
                "LONG_ONLY": lo,
                "MOM_BLEND": mb,
                "SEC_MOM_FAST": sf,
                "SEC_MOM_SLOW": sl,
            }
        )
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="CV grid search for MyStrategy")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--quick", action="store_true", help="Smaller grid (~96 configs)")
    ap.add_argument("--lambda-std", type=float, default=0.5, help="Penalty on fold Sharpe std")
    ap.add_argument(
        "--ew-margin",
        type=float,
        default=0.0,
        help="Require min fold Sharpe >= (min EW fold Sharpe) - margin",
    )
    ap.add_argument("--top", type=int, default=12, help="How many configs to print")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.data_dir):
        sys.exit(f"Not a directory: {args.data_dir}")

    p = os.path.join(args.data_dir, "prices.csv")
    m = os.path.join(args.data_dir, "meta.csv")
    prices = load_prices(p)
    meta = load_meta(m)

    print("Computing equal-weight CV floor (min fold Sharpe)...")
    ew_min = baseline_equal_weight_floor(prices, meta)
    print(f"  Equal-weight min fold Sharpe: {ew_min:+.4f}")
    floor = ew_min - float(args.ew_margin)
    print(f"  Hard floor for candidates:    {floor:+.4f}  (margin={args.ew_margin:g})")

    grid = quick_grid() if args.quick else default_grid()
    print(f"\nGrid size: {len(grid)} configs × {len(cv_folds(prices))} folds")

    print("\nEvaluating baseline MyStrategy() (class defaults)...")
    baseline_srs, baseline_mto, baseline_blown = _cv_sharpes(
        prices, meta, factory=MyStrategy
    )
    baseline_row = EvalRow(
        params={"baseline": "MyStrategy defaults"},
        sharpes=baseline_srs,
        mean_turnover=baseline_mto,
        blown_up=baseline_blown,
    )

    rows: list[EvalRow] = []
    for params in grid:

        def _factory(p=params):
            return create_strategy_with_params(**p)

        srs, mto, blown = _cv_sharpes(prices, meta, factory=_factory)
        row = EvalRow(params=params, sharpes=srs, mean_turnover=mto, blown_up=blown)
        rows.append(row)

    lam = float(args.lambda_std)

    def robust_score(r: EvalRow) -> float:
        return r.mean_sr - lam * r.std_sr

    eligible = [
        r
        for r in rows
        if not r.blown_up and r.min_sr >= floor
    ]
    eligible.sort(
        key=lambda r: (robust_score(r), r.min_sr, -r.mean_turnover),
        reverse=True,
    )

    print("\n" + "=" * 70)
    print("TOP CONFIGS (robust score = mean Sharpe - λ·std)")
    print("=" * 70)
    for i, r in enumerate(eligible[: args.top], 1):
        rs = robust_score(r)
        print(
            f"\n#{i}  robust={rs:+.4f}  mean={r.mean_sr:+.4f}  std={r.std_sr:.4f}  min={r.min_sr:+.4f}  "
            f"mean|Δw|={r.mean_turnover:.4f}"
        )
        print(f"     {r.params}")

    if not eligible:
        print("\nNo configs passed filters; relax --ew-margin or check blow-ups.")
        return

    best = eligible[0]
    print("\n" + "=" * 70)
    print("RECOMMENDED (highest robust score passing filters)")
    print("=" * 70)
    print(f"  robust score: {robust_score(best):+.4f}")
    print(f"  fold Sharpes: {best.sharpes}")
    print(f"  params: {best.params}")

    if baseline_row is not None:
        b = baseline_row
        print("\n--- Current MyStrategy defaults (same grid encoding) ---")
        print(
            f"  robust={robust_score(b):+.4f}  mean={b.mean_sr:+.4f}  std={b.std_sr:.4f}  min={b.min_sr:+.4f}"
        )

    if baseline_row is not None and robust_score(best) > robust_score(baseline_row) + 1e-6:
        print(
            "\nSuggested: update class defaults in case2.py to the RECOMMENDED params "
            "(or keep as create_strategy_with_params for experiments)."
        )
    else:
        print("\nBaseline defaults already competitive on robust score; consider manual tweaks.")


if __name__ == "__main__":
    main()
