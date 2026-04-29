# uchicagotrading

## Project
- Repository for a portfolio optimization competition project.
- Universe: 25 assets (`A00` to `A24`) grouped into 5 sectors.
- Decision frequency: one weight vector per day.
- Data frequency: 30 ticks per day.
- Core portfolio constraint: `sum(abs(weights)) <= 1`.

## Data Given
- `prices.csv`
  - Tick-level prices for all 25 assets.
  - Used to build daily closes, daily returns, and intraday realized volatility.
- `meta.csv`
  - `sector_id`: sector membership for each asset.
  - `spread_bps`: transaction-cost proxy per asset.
  - `borrow_bps_annual`: annualized borrow cost for shorts.

## Objective
- Maximize annualized Sharpe ratio of daily returns under the provided evaluation mechanics.
- Evaluation uses full intraday wealth path while holding daily weights.
- Costs included in scoring:
  - linear trading cost: `0.5 * spread * abs(delta_weight)`
  - quadratic impact: `2.5 * spread * (delta_weight^2)`
  - borrow cost on short exposure each tick
- Strategy is validated with:
  - single split: 4y train / 1y holdout
  - expanding-window CV folds
  - robustness metric: `mean(fold_sharpe) - 0.5 * std(fold_sharpe)`

## Strategy
- Main strategy implementation file: `case2.py`.
- Baseline logic:
  - sector momentum view (fast/slow blend)
  - inverse intraday-vol weighting within sector
  - covariance estimate (Ledoit-Wolf with fallback)
  - soft tilt from equal-weight prior
  - volatility targeting
  - execution cost gate (skip low-value rebalances)
- Research toggles exist for ablation/testing (momentum, inv-vol, vol-target, cost gate, long-only).

## Evaluation Utilities
- `harness.py`
  - Local evaluator aligned with the provided scoring mechanics.
  - Computes daily returns, costs, drawdown, and blow-up handling.
- `run_evaluation.py`
  - Runs single split and CV comparison across strategies.
- `tune_cv.py`
  - Coarse CV parameter search using robust-score ranking.
- `experiment_batch.py`
  - Fixed ablation/stress batch for component-level checks.

## Typical Workflow
- Validate baseline:
  - `python run_evaluation.py --data-dir "/path/to/data" --cv --strategies my equal_weight`
- Inspect single split:
  - `python run_evaluation.py --data-dir "/path/to/data" --strategies my`
- Run coarse robust search:
  - `python tune_cv.py --data-dir "/path/to/data" --quick`