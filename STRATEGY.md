# Case 2 — Systematic portfolio architecture (new branch)

## 1. Why naive 1/N is weak

- **No risk model:** ignores covariance; volatile names and diversifiers get the same weight, so portfolio volatility is often suboptimal for a given gross exposure.
- **No expected-return structure:** does not use sector clustering or cross-sectional information the case provides.
- **Blind to costs:** equal turnover across names with different spreads and impact; shorts ignore borrow drag at tick frequency.

## 2. Why sector-only momentum is insufficient

- **No within-sector ranking:** winners and losers inside the same industry are treated identically.
- **Single horizon / single signal:** misses mean-reversion opportunities when dispersion is high and trends are weak.
- **Still heuristic on risk:** without an explicit Σ and optimizer, sizing is not tied to marginal risk contribution or constraints.

## 3. Architecture (modular)

| Module | Role |
|--------|------|
| `pfo/data_pipeline.py` | Daily closes, log returns, intraday realized vol, sector and residual returns. |
| `pfo/signals.py` | (A) multi-horizon sector momentum; (B) residual momentum; (C) gated residual mean reversion; (D) spread/borrow/hurdle adjustment. |
| `pfo/regime.py` | Smooth weights between momentum vs mean-reversion and a high-vol de-risk scaler (no hard switches). |
| `pfo/risk.py` | Rolling Ledoit–Wolf covariance on daily returns. |
| `pfo/optimizer.py` | Long-only, risk-aware sizing: risk-parity anchor plus positive-μ tilt vs diagonal vol (fast, stable; avoids daily heavy QP under tick costs). |
| `pfo/execution.py` | No-trade band + partial blend toward target; marginal txn cost check (linear + quadratic in Δw). |
| `pfo/strategy.py` | Wires the daily loop; `case2.py` exposes `create_strategy()` for the grader. |

## 4. Mathematical core

**Signals → expected return proxy**

- Sector momentum uses overlapping horizons (5 / 20 / 60 days) on **sector equal-weight returns**, z-scored across sectors, then mapped to each asset.
- Residuals \(r^{\perp}_{i,t} = r_{i,t} - r_{\text{sector}(i),t}\) isolate **within-sector** alpha; momentum and short-horizon reversal are built on \(r^{\perp}\).
- Regime module outputs convex weights \((\omega_{\text{mom}}, \omega_{\text{rev}})\) and a volatility scaler \(\psi\) blending **trend strength**, **dispersion / vol ratio**, and **high-vol** pressure.

**Optimization (explicit risk–return, long-only)**

Each day we form **long-only** weights with a **closed-form style** map (production-style when full QP is too costly at tick frequency):

- Start from a **risk-parity anchor** \(1/\sigma_i\).
- Add a **positive-μ tilt** \(\max(\hat\mu_i,0)/(\gamma\sigma_i^2)\).
- **Cap** names, **renormalize** to \(\sum w_i = 1\) (gross \(\le 1\)).
- **Blend** with \(1/N\) before temporal smoothing to stabilize short-training folds.

Borrow and bid–ask impact enter via **signal gating** and **execution**, not inside this smooth core (two-stage construction vs implementation).

**Execution**

- Target move \(\Delta = w^\* - w^{\text{prev}}\); if \(\|\Delta\|_\infty\) is below a spread-based band, **do not trade**.
- Otherwise \(w^{\text{new}} = w^{\text{prev}} + \lambda (w^\* - w^{\text{prev}})\) with \(\lambda \in (0,1)\), then re-scale to respect gross \(\le 1\).

## 5. Robustness objective

Primary metric used in research (not raw Sharpe):

`robust_score = mean(CV fold Sharpe) - 0.5 * std(CV fold Sharpe)`

Secondary: **minimum** fold Sharpe (tail). Reject strategies that win on one fold only.

## 6. Submission layout

- Implementations: `pfo/*.py`, `case2.py` (API), optional `submission.py` (re-export).
- Competition environment: Python 3.12 with numpy, pandas, scikit-learn (and scipy optional in `requirements.txt` for ecosystem parity).

## 7. Practice-data note (honest)

On the released training CSV with expanding-window CV, **equal-weight** can still edge this build on **robust_score** (mean − ½·std) while **pfo_systematic** remains competitive on **mean** Sharpe and the **4y/1y** split. Treat practice metrics as **sanity**, not the hidden test; the architecture is built for **regime/cost-aware** structure, not in-sample peak Sharpe.
