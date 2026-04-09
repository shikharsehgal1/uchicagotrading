from __future__ import annotations

N_ASSETS = 25
TICKS_PER_DAY = 30
ASSET_COLUMNS = tuple(f"A{i:02d}" for i in range(N_ASSETS))
TRADING_DAYS = 252

# Risk / optimization (fixed policy constants — not a tuned grid)
COV_WINDOW = 78
MIN_HISTORY_DAYS = 80
GAMMA_MV = 22.0
MAX_ABS_WEIGHT = 0.12
IMPACT_MULT = 2.5  # aligns with competition evaluator for marginal cost thinking

# Execution (conservative to match tick-level cost environment)
EXEC_BLEND = 0.09
NO_TRADE_FRAC_OF_HALFSPREAD = 2.8
TARGET_EMA = 0.93  # smooth optimizer output before execution

# Signal horizons (trading days)
H_SHORT = 5
H_MED = 20
H_LONG = 60
H_RES_REV = 2
H_RES_VOL = 20
