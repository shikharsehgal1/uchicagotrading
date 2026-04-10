"""
Calibration module: back out gamma, beta_y (and potentially other params)
from observed exchange data using regression.

Data we collect each tick/day:
- timestamp
- C market price (bid, ask, last)
- q_hike, q_hold, q_cut (from prediction market)
- EPS (from earnings releases, updates twice/day)

What is gamma and beta_y?
- gamma: sensitivity of PE to changes in y (the "risk factor" derived from q
- beta_y: sensitivity of y to changes in expected return (E[Δr])

Strategy:
1. Linearized OLS for a quick dirty estimate
2. Nonlinear least squares for the full model
3. Rolling window to detect parameter drift
"""

import numpy as np
from scipy.optimize import least_squares
from collections import deque

# ---- Data Collection ----

class ExchangeDataStore:
    """
    Ring buffer of observations from the exchange.
    Call .record() every tick or every day depending on your frequency.
    """
    def __init__(self, max_obs=500):
        self.data = deque(maxlen=max_obs)
    
    def record(self, timestamp, price_c, q_hike, q_hold, q_cut, eps):
        self.data.append({
            't': timestamp,
            'price': price_c,
            'q_hike': q_hike,
            'q_hold': q_hold,
            'q_cut': q_cut,
            'eps': eps,
        })
    
    def to_arrays(self):
        """Convert to numpy arrays for regression."""
        d = list(self.data)
        return {
            'price': np.array([x['price'] for x in d]),
            'q_hike': np.array([x['q_hike'] for x in d]),
            'q_hold': np.array([x['q_hold'] for x in d]),
            'q_cut': np.array([x['q_cut'] for x in d]),
            'eps': np.array([x['eps'] for x in d]),
        }


# ---- Method 1: Linearized OLS ----

def calibrate_linear(store, y0=0.045, PE0=14.0, B0_per_share=40.0, 
                     D=7.5, C_conv=55.0, lam=0.65):
    """
    Linearize the model around y0 for a quick estimate.
    
    P ≈ EPS * PE0 * (1 - gamma*Δy) + lam * B0/N * (-D*Δy + 0.5*C*Δy²)
    
    Rearrange:
    P - EPS*PE0 - lam*B0/N*(0.5*C*Δy²) = -[EPS*PE0*gamma + lam*B0/N*D] * Δy
    
    But Δy = beta_y * E[Δr]/10000, so we're estimating gamma and beta_y jointly.
    
    Simplification: treat beta_y*E[Δr]/10000 as the single regressor x_t,
    then regress to get composite coefficients.
    
    Actually cleaner: two-step.
    Step 1: Assume beta_y = 1, compute Δy from probs, run regression for gamma.
    Step 2: Use residuals to refine beta_y.
    """
    arr = store.to_arrays()
    n = len(arr['price'])
    if n < 10:
        return None  # not enough data
    
    # Compute E[Δr] for each observation
    e_dr = 25 * arr['q_hike'] + (-25) * arr['q_cut']  # in bps
    
    # For now assume beta_y = 1, Δy = e_dr / 10000
    dy = e_dr / 10000
    dy2 = dy ** 2
    
    # Bond component (known params): lam * B0/N * (-D*Δy + 0.5*C*Δy²)
    bond_component = lam * B0_per_share * (-D * dy + 0.5 * C_conv * dy2)
    
    # Residual after removing bond component:
    # P - lam*ΔB/N ≈ EPS * PE0 * exp(-gamma * Δy)
    # log(P - lam*ΔB/N) - log(EPS) - log(PE0) ≈ -gamma * Δy
    
    ops_residual = arr['price'] - bond_component
    
    # Filter out non-positive values (can't take log)
    valid = ops_residual > 0
    if valid.sum() < 10:
        return None
    
    log_resid = np.log(ops_residual[valid]) - np.log(arr['eps'][valid]) - np.log(PE0)
    dy_valid = dy[valid]
    
    # OLS: log_resid = -gamma * dy + intercept
    X = np.column_stack([dy_valid, np.ones(valid.sum())])
    beta, residuals, _, _ = np.linalg.lstsq(X, log_resid, rcond=None)
    
    gamma_est = -beta[0]
    intercept = beta[1]  # should be ~0 if model is correct
    
    return {
        'gamma': gamma_est,
        'intercept': intercept,  # diagnostic: how far from 0?
        'n_obs': int(valid.sum()),
        'method': 'linearized_ols',
    }


# ---- Method 2: Nonlinear Least Squares (full model) ----

def calibrate_nonlinear(store, y0=0.045, PE0=14.0, B0_per_share=40.0,
                        D=7.5, C_conv=55.0, lam=0.65,
                        gamma_init=0.5, beta_y_init=1.0):
    """
    Full nonlinear calibration via scipy.optimize.least_squares.
    Jointly estimate gamma and beta_y.
    """
    arr = store.to_arrays()
    n = len(arr['price'])
    if n < 20:
        return None
    
    e_dr = 25 * arr['q_hike'] + (-25) * arr['q_cut']
    var_dr = (625 * arr['q_hike'] + 625 * arr['q_cut']) - e_dr**2
    
    def model_price(params):
        gamma, beta_y = params
        
        dy = beta_y * e_dr / 10000
        
        # Ops component
        pe_t = PE0 * np.exp(-gamma * dy)
        ops = arr['eps'] * pe_t
        
        # Bond component -- use expected value including variance
        e_dy = beta_y * e_dr / 10000
        e_dy2 = (beta_y / 10000)**2 * (var_dr + e_dr**2)
        bond = lam * B0_per_share * (-D * e_dy + 0.5 * C_conv * e_dy2)
        
        return ops + bond
    
    def residuals(params):
        return arr['price'] - model_price(params)
    
    result = least_squares(
        residuals,
        x0=[gamma_init, beta_y_init],
        bounds=([0.01, 0.01], [10.0, 5.0]),  # reasonable bounds
        method='trf',
    )
    
    # Estimate standard errors from Jacobian
    J = result.jac
    residual_var = np.mean(result.fun**2)
    try:
        cov = residual_var * np.linalg.inv(J.T @ J)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.array([np.nan, np.nan])
    
    return {
        'gamma': result.x[0],
        'beta_y': result.x[1],
        'gamma_se': se[0],
        'beta_y_se': se[1],
        'rmse': np.sqrt(np.mean(result.fun**2)),
        'n_obs': n,
        'converged': result.success,
        'method': 'nonlinear_ls',
    }


# ---- Method 3: Rolling Calibration ----

def rolling_calibrate(store, window=50, **kwargs):
    """
    Run nonlinear calibration on a rolling window to detect param drift.
    Returns list of timestamped parameter estimates.
    """
    data = list(store.data)
    results = []
    
    for i in range(window, len(data)):
        window_store = ExchangeDataStore(max_obs=window)
        for obs in data[i - window:i]:
            window_store.data.append(obs)
        
        cal = calibrate_nonlinear(window_store, **kwargs)
        if cal and cal['converged']:
            results.append({
                't': data[i]['t'],
                **cal,
            })
    
    return results


# ---- Diagnostic: Implied vs Market Price ----

def compute_mispricing(store, gamma, beta_y, y0=0.045, PE0=14.0,
                       B0_per_share=40.0, D=7.5, C_conv=55.0, lam=0.65):
    """
    Given calibrated params, compute fair value vs market price for each obs.
    The residual IS your trade signal.
    """
    arr = store.to_arrays()
    e_dr = 25 * arr['q_hike'] + (-25) * arr['q_cut']
    var_dr = (625 * arr['q_hike'] + 625 * arr['q_cut']) - e_dr**2
    
    dy = beta_y * e_dr / 10000
    
    pe_t = PE0 * np.exp(-gamma * dy)
    ops = arr['eps'] * pe_t
    
    e_dy = beta_y * e_dr / 10000
    e_dy2 = (beta_y / 10000)**2 * (var_dr + e_dr**2)
    bond = lam * B0_per_share * (-D * e_dy + 0.5 * C_conv * e_dy2)
    
    fair_value = ops + bond
    mispricing = arr['price'] - fair_value
    
    return {
        'fair_value': fair_value,
        'market_price': arr['price'],
        'mispricing': mispricing,
        'mispricing_pct': mispricing / fair_value * 100,
    }


# ---- Example Usage ----
if __name__ == "__main__":
    store = ExchangeDataStore()
    
    # Simulate some fake data for testing
    np.random.seed(42)
    TRUE_GAMMA = 0.8
    TRUE_BETA_Y = 1.2
    
    for t in range(200):
        q_h = np.clip(0.3 + 0.1 * np.sin(t / 20) + np.random.normal(0, 0.05), 0.05, 0.9)
        q_c = np.clip(0.2 + 0.05 * np.cos(t / 15) + np.random.normal(0, 0.03), 0.05, 0.9)
        q_hold = 1 - q_h - q_c
        if q_hold < 0.05:
            scale = (q_h + q_c) / 0.95
            q_h /= scale
            q_c /= scale
            q_hold = 0.05
        
        eps = 2.0 + np.random.normal(0, 0.02)
        
        # True price from true params + noise
        e_dr = 25 * q_h - 25 * q_c
        dy = TRUE_BETA_Y * e_dr / 10000
        pe = 14.0 * np.exp(-TRUE_GAMMA * dy)
        ops = eps * pe
        var_dr = 625 * q_h + 625 * q_c - e_dr**2
        e_dy2 = (TRUE_BETA_Y / 10000)**2 * (var_dr + e_dr**2)
        bond = 0.65 * 40.0 * (-7.5 * dy + 0.5 * 55.0 * e_dy2)
        price = ops + bond + np.random.normal(0, 0.15)  # noise term
        
        store.record(t, price, q_h, q_hold, q_c, eps)
    
    # Calibrate
    print("=== Linear Calibration ===")
    lin = calibrate_linear(store)
    assert lin is not None, "linear calibration returned None"
    print(f"gamma = {lin['gamma']:.4f}  (true={TRUE_GAMMA})")

    print("\n=== Nonlinear Calibration ===")
    nlin = calibrate_nonlinear(store)
    assert nlin is not None, "nonlinear calibration returned None"
    print(f"gamma  = {nlin['gamma']:.4f} ± {nlin['gamma_se']:.4f}  (true={TRUE_GAMMA})")
    print(f"beta_y = {nlin['beta_y']:.4f} ± {nlin['beta_y_se']:.4f}  (true={TRUE_BETA_Y})")
    print(f"RMSE   = {nlin['rmse']:.4f}")

    print("\n=== Mispricing Signal ===")
    mp = compute_mispricing(store, nlin['gamma'], nlin['beta_y'])
    print(f"Mean mispricing: {np.mean(mp['mispricing']):.4f}")
    print(f"Std mispricing:  {np.std(mp['mispricing']):.4f}")
    print(f"Max mispricing:  {np.max(np.abs(mp['mispricing'])):.4f}")