"""
The goal is to build a fair value model for Stock C.
We will use PCA to determine the movement of the stock given the rate change of 
the FED prediction market increasing on decreasing the interest rate.

1. Pull current q_hike, q_hold, q_cut from the prediction market
2. Compute E[Δr] and Var[Δr] = E[Δr²] - (E[Δr])² using the three-point distribution
3. Compare the implied variance to what C's price is embedding 
(back out what variance C is pricing from the convexity term)
4. If there's a gap, go long/short C and hedge out the level exposure using the prediction market

the PE ratio term (e^{-γ(y-y₀)}) also has convexity in it (exponential is convex), 
so C actually has convexity exposure from two sources -- the bond portfolio and the 
PE multiple. That amplifies your wings trade but also means your hedge ratios need 
to account for both channels.
"""

import numpy as np

# ---- Hyperparameters ----
y0 = 0.045
PE0 = 14.0
EPS0 = 2.00
D = 7.5
C_conv = 55.0  # convexity constant (C in the formulas, renamed to avoid clash)
B0_per_share = 40.0  # B0/N
lam = 0.65
beta_y = 1.0  # need to calibrate -- maps E[Δr] in bps to yield change
gamma = 0.5   # need to calibrate -- PE sensitivity to yields

# ---- Core Functions ----

def expected_rate_change(q_hike, q_hold, q_cut):
    """E[Δr] in bps"""
    return 25 * q_hike + 0 * q_hold + (-25) * q_cut

def variance_rate_change(q_hike, q_hold, q_cut):
    """Var[Δr] in bps^2"""
    e_dr = expected_rate_change(q_hike, q_hold, q_cut)
    e_dr2 = (25**2) * q_hike + 0 * q_hold + (25**2) * q_cut
    return e_dr2 - e_dr**2

def yield_from_probs(q_hike, q_hold, q_cut):
    """Current implied yield: y_t = y0 + beta_y * E[Δr] (convert bps to decimal)"""
    e_dr = expected_rate_change(q_hike, q_hold, q_cut)
    return y0 + beta_y * (e_dr / 10000)

def fair_value_C(q_hike, q_hold, q_cut, eps=EPS0):
    """
    P_t = EPS_t * PE_t + lambda * (ΔB_t / N) + noise
    
    We compute the deterministic fair value (no noise).
    For the bond component, we use the EXPECTED ΔB which includes
    both the duration (level) and convexity (variance) terms.
    """
    y_t = yield_from_probs(q_hike, q_hold, q_cut)
    dy = y_t - y0
    
    # Operations component: EPS * PE(y)
    pe_t = PE0 * np.exp(-gamma * (y_t - y0))
    ops = eps * pe_t
    
    # Bond portfolio component: E[ΔB/N]
    # For expected value, we need E[Δy] and E[(Δy)^2]
    # E[(Δy)^2] = Var[Δy] + (E[Δy])^2
    e_dr = expected_rate_change(q_hike, q_hold, q_cut)
    var_dr = variance_rate_change(q_hike, q_hold, q_cut)
    
    # Convert bps to decimal for yield
    e_dy = beta_y * (e_dr / 10000)
    e_dy2 = (beta_y / 10000)**2 * (var_dr + e_dr**2)
    
    delta_B_per_share = B0_per_share * (-D * e_dy + 0.5 * C_conv * e_dy2)
    
    bonds = lam * delta_B_per_share
    
    return ops + bonds, ops, bonds

def dP_dy(q_hike, q_hold, q_cut, eps=EPS0):
    """
    Delta: dP/dy -- sensitivity of fair value to parallel yield shift.
    Two channels: PE multiple and bond portfolio.
    """
    y_t = yield_from_probs(q_hike, q_hold, q_cut)
    dy = y_t - y0
    
    # PE channel: d/dy [EPS * PE0 * exp(-gamma*(y-y0))] = -gamma * EPS * PE_t
    pe_t = PE0 * np.exp(-gamma * dy)
    d_ops = -gamma * eps * pe_t
    
    # Bond channel: d/dy [lam * B0/N * (-D*dy + 0.5*C*dy^2)] = lam * B0/N * (-D + C*dy)
    d_bonds = lam * B0_per_share * (-D + C_conv * dy)
    
    return d_ops + d_bonds, d_ops, d_bonds

def d2P_dy2(q_hike, q_hold, q_cut, eps=EPS0):
    """
    Gamma: d²P/dy² -- convexity exposure (this is what we're trading).
    """
    y_t = yield_from_probs(q_hike, q_hold, q_cut)
    dy = y_t - y0
    
    # PE channel: gamma^2 * EPS * PE_t (convex -- positive)
    pe_t = PE0 * np.exp(-gamma * dy)
    g_ops = gamma**2 * eps * pe_t
    
    # Bond channel: lam * B0/N * C
    g_bonds = lam * B0_per_share * C_conv
    
    return g_ops + g_bonds, g_ops, g_bonds


# ---- Live Bot ----
#
# Connects to the UTC Xchange (same pattern as order-sniper/base_sniper.py),
# reads R_HIKE / R_HOLD / R_CUT mids as Fed probabilities, reads C's mid,
# recalibrates gamma / beta_y from hyper_parameter_regress, then every 90s
# (== one trading day) rebalances C toward a target position derived from
# the mispricing signal. Prints WIN / LOSS on every closing fill.

if __name__ == "__main__":
    import os
    import sys
    import time
    import asyncio
    import logging
    from pathlib import Path

    from dotenv import load_dotenv
    from utcxchangelib import XChangeClient
    import utcxchangelib.service_pb2 as utc_pb2  # type: ignore[attr-defined]

    HERE = Path(__file__).resolve().parent
    sys.path.insert(0, str(HERE))
    from hyper_parameter_regress import ExchangeDataStore, calibrate_nonlinear

    load_dotenv(HERE.parent / "order-sniper" / ".env")
    logging.getLogger("xchange-client").setLevel(logging.WARNING)

    SYMBOL = "C"
    PROB_SYMS = ("R_HIKE", "R_HOLD", "R_CUT")
    REBALANCE_SEC = 80.0            # one trading day
    MAX_ORDER_SIZE = 40
    MAX_TARGET_POS = 150
    SIZE_PER_DOLLAR = 20.0          # shares per $1 of mispricing
    CALIB_MIN_OBS = 25
    MAX_HEDGE_POS = 120              # cap on R_HIKE / R_CUT hedge size

    def _mid(book) -> float | None:
        if not book:
            return None
        bid = max(book.bids.keys(), default=None)
        ask = min(book.asks.keys(), default=None)
        if bid is not None and ask is not None:
            return 0.5 * (bid + ask)
        # one-sided book: fall back to whichever side exists
        if bid is not None:
            return float(bid)
        if ask is not None:
            return float(ask)
        return None

    class StockCBot(XChangeClient):
        def __init__(self, host, user, password):
            super().__init__(host, user, password, silent=True)
            self.store = ExchangeDataStore(max_obs=1000)
            self.eps = EPS0
            self.gamma_live = gamma
            self.beta_y_live = beta_y
            # avg-cost W/L tracker
            self.avg_cost = 0.0
            self.net_pos = 0
            self.wins = 0
            self.losses = 0
            self.total_pnl = 0.0
            self.day = 0
            # anchor: treat fv as price_base + (model_now - model_base) so it lives
            # on the same scale as C, regardless of the absolute constants.
            self._model_base: float | None = None
            self._price_base: float | None = None
            self._market_open = True

        def _read_probs(self):
            mids = []
            for s in PROB_SYMS:
                m = _mid(self.order_books.get(s))
                if m is None or m <= 0:
                    return None
                mids.append(m)
            tot = sum(mids)
            return tuple(m / tot for m in mids) if tot > 0 else None

        def _fair_value(self):
            probs = self._read_probs()
            if probs is None:
                return None
            q_h, q_hold_, q_c = probs
            # swap in calibrated params around the call
            global gamma, beta_y
            g0, b0 = gamma, beta_y
            gamma, beta_y = self.gamma_live, self.beta_y_live
            try:
                fv, ops, bonds = fair_value_C(q_h, q_hold_, q_c, eps=self.eps)
                dpdy_tot, dpdy_ops, dpdy_bonds = dP_dy(q_h, q_hold_, q_c, eps=self.eps)
                gpdy_tot, _, _ = d2P_dy2(q_h, q_hold_, q_c, eps=self.eps)
                e_dr = expected_rate_change(q_h, q_hold_, q_c)
                v_dr = variance_rate_change(q_h, q_hold_, q_c)
            finally:
                gamma, beta_y = g0, b0
            return {
                "probs": probs,
                "fv_model": fv,
                "ops": ops,
                "bonds": bonds,
                "dpdy": dpdy_tot,
                "dpdy_ops": dpdy_ops,
                "dpdy_bonds": dpdy_bonds,
                "gpdy": gpdy_tot,
                "e_dr": e_dr,
                "var_dr": v_dr,
            }

        def _calibrate(self):
            if len(self.store.data) < CALIB_MIN_OBS:
                return
            r = calibrate_nonlinear(
                self.store,
                gamma_init=self.gamma_live,
                beta_y_init=self.beta_y_live,
            )
            if r and r.get("converged"):
                self.gamma_live = 0.5 * self.gamma_live + 0.5 * float(r["gamma"])
                self.beta_y_live = 0.5 * self.beta_y_live + 0.5 * float(r["beta_y"])
                print(
                    f"[CALIB] gamma={self.gamma_live:.4f}  "
                    f"beta_y={self.beta_y_live:.4f}  rmse={r['rmse']:.3f}  n={r['n_obs']}"
                )

        async def _cancel_all(self):
            for oid in list(self.open_orders.keys()):
                try:
                    await self.cancel_order(oid)
                except Exception as e:
                    print(f"[CANCEL ERR] {oid}: {e}")

        async def _rebalance(self):
            self.day += 1
            if not self._market_open:
                print(f"[DAY {self.day}] skip — market not open (retrying next tick)")
                self._market_open = True
                return
            info = self._fair_value()
            c_mid = _mid(self.order_books.get(SYMBOL))
            if info is None or c_mid is None:
                print(f"[DAY {self.day}] skip — book not ready")
                return

            probs = info["probs"]
            q_h, q_hold_, q_c = probs
            model_fv = info["fv_model"]

            # Anchor model output to first observed price so fv lives on C's scale.
            if self._model_base is None:
                self._model_base = model_fv
                self._price_base = c_mid
                print(
                    f"[ANCHOR] model_base={model_fv:.3f}  price_base={c_mid:.2f}  "
                    "(fv now reads as price_base + Δmodel)"
                )
            fv = self._price_base + (model_fv - self._model_base)
            mis = fv - c_mid  # +ve → C cheap → want long

            self.store.record(time.time(), c_mid, q_h, q_hold_, q_c, self.eps)
            self._calibrate()

            target_c = int(round(SIZE_PER_DOLLAR * mis))
            target_c = max(-MAX_TARGET_POS, min(MAX_TARGET_POS, target_c))
            pos_c = int(self.positions.get(SYMBOL, 0))
            delta_c = target_c - pos_c

            # ---- Logic print: what the model thinks and why ----
            print(
                f"\n[DAY {self.day}] ========================================\n"
                f"  probs     : HIKE={q_h:.3f}  HOLD={q_hold_:.3f}  CUT={q_c:.3f}\n"
                f"  E[Δr]     : {info['e_dr']:+.2f} bps   Var[Δr]: {info['var_dr']:.2f} bps²\n"
                f"  EPS       : {self.eps:.3f}   γ={self.gamma_live:.3f}  β_y={self.beta_y_live:.3f}\n"
                f"  model_fv  : {model_fv:.3f}  (ops={info['ops']:.3f}  bonds={info['bonds']:+.3f})\n"
                f"  anchored  : fv = price_base + Δmodel = {fv:.2f}\n"
                f"  C mid     : {c_mid:.2f}   mispricing = fv - C = {mis:+.2f}\n"
                f"  dP/dy     : {info['dpdy']:+.2f}  (ops={info['dpdy_ops']:+.2f}  bonds={info['dpdy_bonds']:+.2f})\n"
                f"  d²P/dy²   : {info['gpdy']:+.2f}   ← convexity (what we're actually trading)\n"
                f"  C target  : pos {pos_c:+d} → {target_c:+d}   (size = {SIZE_PER_DOLLAR:.0f}·mis)\n"
                f"  score     : W{self.wins}/L{self.losses}  cum pnl={self.total_pnl:+.2f}"
            )

            await self._cancel_all()

            # ---- 1) trade C toward target ----
            await self._place_toward(SYMBOL, delta_c)

            # ---- 2) delta-hedge using the prediction market ----
            # Total $/yield sensitivity at the new target (use target, not current pos,
            # because we're committing to it this tick).
            # dP/dy is in $/(unit yield). Convert 25bps move = 0.0025.
            hike_pnl_c = target_c * info["dpdy"] * 0.0025   # C PnL if HIKE realizes
            cut_pnl_c  = target_c * info["dpdy"] * (-0.0025)  # C PnL if CUT realizes

            # Solve 2x2 for hedge positions (N_h, N_c) in R_HIKE/R_CUT:
            #   N_h*(1-q_h) - N_c*q_c = -hike_pnl_c
            #  -N_h*q_h     + N_c*(1-q_c) = -cut_pnl_c
            # det = 1 - q_h - q_c = q_hold
            if q_hold_ > 1e-6:
                det = q_hold_
                N_h_f = (-(1 - q_c) * hike_pnl_c - q_c * cut_pnl_c) / det
                N_c_f = (-(1 - q_h) * cut_pnl_c - q_h * hike_pnl_c) / det
            else:
                N_h_f = N_c_f = 0.0

            target_hike = max(-MAX_HEDGE_POS, min(MAX_HEDGE_POS, int(round(N_h_f))))
            target_cut  = max(-MAX_HEDGE_POS, min(MAX_HEDGE_POS, int(round(N_c_f))))

            print(
                f"[HEDGE] C pnl if HIKE={hike_pnl_c:+.2f}  if CUT={cut_pnl_c:+.2f}\n"
                f"        solve → want HIKE={target_hike:+d}  CUT={target_cut:+d}  "
                f"(raw {N_h_f:+.1f} / {N_c_f:+.1f})"
            )

            pos_hike = int(self.positions.get("R_HIKE", 0))
            pos_cut  = int(self.positions.get("R_CUT", 0))
            await self._place_toward("R_HIKE", target_hike - pos_hike)
            await self._place_toward("R_CUT",  target_cut  - pos_cut)

        async def _place_toward(self, symbol: str, delta: int) -> None:
            if abs(delta) < 1:
                return
            book = self.order_books.get(symbol)
            if not book:
                print(f"[SKIP {symbol}] no book")
                return
            best_bid = max(book.bids.keys(), default=None)
            best_ask = min(book.asks.keys(), default=None)
            if delta > 0:
                px = best_ask if best_ask is not None else best_bid
                if px is None:
                    print(f"[SKIP {symbol}] no ask")
                    return
                qty = min(delta, MAX_ORDER_SIZE)
                await self.place_order(symbol, qty, "buy", int(px))
                print(f"[ORDER] {symbol} BUY  {qty}@{int(px)}")
            else:
                px = best_bid if best_bid is not None else best_ask
                if px is None:
                    print(f"[SKIP {symbol}] no bid")
                    return
                qty = min(-delta, MAX_ORDER_SIZE)
                await self.place_order(symbol, qty, "sell", int(px))
                print(f"[ORDER] {symbol} SELL {qty}@{int(px)}")

        async def _rebalance_loop(self):
            await asyncio.sleep(5.0)  # warm-up
            while True:
                try:
                    await self._rebalance()
                except Exception as e:
                    print(f"[REBAL ERR] {e}")
                await asyncio.sleep(REBALANCE_SEC)

        # ---------- exchange hooks ----------
        async def bot_handle_book_update(self, symbol: str) -> None:
            return

        async def bot_handle_order_fill(self, order_id, qty, price):
            info = self.open_orders.get(order_id)
            if info is None:
                return
            is_buy = info[0].side == utc_pb2.NewOrderRequest.Side.BUY  # type: ignore[attr-defined]
            side = "buy" if is_buy else "sell"
            signed = qty if is_buy else -qty
            # avg-cost W/L: if this trade reduces |pos|, the reducing part is a round-trip close
            if self.net_pos != 0 and ((self.net_pos > 0) != (signed > 0)):
                close_qty = min(abs(signed), abs(self.net_pos))
                pnl = (price - self.avg_cost) * close_qty * (1 if self.net_pos > 0 else -1)
                self.total_pnl += pnl
                tag = "WIN " if pnl > 0 else "LOSS"
                if pnl > 0:
                    self.wins += 1
                else:
                    self.losses += 1
                print(
                    f"[{tag}] {side.upper()} {close_qty}@{price}  "
                    f"avg_cost={self.avg_cost:.2f}  pnl={pnl:+.2f}  "
                    f"cum={self.total_pnl:+.2f}"
                )
                # apply the closing portion
                new_pos = self.net_pos + (close_qty if signed > 0 else -close_qty)
                remainder = abs(signed) - close_qty
                self.net_pos = new_pos
                if self.net_pos == 0:
                    self.avg_cost = 0.0
                # any leftover after the flip opens a new position
                if remainder > 0:
                    self.avg_cost = float(price)
                    self.net_pos += remainder if signed > 0 else -remainder
            else:
                # purely adding to / opening position
                new_pos = self.net_pos + signed
                if new_pos != 0:
                    self.avg_cost = (
                        self.avg_cost * abs(self.net_pos) + price * abs(signed)
                    ) / abs(new_pos)
                self.net_pos = new_pos
                print(f"[OPEN] {side.upper()} {qty}@{price}  pos={self.net_pos:+d}  avg={self.avg_cost:.2f}")

        async def bot_handle_order_rejected(self, order_id, reason):
            print(f"[REJECT] {order_id}: {reason}")
            if isinstance(reason, str) and "not opened" in reason.lower():
                self._market_open = False

        async def bot_handle_news(self, news_release: dict) -> None:
            data = news_release.get("new_data") or {}
            if "value" in data and data.get("asset") in (None, "", "C"):
                try:
                    self.eps = float(data["value"])
                    print(f"[NEWS] EPS → {self.eps:.3f}")
                except (TypeError, ValueError):
                    pass

        async def start(self):
            self._rebal_task = asyncio.create_task(self._rebalance_loop())
            try:
                await self.connect()
            finally:
                if self._rebal_task:
                    self._rebal_task.cancel()

    async def _main():
        server = os.environ["server"]
        user = os.environ["user"]
        password = os.environ["password"]
        while True:
            try:
                print(f"[CONNECT] {server} as {user}")
                bot = StockCBot(server, user, password)
                await bot.start()
            except (KeyboardInterrupt, asyncio.CancelledError):
                print("\n[SHUTDOWN]")
                return
            except Exception as e:
                print(f"[DISCONNECT] {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)

    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        print("\n[SHUTDOWN]")