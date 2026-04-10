"""
The goal is to build a fair value model for Stock C.
"""

import numpy as np

# ---- Hyperparameters ----
y0 = 0.045
PE0 = 14.0
EPS0 = 2.00
D = 7.5
C_conv = 55.0
B0_per_share = 40.0
lam = 0.65
beta_y = 1.0
gamma = 0.5

# Reference lot size for the displayed hedge guide
HEDGE_LOT = 100

# ---- Core Functions ----

def expected_rate_change(q_hike, q_hold, q_cut):
    return 25 * q_hike + 0 * q_hold + (-25) * q_cut

def variance_rate_change(q_hike, q_hold, q_cut):
    e_dr = expected_rate_change(q_hike, q_hold, q_cut)
    e_dr2 = (25**2) * q_hike + (25**2) * q_cut
    return e_dr2 - e_dr**2

def yield_from_probs(q_hike, q_hold, q_cut):
    e_dr = expected_rate_change(q_hike, q_hold, q_cut)
    return y0 + beta_y * (e_dr / 10000)

def fair_value_C(q_hike, q_hold, q_cut, eps=EPS0):
    y_t = yield_from_probs(q_hike, q_hold, q_cut)
    dy = y_t - y0

    pe_t = PE0 * np.exp(-gamma * dy)
    ops = eps * pe_t

    e_dr = expected_rate_change(q_hike, q_hold, q_cut)
    var_dr = variance_rate_change(q_hike, q_hold, q_cut)

    e_dy = beta_y * (e_dr / 10000)
    e_dy2 = (beta_y / 10000)**2 * (var_dr + e_dr**2)

    delta_B_per_share = B0_per_share * (-D * e_dy + 0.5 * C_conv * e_dy2)
    bonds = lam * delta_B_per_share

    return ops + bonds, ops, bonds

def dP_dy(q_hike, q_hold, q_cut, eps=EPS0):
    y_t = yield_from_probs(q_hike, q_hold, q_cut)
    dy = y_t - y0

    pe_t = PE0 * np.exp(-gamma * dy)
    d_ops = -gamma * eps * pe_t
    d_bonds = lam * B0_per_share * (-D + C_conv * dy)

    return d_ops + d_bonds, d_ops, d_bonds

def d2P_dy2(q_hike, q_hold, q_cut, eps=EPS0):
    y_t = yield_from_probs(q_hike, q_hold, q_cut)
    dy = y_t - y0

    pe_t = PE0 * np.exp(-gamma * dy)
    g_ops = gamma**2 * eps * pe_t
    g_bonds = lam * B0_per_share * C_conv

    return g_ops + g_bonds, g_ops, g_bonds


# ---- Live Bot ----

if __name__ == "__main__":
    import os, sys, time, asyncio, logging
    from pathlib import Path
    from dotenv import load_dotenv
    from utcxchangelib import XChangeClient
    import utcxchangelib.service_pb2 as utc_pb2

    HERE = Path(__file__).resolve().parent
    sys.path.insert(0, str(HERE))
    from hyper_parameter_regress import ExchangeDataStore, calibrate_nonlinear

    load_dotenv(HERE.parent / "order-sniper" / ".env")
    logging.getLogger("xchange-client").setLevel(logging.WARNING)

    SYMBOL = "C"
    PROB_SYMS = ("R_HIKE", "R_HOLD", "R_CUT")
    WATCHED_SYMS = {SYMBOL, *PROB_SYMS}

    HEARTBEAT_SEC = 45      # fallback re-print if nothing moves
    BOOK_DEBOUNCE_SEC = 1.5 # min gap between book-triggered reprints
    CALIB_MIN_OBS = 25

    def _mid(book):
        if not book:
            return None
        bid = max(book.bids.keys(), default=None)
        ask = min(book.asks.keys(), default=None)
        if bid is not None and ask is not None:
            return 0.5 * (bid + ask)
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

            self.net_pos = 0
            self.avg_cost = 0
            self.total_pnl = 0
            self.wins = 0
            self.losses = 0
            self.day = 0

            self._model_base = None
            self._price_base = None
            self._market_open = True

            # event-driven dashboard
            self._last_print_ts = 0.0
            self._print_lock = asyncio.Lock()

        def _read_probs(self):
            mids: list[float] = []
            for s in PROB_SYMS:
                m = _mid(self.order_books.get(s))
                if m is None or m <= 0:
                    return None
                mids.append(m)
            tot = sum(mids)
            if tot <= 0:
                return None
            return tuple(m / tot for m in mids)

        def _fair_value(self):
            probs = self._read_probs()
            if probs is None:
                return None

            q_h, q_hold_, q_c = probs
            global gamma, beta_y
            g0, b0 = gamma, beta_y
            gamma, beta_y = self.gamma_live, self.beta_y_live

            try:
                fv, ops, bonds = fair_value_C(q_h, q_hold_, q_c, self.eps)
                dpdy_tot, dpdy_ops, dpdy_bonds = dP_dy(q_h, q_hold_, q_c, self.eps)
                gpdy_tot, _, _ = d2P_dy2(q_h, q_hold_, q_c, self.eps)
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

        def _hedge_for(self, pos_c: int, info: dict) -> tuple[int, int]:
            """
            Neutralize C's dP/dy exposure across HIKE and CUT scenarios.
            Returns (N_hike, N_cut) rounded to ints.
            """
            q_h, q_hold_, q_c = info["probs"]
            hike_pnl = pos_c * info["dpdy"] * 0.0025   # C P&L if HIKE (+25bps) realizes
            cut_pnl  = -hike_pnl                       # C P&L if CUT (-25bps) realizes

            if q_hold_ <= 1e-6:
                return 0, 0

            det = q_hold_
            N_h = (-(1 - q_c) * hike_pnl - q_c * cut_pnl) / det
            N_c = (-(1 - q_h) * cut_pnl - q_h * hike_pnl) / det
            return int(round(N_h)), int(round(N_c))

        async def _maybe_print(self, *, reason: str, force: bool = False):
            """Debounced dashboard print. Force=True bypasses the debounce window."""
            now = time.monotonic()
            if not force and (now - self._last_print_ts) < BOOK_DEBOUNCE_SEC:
                return
            if self._print_lock.locked():
                return
            async with self._print_lock:
                self._last_print_ts = time.monotonic()
                await self._print_dashboard(reason)

        async def _print_dashboard(self, reason: str = "heartbeat"):
            self.day += 1
            info = self._fair_value()
            c_mid = _mid(self.order_books.get(SYMBOL))

            if info is None or c_mid is None:
                print(f"[TICK {self.day}] skip — book not ready")
                return

            if self._model_base is None:
                self._model_base = info["fv_model"]
                self._price_base = c_mid

            fv = self._price_base + (info["fv_model"] - self._model_base)
            mis = fv - c_mid
            q_h, q_hold_, q_c = info["probs"]
            dpdy = info["dpdy"]
            dpdy_bp = dpdy * 1e-4  # $ per bp per share

            signal = (
                "CHEAP  (consider LONG)" if mis > 0.5
                else "RICH   (consider SHORT)" if mis < -0.5
                else "FAIR   (flat)"
            )

            hike_long, cut_long   = self._hedge_for(+HEDGE_LOT, info)
            hike_short, cut_short = self._hedge_for(-HEDGE_LOT, info)

            pos_c_now = int(self.positions.get(SYMBOL, 0))
            if abs(pos_c_now) > 0:
                hike_now, cut_now = self._hedge_for(pos_c_now, info)
                live_line = (
                    f"  LIVE (pos {pos_c_now:+d} C): HIKE {hike_now:+d}  CUT {cut_now:+d}"
                )
            else:
                live_line = "  LIVE (pos 0): no hedge needed"

            lines = [
                "",
                "═══════════════════════════════════════════════════════════",
                f"  TICK {self.day:3d}  [{reason}]  EPS={self.eps:.3f}    "
                f"γ={self.gamma_live:.3f}  β_y={self.beta_y_live:.3f}",
                "═══════════════════════════════════════════════════════════",
                f"  Prediction market   HIKE={q_h:.3f}  HOLD={q_hold_:.3f}  CUT={q_c:.3f}",
                f"  E[Δr] = {info['e_dr']:+6.2f} bps   Var[Δr] = {info['var_dr']:8.2f} bps²",
                "",
                f"  Stock C",
                f"    market mid     {c_mid:8.2f}",
                f"    fair value     {fv:8.2f}   (mispricing = {mis:+.2f})",
                f"    model breakdown: ops={info['ops']:+.3f}  bonds={info['bonds']:+.3f}",
                f"    signal         {signal}",
                "",
                f"  Sensitivities (per share of C)",
                f"    dP/dy        = {dpdy:+9.2f}  $/unit-yield",
                f"                 = {dpdy_bp:+9.4f}  $/bp",
                f"      ops channel  = {info['dpdy_ops']:+9.2f}",
                f"      bonds channel= {info['dpdy_bonds']:+9.2f}",
                f"    d²P/dy²      = {info['gpdy']:+9.2f}  (convexity)",
                "",
                f"  Hedge guide — neutralize ±25bps scenarios in prob market",
                f"   (solved 2x2: long N_H HIKE + long N_C CUT)",
                f"    if LONG  {HEDGE_LOT} C : HIKE {hike_long:+d}   CUT {cut_long:+d}",
                f"    if SHORT {HEDGE_LOT} C : HIKE {hike_short:+d}   CUT {cut_short:+d}",
                live_line,
                "═══════════════════════════════════════════════════════════",
                "",
            ]
            print("\n".join(lines))

            # keep the calibration pipeline fed so gamma/beta_y stay live
            self.store.record(time.time(), c_mid, q_h, q_hold_, q_c, self.eps)
            self._calibrate()

        async def bot_handle_book_update(self, symbol: str) -> None:
            if symbol in WATCHED_SYMS:
                await self._maybe_print(reason=f"book:{symbol}")

        async def bot_handle_order_fill(self, order_id, qty, price):
            # user is trading manually; just log fills for visibility
            info = self.open_orders.get(order_id)
            if not info:
                return
            is_buy = info[0].side == utc_pb2.NewOrderRequest.Side.BUY  # type: ignore[attr-defined]
            side = "BUY " if is_buy else "SELL"
            sym = info[0].symbol
            print(f"[FILL] {sym} {side} {qty}@{price}")

        async def bot_handle_news(self, news_release: dict) -> None:
            data = news_release.get("new_data") or {}
            subtype = data.get("structured_subtype")

            if subtype == "earnings":
                asset = str(data.get("asset", "")).upper()
                if asset != SYMBOL:
                    # earnings for A or B — ignore, we only model C
                    print(f"[NEWS] {asset} EPS={data.get('value')} (ignored)")
                    return
                try:
                    self.eps = float(data["value"])
                    print(f"[NEWS] C EPS → {self.eps:.3f}")
                except (TypeError, ValueError):
                    return
                await self._maybe_print(reason="C EPS news", force=True)
                return

            if subtype == "cpi_print":
                try:
                    actual = float(data["actual"])
                    forecast = float(data["forecast"])
                except (TypeError, ValueError, KeyError):
                    return
                surprise = actual - forecast
                tag = "HOT " if surprise > 0 else ("COLD" if surprise < 0 else "FLAT")
                bias = (
                    "buy HIKE / sell CUT" if surprise > 0
                    else "sell HIKE / buy CUT" if surprise < 0
                    else "no bias"
                )
                print(
                    f"\n[CPI ALERT] {tag}  actual={actual:.4f}  forecast={forecast:.4f}  "
                    f"surprise={surprise:+.4f}  →  {bias}\n"
                )
                await self._maybe_print(reason="CPI news", force=True)

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

        async def _loop(self):
            await asyncio.sleep(5)
            while True:
                try:
                    await self._maybe_print(reason="heartbeat", force=True)
                except Exception as e:
                    print(f"[PRINT ERR] {e}")
                await asyncio.sleep(HEARTBEAT_SEC)

        async def start(self):
            asyncio.create_task(self._loop())
            await self.connect()

    async def main():
        bot = StockCBot(os.environ["server"], os.environ["user"], os.environ["password"])
        await bot.start()

    asyncio.run(main())