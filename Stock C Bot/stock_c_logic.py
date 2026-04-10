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

# CPI-surprise → prob-market sizing buckets (|actual - forecast| in decimal)
CPI_BUCKETS = [
    (0.0005, 0),    # noise → no trade
    (0.0015, 20),   # small surprise
    (0.0030, 60),   # medium surprise
    (float("inf"), 120),  # large surprise → full size
]

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

    REBALANCE_SEC = 80
    MAX_ORDER_SIZE = 40
    MAX_TARGET_POS = 150
    SIZE_PER_DOLLAR = 20.0
    MAX_HEDGE_POS = 120
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
                dpdy, _, _ = dP_dy(q_h, q_hold_, q_c, self.eps)
            finally:
                gamma, beta_y = g0, b0

            return {"probs": probs, "fv": fv, "dpdy": dpdy}

        async def _cpi_trade(self, actual: float, forecast: float):
            """
            Crude CPI-surprise directional trade in the prob market.
            Hot CPI (actual > forecast) → buy HIKE, sell CUT.
            Cold CPI (actual < forecast) → sell HIKE, buy CUT.
            Size by |surprise| via CPI_BUCKETS. Fires on every CPI print.
            """
            surprise = actual - forecast
            mag = abs(surprise)

            # bucketed size (crude if-ladder)
            size = 0
            for thresh, sz in CPI_BUCKETS:
                if mag < thresh:
                    size = sz
                    break

            print(
                f"[CPI] actual={actual:.4f} forecast={forecast:.4f} "
                f"surprise={surprise:+.4f} → size={size}"
            )

            if size == 0:
                return

            # hot → long HIKE, short CUT.  cold → flip.
            if surprise > 0:
                target_hike, target_cut = +size, -size
            else:
                target_hike, target_cut = -size, +size

            pos_h = int(self.positions.get("R_HIKE", 0))
            pos_c = int(self.positions.get("R_CUT", 0))

            print(
                f"[CPI TRADE] HIKE {pos_h:+d}→{target_hike:+d}   "
                f"CUT {pos_c:+d}→{target_cut:+d}"
            )

            # fire both legs concurrently so we hit the book before the rest of the
            # market re-prices on the same print
            await asyncio.gather(
                self._place_toward("R_HIKE", target_hike - pos_h),
                self._place_toward("R_CUT", target_cut - pos_c),
            )

        async def _place_toward(self, symbol, delta):
            if abs(delta) < 1:
                return

            book = self.order_books.get(symbol)
            if not book:
                return

            bid = max(book.bids.keys(), default=None)
            ask = min(book.asks.keys(), default=None)

            if delta > 0:
                px = ask if ask is not None else bid
                if px is None:
                    return
                qty = min(delta, MAX_ORDER_SIZE)
                await self.place_order(symbol, qty, "buy", int(px))
                print(f"[ORDER] {symbol} BUY {qty}@{px}")
            else:
                px = bid if bid is not None else ask
                if px is None:
                    return
                qty = min(-delta, MAX_ORDER_SIZE)
                await self.place_order(symbol, qty, "sell", int(px))
                print(f"[ORDER] {symbol} SELL {qty}@{px}")

        async def _rebalance(self):
            self.day += 1
            info = self._fair_value()
            c_mid = _mid(self.order_books.get(SYMBOL))

            if info is None or c_mid is None:
                return

            if self._model_base is None:
                self._model_base = info["fv"]
                self._price_base = c_mid

            fv = self._price_base + (info["fv"] - self._model_base)
            mis = fv - c_mid

            target = int(np.clip(SIZE_PER_DOLLAR * mis, -MAX_TARGET_POS, MAX_TARGET_POS))
            pos = int(self.positions.get(SYMBOL, 0))

            print(f"[DAY {self.day}] C mid={c_mid:.2f} fv={fv:.2f} mis={mis:.2f} pos={pos}→{target}")

            await self._place_toward(SYMBOL, target - pos)

        async def bot_handle_order_fill(self, order_id, qty, price):
            info = self.open_orders.get(order_id)
            if not info:
                return

            is_buy = info[0].side == utc_pb2.NewOrderRequest.Side.BUY  # type: ignore[attr-defined]
            signed = qty if is_buy else -qty

            self.net_pos += signed
            print(f"[FILL] pos={self.net_pos}")

        async def bot_handle_news(self, news_release: dict) -> None:
            data = news_release.get("new_data") or {}
            subtype = data.get("structured_subtype")

            if subtype == "earnings":
                try:
                    self.eps = float(data["value"])
                    print(f"[NEWS] EPS → {self.eps:.3f}")
                except (TypeError, ValueError):
                    pass
                return

            if subtype == "cpi_print":
                try:
                    actual = float(data["actual"])
                    forecast = float(data["forecast"])
                except (TypeError, ValueError, KeyError):
                    return
                await self._cpi_trade(actual, forecast)

        async def _loop(self):
            await asyncio.sleep(5)
            while True:
                await self._rebalance()
                await asyncio.sleep(REBALANCE_SEC)

        async def start(self):
            asyncio.create_task(self._loop())
            await self.connect()

    async def main():
        bot = StockCBot(os.environ["server"], os.environ["user"], os.environ["password"])
        await bot.start()

    asyncio.run(main())