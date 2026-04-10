"""
PassiveSniper — book-driven small-passive liquidity provider.

Strategy:
- Place small limit orders just inside the spread.
- Orders rest to get filled naturally.
- No aggressive crossed-book sniping.
- Risk controls and stale order cancellation remain.

Author: Adapted from BaseSniper
"""

import os
import time
import asyncio
import logging
from dotenv import load_dotenv
from utcxchangelib import XChangeClient
import utcxchangelib.service_pb2 as utc_pb2

from performance import PerformanceTracker

load_dotenv()
logging.getLogger("xchange-client").setLevel(logging.WARNING)


class PassiveSniper(XChangeClient):
    # Exchange-enforced limits (per symbol)
    MAX_ORDER_SIZE: int = 40
    MAX_OPEN_ORDERS: int = 50
    MAX_OUTSTANDING_VOLUME: int = 120
    MAX_ABSOLUTE_POSITION: int = 200

    # Risk controls
    STALE_ORDER_MAX_AGE: float = 5.0       # Give orders time to fill passively
    MAX_COMMIT_PER_ORDER: int = 10         # Small order size for passive strategy
    SNAPSHOT_INTERVAL: float = 10.0
    CANCEL_CHECK_INTERVAL: float = 0.25

    def __init__(self, host, username, password, tracker: PerformanceTracker | None = None, **kwargs):
        super().__init__(host, username, password, silent=True, **kwargs)
        self.tracker = tracker or PerformanceTracker()
        self._order_placed_at: dict[str, float] = {}
        self._snapshot_task: asyncio.Task | None = None
        self._cancel_task: asyncio.Task | None = None

    # ---------- place_order wrapper ----------
    async def place_order(self, symbol, qty, side, px=None):
        oid = await super().place_order(symbol, qty, side, px)
        self._order_placed_at[oid] = time.monotonic()
        return oid

    # ---------- capacity helpers ----------
    def _in_flight(self, symbol: str) -> tuple[int, int]:
        buys = sells = 0
        for info in self.open_orders.values():
            if info[0].symbol != symbol:
                continue
            remaining = info[1]
            if info[0].side == utc_pb2.NewOrderRequest.Side.BUY:
                buys += remaining
            else:
                sells += remaining
        return buys, sells

    def capacity_buy(self, symbol: str) -> int:
        pos = self.positions[symbol]
        open_buys, open_sells = self._in_flight(symbol)
        by_position = self.MAX_ABSOLUTE_POSITION - pos - open_buys
        by_volume = self.MAX_OUTSTANDING_VOLUME - open_buys - open_sells
        return max(0, min(by_position, by_volume))

    def capacity_sell(self, symbol: str) -> int:
        pos = self.positions[symbol]
        open_buys, open_sells = self._in_flight(symbol)
        by_position = self.MAX_ABSOLUTE_POSITION + pos - open_sells
        by_volume = self.MAX_OUTSTANDING_VOLUME - open_buys - open_sells
        return max(0, min(by_position, by_volume))

    # ---------- passive liquidity logic ----------
    async def _provide_liquidity(self, symbol: str) -> None:
        book = self.order_books.get(symbol)
        if not book:
            return

        best_bid_px = max(book.bids.keys(), default=0)
        best_ask_px = min(book.asks.keys(), default=None)
        if best_bid_px == 0 or best_ask_px is None:
            return

        # Small passive orders just inside the spread
        buy_px = best_bid_px + 1
        sell_px = best_ask_px - 1

        buy_qty = min(self.MAX_COMMIT_PER_ORDER, self.capacity_buy(symbol))
        sell_qty = min(self.MAX_COMMIT_PER_ORDER, self.capacity_sell(symbol))

        if buy_qty > 0:
            await self.place_order(symbol, buy_qty, "buy", buy_px)
        if sell_qty > 0:
            await self.place_order(symbol, sell_qty, "sell", sell_px)

        if buy_qty > 0 or sell_qty > 0:
            parts = []
            if buy_qty > 0:
                parts.append(f"BUY {buy_qty}@{buy_px}")
            if sell_qty > 0:
                parts.append(f"SELL {sell_qty}@{sell_px}")
            print(f"[PASSIVE {symbol}] {' '.join(parts)}")

    # ---------- book update handler ----------
    async def bot_handle_book_update(self, symbol: str) -> None:
        await self._provide_liquidity(symbol)

    # ---------- fill / cancel / reject ----------
    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int) -> None:
        info = self.open_orders.get(order_id)
        if info is None:
            return
        is_buy = info[0].side == utc_pb2.NewOrderRequest.Side.BUY
        side = "buy" if is_buy else "sell"
        symbol = info[0].symbol
        self.tracker.record_fill(symbol, side, qty, price)
        print(f"[FILL] {symbol} {side.upper()} {qty}@{price}")
        if info[1] == 0:
            self._order_placed_at.pop(order_id, None)

    async def bot_handle_cancel_response(self, order_id: str, success: bool, error) -> None:
        self._order_placed_at.pop(order_id, None)

    async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
        print(f"[REJECT] {order_id}: {reason}")
        self._order_placed_at.pop(order_id, None)

    # ---------- stale-order cancel loop ----------
    async def _cancel_stale_orders_loop(self):
        while True:
            try:
                await asyncio.sleep(self.CANCEL_CHECK_INTERVAL)
                now = time.monotonic()
                stale = [oid for oid, ts in self._order_placed_at.items() if now - ts > self.STALE_ORDER_MAX_AGE]
                for oid in stale:
                    try:
                        await self.cancel_order(oid)
                        self._order_placed_at.pop(oid, None)
                    except Exception as e:
                        print(f"[CANCEL ERR] {oid}: {e}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[CANCEL LOOP] {e}")

    # ---------- performance snapshot loop ----------
    async def _snapshot_loop(self):
        last_trades = 0
        last_eq = None
        while True:
            try:
                await asyncio.sleep(self.SNAPSHOT_INTERVAL)
                eq = self.tracker.snapshot(self.order_books)
                trades = len(self.tracker.trades)
                if trades == last_trades and last_eq is not None and abs(eq - last_eq) < 0.01:
                    continue
                last_trades = trades
                last_eq = eq
                realized = self.tracker.total_realized()
                unreal = eq - realized
                live_pos = [
                    f"{s}={p:+d}"
                    for s, p in sorted(self.tracker.positions.items())
                    if p != 0
                ]
                print(
                    f"[PERF {int(time.time() - self.tracker.start_time):4d}s] "
                    f"trades={trades:4d}  "
                    f"realized={realized:+9.2f}  "
                    f"unreal={unreal:+9.2f}  "
                    f"eq={eq:+9.2f}  "
                    + (" ".join(live_pos) if live_pos else "flat")
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[PERF] {e}")

    # ---------- start / shutdown ----------
    async def start(self):
        self._snapshot_task = asyncio.create_task(self._snapshot_loop())
        self._cancel_task = asyncio.create_task(self._cancel_stale_orders_loop())
        try:
            await self.connect()
        finally:
            for t in (self._snapshot_task, self._cancel_task):
                if t:
                    t.cancel()

    def shutdown_report(self) -> None:
        print(self.tracker.summary(self.order_books))
        base = self.tracker.save()
        self.tracker.plot(f"{base}_equity.png")


async def main():
    server = os.environ["server"]
    user = os.environ["user"]
    password = os.environ["password"]
    tracker = PerformanceTracker()
    client: PassiveSniper | None = None
    try:
        while True:
            try:
                print(f"[CONNECT] {server} as {user}")
                client = PassiveSniper(server, user, password, tracker=tracker)
                await client.start()
            except Exception as e:
                print(f"[DISCONNECT] {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n[SHUTDOWN] interrupted")
    finally:
        if client is not None:
            client.shutdown_report()


if __name__ == "__main__":
    asyncio.run(main())