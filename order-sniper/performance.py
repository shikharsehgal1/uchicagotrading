"""
Performance tracker for sniper bots.

Tracks realized and unrealized P&L per symbol using average-cost accounting,
snapshots total equity over time, and computes Sharpe ratio, max drawdown,
and Calmar ratio.

Usage from a bot:
    self.tracker = PerformanceTracker()
    self.tracker.record_fill("A", "buy", 10, 945)
    self.tracker.snapshot(self.order_books)
    print(self.tracker.summary(self.order_books))
    self.tracker.save()          # writes equity/trades CSV under perf_logs/
    self.tracker.plot("eq.png")  # optional matplotlib plot

Standalone (re-plot a saved run):
    python3 performance.py perf_logs/perf_20260409_180000_equity.csv
"""

import csv
import time
from collections import defaultdict
from math import sqrt
from pathlib import Path


class PerformanceTracker:
    def __init__(self, log_dir: str = "perf_logs"):
        self.realized_pnl: dict[str, float] = defaultdict(float)
        self.positions: dict[str, int] = defaultdict(int)
        self.avg_cost: dict[str, float] = defaultdict(float)
        self.trades: list[dict] = []
        self.equity_curve: list[tuple[float, float]] = []  # (time, equity)
        self.start_time = time.time()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

    # ---------- fills / accounting ----------

    def record_fill(self, symbol: str, side: str, qty: int, price: int) -> None:
        """
        Average-cost accounting. Handles opening, adding, reducing, and flipping
        through zero. Updates realized P&L on any position reduction.
        """
        signed_qty = qty if side == "buy" else -qty
        pos = self.positions[symbol]
        avg = self.avg_cost[symbol]

        if pos == 0:
            # Open new position
            self.positions[symbol] = signed_qty
            self.avg_cost[symbol] = float(price)
        elif (pos > 0) == (signed_qty > 0):
            # Same direction — add to position, recompute weighted avg
            new_pos = pos + signed_qty
            self.avg_cost[symbol] = (pos * avg + signed_qty * price) / new_pos
            self.positions[symbol] = new_pos
        else:
            # Opposite direction — reducing or flipping
            closing_qty = min(abs(signed_qty), abs(pos))
            pnl_per_unit = (price - avg) if pos > 0 else (avg - price)
            self.realized_pnl[symbol] += closing_qty * pnl_per_unit

            if abs(signed_qty) <= abs(pos):
                # Reduce only
                self.positions[symbol] = pos + signed_qty
                if self.positions[symbol] == 0:
                    self.avg_cost[symbol] = 0.0
            else:
                # Flip through zero: close existing, open opposite
                self.positions[symbol] = signed_qty + pos  # signed remainder
                self.avg_cost[symbol] = float(price)

        self.trades.append({
            "t": time.time() - self.start_time,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": price,
            "position_after": self.positions[symbol],
            "avg_cost_after": round(self.avg_cost[symbol], 4),
            "realized_pnl_after": round(self.realized_pnl[symbol], 4),
        })

    # ---------- mark-to-market ----------

    def unrealized_pnl(self, order_books) -> float:
        total = 0.0
        for symbol, pos in self.positions.items():
            if pos == 0:
                continue
            book = order_books.get(symbol)
            if not book:
                continue
            bids = [p for p, q in book.bids.items() if q > 0]
            asks = [p for p, q in book.asks.items() if q > 0]
            if not bids or not asks:
                continue
            mid = (max(bids) + min(asks)) / 2
            avg = self.avg_cost[symbol]
            total += pos * (mid - avg)  # works for both long and short
        return total

    def total_realized(self) -> float:
        return sum(self.realized_pnl.values())

    def total_equity(self, order_books) -> float:
        return self.total_realized() + self.unrealized_pnl(order_books)

    def snapshot(self, order_books) -> float:
        eq = self.total_equity(order_books)
        self.equity_curve.append((time.time() - self.start_time, eq))
        return eq

    # ---------- stats ----------

    def max_drawdown(self) -> float:
        if len(self.equity_curve) < 2:
            return 0.0
        peak = self.equity_curve[0][1]
        max_dd = 0.0
        for _, e in self.equity_curve:
            peak = max(peak, e)
            max_dd = max(max_dd, peak - e)
        return max_dd

    def sharpe_ratio(self) -> float:
        """
        Per-snapshot Sharpe (unitless): mean(return) / std(return).
        Not annualized — for a competition run the raw number is more useful
        than an annualized extrapolation of a short sample.
        """
        if len(self.equity_curve) < 3:
            return 0.0
        eq = [e for _, e in self.equity_curve]
        returns = [eq[i] - eq[i - 1] for i in range(1, len(eq))]
        n = len(returns)
        mean = sum(returns) / n
        var = sum((r - mean) ** 2 for r in returns) / n
        std = sqrt(var)
        return mean / std if std > 0 else 0.0

    def calmar_ratio(self) -> float:
        """Total P&L over the run divided by max drawdown. Some traders call
        this the Damian coefficient."""
        dd = self.max_drawdown()
        if dd <= 0 or not self.equity_curve:
            return 0.0
        total_return = self.equity_curve[-1][1] - self.equity_curve[0][1]
        return total_return / dd

    # ---------- reporting ----------

    def summary(self, order_books=None) -> str:
        lines = ["=" * 56, "PERFORMANCE SUMMARY", "=" * 56]
        lines.append(f"Elapsed:         {time.time() - self.start_time:>12.1f} s")
        lines.append(f"Realized P&L:    {self.total_realized():>12.2f}")
        if order_books is not None:
            lines.append(f"Unrealized P&L:  {self.unrealized_pnl(order_books):>12.2f}")
            lines.append(f"Total equity:    {self.total_equity(order_books):>12.2f}")
        lines.append(f"Trades:          {len(self.trades):>12d}")
        lines.append(f"Max drawdown:    {self.max_drawdown():>12.2f}")
        lines.append(f"Sharpe:          {self.sharpe_ratio():>12.3f}")
        lines.append(f"Calmar:          {self.calmar_ratio():>12.3f}")
        lines.append("")
        lines.append(f"{'symbol':10s} {'realized':>12s} {'position':>10s} {'avg_cost':>10s}")
        for s in sorted(set(list(self.realized_pnl) + list(self.positions))):
            pnl = self.realized_pnl.get(s, 0.0)
            pos = self.positions.get(s, 0)
            if pnl == 0 and pos == 0:
                continue
            avg = self.avg_cost.get(s, 0.0)
            lines.append(f"{s:10s} {pnl:>12.2f} {pos:>10d} {avg:>10.2f}")
        lines.append("=" * 56)
        return "\n".join(lines)

    # ---------- persistence ----------

    def save(self, tag: str | None = None) -> Path:
        tag = tag or time.strftime("%Y%m%d_%H%M%S")
        base = self.log_dir / f"perf_{tag}"
        with open(str(base) + "_equity.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_seconds", "equity"])
            for t, e in self.equity_curve:
                w.writerow([f"{t:.3f}", f"{e:.4f}"])
        if self.trades:
            with open(str(base) + "_trades.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(self.trades[0].keys()))
                w.writeheader()
                w.writerows(self.trades)
        return base

    def plot(self, path: str = "performance.png") -> bool:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[PERF] matplotlib not installed — skipping plot")
            return False
        if len(self.equity_curve) < 2:
            print("[PERF] not enough equity points to plot")
            return False
        ts = [t for t, _ in self.equity_curve]
        eq = [e for _, e in self.equity_curve]
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(ts, eq, linewidth=1.2, color="#1f77b4")
        ax.fill_between(ts, eq, alpha=0.15, color="#1f77b4")
        ax.axhline(0, color="k", linewidth=0.6)
        ax.set_title(
            f"Equity — realized {self.total_realized():+.0f}  "
            f"Sharpe {self.sharpe_ratio():+.2f}  Calmar {self.calmar_ratio():+.2f}  "
            f"trades {len(self.trades)}"
        )
        ax.set_xlabel("seconds since start")
        ax.set_ylabel("equity")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"[PERF] saved plot to {path}")
        return True


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 performance.py <equity_csv>")
        sys.exit(1)
    t = PerformanceTracker()
    t.start_time = 0.0
    with open(sys.argv[1]) as f:
        r = csv.reader(f)
        next(r)  # header
        for row in r:
            t.equity_curve.append((float(row[0]), float(row[1])))
    out = sys.argv[1].replace(".csv", ".png")
    t.plot(out)
