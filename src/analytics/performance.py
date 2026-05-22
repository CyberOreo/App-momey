"""
Performance analytics engine.

Computes comprehensive trading metrics from a list of closed Trade records:
Sharpe, Sortino, Calmar, max-drawdown, profit-factor, and more.
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich import box

from src.core.models import PerformanceMetrics, Trade, TradeOutcome


_console = Console()


class PerformanceAnalyzer:
    """Compute, visualize, and export performance metrics for completed trades."""

    # ── Public API ─────────────────────────────────────────────────────────────

    def compute_metrics(
        self,
        trades: List[Trade],
        initial_balance: float,
    ) -> PerformanceMetrics:
        """
        Compute full performance metrics from a list of trades.

        Only closed trades (WIN / LOSS / BREAK_EVEN) are included.
        OPEN trades are ignored.
        """
        closed = [
            t for t in trades
            if t.outcome in (TradeOutcome.WIN, TradeOutcome.LOSS, TradeOutcome.BREAK_EVEN)
            and t.realized_pnl is not None
        ]

        if not closed:
            now = datetime.utcnow()
            return PerformanceMetrics(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                total_pnl_pct=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown=0.0,
                max_drawdown_pct=0.0,
                calmar_ratio=0.0,
                avg_holding_time_hours=0.0,
                start_date=now,
                end_date=now,
            )

        # ── Basic counts ────────────────────────────────────────────────────────
        wins = [t for t in closed if t.outcome == TradeOutcome.WIN]
        losses = [t for t in closed if t.outcome == TradeOutcome.LOSS]

        total_trades = len(closed)
        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        # ── PnL aggregates ───────────────────────────────────────────────────────
        all_pnls = [t.realized_pnl for t in closed]  # type: ignore[misc]
        total_pnl = sum(all_pnls)
        total_pnl_pct = total_pnl / initial_balance if initial_balance > 0 else 0.0

        win_pnls = [t.realized_pnl for t in wins]  # type: ignore[misc]
        loss_pnls = [abs(t.realized_pnl) for t in losses]  # type: ignore[misc]

        avg_win = float(np.mean(win_pnls)) if win_pnls else 0.0
        avg_loss = float(np.mean(loss_pnls)) if loss_pnls else 0.0

        gross_profit = sum(p for p in all_pnls if p > 0)
        gross_loss = sum(abs(p) for p in all_pnls if p < 0)
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # ── Equity curve & drawdown ─────────────────────────────────────────────
        equity_curve = self.compute_equity_curve(closed, initial_balance)
        daily_returns = self.compute_daily_returns(equity_curve)

        sharpe_ratio = self._compute_sharpe(daily_returns)
        sortino_ratio = self._compute_sortino(daily_returns)

        dd_series = self.compute_drawdown_series(equity_curve)
        if dd_series:
            max_drawdown_pct = max(abs(pct) for _, pct in dd_series)
        else:
            max_drawdown_pct = 0.0

        balances = [bal for _, bal in equity_curve]
        peak = initial_balance
        max_drawdown_abs = 0.0
        for bal in balances:
            peak = max(peak, bal)
            dd = peak - bal
            max_drawdown_abs = max(max_drawdown_abs, dd)
        max_drawdown = max_drawdown_abs

        # ── Calmar ratio ────────────────────────────────────────────────────────
        sorted_trades = sorted(closed, key=lambda t: t.entry_time)
        start_date = sorted_trades[0].entry_time
        end_date = sorted_trades[-1].exit_time or sorted_trades[-1].entry_time
        days_elapsed = max(1.0, (end_date - start_date).total_seconds() / 86400)
        annualized_return = total_pnl_pct * (365.0 / days_elapsed)
        calmar_ratio = annualized_return / max_drawdown_pct if max_drawdown_pct > 0 else 0.0

        # ── Holding time ────────────────────────────────────────────────────────
        holding_hours = [t.holding_hours for t in closed if t.holding_hours is not None]
        avg_holding_time_hours = float(np.mean(holding_hours)) if holding_hours else 0.0

        logger.info(
            "Performance metrics computed",
            total_trades=total_trades,
            win_rate=round(win_rate, 3),
            total_pnl=round(total_pnl, 2),
            sharpe=round(sharpe_ratio, 3),
        )

        return PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            calmar_ratio=calmar_ratio,
            avg_holding_time_hours=avg_holding_time_hours,
            start_date=start_date,
            end_date=end_date,
        )

    def compute_equity_curve(
        self,
        trades: List[Trade],
        initial_balance: float,
    ) -> List[Tuple[datetime, float]]:
        """
        Build equity curve from closed trades, sorted chronologically.

        Returns a list of (timestamp, balance) tuples.
        The first entry is (earliest_trade_entry_time, initial_balance).
        """
        closed = [
            t for t in trades
            if t.outcome in (TradeOutcome.WIN, TradeOutcome.LOSS, TradeOutcome.BREAK_EVEN)
            and t.realized_pnl is not None
            and t.exit_time is not None
        ]
        closed_sorted = sorted(closed, key=lambda t: t.exit_time)  # type: ignore[arg-type, return-value]

        if not closed_sorted:
            now = datetime.utcnow()
            return [(now, initial_balance)]

        curve: List[Tuple[datetime, float]] = []
        balance = initial_balance
        # Seed the curve at the entry time of the first trade
        curve.append((closed_sorted[0].entry_time, initial_balance))

        for trade in closed_sorted:
            balance += trade.realized_pnl  # type: ignore[operator]
            balance -= trade.fees_paid
            curve.append((trade.exit_time, balance))  # type: ignore[arg-type]

        return curve

    def compute_daily_returns(
        self,
        equity_curve: List[Tuple[datetime, float]],
    ) -> List[float]:
        """
        Group equity curve points by calendar day, then compute
        daily percentage returns: (end_balance - start_balance) / start_balance.
        """
        if len(equity_curve) < 2:
            return []

        # Group by date
        daily_balances: Dict[str, List[float]] = defaultdict(list)
        for ts, bal in equity_curve:
            day_key = ts.strftime("%Y-%m-%d")
            daily_balances[day_key].append(bal)

        sorted_days = sorted(daily_balances.keys())
        # Use last balance of each day
        eod_balances = [daily_balances[d][-1] for d in sorted_days]

        if len(eod_balances) < 2:
            return []

        returns: List[float] = []
        for i in range(1, len(eod_balances)):
            prev = eod_balances[i - 1]
            curr = eod_balances[i]
            if prev > 0:
                returns.append((curr - prev) / prev)

        return returns

    def compute_drawdown_series(
        self,
        equity_curve: List[Tuple[datetime, float]],
    ) -> List[Tuple[datetime, float]]:
        """
        Compute drawdown percentage at each point in the equity curve.

        Returns a list of (timestamp, drawdown_pct) tuples where drawdown_pct
        is a non-negative fraction (0.05 = 5% below peak).
        """
        if not equity_curve:
            return []

        series: List[Tuple[datetime, float]] = []
        peak = equity_curve[0][1]

        for ts, bal in equity_curve:
            peak = max(peak, bal)
            drawdown_pct = (peak - bal) / peak if peak > 0 else 0.0
            series.append((ts, drawdown_pct))

        return series

    def export_to_csv(self, trades: List[Trade], filepath: str) -> None:
        """Write a trade journal CSV file to *filepath*."""
        fieldnames = [
            "trade_id", "market_id", "condition_id", "token_id",
            "direction", "size", "entry_price", "exit_price",
            "entry_time", "exit_time", "holding_hours",
            "realized_pnl", "realized_pnl_pct",
            "outcome", "confidence", "fees_paid",
            "paper_trade", "signal_reasons",
        ]

        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for trade in trades:
                pnl_pct = 0.0
                if trade.realized_pnl is not None and trade.entry_price > 0 and trade.size > 0:
                    cost_basis = trade.size * trade.entry_price
                    pnl_pct = trade.realized_pnl / cost_basis if cost_basis > 0 else 0.0

                writer.writerow({
                    "trade_id": trade.trade_id,
                    "market_id": trade.market_id,
                    "condition_id": trade.condition_id,
                    "token_id": trade.token_id,
                    "direction": trade.direction.value,
                    "size": round(trade.size, 4),
                    "entry_price": round(trade.entry_price, 4),
                    "exit_price": round(trade.exit_price, 4) if trade.exit_price is not None else "",
                    "entry_time": trade.entry_time.isoformat(),
                    "exit_time": trade.exit_time.isoformat() if trade.exit_time else "",
                    "holding_hours": round(trade.holding_hours, 2) if trade.holding_hours else "",
                    "realized_pnl": round(trade.realized_pnl, 4) if trade.realized_pnl is not None else "",
                    "realized_pnl_pct": round(pnl_pct * 100, 3),
                    "outcome": trade.outcome.value,
                    "confidence": round(trade.confidence, 2),
                    "fees_paid": round(trade.fees_paid, 4),
                    "paper_trade": trade.paper_trade,
                    "signal_reasons": "|".join(trade.signal_reasons),
                })

        logger.info("Trade journal exported", filepath=filepath, count=len(trades))

    def print_summary(self, metrics: PerformanceMetrics) -> None:
        """Print a Rich-formatted performance summary table to the console."""
        table = Table(
            title="[bold cyan]PolyBTC Trader — Performance Summary[/bold cyan]",
            box=box.DOUBLE_EDGE,
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Metric", style="dim", width=28)
        table.add_column("Value", justify="right", width=18)

        def pct(v: float) -> str:
            color = "green" if v >= 0 else "red"
            return f"[{color}]{v * 100:.2f}%[/{color}]"

        def currency(v: float) -> str:
            color = "green" if v >= 0 else "red"
            return f"[{color}]${v:,.2f}[/{color}]"

        win_color = "green" if metrics.win_rate >= 0.5 else "red"
        pf_color = "green" if metrics.profit_factor >= 1.5 else ("yellow" if metrics.profit_factor >= 1.0 else "red")

        rows = [
            ("Period", f"{metrics.start_date.date()} → {metrics.end_date.date()}"),
            ("Total Trades", str(metrics.total_trades)),
            ("Wins / Losses", f"[green]{metrics.winning_trades}[/green] / [red]{metrics.losing_trades}[/red]"),
            ("Win Rate", f"[{win_color}]{metrics.win_rate * 100:.1f}%[/{win_color}]"),
            ("Total P&L", currency(metrics.total_pnl)),
            ("Total P&L %", pct(metrics.total_pnl_pct)),
            ("Avg Win", currency(metrics.avg_win)),
            ("Avg Loss", f"[red]${metrics.avg_loss:,.2f}[/red]"),
            ("Profit Factor", f"[{pf_color}]{metrics.profit_factor:.2f}[/{pf_color}]"),
            ("Sharpe Ratio", f"{metrics.sharpe_ratio:.3f}"),
            ("Sortino Ratio", f"{metrics.sortino_ratio:.3f}"),
            ("Max Drawdown", currency(-metrics.max_drawdown)),
            ("Max Drawdown %", f"[red]{metrics.max_drawdown_pct * 100:.2f}%[/red]"),
            ("Calmar Ratio", f"{metrics.calmar_ratio:.3f}"),
            ("Avg Holding Time", f"{metrics.avg_holding_time_hours:.1f}h"),
        ]

        for metric, value in rows:
            table.add_row(metric, value)

        _console.print(table)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _compute_sharpe(self, daily_returns: List[float], risk_free_daily: float = 0.0) -> float:
        """Annualized Sharpe ratio (sqrt(252) * mean/std)."""
        if len(daily_returns) < 2:
            return 0.0
        arr = np.array(daily_returns, dtype=float)
        excess = arr - risk_free_daily
        std = float(np.std(excess, ddof=1))
        if std == 0.0:
            return 0.0
        return float(np.mean(excess) / std * math.sqrt(252))

    def _compute_sortino(self, daily_returns: List[float], target: float = 0.0) -> float:
        """Annualized Sortino ratio using downside deviation."""
        if len(daily_returns) < 2:
            return 0.0
        arr = np.array(daily_returns, dtype=float)
        downside = arr[arr < target]
        if len(downside) == 0:
            return float("inf")
        downside_dev = float(np.std(downside, ddof=1))
        if downside_dev == 0.0:
            return 0.0
        mean_excess = float(np.mean(arr - target))
        return mean_excess / downside_dev * math.sqrt(252)
