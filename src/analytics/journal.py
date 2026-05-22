"""
Trade journal: CSV export and in-memory querying of Trade records.

Provides a lightweight async interface for logging, querying, and
exporting trades without requiring a database connection.
"""
from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from src.core.models import Trade, TradeOutcome


class TradeJournal:
    """
    Async trade journal backed by an in-memory list with optional DB pass-through.

    All write methods are async to allow future DB integration; the current
    implementation stores trades in memory and syncs to DB when a connection
    is provided.

    Parameters
    ----------
    db:
        Optional SQLAlchemy AsyncEngine.  When provided, trades are also
        persisted via TradeRepository.  Pass None for memory-only mode.
    export_dir:
        Directory where CSV exports are written (created on first export).
    """

    def __init__(self, db=None, export_dir: str = "data/journal") -> None:
        self._db = db
        self._export_dir = export_dir
        self._trades: List[Trade] = []

    # ── Write ──────────────────────────────────────────────────────────────────

    async def add_trade(self, trade: Trade) -> None:
        """
        Record a trade in the journal.

        Appends to the in-memory list and, when a DB engine is available,
        persists via TradeRepository.
        """
        self._trades.append(trade)

        if self._db is not None:
            try:
                from src.core.database import TradeRepository
                repo = TradeRepository(self._db)
                await repo.save_trade(trade)
            except Exception as exc:
                logger.warning(
                    "TradeJournal: failed to persist trade to DB",
                    trade_id=trade.trade_id,
                    error=str(exc),
                )

        logger.debug(
            "Trade recorded in journal",
            trade_id=trade.trade_id,
            direction=trade.direction.value,
            outcome=trade.outcome.value,
        )

    # ── Query ──────────────────────────────────────────────────────────────────

    async def get_trades(
        self,
        since: Optional[datetime] = None,
        outcome: Optional[TradeOutcome] = None,
    ) -> List[Trade]:
        """
        Return trades filtered by entry time and/or outcome.

        Parameters
        ----------
        since:
            If provided, only trades with entry_time >= since are returned.
        outcome:
            If provided, only trades with the matching outcome are returned.
        """
        result = self._trades

        if since is not None:
            result = [t for t in result if t.entry_time >= since]

        if outcome is not None:
            result = [t for t in result if t.outcome == outcome]

        return sorted(result, key=lambda t: t.entry_time)

    async def get_recent_trades(self, n: int = 20) -> List[Trade]:
        """Return the *n* most recently entered trades."""
        sorted_trades = sorted(self._trades, key=lambda t: t.entry_time, reverse=True)
        return sorted_trades[:n]

    # ── Stats ──────────────────────────────────────────────────────────────────

    async def get_summary_stats(self) -> Dict[str, Any]:
        """
        Compute summary statistics over all recorded trades.

        Returns
        -------
        dict with keys:
            trade_count, open_trades, closed_trades,
            win_rate, total_pnl, avg_confidence,
            winning_trades, losing_trades.
        """
        total = len(self._trades)
        closed = [
            t for t in self._trades
            if t.outcome in (TradeOutcome.WIN, TradeOutcome.LOSS, TradeOutcome.BREAK_EVEN)
            and t.realized_pnl is not None
        ]
        open_trades = [t for t in self._trades if t.outcome == TradeOutcome.OPEN]
        wins = [t for t in closed if t.outcome == TradeOutcome.WIN]
        losses = [t for t in closed if t.outcome == TradeOutcome.LOSS]

        win_rate = len(wins) / len(closed) if closed else 0.0
        total_pnl = sum(t.realized_pnl for t in closed if t.realized_pnl is not None)
        avg_confidence = (
            sum(t.confidence for t in self._trades) / total if total > 0 else 0.0
        )

        return {
            "trade_count": total,
            "open_trades": len(open_trades),
            "closed_trades": len(closed),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": round(win_rate, 4),
            "total_pnl": round(total_pnl, 4),
            "avg_confidence": round(avg_confidence, 2),
        }

    # ── Export ─────────────────────────────────────────────────────────────────

    async def export_csv(self, filepath: Optional[str] = None) -> str:
        """
        Export all trades to a CSV file.

        Parameters
        ----------
        filepath:
            Full path for the output file.  If None, a timestamped file is
            created inside ``self._export_dir``.

        Returns
        -------
        str
            Absolute path of the written CSV file.
        """
        if filepath is None:
            os.makedirs(self._export_dir, exist_ok=True)
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self._export_dir, f"trades_{ts}.csv")

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
            for trade in sorted(self._trades, key=lambda t: t.entry_time):
                writer.writerow(self._trade_to_dict(trade))

        logger.info(
            "Trade journal exported to CSV",
            filepath=filepath,
            trade_count=len(self._trades),
        )
        return filepath

    # ── Report ─────────────────────────────────────────────────────────────────

    def generate_report(self, trades: List[Trade]) -> str:
        """
        Generate a plain-text performance report for the given trades.

        Parameters
        ----------
        trades:
            Trades to include in the report (any subset).

        Returns
        -------
        str — multi-line formatted report.
        """
        closed = [
            t for t in trades
            if t.outcome in (TradeOutcome.WIN, TradeOutcome.LOSS, TradeOutcome.BREAK_EVEN)
            and t.realized_pnl is not None
        ]
        open_t = [t for t in trades if t.outcome == TradeOutcome.OPEN]
        wins = [t for t in closed if t.outcome == TradeOutcome.WIN]
        losses = [t for t in closed if t.outcome == TradeOutcome.LOSS]

        total_pnl = sum(t.realized_pnl for t in closed if t.realized_pnl is not None)
        win_rate = len(wins) / len(closed) if closed else 0.0
        avg_conf = sum(t.confidence for t in trades) / len(trades) if trades else 0.0

        lines = [
            "=" * 60,
            "  PolyBTC Trader — Trade Journal Report",
            f"  Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
            "=" * 60,
            f"  Total Trades  : {len(trades)}",
            f"  Open Trades   : {len(open_t)}",
            f"  Closed Trades : {len(closed)}",
            f"  Wins          : {len(wins)}",
            f"  Losses        : {len(losses)}",
            f"  Win Rate      : {win_rate * 100:.1f}%",
            f"  Total P&L     : ${total_pnl:+.2f}",
            f"  Avg Confidence: {avg_conf:.1f}",
            "",
        ]

        if closed:
            lines.append("  Recent Closed Trades:")
            lines.append(f"  {'ID':10s} {'Dir':4s} {'Entry':8s} {'Exit':8s} {'P&L':10s} {'Outcome':12s}")
            lines.append("  " + "-" * 54)
            for t in sorted(closed, key=lambda x: x.entry_time, reverse=True)[:10]:
                entry = f"{t.entry_price:.4f}"
                exit_p = f"{t.exit_price:.4f}" if t.exit_price else "N/A"
                pnl_str = f"${t.realized_pnl:+.2f}" if t.realized_pnl is not None else "N/A"
                lines.append(
                    f"  {t.trade_id[:10]:10s} {t.direction.value:4s} {entry:8s} "
                    f"{exit_p:8s} {pnl_str:10s} {t.outcome.value:12s}"
                )

        lines.append("=" * 60)
        return "\n".join(lines)

    # ── Serialization ──────────────────────────────────────────────────────────

    def _trade_to_dict(self, trade: Trade) -> Dict[str, Any]:
        """
        Convert a Trade to a flat dict suitable for CSV serialization.

        All numeric values are rounded to 4 decimal places.
        """
        pnl_pct = 0.0
        if trade.realized_pnl is not None and trade.entry_price > 0 and trade.size > 0:
            cost_basis = trade.size * trade.entry_price
            if cost_basis > 0:
                pnl_pct = (trade.realized_pnl / cost_basis) * 100.0

        return {
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
            "holding_hours": round(trade.holding_hours, 2) if trade.holding_hours is not None else "",
            "realized_pnl": round(trade.realized_pnl, 4) if trade.realized_pnl is not None else "",
            "realized_pnl_pct": round(pnl_pct, 3),
            "outcome": trade.outcome.value,
            "confidence": round(trade.confidence, 2),
            "fees_paid": round(trade.fees_paid, 4),
            "paper_trade": trade.paper_trade,
            "signal_reasons": "|".join(trade.signal_reasons),
        }
