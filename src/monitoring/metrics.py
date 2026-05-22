"""
Live metrics collector for monitoring system health and trading performance.

Thread-safe in-memory store; all methods are synchronous since they only
manipulate Python lists/dicts — no I/O occurs.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Tuple

from loguru import logger

from src.core.models import Trade, TradeOutcome, TradeSignal


# Maximum history kept in memory per metric
_MAX_SIGNALS = 500
_MAX_TRADES = 500
_MAX_ERRORS = 200
_MAX_LATENCY_SAMPLES = 100


class MetricsCollector:
    """
    In-memory metrics store for signals, trades, errors, and API latency.

    Usage
    -----
    collector = MetricsCollector()
    collector.record_signal(signal)
    collector.record_api_latency("polymarket", 123.4)
    snapshot = collector.get_current_metrics()
    """

    def __init__(self) -> None:
        self._start_time = time.monotonic()
        self._start_dt = datetime.utcnow()

        self._signals: Deque[TradeSignal] = deque(maxlen=_MAX_SIGNALS)
        self._trades: Deque[Trade] = deque(maxlen=_MAX_TRADES)
        self._errors: Deque[Dict[str, Any]] = deque(maxlen=_MAX_ERRORS)

        # {api_name: deque of latency_ms floats}
        self._latencies: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=_MAX_LATENCY_SAMPLES)
        )

        self._last_price_update: Optional[datetime] = None
        self._last_scan_time: Optional[datetime] = None

    # ── Recording ─────────────────────────────────────────────────────────────

    def record_signal(self, signal: TradeSignal) -> None:
        """
        Record a generated TradeSignal for tracking.

        Parameters
        ----------
        signal:
            The TradeSignal to record.
        """
        self._signals.append(signal)
        logger.debug(
            "MetricsCollector: signal recorded",
            market=signal.market_id,
            direction=signal.direction.value,
            confidence=signal.confidence,
        )

    def record_trade(self, trade: Trade) -> None:
        """
        Record a Trade (open or closed) for tracking.

        Parameters
        ----------
        trade:
            The Trade to record.
        """
        self._trades.append(trade)
        logger.debug(
            "MetricsCollector: trade recorded",
            trade_id=trade.trade_id,
            outcome=trade.outcome.value,
        )

    def record_error(self, error: Exception, context: str) -> None:
        """
        Record an exception with its context string.

        Parameters
        ----------
        error:
            The exception instance.
        context:
            Human-readable description of where the error occurred.
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "error_type": type(error).__name__,
            "message": str(error),
            "context": context,
        }
        self._errors.append(entry)
        logger.debug(
            "MetricsCollector: error recorded",
            type=type(error).__name__,
            context=context,
        )

    def record_api_latency(self, api_name: str, latency_ms: float) -> None:
        """
        Record an API call latency sample.

        Parameters
        ----------
        api_name:
            Identifier for the API (e.g. "polymarket", "binance_rest").
        latency_ms:
            Round-trip time in milliseconds.
        """
        self._latencies[api_name].append(latency_ms)

    def mark_price_update(self) -> None:
        """Update the last price update timestamp."""
        self._last_price_update = datetime.utcnow()

    def mark_scan_complete(self) -> None:
        """Update the last market scan timestamp."""
        self._last_scan_time = datetime.utcnow()

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Return a full snapshot of all collected metrics.

        Returns
        -------
        dict with keys:
            signals_today      — int
            trades_today       — int
            win_rate_today     — float
            errors_today       — int
            api_latencies      — {api_name: avg_latency_ms}
            uptime_seconds     — float
            last_price_update  — ISO timestamp or None
            last_scan_time     — ISO timestamp or None
            total_signals      — int
            total_trades       — int
            total_errors       — int
        """
        today = datetime.utcnow().date()

        signals_today = sum(
            1 for s in self._signals
            if s.timestamp.date() == today
        )

        trades_today = sum(
            1 for t in self._trades
            if t.entry_time.date() == today
        )

        closed_today = [
            t for t in self._trades
            if t.entry_time.date() == today
            and t.outcome in (TradeOutcome.WIN, TradeOutcome.LOSS, TradeOutcome.BREAK_EVEN)
        ]
        wins_today = sum(1 for t in closed_today if t.outcome == TradeOutcome.WIN)
        win_rate_today = wins_today / len(closed_today) if closed_today else 0.0

        errors_today = sum(
            1 for e in self._errors
            if e["timestamp"].startswith(today.isoformat())
        )

        api_latencies = {
            api: (
                sum(samples) / len(samples) if samples else 0.0
            )
            for api, samples in self._latencies.items()
        }

        uptime = time.monotonic() - self._start_time

        return {
            "signals_today": signals_today,
            "trades_today": trades_today,
            "win_rate_today": round(win_rate_today, 4),
            "errors_today": errors_today,
            "api_latencies": {k: round(v, 2) for k, v in api_latencies.items()},
            "uptime_seconds": round(uptime, 1),
            "last_price_update": (
                self._last_price_update.isoformat()
                if self._last_price_update else None
            ),
            "last_scan_time": (
                self._last_scan_time.isoformat()
                if self._last_scan_time else None
            ),
            "total_signals": len(self._signals),
            "total_trades": len(self._trades),
            "total_errors": len(self._errors),
        }

    def get_recent_signals(self, n: int = 10) -> List[TradeSignal]:
        """
        Return the *n* most recently recorded signals (newest first).

        Parameters
        ----------
        n:
            Maximum number of signals to return.
        """
        signals = list(self._signals)
        signals.reverse()
        return signals[:n]

    def get_error_summary(self) -> Dict[str, Any]:
        """
        Return an aggregated error summary.

        Returns
        -------
        dict with keys:
            total_errors      — int
            error_types       — {ErrorType: count}
            recent_errors     — last 5 error dicts
        """
        total = len(self._errors)
        type_counts: Dict[str, int] = defaultdict(int)
        for e in self._errors:
            type_counts[e.get("error_type", "Unknown")] += 1

        recent = list(self._errors)
        recent.reverse()

        return {
            "total_errors": total,
            "error_types": dict(type_counts),
            "recent_errors": recent[:5],
        }

    def get_uptime(self) -> float:
        """Return uptime in seconds since MetricsCollector was created."""
        return time.monotonic() - self._start_time

    def get_trade_count_by_outcome(self) -> Dict[str, int]:
        """Return trade counts grouped by outcome."""
        counts: Dict[str, int] = defaultdict(int)
        for t in self._trades:
            counts[t.outcome.value] += 1
        return dict(counts)
