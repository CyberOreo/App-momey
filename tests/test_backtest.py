"""
Tests for BacktestEngine (src/analytics/backtest.py).

Verifies correct metric computation, equity curve behavior, and
trade injection for the backtesting pipeline.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List

import pytest

from src.analytics.backtest import BacktestEngine
from src.analytics.performance import PerformanceAnalyzer
from src.core.models import (
    Candle,
    Direction,
    PerformanceMetrics,
    Trade,
    TradeOutcome,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_candles(n: int, start_price: float = 95_000.0, drift: float = 0.0001) -> List[Candle]:
    """
    Build n synthetic 1-hour candles with mild upward drift.
    Prices are deterministic so tests are reproducible.
    """
    import math, numpy as np
    rng = np.random.default_rng(99)
    start = datetime(2025, 1, 1, 0, 0, 0)
    candles = []
    price = start_price
    for i in range(n):
        log_ret = drift + 0.007 * rng.standard_normal()
        close = price * math.exp(log_ret)
        candles.append(
            Candle(
                timestamp=start + timedelta(hours=i),
                open=price,
                high=max(price, close) * 1.002,
                low=min(price, close) * 0.998,
                close=round(close, 2),
                volume=abs(rng.normal(500, 80)),
                timeframe="1h",
            )
        )
        price = close
    return candles


def _make_closed_trade(
    outcome: TradeOutcome,
    pnl: float,
    entry: datetime,
    exit_: datetime,
) -> Trade:
    return Trade(
        trade_id=str(uuid.uuid4()),
        market_id="test-market",
        condition_id="test-market",
        token_id="yes-tok",
        direction=Direction.YES,
        size=100.0,
        entry_price=0.55,
        entry_time=entry,
        confidence=70.0,
        paper_trade=True,
        exit_price=0.75 if outcome == TradeOutcome.WIN else 0.03,
        realized_pnl=pnl,
        exit_time=exit_,
        outcome=outcome,
        fees_paid=0.05,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestBacktestRunsWithoutError:
    def test_backtest_completes_on_synthetic_data(self, mock_settings):
        """
        BacktestEngine.run() should complete without raising an exception
        on 300 candles of synthetic data.
        """
        candles = _build_candles(300)
        candles_by_tf = {"1h": candles}
        markets = BacktestEngine.generate_mock_markets(n=3, current_btc_price=95_000.0)

        engine = BacktestEngine(mock_settings, initial_balance=1000.0)
        metrics = asyncio.get_event_loop().run_until_complete(
            engine.run(candles_by_tf, markets)
        )

        assert isinstance(metrics, PerformanceMetrics)

    def test_backtest_with_multiple_timeframes(self, mock_settings):
        """BacktestEngine should handle multiple timeframes without error."""
        candles_1h = _build_candles(300)
        candles_4h = _build_candles(300, drift=0.0001)
        candles_by_tf = {"1h": candles_1h, "4h": candles_4h}
        markets = BacktestEngine.generate_mock_markets(n=2)

        engine = BacktestEngine(mock_settings, initial_balance=1000.0)
        metrics = asyncio.get_event_loop().run_until_complete(
            engine.run(candles_by_tf, markets)
        )

        assert isinstance(metrics, PerformanceMetrics)

    def test_backtest_returns_correct_type(self, mock_settings):
        """get_results() should return a dict with expected keys."""
        candles = _build_candles(300)
        candles_by_tf = {"1h": candles}
        markets = BacktestEngine.generate_mock_markets(n=2)

        engine = BacktestEngine(mock_settings, initial_balance=1000.0)
        asyncio.get_event_loop().run_until_complete(
            engine.run(candles_by_tf, markets)
        )

        results = engine.get_results()
        assert "metrics" in results
        assert "equity_curve" in results
        assert "trades" in results
        assert "parameters" in results


class TestEquityCurveStartsAtInitial:
    def test_first_equity_point_equals_initial_balance(self, mock_settings):
        """The first equity curve point must equal the initial_balance."""
        initial = 1000.0
        candles = _build_candles(300)
        candles_by_tf = {"1h": candles}
        markets = BacktestEngine.generate_mock_markets(n=2)

        engine = BacktestEngine(mock_settings, initial_balance=initial)
        asyncio.get_event_loop().run_until_complete(
            engine.run(candles_by_tf, markets)
        )

        equity_curve = engine.get_results()["equity_curve"]
        assert len(equity_curve) > 0

        first_balance = equity_curve[0][1]
        assert abs(first_balance - initial) < 0.01, (
            f"First equity point {first_balance:.2f} != initial {initial:.2f}"
        )

    def test_equity_curve_is_monotone_timestamps(self, mock_settings):
        """Equity curve timestamps should be non-decreasing."""
        candles = _build_candles(300)
        candles_by_tf = {"1h": candles}
        markets = BacktestEngine.generate_mock_markets(n=2)

        engine = BacktestEngine(mock_settings, initial_balance=1000.0)
        asyncio.get_event_loop().run_until_complete(
            engine.run(candles_by_tf, markets)
        )

        equity_curve = engine.get_results()["equity_curve"]
        timestamps = [ts for ts, _ in equity_curve]
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1], (
                f"Equity curve timestamp regression at index {i}"
            )


class TestNoTradesNoDrawdown:
    def test_no_trades_balance_unchanged(self, mock_settings):
        """
        When no signals are generated (e.g. very few candles → no indicators),
        the final balance should equal the initial balance.
        """
        # Only 100 candles → not enough for indicators → no trades
        candles = _build_candles(150)
        candles_by_tf = {"1h": candles}
        markets = BacktestEngine.generate_mock_markets(n=2)

        engine = BacktestEngine(mock_settings, initial_balance=1000.0)
        asyncio.get_event_loop().run_until_complete(
            engine.run(candles_by_tf, markets)
        )

        results = engine.get_results()
        trades = results["trades"]

        # If no signals generated, equity should not have dropped
        equity_curve = results["equity_curve"]
        if len(equity_curve) > 1:
            final_balance = equity_curve[-1][1]
            # Allow $5 tolerance for any fees on very few trades
            initial_if_no_trades = 1000.0
            # If no trades were made, balance == initial
            if len(trades) == 0:
                assert abs(final_balance - initial_if_no_trades) < 1.0

    def test_metrics_zero_trades_is_valid(self, mock_settings):
        """PerformanceMetrics with zero trades must be valid (not error)."""
        analyzer = PerformanceAnalyzer()
        metrics = analyzer.compute_metrics([], initial_balance=1000.0)

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_trades == 0
        assert metrics.win_rate == 0.0
        assert metrics.total_pnl == 0.0
        assert metrics.max_drawdown == 0.0


class TestMetricsComputation:
    def test_metrics_valid_structure(self, mock_settings):
        """BacktestEngine.run() metrics should have all expected fields."""
        candles = _build_candles(300)
        candles_by_tf = {"1h": candles}
        markets = BacktestEngine.generate_mock_markets(n=3)

        engine = BacktestEngine(mock_settings, initial_balance=1000.0)
        metrics = asyncio.get_event_loop().run_until_complete(
            engine.run(candles_by_tf, markets)
        )

        # All numeric fields should be real numbers (not NaN or inf)
        import math
        assert not math.isnan(metrics.win_rate)
        assert not math.isnan(metrics.total_pnl)
        assert not math.isnan(metrics.max_drawdown_pct)
        assert not math.isnan(metrics.sharpe_ratio)
        assert not math.isnan(metrics.sortino_ratio)
        assert not math.isnan(metrics.calmar_ratio)
        assert not math.isnan(metrics.avg_holding_time_hours)

    def test_metrics_start_end_dates(self, mock_settings):
        """
        When trades exist, start_date <= end_date in PerformanceMetrics.
        """
        entry_base = datetime(2025, 1, 1, 8, 0, 0)
        trades = [
            _make_closed_trade(
                TradeOutcome.WIN, 30.0,
                entry_base, entry_base + timedelta(hours=12)
            ),
            _make_closed_trade(
                TradeOutcome.LOSS, -15.0,
                entry_base + timedelta(hours=24),
                entry_base + timedelta(hours=36)
            ),
        ]

        analyzer = PerformanceAnalyzer()
        metrics = analyzer.compute_metrics(trades, initial_balance=1000.0)
        assert metrics.start_date <= metrics.end_date

    def test_performance_analyzer_profit_factor(self):
        """Profit factor = gross_profit / gross_loss."""
        entry = datetime(2025, 1, 1, 8)
        trades = [
            _make_closed_trade(TradeOutcome.WIN, 60.0, entry, entry + timedelta(hours=6)),
            _make_closed_trade(TradeOutcome.WIN, 40.0, entry + timedelta(hours=8), entry + timedelta(hours=16)),
            _make_closed_trade(TradeOutcome.LOSS, -20.0, entry + timedelta(hours=20), entry + timedelta(hours=28)),
        ]

        analyzer = PerformanceAnalyzer()
        metrics = analyzer.compute_metrics(trades, initial_balance=1000.0)

        # gross_profit = 100, gross_loss = 20, PF = 5.0
        assert abs(metrics.profit_factor - 5.0) < 0.01, (
            f"Expected profit_factor=5.0, got {metrics.profit_factor}"
        )


class TestWinRateCalculation:
    def test_three_wins_one_loss_win_rate_75_pct(self):
        """
        Manually injecting 3 WIN and 1 LOSS trade must produce win_rate = 0.75.
        """
        entry = datetime(2025, 2, 1, 8)
        trades = [
            _make_closed_trade(TradeOutcome.WIN, 25.0, entry, entry + timedelta(hours=8)),
            _make_closed_trade(TradeOutcome.WIN, 30.0, entry + timedelta(hours=10), entry + timedelta(hours=18)),
            _make_closed_trade(TradeOutcome.WIN, 20.0, entry + timedelta(hours=20), entry + timedelta(hours=28)),
            _make_closed_trade(TradeOutcome.LOSS, -18.0, entry + timedelta(hours=32), entry + timedelta(hours=40)),
        ]

        analyzer = PerformanceAnalyzer()
        metrics = analyzer.compute_metrics(trades, initial_balance=1000.0)

        assert metrics.total_trades == 4
        assert metrics.winning_trades == 3
        assert metrics.losing_trades == 1
        assert abs(metrics.win_rate - 0.75) < 1e-9, (
            f"Expected win_rate=0.75, got {metrics.win_rate}"
        )

    def test_all_wins_win_rate_100(self):
        entry = datetime(2025, 2, 1, 8)
        trades = [
            _make_closed_trade(TradeOutcome.WIN, 10.0, entry + timedelta(hours=i * 4), entry + timedelta(hours=i * 4 + 3))
            for i in range(5)
        ]
        analyzer = PerformanceAnalyzer()
        metrics = analyzer.compute_metrics(trades, initial_balance=1000.0)
        assert metrics.win_rate == 1.0

    def test_all_losses_win_rate_0(self):
        entry = datetime(2025, 2, 1, 8)
        trades = [
            _make_closed_trade(TradeOutcome.LOSS, -10.0, entry + timedelta(hours=i * 4), entry + timedelta(hours=i * 4 + 3))
            for i in range(5)
        ]
        analyzer = PerformanceAnalyzer()
        metrics = analyzer.compute_metrics(trades, initial_balance=1000.0)
        assert metrics.win_rate == 0.0

    def test_open_trades_excluded_from_win_rate(self):
        """OPEN trades must not be counted in win_rate calculation."""
        entry = datetime(2025, 2, 1, 8)
        closed_win = _make_closed_trade(
            TradeOutcome.WIN, 15.0, entry, entry + timedelta(hours=6)
        )
        open_trade = Trade(
            trade_id=str(uuid.uuid4()),
            market_id="test",
            condition_id="test",
            token_id="yes",
            direction=Direction.YES,
            size=100.0,
            entry_price=0.50,
            entry_time=entry + timedelta(hours=12),
            confidence=70.0,
            outcome=TradeOutcome.OPEN,
        )

        analyzer = PerformanceAnalyzer()
        metrics = analyzer.compute_metrics([closed_win, open_trade], initial_balance=1000.0)

        assert metrics.total_trades == 1  # only 1 closed trade counted
        assert metrics.win_rate == 1.0

    def test_simulate_resolution_yes_wins_when_btc_above_target(self, mock_settings):
        """simulate_resolution should return ~0.97 for YES when BTC > target."""
        engine = BacktestEngine(mock_settings, initial_balance=1000.0)
        trade = Trade(
            trade_id="t1", market_id="m1", condition_id="m1",
            token_id="yes", direction=Direction.YES,
            size=100, entry_price=0.55, entry_time=datetime.utcnow(),
            confidence=70.0,
        )
        price = engine.simulate_resolution(
            trade,
            btc_price_at_resolution=100_000.0,
            market_question="Will BTC be above $95,000 by March 31?",
        )
        assert price >= 0.90, f"YES position should win when BTC > target, got {price}"

    def test_simulate_resolution_yes_loses_when_btc_below_target(self, mock_settings):
        """simulate_resolution should return ~0.03 for YES when BTC < target."""
        engine = BacktestEngine(mock_settings, initial_balance=1000.0)
        trade = Trade(
            trade_id="t1", market_id="m1", condition_id="m1",
            token_id="yes", direction=Direction.YES,
            size=100, entry_price=0.55, entry_time=datetime.utcnow(),
            confidence=70.0,
        )
        price = engine.simulate_resolution(
            trade,
            btc_price_at_resolution=80_000.0,
            market_question="Will BTC be above $95,000 by March 31?",
        )
        assert price <= 0.10, f"YES position should lose when BTC < target, got {price}"
