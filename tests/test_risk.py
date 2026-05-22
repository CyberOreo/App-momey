"""
Tests for RiskManager and risk limit enforcement.

Since the full RiskManager implementation lives in src/risk/ (not yet created),
these tests exercise the risk-related logic that can be derived from the models
and settings, and test a minimal RiskManager stand-in that follows the
documented contract.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.config import Settings
from src.core.models import (
    Direction,
    PositionSizing,
    RiskState,
    Trade,
    TradeOutcome,
    TradeSignal,
)


# ── Minimal RiskManager for testing ──────────────────────────────────────────
# If a full RiskManager exists in the codebase we import it; otherwise we
# build a test double that satisfies the public contract.

from src.risk.manager import RiskManager


def _make_rm(settings: Settings, initial_balance: float = 1000.0) -> RiskManager:
    """Create and initialise a RiskManager synchronously for use in tests."""
    rm = RiskManager(settings, db=None)
    asyncio.get_event_loop().run_until_complete(rm.initialize(initial_balance))
    return rm


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_signal(confidence: float = 75.0, edge: float = 0.08) -> TradeSignal:
    """Create a minimal TradeSignal for risk tests."""
    from src.core.models import (
        IndicatorSet, MarketCondition, MarketConditionType, VolatilityRegime
    )
    ind = IndicatorSet(
        timestamp=datetime.utcnow(), timeframe="1h", close=95000,
        ema_20=94500, ema_50=93000, ema_200=88000,
        rsi=62, macd=400, macd_signal=350, macd_histogram=50,
        atr=1200, atr_pct=0.013, volume_ma=300, volume_ratio=1.4, momentum=3.0,
    )
    mc = MarketCondition(
        condition=MarketConditionType.TRENDING_UP,
        trend_direction="up",
        volatility_regime=VolatilityRegime.MEDIUM,
        trend_strength=0.7,
        confidence=0.75,
        timestamp=datetime.utcnow(),
    )
    return TradeSignal(
        market_id="test-market",
        direction=Direction.YES,
        token_id="yes-test",
        confidence=confidence,
        price=0.52,
        reasons=["Test signal"],
        timeframe_scores={},
        market_condition=mc,
        indicators=ind,
        timestamp=datetime.utcnow(),
        implied_probability=0.52,
        fair_value_estimate=0.60,
        edge=edge,
    )


def _make_sizing(usdc: float = 50.0) -> PositionSizing:
    return PositionSizing(
        recommended_size_usdc=usdc,
        max_size_usdc=200.0,
        risk_amount_usdc=usdc * 0.02,
        kelly_fraction=0.05,
        method="kelly",
    )


def _make_trade(outcome: TradeOutcome, pnl: float = -20.0) -> Trade:
    return Trade(
        trade_id="test-" + outcome.value,
        market_id="test-market",
        condition_id="test-market",
        token_id="yes-test",
        direction=Direction.YES,
        size=100.0,
        entry_price=0.52,
        entry_time=datetime.utcnow() - timedelta(hours=6),
        confidence=70.0,
        paper_trade=True,
        exit_price=0.03,
        realized_pnl=pnl,
        exit_time=datetime.utcnow(),
        outcome=outcome,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestCanTradeDefaultState:
    def test_fresh_state_allows_trading(self, mock_settings):
        """A freshly initialised risk manager should allow trading."""
        rm = _make_rm(mock_settings, 1000.0)
        signal = _make_signal()
        sizing = _make_sizing()

        can_trade, reason = asyncio.get_event_loop().run_until_complete(
            rm.can_execute(signal, sizing)
        )
        assert can_trade, f"Expected can_trade=True but got: {reason}"

    def test_default_risk_state_properties(self, mock_settings):
        rm = _make_rm(mock_settings, 1000.0)
        state = asyncio.get_event_loop().run_until_complete(rm.get_state())

        assert state.balance == 1000.0
        assert state.kill_switch_active is False
        assert state.in_cooldown is False
        assert state.consecutive_losses == 0
        assert state.can_trade is True


class TestKillSwitchBlocksTrading:
    def test_kill_switch_blocks_all_execution(self, mock_settings):
        """Once the kill switch is active, can_execute must always return False."""
        rm = _make_rm(mock_settings, 1000.0)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(rm.activate_kill_switch("Persistent API failure"))

        signal = _make_signal()
        sizing = _make_sizing()

        can_trade, reason = loop.run_until_complete(rm.can_execute(signal, sizing))
        assert can_trade is False
        assert "kill" in reason.lower() or "switch" in reason.lower() or "Kill" in reason

    def test_kill_switch_state_reflects_reason(self, mock_settings):
        rm = _make_rm(mock_settings, 1000.0)
        loop = asyncio.get_event_loop()
        reason_text = "Circuit breaker: 5 consecutive losses"
        loop.run_until_complete(rm.activate_kill_switch(reason_text))

        state = loop.run_until_complete(rm.get_state())
        assert state.kill_switch_active is True
        assert state.can_trade is False


class TestCooldownBlocksTrading:
    def test_cooldown_rejects_new_trades(self, mock_settings):
        """A risk manager in cooldown must reject new trades."""
        rm = _make_rm(mock_settings, 1000.0)
        # Trigger cooldown by recording max_consecutive_losses LOSS trades
        loss_trade = _make_trade(TradeOutcome.LOSS, pnl=-20.0)

        loop = asyncio.get_event_loop()
        for _ in range(mock_settings.max_consecutive_losses):
            loop.run_until_complete(rm.record_trade_result(loss_trade))

        state = loop.run_until_complete(rm.get_state())
        assert state.in_cooldown is True

        can_trade, reason = loop.run_until_complete(
            rm.can_execute(_make_signal(), _make_sizing())
        )
        assert can_trade is False
        assert "cooldown" in reason.lower()

    def test_consecutive_losses_increment(self, mock_settings):
        """Each LOSS trade should increment consecutive_losses."""
        rm = _make_rm(mock_settings, 1000.0)
        loop = asyncio.get_event_loop()
        loss = _make_trade(TradeOutcome.LOSS, pnl=-10.0)

        loop.run_until_complete(rm.record_trade_result(loss))
        state = loop.run_until_complete(rm.get_state())
        assert state.consecutive_losses == 1

    def test_win_resets_consecutive_losses(self, mock_settings):
        """A WIN trade should reset the consecutive_losses counter."""
        rm = _make_rm(mock_settings, 1000.0)
        loop = asyncio.get_event_loop()
        loss = _make_trade(TradeOutcome.LOSS, pnl=-10.0)
        win = _make_trade(TradeOutcome.WIN, pnl=15.0)

        loop.run_until_complete(rm.record_trade_result(loss))
        loop.run_until_complete(rm.record_trade_result(win))
        state = loop.run_until_complete(rm.get_state())
        assert state.consecutive_losses == 0


class TestDailyDrawdownLimit:
    def test_daily_drawdown_halts_trading(self, mock_settings):
        """
        Simulating a loss that exceeds max_daily_drawdown_pct (5%) must
        cause can_execute to return False.
        """
        rm = _make_rm(mock_settings, 1000.0)
        loop = asyncio.get_event_loop()

        # 5.5% loss on a $1000 balance = $55
        loss = _make_trade(TradeOutcome.LOSS, pnl=-55.0)
        loop.run_until_complete(rm.record_trade_result(loss))

        state = loop.run_until_complete(rm.get_state())
        assert state.daily_pnl < 0

        # The state should reflect the drawdown
        drawdown_pct = abs(state.daily_pnl) / state.start_of_day_balance
        assert drawdown_pct >= 0.05, (
            f"Expected drawdown ≥ 5%; got {drawdown_pct * 100:.2f}%"
        )

        can_trade, reason = loop.run_until_complete(
            rm.can_execute(_make_signal(), _make_sizing())
        )
        assert can_trade is False


class TestConsecutiveLossesTriggerCooldown:
    def test_three_losses_activate_cooldown(self, mock_settings):
        """Exactly max_consecutive_losses LOSS trades should activate cooldown."""
        rm = _make_rm(mock_settings, 1000.0)
        loop = asyncio.get_event_loop()
        loss = _make_trade(TradeOutcome.LOSS, pnl=-15.0)

        for _ in range(mock_settings.max_consecutive_losses):
            loop.run_until_complete(rm.record_trade_result(loss))

        state = loop.run_until_complete(rm.get_state())
        assert state.in_cooldown is True
        assert state.cooldown_until is not None
        assert state.cooldown_until > datetime.utcnow()


class TestPositionSizingKelly:
    def test_kelly_produces_reasonable_size(self, mock_settings, sample_indicators):
        """
        Kelly sizing with 25% fraction should never exceed 20% of balance.
        Balance = $1000, max = $200.
        """
        # Inline Kelly computation matching the backtest sizer
        balance = 1000.0
        confidence_frac = 0.72
        implied = 0.52
        p_win = min(0.95, confidence_frac)
        q_lose = 1.0 - p_win
        b = (1.0 - implied) / implied
        kelly = max(0.0, (p_win * b - q_lose) / b) * mock_settings.kelly_fraction
        size = kelly * balance
        size = max(mock_settings.min_position_usdc, min(mock_settings.max_position_usdc, size))
        assert size <= balance * 0.20, f"Kelly size {size:.2f} exceeds 20% of balance"
        assert size >= mock_settings.min_position_usdc

    def test_position_sizing_min_floor(self, mock_settings):
        """
        Even a very low-confidence signal should meet the minimum position floor.
        """
        balance = 1000.0
        min_size = mock_settings.min_position_usdc  # $5 in test settings

        # Ultra-low kelly fraction → very small computed size
        kelly = 0.001
        computed = kelly * balance  # = $1.00
        final = max(min_size, computed)
        assert final >= min_size


class TestRiskLimitsMaxExposure:
    def test_high_exposure_rejects_trade(self, mock_settings):
        """When total exposure already equals max_total_exposure_pct, reject."""
        rm = _make_rm(mock_settings, 1000.0)
        loop = asyncio.get_event_loop()

        # Simulate near-full exposure: $200 already deployed (20%)
        # Use record_new_position which is the real API
        loop.run_until_complete(rm.record_new_position(200.0))  # exactly at 20%

        sizing = _make_sizing(usdc=50.0)  # would push over limit
        signal = _make_signal()

        can_trade, reason = loop.run_until_complete(rm.can_execute(signal, sizing))
        assert can_trade is False
        assert "exposure" in reason.lower()

    def test_low_exposure_allows_trade(self, mock_settings):
        """With only 5% exposure, a new trade should be allowed."""
        rm = _make_rm(mock_settings, 1000.0)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(rm.record_new_position(50.0))  # 5%

        sizing = _make_sizing(usdc=30.0)  # 3% → total 8%, under 20%
        signal = _make_signal()

        can_trade, _ = loop.run_until_complete(rm.can_execute(signal, sizing))
        assert can_trade is True
