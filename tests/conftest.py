"""
Shared pytest fixtures for the PolyBTC Trader test suite.

All fixtures produce deterministic synthetic data so tests are
reproducible and do not require any external API connections.
"""
from __future__ import annotations

import math
import uuid
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pytest

from src.core.config import Settings
from src.core.models import (
    Candle,
    Direction,
    IndicatorSet,
    Market,
    MarketCondition,
    MarketConditionType,
    PolymarketToken,
    RiskState,
    TimeframeScore,
    Trade,
    TradeOutcome,
    VolatilityRegime,
)


# ── Candle helpers ────────────────────────────────────────────────────────────

def _make_candles(
    n: int,
    start_price: float,
    drift: float,
    volatility: float,
    start_dt: datetime,
    interval_minutes: int = 60,
    timeframe: str = "1h",
    seed: int = 42,
) -> List[Candle]:
    """
    Generate synthetic BTCUSDT OHLCV candles with a geometric random walk.

    Parameters
    ----------
    n:
        Number of candles.
    start_price:
        Initial close price.
    drift:
        Per-bar drift in log-return units (positive = uptrend).
    volatility:
        Per-bar log-return standard deviation.
    start_dt:
        Timestamp of the first candle.
    interval_minutes:
        Bar duration in minutes.
    timeframe:
        Candle timeframe label.
    seed:
        RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    candles: List[Candle] = []
    price = start_price

    for i in range(n):
        ts = start_dt + timedelta(minutes=i * interval_minutes)
        log_ret = drift + volatility * rng.standard_normal()
        close = price * math.exp(log_ret)

        # Realistic OHLC within the bar
        bar_range = abs(close - price) * (1.0 + abs(rng.standard_normal()) * 0.5)
        high = max(price, close) + bar_range * 0.3 * abs(rng.standard_normal())
        low = min(price, close) - bar_range * 0.3 * abs(rng.standard_normal())
        open_ = price * math.exp(rng.standard_normal() * volatility * 0.3)

        # Clamp to avoid negative prices
        high = max(high, max(open_, close) * 1.001)
        low = min(low, min(open_, close) * 0.999)
        low = max(low, 1.0)

        volume = abs(rng.normal(500.0, 100.0)) * (close / start_price)

        candles.append(
            Candle(
                timestamp=ts,
                open=round(open_, 2),
                high=round(high, 2),
                low=round(low, 2),
                close=round(close, 2),
                volume=round(volume, 4),
                timeframe=timeframe,
            )
        )
        price = close

    return candles


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_candles() -> List[Candle]:
    """
    300 hourly BTCUSDT candles trending upward from $90,000.

    Drift is +0.0002 per bar (≈ +5 % over 300 bars) with moderate volatility.
    """
    start_dt = datetime(2025, 1, 1, 0, 0, 0)
    return _make_candles(
        n=300,
        start_price=90_000.0,
        drift=0.0002,
        volatility=0.008,
        start_dt=start_dt,
        interval_minutes=60,
        timeframe="1h",
        seed=42,
    )


@pytest.fixture
def sample_candles_bearish() -> List[Candle]:
    """
    300 hourly BTCUSDT candles trending downward from $95,000.

    Drift is -0.0003 per bar with moderate volatility.
    """
    start_dt = datetime(2025, 1, 1, 0, 0, 0)
    return _make_candles(
        n=300,
        start_price=95_000.0,
        drift=-0.0003,
        volatility=0.009,
        start_dt=start_dt,
        interval_minutes=60,
        timeframe="1h",
        seed=7,
    )


@pytest.fixture
def sample_indicators() -> IndicatorSet:
    """
    An IndicatorSet with realistic bullish values for a BTC at ~$95,000.

    All numeric values are consistent (EMA stack bullish, RSI in zone, etc.)
    """
    return IndicatorSet(
        timestamp=datetime(2025, 3, 1, 12, 0, 0),
        timeframe="1h",
        close=95_000.0,
        ema_20=94_500.0,
        ema_50=93_000.0,
        ema_200=88_000.0,
        rsi=62.0,
        macd=450.0,
        macd_signal=380.0,
        macd_histogram=70.0,
        atr=1_200.0,
        atr_pct=0.013,
        volume_ma=350.0,
        volume_ratio=1.45,
        momentum=3.5,
    )


@pytest.fixture
def sample_market() -> Market:
    """
    A Polymarket market: "Will BTC be above $95,000 by March 31?"
    with YES price ≈ 0.52, 24 h to resolution.
    """
    now = datetime.utcnow()
    return Market(
        condition_id="btc-above-95000-20250331",
        question="Will BTC be above $95,000 by March 31, 2025?",
        tokens=[
            PolymarketToken(
                token_id="yes-btc-95k",
                outcome="Yes",
                price=0.52,
            ),
            PolymarketToken(
                token_id="no-btc-95k",
                outcome="No",
                price=0.48,
            ),
        ],
        end_date=now + timedelta(hours=24),
        active=True,
        volume=150_000.0,
        liquidity=8_000.0,
    )


@pytest.fixture
def sample_trade() -> Trade:
    """
    A completed WIN trade.

    Entry: 0.52, Exit: 0.78, PnL: +$52, holding 12 h.
    """
    entry = datetime(2025, 3, 1, 8, 0, 0)
    exit_ = datetime(2025, 3, 1, 20, 0, 0)
    trade_id = str(uuid.uuid4())

    return Trade(
        trade_id=trade_id,
        market_id="btc-above-95000-20250331",
        condition_id="btc-above-95000-20250331",
        token_id="yes-btc-95k",
        direction=Direction.YES,
        size=200.0,          # 200 tokens
        entry_price=0.52,
        entry_time=entry,
        confidence=72.0,
        signal_reasons=["BTC above EMA-200", "MACD bullish", "Volume confirmed"],
        paper_trade=True,
        exit_price=0.78,
        realized_pnl=52.0,
        exit_time=exit_,
        outcome=TradeOutcome.WIN,
        fees_paid=0.104,
    )


@pytest.fixture
def sample_trade_loss() -> Trade:
    """
    A completed LOSS trade.

    Entry: 0.55, Exit: 0.03 (market resolved against us), PnL: -$52.
    """
    entry = datetime(2025, 3, 2, 9, 0, 0)
    exit_ = datetime(2025, 3, 3, 9, 0, 0)
    trade_id = str(uuid.uuid4())

    return Trade(
        trade_id=trade_id,
        market_id="btc-above-98000-20250310",
        condition_id="btc-above-98000-20250310",
        token_id="yes-btc-98k",
        direction=Direction.YES,
        size=100.0,
        entry_price=0.55,
        entry_time=entry,
        confidence=66.0,
        signal_reasons=["RSI bullish zone", "MACD positive"],
        paper_trade=True,
        exit_price=0.03,
        realized_pnl=-52.0,
        exit_time=exit_,
        outcome=TradeOutcome.LOSS,
        fees_paid=0.055,
    )


@pytest.fixture
def mock_settings() -> Settings:
    """
    Settings with paper trading enabled and relaxed thresholds for unit tests.

    Key overrides:
        - paper_trading = True
        - min_confidence_threshold = 50.0 (lower than production 65)
        - require_volume_confirmation = False
        - min_position_usdc = 5.0
        - max_position_usdc = 100.0
    """
    return Settings(
        paper_trading=True,
        min_confidence_threshold=50.0,
        require_volume_confirmation=False,
        rsi_long_min=45.0,
        rsi_long_max=75.0,
        rsi_short_min=25.0,
        rsi_short_max=55.0,
        min_volume_ratio=1.0,
        min_position_usdc=5.0,
        max_position_usdc=100.0,
        kelly_fraction=0.25,
        use_kelly=True,
        max_risk_per_trade_pct=0.02,
        max_daily_drawdown_pct=0.05,
        max_consecutive_losses=3,
        cooldown_minutes=30,
        max_open_positions=3,
        max_total_exposure_pct=0.20,
        min_time_to_resolution_hours=1.0,
        max_time_to_resolution_hours=168.0,
        paper_balance=1000.0,
    )


@pytest.fixture
def sample_risk_state() -> RiskState:
    """
    A safe RiskState: no kill switch, no cooldown, healthy balance.
    """
    return RiskState(
        balance=1_000.0,
        start_of_day_balance=1_000.0,
        daily_pnl=0.0,
        consecutive_losses=0,
        open_positions=0,
        total_exposure_usdc=0.0,
        in_cooldown=False,
        cooldown_until=None,
        kill_switch_active=False,
        kill_switch_reason="",
    )
