"""
Tests for SignalGenerator and ConfidenceScorer.

These tests exercise the full signal-generation pipeline with synthetic
market data crafted to trigger (or not trigger) specific conditions.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Dict, List

import pytest

from src.core.models import (
    Direction,
    IndicatorSet,
    Market,
    MarketCondition,
    MarketConditionType,
    PolymarketToken,
    TimeframeScore,
    VolatilityRegime,
)
from src.trading.scoring import ConfidenceScorer
from src.trading.signals import SignalGenerator


# ── Builders ──────────────────────────────────────────────────────────────────

def _make_market(
    yes_price: float = 0.50,
    no_price: float = 0.50,
    hours_to_resolution: float = 24.0,
    question: str = "Will BTC be above $95,000 by March 31?",
) -> Market:
    end_date = datetime.utcnow() + timedelta(hours=hours_to_resolution)
    return Market(
        condition_id="test-market-" + str(uuid.uuid4())[:8],
        question=question,
        tokens=[
            PolymarketToken(token_id="yes-tok", outcome="Yes", price=yes_price),
            PolymarketToken(token_id="no-tok", outcome="No", price=no_price),
        ],
        end_date=end_date,
        active=True,
        volume=200_000.0,
        liquidity=10_000.0,
    )


def _bullish_indicators(close: float = 95_000.0) -> IndicatorSet:
    """IndicatorSet with a perfect bullish EMA stack, RSI in zone, positive MACD."""
    return IndicatorSet(
        timestamp=datetime.utcnow(),
        timeframe="1h",
        close=close,
        ema_20=close * 0.994,    # 20 > 50 > 200
        ema_50=close * 0.980,
        ema_200=close * 0.920,
        rsi=62.0,                # in [50, 70] bullish zone
        macd=450.0,
        macd_signal=380.0,
        macd_histogram=70.0,     # positive → bullish
        atr=1_200.0,
        atr_pct=0.013,
        volume_ma=350.0,
        volume_ratio=1.5,        # above-average volume
        momentum=4.2,            # positive
    )


def _bearish_indicators(close: float = 85_000.0) -> IndicatorSet:
    """IndicatorSet with a perfect bearish EMA stack, RSI in bearish zone."""
    return IndicatorSet(
        timestamp=datetime.utcnow(),
        timeframe="1h",
        close=close,
        ema_20=close * 1.006,    # 20 < 50 < 200 reversed
        ema_50=close * 1.020,
        ema_200=close * 1.080,
        rsi=38.0,                # in [30, 50] bearish zone
        macd=-350.0,
        macd_signal=-280.0,
        macd_histogram=-70.0,    # negative → bearish
        atr=1_100.0,
        atr_pct=0.013,
        volume_ma=300.0,
        volume_ratio=1.4,
        momentum=-3.8,           # negative
    )


def _choppy_indicators(close: float = 90_000.0) -> IndicatorSet:
    """IndicatorSet where EMAs are tangled and RSI is neutral (50)."""
    return IndicatorSet(
        timestamp=datetime.utcnow(),
        timeframe="1h",
        close=close,
        ema_20=close * 1.001,   # barely above 50
        ema_50=close * 0.999,   # barely below 20
        ema_200=close * 0.998,
        rsi=52.0,               # neutral
        macd=10.0,
        macd_signal=8.0,
        macd_histogram=2.0,
        atr=900.0,
        atr_pct=0.010,
        volume_ma=200.0,
        volume_ratio=0.95,
        momentum=0.3,
    )


def _bullish_tf_scores(n: int = 4) -> Dict[str, TimeframeScore]:
    tfs = ["1h", "4h", "15m", "5m"]
    return {
        tf: TimeframeScore(
            timeframe=tf,
            trend_score=0.8,
            momentum_score=0.7,
            volume_score=0.9,
            overall_score=-0.75,  # negative = bullish in this convention
        )
        for tf in tfs[:n]
    }


def _bearish_tf_scores(n: int = 4) -> Dict[str, TimeframeScore]:
    tfs = ["1h", "4h", "15m", "5m"]
    return {
        tf: TimeframeScore(
            timeframe=tf,
            trend_score=-0.8,
            momentum_score=-0.7,
            volume_score=0.9,
            overall_score=0.75,  # positive = bearish in this convention
        )
        for tf in tfs[:n]
    }


def _neutral_tf_scores() -> Dict[str, TimeframeScore]:
    return {
        "1h": TimeframeScore(
            timeframe="1h",
            trend_score=0.05,
            momentum_score=0.02,
            volume_score=0.4,
            overall_score=0.0,   # neutral
        ),
    }


def _make_market_condition(
    condition: MarketConditionType = MarketConditionType.TRENDING_UP,
    vol_regime: VolatilityRegime = VolatilityRegime.MEDIUM,
    strength: float = 0.7,
) -> MarketCondition:
    return MarketCondition(
        condition=condition,
        trend_direction="up" if condition == MarketConditionType.TRENDING_UP else (
            "down" if condition == MarketConditionType.TRENDING_DOWN else None
        ),
        volatility_regime=vol_regime,
        trend_strength=strength,
        confidence=strength,
        timestamp=datetime.utcnow(),
    )


# ── No-signal conditions ──────────────────────────────────────────────────────

class TestNoSignalInChoppyMarket:
    def test_choppy_condition_returns_none(self, mock_settings):
        """
        SignalGenerator must return None when market_condition is CHOPPY,
        regardless of technical indicators.
        """
        gen = SignalGenerator(mock_settings)
        market = _make_market(yes_price=0.45, no_price=0.55)
        indicators = _bullish_indicators()
        choppy_condition = _make_market_condition(MarketConditionType.CHOPPY)
        tf_scores = _bullish_tf_scores()

        signal = gen.generate(
            market=market,
            indicators_1h=indicators,
            timeframe_scores=tf_scores,
            market_condition=choppy_condition,
            btc_price=95_000.0,
            fair_value=0.62,
        )
        assert signal is None, "Expected no signal in CHOPPY market"

    def test_choppy_with_neutral_tfs_returns_none(self, mock_settings):
        gen = SignalGenerator(mock_settings)
        market = _make_market()
        indicators = _choppy_indicators()
        choppy_condition = _make_market_condition(MarketConditionType.CHOPPY)
        tf_scores = _neutral_tf_scores()

        signal = gen.generate(
            market=market,
            indicators_1h=indicators,
            timeframe_scores=tf_scores,
            market_condition=choppy_condition,
            btc_price=90_000.0,
            fair_value=0.50,
        )
        assert signal is None


class TestNoSignalExtremeVolatility:
    def test_extreme_atr_returns_none(self, mock_settings):
        """
        When volatility regime is EXTREME, SignalGenerator must veto all signals.
        """
        gen = SignalGenerator(mock_settings)
        market = _make_market(yes_price=0.42)
        indicators = _bullish_indicators()
        extreme_condition = _make_market_condition(
            MarketConditionType.TRENDING_UP, VolatilityRegime.EXTREME
        )
        tf_scores = _bullish_tf_scores()

        signal = gen.generate(
            market=market,
            indicators_1h=indicators,
            timeframe_scores=tf_scores,
            market_condition=extreme_condition,
            btc_price=95_000.0,
            fair_value=0.65,
        )
        assert signal is None, "Expected no signal under EXTREME volatility"

    def test_extreme_vol_blocks_even_high_edge(self, mock_settings):
        """Even a very large edge should not overcome EXTREME volatility veto."""
        gen = SignalGenerator(mock_settings)
        market = _make_market(yes_price=0.20, no_price=0.80)  # huge apparent edge
        indicators = _bullish_indicators()
        extreme_condition = _make_market_condition(
            MarketConditionType.TRENDING_UP, VolatilityRegime.EXTREME
        )

        signal = gen.generate(
            market=market,
            indicators_1h=indicators,
            timeframe_scores=_bullish_tf_scores(),
            market_condition=extreme_condition,
            btc_price=95_000.0,
            fair_value=0.85,
        )
        assert signal is None


class TestNoSignalLowEdge:
    def test_edge_below_minimum_returns_none(self, mock_settings):
        """
        When the Polymarket edge is below the minimum (0.03), no signal should
        be generated even if all technical conditions are perfect.
        """
        gen = SignalGenerator(mock_settings)
        # fair_value = 0.52, yes_price = 0.52 → edge = 0.00 (no edge)
        market = _make_market(yes_price=0.52, no_price=0.48)
        indicators = _bullish_indicators()
        condition = _make_market_condition()
        tf_scores = _bullish_tf_scores()

        signal = gen.generate(
            market=market,
            indicators_1h=indicators,
            timeframe_scores=tf_scores,
            market_condition=condition,
            btc_price=95_000.0,
            fair_value=0.52,  # = implied → no edge
        )
        assert signal is None, "Expected no signal when edge = 0"


# ── Signal generation ─────────────────────────────────────────────────────────

class TestLongSignalStrongTrend:
    def test_bullish_setup_generates_yes_signal(self, mock_settings):
        """
        Perfect bullish alignment (EMA stack, RSI in zone, positive MACD,
        volume confirmed, multi-TF agreement) with a meaningful Polymarket edge
        must produce a YES signal.
        """
        gen = SignalGenerator(mock_settings)
        # yes_price=0.48 while fair_value=0.65 → edge = 0.17
        market = _make_market(yes_price=0.48, no_price=0.52)
        indicators = _bullish_indicators()
        condition = _make_market_condition()
        tf_scores = _bullish_tf_scores()

        signal = gen.generate(
            market=market,
            indicators_1h=indicators,
            timeframe_scores=tf_scores,
            market_condition=condition,
            btc_price=95_000.0,
            fair_value=0.65,
        )

        assert signal is not None, "Expected a YES signal on strong bullish setup"
        assert signal.direction == Direction.YES
        assert signal.confidence >= mock_settings.min_confidence_threshold
        assert signal.edge > 0

    def test_signal_includes_reasons(self, mock_settings):
        """Generated signals must include non-empty reason list."""
        gen = SignalGenerator(mock_settings)
        market = _make_market(yes_price=0.44, no_price=0.56)
        indicators = _bullish_indicators()
        condition = _make_market_condition()
        tf_scores = _bullish_tf_scores()

        signal = gen.generate(
            market=market,
            indicators_1h=indicators,
            timeframe_scores=tf_scores,
            market_condition=condition,
            btc_price=95_000.0,
            fair_value=0.65,
        )
        if signal is not None:
            assert len(signal.reasons) > 0


class TestShortSignalStrongDowntrend:
    def test_bearish_setup_generates_no_signal(self, mock_settings):
        """
        Perfect bearish alignment with a Polymarket NO edge must produce
        a NO (short) signal.
        """
        gen = SignalGenerator(mock_settings)
        # no_price=0.42, fair_value_yes=0.30 → no_fair = 0.70, no_edge = 0.28
        market = _make_market(yes_price=0.58, no_price=0.42)
        indicators = _bearish_indicators()
        condition = _make_market_condition(
            MarketConditionType.TRENDING_DOWN, VolatilityRegime.MEDIUM
        )
        tf_scores = _bearish_tf_scores()

        signal = gen.generate(
            market=market,
            indicators_1h=indicators,
            timeframe_scores=tf_scores,
            market_condition=condition,
            btc_price=85_000.0,
            fair_value=0.30,   # YES fair value low → NO is attractive
        )

        assert signal is not None, "Expected a NO signal on bearish setup"
        assert signal.direction == Direction.NO
        assert signal.confidence >= mock_settings.min_confidence_threshold


# ── Confidence scorer ─────────────────────────────────────────────────────────

class TestConfidenceScoreRange:
    def test_confidence_always_0_to_100(self, mock_settings):
        """ConfidenceScorer.score() must always return a value in [0, 100]."""
        scorer = ConfidenceScorer()

        # Test a wide variety of component combinations
        test_cases = [
            {"trend_alignment": 1.0, "momentum": 1.0, "volume": 1.0,
             "rsi_position": 1.0, "timeframe_agreement": 1.0, "edge_magnitude": 1.0,
             "volatility_penalty": 0.0, "choppy_penalty": 0.0, "market_condition_bonus": 0.0},
            {"trend_alignment": 0.0, "momentum": 0.0, "volume": 0.0,
             "rsi_position": 0.0, "timeframe_agreement": 0.0, "edge_magnitude": 0.0,
             "volatility_penalty": 1.0, "choppy_penalty": 1.0, "market_condition_bonus": 0.0},
            {"trend_alignment": 0.5, "momentum": 0.3, "volume": 0.8,
             "rsi_position": 0.6, "timeframe_agreement": 0.7, "edge_magnitude": 0.4,
             "volatility_penalty": 0.3, "choppy_penalty": 0.0, "market_condition_bonus": 0.0},
        ]

        for components in test_cases:
            score = scorer.score(components)
            assert 0.0 <= score <= 100.0, (
                f"Score {score} out of range [0, 100] for components {components}"
            )

    def test_perfect_components_max_score(self):
        """All components at 1.0 with no penalties → score near 100."""
        scorer = ConfidenceScorer()
        components = {
            "trend_alignment": 1.0, "momentum": 1.0, "volume": 1.0,
            "rsi_position": 1.0, "timeframe_agreement": 1.0, "edge_magnitude": 1.0,
            "volatility_penalty": 0.0, "choppy_penalty": 0.0, "market_condition_bonus": 0.0,
        }
        score = scorer.score(components)
        assert score == 100.0, f"Perfect components should yield 100.0, got {score}"

    def test_all_zero_components_min_score(self):
        """All components at 0.0 with max penalties → score = 0."""
        scorer = ConfidenceScorer()
        components = {
            "trend_alignment": 0.0, "momentum": 0.0, "volume": 0.0,
            "rsi_position": 0.0, "timeframe_agreement": 0.0, "edge_magnitude": 0.0,
            "volatility_penalty": 1.0, "choppy_penalty": 1.0, "market_condition_bonus": 0.0,
        }
        score = scorer.score(components)
        assert score == 0.0, f"Zero components + max penalties should yield 0.0, got {score}"


class TestConfidenceIncreasesWithAlignment:
    def test_more_aligned_tfs_higher_confidence(self, mock_settings, sample_indicators):
        """
        Adding more timeframes in alignment should monotonically increase
        the timeframe_agreement component and thus the overall confidence.
        """
        scorer = ConfidenceScorer()
        mc = _make_market_condition()

        scores_by_n: list[float] = []
        for n in [1, 2, 3, 4]:
            tf_scores = _bullish_tf_scores(n)
            components = scorer.compute_components(
                indicators=sample_indicators,
                timeframe_scores=tf_scores,
                direction=Direction.YES,
                market_condition=mc,
                edge=0.10,
            )
            scores_by_n.append(scorer.score(components))

        # More aligned timeframes → at least as high or higher confidence
        for i in range(len(scores_by_n) - 1):
            assert scores_by_n[i] <= scores_by_n[i + 1], (
                f"Confidence should not decrease with more TF alignment: "
                f"{scores_by_n}"
            )

    def test_higher_edge_higher_confidence(self, mock_settings, sample_indicators):
        """Larger Polymarket edge should produce a higher confidence score."""
        scorer = ConfidenceScorer()
        mc = _make_market_condition()
        tf_scores = _bullish_tf_scores()

        low_edge_components = scorer.compute_components(
            indicators=sample_indicators,
            timeframe_scores=tf_scores,
            direction=Direction.YES,
            market_condition=mc,
            edge=0.04,
        )
        high_edge_components = scorer.compute_components(
            indicators=sample_indicators,
            timeframe_scores=tf_scores,
            direction=Direction.YES,
            market_condition=mc,
            edge=0.18,
        )

        low_score = scorer.score(low_edge_components)
        high_score = scorer.score(high_edge_components)

        assert high_score > low_score, (
            f"Higher edge should yield higher score. "
            f"Got low={low_score:.1f}, high={high_score:.1f}"
        )
