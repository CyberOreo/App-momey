"""Signal generation from multi-timeframe technical analysis.

Combines BTC technical indicators, Polymarket fair-value estimates, and
multi-timeframe alignment to produce high-conviction TradeSignals.
Only returns a signal when all required filters pass — returns None otherwise.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from loguru import logger

from src.core.models import (
    Direction,
    IndicatorSet,
    Market,
    MarketCondition,
    MarketConditionType,
    TimeframeScore,
    TradeSignal,
    VolatilityRegime,
)
from src.trading.scoring import ConfidenceScorer

_MIN_EDGE = 0.03          # absolute minimum Polymarket mispricing
_MIN_BULLISH_TFS = 2      # at least this many timeframes must agree
_MIN_BEARISH_TFS = 2


class SignalGenerator:
    """Generate TradeSignals from technical and Polymarket data."""

    def __init__(self, settings) -> None:
        self._settings = settings
        self._scorer = ConfidenceScorer()

    # ── main entry point ──────────────────────────────────────────────────────

    def generate(
        self,
        market: Market,
        indicators_1h: IndicatorSet,
        timeframe_scores: Dict[str, TimeframeScore],
        market_condition: MarketCondition,
        btc_price: float,
        fair_value: float,
    ) -> Optional[TradeSignal]:
        """
        Evaluate a market for a tradeable signal.

        Returns a TradeSignal if all conditions pass, or None if any
        no-trade filter fires or neither direction has a valid setup.

        Parameters
        ----------
        market:
            Polymarket market object (YES / NO tokens, resolution time).
        indicators_1h:
            1-hour IndicatorSet for the primary timeframe check.
        timeframe_scores:
            Dict of timeframe label → TimeframeScore for multi-TF alignment.
        market_condition:
            Detected market condition (trend / volatility regime).
        btc_price:
            Current BTC/USD price (used for context logging).
        fair_value:
            Our model's estimate of true YES probability (0–1).
        """
        # ── global no-trade filters ───────────────────────────────────────────
        veto, veto_reason = self._global_veto(market, market_condition)
        if veto:
            logger.debug("Signal vetoed", reason=veto_reason, market=market.condition_id)
            return None

        yes_token = market.yes_token
        no_token = market.no_token
        if yes_token is None or no_token is None:
            logger.debug("Market missing YES or NO token", market=market.condition_id)
            return None

        # ── try long (YES) ────────────────────────────────────────────────────
        yes_price = yes_token.price
        yes_edge = fair_value - yes_price
        long_ok, long_reasons, long_raw_confidence = self._check_long_conditions(
            indicators_1h, timeframe_scores, market_condition
        )

        if long_ok and yes_edge >= _MIN_EDGE:
            components = self._scorer.compute_components(
                indicators=indicators_1h,
                timeframe_scores=timeframe_scores,
                direction=Direction.YES,
                market_condition=market_condition,
                edge=yes_edge,
            )
            confidence = self._scorer.score(components)

            if confidence < self._settings.min_confidence_threshold:
                logger.debug(
                    "Long signal below confidence threshold",
                    confidence=round(confidence, 1),
                    threshold=self._settings.min_confidence_threshold,
                )
            else:
                signal = TradeSignal(
                    market_id=market.condition_id,
                    direction=Direction.YES,
                    token_id=yes_token.token_id,
                    confidence=confidence,
                    price=yes_price,
                    reasons=long_reasons,
                    timeframe_scores=timeframe_scores,
                    market_condition=market_condition,
                    indicators=indicators_1h,
                    timestamp=datetime.utcnow(),
                    implied_probability=yes_price,
                    fair_value_estimate=fair_value,
                    edge=yes_edge,
                )
                logger.info(
                    "Long signal generated",
                    market=market.condition_id,
                    confidence=round(confidence, 1),
                    edge=round(yes_edge, 4),
                    reasons=long_reasons,
                )
                return signal

        # ── try short (NO) ────────────────────────────────────────────────────
        no_price = no_token.price
        # NO edge: we think YES is unlikely, so buying NO is mispriced if
        # fair_value is low and no_price is also low.
        no_fair_value = 1.0 - fair_value
        no_edge = no_fair_value - no_price
        short_ok, short_reasons, short_raw_confidence = self._check_short_conditions(
            indicators_1h, timeframe_scores, market_condition
        )

        if short_ok and no_edge >= _MIN_EDGE:
            components = self._scorer.compute_components(
                indicators=indicators_1h,
                timeframe_scores=timeframe_scores,
                direction=Direction.NO,
                market_condition=market_condition,
                edge=no_edge,
            )
            confidence = self._scorer.score(components)

            if confidence < self._settings.min_confidence_threshold:
                logger.debug(
                    "Short signal below confidence threshold",
                    confidence=round(confidence, 1),
                    threshold=self._settings.min_confidence_threshold,
                )
            else:
                signal = TradeSignal(
                    market_id=market.condition_id,
                    direction=Direction.NO,
                    token_id=no_token.token_id,
                    confidence=confidence,
                    price=no_price,
                    reasons=short_reasons,
                    timeframe_scores=timeframe_scores,
                    market_condition=market_condition,
                    indicators=indicators_1h,
                    timestamp=datetime.utcnow(),
                    implied_probability=no_price,
                    fair_value_estimate=no_fair_value,
                    edge=no_edge,
                )
                logger.info(
                    "Short signal generated",
                    market=market.condition_id,
                    confidence=round(confidence, 1),
                    edge=round(no_edge, 4),
                    reasons=short_reasons,
                )
                return signal

        logger.debug(
            "No valid signal",
            market=market.condition_id,
            long_ok=long_ok,
            short_ok=short_ok,
            yes_edge=round(yes_edge, 4),
            no_edge=round(no_edge, 4),
        )
        return None

    # ── condition checkers ────────────────────────────────────────────────────

    def _check_long_conditions(
        self,
        indicators: IndicatorSet,
        timeframe_scores: Dict[str, TimeframeScore],
        market_condition: MarketCondition,
    ) -> Tuple[bool, List[str], float]:
        """
        Evaluate whether a YES (long BTC above target) setup is present.

        Returns (is_valid, reasons_list, raw_confidence_hint).
        reasons_list is non-empty only when is_valid is True.
        """
        reasons: List[str] = []
        checks_passed = 0

        # 1. BTC above EMA-200 (macro uptrend)
        if indicators.close > indicators.ema_200:
            reasons.append("BTC above EMA-200 (macro uptrend confirmed)")
            checks_passed += 1
        else:
            return False, [], 0.0

        # 2. EMA-20 > EMA-50 (short-term uptrend)
        if indicators.ema_20 > indicators.ema_50:
            reasons.append("EMA-20 above EMA-50 (short-term uptrend)")
            checks_passed += 1
        else:
            return False, [], 0.0

        # 3. RSI in bullish momentum zone
        rsi = indicators.rsi
        if self._settings.rsi_long_min <= rsi <= self._settings.rsi_long_max:
            reasons.append(
                f"RSI {rsi:.1f} in bullish momentum zone "
                f"({self._settings.rsi_long_min}–{self._settings.rsi_long_max})"
            )
            checks_passed += 1
        else:
            return False, [], 0.0

        # 4. MACD histogram positive and rising (bullish momentum confirmation)
        if indicators.macd_histogram > 0 and indicators.macd > indicators.macd_signal:
            reasons.append(
                f"MACD histogram positive ({indicators.macd_histogram:.5f}), "
                "MACD above signal line"
            )
            checks_passed += 1
        else:
            return False, [], 0.0

        # 5. Volume confirmation (optional per config)
        if self._settings.require_volume_confirmation:
            if indicators.volume_ratio >= self._settings.min_volume_ratio:
                reasons.append(
                    f"Volume ratio {indicators.volume_ratio:.2f}x above MA "
                    f"(min {self._settings.min_volume_ratio}x)"
                )
                checks_passed += 1
            else:
                logger.debug(
                    "Long: volume confirmation failed",
                    ratio=indicators.volume_ratio,
                    required=self._settings.min_volume_ratio,
                )
                return False, [], 0.0
        else:
            if indicators.volume_ratio >= self._settings.min_volume_ratio:
                reasons.append(f"Volume ratio {indicators.volume_ratio:.2f}x (supportive)")
                checks_passed += 1

        # 6. Multi-timeframe alignment: at least _MIN_BULLISH_TFS timeframes bullish
        bullish_tfs = [
            tf_name
            for tf_name, tfs in timeframe_scores.items()
            if tfs.overall_score < -0.2
        ]
        if len(bullish_tfs) >= _MIN_BULLISH_TFS:
            reasons.append(
                f"Bullish alignment on {len(bullish_tfs)} timeframes: "
                + ", ".join(bullish_tfs)
            )
            checks_passed += 1
        else:
            logger.debug(
                "Long: insufficient timeframe alignment",
                bullish_count=len(bullish_tfs),
                required=_MIN_BULLISH_TFS,
            )
            return False, [], 0.0

        raw_confidence = min(100.0, checks_passed * 14.0)
        return True, reasons, raw_confidence

    def _check_short_conditions(
        self,
        indicators: IndicatorSet,
        timeframe_scores: Dict[str, TimeframeScore],
        market_condition: MarketCondition,
    ) -> Tuple[bool, List[str], float]:
        """
        Evaluate whether a NO (short BTC below target) setup is present.

        Returns (is_valid, reasons_list, raw_confidence_hint).
        """
        reasons: List[str] = []
        checks_passed = 0

        # 1. BTC below EMA-200 (macro downtrend)
        if indicators.close < indicators.ema_200:
            reasons.append("BTC below EMA-200 (macro downtrend confirmed)")
            checks_passed += 1
        else:
            return False, [], 0.0

        # 2. EMA-20 < EMA-50 (short-term downtrend)
        if indicators.ema_20 < indicators.ema_50:
            reasons.append("EMA-20 below EMA-50 (short-term downtrend)")
            checks_passed += 1
        else:
            return False, [], 0.0

        # 3. RSI in bearish momentum zone
        rsi = indicators.rsi
        if self._settings.rsi_short_min <= rsi <= self._settings.rsi_short_max:
            reasons.append(
                f"RSI {rsi:.1f} in bearish momentum zone "
                f"({self._settings.rsi_short_min}–{self._settings.rsi_short_max})"
            )
            checks_passed += 1
        else:
            return False, [], 0.0

        # 4. MACD histogram negative and falling
        if indicators.macd_histogram < 0 and indicators.macd < indicators.macd_signal:
            reasons.append(
                f"MACD histogram negative ({indicators.macd_histogram:.5f}), "
                "MACD below signal line"
            )
            checks_passed += 1
        else:
            return False, [], 0.0

        # 5. Volume confirmation (optional per config)
        if self._settings.require_volume_confirmation:
            if indicators.volume_ratio >= self._settings.min_volume_ratio:
                reasons.append(
                    f"Volume ratio {indicators.volume_ratio:.2f}x above MA "
                    f"(min {self._settings.min_volume_ratio}x)"
                )
                checks_passed += 1
            else:
                logger.debug(
                    "Short: volume confirmation failed",
                    ratio=indicators.volume_ratio,
                    required=self._settings.min_volume_ratio,
                )
                return False, [], 0.0
        else:
            if indicators.volume_ratio >= self._settings.min_volume_ratio:
                reasons.append(f"Volume ratio {indicators.volume_ratio:.2f}x (supportive)")
                checks_passed += 1

        # 6. Multi-timeframe alignment: at least _MIN_BEARISH_TFS timeframes bearish
        bearish_tfs = [
            tf_name
            for tf_name, tfs in timeframe_scores.items()
            if tfs.overall_score > 0.2
        ]
        if len(bearish_tfs) >= _MIN_BEARISH_TFS:
            reasons.append(
                f"Bearish alignment on {len(bearish_tfs)} timeframes: "
                + ", ".join(bearish_tfs)
            )
            checks_passed += 1
        else:
            logger.debug(
                "Short: insufficient timeframe alignment",
                bearish_count=len(bearish_tfs),
                required=_MIN_BEARISH_TFS,
            )
            return False, [], 0.0

        raw_confidence = min(100.0, checks_passed * 14.0)
        return True, reasons, raw_confidence

    # ── global veto filters ───────────────────────────────────────────────────

    def _global_veto(
        self,
        market: Market,
        market_condition: MarketCondition,
    ) -> Tuple[bool, str]:
        """
        Return (True, reason) if trading should be refused regardless of direction.

        Checked before any technical analysis to avoid wasted computation.
        """
        # Choppy regime — no directional edge
        if market_condition.condition == MarketConditionType.CHOPPY:
            return True, "Market condition is CHOPPY"

        # Extreme volatility — position sizing breaks down
        if market_condition.volatility_regime == VolatilityRegime.EXTREME:
            return True, "Volatility regime is EXTREME"

        # Time-to-resolution gates
        hours = market.hours_to_resolution
        if hours < self._settings.min_time_to_resolution_hours:
            return (
                True,
                f"Only {hours:.1f}h to resolution "
                f"(min {self._settings.min_time_to_resolution_hours}h)",
            )
        if hours > self._settings.max_time_to_resolution_hours:
            return (
                True,
                f"{hours:.1f}h to resolution exceeds max "
                f"{self._settings.max_time_to_resolution_hours}h",
            )

        return False, ""
