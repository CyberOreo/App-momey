"""
Multi-timeframe market analysis engine.

Scores each timeframe independently on trend, momentum, and volume then
synthesises a consensus direction to support Polymarket edge detection.
"""
from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

import numpy as np
from loguru import logger

from src.core.models import (
    Candle,
    IndicatorSet,
    MarketCondition,
    TimeframeScore,
)
from src.market.indicators import IndicatorEngine


# ── Timeframe consensus weights ───────────────────────────────────────────────
# Keys are canonical labels used throughout the system.
_TF_WEIGHTS: Dict[str, float] = {
    "1h": 0.40,
    "4h": 0.30,
    "15m": 0.20,
    "5m": 0.10,
}

# Minimum edge (in probability points) required for an actionable signal.
_MIN_EDGE: float = 0.05


class MultiTimeframeAnalyzer:
    """
    Scores BTC market data across multiple timeframes and derives a consensus
    directional view to be used for Polymarket fair-value estimation.
    """

    def __init__(self, indicator_engine: IndicatorEngine, settings) -> None:
        """
        Parameters
        ----------
        indicator_engine:
            Shared :class:`~src.market.indicators.IndicatorEngine` instance.
        settings:
            Application :class:`~src.core.config.Settings` object; used for
            ``min_volume_ratio`` and ``require_volume_confirmation``.
        """
        self._engine = indicator_engine
        self._settings = settings

    # ── Core scoring ──────────────────────────────────────────────────────────

    def analyze(
        self,
        candles_by_tf: Dict[str, list[Candle]],
    ) -> Dict[str, TimeframeScore]:
        """
        Compute an :class:`~src.core.models.IndicatorSet` for every timeframe
        and convert it to a :class:`~src.core.models.TimeframeScore`.

        Timeframes that fail indicator computation (e.g. insufficient candles)
        are logged and omitted from the result.

        Parameters
        ----------
        candles_by_tf:
            Mapping of timeframe label → ordered candle list (oldest first).

        Returns
        -------
        Dict[str, TimeframeScore]
            Keyed by timeframe label; only successful timeframes included.
        """
        indicator_sets = self._engine.compute_all_timeframes(candles_by_tf)

        scores: Dict[str, TimeframeScore] = {}
        for tf, ind in indicator_sets.items():
            try:
                tf_score = self._score_timeframe(tf, ind)
                scores[tf] = tf_score
                logger.debug(
                    "TimeframeScore computed",
                    timeframe=tf,
                    trend=round(tf_score.trend_score, 3),
                    momentum=round(tf_score.momentum_score, 3),
                    volume=round(tf_score.volume_score, 3),
                    overall=round(tf_score.overall_score, 3),
                )
            except Exception as exc:  # pragma: no cover
                logger.error(
                    "Failed to score timeframe",
                    timeframe=tf,
                    error=str(exc),
                )

        return scores

    def _score_timeframe(self, timeframe: str, ind: IndicatorSet) -> TimeframeScore:
        """Convert a single :class:`IndicatorSet` to a :class:`TimeframeScore`.

        Scoring conventions
        -------------------
        * trend_score  : +1.0 = fully bullish, -1.0 = fully bearish
        * momentum_score: +1.0 = strong bullish momentum, -1.0 = bearish
        * volume_score : 0.0–1.0 (0 = no confirmation, 1 = strong volume)
        * overall_score: weighted composite of the three components
                         (trend 50 %, momentum 30 %, volume 20 %)
        """
        trend_score = self._compute_trend_score(ind)
        momentum_score = self._compute_momentum_score(ind)
        volume_score = self._compute_volume_score(ind)

        # overall_score: trend 50 %, momentum 30 %, volume 20 %
        # volume is direction-neutral so it amplifies rather than reverses.
        overall_score = (
            0.50 * trend_score
            + 0.30 * momentum_score
            + 0.20 * (volume_score * np.sign(trend_score) if trend_score != 0.0 else 0.0)
        )
        overall_score = float(np.clip(overall_score, -1.0, 1.0))

        return TimeframeScore(
            timeframe=timeframe,
            trend_score=trend_score,
            momentum_score=momentum_score,
            volume_score=volume_score,
            overall_score=overall_score,
        )

    @staticmethod
    def _compute_trend_score(ind: IndicatorSet) -> float:
        """
        EMA alignment score in [-1.0, +1.0].

        Full bullish (+1.0): close > EMA20 > EMA50 > EMA200
        Full bearish (-1.0): close < EMA20 < EMA50 < EMA200
        Partial alignments yield proportional scores.

        Scoring breakdown (bullish side, mirror for bearish):
            close > EMA20         → +0.25
            EMA20 > EMA50         → +0.30
            EMA50 > EMA200        → +0.25
            close > EMA50         → +0.10  (bonus confirmation)
            close > EMA200        → +0.10  (bonus confirmation)
        """
        close = ind.close
        ema20 = ind.ema_20
        ema50 = ind.ema_50
        ema200 = ind.ema_200

        # Guard against degenerate values
        if any(v == 0.0 for v in [ema20, ema50, ema200]):
            return 0.0

        bull_score = 0.0
        bear_score = 0.0

        # Primary alignment components
        if close > ema20:
            bull_score += 0.25
        else:
            bear_score += 0.25

        if ema20 > ema50:
            bull_score += 0.30
        else:
            bear_score += 0.30

        if ema50 > ema200:
            bull_score += 0.25
        else:
            bear_score += 0.25

        if close > ema50:
            bull_score += 0.10
        else:
            bear_score += 0.10

        if close > ema200:
            bull_score += 0.10
        else:
            bear_score += 0.10

        # Net score: positive = bullish, negative = bearish
        return float(np.clip(bull_score - bear_score, -1.0, 1.0))

    @staticmethod
    def _compute_momentum_score(ind: IndicatorSet) -> float:
        """
        Momentum score in [-1.0, +1.0].

        Two sub-components combined equally:
        1. RSI normalised: (RSI - 50) / 50  → -1.0 (RSI=0) to +1.0 (RSI=100)
        2. MACD histogram sign * relative strength:
               histogram / max(|macd_line|, ε) clamped to [-1, +1]
        """
        # RSI component: normalised around 50
        rsi_norm = float(np.clip((ind.rsi - 50.0) / 50.0, -1.0, 1.0))

        # MACD histogram component
        denom = max(abs(ind.macd), 1e-8)
        macd_norm = float(np.clip(ind.macd_histogram / denom, -1.0, 1.0))

        # Fallback when MACD is near zero: use raw histogram sign only
        if abs(ind.macd) < 1e-4:
            macd_norm = float(np.sign(ind.macd_histogram))

        # Momentum rate-of-change: cap at ±10 % ROC → ±1.0
        mom_norm = float(np.clip(ind.momentum / 10.0, -1.0, 1.0))

        # Weighted: RSI 35 %, MACD 35 %, momentum ROC 30 %
        score = 0.35 * rsi_norm + 0.35 * macd_norm + 0.30 * mom_norm
        return float(np.clip(score, -1.0, 1.0))

    def _compute_volume_score(self, ind: IndicatorSet) -> float:
        """
        Volume confirmation score in [0.0, 1.0].

        volume_ratio >= min_volume_ratio (default 1.2) → 1.0
        Scales proportionally below that threshold.
        0.0 when volume is below the moving average.
        """
        min_ratio = getattr(self._settings, "min_volume_ratio", 1.2)
        ratio = ind.volume_ratio

        if ratio <= 1.0:
            # Below-average volume: no confirmation
            return 0.0

        # Linearly scale 1.0 → min_ratio to 0.0 → 1.0
        if ratio >= min_ratio:
            return 1.0

        return float((ratio - 1.0) / max(min_ratio - 1.0, 1e-8))

    # ── Consensus ─────────────────────────────────────────────────────────────

    def get_consensus_direction(
        self,
        scores: Dict[str, TimeframeScore],
    ) -> Tuple[float, str]:
        """
        Derive a composite directional score from all available timeframes.

        Weights (applied when timeframe present):
            1h → 40 %,  4h → 30 %,  15m → 20 %,  5m → 10 %

        If a timeframe is absent its weight is redistributed proportionally
        among those that are present.

        Parameters
        ----------
        scores:
            Output of :meth:`analyze`.

        Returns
        -------
        Tuple[float, str]
            ``(composite_score, direction)`` where *composite_score* is in
            [-1.0, +1.0] and *direction* is ``'bullish'`` | ``'bearish'`` |
            ``'neutral'``.
        """
        if not scores:
            logger.warning("get_consensus_direction called with empty scores")
            return 0.0, "neutral"

        # Collect weights for present timeframes only
        present_weights: Dict[str, float] = {}
        for tf in scores:
            present_weights[tf] = _TF_WEIGHTS.get(tf, 0.05)

        total_weight = sum(present_weights.values())
        if total_weight == 0.0:
            return 0.0, "neutral"

        composite = 0.0
        for tf, tf_score in scores.items():
            w = present_weights[tf] / total_weight
            composite += w * tf_score.overall_score

        composite = float(np.clip(composite, -1.0, 1.0))

        if composite > 0.15:
            direction = "bullish"
        elif composite < -0.15:
            direction = "bearish"
        else:
            direction = "neutral"

        logger.debug(
            "Consensus direction",
            composite_score=round(composite, 4),
            direction=direction,
            timeframes=list(scores.keys()),
        )
        return composite, direction

    def is_aligned(
        self,
        scores: Dict[str, TimeframeScore],
        direction: str,
        min_timeframes: int = 2,
    ) -> bool:
        """
        Check whether at least *min_timeframes* timeframes agree on *direction*.

        A timeframe "agrees" when its ``overall_score`` crosses the ±0.15
        neutral band in the expected direction.

        Parameters
        ----------
        scores:
            Output of :meth:`analyze`.
        direction:
            ``'bullish'`` or ``'bearish'``.
        min_timeframes:
            Minimum number of agreeing timeframes required (default 2).

        Returns
        -------
        bool
        """
        threshold = 0.15
        count = 0

        for tf_score in scores.values():
            if direction == "bullish" and tf_score.overall_score > threshold:
                count += 1
            elif direction == "bearish" and tf_score.overall_score < -threshold:
                count += 1

        aligned = count >= min_timeframes
        logger.debug(
            "Alignment check",
            direction=direction,
            agreeing=count,
            required=min_timeframes,
            aligned=aligned,
        )
        return aligned

    # ── Fair-value estimation ─────────────────────────────────────────────────

    def compute_fair_value(
        self,
        btc_price: float,
        market_question: str,
        indicators: IndicatorSet,
        market_condition: MarketCondition,
    ) -> float:
        """
        Estimate the YES probability for a binary BTC price market.

        Algorithm
        ---------
        1. Parse the price target from the question text (e.g. "$100,000").
        2. Compute the raw distance ratio: ``(btc_price - target) / target``.
        3. Apply a momentum adjustment derived from RSI and MACD histogram.
        4. Map the adjusted ratio through a logistic function to [0, 1].
        5. Clamp to [0.02, 0.98] — no prediction is ever certain.

        "Will BTC be **above** $X" → YES = P(BTC > X at resolution).
        "Will BTC be **below** $X" → YES = P(BTC < X at resolution).
        If the question direction cannot be determined, defaults to "above".

        Parameters
        ----------
        btc_price:
            Current BTC spot price in USD.
        market_question:
            Full text of the Polymarket market question.
        indicators:
            Latest :class:`IndicatorSet` for the primary timeframe.
        market_condition:
            Current :class:`MarketCondition` regime.

        Returns
        -------
        float
            Fair value probability in [0.02, 0.98].
        """
        target_price, question_direction = self._parse_question(market_question)

        if target_price is None or target_price <= 0:
            # Cannot parse question → assume 50/50 with small momentum tilt
            logger.warning(
                "Could not parse BTC target from question; using 0.5 baseline",
                question=market_question,
            )
            base_prob = 0.5
        else:
            # Distance ratio: positive = BTC already above target
            distance_ratio = (btc_price - target_price) / target_price

            # Logistic scale factor: 10 % move away from target ≈ 80 % probability
            k = 20.0
            raw_prob = 1.0 / (1.0 + np.exp(-k * distance_ratio))

            # If question is "below", invert
            if question_direction == "below":
                raw_prob = 1.0 - raw_prob

            base_prob = float(raw_prob)

        # Momentum adjustment: RSI and MACD histogram push prob toward extremes
        momentum_adj = self._momentum_adjustment(indicators)

        # Trend strength attenuates adjustment in volatile conditions
        from src.core.models import VolatilityRegime
        vol_attenuation = {
            VolatilityRegime.LOW: 1.0,
            VolatilityRegime.MEDIUM: 0.8,
            VolatilityRegime.HIGH: 0.5,
            VolatilityRegime.EXTREME: 0.2,
        }.get(market_condition.volatility_regime, 0.5)

        confidence_weight = market_condition.confidence * vol_attenuation
        adjusted_prob = base_prob + momentum_adj * confidence_weight * 0.10

        fair_value = float(np.clip(adjusted_prob, 0.02, 0.98))
        logger.debug(
            "Fair value computed",
            btc_price=btc_price,
            question=market_question,
            base_prob=round(base_prob, 4),
            momentum_adj=round(momentum_adj, 4),
            fair_value=round(fair_value, 4),
        )
        return fair_value

    @staticmethod
    def _parse_question(question: str) -> Tuple[Optional[float], str]:
        """
        Extract a numeric price target and direction keyword from a question.

        Handles formats like:
            "Will BTC be above $100,000 by end of month?"
            "Will Bitcoin close below $90000?"
            "Does BTC hit $105,000?"

        Returns
        -------
        Tuple[Optional[float], str]
            ``(target_price, direction)`` where direction is ``'above'`` or
            ``'below'``.  target_price is None if parsing failed.
        """
        text = question.lower()

        # Detect direction keyword
        direction = "above"  # default assumption
        if "below" in text or "under" in text or "less than" in text:
            direction = "below"

        # Extract dollar amount — allow commas and optional decimal
        # Patterns: $100,000  $100000  $100,000.00
        pattern = re.compile(r"\$\s*([\d,]+(?:\.\d+)?)")
        matches = pattern.findall(question)

        if not matches:
            return None, direction

        # Use the last dollar value found (most specific in compound questions)
        raw = matches[-1].replace(",", "").strip()
        try:
            target = float(raw)
            return target, direction
        except ValueError:
            return None, direction

    @staticmethod
    def _momentum_adjustment(indicators: IndicatorSet) -> float:
        """
        Derive a [-1, +1] momentum adjustment from RSI and MACD histogram.

        Used to nudge the base probability slightly in the direction of
        short-term momentum.
        """
        rsi_component = (indicators.rsi - 50.0) / 50.0  # -1 to +1

        denom = max(abs(indicators.macd), 1e-8)
        macd_component = float(np.clip(indicators.macd_histogram / denom, -1.0, 1.0))
        if abs(indicators.macd) < 1e-4:
            macd_component = float(np.sign(indicators.macd_histogram))

        return float(np.clip(0.5 * rsi_component + 0.5 * macd_component, -1.0, 1.0))

    # ── Polymarket edge analysis ───────────────────────────────────────────────

    def analyze_implied_vs_fair(
        self,
        market,
        btc_price: float,
        fair_value: float,
    ) -> dict:
        """
        Compare our fair-value estimate to the market's implied probability.

        Parameters
        ----------
        market:
            A :class:`~src.core.models.Market` instance.
        btc_price:
            Current BTC spot price (used for logging context).
        fair_value:
            Our probability estimate from :meth:`compute_fair_value`.

        Returns
        -------
        dict with keys:
            ``edge``          – absolute difference between fair and implied (0–1)
            ``direction``     – ``'YES'`` | ``'NO'`` | ``'none'``
            ``overpriced_side`` – which side the market has mispriced
            ``magnitude``     – same as edge (alias for downstream consumers)
            ``actionable``    – bool; True when edge > 5 pp
        """
        implied_prob = market.yes_implied_prob  # P(YES) from order book

        # Raw signed edge from our model's perspective
        signed_edge = fair_value - implied_prob
        edge = abs(signed_edge)

        if signed_edge > 0:
            # We think YES is underpriced → buy YES (go long)
            direction = "YES"
            overpriced_side = "NO"
        elif signed_edge < 0:
            # We think YES is overpriced → buy NO
            direction = "NO"
            overpriced_side = "YES"
        else:
            direction = "none"
            overpriced_side = "none"

        actionable = edge > _MIN_EDGE

        result = {
            "edge": round(edge, 4),
            "direction": direction,
            "overpriced_side": overpriced_side,
            "magnitude": round(edge, 4),
            "actionable": actionable,
            "fair_value": round(fair_value, 4),
            "implied_prob": round(implied_prob, 4),
            "btc_price": btc_price,
        }

        logger.info(
            "Implied vs fair analysis",
            market_id=getattr(market, "condition_id", "unknown"),
            implied=round(implied_prob, 4),
            fair=round(fair_value, 4),
            edge=round(edge, 4),
            direction=direction,
            actionable=actionable,
        )
        return result
