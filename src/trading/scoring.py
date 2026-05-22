"""Confidence scorer: converts signal components into a 0–100 score.

Only genuinely strong, multi-factor setups should pass the 65-point threshold.
Every component is independently bounded so a single great indicator cannot
carry a weak signal over the line.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict

from loguru import logger

from src.core.models import (
    Direction,
    IndicatorSet,
    MarketCondition,
    MarketConditionType,
    TimeframeScore,
    VolatilityRegime,
)

if TYPE_CHECKING:
    from src.core.models import TradeSignal


# ── weight table ──────────────────────────────────────────────────────────────
_WEIGHTS: Dict[str, float] = {
    "trend_alignment": 25.0,   # EMA stack (20 > 50 > 200 for longs)
    "momentum": 20.0,          # MACD + rate-of-change
    "volume": 15.0,            # volume vs moving average
    "rsi_position": 15.0,      # RSI inside ideal zone
    "timeframe_agreement": 15.0,  # multiple timeframes agree
    "edge_magnitude": 10.0,    # Polymarket mispricing depth
}

_MAX_VOLATILITY_PENALTY = 15.0
_CHOPPY_PENALTY = 20.0


class ConfidenceScorer:
    """Convert individual signal-quality components into a bounded 0–100 score."""

    # ── public API ────────────────────────────────────────────────────────────

    def score(self, signal_components: Dict[str, float]) -> float:
        """
        Compute final confidence from pre-computed component scores.

        Expected keys (all 0–1 fractions unless noted):
            trend_alignment, momentum, volume, rsi_position,
            timeframe_agreement, edge_magnitude,
            volatility_penalty (0–1 fraction of max penalty),
            market_condition_bonus (unused — kept for forward compat),
            choppy_penalty (bool-like 0 or 1).
        """
        raw = 0.0

        for key, weight in _WEIGHTS.items():
            component_val = float(signal_components.get(key, 0.0))
            # Components are 0–1 fractions; multiply by weight for points
            raw += max(0.0, min(1.0, component_val)) * weight

        # Volatility penalty: up to -15 pts
        vol_penalty_frac = float(signal_components.get("volatility_penalty", 0.0))
        raw -= max(0.0, min(1.0, vol_penalty_frac)) * _MAX_VOLATILITY_PENALTY

        # Choppy market penalty: flat -20 pts when set
        choppy = float(signal_components.get("choppy_penalty", 0.0))
        if choppy > 0.5:
            raw -= _CHOPPY_PENALTY

        score = max(0.0, min(100.0, raw))
        logger.debug(
            "ConfidenceScorer.score",
            raw=round(raw, 2),
            final=round(score, 2),
            components=signal_components,
        )
        return score

    def compute_components(
        self,
        indicators: IndicatorSet,
        timeframe_scores: Dict[str, TimeframeScore],
        direction: Direction,
        market_condition: MarketCondition,
        edge: float,
    ) -> Dict[str, float]:
        """
        Derive each component score (0–1) from raw indicator data.

        Returns a dict ready to pass to :meth:`score`.
        """
        components: Dict[str, float] = {}

        # ── trend_alignment ───────────────────────────────────────────────────
        # Full score: EMA-20 > EMA-50 > EMA-200 for longs; reversed for shorts.
        if direction == Direction.YES:
            ema_stack = (indicators.ema_20 > indicators.ema_50) and (
                indicators.ema_50 > indicators.ema_200
            )
            price_above_200 = indicators.close > indicators.ema_200
            price_above_50 = indicators.close > indicators.ema_50
        else:
            ema_stack = (indicators.ema_20 < indicators.ema_50) and (
                indicators.ema_50 < indicators.ema_200
            )
            price_above_200 = indicators.close < indicators.ema_200
            price_above_50 = indicators.close < indicators.ema_50

        alignment_score = 0.0
        if ema_stack:
            alignment_score += 0.6
        if price_above_200:
            alignment_score += 0.25
        if price_above_50:
            alignment_score += 0.15
        components["trend_alignment"] = min(1.0, alignment_score)

        # ── momentum ──────────────────────────────────────────────────────────
        # MACD histogram positive/negative + 10-bar momentum sign
        if direction == Direction.YES:
            macd_ok = indicators.macd_histogram > 0
            mom_ok = indicators.momentum > 0
            macd_strength = min(1.0, abs(indicators.macd_histogram) / max(1e-8, abs(indicators.macd)))
        else:
            macd_ok = indicators.macd_histogram < 0
            mom_ok = indicators.momentum < 0
            macd_strength = min(1.0, abs(indicators.macd_histogram) / max(1e-8, abs(indicators.macd)))

        momentum_score = 0.0
        if macd_ok:
            momentum_score += 0.5 + 0.3 * macd_strength
        if mom_ok:
            momentum_score += 0.2
        components["momentum"] = min(1.0, momentum_score)

        # ── volume ────────────────────────────────────────────────────────────
        # volume_ratio > 1 = above average; scale to 0–1 capped at ratio 3×
        vol_ratio = indicators.volume_ratio
        if vol_ratio >= 1.0:
            vol_score = min(1.0, (vol_ratio - 1.0) / 2.0)
        else:
            vol_score = 0.0
        components["volume"] = vol_score

        # ── rsi_position ──────────────────────────────────────────────────────
        # Ideal long zone: 50–70; ideal short zone: 30–50.
        rsi = indicators.rsi
        if direction == Direction.YES:
            if 50.0 <= rsi <= 70.0:
                # Peak score at RSI 55–65
                deviation = abs(rsi - 60.0)
                rsi_score = max(0.3, 1.0 - deviation / 15.0)
            elif 45.0 <= rsi < 50.0:
                rsi_score = 0.2  # borderline — partial credit
            else:
                rsi_score = 0.0
        else:
            if 30.0 <= rsi <= 50.0:
                deviation = abs(rsi - 40.0)
                rsi_score = max(0.3, 1.0 - deviation / 15.0)
            elif 50.0 < rsi <= 55.0:
                rsi_score = 0.2
            else:
                rsi_score = 0.0
        components["rsi_position"] = rsi_score

        # ── timeframe_agreement ───────────────────────────────────────────────
        # Count how many timeframes show alignment in the right direction.
        # overall_score convention: negative = bullish (YES), positive = bearish (NO)
        bullish_tfs = sum(
            1 for tf in timeframe_scores.values() if tf.overall_score < -0.2
        )
        bearish_tfs = sum(
            1 for tf in timeframe_scores.values() if tf.overall_score > 0.2
        )
        total_tfs = max(1, len(timeframe_scores))

        if direction == Direction.YES:
            aligned_frac = bullish_tfs / total_tfs
        else:
            aligned_frac = bearish_tfs / total_tfs

        components["timeframe_agreement"] = min(1.0, aligned_frac)

        # ── edge_magnitude ────────────────────────────────────────────────────
        # Edge of 0.03 = minimal, 0.15+ = excellent. Scale 0.03→0, 0.15→1.
        clamped_edge = max(0.0, min(0.20, edge))
        if clamped_edge >= 0.03:
            edge_score = min(1.0, (clamped_edge - 0.03) / 0.12)
        else:
            edge_score = 0.0
        components["edge_magnitude"] = edge_score

        # ── penalties ─────────────────────────────────────────────────────────
        vol_regime = market_condition.volatility_regime
        vol_penalty_map = {
            VolatilityRegime.LOW: 0.0,
            VolatilityRegime.MEDIUM: 0.2,
            VolatilityRegime.HIGH: 0.6,
            VolatilityRegime.EXTREME: 1.0,
        }
        components["volatility_penalty"] = vol_penalty_map.get(vol_regime, 0.0)

        choppy = market_condition.condition == MarketConditionType.CHOPPY
        components["choppy_penalty"] = 1.0 if choppy else 0.0

        # forward-compat placeholder
        components["market_condition_bonus"] = 0.0

        return components

    def score_from_signal(self, signal: "TradeSignal") -> float:
        """Convenience: re-score an existing TradeSignal from its embedded data."""
        components = self.compute_components(
            indicators=signal.indicators,
            timeframe_scores=signal.timeframe_scores,
            direction=signal.direction,
            market_condition=signal.market_condition,
            edge=signal.edge,
        )
        return self.score(components)
