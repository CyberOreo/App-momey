"""
Market regime and condition detection.

Classifies the current BTC market into one of the MarketConditionType buckets
and provides supporting utilities: ADX computation, tradeability gating, and
RSI divergence detection.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
from loguru import logger

from src.core.models import (
    Candle,
    IndicatorSet,
    MarketCondition,
    MarketConditionType,
    VolatilityRegime,
)


# ── Threshold constants ───────────────────────────────────────────────────────

# ATR as a fraction of price
_VOL_LOW_THRESHOLD: float = 0.010       # < 1.0 % → LOW
_VOL_MEDIUM_THRESHOLD: float = 0.025    # 1.0–2.5 % → MEDIUM
_VOL_HIGH_THRESHOLD: float = 0.050      # 2.5–5.0 % → HIGH
# >= 5.0 % → EXTREME

# Trend classification thresholds
_TREND_STRENGTH_THRESHOLD: float = 0.60  # EMA-spread derived

# Volume regime
_LOW_VOLUME_RATIO: float = 0.50          # < 50 % of moving average = illiquid

# ADX above this value indicates a trending market
_ADX_TREND_THRESHOLD: float = 25.0

# Number of recent bars to scan for EMA crossovers (choppy detection)
_CHOPPY_CROSSOVER_BARS: int = 20
_CHOPPY_CROSSOVER_MIN: int = 2           # ≥ this many crossovers = choppy


class MarketConditionDetector:
    """
    Classifies a BTC market into one of five regime buckets and exposes
    helper utilities for downstream decision-making.
    """

    # ── Primary detection ─────────────────────────────────────────────────────

    def detect(
        self,
        indicators: IndicatorSet,
        candles: List[Candle],
    ) -> MarketCondition:
        """
        Determine the dominant market condition from the latest indicators and
        raw candle history.

        Detection priority (first match wins):
        1. HIGH_VOLATILITY  – extreme ATR or RSI at extremes
        2. LOW_LIQUIDITY    – volume_ratio < 0.5
        3. TRENDING_UP      – price > EMA20 > EMA50 > EMA200 + momentum
        4. TRENDING_DOWN    – price < EMA20 < EMA50 < EMA200 + momentum
        5. CHOPPY           – multiple recent EMA crossovers / low momentum

        Parameters
        ----------
        indicators:
            Latest indicator values for the primary timeframe.
        candles:
            Raw OHLCV candles (oldest first) for the same timeframe.

        Returns
        -------
        MarketCondition
        """
        volatility_regime = self._classify_volatility(indicators.atr_pct)
        trend_strength = self._compute_trend_strength(indicators)
        adx = self.compute_adx(candles) if len(candles) >= 28 else 0.0
        # Blend EMA-spread trend strength with ADX signal
        adx_norm = float(np.clip(adx / 100.0, 0.0, 1.0))
        blended_strength = 0.5 * trend_strength + 0.5 * adx_norm

        condition, trend_direction = self._classify_condition(
            indicators=indicators,
            candles=candles,
            volatility_regime=volatility_regime,
            trend_strength=blended_strength,
        )

        # Confidence: how clearly this condition is expressed
        confidence = self._compute_confidence(
            condition=condition,
            indicators=indicators,
            trend_strength=blended_strength,
            volatility_regime=volatility_regime,
        )

        mc = MarketCondition(
            condition=condition,
            trend_direction=trend_direction,
            volatility_regime=volatility_regime,
            trend_strength=blended_strength,
            confidence=confidence,
            timestamp=indicators.timestamp,
        )

        logger.debug(
            "Market condition detected",
            condition=condition.value,
            direction=trend_direction,
            volatility=volatility_regime.value,
            trend_strength=round(blended_strength, 3),
            adx=round(adx, 2),
            confidence=round(confidence, 3),
        )
        return mc

    def _classify_condition(
        self,
        indicators: IndicatorSet,
        candles: List[Candle],
        volatility_regime: VolatilityRegime,
        trend_strength: float,
    ) -> Tuple[MarketConditionType, Optional[str]]:
        """Return (condition, trend_direction)."""

        close = indicators.close
        ema20 = indicators.ema_20
        ema50 = indicators.ema_50
        ema200 = indicators.ema_200
        rsi = indicators.rsi
        momentum = indicators.momentum
        atr_pct = indicators.atr_pct
        volume_ratio = indicators.volume_ratio

        # 1. HIGH_VOLATILITY
        if atr_pct > _VOL_MEDIUM_THRESHOLD or rsi > 80.0 or rsi < 20.0:
            return MarketConditionType.HIGH_VOLATILITY, None

        # 2. LOW_LIQUIDITY
        if volume_ratio < _LOW_VOLUME_RATIO:
            return MarketConditionType.LOW_LIQUIDITY, None

        # 3. TRENDING_UP
        ema_bull_stack = (
            close > ema20
            and ema20 > ema50
            and ema50 > ema200
        )
        if ema_bull_stack and momentum > 0.0 and trend_strength > _TREND_STRENGTH_THRESHOLD:
            return MarketConditionType.TRENDING_UP, "up"

        # 4. TRENDING_DOWN
        ema_bear_stack = (
            close < ema20
            and ema20 < ema50
            and ema50 < ema200
        )
        if ema_bear_stack and momentum < 0.0 and trend_strength > _TREND_STRENGTH_THRESHOLD:
            return MarketConditionType.TRENDING_DOWN, "down"

        # 5. CHOPPY (default when no clear trend)
        # Check for recent EMA20/EMA50 crossovers as a choppiness signal
        crossover_count = self._count_ema_crossovers(candles, window=_CHOPPY_CROSSOVER_BARS)
        if crossover_count >= _CHOPPY_CROSSOVER_MIN or trend_strength < 0.30:
            return MarketConditionType.CHOPPY, None

        # Weak trend — still classify directionally but at low strength
        if close > ema50:
            return MarketConditionType.TRENDING_UP, "up"
        return MarketConditionType.TRENDING_DOWN, "down"

    # ── Volatility regime ─────────────────────────────────────────────────────

    @staticmethod
    def _classify_volatility(atr_pct: float) -> VolatilityRegime:
        """Map ATR-as-fraction-of-price to a VolatilityRegime bucket."""
        if atr_pct < _VOL_LOW_THRESHOLD:
            return VolatilityRegime.LOW
        if atr_pct < _VOL_MEDIUM_THRESHOLD:
            return VolatilityRegime.MEDIUM
        if atr_pct < _VOL_HIGH_THRESHOLD:
            return VolatilityRegime.HIGH
        return VolatilityRegime.EXTREME

    # ── Trend strength (EMA-spread based) ────────────────────────────────────

    @staticmethod
    def _compute_trend_strength(indicators: IndicatorSet) -> float:
        """
        Derive a 0–1 trend strength score from EMA spread ratios.

        A perfectly aligned, wide-spread EMA stack yields 1.0.
        A flat, tangled EMA cluster yields 0.0.

        We measure the spreads relative to price to make it scale-invariant:
            spread_20_50  = |EMA20 - EMA50|  / close
            spread_50_200 = |EMA50 - EMA200| / close

        Each spread is normalised against a typical "strong trend" spread
        of 1 % and 2 % respectively, then combined.
        """
        close = indicators.close
        if close <= 0:
            return 0.0

        spread_20_50 = abs(indicators.ema_20 - indicators.ema_50) / close
        spread_50_200 = abs(indicators.ema_50 - indicators.ema_200) / close

        # Normalize: 1 % spread_20_50 → 1.0,  2 % spread_50_200 → 1.0
        norm_20_50 = float(np.clip(spread_20_50 / 0.01, 0.0, 1.0))
        norm_50_200 = float(np.clip(spread_50_200 / 0.02, 0.0, 1.0))

        # Additionally check whether EMA stack is properly ordered
        aligned_bull = (
            indicators.ema_20 > indicators.ema_50 > indicators.ema_200
        )
        aligned_bear = (
            indicators.ema_20 < indicators.ema_50 < indicators.ema_200
        )
        alignment_bonus = 0.20 if (aligned_bull or aligned_bear) else 0.0

        raw = 0.40 * norm_20_50 + 0.40 * norm_50_200 + alignment_bonus
        return float(np.clip(raw, 0.0, 1.0))

    # ── EMA crossover count (choppy detector) ─────────────────────────────────

    @staticmethod
    def _count_ema_crossovers(candles: List[Candle], window: int = 20) -> int:
        """
        Count EMA20/EMA50 crossovers in the last ``window`` bars.

        Uses the close prices directly and computes two short EMA series to
        detect whipsaw crossings without importing the full engine.
        """
        from src.market.indicators import ema as compute_ema

        n = min(len(candles), window + 50)  # extra bars for EMA warm-up
        subset = candles[-n:] if len(candles) > n else candles
        closes = np.array([c.close for c in subset], dtype=float)

        if len(closes) < 52:  # need at least 50 bars for EMA-50
            return 0

        ema20 = compute_ema(closes, 20)
        ema50 = compute_ema(closes, 50)

        # Only examine the last `window` bars that are fully computed
        ema20_tail = ema20[-window:]
        ema50_tail = ema50[-window:]

        # Filter out NaN pairs
        valid_mask = ~(np.isnan(ema20_tail) | np.isnan(ema50_tail))
        e20 = ema20_tail[valid_mask]
        e50 = ema50_tail[valid_mask]

        if len(e20) < 2:
            return 0

        diff = e20 - e50
        # Crossover = sign change in (EMA20 - EMA50)
        signs = np.sign(diff)
        crossovers = int(np.sum(np.abs(np.diff(signs)) > 0))
        return crossovers

    # ── Confidence scoring ────────────────────────────────────────────────────

    @staticmethod
    def _compute_confidence(
        condition: MarketConditionType,
        indicators: IndicatorSet,
        trend_strength: float,
        volatility_regime: VolatilityRegime,
    ) -> float:
        """
        Heuristic confidence (0–1) for how clearly a condition is expressed.

        Higher when:
        - Trend conditions have strong EMA alignment and healthy volume
        - Choppy market has low trend strength (i.e. clearly choppy)
        - Volatility readings are unambiguous
        """
        if condition in (MarketConditionType.TRENDING_UP, MarketConditionType.TRENDING_DOWN):
            # Confidence = trend strength × volume confirmation × RSI alignment
            rsi = indicators.rsi
            # Ideal RSI for trending: 45–65 (not overbought/oversold)
            rsi_ok = 1.0 if 40.0 <= rsi <= 70.0 else 0.5
            volume_ok = min(1.0, indicators.volume_ratio / 1.5)
            return float(np.clip(0.50 * trend_strength + 0.30 * volume_ok + 0.20 * rsi_ok, 0.0, 1.0))

        if condition == MarketConditionType.HIGH_VOLATILITY:
            # More extreme = clearer signal
            atr_signal = float(np.clip(indicators.atr_pct / _VOL_HIGH_THRESHOLD, 0.5, 1.0))
            return atr_signal

        if condition == MarketConditionType.LOW_LIQUIDITY:
            ratio_deficit = 1.0 - indicators.volume_ratio / _LOW_VOLUME_RATIO
            return float(np.clip(ratio_deficit, 0.3, 1.0))

        if condition == MarketConditionType.CHOPPY:
            # Clearly choppy when trend strength is very low
            return float(np.clip(1.0 - trend_strength, 0.2, 1.0))

        return 0.5

    # ── ADX ───────────────────────────────────────────────────────────────────

    def compute_adx(self, candles: List[Candle], period: int = 14) -> float:
        """
        Average Directional Index (Wilder, 1978).

        Returns a value in [0, 100]:
            < 25 → weak or absent trend
            25–50 → moderate trend
            > 50 → strong trend

        Requires at least ``2 * period + 1`` candles; returns 0.0 if
        insufficient data is available.

        Parameters
        ----------
        candles:
            Raw OHLCV candle list (oldest first).
        period:
            ATR / DI smoothing period (default 14).

        Returns
        -------
        float
            ADX value (latest bar).
        """
        min_bars = 2 * period + 1
        if len(candles) < min_bars:
            logger.debug(
                "Insufficient candles for ADX",
                have=len(candles),
                need=min_bars,
            )
            return 0.0

        highs = np.array([c.high for c in candles], dtype=float)
        lows = np.array([c.low for c in candles], dtype=float)
        closes = np.array([c.close for c in candles], dtype=float)
        n = len(closes)

        # True Range
        prev_closes = closes[:-1]
        hl = highs[1:] - lows[1:]
        hc = np.abs(highs[1:] - prev_closes)
        lc = np.abs(lows[1:] - prev_closes)
        tr = np.maximum(hl, np.maximum(hc, lc))

        # Directional Movement
        up_move = highs[1:] - highs[:-1]
        down_move = lows[:-1] - lows[1:]

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        # Wilder smoothing
        def _wilder_smooth(arr: np.ndarray, p: int) -> np.ndarray:
            """Wilder running sum / smoothing for ATR / DM."""
            result = np.empty(len(arr))
            result[:] = np.nan
            if len(arr) < p:
                return result
            result[p - 1] = np.sum(arr[:p])
            alpha = (p - 1) / p
            for i in range(p, len(arr)):
                result[i] = result[i - 1] * alpha + arr[i]
            return result

        smooth_tr = _wilder_smooth(tr, period)
        smooth_plus = _wilder_smooth(plus_dm, period)
        smooth_minus = _wilder_smooth(minus_dm, period)

        with np.errstate(divide="ignore", invalid="ignore"):
            plus_di = np.where(smooth_tr > 0, 100.0 * smooth_plus / smooth_tr, 0.0)
            minus_di = np.where(smooth_tr > 0, 100.0 * smooth_minus / smooth_tr, 0.0)
            di_sum = plus_di + minus_di
            dx = np.where(di_sum > 0, 100.0 * np.abs(plus_di - minus_di) / di_sum, 0.0)

        # ADX = Wilder smooth of DX
        smooth_dx = _wilder_smooth(dx, period)

        # Find the last non-NaN ADX value
        valid = smooth_dx[~np.isnan(smooth_dx)]
        if len(valid) == 0:
            return 0.0

        return float(valid[-1])

    # ── Tradeability gate ─────────────────────────────────────────────────────

    def is_tradeable_condition(
        self,
        condition: MarketCondition,
    ) -> Tuple[bool, str]:
        """
        Determine whether the current market condition is safe to trade in.

        Rules
        -----
        Cannot trade in:
            - EXTREME volatility regime (unpredictable price swings)
            - LOW_LIQUIDITY condition (wide spreads, execution risk)
            - CHOPPY market condition (no persistent edge)

        Can trade in:
            - TRENDING_UP or TRENDING_DOWN (directional edge exists)
            - LOW or MEDIUM volatility with a clear trend
            - HIGH volatility when a trend is confirmed (reduced size recommended)

        Parameters
        ----------
        condition:
            :class:`MarketCondition` from :meth:`detect`.

        Returns
        -------
        Tuple[bool, str]
            ``(can_trade, reason)``
        """
        # Hard blocks
        if condition.volatility_regime == VolatilityRegime.EXTREME:
            return (
                False,
                f"Extreme volatility (atr_pct implied ≥ {_VOL_HIGH_THRESHOLD * 100:.0f}%)"
                " — price action too erratic for reliable edge.",
            )

        if condition.condition == MarketConditionType.LOW_LIQUIDITY:
            return (
                False,
                "Low liquidity detected (volume_ratio < 0.5)"
                " — execution risk is unacceptably high.",
            )

        if condition.condition == MarketConditionType.CHOPPY:
            return (
                False,
                "Choppy market — no persistent directional edge;"
                " multiple recent EMA crossovers indicate whipsaw risk.",
            )

        # Trending with high volatility: allow but flag
        if condition.condition in (
            MarketConditionType.TRENDING_UP,
            MarketConditionType.TRENDING_DOWN,
        ):
            if condition.volatility_regime == VolatilityRegime.HIGH:
                direction = condition.trend_direction or "unknown"
                return (
                    True,
                    f"Trending {direction} with HIGH volatility"
                    " — trade allowed; consider reduced position size.",
                )
            return (
                True,
                f"Trending {condition.trend_direction or 'unknown'}"
                f" ({condition.volatility_regime.value} volatility)"
                f" — strength {condition.trend_strength:.2f}.",
            )

        # High volatility without clear trend: too risky
        if condition.condition == MarketConditionType.HIGH_VOLATILITY:
            return (
                False,
                "High volatility without clear trend direction"
                " — waiting for volatility to normalise.",
            )

        # Any remaining case: cautious allow
        return (
            True,
            f"Condition {condition.condition.value} — trading permitted"
            " with reduced confidence.",
        )

    # ── Divergence detection ──────────────────────────────────────────────────

    def detect_divergence(
        self,
        prices: List[float],
        rsi_values: List[float],
        bars: int = 20,
    ) -> Optional[str]:
        """
        Detect bullish or bearish RSI divergence over the last ``bars`` values.

        Divergence types
        ----------------
        Bullish divergence:
            Price is making a **lower low** while RSI makes a **higher low**.
            Signals potential reversal to the upside.

        Bearish divergence:
            Price is making a **higher high** while RSI makes a **lower high**.
            Signals potential reversal to the downside.

        Parameters
        ----------
        prices:
            Closing price series (most recent last).
        rsi_values:
            Corresponding RSI series (same length as ``prices``).
        bars:
            Number of recent bars to examine (default 20).

        Returns
        -------
        Optional[str]
            ``'bullish'`` | ``'bearish'`` | ``None``
        """
        if len(prices) < bars or len(rsi_values) < bars:
            logger.debug(
                "Insufficient data for divergence detection",
                prices_len=len(prices),
                rsi_len=len(rsi_values),
                bars=bars,
            )
            return None

        p = np.array(prices[-bars:], dtype=float)
        r = np.array(rsi_values[-bars:], dtype=float)

        # Remove NaN bars from both arrays in lock-step
        valid = ~(np.isnan(p) | np.isnan(r))
        p = p[valid]
        r = r[valid]

        if len(p) < 4:
            return None

        # Identify swing lows (local minima) and swing highs (local maxima)
        # A local minimum at index i: p[i] < p[i-1] and p[i] < p[i+1]
        def _swing_lows(arr: np.ndarray) -> np.ndarray:
            """Indices of local minima (strictly lower than both neighbours)."""
            idx = []
            for i in range(1, len(arr) - 1):
                if arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
                    idx.append(i)
            return np.array(idx, dtype=int)

        def _swing_highs(arr: np.ndarray) -> np.ndarray:
            """Indices of local maxima (strictly higher than both neighbours)."""
            idx = []
            for i in range(1, len(arr) - 1):
                if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
                    idx.append(i)
            return np.array(idx, dtype=int)

        low_idx = _swing_lows(p)
        high_idx = _swing_highs(p)

        # ── Bullish divergence: price lower low, RSI higher low ───────────────
        if len(low_idx) >= 2:
            i1, i2 = low_idx[-2], low_idx[-1]
            price_lower_low = p[i2] < p[i1]
            rsi_higher_low = r[i2] > r[i1]
            if price_lower_low and rsi_higher_low:
                logger.info(
                    "Bullish RSI divergence detected",
                    price_low1=round(float(p[i1]), 2),
                    price_low2=round(float(p[i2]), 2),
                    rsi_low1=round(float(r[i1]), 2),
                    rsi_low2=round(float(r[i2]), 2),
                )
                return "bullish"

        # ── Bearish divergence: price higher high, RSI lower high ─────────────
        if len(high_idx) >= 2:
            i1, i2 = high_idx[-2], high_idx[-1]
            price_higher_high = p[i2] > p[i1]
            rsi_lower_high = r[i2] < r[i1]
            if price_higher_high and rsi_lower_high:
                logger.info(
                    "Bearish RSI divergence detected",
                    price_high1=round(float(p[i1]), 2),
                    price_high2=round(float(p[i2]), 2),
                    rsi_high1=round(float(r[i1]), 2),
                    rsi_high2=round(float(r[i2]), 2),
                )
                return "bearish"

        return None
