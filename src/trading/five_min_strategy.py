"""
Five-Minute BTC Up/Down Strategy
=================================
Purpose-built for Polymarket's 5-minute BTC price markets.
Replaces the slow multi-hour strategy with tick-level signals.

Core philosophy:
  - Three independent signals must AGREE before any trade fires
  - If the crowd already priced the move in, we fade them (contrarian)
  - If the market hasn't reacted yet, we ride the flow (momentum)
  - Uncertainty = no trade. Always.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from loguru import logger

from src.core.config import get_settings
from src.core.models import (
    Candle, Direction, IndicatorSet, Market,
    MarketCondition, MarketConditionType, TimeframeScore,
    TradeSignal, VolatilityRegime,
)
from src.market.order_flow import OrderFlowAnalyzer, OrderFlowSnapshot


# ── Setup type ────────────────────────────────────────────────────────────────

@dataclass
class FiveMinSignal:
    """Internal signal before it becomes a TradeSignal."""
    direction: Direction
    setup_type: str            # 'momentum' | 'contrarian'
    confidence: float          # 0-100
    fair_value: float          # our probability estimate
    market_price: float        # token price on Polymarket
    edge: float                # fair_value - market_price (positive = we have edge)
    flow_signal: float         # -1 to +1
    velocity_90s: float        # BTC price velocity over 90s
    acceleration: float        # is the move speeding up or dying
    reasons: List[str]
    skip_reason: Optional[str] = None  # set if we decided to skip


# ── Per-market memory ─────────────────────────────────────────────────────────

@dataclass
class MarketMemory:
    consecutive_losses: int = 0
    blacklisted_until: Optional[datetime] = None
    last_entry_time: Optional[datetime] = None
    total_trades: int = 0
    wins: int = 0


# ── Main strategy class ───────────────────────────────────────────────────────

class FiveMinStrategy:
    """
    Evaluates every active 5-minute market and produces TradeSignal objects.

    Signal requires ALL THREE of the following to agree:
      1. Order flow  — buy/sell pressure at tick level (30s + 60s windows)
      2. Price velocity — BTC momentum over last 60-90 seconds
      3. Edge — Polymarket implied probability is meaningfully wrong

    Missing any one of the three = no trade.
    """

    # Hard limits — cannot be configured away
    MIN_SECONDS_TO_RESOLUTION = 90
    MAX_SECONDS_TO_RESOLUTION = 360
    MIN_LIQUIDITY_USDC = 150.0
    MAX_SPREAD_PCT = 0.06
    BLACKLIST_DURATION_MINUTES = 30

    # Early exit thresholds
    TAKE_PROFIT_PRICE = 0.82        # sell if token reaches this (locked ~64% gain)
    TIME_PRESSURE_SECONDS = 120     # with this many seconds left, take profit if up
    TIME_PRESSURE_MIN_GAIN = 0.20   # minimum gain % to exit under time pressure
    FLOW_REVERSAL_THRESHOLD = -0.20 # if flow flips this hard against us, exit
    MIN_CONSECUTIVE_LOSSES_TO_BLACKLIST = 2
    MIN_DATA_AGE_SECONDS = 90       # need this much order flow history
    MAX_PRICE_STALENESS_SECONDS = 8  # skip if BTC price feed is stale

    # Signal thresholds
    MIN_FLOW_SIGNAL_STRENGTH = 0.12   # |signal| below this = ambiguous = skip
    MIN_VELOCITY_PCT = 0.0008         # 0.08% move in 90s to confirm direction
    MIN_EDGE = 0.04                   # 4 percentage point minimum edge
    MIN_CONFIDENCE = 65.0

    # Contrarian trigger: if crowd moved price this far from 50/50, consider fading
    CONTRARIAN_TRIGGER_PCT = 0.12     # market at 0.62+ or 0.38- triggers contrarian check

    # Mean reversion penalty: big move in 120s → fade rather than momentum
    MEAN_REVERSION_THRESHOLD_PCT = 0.003   # 0.30% in 2 min = likely to revert

    def __init__(
        self,
        order_flow: OrderFlowAnalyzer,
        settings=None,
    ):
        self._flow = order_flow
        self._settings = settings or get_settings()
        self._market_memory: Dict[str, MarketMemory] = {}
        self._last_btc_price: float = 0.0
        self._last_btc_update: float = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def update_btc_price(self, price: float) -> None:
        self._last_btc_price = price
        self._last_btc_update = time.time()

    def evaluate_market(
        self,
        market: Market,
        candles_1m: List[Candle],
    ) -> Optional[TradeSignal]:
        """
        Main entry point. Returns a TradeSignal if a valid setup exists,
        None if no trade should be taken.
        """
        sig = self._build_signal(market, candles_1m)
        if sig is None or sig.skip_reason:
            if sig and sig.skip_reason:
                logger.debug(f"[5MIN] Skip {market.condition_id[:8]}: {sig.skip_reason}")
            return None

        if sig.confidence < self.MIN_CONFIDENCE:
            logger.debug(
                f"[5MIN] Low confidence {sig.confidence:.0f} on {market.condition_id[:8]}"
            )
            return None

        return self._to_trade_signal(sig, market)

    def record_result(self, market_id: str, won: bool) -> None:
        """Call after a trade resolves so we can track per-market performance."""
        mem = self._market_memory.setdefault(market_id, MarketMemory())
        mem.total_trades += 1
        if won:
            mem.wins += 1
            mem.consecutive_losses = 0
        else:
            mem.consecutive_losses += 1
            if mem.consecutive_losses >= self.MIN_CONSECUTIVE_LOSSES_TO_BLACKLIST:
                mem.blacklisted_until = datetime.utcnow() + timedelta(
                    minutes=self.BLACKLIST_DURATION_MINUTES
                )
                logger.warning(
                    f"[5MIN] Market {market_id[:16]} blacklisted for "
                    f"{self.BLACKLIST_DURATION_MINUTES} min after "
                    f"{mem.consecutive_losses} consecutive losses"
                )

    # ── Core signal construction ───────────────────────────────────────────────

    def _build_signal(
        self,
        market: Market,
        candles_1m: List[Candle],
    ) -> Optional[FiveMinSignal]:

        reasons: List[str] = []

        # ── Gate 1: Data freshness ─────────────────────────────────────────────
        price_age = time.time() - self._last_btc_update
        if price_age > self.MAX_PRICE_STALENESS_SECONDS or self._last_btc_price == 0:
            return FiveMinSignal(
                direction=Direction.YES, setup_type="", confidence=0,
                fair_value=0.5, market_price=0.5, edge=0, flow_signal=0,
                velocity_90s=0, acceleration=0, reasons=[],
                skip_reason=f"BTC price stale ({price_age:.0f}s old)",
            )

        if not self._flow.is_ready():
            return FiveMinSignal(
                direction=Direction.YES, setup_type="", confidence=0,
                fair_value=0.5, market_price=0.5, edge=0, flow_signal=0,
                velocity_90s=0, acceleration=0, reasons=[],
                skip_reason="Order flow not ready (< 90s of data)",
            )

        # ── Gate 2: Market quality ─────────────────────────────────────────────
        quality_ok, quality_reason = self._check_market_quality(market)
        if not quality_ok:
            return FiveMinSignal(
                direction=Direction.YES, setup_type="", confidence=0,
                fair_value=0.5, market_price=0.5, edge=0, flow_signal=0,
                velocity_90s=0, acceleration=0, reasons=[],
                skip_reason=quality_reason,
            )

        # ── Gate 3: Per-market blacklist ───────────────────────────────────────
        mem = self._market_memory.get(market.condition_id)
        if mem and mem.blacklisted_until and datetime.utcnow() < mem.blacklisted_until:
            remaining = (mem.blacklisted_until - datetime.utcnow()).seconds // 60
            return FiveMinSignal(
                direction=Direction.YES, setup_type="", confidence=0,
                fair_value=0.5, market_price=0.5, edge=0, flow_signal=0,
                velocity_90s=0, acceleration=0, reasons=[],
                skip_reason=f"Market blacklisted ({remaining}m remaining)",
            )

        # ── Collect all three signals ──────────────────────────────────────────
        snapshots = self._flow.multi_snapshot()
        snap_30 = snapshots["30s"]
        snap_60 = snapshots["60s"]

        if snap_30 is None or snap_60 is None:
            return FiveMinSignal(
                direction=Direction.YES, setup_type="", confidence=0,
                fair_value=0.5, market_price=0.5, edge=0, flow_signal=0,
                velocity_90s=0, acceleration=0, reasons=[],
                skip_reason="Insufficient order flow data",
            )

        velocity_90s = self._flow.recent_price_velocity(90)
        velocity_120s = self._flow.recent_price_velocity(120)
        acceleration = self._flow.acceleration()

        # Combine 30s and 60s flow signals (30s has more weight for fast markets)
        flow_signal = snap_30.signal_strength * 0.60 + snap_60.signal_strength * 0.40

        # ── Gate 4: Volatility spike check ────────────────────────────────────
        one_min_range = self._compute_1m_range(candles_1m)
        if one_min_range > 0.005:  # 0.5% range in 1 minute = extreme spike
            return FiveMinSignal(
                direction=Direction.YES, setup_type="", confidence=0,
                fair_value=0.5, market_price=0.5, edge=0, flow_signal=0,
                velocity_90s=0, acceleration=0, reasons=[],
                skip_reason=f"Extreme volatility spike ({one_min_range*100:.2f}% 1m range)",
            )

        # ── Gate 5: Minimum signal strength check ─────────────────────────────
        if abs(flow_signal) < self.MIN_FLOW_SIGNAL_STRENGTH:
            return FiveMinSignal(
                direction=Direction.YES, setup_type="", confidence=0,
                fair_value=0.5, market_price=0.5, edge=0, flow_signal=flow_signal,
                velocity_90s=0, acceleration=0, reasons=[],
                skip_reason=f"Ambiguous order flow ({flow_signal:.3f})",
            )

        if abs(velocity_90s) < self.MIN_VELOCITY_PCT:
            return FiveMinSignal(
                direction=Direction.YES, setup_type="", confidence=0,
                fair_value=0.5, market_price=0.5, edge=0, flow_signal=flow_signal,
                velocity_90s=velocity_90s, acceleration=0, reasons=[],
                skip_reason=f"No directional velocity ({velocity_90s*100:.3f}%)",
            )

        # ── Compute fair value ─────────────────────────────────────────────────
        fair_value = self._compute_fair_value(
            velocity_90s=velocity_90s,
            velocity_120s=velocity_120s,
            flow_signal=flow_signal,
            acceleration=acceleration,
        )

        # ── Determine direction and setup type ────────────────────────────────
        yes_token = market.yes_token
        no_token = market.no_token
        if not yes_token or not no_token:
            return None

        yes_price = yes_token.price
        no_price = no_token.price

        direction, setup_type, market_price, edge = self._determine_setup(
            fair_value=fair_value,
            yes_price=yes_price,
            no_price=no_price,
            velocity_90s=velocity_90s,
            flow_signal=flow_signal,
            reasons=reasons,
        )

        if direction is None:
            return FiveMinSignal(
                direction=Direction.YES, setup_type="none", confidence=0,
                fair_value=fair_value, market_price=yes_price, edge=0,
                flow_signal=flow_signal, velocity_90s=velocity_90s,
                acceleration=acceleration, reasons=reasons,
                skip_reason=f"No edge (fair={fair_value:.2f}, YES={yes_price:.2f})",
            )

        if edge < self.MIN_EDGE:
            return FiveMinSignal(
                direction=Direction.YES, setup_type="none", confidence=0,
                fair_value=fair_value, market_price=market_price, edge=edge,
                flow_signal=flow_signal, velocity_90s=velocity_90s,
                acceleration=acceleration, reasons=reasons,
                skip_reason=f"Edge too small ({edge*100:.1f}% < {self.MIN_EDGE*100:.0f}%)",
            )

        # ── Compute confidence ─────────────────────────────────────────────────
        confidence = self._compute_confidence(
            flow_signal=flow_signal,
            edge=edge,
            velocity_90s=velocity_90s,
            acceleration=acceleration,
            market=market,
            yes_price=yes_price,
            snap_30=snap_30,
            one_min_range=one_min_range,
            setup_type=setup_type,
        )

        return FiveMinSignal(
            direction=direction,
            setup_type=setup_type,
            confidence=confidence,
            fair_value=fair_value,
            market_price=market_price,
            edge=edge,
            flow_signal=flow_signal,
            velocity_90s=velocity_90s,
            acceleration=acceleration,
            reasons=reasons,
        )

    # ── Fair value model ──────────────────────────────────────────────────────

    def _compute_fair_value(
        self,
        velocity_90s: float,
        velocity_120s: float,
        flow_signal: float,
        acceleration: float,
    ) -> float:
        """
        Estimate the true probability that BTC closes higher in 5 minutes.

        Signals used:
          - velocity_90s: BTC price momentum (how much it moved in 90 seconds)
          - flow_signal:  order flow buy/sell pressure (-1 to +1)
          - acceleration: is the move speeding up or slowing down
          - mean_reversion: if too big a move already happened, it tends to revert
        """
        base = 0.50

        # Momentum factor: 0.4% move in 90s = ±15% adjustment
        # Scaled so 0.267% move = ±10%, 0.133% = ±5%
        momentum_factor = _clip(velocity_90s / 0.0027, -0.15, 0.15)

        # Order flow factor: strong smart-money buying = +12%
        flow_factor = _clip(flow_signal * 0.12, -0.12, 0.12)

        # Acceleration: move speeding up = more confidence in direction
        accel_factor = _clip(acceleration * 8, -0.06, 0.06)

        # Mean reversion: if 2-min move > 0.3%, the easy money is gone
        # Fade back toward 50/50 proportionally
        abs_move_120s = abs(velocity_120s)
        if abs_move_120s > self.MEAN_REVERSION_THRESHOLD_PCT:
            revert_amount = (abs_move_120s - self.MEAN_REVERSION_THRESHOLD_PCT) / 0.002 * 0.06
            revert_amount = min(revert_amount, 0.10)
            reversion_factor = -revert_amount * (1 if velocity_120s > 0 else -1)
        else:
            reversion_factor = 0.0

        raw = base + momentum_factor + flow_factor + accel_factor + reversion_factor
        return _clip(raw, 0.05, 0.95)

    # ── Setup determination ───────────────────────────────────────────────────

    def _determine_setup(
        self,
        fair_value: float,
        yes_price: float,
        no_price: float,
        velocity_90s: float,
        flow_signal: float,
        reasons: List[str],
    ) -> Tuple[Optional[Direction], str, float, float]:
        """
        Decide between momentum and contrarian, and which side to take.
        Returns (direction, setup_type, token_price, edge) or (None, ...) if no trade.
        """
        # -- Momentum setups --------------------------------------------------
        # BTC moving up, flow bullish, market hasn't adjusted → buy YES
        if (fair_value > 0.55
                and yes_price < fair_value - self.MIN_EDGE
                and velocity_90s > 0
                and flow_signal > 0):
            edge = fair_value - yes_price
            reasons.append(f"Momentum UP: fair={fair_value:.2f}, YES={yes_price:.2f}, edge={edge:.2f}")
            reasons.append(f"Flow: {flow_signal:+.2f}, velocity: {velocity_90s*100:+.3f}%/90s")
            return Direction.YES, "momentum", yes_price, edge

        # BTC moving down, flow bearish, market hasn't adjusted → buy NO
        if (fair_value < 0.45
                and no_price < (1 - fair_value) - self.MIN_EDGE
                and velocity_90s < 0
                and flow_signal < 0):
            no_fair = 1 - fair_value
            edge = no_fair - no_price
            reasons.append(f"Momentum DOWN: fair_no={no_fair:.2f}, NO={no_price:.2f}, edge={edge:.2f}")
            reasons.append(f"Flow: {flow_signal:+.2f}, velocity: {velocity_90s*100:+.3f}%/90s")
            return Direction.NO, "momentum", no_price, edge

        # -- Contrarian setups ------------------------------------------------
        # Crowd overbought YES (panic), but flow is already fading → buy NO
        if (yes_price > 0.50 + self.CONTRARIAN_TRIGGER_PCT
                and fair_value < yes_price - self.MIN_EDGE
                and (flow_signal < 0.05 or velocity_90s < 0)):
            no_fair = 1 - fair_value
            edge = no_fair - no_price
            if edge >= self.MIN_EDGE:
                reasons.append(f"Contrarian: crowd overbought YES={yes_price:.2f}, fair={fair_value:.2f}")
                reasons.append(f"Flow fading: {flow_signal:+.2f}")
                return Direction.NO, "contrarian", no_price, edge

        # Crowd oversold NO (panic sell), but flow is stabilising → buy YES
        if (no_price > 0.50 + self.CONTRARIAN_TRIGGER_PCT
                and (1 - fair_value) < no_price - self.MIN_EDGE
                and (flow_signal > -0.05 or velocity_90s > 0)):
            edge = fair_value - yes_price
            if edge >= self.MIN_EDGE:
                reasons.append(f"Contrarian: crowd oversold NO={no_price:.2f}, fair_yes={fair_value:.2f}")
                reasons.append(f"Flow stabilising: {flow_signal:+.2f}")
                return Direction.YES, "contrarian", yes_price, edge

        return None, "none", yes_price, 0.0

    # ── Confidence scoring ────────────────────────────────────────────────────

    def _compute_confidence(
        self,
        flow_signal: float,
        edge: float,
        velocity_90s: float,
        acceleration: float,
        market: Market,
        yes_price: float,
        snap_30: OrderFlowSnapshot,
        one_min_range: float,
        setup_type: str,
    ) -> float:
        score = 50.0

        # Order flow strength (0-25 pts) ──────────────────────────────────────
        # |signal| of 0.12 = 0 pts, 0.30+ = 25 pts
        flow_pts = _scale(abs(flow_signal), 0.12, 0.40, 0, 25)
        score += flow_pts

        # Edge magnitude (0-20 pts) ───────────────────────────────────────────
        # 4% edge = 0 pts, 15%+ edge = 20 pts
        edge_pts = _scale(edge, 0.04, 0.15, 0, 20)
        score += edge_pts

        # Velocity strength (0-15 pts) ────────────────────────────────────────
        # 0.08% / 90s = 0 pts, 0.4%+ = 15 pts
        vel_pts = _scale(abs(velocity_90s), 0.0008, 0.004, 0, 15)
        score += vel_pts

        # Smart money agreement (0-10 pts) ────────────────────────────────────
        # Large trades flowing same direction as our signal
        large_delta = abs(snap_30.large_trade_delta)
        smart_pts = _scale(large_delta, 0, 0.5, 0, 10)
        score += smart_pts

        # Acceleration bonus (+5 if accelerating in our direction) ─────────────
        if acceleration > 0.0001:   # move is speeding up
            score += 5
        elif acceleration < -0.0001:  # move is dying
            score -= 8

        # Contrarian setup bonus (+5 — contrarian setups have higher base edge) ─
        if setup_type == "contrarian":
            score += 5

        # Penalties ──────────────────────────────────────────────────────────
        # Time to resolution
        secs = market.hours_to_resolution * 3600
        if secs < self.MIN_SECONDS_TO_RESOLUTION:
            score -= 40  # hard block
        elif secs < 120:
            score -= 25
        elif secs < 180:
            score -= 10

        # Spread penalty
        spread = abs(yes_price - (1 - yes_price)) if market.no_token else 0
        if spread > 0.04:
            score -= _scale(spread, 0.04, 0.08, 0, 15)

        # Mild volatility (slight positive — gives us better prices)
        # But extreme volatility (already gated above) = already rejected
        if one_min_range > 0.003:
            score -= 10

        return _clip(score, 0, 100)

    # ── Market quality checks ─────────────────────────────────────────────────

    def _check_market_quality(self, market: Market) -> Tuple[bool, str]:
        if not market.active:
            return False, "Market inactive"

        secs = market.hours_to_resolution * 3600
        if secs < self.MIN_SECONDS_TO_RESOLUTION:
            return False, f"Too close to resolution ({secs:.0f}s)"
        if secs > self.MAX_SECONDS_TO_RESOLUTION:
            return False, f"Too far from resolution ({secs:.0f}s > {self.MAX_SECONDS_TO_RESOLUTION}s)"

        yes = market.yes_token
        no = market.no_token
        if not yes or not no:
            return False, "Missing YES or NO token"

        # Spread check
        if yes.price + no.price > 0:
            spread = abs((yes.price + no.price) - 1.0)
            if spread > self.MAX_SPREAD_PCT:
                return False, f"Spread too wide ({spread*100:.1f}%)"

        if market.liquidity < self.MIN_LIQUIDITY_USDC:
            return False, f"Low liquidity (${market.liquidity:.0f} < ${self.MIN_LIQUIDITY_USDC:.0f})"

        return True, ""

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _compute_1m_range(self, candles_1m: List[Candle]) -> float:
        """High-low range of the most recent 1-minute candle as a fraction of close."""
        if not candles_1m:
            return 0.0
        c = candles_1m[-1]
        if c.close == 0:
            return 0.0
        return (c.high - c.low) / c.close

    def _to_trade_signal(self, sig: FiveMinSignal, market: Market) -> TradeSignal:
        """Convert internal FiveMinSignal to the shared TradeSignal model."""
        token = market.yes_token if sig.direction == Direction.YES else market.no_token
        assert token is not None

        # Build a minimal IndicatorSet so the rest of the system can log it
        now = datetime.utcnow()
        btc = self._last_btc_price
        indicators = IndicatorSet(
            timestamp=now,
            timeframe="1m",
            close=btc,
            ema_20=btc,   # not computed at this timescale, use price as proxy
            ema_50=btc,
            ema_200=btc,
            rsi=50 + sig.flow_signal * 20,   # mapped for logging
            macd=sig.velocity_90s * btc,
            macd_signal=0.0,
            macd_histogram=sig.acceleration * btc,
            atr=self._compute_1m_range([]) * btc,
            atr_pct=0.0,
            volume_ma=0.0,
            volume_ratio=1.0,
            momentum=sig.velocity_90s * 100,
        )

        condition = MarketCondition(
            condition=MarketConditionType.TRENDING_UP if sig.velocity_90s > 0
                      else MarketConditionType.TRENDING_DOWN,
            trend_direction="up" if sig.velocity_90s > 0 else "down",
            volatility_regime=VolatilityRegime.MEDIUM,
            trend_strength=min(abs(sig.velocity_90s) / 0.003, 1.0),
            confidence=sig.confidence / 100,
            timestamp=now,
        )

        tf_score = TimeframeScore(
            timeframe="1m",
            trend_score=sig.flow_signal,
            momentum_score=_clip(sig.velocity_90s / 0.003, -1, 1),
            volume_score=0.5,
            overall_score=sig.flow_signal,
        )

        return TradeSignal(
            market_id=market.condition_id,
            direction=sig.direction,
            token_id=token.token_id,
            confidence=sig.confidence,
            price=sig.market_price,
            reasons=sig.reasons + [f"Setup: {sig.setup_type}"],
            timeframe_scores={"1m": tf_score},
            market_condition=condition,
            indicators=indicators,
            timestamp=now,
            implied_probability=sig.market_price,
            fair_value_estimate=sig.fair_value,
            edge=sig.edge,
        )

    # ── Market memory stats ───────────────────────────────────────────────────

    # ── Early exit logic ──────────────────────────────────────────────────────

    def should_exit_early(
        self,
        entry_price: float,
        current_token_price: float,
        seconds_to_resolution: float,
        direction: Direction,
    ) -> tuple[bool, str]:
        """
        Called every ~10 seconds for each open position.
        Returns (should_exit, reason) so the executor can act.

        Three triggers:

        1. TAKE PROFIT — token price hit 0.82+
           You entered at ~0.50, it's now 0.82. That's a 64% gain.
           Waiting for 1.00 risks a last-minute reversal wiping it out.
           Sell now, pocket the profit, move on.

        2. TIME PRESSURE — under 2 minutes left AND already in profit 20%+
           With 90-120 seconds left BTC can easily reverse. If you're sitting
           on a 20%+ gain, take it. The expected extra gain from waiting
           doesn't justify the reversal risk.

        3. FLOW REVERSAL — the order flow that drove our entry completely flipped
           This is the most important one. It means our thesis broke.
           We said 'buyers are in control' — now sellers have taken over.
           Get out before the price catches up with the new reality.
        """
        gain_pct = (current_token_price - entry_price) / entry_price

        # ── Trigger 1: Price hit take-profit level ────────────────────────────
        if current_token_price >= self.TAKE_PROFIT_PRICE:
            return True, (
                f"Take profit hit: token={current_token_price:.2f} >= {self.TAKE_PROFIT_PRICE} "
                f"(gain={gain_pct*100:.1f}%)"
            )

        # ── Trigger 2: Time pressure with profit ─────────────────────────────
        if (seconds_to_resolution <= self.TIME_PRESSURE_SECONDS
                and gain_pct >= self.TIME_PRESSURE_MIN_GAIN):
            return True, (
                f"Time pressure exit: {seconds_to_resolution:.0f}s left, "
                f"gain={gain_pct*100:.1f}% >= {self.TIME_PRESSURE_MIN_GAIN*100:.0f}%"
            )

        # ── Trigger 3: Order flow completely reversed ─────────────────────────
        snap = self._flow.snapshot(30)
        if snap is not None:
            flow = snap.signal_strength
            # If we went YES (bullish thesis), exit if flow is now strongly bearish
            if direction == Direction.YES and flow <= self.FLOW_REVERSAL_THRESHOLD:
                return True, (
                    f"Flow reversal exit: was bullish, flow now {flow:.2f} "
                    f"(threshold {self.FLOW_REVERSAL_THRESHOLD})"
                )
            # If we went NO (bearish thesis), exit if flow is now strongly bullish
            if direction == Direction.NO and flow >= -self.FLOW_REVERSAL_THRESHOLD:
                return True, (
                    f"Flow reversal exit: was bearish, flow now {flow:.2f}"
                )

        return False, ""

    def get_market_stats(self) -> List[dict]:
        results = []
        for market_id, mem in self._market_memory.items():
            win_rate = mem.wins / mem.total_trades if mem.total_trades else 0
            results.append({
                "market_id": market_id[:16],
                "total": mem.total_trades,
                "wins": mem.wins,
                "win_rate": round(win_rate, 2),
                "consecutive_losses": mem.consecutive_losses,
                "blacklisted": mem.blacklisted_until is not None
                    and datetime.utcnow() < (mem.blacklisted_until or datetime.min),
            })
        return sorted(results, key=lambda x: x["total"], reverse=True)


# ── Pure utility functions ────────────────────────────────────────────────────

def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _scale(value: float, in_lo: float, in_hi: float, out_lo: float, out_hi: float) -> float:
    """Linear interpolation: map value from [in_lo, in_hi] to [out_lo, out_hi]."""
    if in_hi <= in_lo:
        return out_lo
    t = (value - in_lo) / (in_hi - in_lo)
    t = _clip(t, 0.0, 1.0)
    return out_lo + t * (out_hi - out_lo)
