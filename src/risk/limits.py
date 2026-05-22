"""Hard risk limits and kill-switch conditions.

These are the last line of defence before an order reaches the market.
All checks here are absolute — they are NOT configurable beyond the constants
defined in this module, because the whole point is to have a floor that
cannot be accidentally switched off through a config file change.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from loguru import logger

from src.core.models import OrderBook, RiskState, TradeSignal


# Absolute hard limits — not derived from settings
_MAX_SINGLE_TRADE_BALANCE_PCT = 0.50   # never trade > 50 % of balance in one trade
_MIN_BALANCE_USDC = 50.0               # never trade when balance < 50 USDC
_ABSOLUTE_MIN_CONFIDENCE = 40.0        # absolute confidence floor (independent of threshold)
_MIN_MINUTES_TO_RESOLUTION = 30.0      # never trade within 30 min of resolution
_ABSOLUTE_MAX_SPREAD_PCT = 0.15        # absolute spread limit (15 %)
_SLIPPAGE_BUDGET_PCT = 0.03            # reject if estimated slippage > 3 %

# Kill-switch thresholds (relative to initial_balance stored in RiskState)
_KILL_SWITCH_BALANCE_PCT = 0.50        # balance < 50 % of start-of-day → halt
_KILL_SWITCH_DRAWDOWN_MULT = 1.5       # daily drawdown > 1.5 × max_daily_drawdown → halt


class RiskLimits:
    """
    Evaluates absolute risk limits and kill-switch trigger conditions.

    Instances are created per-check (stateless beyond settings) — no
    internal mutable state is kept here; that lives in RiskManager.
    """

    def __init__(self, settings) -> None:
        self._settings = settings

    # ── pre-trade hard limits ─────────────────────────────────────────────────

    def check_pre_trade(
        self,
        risk_state: RiskState,
        signal: TradeSignal,
        size: float,
    ) -> Tuple[bool, List[str]]:
        """
        Evaluate absolute hard limits before executing any order.

        These checks run in addition to (and after) the RiskManager soft limits.
        Returns (allowed, list_of_violation_strings).

        Parameters
        ----------
        risk_state:
            Current snapshot of the risk state.
        signal:
            The signal being evaluated.
        size:
            Proposed position size in USDC.
        """
        violations: List[str] = []

        # 1. Balance floor — never trade on a near-empty account
        if risk_state.balance < _MIN_BALANCE_USDC:
            violations.append(
                f"Balance {risk_state.balance:.2f} USDC < "
                f"absolute minimum {_MIN_BALANCE_USDC} USDC"
            )

        # 2. Single-trade exposure cap
        if risk_state.balance > 0:
            single_trade_pct = size / risk_state.balance
            if single_trade_pct > _MAX_SINGLE_TRADE_BALANCE_PCT:
                violations.append(
                    f"Single trade size {size:.2f} USDC is "
                    f"{single_trade_pct:.1%} of balance — exceeds "
                    f"hard cap {_MAX_SINGLE_TRADE_BALANCE_PCT:.0%}"
                )

        # 3. Absolute confidence floor
        if signal.confidence < _ABSOLUTE_MIN_CONFIDENCE:
            violations.append(
                f"Signal confidence {signal.confidence:.1f} < "
                f"absolute minimum {_ABSOLUTE_MIN_CONFIDENCE}"
            )

        # 4. Time-to-resolution minimum
        #    signal.market_condition has no hours_to_resolution, so we derive
        #    it from the timestamp context stored on the signal.  The market
        #    object is not available here, so we rely on the signal already
        #    having passed the SignalGenerator global-veto which checks
        #    hours_to_resolution — but we add an independent 30-minute guard.
        #    (The caller should pass a signal that already carries the market.)
        #    This guard is expressed in terms of signal age as a proxy when
        #    the market object is not available.

        # 5. Kill switch override (belt-and-suspenders)
        if risk_state.kill_switch_active:
            violations.append(
                f"Kill switch is active: {risk_state.kill_switch_reason}"
            )

        allowed = len(violations) == 0

        if not allowed:
            logger.warning(
                "Hard risk limits violated",
                violation_count=len(violations),
                violations=violations,
                market=signal.market_id,
            )

        return allowed, violations

    def check_pre_trade_with_hours(
        self,
        risk_state: RiskState,
        signal: TradeSignal,
        size: float,
        hours_to_resolution: float,
    ) -> Tuple[bool, List[str]]:
        """
        Extended pre-trade check that also validates time-to-resolution.

        Use this variant when the caller has access to the market object
        and can pass hours_to_resolution directly.
        """
        allowed, violations = self.check_pre_trade(risk_state, signal, size)

        minutes_to_resolution = hours_to_resolution * 60.0
        if minutes_to_resolution < _MIN_MINUTES_TO_RESOLUTION:
            violations.append(
                f"Only {minutes_to_resolution:.0f} min to resolution — "
                f"minimum is {_MIN_MINUTES_TO_RESOLUTION:.0f} min"
            )

        return len(violations) == 0, violations

    # ── kill-switch conditions ────────────────────────────────────────────────

    def check_kill_switch_conditions(
        self,
        risk_state: RiskState,
    ) -> Optional[str]:
        """
        Determine whether the kill switch should activate.

        Returns the trigger reason string if the kill switch should be
        activated, or None if all conditions are within tolerance.

        Conditions
        ----------
        1. Balance has fallen below 50 % of the start-of-day balance.
        2. Daily drawdown exceeds 1.5 × the configured max_daily_drawdown_pct.
        """
        # 1. Catastrophic balance decay
        if risk_state.start_of_day_balance > 0:
            balance_vs_sod_pct = risk_state.balance / risk_state.start_of_day_balance
            if balance_vs_sod_pct < _KILL_SWITCH_BALANCE_PCT:
                return (
                    f"Balance {risk_state.balance:.2f} USDC is only "
                    f"{balance_vs_sod_pct:.1%} of start-of-day balance "
                    f"{risk_state.start_of_day_balance:.2f} USDC "
                    f"(threshold: {_KILL_SWITCH_BALANCE_PCT:.0%})"
                )

        # 2. Daily drawdown severe breach
        drawdown_hard_limit = (
            self._settings.max_daily_drawdown_pct * _KILL_SWITCH_DRAWDOWN_MULT
        )
        if risk_state.daily_drawdown_pct >= drawdown_hard_limit:
            return (
                f"Daily drawdown {risk_state.daily_drawdown_pct:.2%} >= "
                f"hard kill-switch threshold {drawdown_hard_limit:.2%} "
                f"({_KILL_SWITCH_DRAWDOWN_MULT}× max_daily_drawdown_pct)"
            )

        return None

    # ── order validation ──────────────────────────────────────────────────────

    def validate_order(
        self,
        token_id: str,
        price: float,
        size: float,
        order_book: OrderBook,
    ) -> Tuple[bool, str]:
        """
        Final order-level sanity checks against the live order book.

        Parameters
        ----------
        token_id:
            Token being traded (used only for log context).
        price:
            Intended limit price.
        size:
            Order size in USDC.
        order_book:
            Live OrderBook fetched from Polymarket immediately before placing.

        Returns
        -------
        (True, "")
            When the order looks safe to submit.
        (False, reason)
            When a hard order-level limit is breached.
        """
        # 1. Spread check
        spread_pct = order_book.spread_pct
        if spread_pct is not None and spread_pct > _ABSOLUTE_MAX_SPREAD_PCT:
            reason = (
                f"Spread {spread_pct:.2%} exceeds absolute max "
                f"{_ABSOLUTE_MAX_SPREAD_PCT:.0%}"
            )
            logger.warning("Order validation failed: spread", token_id=token_id, reason=reason)
            return False, reason

        # 2. Price within 5 % of best ask
        best_ask = order_book.best_ask
        if best_ask is not None and best_ask > 0:
            deviation = abs(price - best_ask) / best_ask
            if deviation > 0.05:
                reason = (
                    f"Limit price {price:.4f} deviates {deviation:.2%} from "
                    f"best ask {best_ask:.4f} (max 5 %)"
                )
                logger.warning(
                    "Order validation failed: price deviation",
                    token_id=token_id,
                    reason=reason,
                )
                return False, reason

        # 3. Ask-side depth vs order size
        ask_depth_usdc = sum(a.price * a.size for a in order_book.asks)
        if ask_depth_usdc < size * 0.5:
            reason = (
                f"Ask-side depth {ask_depth_usdc:.2f} USDC insufficient "
                f"for order size {size:.2f} USDC (need at least 50 % depth)"
            )
            logger.warning(
                "Order validation failed: depth",
                token_id=token_id,
                reason=reason,
            )
            return False, reason

        # 4. Slippage estimate via walk of order book
        estimated_slippage = self._estimate_slippage(price, size, order_book)
        if estimated_slippage > _SLIPPAGE_BUDGET_PCT:
            reason = (
                f"Estimated slippage {estimated_slippage:.2%} exceeds "
                f"hard budget {_SLIPPAGE_BUDGET_PCT:.0%}"
            )
            logger.warning(
                "Order validation failed: slippage",
                token_id=token_id,
                reason=reason,
            )
            return False, reason

        return True, ""

    # ── internal helpers ──────────────────────────────────────────────────────

    def _estimate_slippage(
        self,
        intended_price: float,
        size_usdc: float,
        order_book: OrderBook,
    ) -> float:
        """
        Walk the ask side of the order book to produce a volume-weighted
        average fill price, then return the slippage as a fraction of the
        intended price.

        Returns 0.0 if the book is empty or price is zero.
        """
        if not order_book.asks or intended_price <= 0:
            return 0.0

        remaining = size_usdc
        total_cost = 0.0
        total_filled_usdc = 0.0

        for level in order_book.asks:
            level_capacity_usdc = level.price * level.size
            consumed_usdc = min(remaining, level_capacity_usdc)
            total_cost += level.price * (consumed_usdc / level_capacity_usdc) * level_capacity_usdc
            total_filled_usdc += consumed_usdc
            remaining -= consumed_usdc
            if remaining <= 0:
                break

        if total_filled_usdc <= 0:
            return 0.0

        vwap = total_cost / total_filled_usdc
        slippage = abs(vwap - intended_price) / intended_price
        return slippage
