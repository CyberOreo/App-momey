"""Position sizing calculator.

Implements fractional Kelly sizing with hard USDC caps/floors.
Conservative by design — a quarter-Kelly cap means the system must have
a very strong edge to deploy meaningful capital.
"""
from __future__ import annotations

from typing import Tuple

from loguru import logger

from src.core.models import Direction, PositionSizing


class PositionSizer:
    """
    Calculate trade size in USDC using Kelly criterion or fixed-fraction fallback.

    All sizes are bounded by:
      - settings.min_position_usdc  (absolute floor)
      - settings.max_position_usdc  (absolute ceiling)
      - settings.max_risk_per_trade_pct × balance  (risk percentage cap)
      - available_capital  (can't deploy more than you have)
    """

    def __init__(self, settings) -> None:
        self._settings = settings

    # ── primary compute ───────────────────────────────────────────────────────

    def compute(
        self,
        balance: float,
        confidence: float,
        edge: float,
        price: float,
        direction: Direction = Direction.YES,
        available_capital: float | None = None,
    ) -> PositionSizing:
        """
        Compute the recommended position size in USDC.

        Parameters
        ----------
        balance:
            Total account balance in USDC.
        confidence:
            Signal confidence score (0–100).
        edge:
            Estimated probability edge: fair_value − implied_probability (0–1).
        price:
            Current token price (0–1).  For YES tokens this is the implied
            probability directly; for NO tokens it is 1 − yes_price.
        direction:
            YES or NO direction (affects Kelly odds calculation).
        available_capital:
            Capital not currently deployed.  If None, defaults to *balance*.

        Returns
        -------
        PositionSizing
            All sizing fields filled in.
        """
        if available_capital is None:
            available_capital = balance

        if balance <= 0:
            return self._zero_sizing("zero_balance")

        if price <= 0.0 or price >= 1.0:
            logger.warning(
                "Token price out of range — using fixed minimum size",
                price=price,
            )
            price = max(0.01, min(0.99, price))

        if self._settings.use_kelly:
            kelly_f, method = self._kelly_size(edge, price, direction)
        else:
            kelly_f = 0.0
            method = "fixed_pct"

        # Apply the fractional Kelly multiplier (quarter-Kelly by default)
        fractional_kelly = kelly_f * self._settings.kelly_fraction

        if self._settings.use_kelly and fractional_kelly > 0:
            raw_size = fractional_kelly * balance
            method = "kelly"
        else:
            # Fixed-fraction fallback
            raw_size = self.fixed_fractional(balance, self._settings.max_risk_per_trade_pct)
            method = "fixed_pct"

        # Apply confidence scaling: reduce size for borderline signals
        confidence_scalar = self._confidence_scalar(confidence)
        raw_size *= confidence_scalar

        # Apply risk per-trade cap
        risk_cap = self._settings.max_risk_per_trade_pct * balance
        capped_size = min(raw_size, risk_cap)

        # Final clamping: min / max USDC bounds and available capital
        final_size, adj_reason = self.validate_size(capped_size, balance, available_capital)

        risk_amount = final_size * self._settings.max_risk_per_trade_pct  # approximate loss at max risk

        sizing = PositionSizing(
            recommended_size_usdc=round(final_size, 2),
            max_size_usdc=round(min(self._settings.max_position_usdc, available_capital), 2),
            risk_amount_usdc=round(risk_amount, 2),
            kelly_fraction=round(fractional_kelly, 6),
            method=method,
        )

        logger.debug(
            "Position sizing computed",
            method=method,
            balance=balance,
            confidence=confidence,
            edge=round(edge, 4),
            price=price,
            kelly_f=round(kelly_f, 6),
            fractional_kelly=round(fractional_kelly, 6),
            raw_size=round(raw_size, 2),
            final_size=final_size,
            adj_reason=adj_reason,
        )
        return sizing

    # ── Kelly criterion ───────────────────────────────────────────────────────

    def kelly_criterion(self, win_prob: float, odds: float) -> float:
        """
        Classic Kelly formula for binary bet sizing.

        Formula: f = (p × b − q) / b
        where:
          b = net fractional odds (profit per unit stake)
          p = estimated win probability
          q = 1 − p

        For prediction markets:
          b = (1 / token_price) − 1

        Returns the raw Kelly fraction (0–1).  Negative values (negative
        edge) are clamped to 0.
        """
        if odds <= 0:
            return 0.0
        p = max(0.0, min(1.0, win_prob))
        q = 1.0 - p
        f = (p * odds - q) / odds
        return max(0.0, f)

    def _kelly_size(
        self,
        edge: float,
        price: float,
        direction: Direction,
    ) -> Tuple[float, str]:
        """
        Compute the raw Kelly fraction for a binary prediction-market token.

        For a YES token at price *p*:
          win_prob = p + edge  (our estimate of the true probability)
          b = (1/p) − 1       (implied odds per unit invested)

        For a NO token at price *p*:
          win_prob = p + edge  (same logic, mirrored)
          b = (1/p) − 1
        """
        if price <= 0.001:
            return 0.0, "kelly"

        # Implied odds: what you win per USDC staked if the token goes to 1
        b = (1.0 / price) - 1.0

        # Our estimate of the true win probability
        win_prob = min(0.99, max(0.01, price + edge))

        kelly_f = self.kelly_criterion(win_prob, b)
        return kelly_f, "kelly"

    # ── fixed fraction fallback ───────────────────────────────────────────────

    def fixed_fractional(self, balance: float, risk_pct: float) -> float:
        """
        Simple fixed-fraction sizing: risk_pct × balance.

        Used when Kelly is disabled or returns zero.
        """
        return max(0.0, balance * risk_pct)

    # ── validation / clamping ─────────────────────────────────────────────────

    def validate_size(
        self,
        size: float,
        balance: float,
        available_capital: float,
    ) -> Tuple[float, str]:
        """
        Clamp *size* to all applicable bounds.

        Returns
        -------
        (final_size, adjustment_reason)
            *adjustment_reason* is an empty string when no adjustment was made.
        """
        reasons: list[str] = []
        final = size

        # Can't spend more than available capital
        if final > available_capital:
            final = available_capital
            reasons.append(f"capped to available capital {available_capital:.2f}")

        # Hard position ceiling
        if final > self._settings.max_position_usdc:
            final = self._settings.max_position_usdc
            reasons.append(f"capped to max_position_usdc {self._settings.max_position_usdc}")

        # Risk per-trade ceiling
        risk_cap = self._settings.max_risk_per_trade_pct * balance
        if final > risk_cap:
            final = risk_cap
            reasons.append(f"capped to risk cap {risk_cap:.2f}")

        # Minimum position floor
        if 0 < final < self._settings.min_position_usdc:
            final = 0.0  # too small to place — return zero so caller skips
            reasons.append(f"below min_position_usdc {self._settings.min_position_usdc} — zeroed")

        final = max(0.0, round(final, 2))
        adj_reason = "; ".join(reasons) if reasons else ""
        return final, adj_reason

    # ── confidence scaling ────────────────────────────────────────────────────

    def _confidence_scalar(self, confidence: float) -> float:
        """
        Return a multiplier in [0.5, 1.0] that scales position size down
        for lower-confidence signals.

        At threshold (65) → 0.5×.  At maximum (100) → 1.0×.
        Linear between.
        """
        threshold = self._settings.min_confidence_threshold
        conf = max(threshold, min(100.0, confidence))
        scalar = 0.5 + 0.5 * (conf - threshold) / (100.0 - threshold)
        return round(scalar, 4)

    # ── zero-sizing helper ────────────────────────────────────────────────────

    def _zero_sizing(self, method: str) -> PositionSizing:
        return PositionSizing(
            recommended_size_usdc=0.0,
            max_size_usdc=0.0,
            risk_amount_usdc=0.0,
            kelly_fraction=0.0,
            method=method,
        )
