"""Risk management orchestrator.

Tracks account-level risk state, enforces per-trade gates, manages
cooldowns and the kill switch, and logs all risk events to the database.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional, Tuple

from loguru import logger

from src.core.models import (
    PositionSizing,
    RiskState,
    Trade,
    TradeOutcome,
    TradeSignal,
)


class RiskManager:
    """
    Centralised risk orchestrator for the trading system.

    Holds the authoritative RiskState and exposes async methods that the
    TradeExecutor calls before and after every trade.  The optional *db*
    engine is used only for writing RiskEventRow records — all other state
    is kept in memory and refreshed on initialisation from the current
    account balance.
    """

    def __init__(self, settings, db=None) -> None:
        self._settings = settings
        self._db = db

        # Internal state — fully reset by initialize()
        self._balance: float = 0.0
        self._start_of_day_balance: float = 0.0
        self._daily_pnl: float = 0.0
        self._consecutive_losses: int = 0
        self._open_positions: int = 0
        self._total_exposure_usdc: float = 0.0
        self._in_cooldown: bool = False
        self._cooldown_until: Optional[datetime] = None
        self._kill_switch_active: bool = False
        self._kill_switch_reason: str = ""
        self._last_reset_date: Optional[datetime] = None
        self._initial_balance: float = 0.0   # set once at initialise() for kill-switch math

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def initialize(self, balance: float) -> None:
        """
        Set the starting balance and reset all counters.

        Must be called once at system start before any trading occurs.
        """
        if balance <= 0:
            raise ValueError(f"balance must be positive, got {balance}")

        self._balance = balance
        self._initial_balance = balance
        self._start_of_day_balance = balance
        self._daily_pnl = 0.0
        self._consecutive_losses = 0
        self._open_positions = 0
        self._total_exposure_usdc = 0.0
        self._in_cooldown = False
        self._cooldown_until = None
        self._kill_switch_active = False
        self._kill_switch_reason = ""
        self._last_reset_date = datetime.utcnow().date()

        await self.log_risk_event(
            "initialised",
            f"RiskManager initialised with balance={balance:.2f} USDC",
        )
        logger.info("RiskManager initialised", balance=balance)

    # ── state snapshot ────────────────────────────────────────────────────────

    async def get_state(self) -> RiskState:
        """Return a snapshot of the current risk state."""
        await self.check_cooldown_expiry()
        await self.check_daily_reset()

        return RiskState(
            balance=self._balance,
            start_of_day_balance=self._start_of_day_balance,
            daily_pnl=self._daily_pnl,
            consecutive_losses=self._consecutive_losses,
            open_positions=self._open_positions,
            total_exposure_usdc=self._total_exposure_usdc,
            in_cooldown=self._in_cooldown,
            cooldown_until=self._cooldown_until,
            kill_switch_active=self._kill_switch_active,
            kill_switch_reason=self._kill_switch_reason,
        )

    # ── pre-trade gate ────────────────────────────────────────────────────────

    async def can_execute(
        self,
        signal: TradeSignal,
        sizing: PositionSizing,
    ) -> Tuple[bool, str]:
        """
        Evaluate whether a trade may proceed.

        Checks are ordered from cheapest to most expensive.  The first
        failure short-circuits — only one rejection reason is returned.

        Returns
        -------
        (True, "")
            When all checks pass.
        (False, reason_str)
            When any check fails; *reason_str* is human-readable.
        """
        await self.check_cooldown_expiry()
        await self.check_daily_reset()

        state = await self.get_state()

        # 1. Kill switch — hard stop, no bypass
        if state.kill_switch_active:
            return False, f"Kill switch active: {state.kill_switch_reason}"

        # 2. Cooldown — temporary trading halt after consecutive losses
        if state.in_cooldown:
            remaining_secs = (
                (self._cooldown_until - datetime.utcnow()).total_seconds()
                if self._cooldown_until
                else 0
            )
            remaining_min = max(0, int(remaining_secs / 60))
            return False, f"In cooldown — {remaining_min} min remaining"

        # 3. Daily drawdown limit
        if state.daily_drawdown_pct >= self._settings.max_daily_drawdown_pct:
            reason = (
                f"Daily drawdown {state.daily_drawdown_pct:.2%} >= "
                f"max {self._settings.max_daily_drawdown_pct:.2%}"
            )
            await self.log_risk_event("daily_drawdown_breach", reason)
            return False, reason

        # 4. Max open positions
        if state.open_positions >= self._settings.max_open_positions:
            return (
                False,
                f"Max open positions reached ({state.open_positions}/"
                f"{self._settings.max_open_positions})",
            )

        # 5. Max total exposure
        if state.total_exposure_pct >= self._settings.max_total_exposure_pct:
            return (
                False,
                f"Total exposure {state.total_exposure_pct:.2%} >= "
                f"max {self._settings.max_total_exposure_pct:.2%}",
            )

        # 6. Minimum confidence threshold
        if signal.confidence < self._settings.min_confidence_threshold:
            return (
                False,
                f"Signal confidence {signal.confidence:.1f} < "
                f"threshold {self._settings.min_confidence_threshold:.1f}",
            )

        # 7. Minimum available capital
        if state.available_capital < sizing.recommended_size_usdc:
            return (
                False,
                f"Insufficient available capital "
                f"({state.available_capital:.2f} USDC available, "
                f"{sizing.recommended_size_usdc:.2f} USDC required)",
            )

        return True, ""

    # ── post-trade updates ────────────────────────────────────────────────────

    async def record_trade_result(self, trade: Trade) -> None:
        """
        Update internal counters after a trade closes.

        Triggers cooldown if the consecutive-loss limit is reached.
        Also checks whether the kill switch should activate.
        """
        if trade.realized_pnl is not None:
            self._daily_pnl += trade.realized_pnl
            self._balance += trade.realized_pnl

        # Release exposure when a trade closes
        cost_basis = trade.size * trade.entry_price
        self._total_exposure_usdc = max(0.0, self._total_exposure_usdc - cost_basis)
        self._open_positions = max(0, self._open_positions - 1)

        if trade.outcome == TradeOutcome.LOSS:
            self._consecutive_losses += 1
            logger.warning(
                "Consecutive losses updated",
                consecutive_losses=self._consecutive_losses,
                limit=self._settings.max_consecutive_losses,
            )
        elif trade.outcome in (TradeOutcome.WIN, TradeOutcome.BREAK_EVEN):
            self._consecutive_losses = 0

        # Trigger cooldown after too many losses
        if (
            not self._in_cooldown
            and self._consecutive_losses >= self._settings.max_consecutive_losses
        ):
            await self.enter_cooldown()

        # Check kill switch conditions
        from src.risk.limits import RiskLimits
        limits = RiskLimits(self._settings)
        state = await self.get_state()
        ks_reason = limits.check_kill_switch_conditions(state)
        if ks_reason and not self._kill_switch_active:
            await self.activate_kill_switch(ks_reason)

        await self.log_risk_event(
            "trade_closed",
            (
                f"trade={trade.trade_id} outcome={trade.outcome.value} "
                f"pnl={trade.realized_pnl:.4f} "
                f"consecutive_losses={self._consecutive_losses}"
            ),
        )

    async def record_new_position(self, cost_basis_usdc: float) -> None:
        """
        Register a new open position's capital deployment.

        Called by the executor immediately after a successful fill so that
        exposure and position-count counters stay accurate.
        """
        self._open_positions += 1
        self._total_exposure_usdc += cost_basis_usdc

    async def update_balance(self, new_balance: float) -> None:
        """Synchronise the internal balance with a freshly fetched value."""
        if new_balance < 0:
            logger.warning("Received negative balance", new_balance=new_balance)
            return
        old = self._balance
        self._balance = new_balance
        self._daily_pnl = new_balance - self._start_of_day_balance
        if abs(new_balance - old) > 0.01:
            logger.debug(
                "Balance updated",
                old=round(old, 2),
                new=round(new_balance, 2),
                daily_pnl=round(self._daily_pnl, 4),
            )

    # ── daily reset ───────────────────────────────────────────────────────────

    async def check_daily_reset(self) -> None:
        """
        Reset daily counters at UTC midnight.

        Safe to call on every event loop tick — the date comparison is O(1).
        """
        today = datetime.utcnow().date()
        if self._last_reset_date is not None and today != self._last_reset_date:
            prev_balance = self._start_of_day_balance
            self._start_of_day_balance = self._balance
            self._daily_pnl = 0.0
            self._last_reset_date = today
            await self.log_risk_event(
                "daily_reset",
                f"New trading day. Previous SoD balance={prev_balance:.2f} "
                f"New SoD balance={self._balance:.2f}",
            )
            logger.info(
                "Daily risk counters reset",
                new_start_of_day_balance=self._balance,
            )
        elif self._last_reset_date is None:
            self._last_reset_date = today

    # ── kill switch ───────────────────────────────────────────────────────────

    async def activate_kill_switch(self, reason: str) -> None:
        """
        Activate the kill switch, halting all new trade execution.

        The kill switch can only be deactivated by an explicit call to
        :meth:`deactivate_kill_switch` (typically via an admin command).
        """
        self._kill_switch_active = True
        self._kill_switch_reason = reason
        await self.log_risk_event("kill_switch_activated", reason)
        logger.error("KILL SWITCH ACTIVATED", reason=reason)

    async def deactivate_kill_switch(self) -> None:
        """
        Deactivate the kill switch to resume trading.

        Should only be called after the underlying issue has been reviewed
        and resolved by a human operator.
        """
        self._kill_switch_active = False
        old_reason = self._kill_switch_reason
        self._kill_switch_reason = ""
        await self.log_risk_event(
            "kill_switch_deactivated",
            f"Kill switch cleared (was: {old_reason})",
        )
        logger.info("Kill switch deactivated", was_reason=old_reason)

    # ── cooldown ──────────────────────────────────────────────────────────────

    async def enter_cooldown(self) -> None:
        """
        Enter a trading cooldown for *cooldown_minutes* minutes.

        Called automatically when consecutive-loss limit is reached, or
        it can be triggered manually.
        """
        self._in_cooldown = True
        self._cooldown_until = datetime.utcnow() + timedelta(
            minutes=self._settings.cooldown_minutes
        )
        reason = (
            f"{self._consecutive_losses} consecutive losses — "
            f"cooldown until {self._cooldown_until.strftime('%H:%M:%S UTC')}"
        )
        await self.log_risk_event("cooldown_entered", reason)
        logger.warning("Cooldown entered", cooldown_until=self._cooldown_until.isoformat())

    async def check_cooldown_expiry(self) -> None:
        """
        Clear the cooldown flag if the cooldown period has elapsed.

        Safe to call on every event loop tick — only the datetime comparison
        is executed when not in cooldown.
        """
        if self._in_cooldown and self._cooldown_until is not None:
            if datetime.utcnow() >= self._cooldown_until:
                self._in_cooldown = False
                self._cooldown_until = None
                self._consecutive_losses = 0   # reset streak on cooldown exit
                await self.log_risk_event(
                    "cooldown_expired",
                    "Cooldown period elapsed — trading resumed",
                )
                logger.info("Cooldown expired, trading resumed")

    # ── risk event logging ────────────────────────────────────────────────────

    async def log_risk_event(self, event_type: str, description: str) -> None:
        """
        Write a risk event to the database.

        Failures are silently swallowed so that logging never blocks trading.
        """
        if self._db is None:
            return
        try:
            from src.core.database import RiskEventRow, get_session
            async with get_session(self._db) as session:
                row = RiskEventRow(
                    event_type=event_type,
                    description=description,
                    balance_at_event=self._balance,
                    timestamp=datetime.utcnow(),
                )
                session.add(row)
        except Exception as exc:
            logger.warning(
                "Failed to log risk event",
                event_type=event_type,
                error=str(exc),
            )
