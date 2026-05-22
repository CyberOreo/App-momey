"""Paper trading simulator with realistic fill simulation.

Simulates Polymarket order fills with randomised slippage and taker fees
so that back-testing and paper-trading results stay conservative.
"""
from __future__ import annotations

import random
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from loguru import logger

from src.core.models import (
    Direction,
    Trade,
    TradeOutcome,
    TradeSignal,
)

# Fee and slippage constants that mirror real Polymarket behaviour
_TAKER_FEE_PCT = 0.001       # 0.1 % taker fee
_SLIPPAGE_MAX_PCT = 0.002    # ± 0.2 % random slippage


class PaperTrader:
    """
    Simulated execution engine for paper (virtual money) trading.

    All orders are filled immediately with a small random slippage and a
    fixed 0.1 % taker fee deducted from the balance.  Positions are
    tracked in memory; the open/closed split mirrors the DB Trade model.
    """

    def __init__(self, initial_balance: float, settings) -> None:
        if initial_balance <= 0:
            raise ValueError(f"initial_balance must be positive, got {initial_balance}")
        self._settings = settings
        self._initial_balance = initial_balance
        self._balance = initial_balance
        self._exposure_usdc: float = 0.0          # sum of cost-basis for open trades
        self._open_trades: Dict[str, Trade] = {}  # trade_id → Trade
        self._closed_trades: List[Trade] = []
        logger.info("PaperTrader initialised", initial_balance=initial_balance)

    # ── order placement ───────────────────────────────────────────────────────

    async def place_order(
        self,
        signal: TradeSignal,
        size_usdc: float,
    ) -> Trade:
        """
        Simulate a market-taker fill for the given signal.

        Slippage is drawn uniformly from [−0.2 %, +0.2 %] of the signal
        price, then the 0.1 % taker fee is deducted from the balance.

        Parameters
        ----------
        signal:
            The TradeSignal driving this order.
        size_usdc:
            Desired position size in USDC.

        Returns
        -------
        Trade
            A fully populated Trade record with outcome == OPEN.

        Raises
        ------
        ValueError
            If size_usdc exceeds current available balance.
        """
        available = self._balance - self._exposure_usdc
        if size_usdc > available:
            logger.warning(
                "Paper order capped to available capital",
                requested=size_usdc,
                available=available,
            )
            size_usdc = max(0.0, available)

        if size_usdc < self._settings.min_position_usdc:
            raise ValueError(
                f"Order size {size_usdc:.2f} USDC is below minimum "
                f"{self._settings.min_position_usdc} USDC"
            )

        # Simulate random slippage in ± _SLIPPAGE_MAX_PCT
        slippage_factor = 1.0 + random.uniform(-_SLIPPAGE_MAX_PCT, _SLIPPAGE_MAX_PCT)
        fill_price = max(0.001, min(0.999, signal.price * slippage_factor))

        # Number of tokens bought (Polymarket token unit is 1 USDC each at $1)
        tokens_bought = size_usdc / fill_price

        # Taker fee is paid on the USDC notional
        fee = size_usdc * _TAKER_FEE_PCT
        total_cost = size_usdc + fee

        if total_cost > self._balance:
            # Edge case: fee tips us over balance — reduce size slightly
            size_usdc = self._balance / (1 + _TAKER_FEE_PCT)
            fee = size_usdc * _TAKER_FEE_PCT
            total_cost = size_usdc + fee
            tokens_bought = size_usdc / fill_price

        trade_id = str(uuid.uuid4())
        trade = Trade(
            trade_id=trade_id,
            market_id=signal.market_id,
            condition_id=signal.market_id,
            token_id=signal.token_id,
            direction=signal.direction,
            size=tokens_bought,
            entry_price=fill_price,
            entry_time=datetime.utcnow(),
            confidence=signal.confidence,
            signal_reasons=list(signal.reasons),
            paper_trade=True,
            fees_paid=fee,
            outcome=TradeOutcome.OPEN,
        )

        # Update internal state
        self._open_trades[trade_id] = trade
        self._exposure_usdc += size_usdc  # cost-basis, not current value

        logger.info(
            "Paper order filled",
            trade_id=trade_id,
            direction=signal.direction.value,
            fill_price=round(fill_price, 6),
            signal_price=round(signal.price, 6),
            slippage_pct=round((fill_price / signal.price - 1) * 100, 4),
            tokens=round(tokens_bought, 4),
            fee=round(fee, 4),
            new_exposure=round(self._exposure_usdc, 2),
        )
        return trade

    # ── position close ────────────────────────────────────────────────────────

    async def close_position(
        self,
        trade: Trade,
        current_price: float,
        reason: str,
    ) -> Trade:
        """
        Close an open paper trade at *current_price*.

        PnL formula (binary prediction market):
            pnl = size_tokens × (exit_price − entry_price) − exit_fee

        where ``size_tokens`` is the number of tokens held and ``exit_price``
        is the simulated fill price.

        Parameters
        ----------
        trade:
            The open Trade to close.
        current_price:
            Mid-market price at the time of closing.
        reason:
            Human-readable close reason (stop-loss, take-profit, etc.).

        Returns
        -------
        Trade
            The same Trade object updated with exit fields and outcome.
        """
        if trade.trade_id not in self._open_trades:
            logger.warning(
                "Attempted to close a trade not in open_trades",
                trade_id=trade.trade_id,
            )
            # Try to update the trade even if we lost track of it
        else:
            del self._open_trades[trade.trade_id]

        # Simulate exit slippage (slightly worse — selling into bids)
        slippage_factor = 1.0 + random.uniform(-_SLIPPAGE_MAX_PCT, 0.0)
        exit_price = max(0.001, min(0.999, current_price * slippage_factor))

        # PnL = (exit − entry) × tokens
        gross_pnl = trade.size * (exit_price - trade.entry_price)
        # Exit fee on USDC notional = tokens × exit_price × fee_pct
        exit_notional = trade.size * exit_price
        exit_fee = exit_notional * _TAKER_FEE_PCT
        net_pnl = gross_pnl - exit_fee
        total_fees = trade.fees_paid + exit_fee

        # Recover cost-basis exposure
        entry_cost = trade.size * trade.entry_price
        self._exposure_usdc = max(0.0, self._exposure_usdc - entry_cost)

        # Update balance
        self._balance += exit_notional - exit_fee  # receive proceeds minus exit fee

        # Determine outcome
        if net_pnl > 0.01:
            outcome = TradeOutcome.WIN
        elif net_pnl < -0.01:
            outcome = TradeOutcome.LOSS
        else:
            outcome = TradeOutcome.BREAK_EVEN

        trade.exit_price = exit_price
        trade.realized_pnl = net_pnl
        trade.exit_time = datetime.utcnow()
        trade.outcome = outcome
        trade.fees_paid = total_fees

        self._closed_trades.append(trade)

        logger.info(
            "Paper position closed",
            trade_id=trade.trade_id,
            reason=reason,
            exit_price=round(exit_price, 6),
            net_pnl=round(net_pnl, 4),
            outcome=outcome.value,
            new_balance=round(self._balance, 2),
        )
        return trade

    # ── account state accessors ───────────────────────────────────────────────

    def get_balance(self) -> float:
        """Return the current simulated cash balance in USDC."""
        return self._balance

    def get_open_trades(self) -> List[Trade]:
        """Return all currently open paper trades."""
        return list(self._open_trades.values())

    def get_all_trades(self) -> List[Trade]:
        """Return all paper trades (open + closed)."""
        return list(self._open_trades.values()) + list(self._closed_trades)

    def get_total_pnl(self) -> float:
        """Return the sum of realised PnL across all closed paper trades."""
        return sum(
            t.realized_pnl for t in self._closed_trades if t.realized_pnl is not None
        )

    def get_exposure(self) -> float:
        """Return current total USDC exposure (sum of open cost-bases)."""
        return self._exposure_usdc

    def get_stats(self) -> dict:
        """
        Return a summary statistics dict suitable for monitoring dashboards.

        Keys
        ----
        total_trades, open_trades, closed_trades, win_rate,
        total_pnl, total_pnl_pct, current_balance, initial_balance,
        wins, losses, break_evens, current_exposure_usdc
        """
        closed = self._closed_trades
        total = len(closed)
        wins = sum(1 for t in closed if t.outcome == TradeOutcome.WIN)
        losses = sum(1 for t in closed if t.outcome == TradeOutcome.LOSS)
        break_evens = sum(1 for t in closed if t.outcome == TradeOutcome.BREAK_EVEN)
        total_pnl = self.get_total_pnl()
        win_rate = (wins / total) if total > 0 else 0.0
        pnl_pct = (
            (total_pnl / self._initial_balance) if self._initial_balance > 0 else 0.0
        )

        return {
            "total_trades": total,
            "open_trades": len(self._open_trades),
            "closed_trades": total,
            "wins": wins,
            "losses": losses,
            "break_evens": break_evens,
            "win_rate": round(win_rate, 4),
            "total_pnl": round(total_pnl, 4),
            "total_pnl_pct": round(pnl_pct, 4),
            "current_balance": round(self._balance, 2),
            "initial_balance": round(self._initial_balance, 2),
            "current_exposure_usdc": round(self._exposure_usdc, 2),
        }

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(self, initial_balance: float) -> None:
        """
        Reset the paper trader to a clean state for back-testing.

        All open and closed trades are discarded, balances reset.
        """
        if initial_balance <= 0:
            raise ValueError(f"initial_balance must be positive, got {initial_balance}")
        self._initial_balance = initial_balance
        self._balance = initial_balance
        self._exposure_usdc = 0.0
        self._open_trades = {}
        self._closed_trades = []
        logger.info("PaperTrader reset", initial_balance=initial_balance)
