"""Trade execution engine.

Orchestrates paper vs live trading paths, pre-execution risk checks,
liquidity validation, slippage estimation, and position lifecycle management.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, List, Optional

from loguru import logger

from src.core.models import (
    Direction,
    Order,
    OrderBook,
    OrderStatus,
    Position,
    PositionSizing,
    Trade,
    TradeOutcome,
    TradeSignal,
)


# Default stop-loss / take-profit parameters
_STOP_LOSS_MULTIPLIER = 0.30   # stop at 30% of entry price (relative to entry)
_TAKE_PROFIT_DEFAULT = 0.85    # take profit at 0.85 token price


class TradeExecutor:
    """
    Async execution engine that routes signals to paper or live trading.

    Responsibilities:
      - Pre-execution risk gating (delegates to RiskManager)
      - Liquidity and slippage checks against live order book
      - Creating Trade records and persisting them
      - Stop-loss / take-profit monitoring for open positions
      - Closing positions and calculating realised PnL
    """

    def __init__(
        self,
        polymarket_client,
        paper_trader,
        position_manager,
        risk_manager,
        settings,
        db,
    ) -> None:
        self._client = polymarket_client
        self._paper_trader = paper_trader
        self._position_manager = position_manager
        self._risk_manager = risk_manager
        self._settings = settings
        self._db = db          # AsyncEngine; repositories instantiated per-call

    # ── main entry points ─────────────────────────────────────────────────────

    async def execute_signal(
        self,
        signal: TradeSignal,
        sizing: PositionSizing,
    ) -> Optional[Trade]:
        """
        Attempt to execute a TradeSignal.

        Flow:
          1. Risk manager gate — reject if any limit is breached.
          2. Live order-book liquidity check (even in paper mode).
          3. Slippage estimate check.
          4. Route to paper_trader or live order placement.
          5. Persist trade record and update risk state.

        Returns the Trade on success, None if rejected at any gate.
        """
        # ── 1. Risk manager gate ──────────────────────────────────────────────
        can_trade, reject_reason = await self._risk_manager.can_execute(signal, sizing)
        if not can_trade:
            logger.warning(
                "Signal rejected by risk manager",
                market=signal.market_id,
                reason=reject_reason,
                direction=signal.direction.value,
            )
            return None

        size_usdc = sizing.recommended_size_usdc

        # ── 2. Liquidity check ────────────────────────────────────────────────
        try:
            order_book = await self._client.get_order_book(signal.token_id)
        except Exception as exc:
            logger.error(
                "Failed to fetch order book before execution",
                token_id=signal.token_id,
                error=str(exc),
            )
            return None

        liquidity_ok, liquidity_reason = self._check_liquidity(order_book, size_usdc)
        if not liquidity_ok:
            logger.warning(
                "Signal rejected: insufficient liquidity",
                market=signal.market_id,
                reason=liquidity_reason,
            )
            return None

        # ── 3. Slippage estimate ──────────────────────────────────────────────
        slippage_ok, slippage_reason = self._check_slippage(signal, order_book, size_usdc)
        if not slippage_ok:
            logger.warning(
                "Signal rejected: slippage too high",
                market=signal.market_id,
                reason=slippage_reason,
            )
            return None

        # ── 4. Execute ────────────────────────────────────────────────────────
        stop_loss = self._calculate_stop_loss(signal, size_usdc)
        take_profit = self._calculate_take_profit(signal)

        if self._settings.paper_trading:
            trade = await self._paper_trader.place_order(signal, size_usdc)
        else:
            trade = await self._execute_live(signal, size_usdc)
            if trade is None:
                return None

        # ── 5. Register position and update risk state ────────────────────────
        try:
            position = await self._position_manager.add_position(trade)
            # Attach stop / TP after position creation
            position.stop_loss = stop_loss
            position.take_profit = take_profit
            await self._position_manager.update_stop_take(position)
        except Exception as exc:
            logger.error(
                "Failed to register position after fill",
                trade_id=trade.trade_id,
                error=str(exc),
            )
            # Trade executed but position not tracked — still return the trade

        await self._risk_manager.update_balance(
            await self._fetch_current_balance()
        )

        logger.info(
            "Trade executed",
            trade_id=trade.trade_id,
            market=trade.market_id,
            direction=trade.direction.value,
            size=trade.size,
            entry_price=trade.entry_price,
            paper=trade.paper_trade,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        return trade

    async def close_position(
        self,
        position: Position,
        reason: str,
    ) -> Optional[Trade]:
        """
        Close an open position.

        Calculates realised PnL, marks the trade record as closed in the DB,
        updates the risk manager, and removes the position from active tracking.

        Returns the updated Trade or None if the close failed.
        """
        try:
            current_price = position.current_price

            if self._settings.paper_trading:
                trade = await self._paper_trader.close_position(
                    await self._find_trade_for_position(position),
                    current_price,
                    reason,
                )
            else:
                trade = await self._close_live_position(position, current_price)
                if trade is None:
                    return None

            await self._position_manager.close_position(position, current_price)
            await self._risk_manager.record_trade_result(trade)
            await self._risk_manager.update_balance(
                await self._fetch_current_balance()
            )

            logger.info(
                "Position closed",
                position_id=position.position_id,
                reason=reason,
                exit_price=current_price,
                realized_pnl=trade.realized_pnl,
                outcome=trade.outcome.value,
            )
            return trade

        except Exception as exc:
            logger.error(
                "Error closing position",
                position_id=position.position_id,
                reason=reason,
                error=str(exc),
            )
            return None

    async def check_and_close_positions(
        self,
        positions: List[Position],
        current_prices: Dict[str, float],
    ) -> None:
        """
        Evaluate all open positions against stop-loss, take-profit, and
        market resolution (price → 0 or 1) and close any that are triggered.

        Parameters
        ----------
        positions:
            List of open Position objects.
        current_prices:
            Map of token_id → current market price.
        """
        for position in positions:
            price = current_prices.get(position.token_id)
            if price is None:
                logger.debug(
                    "No current price available for position",
                    token_id=position.token_id,
                )
                continue

            # Update current price in position tracker
            await self._position_manager.update_price(position.token_id, price)
            position.current_price = price

            close_reason: Optional[str] = None

            # Market resolved YES (price went to 1.0)
            if price >= 0.98:
                close_reason = "market_resolved_yes"

            # Market resolved NO (price went to 0.0)
            elif price <= 0.02:
                close_reason = "market_resolved_no"

            # Stop-loss triggered
            elif position.stop_loss is not None and price <= position.stop_loss:
                close_reason = f"stop_loss_triggered at {price:.4f} (stop={position.stop_loss:.4f})"

            # Take-profit triggered
            elif position.take_profit is not None and price >= position.take_profit:
                close_reason = f"take_profit_triggered at {price:.4f} (tp={position.take_profit:.4f})"

            if close_reason:
                await self.close_position(position, close_reason)

    # ── internal execution helpers ────────────────────────────────────────────

    async def _execute_live(
        self,
        signal: TradeSignal,
        size_usdc: float,
    ) -> Optional[Trade]:
        """Place a real order via the Polymarket client and build a Trade record."""
        order = await self._execute_live_order(signal, size_usdc)
        if order is None:
            return None

        # Determine fill price — fall back to signal price if not available
        fill_price = (
            order.average_fill_price
            if order.average_fill_price > 0
            else signal.price
        )
        filled_size = order.filled_size if order.filled_size > 0 else size_usdc

        # Calculate fee: Polymarket charges 0.1% taker fee
        fee = filled_size * 0.001

        trade = Trade(
            trade_id=str(uuid.uuid4()),
            market_id=signal.market_id,
            condition_id=signal.market_id,   # condition_id == market_id for Polymarket
            token_id=signal.token_id,
            direction=signal.direction,
            size=filled_size / fill_price if fill_price > 0 else 0.0,
            entry_price=fill_price,
            entry_time=datetime.utcnow(),
            confidence=signal.confidence,
            signal_reasons=signal.reasons,
            paper_trade=False,
            fees_paid=fee,
        )

        # Persist
        from src.core.database import TradeRepository
        repo = TradeRepository(self._db)
        await repo.save_trade(trade)

        return trade

    async def _execute_live_order(
        self,
        signal: TradeSignal,
        size_usdc: float,
    ) -> Optional[Order]:
        """
        Place an actual limit order on Polymarket.

        Handles partial fills and rejections gracefully — returns None on
        complete failure so the caller can abort cleanly.
        """
        try:
            order = await self._client.place_order(
                token_id=signal.token_id,
                side="BUY",
                price=round(signal.price, 4),
                size=round(size_usdc, 2),
                order_type="GTC",
            )

            if order.status == OrderStatus.REJECTED:
                logger.warning(
                    "Live order rejected",
                    token_id=signal.token_id,
                    size=size_usdc,
                    price=signal.price,
                )
                return None

            if order.status == OrderStatus.CANCELLED:
                logger.warning(
                    "Live order cancelled immediately",
                    order_id=order.order_id,
                )
                return None

            # Accept FILLED and PARTIALLY_FILLED
            if order.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
                if order.status == OrderStatus.PARTIALLY_FILLED:
                    logger.warning(
                        "Order partially filled",
                        order_id=order.order_id,
                        filled=order.filled_size,
                        requested=order.size,
                    )
                return order

            # PENDING — the order was accepted but not yet filled
            # Return it and let the position manager track the fill later
            logger.info(
                "Live order pending",
                order_id=order.order_id,
                price=order.price,
                size=order.size,
            )
            return order

        except Exception as exc:
            logger.error(
                "Live order placement failed",
                token_id=signal.token_id,
                error=str(exc),
            )
            return None

    async def _close_live_position(
        self,
        position: Position,
        exit_price: float,
    ) -> Optional[Trade]:
        """
        Close a live position by selling the held tokens back to the market.
        Builds a Trade record from the resulting order fill.
        """
        size_usdc = position.size * exit_price  # approximate USDC value

        try:
            order = await self._client.place_order(
                token_id=position.token_id,
                side="SELL",
                price=round(exit_price, 4),
                size=round(size_usdc, 2),
                order_type="GTC",
            )

            if order.status == OrderStatus.REJECTED:
                logger.error(
                    "Close order rejected — position remains open",
                    position_id=position.position_id,
                )
                return None

            actual_exit = (
                order.average_fill_price if order.average_fill_price > 0 else exit_price
            )
            fee = size_usdc * 0.001
            pnl = position.size * (actual_exit - position.entry_price) - fee

            outcome = TradeOutcome.BREAK_EVEN
            if pnl > 0.01:
                outcome = TradeOutcome.WIN
            elif pnl < -0.01:
                outcome = TradeOutcome.LOSS

            trade = Trade(
                trade_id=str(uuid.uuid4()),
                market_id=position.market_id,
                condition_id=position.condition_id,
                token_id=position.token_id,
                direction=position.direction,
                size=position.size,
                entry_price=position.entry_price,
                entry_time=position.entry_time,
                confidence=position.confidence,
                paper_trade=False,
                exit_price=actual_exit,
                realized_pnl=pnl,
                exit_time=datetime.utcnow(),
                outcome=outcome,
                fees_paid=fee,
            )

            from src.core.database import TradeRepository
            repo = TradeRepository(self._db)
            await repo.update_trade(trade)

            return trade

        except Exception as exc:
            logger.error(
                "Failed to close live position",
                position_id=position.position_id,
                error=str(exc),
            )
            return None

    # ── liquidity and slippage checks ────────────────────────────────────────

    def _check_liquidity(
        self,
        order_book: OrderBook,
        size_usdc: float,
    ) -> tuple[bool, str]:
        """
        Verify there is enough liquidity to fill the desired size without
        catastrophic slippage.

        Returns (ok, reason_if_rejected).
        """
        # Absolute minimum total liquidity
        total_liq = order_book.total_liquidity
        if total_liq < self._settings.min_liquidity_usdc:
            return (
                False,
                f"Total liquidity {total_liq:.2f} USDC < "
                f"minimum {self._settings.min_liquidity_usdc} USDC",
            )

        # Spread gate
        spread_pct = order_book.spread_pct
        if spread_pct is not None and spread_pct > self._settings.max_spread_pct:
            return (
                False,
                f"Spread {spread_pct:.2%} exceeds max {self._settings.max_spread_pct:.2%}",
            )

        # Ask-side depth: can we actually fill our order?
        ask_depth = sum(a.price * a.size for a in order_book.asks)
        if ask_depth < size_usdc * 0.8:  # allow up to 20% shortfall before rejection
            return (
                False,
                f"Ask-side depth {ask_depth:.2f} USDC insufficient for "
                f"order size {size_usdc:.2f} USDC",
            )

        return True, ""

    def _check_slippage(
        self,
        signal: TradeSignal,
        order_book: OrderBook,
        size_usdc: float,
    ) -> tuple[bool, str]:
        """
        Walk the ask-side of the order book and estimate volume-weighted
        average fill price. Reject if estimated slippage exceeds max_slippage_pct.
        """
        if not order_book.asks:
            return False, "Order book has no asks"

        remaining = size_usdc
        weighted_price_sum = 0.0

        for level in order_book.asks:
            level_usdc = level.price * level.size
            consumed = min(remaining, level_usdc)
            weighted_price_sum += level.price * (consumed / level_usdc) * level_usdc
            remaining -= consumed
            if remaining <= 0:
                break

        if remaining > 0:
            # Could not fill entire order — last level price used for remainder
            last_price = order_book.asks[-1].price
            weighted_price_sum += last_price * remaining

        total_filled = size_usdc
        vwap = weighted_price_sum / total_filled if total_filled > 0 else signal.price

        slippage = abs(vwap - signal.price) / signal.price if signal.price > 0 else 0.0

        if slippage > self._settings.max_slippage_pct:
            return (
                False,
                f"Estimated slippage {slippage:.2%} exceeds max "
                f"{self._settings.max_slippage_pct:.2%}",
            )

        return True, ""

    # ── stop-loss / take-profit calculation ───────────────────────────────────

    def _calculate_stop_loss(
        self,
        signal: TradeSignal,
        size_usdc: float,
    ) -> Optional[float]:
        """
        Compute the stop-loss price for a new position.

        Strategy: stop at 30% of the entry price (e.g. entry at 0.60 → stop at 0.18).
        Clamped to a minimum of 0.01 so we never set a negative/zero stop.
        """
        if signal.price <= 0:
            return None
        stop = signal.price * _STOP_LOSS_MULTIPLIER
        return max(0.01, round(stop, 4))

    def _calculate_take_profit(
        self,
        signal: TradeSignal,
    ) -> Optional[float]:
        """
        Compute the take-profit price for a new position.

        Default target: 0.85 token price (scale out near resolution).
        For NO tokens, the target is the same absolute level — the market
        resolves to 1.0 if NO wins.
        """
        return _TAKE_PROFIT_DEFAULT

    # ── helper utilities ──────────────────────────────────────────────────────

    async def _fetch_current_balance(self) -> float:
        """
        Return the current account balance.

        In paper mode, delegate to the paper trader. In live mode, query
        the Polymarket client; fall back to the last known balance on error.
        """
        if self._settings.paper_trading:
            return self._paper_trader.get_balance()
        try:
            return await self._client.get_balance()
        except Exception as exc:
            logger.warning(
                "Failed to fetch live balance after trade",
                error=str(exc),
            )
            state = await self._risk_manager.get_state()
            return state.balance

    async def _find_trade_for_position(self, position: Position) -> Trade:
        """
        Reconstruct a minimal Trade object from a Position record so that
        the paper trader's close_position method can compute PnL correctly.
        """
        from src.core.database import TradeRepository
        repo = TradeRepository(self._db)
        open_trades = await repo.get_open_trades()
        for trade in open_trades:
            if trade.token_id == position.token_id and trade.market_id == position.market_id:
                return trade

        # Fallback: synthesise a Trade from the Position fields
        return Trade(
            trade_id=str(uuid.uuid4()),
            market_id=position.market_id,
            condition_id=position.condition_id,
            token_id=position.token_id,
            direction=position.direction,
            size=position.size,
            entry_price=position.entry_price,
            entry_time=position.entry_time,
            confidence=position.confidence,
            paper_trade=self._settings.paper_trading,
        )
