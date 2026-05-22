"""Position state manager.

Keeps a live in-memory mirror of all open positions (for O(1) lookups and
stop/TP checks) backed by the SQLAlchemy PositionRow table for persistence
across restarts.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, List, Optional

from loguru import logger
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncEngine

from src.core.database import PositionRow, get_session
from src.core.models import Direction, Position, Trade


class PositionManager:
    """
    Async manager for open positions.

    Maintains an in-memory cache that is the source of truth for hot-path
    operations (stop/TP checks, price updates).  All writes are also
    persisted to the database so the state survives restarts.

    The cache is bootstrapped from the DB on the first call to
    :meth:`get_open_positions` or by explicitly calling :meth:`load_from_db`.
    """

    def __init__(self, db: AsyncEngine, settings) -> None:
        self._db = db
        self._settings = settings
        # position_id → Position
        self._open: Dict[str, Position] = {}
        self._loaded = False   # True once the DB has been read into the cache

    # ── bootstrap ─────────────────────────────────────────────────────────────

    async def load_from_db(self) -> None:
        """Populate the in-memory cache from all open positions in the DB."""
        async with get_session(self._db) as session:
            stmt = select(PositionRow).where(PositionRow.open == True)  # noqa: E712
            result = await session.execute(stmt)
            rows = result.scalars().all()

        self._open = {}
        for row in rows:
            pos = self._row_to_position(row)
            self._open[pos.position_id] = pos

        self._loaded = True
        logger.info("PositionManager loaded from DB", open_count=len(self._open))

    async def _ensure_loaded(self) -> None:
        if not self._loaded:
            await self.load_from_db()

    # ── writes ────────────────────────────────────────────────────────────────

    async def add_position(self, trade: Trade) -> Position:
        """
        Create a new Position from an executed trade and persist it.

        Returns the newly created Position object.
        """
        await self._ensure_loaded()

        position_id = str(uuid.uuid4())
        position = Position(
            position_id=position_id,
            market_id=trade.market_id,
            condition_id=trade.condition_id,
            token_id=trade.token_id,
            direction=trade.direction,
            size=trade.size,
            entry_price=trade.entry_price,
            current_price=trade.entry_price,   # starts at entry
            entry_time=trade.entry_time,
            confidence=trade.confidence,
            stop_loss=None,     # set by executor after creation
            take_profit=None,
        )

        async with get_session(self._db) as session:
            row = PositionRow(
                position_id=position_id,
                market_id=position.market_id,
                condition_id=position.condition_id,
                token_id=position.token_id,
                direction=position.direction.value,
                size=position.size,
                entry_price=position.entry_price,
                current_price=position.current_price,
                entry_time=position.entry_time,
                confidence=position.confidence,
                stop_loss=position.stop_loss,
                take_profit=position.take_profit,
                open=True,
                closed_at=None,
            )
            session.add(row)

        self._open[position_id] = position
        logger.info(
            "Position added",
            position_id=position_id,
            market=position.market_id,
            direction=position.direction.value,
            size=position.size,
            entry_price=position.entry_price,
        )
        return position

    async def update_stop_take(self, position: Position) -> None:
        """Persist stop_loss and take_profit changes to the DB."""
        async with get_session(self._db) as session:
            stmt = (
                update(PositionRow)
                .where(PositionRow.position_id == position.position_id)
                .values(
                    stop_loss=position.stop_loss,
                    take_profit=position.take_profit,
                )
            )
            await session.execute(stmt)

        # Update in-memory cache
        if position.position_id in self._open:
            self._open[position.position_id].stop_loss = position.stop_loss
            self._open[position.position_id].take_profit = position.take_profit

    async def update_price(self, token_id: str, new_price: float) -> None:
        """
        Update the current_price for every open position holding *token_id*.

        Also persists the change to the DB so that dashboards see live values.
        The DB write is batched per token — one UPDATE per token_id invocation.
        """
        await self._ensure_loaded()

        affected_ids = [
            pid
            for pid, pos in self._open.items()
            if pos.token_id == token_id
        ]

        if not affected_ids:
            return

        for pid in affected_ids:
            self._open[pid].current_price = new_price

        async with get_session(self._db) as session:
            stmt = (
                update(PositionRow)
                .where(
                    PositionRow.token_id == token_id,
                    PositionRow.open == True,  # noqa: E712
                )
                .values(current_price=new_price)
            )
            await session.execute(stmt)

    async def close_position(self, position: Position, exit_price: float) -> None:
        """
        Mark a position as closed in both the in-memory cache and the DB.
        """
        await self._ensure_loaded()

        closed_at = datetime.utcnow()

        async with get_session(self._db) as session:
            stmt = (
                update(PositionRow)
                .where(PositionRow.position_id == position.position_id)
                .values(
                    open=False,
                    current_price=exit_price,
                    closed_at=closed_at,
                )
            )
            await session.execute(stmt)

        self._open.pop(position.position_id, None)
        logger.info(
            "Position closed in manager",
            position_id=position.position_id,
            exit_price=exit_price,
        )

    # ── reads ─────────────────────────────────────────────────────────────────

    async def get_open_positions(self) -> List[Position]:
        """Return all currently open positions (from in-memory cache)."""
        await self._ensure_loaded()
        return list(self._open.values())

    async def get_position(self, position_id: str) -> Optional[Position]:
        """Return a specific open position by ID, or None if not found."""
        await self._ensure_loaded()
        return self._open.get(position_id)

    async def get_total_exposure(self) -> float:
        """
        Return the total USDC cost-basis across all open positions.

        This is the sum of (size × entry_price) for each open position —
        i.e. how much capital is currently deployed.
        """
        await self._ensure_loaded()
        return sum(pos.cost_basis for pos in self._open.values())

    async def count_open(self) -> int:
        """Return the number of currently open positions."""
        await self._ensure_loaded()
        return len(self._open)

    # ── stop-loss / take-profit scanning ─────────────────────────────────────

    async def check_stop_losses(
        self, current_prices: Dict[str, float]
    ) -> List[Position]:
        """
        Identify positions whose current price has breached their stop-loss.

        Returns a list of Position objects that need to be closed immediately.
        Does NOT close them — the caller (TradeExecutor) handles the close.
        """
        await self._ensure_loaded()
        triggered: List[Position] = []

        for pos in self._open.values():
            if pos.stop_loss is None:
                continue
            price = current_prices.get(pos.token_id)
            if price is None:
                continue
            if price <= pos.stop_loss:
                logger.warning(
                    "Stop-loss triggered",
                    position_id=pos.position_id,
                    current_price=price,
                    stop_loss=pos.stop_loss,
                    unrealized_pnl=round(pos.size * (price - pos.entry_price), 4),
                )
                triggered.append(pos)

        return triggered

    async def check_take_profits(
        self, current_prices: Dict[str, float]
    ) -> List[Position]:
        """
        Identify positions whose current price has reached their take-profit.

        Returns a list of Position objects ready to be closed for a win.
        Does NOT close them — the caller handles the close.
        """
        await self._ensure_loaded()
        triggered: List[Position] = []

        for pos in self._open.values():
            if pos.take_profit is None:
                continue
            price = current_prices.get(pos.token_id)
            if price is None:
                continue
            if price >= pos.take_profit:
                logger.info(
                    "Take-profit triggered",
                    position_id=pos.position_id,
                    current_price=price,
                    take_profit=pos.take_profit,
                    unrealized_pnl=round(pos.size * (price - pos.entry_price), 4),
                )
                triggered.append(pos)

        return triggered

    # ── internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _row_to_position(row: PositionRow) -> Position:
        return Position(
            position_id=row.position_id,
            market_id=row.market_id,
            condition_id=row.condition_id,
            token_id=row.token_id,
            direction=Direction(row.direction),
            size=row.size,
            entry_price=row.entry_price,
            current_price=row.current_price,
            entry_time=row.entry_time,
            confidence=row.confidence,
            stop_loss=row.stop_loss,
            take_profit=row.take_profit,
        )
