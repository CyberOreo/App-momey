"""SQLAlchemy 2.0 async database layer — engines, session factory, ORM models, repositories."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncIterator, Dict, List, Optional, Tuple

from loguru import logger
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    select,
    update,
)
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, mapped_column, Mapped

from src.core.models import (
    Candle,
    Direction,
    Market,
    PolymarketToken,
    Trade,
    TradeOutcome,
)


# ── ORM base ─────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── ORM table definitions ─────────────────────────────────────────────────────

class CandleRow(Base):
    __tablename__ = "candles"
    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "timestamp", name="uq_candle_symbol_tf_ts"),
        Index("ix_candles_symbol_tf_ts", "symbol", "timeframe", "timestamp"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(10), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class MarketRow(Base):
    __tablename__ = "markets"
    __table_args__ = (
        UniqueConstraint("condition_id", name="uq_market_condition_id"),
        Index("ix_markets_active", "active"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    condition_id: Mapped[str] = mapped_column(String(100), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    yes_token_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    no_token_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    end_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    volume: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    liquidity: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    last_updated: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class TradeRow(Base):
    __tablename__ = "trades"
    __table_args__ = (
        UniqueConstraint("trade_id", name="uq_trade_trade_id"),
        Index("ix_trades_outcome", "outcome"),
        Index("ix_trades_entry_time", "entry_time"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_id: Mapped[str] = mapped_column(String(100), nullable=False)
    market_id: Mapped[str] = mapped_column(String(100), nullable=False)
    condition_id: Mapped[str] = mapped_column(String(100), nullable=False)
    token_id: Mapped[str] = mapped_column(String(100), nullable=False)
    direction: Mapped[str] = mapped_column(String(10), nullable=False)
    size: Mapped[float] = mapped_column(Float, nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    entry_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    exit_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    exit_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    realized_pnl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    outcome: Mapped[str] = mapped_column(String(20), default=TradeOutcome.OPEN.value, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    paper_trade: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    signal_reasons: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON array
    fees_paid: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class PositionRow(Base):
    __tablename__ = "positions"
    __table_args__ = (
        UniqueConstraint("position_id", name="uq_position_position_id"),
        Index("ix_positions_open", "open"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    position_id: Mapped[str] = mapped_column(String(100), nullable=False)
    market_id: Mapped[str] = mapped_column(String(100), nullable=False)
    condition_id: Mapped[str] = mapped_column(String(100), nullable=False)
    token_id: Mapped[str] = mapped_column(String(100), nullable=False)
    direction: Mapped[str] = mapped_column(String(10), nullable=False)
    size: Mapped[float] = mapped_column(Float, nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    current_price: Mapped[float] = mapped_column(Float, nullable=False)
    entry_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    stop_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    take_profit: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    open: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    closed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class RiskEventRow(Base):
    __tablename__ = "risk_events"
    __table_args__ = (
        Index("ix_risk_events_timestamp", "timestamp"),
        Index("ix_risk_events_event_type", "event_type"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    balance_at_event: Mapped[float] = mapped_column(Float, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


# ── Engine / session factory ──────────────────────────────────────────────────

def _build_engine(url: str) -> AsyncEngine:
    """Create an async engine with dialect-appropriate options."""
    if url.startswith("sqlite"):
        return create_async_engine(
            url,
            echo=False,
            connect_args={"check_same_thread": False},
        )
    # PostgreSQL — use a connection pool sized for production async usage
    return create_async_engine(
        url,
        echo=False,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=3600,
    )


async def init_db(url: str) -> AsyncEngine:
    """Create all tables and return the engine."""
    engine = _build_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialised", url=url)
    return engine


_session_factories: Dict[int, async_sessionmaker[AsyncSession]] = {}


def _get_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    key = id(engine)
    if key not in _session_factories:
        _session_factories[key] = async_sessionmaker(
            engine,
            expire_on_commit=False,
            class_=AsyncSession,
        )
    return _session_factories[key]


@asynccontextmanager
async def get_session(engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    """Yield an AsyncSession that is automatically committed or rolled-back."""
    factory = _get_factory(engine)
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ── CandleRepository ──────────────────────────────────────────────────────────

class CandleRepository:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def upsert_candles(self, symbol: str, candles: List[Candle]) -> int:
        """
        Insert or update candles for *symbol*.
        Uses a manual upsert so it works for both SQLite and PostgreSQL.
        Returns the number of rows written.
        """
        if not candles:
            return 0

        written = 0
        async with get_session(self._engine) as session:
            for candle in candles:
                # Try to find an existing row
                stmt = select(CandleRow).where(
                    CandleRow.symbol == symbol,
                    CandleRow.timeframe == candle.timeframe,
                    CandleRow.timestamp == candle.timestamp,
                )
                result = await session.execute(stmt)
                row = result.scalar_one_or_none()

                if row is None:
                    row = CandleRow(
                        symbol=symbol,
                        timeframe=candle.timeframe,
                        timestamp=candle.timestamp,
                        open=candle.open,
                        high=candle.high,
                        low=candle.low,
                        close=candle.close,
                        volume=candle.volume,
                        created_at=datetime.utcnow(),
                    )
                    session.add(row)
                else:
                    row.open = candle.open
                    row.high = candle.high
                    row.low = candle.low
                    row.close = candle.close
                    row.volume = candle.volume

                written += 1

        logger.debug("Upserted candles", symbol=symbol, count=written)
        return written

    async def get_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
        since: Optional[datetime] = None,
    ) -> List[Candle]:
        """Return candles ordered oldest-first."""
        async with get_session(self._engine) as session:
            stmt = select(CandleRow).where(
                CandleRow.symbol == symbol,
                CandleRow.timeframe == timeframe,
            )
            if since is not None:
                stmt = stmt.where(CandleRow.timestamp >= since)
            stmt = stmt.order_by(CandleRow.timestamp.desc()).limit(limit)

            result = await session.execute(stmt)
            rows = result.scalars().all()

        # Return in chronological order
        return [
            Candle(
                timestamp=r.timestamp,
                open=r.open,
                high=r.high,
                low=r.low,
                close=r.close,
                volume=r.volume,
                timeframe=r.timeframe,
            )
            for r in reversed(rows)
        ]


# ── TradeRepository ───────────────────────────────────────────────────────────

class TradeRepository:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    def _row_to_trade(self, row: TradeRow) -> Trade:
        reasons: List[str] = []
        if row.signal_reasons:
            try:
                reasons = json.loads(row.signal_reasons)
            except (json.JSONDecodeError, TypeError):
                reasons = []

        return Trade(
            trade_id=row.trade_id,
            market_id=row.market_id,
            condition_id=row.condition_id,
            token_id=row.token_id,
            direction=Direction(row.direction),
            size=row.size,
            entry_price=row.entry_price,
            entry_time=row.entry_time,
            confidence=row.confidence,
            signal_reasons=reasons,
            paper_trade=row.paper_trade,
            exit_price=row.exit_price,
            realized_pnl=row.realized_pnl,
            exit_time=row.exit_time,
            outcome=TradeOutcome(row.outcome),
            fees_paid=row.fees_paid,
        )

    async def save_trade(self, trade: Trade) -> None:
        """Persist a new trade record."""
        async with get_session(self._engine) as session:
            row = TradeRow(
                trade_id=trade.trade_id,
                market_id=trade.market_id,
                condition_id=trade.condition_id,
                token_id=trade.token_id,
                direction=trade.direction.value,
                size=trade.size,
                entry_price=trade.entry_price,
                entry_time=trade.entry_time,
                exit_price=trade.exit_price,
                exit_time=trade.exit_time,
                realized_pnl=trade.realized_pnl,
                outcome=trade.outcome.value,
                confidence=trade.confidence,
                paper_trade=trade.paper_trade,
                signal_reasons=json.dumps(trade.signal_reasons),
                fees_paid=trade.fees_paid,
                created_at=datetime.utcnow(),
            )
            session.add(row)
        logger.debug("Trade saved", trade_id=trade.trade_id)

    async def update_trade(self, trade: Trade) -> None:
        """Update mutable fields of an existing trade (exit, PnL, outcome)."""
        async with get_session(self._engine) as session:
            stmt = (
                update(TradeRow)
                .where(TradeRow.trade_id == trade.trade_id)
                .values(
                    exit_price=trade.exit_price,
                    exit_time=trade.exit_time,
                    realized_pnl=trade.realized_pnl,
                    outcome=trade.outcome.value,
                    fees_paid=trade.fees_paid,
                )
            )
            await session.execute(stmt)
        logger.debug("Trade updated", trade_id=trade.trade_id, outcome=trade.outcome.value)

    async def get_open_trades(self) -> List[Trade]:
        """Return all trades with outcome == OPEN."""
        async with get_session(self._engine) as session:
            stmt = select(TradeRow).where(TradeRow.outcome == TradeOutcome.OPEN.value)
            result = await session.execute(stmt)
            rows = result.scalars().all()
        return [self._row_to_trade(r) for r in rows]

    async def get_all_trades(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> List[Trade]:
        """Return all trades, optionally bounded by time window."""
        async with get_session(self._engine) as session:
            stmt = select(TradeRow).order_by(TradeRow.entry_time)
            if since is not None:
                stmt = stmt.where(TradeRow.entry_time >= since)
            if until is not None:
                stmt = stmt.where(TradeRow.entry_time <= until)
            result = await session.execute(stmt)
            rows = result.scalars().all()
        return [self._row_to_trade(r) for r in rows]


# ── MarketRepository ──────────────────────────────────────────────────────────

class MarketRepository:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    def _row_to_market(self, row: MarketRow) -> Market:
        tokens: List[PolymarketToken] = []
        if row.yes_token_id:
            tokens.append(PolymarketToken(token_id=row.yes_token_id, outcome="Yes", price=0.5))
        if row.no_token_id:
            tokens.append(PolymarketToken(token_id=row.no_token_id, outcome="No", price=0.5))

        return Market(
            condition_id=row.condition_id,
            question=row.question,
            tokens=tokens,
            end_date=row.end_date or datetime.utcnow(),
            active=row.active,
            volume=row.volume,
            liquidity=row.liquidity,
        )

    async def upsert_market(self, market: Market) -> None:
        """Insert or update a market record."""
        yes_id: Optional[str] = market.yes_token.token_id if market.yes_token else None
        no_id: Optional[str] = market.no_token.token_id if market.no_token else None

        async with get_session(self._engine) as session:
            stmt = select(MarketRow).where(MarketRow.condition_id == market.condition_id)
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()

            if row is None:
                row = MarketRow(
                    condition_id=market.condition_id,
                    question=market.question,
                    yes_token_id=yes_id,
                    no_token_id=no_id,
                    end_date=market.end_date,
                    active=market.active,
                    volume=market.volume,
                    liquidity=market.liquidity,
                    last_updated=datetime.utcnow(),
                )
                session.add(row)
            else:
                row.question = market.question
                row.yes_token_id = yes_id
                row.no_token_id = no_id
                row.end_date = market.end_date
                row.active = market.active
                row.volume = market.volume
                row.liquidity = market.liquidity
                row.last_updated = datetime.utcnow()

        logger.debug("Market upserted", condition_id=market.condition_id)

    async def get_active_markets(self) -> List[Market]:
        """Return all markets where active=True."""
        async with get_session(self._engine) as session:
            stmt = select(MarketRow).where(MarketRow.active == True)  # noqa: E712
            result = await session.execute(stmt)
            rows = result.scalars().all()
        return [self._row_to_market(r) for r in rows]
