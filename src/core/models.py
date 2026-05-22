"""Shared domain models used across all modules."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


# ── Enumerations ──────────────────────────────────────────────────────────────

class Direction(str, Enum):
    YES = "YES"
    NO = "NO"


class MarketConditionType(str, Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    CHOPPY = "choppy"
    HIGH_VOLATILITY = "high_volatility"
    LOW_LIQUIDITY = "low_liquidity"


class VolatilityRegime(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class TradeOutcome(str, Enum):
    WIN = "win"
    LOSS = "loss"
    BREAK_EVEN = "break_even"
    OPEN = "open"


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


# ── Price / OHLCV ─────────────────────────────────────────────────────────────

@dataclass
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str = "1m"


@dataclass
class BTCPrice:
    price: float
    timestamp: datetime
    source: str = "binance"


# ── Order book ────────────────────────────────────────────────────────────────

@dataclass
class OrderBookLevel:
    price: float
    size: float


@dataclass
class OrderBook:
    token_id: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    timestamp: datetime

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None

    @property
    def spread_pct(self) -> Optional[float]:
        mid = self.mid_price
        if mid and mid > 0:
            return self.spread / mid  # type: ignore[operator]
        return None

    @property
    def bid_liquidity(self) -> float:
        return sum(b.price * b.size for b in self.bids)

    @property
    def ask_liquidity(self) -> float:
        return sum(a.price * a.size for a in self.asks)

    @property
    def total_liquidity(self) -> float:
        return self.bid_liquidity + self.ask_liquidity


# ── Polymarket markets ────────────────────────────────────────────────────────

@dataclass
class PolymarketToken:
    token_id: str
    outcome: str     # "Yes" or "No"
    price: float
    winner: Optional[bool] = None


@dataclass
class Market:
    condition_id: str
    question: str
    tokens: List[PolymarketToken]
    end_date: datetime
    active: bool
    volume: float = 0.0
    liquidity: float = 0.0
    description: str = ""
    image: str = ""
    slug: str = ""

    @property
    def yes_token(self) -> Optional[PolymarketToken]:
        for t in self.tokens:
            if t.outcome.lower() == "yes":
                return t
        return None

    @property
    def no_token(self) -> Optional[PolymarketToken]:
        for t in self.tokens:
            if t.outcome.lower() == "no":
                return t
        return None

    @property
    def hours_to_resolution(self) -> float:
        delta = self.end_date - datetime.utcnow()
        return max(0.0, delta.total_seconds() / 3600)

    @property
    def yes_implied_prob(self) -> float:
        t = self.yes_token
        return t.price if t else 0.5

    @property
    def no_implied_prob(self) -> float:
        t = self.no_token
        return t.price if t else 0.5


# ── Technical indicators ──────────────────────────────────────────────────────

@dataclass
class IndicatorSet:
    timestamp: datetime
    timeframe: str
    close: float
    ema_20: float
    ema_50: float
    ema_200: float
    rsi: float
    macd: float
    macd_signal: float
    macd_histogram: float
    atr: float
    atr_pct: float          # ATR / close
    volume_ma: float
    volume_ratio: float     # current_volume / volume_ma
    momentum: float         # rate of change, 10-bar


# ── Market conditions ─────────────────────────────────────────────────────────

@dataclass
class MarketCondition:
    condition: MarketConditionType
    trend_direction: Optional[str]          # 'up' | 'down' | None
    volatility_regime: VolatilityRegime
    trend_strength: float                   # 0–1
    confidence: float                       # 0–1
    timestamp: datetime


# ── Signals ───────────────────────────────────────────────────────────────────

@dataclass
class TimeframeScore:
    timeframe: str
    trend_score: float      # -1 bullish … +1 bearish (YES-biased = -1)
    momentum_score: float   # -1 … +1
    volume_score: float     # 0 … 1 (confirmation strength)
    overall_score: float    # composite -1 … +1


@dataclass
class TradeSignal:
    market_id: str
    direction: Direction
    token_id: str
    confidence: float                           # 0–100
    price: float                                # token price (0–1)
    reasons: List[str]
    timeframe_scores: Dict[str, TimeframeScore]
    market_condition: MarketCondition
    indicators: IndicatorSet
    timestamp: datetime
    implied_probability: float                  # from Polymarket price
    fair_value_estimate: float                  # our model estimate
    edge: float                                 # fair_value – implied_prob


# ── Position sizing ───────────────────────────────────────────────────────────

@dataclass
class PositionSizing:
    recommended_size_usdc: float
    max_size_usdc: float
    risk_amount_usdc: float
    kelly_fraction: float
    method: str             # 'kelly' | 'fixed_pct' | 'fixed_usdc'


# ── Orders & positions ────────────────────────────────────────────────────────

@dataclass
class Order:
    order_id: str
    market_id: str
    token_id: str
    side: str               # 'BUY' | 'SELL'
    price: float
    size: float
    status: OrderStatus
    timestamp: datetime
    filled_size: float = 0.0
    average_fill_price: float = 0.0


@dataclass
class Position:
    position_id: str
    market_id: str
    condition_id: str
    token_id: str
    direction: Direction
    size: float
    entry_price: float
    current_price: float
    entry_time: datetime
    confidence: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    @property
    def unrealized_pnl(self) -> float:
        return self.size * (self.current_price - self.entry_price)

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price

    @property
    def cost_basis(self) -> float:
        return self.size * self.entry_price

    @property
    def current_value(self) -> float:
        return self.size * self.current_price


# ── Trade record ──────────────────────────────────────────────────────────────

@dataclass
class Trade:
    trade_id: str
    market_id: str
    condition_id: str
    token_id: str
    direction: Direction
    size: float
    entry_price: float
    entry_time: datetime
    confidence: float
    signal_reasons: List[str] = field(default_factory=list)
    paper_trade: bool = True
    exit_price: Optional[float] = None
    realized_pnl: Optional[float] = None
    exit_time: Optional[datetime] = None
    outcome: TradeOutcome = TradeOutcome.OPEN
    fees_paid: float = 0.0

    @property
    def holding_hours(self) -> Optional[float]:
        if self.exit_time:
            return (self.exit_time - self.entry_time).total_seconds() / 3600
        return None


# ── Risk state ────────────────────────────────────────────────────────────────

@dataclass
class RiskState:
    balance: float
    start_of_day_balance: float
    daily_pnl: float
    consecutive_losses: int
    open_positions: int
    total_exposure_usdc: float
    in_cooldown: bool
    cooldown_until: Optional[datetime]
    kill_switch_active: bool
    kill_switch_reason: str = ""

    @property
    def daily_pnl_pct(self) -> float:
        if self.start_of_day_balance == 0:
            return 0.0
        return self.daily_pnl / self.start_of_day_balance

    @property
    def daily_drawdown_pct(self) -> float:
        return max(0.0, -self.daily_pnl_pct)

    @property
    def total_exposure_pct(self) -> float:
        if self.balance == 0:
            return 0.0
        return self.total_exposure_usdc / self.balance

    @property
    def available_capital(self) -> float:
        return max(0.0, self.balance - self.total_exposure_usdc)

    @property
    def can_trade(self) -> bool:
        return not self.kill_switch_active and not self.in_cooldown


# ── Performance analytics ─────────────────────────────────────────────────────

@dataclass
class PerformanceMetrics:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_pct: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    calmar_ratio: float
    avg_holding_time_hours: float
    start_date: datetime
    end_date: datetime
