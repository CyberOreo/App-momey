"""
Real-time order flow analysis from Binance trade stream.

Tracks buy vs sell pressure at the tick level — the single best
short-term predictor for 5-minute price direction.
"""
from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Deque, Optional

import websockets
from loguru import logger


@dataclass
class Trade:
    price: float
    qty: float
    is_buyer_maker: bool   # True = seller initiated (sell trade)
    timestamp: float       # unix ms


@dataclass
class OrderFlowSnapshot:
    """Rolling window order flow metrics."""
    window_seconds: int
    buy_volume: float
    sell_volume: float
    buy_count: int
    sell_count: int
    total_volume: float
    large_buy_volume: float    # trades > large_trade_threshold
    large_sell_volume: float
    timestamp: datetime

    @property
    def buy_pressure(self) -> float:
        """0.0 = all selling, 1.0 = all buying, 0.5 = balanced."""
        if self.total_volume == 0:
            return 0.5
        return self.buy_volume / self.total_volume

    @property
    def delta(self) -> float:
        """Buy volume minus sell volume. Positive = buying pressure."""
        return self.buy_volume - self.sell_volume

    @property
    def delta_pct(self) -> float:
        """Delta as % of total volume."""
        if self.total_volume == 0:
            return 0.0
        return self.delta / self.total_volume

    @property
    def large_trade_delta(self) -> float:
        """Smart money delta — large trades only."""
        total = self.large_buy_volume + self.large_sell_volume
        if total == 0:
            return 0.0
        return (self.large_buy_volume - self.large_sell_volume) / total

    @property
    def signal_strength(self) -> float:
        """
        -1.0 = strong sell pressure, +1.0 = strong buy pressure.
        Weighted: large trades count more than small retail trades.
        """
        retail_signal = self.delta_pct              # -1 to +1
        smart_signal = self.large_trade_delta        # -1 to +1
        return retail_signal * 0.35 + smart_signal * 0.65

    @property
    def direction(self) -> str:
        s = self.signal_strength
        if s > 0.15:
            return "bullish"
        if s < -0.15:
            return "bearish"
        return "neutral"


class OrderFlowAnalyzer:
    """
    Connects to Binance trade stream for BTCUSDT and maintains a
    rolling buffer of individual trades for order flow analysis.
    """

    STREAM_URL = "wss://stream.binance.com:9443/ws/btcusdt@aggTrade"

    def __init__(
        self,
        large_trade_usdc: float = 50_000,   # trades above this = "smart money"
        on_snapshot: Optional[Callable[[OrderFlowSnapshot], None]] = None,
    ):
        self._large_threshold = large_trade_usdc
        self._on_snapshot = on_snapshot
        self._trades: Deque[Trade] = deque(maxlen=10_000)
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._reconnect_delay = 1.0

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._connection_loop())
        logger.info("OrderFlowAnalyzer started")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("OrderFlowAnalyzer stopped")

    async def _connection_loop(self) -> None:
        delay = self._reconnect_delay
        while self._running:
            try:
                async with websockets.connect(
                    self.STREAM_URL,
                    ping_interval=20,
                    ping_timeout=10,
                ) as ws:
                    logger.info("Order flow stream connected")
                    delay = self._reconnect_delay
                    async for raw in ws:
                        if not self._running:
                            break
                        try:
                            self._handle(json.loads(raw))
                        except Exception as e:
                            logger.debug(f"Order flow parse error: {e}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._running:
                    logger.warning(f"Order flow stream error: {e} — reconnecting in {delay}s")
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 60.0)

    def _handle(self, msg: dict) -> None:
        """Parse aggTrade event and store."""
        # aggTrade fields: p=price, q=qty, m=is_buyer_maker, T=trade_time
        price = float(msg["p"])
        qty = float(msg["q"])
        is_buyer_maker = bool(msg["m"])  # True = sell trade
        ts = float(msg["T"])

        trade = Trade(price=price, qty=qty, is_buyer_maker=is_buyer_maker, timestamp=ts)
        self._trades.append(trade)

        if self._on_snapshot and len(self._trades) % 20 == 0:
            snap = self.snapshot(30)
            if snap:
                self._on_snapshot(snap)

    def snapshot(self, window_seconds: int = 60) -> Optional[OrderFlowSnapshot]:
        """Compute order flow metrics over the last N seconds."""
        if not self._trades:
            return None

        cutoff_ms = (time.time() - window_seconds) * 1000
        window = [t for t in self._trades if t.timestamp >= cutoff_ms]
        if not window:
            return None

        buy_vol = sell_vol = 0.0
        buy_cnt = sell_cnt = 0
        lg_buy = lg_sell = 0.0

        for t in window:
            usdc_val = t.price * t.qty
            if t.is_buyer_maker:        # seller initiated
                sell_vol += t.qty
                sell_cnt += 1
                if usdc_val >= self._large_threshold:
                    lg_sell += usdc_val
            else:                        # buyer initiated
                buy_vol += t.qty
                buy_cnt += 1
                if usdc_val >= self._large_threshold:
                    lg_buy += usdc_val

        return OrderFlowSnapshot(
            window_seconds=window_seconds,
            buy_volume=buy_vol,
            sell_volume=sell_vol,
            buy_count=buy_cnt,
            sell_count=sell_cnt,
            total_volume=buy_vol + sell_vol,
            large_buy_volume=lg_buy,
            large_sell_volume=lg_sell,
            timestamp=datetime.utcnow(),
        )

    def multi_snapshot(self) -> dict[str, Optional[OrderFlowSnapshot]]:
        """Return snapshots for 30s, 60s, and 120s windows simultaneously."""
        return {
            "30s": self.snapshot(30),
            "60s": self.snapshot(60),
            "120s": self.snapshot(120),
        }

    def is_ready(self) -> bool:
        """True once we have at least 30 seconds of data."""
        if not self._trades:
            return False
        age_s = (time.time() * 1000 - self._trades[0].timestamp) / 1000
        return age_s >= 30

    def recent_price_velocity(self, window_seconds: int = 60) -> float:
        """
        Price velocity: (last_price - first_price) / first_price over window.
        Positive = price rising, negative = falling.
        """
        cutoff_ms = (time.time() - window_seconds) * 1000
        window = [t for t in self._trades if t.timestamp >= cutoff_ms]
        if len(window) < 2:
            return 0.0
        return (window[-1].price - window[0].price) / window[0].price

    def acceleration(self) -> float:
        """
        Compare velocity in first half vs second half of the last 60 seconds.
        Positive = move accelerating (momentum), negative = decelerating.
        """
        now_ms = time.time() * 1000
        cutoff_ms = now_ms - 60_000
        mid_ms = now_ms - 30_000

        first_half = [t for t in self._trades if cutoff_ms <= t.timestamp < mid_ms]
        second_half = [t for t in self._trades if t.timestamp >= mid_ms]

        if len(first_half) < 2 or len(second_half) < 2:
            return 0.0

        v1 = (first_half[-1].price - first_half[0].price) / first_half[0].price
        v2 = (second_half[-1].price - second_half[0].price) / second_half[0].price
        return v2 - v1   # positive = speeding up
