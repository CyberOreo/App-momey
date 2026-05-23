"""
Binance perpetual futures funding rate + mark price tracker.

Funding rate insight:
  Positive funding (>+0.01%) = market is too long = price tends to fall
  Negative funding (<-0.01%) = market is too short = price tends to spike
  Extreme funding (>+0.05%) = imminent long squeeze, strong bearish signal

Mark price vs index price gap (basis) also matters:
  Futures trading above spot = bullish institutional positioning
  Futures below spot = bearish / hedging dominant
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Callable, Deque, Optional
from collections import deque

import websockets
from loguru import logger


@dataclass
class FundingSnapshot:
    mark_price: float
    funding_rate: float    # e.g. 0.0001 = 0.01% per 8-hour period
    index_price: float
    next_funding_time: int  # unix ms
    ts: float = field(default_factory=time.time)

    @property
    def age(self) -> float:
        return time.time() - self.ts

    @property
    def basis_pct(self) -> float:
        """Futures premium over spot. Positive = contango (bullish)."""
        if self.index_price == 0:
            return 0.0
        return (self.mark_price - self.index_price) / self.index_price

    @property
    def signal(self) -> float:
        """
        Directional signal from funding rate.
        Positive funding = over-leveraged longs = bearish pressure.
        Range: -1.0 to +1.0 where +1.0 = strong bearish (too many longs).
        """
        # Normalize: 0.03% funding per 8h is extreme
        return max(-1.0, min(1.0, self.funding_rate / 0.0003))

    @property
    def minutes_to_funding(self) -> float:
        """Minutes until next funding payment."""
        return max(0, (self.next_funding_time / 1000 - time.time()) / 60)


class FundingRateTracker:
    """
    Streams Binance USDT-M perpetual mark price and funding rate for BTCUSDT.

    The markPrice stream updates every 3 seconds with:
    - Mark price (used for liquidations)
    - Index price (spot composite)
    - Funding rate (current rate applied next payment)
    - Next funding time

    Usage:
        tracker = FundingRateTracker()
        await tracker.start()
        signal = tracker.directional_signal  # -1 to +1
    """

    STREAM_URL = "wss://fstream.binance.com/ws/btcusdt@markPrice"

    def __init__(
        self,
        on_update: Optional[Callable[[FundingSnapshot], None]] = None,
    ):
        self._on_update = on_update
        self._latest: Optional[FundingSnapshot] = None
        self._history: Deque[FundingSnapshot] = deque(maxlen=100)
        self._running = False
        self._task: Optional[asyncio.Task] = None

    @property
    def latest(self) -> Optional[FundingSnapshot]:
        return self._latest

    @property
    def is_ready(self) -> bool:
        return self._latest is not None and self._latest.age < 30

    @property
    def directional_signal(self) -> float:
        """
        Combined signal: funding rate + basis.
        Positive = bearish pressure (too many longs, or futures premium).
        Negative = bullish pressure (too many shorts, or futures discount).
        Returns 0.0 if no fresh data.
        """
        if not self.is_ready:
            return 0.0
        snap = self._latest
        # Funding signal: positive funding → bearish
        funding_component = snap.signal * 0.7
        # Basis component: positive basis (futures > spot) is slightly bullish
        basis_component = -snap.basis_pct * 10 * 0.3  # scale and invert
        return max(-1.0, min(1.0, funding_component + basis_component))

    @property
    def funding_rate(self) -> float:
        if self._latest is None:
            return 0.0
        return self._latest.funding_rate

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("[FUNDING] Funding rate tracker started")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("[FUNDING] Funding rate tracker stopped")

    async def _loop(self) -> None:
        delay = 1.0
        while self._running:
            try:
                async with websockets.connect(
                    self.STREAM_URL,
                    ping_interval=20,
                    ping_timeout=10,
                ) as ws:
                    logger.debug("[FUNDING] Stream connected")
                    delay = 1.0
                    async for raw in ws:
                        if not self._running:
                            break
                        try:
                            msg = json.loads(raw)
                            snap = FundingSnapshot(
                                mark_price=float(msg["p"]),
                                funding_rate=float(msg["r"]),
                                index_price=float(msg["i"]),
                                next_funding_time=int(msg.get("T", 0)),
                            )
                            self._latest = snap
                            self._history.append(snap)
                            if self._on_update:
                                self._on_update(snap)
                        except Exception as e:
                            logger.debug(f"[FUNDING] Parse error: {e}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._running:
                    logger.debug(f"[FUNDING] Reconnect in {delay:.0f}s: {e}")
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 30.0)

    def funding_trend(self) -> float:
        """
        Is funding rate rising or falling?
        Positive = rising (more longs piling in = bearish).
        """
        if len(self._history) < 10:
            return 0.0
        recent = list(self._history)[-10:]
        old_rate = sum(s.funding_rate for s in recent[:5]) / 5
        new_rate = sum(s.funding_rate for s in recent[5:]) / 5
        return new_rate - old_rate
