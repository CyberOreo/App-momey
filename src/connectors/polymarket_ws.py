"""
Polymarket CLOB real-time data stream (RTDS).

Connects to Polymarket's live WebSocket to receive orderbook updates
instead of polling REST every 2 seconds. Latency: ~2000ms → ~50ms.

Subscribe to YES and NO token IDs for each active 5-min market.
On each price_change event, rebuilds a BookSnapshot and calls on_book_update.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

import websockets
from loguru import logger


RTDS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


@dataclass
class TokenBook:
    """One side of the market (YES or NO token)."""
    token_id: str
    bid: float = 0.0
    ask: float = 1.0
    bid_size: float = 0.0
    ask_size: float = 0.0
    ts: float = field(default_factory=time.time)

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def microprice(self) -> float:
        total = self.bid_size + self.ask_size
        if total == 0:
            return self.mid
        return (self.bid * self.ask_size + self.ask * self.bid_size) / total


@dataclass
class LiveBookSnapshot:
    """Combined YES+NO book snapshot from the live WebSocket."""
    yes_token_id: str
    no_token_id: str
    yes: TokenBook
    no: TokenBook
    ts: float = field(default_factory=time.time)

    @property
    def sum_ask(self) -> float:
        """YES ask + NO ask. < 1.0 → guaranteed arbitrage."""
        return self.yes.ask + self.no.ask

    @property
    def sum_bid(self) -> float:
        return self.yes.bid + self.no.bid

    @property
    def yes_microprice(self) -> float:
        return self.yes.microprice

    @property
    def no_microprice(self) -> float:
        return self.no.microprice

    @property
    def orderbook_imbalance(self) -> float:
        """
        YES-side pressure vs NO-side.
        +1.0 = all YES buyers, -1.0 = all NO buyers.
        """
        yes_p = self.yes.bid_size - self.yes.ask_size
        no_p  = self.no.bid_size  - self.no.ask_size
        total = abs(yes_p) + abs(no_p)
        if total == 0:
            return 0.0
        return (yes_p - no_p) / total

    @property
    def age(self) -> float:
        return time.time() - self.ts


class PolymarketWebSocket:
    """
    Subscribes to Polymarket CLOB real-time orderbook updates.

    Usage:
        pm_ws = PolymarketWebSocket(on_book_update=my_handler)
        await pm_ws.start()
        await pm_ws.subscribe(yes_token_id, no_token_id)
        # handler called on every price change with LiveBookSnapshot
        await pm_ws.stop()
    """

    def __init__(
        self,
        on_book_update: Optional[Callable[[LiveBookSnapshot], None]] = None,
    ):
        self._on_book_update = on_book_update
        self._token_books: Dict[str, TokenBook] = {}
        self._pairs: List[Tuple[str, str]] = []      # [(yes_id, no_id), ...]
        self._token_to_pair: Dict[str, Tuple[str, str]] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._ws = None
        self._subscribed_tokens: Set[str] = set()
        self._pending_tokens: Set[str] = set()

    async def subscribe(self, yes_token_id: str, no_token_id: str) -> None:
        """
        Subscribe to real-time updates for a market's YES and NO tokens.
        Can be called before or after start().
        """
        if (yes_token_id, no_token_id) in self._pairs:
            return  # already subscribed

        self._pairs.append((yes_token_id, no_token_id))
        self._token_to_pair[yes_token_id] = (yes_token_id, no_token_id)
        self._token_to_pair[no_token_id]  = (yes_token_id, no_token_id)

        for tid in (yes_token_id, no_token_id):
            if tid not in self._subscribed_tokens:
                self._pending_tokens.add(tid)
                self._token_books[tid] = TokenBook(token_id=tid)

        if self._ws is not None:
            await self._send_subscribe(list(self._pending_tokens))
            self._subscribed_tokens.update(self._pending_tokens)
            self._pending_tokens.clear()

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("[POLY-WS] Polymarket RTDS started")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("[POLY-WS] Polymarket RTDS stopped")

    async def _loop(self) -> None:
        delay = 1.0
        while self._running:
            try:
                async with websockets.connect(
                    RTDS_URL,
                    ping_interval=30,
                    ping_timeout=10,
                    extra_headers={"User-Agent": "PolyBTC-Trader/1.0"},
                ) as ws:
                    self._ws = ws
                    logger.info("[POLY-WS] Connected to Polymarket CLOB stream")
                    delay = 1.0

                    # Subscribe to all pending + previously subscribed tokens
                    all_tokens = list(
                        self._subscribed_tokens | self._pending_tokens
                    )
                    if all_tokens:
                        await self._send_subscribe(all_tokens)
                        self._subscribed_tokens.update(all_tokens)
                        self._pending_tokens.clear()

                    async for raw in ws:
                        if not self._running:
                            break
                        try:
                            msgs = json.loads(raw)
                            if isinstance(msgs, list):
                                for m in msgs:
                                    self._handle(m)
                            elif isinstance(msgs, dict):
                                self._handle(msgs)
                        except Exception as e:
                            logger.debug(f"[POLY-WS] Handle error: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._running:
                    logger.debug(f"[POLY-WS] Reconnect in {delay:.0f}s: {e}")
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 30.0)
            finally:
                self._ws = None

    async def _send_subscribe(self, token_ids: List[str]) -> None:
        if self._ws is None or not token_ids:
            return
        try:
            msg = json.dumps({
                "assets_ids": token_ids,
                "type": "market",
            })
            await self._ws.send(msg)
            logger.debug(f"[POLY-WS] Subscribed to {len(token_ids)} tokens")
        except Exception as e:
            logger.debug(f"[POLY-WS] Subscribe send error: {e}")

    def _handle(self, msg: dict) -> None:
        """Process a single RTDS event and update book state."""
        if not isinstance(msg, dict):
            return

        event_type = msg.get("event_type") or msg.get("type", "")
        asset_id = msg.get("asset_id") or msg.get("market", "")

        if not asset_id or asset_id not in self._token_to_pair:
            return

        book = self._token_books.get(asset_id)
        if book is None:
            return

        # price_change events carry side + price + size
        if event_type in ("price_change", "book", "tick", "last_trade_price"):
            price_raw = msg.get("price")
            side = str(msg.get("side", "")).upper()
            size_raw = msg.get("size", 0)

            if price_raw is not None:
                price = float(price_raw)
                size  = float(size_raw) if size_raw else 0.0
                book.ts = time.time()

                if side == "BUY":
                    book.bid = price
                    if size:
                        book.bid_size = size
                elif side == "SELL":
                    book.ask = price
                    if size:
                        book.ask_size = size
                else:
                    # Mid update — use as both bid/ask estimate
                    spread = book.spread
                    book.bid = price - spread / 2
                    book.ask = price + spread / 2

            # Try to emit combined snapshot
            yes_id, no_id = self._token_to_pair[asset_id]
            yes_book = self._token_books.get(yes_id)
            no_book  = self._token_books.get(no_id)

            if yes_book and no_book and yes_book.ask > 0 and no_book.ask > 0:
                snap = LiveBookSnapshot(
                    yes_token_id=yes_id,
                    no_token_id=no_id,
                    yes=yes_book,
                    no=no_book,
                )
                if self._on_book_update:
                    self._on_book_update(snap)
