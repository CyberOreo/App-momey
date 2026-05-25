"""
Direct 1-second feed for Polymarket BTC 5-minute Up/Down markets.

Slug pattern: btc-updown-5m-{close_timestamp}
e.g.  btc-updown-5m-1779696300  (timestamp is the window end in Unix seconds)

Every second:
  1. Compute current window slug from system clock
  2. If new window → fetch market structure (token IDs) from Gamma API
  3. Fetch live YES/NO bid/ask from CLOB orderbook API
  4. Call on_update(MarketSnap) with fresh data
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Callable, Optional

import aiohttp
from loguru import logger

GAMMA = "https://gamma-api.polymarket.com"
CLOB  = "https://clob.polymarket.com"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://polymarket.com/",
    "Origin": "https://polymarket.com",
}


@dataclass
class MarketSnap:
    slug: str
    question: str
    condition_id: str
    yes_token_id: str
    no_token_id: str
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float
    close_ts: int
    seconds_to_close: float

    @property
    def yes_mid(self) -> float:
        return (self.yes_bid + self.yes_ask) / 2 if self.yes_ask > 0 else 0.0

    @property
    def no_mid(self) -> float:
        return (self.no_bid + self.no_ask) / 2 if self.no_ask > 0 else 0.0

    @property
    def spread_pct(self) -> float:
        return (self.yes_ask - self.yes_bid) if self.yes_ask > 0 else 0.0

    @property
    def sum_ask(self) -> float:
        return round(self.yes_ask + self.no_ask, 4)


def _close_ts() -> int:
    """Next 5-minute boundary as Unix timestamp."""
    return (int(time.time()) // 300 + 1) * 300


def _slug() -> str:
    return f"btc-updown-5m-{_close_ts()}"


class BTC5MinFeed:
    """
    Polls Polymarket every second for the current BTC 5-min Up/Down market.
    Delivers real YES/NO bid/ask prices from the CLOB API.
    """

    def __init__(self, on_update: Optional[Callable[[MarketSnap], None]] = None):
        self._on_update = on_update
        self._session: Optional[aiohttp.ClientSession] = None
        self._task: Optional[asyncio.Task] = None
        self._running = False

        # Current window state
        self._current_slug = ""
        self._yes_token = ""
        self._no_token = ""
        self._condition_id = ""
        self._question = ""

        self.last_snap: Optional[MarketSnap] = None
        self._fetch_errors = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        conn = aiohttp.TCPConnector(ssl=False, limit=10)
        self._session = aiohttp.ClientSession(
            connector=conn,
            headers=_HEADERS,
            timeout=aiohttp.ClientTimeout(total=5),
        )
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("[5MIN] BTC5MinFeed started — polling Polymarket every 1s")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._session:
            await self._session.close()
        logger.info("[5MIN] BTC5MinFeed stopped")

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def _loop(self) -> None:
        while self._running:
            t0 = time.monotonic()
            try:
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._fetch_errors += 1
                logger.debug(f"[5MIN] tick error ({self._fetch_errors}): {exc}")
            # Sleep the remainder of 1 second
            elapsed = time.monotonic() - t0
            await asyncio.sleep(max(0.0, 1.0 - elapsed))

    async def _tick(self) -> None:
        slug = _slug()
        close_ts = _close_ts()
        secs = close_ts - time.time()

        # New 5-min window → load market structure from Gamma API
        if slug != self._current_slug:
            logger.info(f"[5MIN] New window: {slug} | T-{secs:.0f}s")
            await self._fetch_market(slug)

        if not self._yes_token or not self._no_token:
            logger.debug("[5MIN] No token IDs yet — waiting for market load")
            return

        # Fetch live bid/ask from CLOB (both tokens in parallel)
        yes_task = asyncio.create_task(self._book(self._yes_token))
        no_task  = asyncio.create_task(self._book(self._no_token))
        yes, no  = await asyncio.gather(yes_task, no_task)

        snap = MarketSnap(
            slug=slug,
            question=self._question,
            condition_id=self._condition_id,
            yes_token_id=self._yes_token,
            no_token_id=self._no_token,
            yes_bid=yes["bid"],
            yes_ask=yes["ask"],
            no_bid=no["bid"],
            no_ask=no["ask"],
            close_ts=close_ts,
            seconds_to_close=max(0.0, secs),
        )
        self.last_snap = snap
        self._fetch_errors = 0

        logger.debug(
            f"[5MIN] T-{secs:.0f}s | YES={yes['bid']:.3f}/{yes['ask']:.3f} | "
            f"NO={no['bid']:.3f}/{no['ask']:.3f} | sum={snap.sum_ask:.3f}"
        )

        if self._on_update:
            self._on_update(snap)

    # ── Gamma API — market structure ──────────────────────────────────────────

    async def _fetch_market(self, slug: str) -> None:
        try:
            async with self._session.get(  # type: ignore[union-attr]
                f"{GAMMA}/events", params={"slug": slug}
            ) as r:
                data = await r.json()

            events = data if isinstance(data, list) else [data]
            if not events or not events[0]:
                logger.warning(f"[5MIN] No event found for slug {slug}")
                return

            event = events[0]
            markets = event.get("markets", [])
            if not markets:
                logger.warning(f"[5MIN] Event has no markets: {slug}")
                return

            mkt = markets[0]
            self._condition_id = str(
                mkt.get("conditionId") or mkt.get("condition_id") or ""
            )
            self._question = str(
                event.get("title") or mkt.get("question") or slug
            )

            # Parse YES/NO token IDs — Polymarket returns them in two formats:
            # Format A: clobTokenIds = ["0xYES...", "0xNO..."]  (array of strings)
            # Format B: tokens = [{"token_id": "...", "outcome": "YES"}, ...]
            yes_id = no_id = ""

            clob_ids = mkt.get("clobTokenIds")
            if isinstance(clob_ids, list) and len(clob_ids) >= 2:
                yes_id = str(clob_ids[0])
                no_id  = str(clob_ids[1])
            else:
                for t in mkt.get("tokens", []):
                    out = str(t.get("outcome", "")).upper()
                    tid = str(t.get("token_id") or t.get("tokenId") or "")
                    if "YES" in out:
                        yes_id = tid
                    elif "NO" in out:
                        no_id = tid

            if yes_id and no_id:
                self._yes_token = yes_id
                self._no_token  = no_id
                self._current_slug = slug
                logger.success(
                    f"[5MIN] Market loaded: {self._question} | "
                    f"YES={yes_id[:12]}... NO={no_id[:12]}..."
                )
            else:
                logger.warning(f"[5MIN] Could not parse token IDs from market data")

        except Exception as exc:
            logger.debug(f"[5MIN] _fetch_market failed: {exc}")

    # ── CLOB API — live orderbook ─────────────────────────────────────────────

    async def _book(self, token_id: str) -> dict:
        """Return best bid/ask for a CLOB token. Falls back to midpoint."""
        # Primary: full orderbook (best prices)
        try:
            async with self._session.get(  # type: ignore[union-attr]
                f"{CLOB}/book", params={"token_id": token_id}
            ) as r:
                data = await r.json()
            bids = data.get("bids", [])
            asks = data.get("asks", [])
            bid = float(bids[0]["price"]) if bids else 0.0
            ask = float(asks[0]["price"]) if asks else 0.0
            return {"bid": bid, "ask": ask}
        except Exception:
            pass

        # Fallback: midpoint endpoint
        try:
            async with self._session.get(  # type: ignore[union-attr]
                f"{CLOB}/midpoint", params={"token_id": token_id}
            ) as r:
                data = await r.json()
            mid = float(data.get("mid", 0.5))
            return {"bid": round(mid - 0.005, 4), "ask": round(mid + 0.005, 4)}
        except Exception:
            return {"bid": 0.0, "ask": 0.0}
