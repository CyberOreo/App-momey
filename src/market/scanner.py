"""BTC market scanner — discovers, enriches, and filters Polymarket prediction markets."""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Callable, Coroutine, Dict, List, Optional

from loguru import logger

from src.core.config import Settings
from src.core.database import MarketRepository
from src.core.models import Market, OrderBook


# ── BTC-related keyword filter ────────────────────────────────────────────────

_BTC_KEYWORDS = (
    "btc",
    "bitcoin",
    "xbt",
    "satoshi",
)

_FIVE_MIN_KEYWORDS = (
    "5 minute",
    "5-minute",
    "5min",
    "next 5",
    "5 min",
    "five minute",
    "in 5",
)


def _is_btc_market(question: str) -> bool:
    """Return True if the market question refers to BTC/Bitcoin."""
    lower = question.lower()
    return any(kw in lower for kw in _BTC_KEYWORDS)


# ── MarketScanner ─────────────────────────────────────────────────────────────

class MarketScanner:
    """
    Discovers and filters active Polymarket markets related to Bitcoin.

    Pipeline:
      1. ``scan()``              — fetch all BTC markets from Polymarket.
      2. ``get_tradeable_markets()`` — apply time-window + keyword + token filters.
      3. ``enrich_with_orderbook()`` — attach live order-book data.
      4. ``filter_by_liquidity()`` / ``filter_by_spread()`` — quality filters.
      5. ``run_continuous()``    — run the above on a schedule.
    """

    def __init__(
        self,
        polymarket_client: Any,          # PolymarketClient (avoid circular import)
        settings: Settings,
        db: MarketRepository,
    ) -> None:
        self._client = polymarket_client
        self._settings = settings
        self._db = db
        self._stop_event: asyncio.Event = asyncio.Event()

    # ── Step 1: full scan ─────────────────────────────────────────────────────

    async def scan(self) -> List[Market]:
        """
        Fetch all active BTC-related markets from Polymarket and persist them
        in the database.

        Returns the raw list of Market objects (not yet filtered by trade-ability).
        """
        try:
            markets = await self._client.get_btc_markets()
            logger.info("Market scan complete", total=len(markets))
        except Exception as exc:
            logger.error("Market scan failed", error=str(exc))
            return []

        # Persist / update each market in the database
        for market in markets:
            try:
                await self._db.upsert_market(market)
            except Exception as exc:
                logger.warning(
                    "Failed to persist market",
                    condition_id=market.condition_id,
                    error=str(exc),
                )

        return markets

    # ── Step 2: tradeable filter ──────────────────────────────────────────────

    async def get_tradeable_markets(self) -> List[Market]:
        """
        Filter scanned markets down to those eligible for trading:

        - Active and not already expired.
        - Time to resolution within [min_time, max_time] hours.
        - Question contains BTC/bitcoin keywords.
        - Market has both a YES and a NO token.
        """
        markets = await self.scan()
        tradeable: List[Market] = []
        now = datetime.utcnow()

        min_hours: float = self._settings.min_time_to_resolution_hours
        max_hours: float = self._settings.max_time_to_resolution_hours

        for market in markets:
            if not market.active:
                logger.debug("Skipping inactive market", condition_id=market.condition_id)
                continue

            if market.end_date <= now:
                logger.debug(
                    "Skipping expired market",
                    condition_id=market.condition_id,
                    end_date=market.end_date.isoformat(),
                )
                continue

            hours = market.hours_to_resolution
            if not (min_hours <= hours <= max_hours):
                logger.debug(
                    "Skipping market outside time window",
                    condition_id=market.condition_id,
                    hours_to_resolution=round(hours, 2),
                    min=min_hours,
                    max=max_hours,
                )
                continue

            if not _is_btc_market(market.question):
                logger.debug(
                    "Skipping non-BTC market",
                    condition_id=market.condition_id,
                    question=market.question[:80],
                )
                continue

            if market.yes_token is None or market.no_token is None:
                logger.debug(
                    "Skipping market missing YES/NO tokens",
                    condition_id=market.condition_id,
                )
                continue

            tradeable.append(market)

        logger.info(
            "Tradeable markets identified",
            total_scanned=len(markets),
            tradeable=len(tradeable),
        )
        return tradeable

    # ── Step 3: order-book enrichment ─────────────────────────────────────────

    async def enrich_with_orderbook(self, markets: List[Market]) -> List[Market]:
        """
        Fetch order books for both YES and NO tokens of each market.
        Markets whose order books cannot be fetched are returned as-is with a
        warning — they will be filtered out in subsequent liquidity/spread checks.

        Returns the same list of markets (order-book data is collected as a side
        effect and returned via the ``_order_books`` dict for use by callers).
        """
        self._order_books: Dict[str, OrderBook] = {}

        async def _fetch_one(token_id: str) -> None:
            try:
                ob = await self._client.get_order_book(token_id)
                self._order_books[token_id] = ob
            except Exception as exc:
                logger.warning(
                    "Failed to fetch order book",
                    token_id=token_id,
                    error=str(exc),
                )

        # Collect all token IDs that need order books
        token_ids: List[str] = []
        for market in markets:
            if market.yes_token:
                token_ids.append(market.yes_token.token_id)
            if market.no_token:
                token_ids.append(market.no_token.token_id)

        # Fetch concurrently with a semaphore to avoid hammering the API
        semaphore = asyncio.Semaphore(10)

        async def _fetch_limited(token_id: str) -> None:
            async with semaphore:
                await _fetch_one(token_id)

        await asyncio.gather(*[_fetch_limited(tid) for tid in token_ids])

        fetched = len(self._order_books)
        logger.info(
            "Order books fetched",
            requested=len(token_ids),
            fetched=fetched,
        )
        return markets

    # ── Step 4a: liquidity filter ─────────────────────────────────────────────

    def filter_by_liquidity(
        self,
        markets: List[Market],
        order_books: Dict[str, "OrderBook"],
    ) -> List[Market]:
        """
        Remove markets whose combined YES+NO order-book liquidity is below
        ``settings.min_liquidity_usdc``.
        """
        min_liq: float = self._settings.min_liquidity_usdc
        filtered: List[Market] = []

        for market in markets:
            yes_id = market.yes_token.token_id if market.yes_token else None
            no_id = market.no_token.token_id if market.no_token else None

            yes_liq: float = order_books[yes_id].total_liquidity if yes_id and yes_id in order_books else 0.0
            no_liq: float = order_books[no_id].total_liquidity if no_id and no_id in order_books else 0.0
            total_liq = yes_liq + no_liq

            if total_liq < min_liq:
                logger.debug(
                    "Market filtered — insufficient liquidity",
                    condition_id=market.condition_id,
                    liquidity=round(total_liq, 2),
                    min_required=min_liq,
                )
                continue

            filtered.append(market)

        logger.info(
            "Liquidity filter applied",
            before=len(markets),
            after=len(filtered),
            min_usdc=min_liq,
        )
        return filtered

    # ── Step 4b: spread filter ────────────────────────────────────────────────

    def filter_by_spread(
        self,
        markets: List[Market],
        order_books: Dict[str, "OrderBook"],
    ) -> List[Market]:
        """
        Remove markets where the YES or NO token has a spread wider than
        ``settings.max_spread_pct``.
        """
        max_spread: float = self._settings.max_spread_pct
        filtered: List[Market] = []

        for market in markets:
            yes_id = market.yes_token.token_id if market.yes_token else None
            no_id = market.no_token.token_id if market.no_token else None

            spreads: List[float] = []
            for token_id in (yes_id, no_id):
                if token_id and token_id in order_books:
                    sp = order_books[token_id].spread_pct
                    if sp is not None:
                        spreads.append(sp)

            if not spreads:
                logger.debug(
                    "Market filtered — no spread data available",
                    condition_id=market.condition_id,
                )
                continue

            worst_spread = max(spreads)
            if worst_spread > max_spread:
                logger.debug(
                    "Market filtered — spread too wide",
                    condition_id=market.condition_id,
                    spread_pct=round(worst_spread, 4),
                    max_allowed=max_spread,
                )
                continue

            filtered.append(market)

        logger.info(
            "Spread filter applied",
            before=len(markets),
            after=len(filtered),
            max_spread_pct=max_spread,
        )
        return filtered

    # ── Continuous runner ─────────────────────────────────────────────────────

    async def run_continuous(
        self,
        interval_seconds: float = 300.0,
        callback: Optional[Callable[[List[Market]], Coroutine[Any, Any, None]]] = None,
    ) -> None:
        """
        Run the full scan + filter + enrich pipeline every *interval_seconds*.

        Calls *callback* with the final list of trade-eligible markets after
        each cycle.  Runs until ``stop()`` is called.
        """
        self._stop_event.clear()
        logger.info(
            "MarketScanner continuous loop started",
            interval_seconds=interval_seconds,
        )

        while not self._stop_event.is_set():
            cycle_start = asyncio.get_event_loop().time()
            try:
                tradeable = await self.get_tradeable_markets()
                if tradeable:
                    enriched = await self.enrich_with_orderbook(tradeable)
                    order_books = getattr(self, "_order_books", {})
                    quality = self.filter_by_liquidity(enriched, order_books)
                    quality = self.filter_by_spread(quality, order_books)
                    logger.info(
                        "Scan cycle complete",
                        tradeable=len(tradeable),
                        after_quality_filters=len(quality),
                    )
                    if callback is not None:
                        try:
                            await callback(quality)
                        except Exception as exc:
                            logger.error("Scanner callback raised", error=str(exc))
                else:
                    logger.info("Scan cycle complete — no tradeable markets found")

            except asyncio.CancelledError:
                logger.info("MarketScanner continuous loop cancelled")
                break
            except Exception as exc:
                logger.error("Error in scanner continuous loop", error=str(exc))

            # Sleep for the remainder of the interval
            elapsed = asyncio.get_event_loop().time() - cycle_start
            sleep_time = max(0.0, interval_seconds - elapsed)
            if sleep_time > 0:
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=sleep_time)
                except asyncio.TimeoutError:
                    pass  # interval elapsed normally

        logger.info("MarketScanner continuous loop stopped")

    async def stop(self) -> None:
        """Signal the continuous loop to exit after the current cycle."""
        self._stop_event.set()
        logger.info("MarketScanner stop requested")

    # ── 5-minute market scanning ──────────────────────────────────────────────

    async def scan_five_minute_markets(self) -> List[Market]:
        """
        Find all active 5-minute BTC up/down markets on Polymarket.

        Returns markets sorted by liquidity (highest first) that are:
        - Active and contain BTC keywords
        - Identified as 5-minute resolution markets by question text
        - Between 90 seconds and 360 seconds from resolution (sweet spot)
        - Have both YES and NO tokens
        """
        try:
            all_markets = await self._client.get_btc_markets()
        except Exception as exc:
            logger.error("5-min scan failed", error=str(exc))
            return []

        result: List[Market] = []
        now = datetime.utcnow()

        for market in all_markets:
            if not market.active:
                continue
            if market.end_date <= now:
                continue
            if not _is_btc_market(market.question):
                continue
            if market.yes_token is None or market.no_token is None:
                continue

            question_lower = market.question.lower()
            is_five_min = any(kw in question_lower for kw in _FIVE_MIN_KEYWORDS)
            if not is_five_min:
                continue

            secs = market.hours_to_resolution * 3600
            if not (90 <= secs <= 360):
                continue

            result.append(market)

        # Sort by liquidity so we trade the deepest markets first
        result.sort(key=lambda m: m.liquidity, reverse=True)

        logger.info(
            "[5MIN] Scan complete",
            found=len(result),
            questions=[m.question[:60] for m in result[:3]],
        )
        return result

    async def run_five_min_continuous(
        self,
        interval_seconds: float = 30.0,
        callback: Optional[Callable[[List[Market]], Coroutine[Any, Any, None]]] = None,
    ) -> None:
        """
        Scan for 5-minute markets every 30 seconds (not 5 minutes like the
        standard scanner — 5-min markets open and close constantly).
        """
        self._stop_event.clear()
        logger.info("[5MIN] Scanner started", interval_seconds=interval_seconds)

        while not self._stop_event.is_set():
            cycle_start = asyncio.get_event_loop().time()
            try:
                markets = await self.scan_five_minute_markets()
                if markets and callback is not None:
                    try:
                        await callback(markets)
                    except Exception as exc:
                        logger.error("[5MIN] Callback error", error=str(exc))
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("[5MIN] Scanner loop error", error=str(exc))

            elapsed = asyncio.get_event_loop().time() - cycle_start
            sleep_time = max(0.0, interval_seconds - elapsed)
            if sleep_time > 0:
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=sleep_time)
                except asyncio.TimeoutError:
                    pass

        logger.info("[5MIN] Scanner stopped")
