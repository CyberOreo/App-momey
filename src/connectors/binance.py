"""Binance public price feed — WebSocket kline streams + REST historical candles."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, List, Optional

import aiohttp
import websockets
import websockets.exceptions
from loguru import logger

from src.core.config import Settings
from src.core.models import BTCPrice, Candle


# ── Constants ─────────────────────────────────────────────────────────────────

_TIMEFRAMES = ("1m", "5m", "15m", "1h", "4h")
_SYMBOL = "btcusdt"
_KLINE_STREAMS = "/".join(f"{_SYMBOL}@kline_{tf}" for tf in _TIMEFRAMES)
_TICKER_STREAM = f"{_SYMBOL}@ticker"


def _build_ws_url(settings: Settings) -> str:
    """Build the combined WebSocket stream URL."""
    # Use the configurable base; strip trailing /ws to normalise
    base = settings.binance_ws_url.rstrip("/").removesuffix("/ws")
    streams = f"{_KLINE_STREAMS}/{_TICKER_STREAM}"
    return f"{base}/stream?streams={streams}"


# ── Kline / candle parsers ────────────────────────────────────────────────────

def _ms_to_dt(ms: int) -> datetime:
    """Convert a millisecond epoch to a UTC-naive datetime."""
    return datetime.utcfromtimestamp(ms / 1000)


def _parse_kline_event(event: Dict[str, Any]) -> Optional[Candle]:
    """
    Parse a Binance kline stream event dict.
    The 'k' sub-object holds: t=open_time, o/h/l/c/v=OHLCV, x=is_closed.
    """
    try:
        k = event.get("k", {})
        timeframe = str(k.get("i", "1m"))
        return Candle(
            timestamp=_ms_to_dt(int(k["t"])),
            open=float(k["o"]),
            high=float(k["h"]),
            low=float(k["l"]),
            close=float(k["c"]),
            volume=float(k["v"]),
            timeframe=timeframe,
        )
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning("Failed to parse kline event", error=str(exc))
        return None


def _parse_rest_kline(raw: List[Any], timeframe: str) -> Optional[Candle]:
    """
    Parse a single REST kline entry.
    Binance REST format: [open_time, open, high, low, close, volume, ...]
    """
    try:
        return Candle(
            timestamp=_ms_to_dt(int(raw[0])),
            open=float(raw[1]),
            high=float(raw[2]),
            low=float(raw[3]),
            close=float(raw[4]),
            volume=float(raw[5]),
            timeframe=timeframe,
        )
    except (IndexError, ValueError, TypeError) as exc:
        logger.warning("Failed to parse REST kline", error=str(exc))
        return None


# ── BinanceFeed ───────────────────────────────────────────────────────────────

class BinanceFeed:
    """
    Binance public market-data feed.

    Subscribes to combined WebSocket streams for BTCUSDT klines (1m, 5m, 15m,
    1h, 4h) and the rolling ticker.  Provides:
      - ``current_price`` property updated on every ticker event.
      - ``on_price_callback`` called on every ticker update.
      - ``on_candle_callback`` called when a kline bar closes (x=True).
      - ``get_historical_candles()`` / ``get_all_historical_candles()`` via REST.
    """

    def __init__(
        self,
        settings: Settings,
        on_price_callback: Optional[Callable[[BTCPrice], Coroutine[Any, Any, None]]] = None,
        on_candle_callback: Optional[Callable[[Candle], Coroutine[Any, Any, None]]] = None,
    ) -> None:
        self._settings = settings
        self._ws_url = _build_ws_url(settings)
        self._rest_url = settings.binance_rest_url.rstrip("/")
        self._on_price = on_price_callback
        self._on_candle = on_candle_callback

        self._current_price: float = 0.0
        self._connected: bool = False
        self._stop_event: asyncio.Event = asyncio.Event()
        self._ws_task: Optional[asyncio.Task[None]] = None

        connector = aiohttp.TCPConnector(limit=20, ssl=False)
        timeout = aiohttp.ClientTimeout(total=30)
        self._http: aiohttp.ClientSession = aiohttp.ClientSession(
            connector=connector, timeout=timeout
        )

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def current_price(self) -> float:
        """Latest BTC/USDT price received from the ticker stream."""
        return self._current_price

    @property
    def connected(self) -> bool:
        return self._connected

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """
        Begin the WebSocket feed.  Spawns a background task that manages the
        connection and auto-reconnects on failure.
        """
        if self._ws_task and not self._ws_task.done():
            logger.warning("BinanceFeed already running")
            return
        self._stop_event.clear()
        self._ws_task = asyncio.create_task(self._connection_loop(), name="binance-ws")
        logger.info("BinanceFeed started", url=self._ws_url)

    async def stop(self) -> None:
        """Gracefully stop the WebSocket feed and close the HTTP session."""
        self._stop_event.set()
        if self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
        self._connected = False
        if not self._http.closed:
            await self._http.close()
        logger.info("BinanceFeed stopped")

    # ── WebSocket connection loop ─────────────────────────────────────────────

    async def _connection_loop(self) -> None:
        """
        Reconnect loop with exponential backoff + jitter.
        Retries up to settings.max_reconnect_attempts times.
        """
        max_attempts: int = self._settings.max_reconnect_attempts
        base_delay: float = self._settings.reconnect_base_delay
        max_delay: float = self._settings.reconnect_max_delay
        consecutive_failures = 0

        import random

        while not self._stop_event.is_set():
            try:
                logger.info(
                    "Connecting to Binance WebSocket",
                    attempt=consecutive_failures + 1,
                    url=self._ws_url,
                )
                async with websockets.connect(
                    self._ws_url,
                    ping_interval=20,
                    ping_timeout=30,
                    close_timeout=10,
                ) as ws:
                    self._connected = True
                    consecutive_failures = 0
                    logger.info("Binance WebSocket connected")
                    await self._stream_loop(ws)

            except asyncio.CancelledError:
                logger.info("BinanceFeed connection loop cancelled")
                break
            except (
                websockets.exceptions.ConnectionClosedError,
                websockets.exceptions.ConnectionClosedOK,
                websockets.exceptions.WebSocketException,
                OSError,
            ) as exc:
                self._connected = False
                consecutive_failures += 1
                logger.warning(
                    "Binance WebSocket disconnected",
                    error=str(exc),
                    consecutive_failures=consecutive_failures,
                )
            except Exception as exc:
                self._connected = False
                consecutive_failures += 1
                logger.error(
                    "Unexpected WebSocket error",
                    error=str(exc),
                    consecutive_failures=consecutive_failures,
                )

            if self._stop_event.is_set():
                break

            if consecutive_failures >= max_attempts:
                logger.error(
                    "Max reconnect attempts reached — BinanceFeed giving up",
                    max=max_attempts,
                )
                break

            cap = min(base_delay * (2 ** (consecutive_failures - 1)), max_delay)
            delay = random.uniform(cap / 2, cap)
            logger.info("Reconnecting Binance WebSocket", delay=round(delay, 2))
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass  # delay elapsed normally

        self._connected = False
        logger.info("BinanceFeed connection loop exited")

    async def _stream_loop(self, ws: Any) -> None:
        """Read messages from the WebSocket and dispatch to handlers."""
        async for raw_msg in ws:
            if self._stop_event.is_set():
                break
            try:
                envelope: Dict[str, Any] = json.loads(raw_msg)
                data: Dict[str, Any] = envelope.get("data", envelope)
                event_type: str = data.get("e", "")

                if event_type == "kline":
                    await self._handle_kline(data)
                elif event_type in ("24hrTicker", "24hrMiniTicker"):
                    await self._handle_ticker(data)
                else:
                    logger.debug("Unhandled WS event type", event_type=event_type)
            except json.JSONDecodeError as exc:
                logger.warning("Failed to decode WS message", error=str(exc))
            except Exception as exc:
                logger.error("Error processing WS message", error=str(exc))

    async def _handle_kline(self, event: Dict[str, Any]) -> None:
        """Handle a kline event; fire on_candle_callback only when bar is closed."""
        candle = _parse_kline_event(event)
        if candle is None:
            return

        k = event.get("k", {})
        is_closed: bool = bool(k.get("x", False))

        # Always update current price from the close
        self._current_price = candle.close

        if is_closed and self._on_candle is not None:
            try:
                await self._on_candle(candle)
            except Exception as exc:
                logger.error("on_candle_callback raised", error=str(exc))

    async def _handle_ticker(self, event: Dict[str, Any]) -> None:
        """Handle a ticker event and invoke on_price_callback."""
        try:
            price = float(event.get("c", event.get("p", 0.0)))
            if price <= 0:
                return
            self._current_price = price
            ts = _ms_to_dt(int(event.get("E", 0))) if event.get("E") else datetime.utcnow()
            btc_price = BTCPrice(price=price, timestamp=ts, source="binance")

            if self._on_price is not None:
                try:
                    await self._on_price(btc_price)
                except Exception as exc:
                    logger.error("on_price_callback raised", error=str(exc))
        except (ValueError, TypeError) as exc:
            logger.warning("Failed to parse ticker event", error=str(exc))

    # ── REST historical candles ───────────────────────────────────────────────

    async def get_historical_candles(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
    ) -> List[Candle]:
        """
        Fetch up to *limit* historical klines via the Binance REST API.

        Args:
            symbol:   E.g. 'BTCUSDT'.
            interval: E.g. '1m', '5m', '1h'.
            limit:    Number of bars (max 1000).

        Returns:
            List of Candle objects, oldest first.
        """
        url = f"{self._rest_url}/klines"
        params: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": min(limit, 1000),
        }
        try:
            async with self._http.get(url, params=params) as resp:
                resp.raise_for_status()
                raw: List[List[Any]] = await resp.json(content_type=None)

            candles: List[Candle] = []
            for entry in raw:
                candle = _parse_rest_kline(entry, interval)
                if candle is not None:
                    candles.append(candle)

            logger.debug(
                "Historical candles fetched",
                symbol=symbol,
                interval=interval,
                count=len(candles),
            )
            return candles

        except aiohttp.ClientResponseError as exc:
            logger.error(
                "REST klines request failed",
                symbol=symbol,
                interval=interval,
                status=exc.status,
            )
            raise
        except Exception as exc:
            logger.error(
                "Unexpected error fetching klines",
                symbol=symbol,
                interval=interval,
                error=str(exc),
            )
            raise

    async def get_all_historical_candles(
        self,
        symbol: str = "BTCUSDT",
        timeframes: Optional[List[str]] = None,
        limit: int = 500,
    ) -> Dict[str, List[Candle]]:
        """
        Fetch historical candles for multiple timeframes concurrently.

        Args:
            symbol:     Trading symbol (default 'BTCUSDT').
            timeframes: List of interval strings; defaults to all standard timeframes.
            limit:      Candles per timeframe.

        Returns:
            Dict mapping timeframe string → list of Candle objects.
        """
        if timeframes is None:
            timeframes = list(_TIMEFRAMES)

        tasks = {
            tf: asyncio.create_task(
                self.get_historical_candles(symbol, tf, limit),
                name=f"klines-{symbol}-{tf}",
            )
            for tf in timeframes
        }

        results: Dict[str, List[Candle]] = {}
        for tf, task in tasks.items():
            try:
                results[tf] = await task
            except Exception as exc:
                logger.error(
                    "Failed to fetch candles for timeframe",
                    symbol=symbol,
                    timeframe=tf,
                    error=str(exc),
                )
                results[tf] = []

        logger.info(
            "All historical candles fetched",
            symbol=symbol,
            timeframes=list(results.keys()),
            counts={tf: len(c) for tf, c in results.items()},
        )
        return results
