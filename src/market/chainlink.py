"""
Chainlink BTC/USD oracle reader for Polygon.

Reads latestRoundData() directly from the Chainlink aggregator contract
via raw JSON-RPC — no web3.py needed, just aiohttp.

KEY EDGE: Polymarket 5-min markets resolve on the Chainlink oracle price,
NOT live Binance. The oracle lags ~27s. Reading it directly tells you
the EXACT settlement price before Polymarket resolves the market.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import aiohttp
from loguru import logger


# Chainlink BTC/USD aggregator on Polygon (verified)
_AGGREGATOR = "0xc907E116054Ad103354f2D350FD2514433D57F6f"
# keccak256("latestRoundData()")[:4] = 0xfeaf968c
_SELECTOR = "0xfeaf968c"
_RPC_URLS: List[str] = [
    "https://polygon-rpc.com",
    "https://rpc.ankr.com/polygon",
    "https://polygon.llamarpc.com",
    "https://rpc-mainnet.matic.quiknode.pro",
]


@dataclass
class OraclePrice:
    price: float        # BTC/USD
    updated_at: int     # unix timestamp of last oracle update
    round_id: int       # Chainlink round number

    @property
    def age_seconds(self) -> float:
        return time.time() - self.updated_at

    @property
    def is_fresh(self) -> bool:
        """Oracle updated within the last 60 seconds."""
        return self.age_seconds < 60

    @property
    def seconds_since_update(self) -> float:
        return time.time() - self.updated_at


class ChainlinkOracle:
    """
    Polls the Chainlink BTC/USD aggregator on Polygon every poll_interval seconds.

    The oracle heartbeat is ~27s on Polygon, so polling at 5s catches
    every update. Rotates through multiple free RPC endpoints on failure.

    Usage:
        oracle = ChainlinkOracle(on_update=my_callback)
        await oracle.start()
        # ... oracle.latest gives current price
        await oracle.stop()
    """

    def __init__(
        self,
        poll_interval: float = 5.0,
        on_update: Optional[Callable[[OraclePrice], None]] = None,
    ):
        self._interval = poll_interval
        self._on_update = on_update
        self._rpc_index = 0
        self._latest: Optional[OraclePrice] = None
        self._prev_round_id: int = -1
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def latest(self) -> Optional[OraclePrice]:
        return self._latest

    def confirms_direction(self, window_open_price: float) -> Optional[str]:
        """
        Compare oracle price to window open price.
        Returns 'YES' if oracle is above open, 'NO' if below, None if no data.
        This is what Polymarket will actually resolve to.
        """
        if self._latest is None or not self._latest.is_fresh:
            return None
        if window_open_price <= 0:
            return None
        return "YES" if self._latest.price > window_open_price else "NO"

    async def start(self) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=8)
        )
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("[ORACLE] Chainlink reader started")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._session and not self._session.closed:
            await self._session.close()
        logger.info("[ORACLE] Chainlink reader stopped")

    async def _loop(self) -> None:
        while self._running:
            try:
                price = await self._fetch()
                if price is not None:
                    if price.round_id != self._prev_round_id:
                        logger.debug(
                            f"[ORACLE] New round: BTC/USD={price.price:,.2f} | "
                            f"age={price.age_seconds:.0f}s | round={price.round_id}"
                        )
                        self._prev_round_id = price.round_id
                    self._latest = price
                    if self._on_update:
                        self._on_update(price)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"[ORACLE] Poll error: {e}")
            await asyncio.sleep(self._interval)

    async def _fetch(self) -> Optional[OraclePrice]:
        """Call latestRoundData() via eth_call JSON-RPC."""
        payload = {
            "jsonrpc": "2.0",
            "method": "eth_call",
            "params": [
                {"to": _AGGREGATOR, "data": _SELECTOR},
                "latest",
            ],
            "id": 1,
        }
        for _ in range(len(_RPC_URLS)):
            rpc = _RPC_URLS[self._rpc_index % len(_RPC_URLS)]
            try:
                async with self._session.post(rpc, json=payload) as resp:
                    data = await resp.json(content_type=None)
                    result = data.get("result", "")
                    if result and result not in ("0x", "", None):
                        decoded = self._decode(result)
                        if decoded:
                            return decoded
            except Exception as e:
                logger.debug(f"[ORACLE] RPC {rpc} failed: {e}")
                self._rpc_index += 1
        return None

    def _decode(self, hex_data: str) -> Optional[OraclePrice]:
        """
        Decode ABI-encoded latestRoundData() return.
        5 × uint256 each padded to 32 bytes (64 hex chars each):
          [0]  roundId     uint80
          [1]  answer      int256   (price × 1e8)
          [2]  startedAt   uint256
          [3]  updatedAt   uint256
          [4]  answeredInRound uint80
        """
        try:
            raw = hex_data[2:] if hex_data.startswith("0x") else hex_data
            if len(raw) < 320:
                return None
            round_id   = int(raw[0:64], 16)
            answer     = int(raw[64:128], 16)
            # handle negative int256 (two's complement)
            if answer >= (1 << 255):
                answer -= (1 << 256)
            updated_at = int(raw[192:256], 16)

            price_usd = answer / 1e8
            if not (1_000 < price_usd < 10_000_000):
                return None  # sanity check

            return OraclePrice(
                price=price_usd,
                updated_at=updated_at,
                round_id=round_id,
            )
        except Exception as e:
            logger.debug(f"[ORACLE] Decode error: {e}")
            return None
