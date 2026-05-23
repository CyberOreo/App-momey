"""Polymarket CLOB API client — L1 wallet auth + L2 API-key auth, full async."""
from __future__ import annotations

import hashlib
import hmac
import json
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp
from loguru import logger

from src.core.config import Settings
from src.core.models import Market, Order, OrderBook, OrderBookLevel, OrderStatus, PolymarketToken


# ── Auth helpers ──────────────────────────────────────────────────────────────

def _now_ms() -> int:
    """Unix timestamp in milliseconds."""
    return int(time.time() * 1000)


def _l1_sign(private_key: str, timestamp: int, nonce: int, method: str, path: str) -> str:
    """
    Level-1 (wallet) signature.
    Signs the string ``{timestamp}{nonce}{method}{path}`` with the Ethereum private key.
    """
    try:
        from eth_account import Account
        from eth_account.messages import encode_defunct
    except ImportError:
        raise RuntimeError(
            "eth-account is required for live trading. "
            "Run: pip install eth-account>=0.10.0"
        )
    message = f"{timestamp}{nonce}{method}{path}"
    msg_hash = encode_defunct(text=message)
    signed = Account.sign_message(msg_hash, private_key=private_key)
    return signed.signature.hex()


def _l2_sign(api_secret: str, timestamp: int, method: str, path: str, body: str = "") -> str:
    """
    Level-2 (API key) HMAC-SHA256 signature.
    Signs the string ``{timestamp}{method}{path}{body}``.
    """
    message = f"{timestamp}{method}{path}{body}"
    sig = hmac.new(api_secret.encode(), message.encode(), hashlib.sha256).hexdigest()
    return sig


def _l1_headers(private_key: str, method: str, path: str) -> Dict[str, str]:
    try:
        from eth_account import Account
    except ImportError:
        raise RuntimeError(
            "eth-account is required for live trading. "
            "Run: pip install eth-account>=0.10.0"
        )
    ts = _now_ms()
    nonce = 0
    sig = _l1_sign(private_key, ts, nonce, method.upper(), path)
    address = Account.from_key(private_key).address
    return {
        "POLY_ADDRESS": address,
        "POLY_SIGNATURE": sig,
        "POLY_TIMESTAMP": str(ts),
        "POLY_NONCE": str(nonce),
    }


def _l2_headers(
    api_key: str,
    api_secret: str,
    passphrase: str,
    method: str,
    path: str,
    body: str = "",
) -> Dict[str, str]:
    ts = _now_ms()
    sig = _l2_sign(api_secret, ts, method.upper(), path, body)
    return {
        "POLY-API-KEY": api_key,
        "POLY-SIGNATURE": sig,
        "POLY-TIMESTAMP": str(ts),
        "POLY-PASSPHRASE": passphrase,
    }


# ── Response parsers ──────────────────────────────────────────────────────────

def _parse_token(raw: Dict[str, Any]) -> PolymarketToken:
    return PolymarketToken(
        token_id=str(raw.get("token_id", raw.get("tokenId", ""))),
        outcome=str(raw.get("outcome", "")),
        price=float(raw.get("price", 0.5)),
        winner=raw.get("winner"),
    )


def _parse_market(raw: Dict[str, Any]) -> Optional[Market]:
    """Convert a raw Gamma/CLOB market dict into a Market dataclass."""
    try:
        condition_id: str = str(raw.get("condition_id", raw.get("conditionId", "")))
        if not condition_id:
            return None

        question: str = str(raw.get("question", ""))
        active: bool = bool(raw.get("active", False))

        # end_date / end_time varies across endpoints
        raw_end = raw.get("end_date_iso") or raw.get("endDateIso") or raw.get("end_date") or ""
        if raw_end:
            try:
                # strip trailing Z / timezone offset
                raw_end_clean = raw_end.replace("Z", "+00:00")
                end_date = datetime.fromisoformat(raw_end_clean).replace(tzinfo=None)
            except (ValueError, AttributeError):
                end_date = datetime.utcnow()
        else:
            end_date = datetime.utcnow()

        raw_tokens: List[Dict[str, Any]] = raw.get("tokens", [])
        tokens: List[PolymarketToken] = [_parse_token(t) for t in raw_tokens]

        return Market(
            condition_id=condition_id,
            question=question,
            tokens=tokens,
            end_date=end_date,
            active=active,
            volume=float(raw.get("volume", 0.0) or 0.0),
            liquidity=float(raw.get("liquidity", 0.0) or 0.0),
            description=str(raw.get("description", "")),
            image=str(raw.get("image", "")),
            slug=str(raw.get("slug", "")),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to parse market", error=str(exc), raw_keys=list(raw.keys()))
        return None


def _parse_order_book(token_id: str, raw: Dict[str, Any]) -> OrderBook:
    """Parse the /book endpoint response into an OrderBook."""
    raw_bids: List[Dict[str, Any]] = raw.get("bids", [])
    raw_asks: List[Dict[str, Any]] = raw.get("asks", [])

    bids = sorted(
        [OrderBookLevel(price=float(b["price"]), size=float(b["size"])) for b in raw_bids],
        key=lambda x: x.price,
        reverse=True,  # highest bid first
    )
    asks = sorted(
        [OrderBookLevel(price=float(a["price"]), size=float(a["size"])) for a in raw_asks],
        key=lambda x: x.price,  # lowest ask first
    )

    return OrderBook(
        token_id=token_id,
        bids=bids,
        asks=asks,
        timestamp=datetime.utcnow(),
    )


def _parse_order(raw: Dict[str, Any]) -> Order:
    status_str = str(raw.get("status", "PENDING")).upper()
    try:
        status = OrderStatus(status_str)
    except ValueError:
        status = OrderStatus.PENDING

    return Order(
        order_id=str(raw.get("id", raw.get("orderID", ""))),
        market_id=str(raw.get("market", raw.get("conditionId", ""))),
        token_id=str(raw.get("asset_id", raw.get("tokenId", ""))),
        side=str(raw.get("side", "BUY")).upper(),
        price=float(raw.get("price", 0.0)),
        size=float(raw.get("original_size", raw.get("size", 0.0))),
        status=status,
        timestamp=datetime.utcfromtimestamp(
            float(raw.get("created_at", time.time()))
        ),
        filled_size=float(raw.get("size_matched", raw.get("filledSize", 0.0))),
        average_fill_price=float(raw.get("average_price", raw.get("avgPrice", 0.0))),
    )


# ── Polymarket client ─────────────────────────────────────────────────────────

class PolymarketClient:
    """
    Async Polymarket CLOB API client.

    Handles both public (unauthenticated) and private (L1/L2 authenticated)
    endpoints.  Uses aiohttp for all HTTP communication.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._base_url = settings.polymarket_base_url.rstrip("/")
        self._gamma_url = settings.polymarket_gamma_url.rstrip("/")
        self._private_key = settings.polymarket_private_key
        self._api_key = settings.polymarket_api_key
        self._api_secret = settings.polymarket_api_secret
        self._passphrase = settings.polymarket_passphrase

        connector = aiohttp.TCPConnector(limit=50, ssl=False)
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self._session: aiohttp.ClientSession = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        logger.info("PolymarketClient initialised", base_url=self._base_url)

    # ── Low-level HTTP helpers ────────────────────────────────────────────────

    async def _get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        headers: Dict[str, str] = {}
        if extra_headers:
            headers.update(extra_headers)
        async with self._session.get(url, params=params, headers=headers) as resp:
            resp.raise_for_status()
            return await resp.json(content_type=None)

    async def _post(
        self,
        url: str,
        payload: Dict[str, Any],
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        body_str = json.dumps(payload)
        headers: Dict[str, str] = {}
        if extra_headers:
            headers.update(extra_headers)
        async with self._session.post(url, data=body_str, headers=headers) as resp:
            resp.raise_for_status()
            return await resp.json(content_type=None)

    async def _delete(
        self,
        url: str,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        headers: Dict[str, str] = {}
        if extra_headers:
            headers.update(extra_headers)
        async with self._session.delete(url, headers=headers) as resp:
            resp.raise_for_status()
            return await resp.json(content_type=None)

    # ── Auth header builders ──────────────────────────────────────────────────

    def _auth_headers_l2(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        """Return L2 (API-key HMAC) auth headers for the given request."""
        return _l2_headers(
            api_key=self._api_key,
            api_secret=self._api_secret,
            passphrase=self._passphrase,
            method=method,
            path=path,
            body=body,
        )

    def _auth_headers_l1(self, method: str, path: str) -> Dict[str, str]:
        """Return L1 (wallet signing) auth headers for the given request."""
        return _l1_headers(
            private_key=self._private_key,
            method=method,
            path=path,
        )

    # ── Public market endpoints ───────────────────────────────────────────────

    async def get_btc_markets(self) -> List[Market]:
        """
        Fetch all active Polymarket markets containing BTC/bitcoin in their question.
        Handles cursor-based pagination from the Gamma API.
        """
        markets: List[Market] = []
        cursor: Optional[str] = None
        keywords = ("btc", "bitcoin")
        page_size = 100

        while True:
            params: Dict[str, Any] = {
                "active": "true",
                "limit": page_size,
            }
            if cursor:
                params["next_cursor"] = cursor

            try:
                url = f"{self._gamma_url}/markets"
                data = await self._get(url, params=params)
            except aiohttp.ClientResponseError as exc:
                logger.error("Failed to fetch markets page", status=exc.status, message=exc.message)
                break
            except Exception as exc:
                logger.error("Unexpected error fetching markets", error=str(exc))
                break

            # Gamma API returns {"data": [...], "next_cursor": "..."}
            if isinstance(data, dict):
                raw_markets: List[Dict[str, Any]] = data.get("data", [])
                next_cursor: Optional[str] = data.get("next_cursor")
            elif isinstance(data, list):
                raw_markets = data
                next_cursor = None
            else:
                logger.warning("Unexpected markets response type", type=type(data).__name__)
                break

            for raw in raw_markets:
                question = str(raw.get("question", "")).lower()
                if not any(kw in question for kw in keywords):
                    continue
                market = _parse_market(raw)
                if market is not None and market.active:
                    markets.append(market)

            logger.debug(
                "Markets page fetched",
                page_size=len(raw_markets),
                btc_found=len(markets),
                has_next=bool(next_cursor),
            )

            if not next_cursor or next_cursor == "LTE=":
                # LTE= is Polymarket's sentinel for "end of results"
                break
            cursor = next_cursor

        logger.info("BTC markets fetched", count=len(markets))
        return markets

    async def get_order_book(self, token_id: str) -> OrderBook:
        """Fetch the order book for a given token ID from /book."""
        url = f"{self._base_url}/book"
        try:
            data = await self._get(url, params={"token_id": token_id})
            return _parse_order_book(token_id, data)
        except aiohttp.ClientResponseError as exc:
            logger.error("Failed to fetch order book", token_id=token_id, status=exc.status)
            raise
        except Exception as exc:
            logger.error("Unexpected error fetching order book", token_id=token_id, error=str(exc))
            raise

    async def get_midpoint(self, token_id: str) -> float:
        """Return the current midpoint price for a token from /midpoint."""
        url = f"{self._base_url}/midpoint"
        try:
            data = await self._get(url, params={"token_id": token_id})
            # Response: {"mid": "0.55"} or {"price": 0.55}
            mid = data.get("mid") or data.get("price") or data.get("midpoint")
            return float(mid) if mid is not None else 0.5
        except aiohttp.ClientResponseError as exc:
            logger.error("Failed to fetch midpoint", token_id=token_id, status=exc.status)
            raise
        except Exception as exc:
            logger.error("Unexpected error fetching midpoint", token_id=token_id, error=str(exc))
            raise

    async def get_price_history(self, market_id: str) -> List[Dict[str, Any]]:
        """
        Fetch price history from /prices-history for a given market_id.
        Returns a list of raw dicts with 't' (timestamp) and 'p' (price) keys.
        """
        url = f"{self._base_url}/prices-history"
        try:
            data = await self._get(url, params={"market": market_id, "interval": "1d"})
            if isinstance(data, dict):
                return data.get("history", [])
            if isinstance(data, list):
                return data
            return []
        except aiohttp.ClientResponseError as exc:
            logger.error("Failed to fetch price history", market_id=market_id, status=exc.status)
            raise
        except Exception as exc:
            logger.error("Unexpected error fetching price history", market_id=market_id, error=str(exc))
            raise

    # ── Authenticated endpoints ───────────────────────────────────────────────

    async def get_balance(self) -> float:
        """
        Return the current USDC balance/allowance for the authenticated account.
        Requires L2 (API-key) authentication.
        """
        path = "/balance-allowance"
        url = f"{self._base_url}{path}"
        auth = self._auth_headers_l2("GET", path)
        try:
            data = await self._get(url, extra_headers=auth)
            # Response contains per-asset allowances; we want USDC
            # Shape: [{"asset": "USDC", "balance": "100.0", ...}, ...]
            #    or  {"balance": "100.0"}
            if isinstance(data, list):
                for entry in data:
                    asset = str(entry.get("asset_id", entry.get("asset", ""))).upper()
                    if "USDC" in asset or asset == "0":
                        return float(entry.get("balance", 0.0))
                # Fall back to first entry
                return float(data[0].get("balance", 0.0)) if data else 0.0
            return float(data.get("balance", 0.0))
        except aiohttp.ClientResponseError as exc:
            logger.error("Failed to fetch balance", status=exc.status, message=exc.message)
            raise
        except Exception as exc:
            logger.error("Unexpected error fetching balance", error=str(exc))
            raise

    async def place_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
        order_type: str = "GTC",
    ) -> Order:
        """
        Place a limit order on Polymarket.

        Args:
            token_id:   The token to trade (yes or no token ID).
            side:       'BUY' or 'SELL'.
            price:      Limit price in USDC (0.0 – 1.0 for binary markets).
            size:       Order size in USDC.
            order_type: 'GTC' (good-till-cancelled) or 'FOK'/'IOC'.

        Returns:
            An Order dataclass representing the placed order.
        """
        path = "/order"
        url = f"{self._base_url}{path}"

        order_id = str(uuid.uuid4())
        payload: Dict[str, Any] = {
            "orderID": order_id,
            "tokenID": token_id,
            "side": side.upper(),
            "price": str(round(price, 4)),
            "size": str(round(size, 2)),
            "type": order_type,
            "timeInForce": order_type,
            "expiration": 0,
            "nonce": _now_ms(),
        }

        body_str = json.dumps(payload)

        # L1 signature to build the on-chain order hash; L2 for the API call
        l1_auth = self._auth_headers_l1("POST", path)
        l2_auth = self._auth_headers_l2("POST", path, body_str)
        headers = {**l1_auth, **l2_auth}

        try:
            data = await self._post(url, payload, extra_headers=headers)
            logger.info(
                "Order placed",
                token_id=token_id,
                side=side,
                price=price,
                size=size,
                response_id=data.get("orderID", data.get("id")),
            )
            return _parse_order(data)
        except aiohttp.ClientResponseError as exc:
            logger.error("Failed to place order", status=exc.status, message=exc.message)
            raise
        except Exception as exc:
            logger.error("Unexpected error placing order", error=str(exc))
            raise

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order by its ID.
        Returns True on success, False if the order could not be cancelled.
        """
        path = f"/order/{order_id}"
        url = f"{self._base_url}{path}"
        auth = self._auth_headers_l2("DELETE", path)
        try:
            data = await self._delete(url, extra_headers=auth)
            success: bool = bool(data.get("success", data.get("cancelled", True)))
            if success:
                logger.info("Order cancelled", order_id=order_id)
            else:
                logger.warning("Order cancel returned non-success", order_id=order_id, response=data)
            return success
        except aiohttp.ClientResponseError as exc:
            if exc.status == 404:
                logger.warning("Order not found for cancellation", order_id=order_id)
                return False
            logger.error("Failed to cancel order", order_id=order_id, status=exc.status)
            raise
        except Exception as exc:
            logger.error("Unexpected error cancelling order", order_id=order_id, error=str(exc))
            raise

    async def get_open_orders(self, market_id: Optional[str] = None) -> List[Order]:
        """
        Return all open orders, optionally filtered by market_id (condition_id).
        Requires L2 authentication.
        """
        path = "/orders"
        url = f"{self._base_url}{path}"
        params: Dict[str, Any] = {"status": "OPEN"}
        if market_id:
            params["market"] = market_id
        auth = self._auth_headers_l2("GET", path)
        try:
            data = await self._get(url, params=params, extra_headers=auth)
            raw_orders: List[Dict[str, Any]] = data if isinstance(data, list) else data.get("data", [])
            orders = [_parse_order(o) for o in raw_orders]
            logger.debug("Open orders fetched", count=len(orders), market_id=market_id)
            return orders
        except aiohttp.ClientResponseError as exc:
            logger.error("Failed to fetch open orders", status=exc.status)
            raise
        except Exception as exc:
            logger.error("Unexpected error fetching open orders", error=str(exc))
            raise

    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        Return current on-chain positions for the authenticated account.
        Returns a list of raw dicts (asset, size, average_price, etc.).
        """
        path = "/positions"
        url = f"{self._base_url}{path}"
        auth = self._auth_headers_l2("GET", path)
        try:
            data = await self._get(url, extra_headers=auth)
            positions: List[Dict[str, Any]] = data if isinstance(data, list) else data.get("data", [])
            logger.debug("Positions fetched", count=len(positions))
            return positions
        except aiohttp.ClientResponseError as exc:
            logger.error("Failed to fetch positions", status=exc.status)
            raise
        except Exception as exc:
            logger.error("Unexpected error fetching positions", error=str(exc))
            raise

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Close the underlying aiohttp session."""
        if not self._session.closed:
            await self._session.close()
            logger.info("PolymarketClient session closed")
