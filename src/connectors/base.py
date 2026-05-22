"""Abstract base connector with exponential-backoff retry, circuit breaker, and token-bucket rate limiter."""
from __future__ import annotations

import asyncio
import functools
import random
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Coroutine, Optional, TypeVar

from loguru import logger


# ── Type var for generic decorator ───────────────────────────────────────────

_T = TypeVar("_T")


# ── Exponential-backoff retry decorator ──────────────────────────────────────

def retry_async(
    max_attempts: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., Coroutine[Any, Any, _T]]], Callable[..., Coroutine[Any, Any, _T]]]:
    """
    Decorator that retries an async function with full-jitter exponential backoff.

    Args:
        max_attempts: Maximum number of total call attempts (including the first).
        base_delay:   Initial delay in seconds before the first retry.
        max_delay:    Cap on the computed delay.
        exceptions:   Only retry on these exception types.
    """
    def decorator(fn: Callable[..., Coroutine[Any, Any, _T]]) -> Callable[..., Coroutine[Any, Any, _T]]:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> _T:
            last_exc: Optional[Exception] = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await fn(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        logger.error(
                            "All retry attempts exhausted",
                            fn=fn.__qualname__,
                            attempts=attempt,
                            error=str(exc),
                        )
                        raise
                    # Full-jitter: sleep for a random value in [0, cap]
                    cap = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    delay = random.uniform(0, cap)
                    logger.warning(
                        "Retrying after error",
                        fn=fn.__qualname__,
                        attempt=attempt,
                        delay=round(delay, 2),
                        error=str(exc),
                    )
                    await asyncio.sleep(delay)
            # Should be unreachable, but satisfies type checker
            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator


# ── Circuit breaker ───────────────────────────────────────────────────────────

class CircuitState(str, Enum):
    CLOSED = "CLOSED"       # Normal operation
    OPEN = "OPEN"           # Failing — reject calls immediately
    HALF_OPEN = "HALF_OPEN" # Probe — allow one call through


class CircuitBreaker:
    """
    Classic three-state circuit breaker.

    When *failure_threshold* consecutive failures occur the circuit opens and
    all subsequent calls raise CircuitOpenError until *recovery_timeout* seconds
    elapse, after which one probe call is allowed (HALF_OPEN).
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        name: str = "circuit",
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._name = name

        self._state: CircuitState = CircuitState.CLOSED
        self._consecutive_failures: int = 0
        self._opened_at: Optional[float] = None
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def is_closed(self) -> bool:
        return self._state == CircuitState.CLOSED

    async def _transition(self, new_state: CircuitState) -> None:
        if self._state != new_state:
            logger.info(
                "Circuit breaker state change",
                name=self._name,
                old=self._state.value,
                new=new_state.value,
            )
        self._state = new_state

    async def call(self, coro: Coroutine[Any, Any, _T]) -> _T:
        """
        Execute *coro* through the circuit breaker.

        Raises:
            CircuitOpenError: When the circuit is OPEN and the recovery timeout
                              has not yet elapsed.
        """
        async with self._lock:
            if self._state == CircuitState.OPEN:
                elapsed = time.monotonic() - (self._opened_at or 0)
                if elapsed < self._recovery_timeout:
                    raise CircuitOpenError(
                        f"Circuit '{self._name}' is OPEN "
                        f"({self._recovery_timeout - elapsed:.1f}s until probe)"
                    )
                await self._transition(CircuitState.HALF_OPEN)

        try:
            result = await coro
            async with self._lock:
                self._consecutive_failures = 0
                await self._transition(CircuitState.CLOSED)
            return result
        except Exception as exc:
            async with self._lock:
                self._consecutive_failures += 1
                if (
                    self._state in (CircuitState.CLOSED, CircuitState.HALF_OPEN)
                    and self._consecutive_failures >= self._failure_threshold
                ):
                    self._opened_at = time.monotonic()
                    await self._transition(CircuitState.OPEN)
            raise exc

    async def reset(self) -> None:
        """Manually reset the circuit breaker to CLOSED."""
        async with self._lock:
            self._consecutive_failures = 0
            self._opened_at = None
            await self._transition(CircuitState.CLOSED)


class CircuitOpenError(Exception):
    """Raised when a call is rejected because the circuit is OPEN."""


# ── Token-bucket rate limiter ─────────────────────────────────────────────────

class RateLimiter:
    """
    Async-safe token-bucket rate limiter.

    Tokens refill continuously at *rate* tokens/second up to *capacity*.
    Callers ``await acquire(n)`` to consume *n* tokens, blocking until
    sufficient tokens are available.
    """

    def __init__(self, rate: float, capacity: float) -> None:
        if rate <= 0 or capacity <= 0:
            raise ValueError("rate and capacity must be positive")
        self._rate = rate
        self._capacity = capacity
        self._tokens: float = capacity
        self._last_refill: float = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> None:
        """Block until *tokens* are available, then consume them."""
        if tokens > self._capacity:
            raise ValueError(
                f"Requested {tokens} tokens exceeds bucket capacity {self._capacity}"
            )
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                # Calculate how long to wait for the deficit to fill
                deficit = tokens - self._tokens
                wait = deficit / self._rate

            await asyncio.sleep(wait)

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last_refill = now

    @property
    def available(self) -> float:
        """Snapshot of available tokens (not thread-safe, informational only)."""
        self._refill()
        return self._tokens


# ── Connection manager ────────────────────────────────────────────────────────

class ConnectionManager:
    """
    Manages the reconnection loop for long-lived async connections.

    Subclasses or callers provide *connect_fn* and *disconnect_fn*.
    ``run()`` drives the loop; ``stop()`` requests a graceful shutdown.
    """

    def __init__(
        self,
        connect_fn: Callable[[], Coroutine[Any, Any, None]],
        disconnect_fn: Callable[[], Coroutine[Any, Any, None]],
        max_attempts: int = 10,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        name: str = "connection",
    ) -> None:
        self._connect_fn = connect_fn
        self._disconnect_fn = disconnect_fn
        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._name = name

        self._stop_event = asyncio.Event()
        self._connected = False
        self._attempt_count = 0

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def run(self) -> None:
        """
        Keep the connection alive.  Reconnects with exponential backoff + jitter
        until *max_attempts* consecutive failures or ``stop()`` is called.
        """
        self._stop_event.clear()
        consecutive_failures = 0

        while not self._stop_event.is_set():
            try:
                logger.info("Connecting", name=self._name, attempt=consecutive_failures + 1)
                await self._connect_fn()
                self._connected = True
                consecutive_failures = 0
                self._attempt_count += 1
                logger.info("Connected", name=self._name)

                # Block here — connect_fn is expected to run until disconnected
                # For WebSocket connectors the connect_fn should raise on disconnect
            except asyncio.CancelledError:
                logger.info("Connection manager cancelled", name=self._name)
                break
            except Exception as exc:
                self._connected = False
                consecutive_failures += 1
                logger.warning(
                    "Connection failed",
                    name=self._name,
                    error=str(exc),
                    consecutive_failures=consecutive_failures,
                )

                if consecutive_failures >= self._max_attempts:
                    logger.error(
                        "Max reconnect attempts reached — giving up",
                        name=self._name,
                        max=self._max_attempts,
                    )
                    break

                cap = min(self._base_delay * (2 ** (consecutive_failures - 1)), self._max_delay)
                delay = random.uniform(cap / 2, cap)  # decorrelated jitter
                logger.info(
                    "Reconnecting after delay",
                    name=self._name,
                    delay=round(delay, 2),
                )
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    pass  # Normal — just means the delay elapsed

        self._connected = False
        try:
            await self._disconnect_fn()
        except Exception as exc:
            logger.debug("Error during disconnect cleanup", name=self._name, error=str(exc))
        logger.info("Connection manager stopped", name=self._name)

    async def stop(self) -> None:
        """Signal the reconnection loop to exit after the current connection."""
        self._stop_event.set()
        self._connected = False


# ── Abstract base connector ───────────────────────────────────────────────────

class BaseConnector(ABC):
    """
    Abstract base for all external service connectors.

    Concrete subclasses must implement ``connect()``, ``disconnect()``,
    and ``is_connected()``.  Resilience primitives (circuit breaker, rate
    limiter, retry decorator) are available as class-level helpers.
    """

    def __init__(
        self,
        name: str = "connector",
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        rate: float = 10.0,
        capacity: float = 20.0,
    ) -> None:
        self._name = name
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            name=name,
        )
        self.rate_limiter = RateLimiter(rate=rate, capacity=capacity)

    @abstractmethod
    async def connect(self) -> None:
        """Establish the connection."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Tear down the connection and release resources."""

    @abstractmethod
    def is_connected(self) -> bool:
        """Return True if the connector currently has an active connection."""

    async def __aenter__(self) -> "BaseConnector":
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.disconnect()
