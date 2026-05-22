"""
Telegram bot for trade alerts and status updates.

Sends async HTTP requests to the Telegram Bot API using aiohttp.
Gracefully handles:
    - missing token (all send calls become no-ops)
    - rate limits (exponential backoff up to 60 s)
    - transient network errors (logged, not re-raised)
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict, Optional

import aiohttp
from loguru import logger

from src.core.models import Trade, TradeOutcome, TradeSignal


_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
_MAX_RETRIES = 4
_BACKOFF_BASE = 1.0    # seconds
_BACKOFF_CAP = 60.0    # seconds


class TelegramAlerter:
    """
    Async Telegram alerter.

    Parameters
    ----------
    settings:
        Application Settings object. Uses:
            ``telegram_bot_token``  — bot token from @BotFather
            ``telegram_chat_id``    — target chat / group id
            ``telegram_enabled``    — master on/off flag
    """

    def __init__(self, settings) -> None:
        self._token: str = getattr(settings, "telegram_bot_token", "")
        self._chat_id: str = getattr(settings, "telegram_chat_id", "")
        self._enabled: bool = bool(getattr(settings, "telegram_enabled", False))

    # ── Core send ─────────────────────────────────────────────────────────────

    async def send_message(self, text: str) -> None:
        """
        Send a plain-text message to the configured Telegram chat.

        Does nothing when ``telegram_enabled`` is False or no token is set.
        Uses exponential backoff on rate-limit (HTTP 429) responses.
        """
        if not self._enabled or not self._token or not self._chat_id:
            return

        url = _TELEGRAM_API.format(token=self._token)
        payload: Dict[str, Any] = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": "Markdown",
        }

        delay = _BACKOFF_BASE
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as resp:
                        if resp.status == 200:
                            logger.debug("Telegram message sent", chars=len(text))
                            return

                        if resp.status == 429:
                            body = await resp.json(content_type=None)
                            retry_after = float(
                                body.get("parameters", {}).get("retry_after", delay)
                            )
                            wait = min(retry_after, _BACKOFF_CAP)
                            logger.warning(
                                "Telegram rate-limited",
                                retry_after=wait,
                                attempt=attempt,
                            )
                            await asyncio.sleep(wait)
                            delay = min(delay * 2, _BACKOFF_CAP)
                            continue

                        logger.warning(
                            "Telegram send failed",
                            status=resp.status,
                            attempt=attempt,
                        )
                        return

            except aiohttp.ClientError as exc:
                logger.warning(
                    "Telegram network error",
                    error=str(exc),
                    attempt=attempt,
                )
                if attempt < _MAX_RETRIES:
                    await asyncio.sleep(min(delay, _BACKOFF_CAP))
                    delay *= 2
            except Exception as exc:
                logger.error("Unexpected Telegram error", error=str(exc))
                return

    # ── Alert methods ─────────────────────────────────────────────────────────

    async def alert_signal(self, signal: TradeSignal) -> None:
        """Send a formatted signal-detected alert."""
        await self.send_message(self._format_signal(signal))

    async def alert_trade_opened(self, trade: Trade) -> None:
        """Send an alert when a new trade is opened."""
        msg = self._format_trade(trade)
        header = "🟢 *Trade Opened*\n" if trade.direction.value == "YES" else "🔴 *Trade Opened*\n"
        await self.send_message(header + msg)

    async def alert_trade_closed(self, trade: Trade) -> None:
        """Send an alert when a trade is closed with P&L summary."""
        pnl = trade.realized_pnl or 0.0
        emoji = "✅" if pnl >= 0 else "❌"
        header = f"{emoji} *Trade Closed*\n"
        msg = self._format_trade(trade)
        outcome_line = f"Outcome: *{trade.outcome.value.upper()}*"
        pnl_line = f"P&L: `${pnl:+.2f}`"
        await self.send_message(header + msg + f"\n{outcome_line}\n{pnl_line}")

    async def alert_risk_event(self, event_type: str, message: str) -> None:
        """
        Send a risk event alert (kill switch, cooldown, drawdown warning).

        Parameters
        ----------
        event_type:
            e.g. "KILL_SWITCH", "COOLDOWN", "DAILY_DRAWDOWN"
        message:
            Human-readable description of the event.
        """
        ts = datetime.utcnow().strftime("%H:%M:%S UTC")
        text = (
            f"⚠️ *Risk Event: {event_type}*\n"
            f"Time: `{ts}`\n"
            f"Details: {message}"
        )
        await self.send_message(text)

    async def alert_daily_summary(self, metrics: dict) -> None:
        """
        Send a daily P&L and performance summary.

        Parameters
        ----------
        metrics:
            Dict with keys: total_pnl, win_rate, trade_count,
            total_pnl_pct, max_drawdown_pct (all optional, default 0).
        """
        total_pnl = metrics.get("total_pnl", 0.0)
        win_rate = metrics.get("win_rate", 0.0)
        trade_count = metrics.get("trade_count", 0)
        pnl_pct = metrics.get("total_pnl_pct", 0.0)
        max_dd = metrics.get("max_drawdown_pct", 0.0)

        emoji = "📈" if total_pnl >= 0 else "📉"
        ts = datetime.utcnow().strftime("%Y-%m-%d")

        text = (
            f"{emoji} *Daily Summary — {ts}*\n"
            f"Trades    : `{trade_count}`\n"
            f"Win Rate  : `{win_rate * 100:.1f}%`\n"
            f"P&L       : `${total_pnl:+.2f}` ({pnl_pct * 100:+.2f}%)\n"
            f"Max DD    : `{max_dd * 100:.2f}%`"
        )
        await self.send_message(text)

    async def alert_error(self, error: str) -> None:
        """Send a system error alert."""
        ts = datetime.utcnow().strftime("%H:%M:%S UTC")
        text = f"🔥 *System Error*\nTime: `{ts}`\n```\n{error[:400]}\n```"
        await self.send_message(text)

    # ── Formatters ────────────────────────────────────────────────────────────

    def _format_signal(self, signal: TradeSignal) -> str:
        """
        Build a Markdown-formatted signal alert message.

        Includes direction, confidence, edge, implied probability, and
        the top-3 signal reasons.
        """
        dir_emoji = "🟢" if signal.direction.value == "YES" else "🔴"
        ts = signal.timestamp.strftime("%H:%M:%S UTC")
        reasons_text = "\n".join(
            f"  • {r}" for r in signal.reasons[:3]
        )

        return (
            f"{dir_emoji} *Signal Detected*\n"
            f"Market    : `{signal.market_id[:30]}`\n"
            f"Direction : *{signal.direction.value}*\n"
            f"Confidence: `{signal.confidence:.1f}/100`\n"
            f"Price     : `{signal.price:.4f}`\n"
            f"Edge      : `{signal.edge * 100:.1f}%`\n"
            f"Impl Prob : `{signal.implied_probability:.2%}`\n"
            f"Time      : `{ts}`\n"
            f"Reasons:\n{reasons_text}"
        )

    def _format_trade(self, trade: Trade) -> str:
        """
        Build a Markdown-formatted trade details message.

        Shows direction, size, entry price, and holding time for closed trades.
        """
        ts = trade.entry_time.strftime("%H:%M:%S UTC")
        holding = (
            f"{trade.holding_hours:.1f}h" if trade.holding_hours is not None else "open"
        )
        paper = "Paper" if trade.paper_trade else "Live"

        exit_line = ""
        if trade.exit_price is not None:
            exit_line = f"Exit Price: `{trade.exit_price:.4f}`\n"

        return (
            f"ID        : `{trade.trade_id[:12]}`\n"
            f"Market    : `{trade.market_id[:28]}`\n"
            f"Direction : *{trade.direction.value}*\n"
            f"Size      : `{trade.size:.4f} tokens`\n"
            f"Entry     : `{trade.entry_price:.4f}` @ {ts}\n"
            f"{exit_line}"
            f"Held      : `{holding}`\n"
            f"Mode      : `{paper}`\n"
            f"Conf      : `{trade.confidence:.1f}`"
        )
