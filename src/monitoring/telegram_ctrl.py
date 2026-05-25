"""
Telegram controller for PolyBTC Trader.

Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env to enable.
Create a bot at t.me/BotFather first to get a token.

Commands (send from iPhone):
  /status   — signal, confidence, market prices, T-countdown
  /balance  — paper balance, P&L, win rate, open position
  /trades   — last 5 closed trades
  /stop     — gracefully stop the bot
  /help     — list commands
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Optional

import aiohttp
from loguru import logger


class TelegramController:
    _BASE = "https://api.telegram.org/bot{token}/{method}"

    def __init__(
        self,
        token: str,
        chat_id: str,
        engine,
        paper,
        stop_event: asyncio.Event,
    ):
        self._token = token
        self._chat_id = str(chat_id)
        self._engine = engine
        self._paper = paper
        self._stop = stop_event
        self._offset = 0
        self._session: Optional[aiohttp.ClientSession] = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=40)
        )
        await self.send("*PolyBTC Trader online* ✅\nSend /help for commands.")
        asyncio.create_task(self._poll_loop())
        logger.info("[TG] Telegram controller active")

    async def stop(self) -> None:
        await self.send("*PolyBTC Trader stopped* 🛑")
        if self._session:
            await self._session.close()

    # ── Long-poll loop ─────────────────────────────────────────────────────────

    async def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                updates = await self._get_updates()
                for u in updates:
                    self._offset = u["update_id"] + 1
                    msg = u.get("message", {})
                    text = msg.get("text", "").strip()
                    from_id = str(msg.get("chat", {}).get("id", ""))
                    if text.startswith("/") and from_id == self._chat_id:
                        asyncio.create_task(self._handle(text.lower()))
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"[TG] Poll error: {e}")
                await asyncio.sleep(5)

    async def _get_updates(self) -> list:
        url = self._BASE.format(token=self._token, method="getUpdates")
        async with self._session.get(
            url, params={"timeout": 30, "offset": self._offset}
        ) as r:
            data = await r.json()
        return data.get("result", []) if data.get("ok") else []

    # ── Command handlers ───────────────────────────────────────────────────────

    async def _handle(self, cmd: str) -> None:
        try:
            if cmd.startswith("/status"):
                await self._cmd_status()
            elif cmd.startswith("/balance"):
                await self._cmd_balance()
            elif cmd.startswith("/trades"):
                await self._cmd_trades()
            elif cmd.startswith("/stop"):
                await self.send("Stopping bot... 🛑")
                self._stop.set()
            elif cmd.startswith("/help"):
                await self.send(
                    "*Commands*\n"
                    "/status — signal & market prices\n"
                    "/balance — P&L & balance\n"
                    "/trades — last 5 trades\n"
                    "/stop — stop the bot\n"
                    "/help — this message"
                )
            else:
                await self.send("Unknown command. Send /help")
        except Exception as e:
            logger.debug(f"[TG] Command error: {e}")

    async def _cmd_status(self) -> None:
        st = self._engine.status()
        mkt = st.get("current_market") or {}
        secs = mkt.get("seconds_to_close", 0)
        mins, s = divmod(int(secs), 60)
        direction = st.get("signal_direction", "—")
        conf = st.get("confidence", 0)
        edge = st.get("edge", 0)
        reason = st.get("no_trade_reason", "")

        lines = [
            f"*Signal:* {direction}  |  *Conf:* {conf:.0f}/100",
            f"*Edge:* {edge*100:+.1f}%",
            f"*T-minus:* {mins}m {s:02d}s",
        ]
        if mkt:
            lines += [
                f"*YES:* {mkt.get('yes_bid',0):.3f} / {mkt.get('yes_ask',0):.3f}",
                f"*NO:*  {mkt.get('no_bid',0):.3f} / {mkt.get('no_ask',0):.3f}",
                f"*Sum ask:* {mkt.get('sum_ask',0):.3f}",
            ]
        if reason:
            lines.append(f"_{reason}_")
        await self.send("\n".join(lines))

    async def _cmd_balance(self) -> None:
        ps = self._paper.stats()
        sign = "+" if ps["total_pnl"] >= 0 else ""
        lines = [
            f"*Balance:* ${ps['balance']:.2f}  (start ${ps['initial']:.0f})",
            f"*Total P&L:* {sign}${ps['total_pnl']:.2f}",
            f"*Today P&L:* {'+' if ps['daily_pnl']>=0 else ''}${ps['daily_pnl']:.2f}",
            f"*Win rate:* {ps['win_rate']:.1f}%  ({ps['wins']}W / {ps['losses']}L)",
            f"*Trades:* {ps['total_trades']}",
        ]
        if ps.get("open"):
            o = ps["open"]
            lines.append(
                f"*Open:* {o['direction']} @ {o['entry_price']:.3f}  "
                f"(${o['size_usdc']:.0f})"
            )
        await self.send("\n".join(lines))

    async def _cmd_trades(self) -> None:
        ps = self._paper.stats()
        recent = ps.get("recent", [])
        if not recent:
            await self.send("No closed trades yet.")
            return
        lines = ["*Last trades:*"]
        for t in recent[:5]:
            icon = "✅" if t.get("outcome") == "win" else "❌"
            pnl = t.get("pnl", 0)
            lines.append(
                f"{icon} {t.get('direction','?')} "
                f"{'+' if pnl>=0 else ''}${pnl:.2f}  "
                f"@ {t.get('exit_time','')[:16]}"
            )
        await self.send("\n".join(lines))

    # ── Trade alert (called externally when a trade fires) ─────────────────────

    async def alert_trade(self, decision) -> None:
        ps = self._paper.stats()
        await self.send(
            f"*Trade entered* 🎯\n"
            f"Direction: *{decision.direction}*\n"
            f"Entry: {decision.entry_price:.3f}  Size: ${decision.size_usdc:.0f}\n"
            f"Confidence: {decision.confidence:.0f}  T-{decision.seconds_to_close:.0f}s\n"
            f"Balance: ${ps['balance']:.2f}"
        )

    async def alert_close(self, direction: str, pnl: float, outcome: str, balance: float) -> None:
        icon = "✅" if outcome == "win" else "❌"
        await self.send(
            f"{icon} *Trade closed* — {outcome.upper()}\n"
            f"Direction: {direction}  P&L: {'+' if pnl>=0 else ''}${pnl:.2f}\n"
            f"Balance: ${balance:.2f}"
        )

    # ── Send ───────────────────────────────────────────────────────────────────

    async def send(self, text: str) -> None:
        if not self._session:
            return
        try:
            url = self._BASE.format(token=self._token, method="sendMessage")
            await self._session.post(url, json={
                "chat_id": self._chat_id,
                "text": text,
                "parse_mode": "Markdown",
            })
        except Exception as e:
            logger.debug(f"[TG] Send error: {e}")
