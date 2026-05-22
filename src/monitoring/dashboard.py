"""
Rich terminal dashboard showing live system state.

Refreshes every 2 seconds using Rich's Live context manager.
All data is sourced via a ``state_provider`` callable so the dashboard
remains decoupled from the rest of the system.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from loguru import logger
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.core.models import (
    Direction,
    Position,
    RiskState,
    Trade,
    TradeOutcome,
    TradeSignal,
)


_console = Console()
_REFRESH_SECONDS = 2


class Dashboard:
    """
    Full-terminal trading dashboard using Rich.

    Parameters
    ----------
    state_provider:
        Async callable ``() -> dict`` that returns the current system state.
        Expected keys (all optional — missing keys show as N/A):
            btc_price       float
            btc_change_24h  float (e.g. 0.023 = +2.3%)
            risk_state      RiskState
            open_positions  List[Position]
            recent_signals  List[TradeSignal]
            recent_trades   List[Trade]
            system_status   dict (api_ok, last_scan, error_count, mode)
    """

    def __init__(self) -> None:
        self._running = False

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def run(self, state_provider: Callable) -> None:
        """
        Start the dashboard main loop.

        Calls ``state_provider()`` every ``_REFRESH_SECONDS`` seconds and
        re-renders the layout.  Stops cleanly when ``self._running`` is set
        to False or when KeyboardInterrupt is raised.
        """
        self._running = True

        with Live(
            self._build_layout({}),
            console=_console,
            refresh_per_second=1,
            screen=True,
        ) as live:
            while self._running:
                try:
                    state = await state_provider()
                    live.update(self._build_layout(state))
                except KeyboardInterrupt:
                    self._running = False
                    break
                except Exception as exc:
                    logger.error("Dashboard state_provider error", error=str(exc))

                await asyncio.sleep(_REFRESH_SECONDS)

    def stop(self) -> None:
        """Signal the run loop to stop on the next iteration."""
        self._running = False

    # ── Layout builder ────────────────────────────────────────────────────────

    def _build_layout(self, state: Dict[str, Any]) -> Layout:
        layout = Layout(name="root")

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="top", size=8),
            Layout(name="middle", size=14),
            Layout(name="bottom"),
        )

        layout["top"].split_row(
            Layout(name="btc_price"),
            Layout(name="risk_state"),
            Layout(name="system_status"),
        )

        layout["middle"].split_row(
            Layout(name="positions"),
            Layout(name="signals"),
        )

        layout["bottom"].update(
            self.render_trades(state.get("recent_trades", []))
        )

        layout["header"].update(self._render_header(state))
        layout["btc_price"].update(self._render_btc_price(state))
        layout["risk_state"].update(
            self.render_risk_state(state.get("risk_state"))
        )
        layout["system_status"].update(self._render_system_status(state))
        layout["positions"].update(
            self.render_positions(state.get("open_positions", []))
        )
        layout["signals"].update(
            self.render_signals(state.get("recent_signals", []))
        )

        return layout

    # ── Section renderers ─────────────────────────────────────────────────────

    def _render_header(self, state: Dict[str, Any]) -> Panel:
        sys_status = state.get("system_status", {})
        mode = sys_status.get("mode", "PAPER")
        env = sys_status.get("environment", "development")
        now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        mode_color = "green" if mode == "PAPER" else "red"
        text = Text()
        text.append("  PolyBTC Trader  ", style="bold cyan")
        text.append(f"| {env.upper()} ", style="dim")
        text.append(f"| [{mode_color}]{mode}[/{mode_color}] ", style="")
        text.append(f"| {now_str}", style="dim")

        return Panel(text, box=box.HORIZONTALS, style="bold")

    def _render_btc_price(self, state: Dict[str, Any]) -> Panel:
        price = state.get("btc_price")
        change = state.get("btc_change_24h", 0.0)

        if price is not None:
            price_str = f"${price:,.2f}"
            change_color = "green" if change >= 0 else "red"
            change_str = f"[{change_color}]{change * 100:+.2f}%[/{change_color}] 24h"
        else:
            price_str = "N/A"
            change_str = "—"

        text = Text()
        text.append("BTC/USD\n", style="bold yellow")
        text.append(f"  {price_str}\n", style="bold white")
        text.append(f"  {change_str}", style="")

        return Panel(text, title="BTC Price", border_style="yellow", box=box.ROUNDED)

    def render_risk_state(self, risk_state: Optional[RiskState]) -> Panel:
        """Render the risk state panel."""
        if risk_state is None:
            return Panel("No risk state available", title="Risk State", border_style="dim")

        balance_color = "green" if risk_state.balance >= 0 else "red"
        pnl_color = "green" if risk_state.daily_pnl >= 0 else "red"
        dd_color = "green" if risk_state.daily_drawdown_pct < 0.02 else (
            "yellow" if risk_state.daily_drawdown_pct < 0.04 else "red"
        )

        ks_status = "[red]ACTIVE[/red]" if risk_state.kill_switch_active else "[green]OK[/green]"
        cd_status = "[yellow]YES[/yellow]" if risk_state.in_cooldown else "[green]NO[/green]"
        exp_pct = risk_state.total_exposure_pct * 100

        text = Text()
        text.append(f"Balance    : [{balance_color}]${risk_state.balance:,.2f}[/{balance_color}]\n")
        text.append(f"Daily P&L  : [{pnl_color}]${risk_state.daily_pnl:+.2f}[/{pnl_color}]\n")
        text.append(f"Drawdown   : [{dd_color}]{risk_state.daily_drawdown_pct * 100:.2f}%[/{dd_color}]\n")
        text.append(f"Positions  : {risk_state.open_positions}\n")
        text.append(f"Exposure   : {exp_pct:.1f}%\n")
        text.append(f"Kill Switch: {ks_status}\n")
        text.append(f"Cooldown   : {cd_status}")

        border = "red" if risk_state.kill_switch_active else (
            "yellow" if risk_state.in_cooldown else "green"
        )
        return Panel(text, title="Risk State", border_style=border, box=box.ROUNDED)

    def _render_system_status(self, state: Dict[str, Any]) -> Panel:
        sys = state.get("system_status", {})

        api_ok = sys.get("api_ok", False)
        last_scan = sys.get("last_scan", "Never")
        error_count = sys.get("error_count", 0)
        binance_ok = sys.get("binance_ok", False)
        poly_ok = sys.get("polymarket_ok", False)

        api_color = "green" if api_ok else "red"
        err_color = "green" if error_count == 0 else ("yellow" if error_count < 5 else "red")

        text = Text()
        text.append(f"Binance WS : [{'green' if binance_ok else 'red'}]{'Connected' if binance_ok else 'Disconnected'}[/{'green' if binance_ok else 'red'}]\n")
        text.append(f"Polymarket : [{'green' if poly_ok else 'red'}]{'Connected' if poly_ok else 'Disconnected'}[/{'green' if poly_ok else 'red'}]\n")
        text.append(f"Last Scan  : {last_scan}\n")
        text.append(f"Errors     : [{err_color}]{error_count}[/{err_color}]")

        return Panel(text, title="System Status", border_style="blue", box=box.ROUNDED)

    def render_positions(self, positions: List[Position]) -> Table:
        """Render open positions as a Rich table."""
        table = Table(
            title="Open Positions",
            box=box.SIMPLE_HEAD,
            border_style="cyan",
            show_lines=False,
            expand=True,
        )
        table.add_column("Market", style="cyan", no_wrap=True, max_width=20)
        table.add_column("Dir", justify="center", width=5)
        table.add_column("Size", justify="right", width=8)
        table.add_column("Entry", justify="right", width=8)
        table.add_column("Current", justify="right", width=9)
        table.add_column("Unr P&L", justify="right", width=10)

        for pos in positions[:10]:
            pnl = pos.unrealized_pnl
            pnl_color = "green" if pnl >= 0 else "red"
            dir_color = "green" if pos.direction == Direction.YES else "red"

            table.add_row(
                pos.market_id[:18],
                f"[{dir_color}]{pos.direction.value}[/{dir_color}]",
                f"{pos.size:.3f}",
                f"{pos.entry_price:.4f}",
                f"{pos.current_price:.4f}",
                f"[{pnl_color}]${pnl:+.2f}[/{pnl_color}]",
            )

        if not positions:
            table.add_row("—", "—", "—", "—", "—", "—")

        return Panel(table, border_style="cyan", box=box.ROUNDED)

    def render_signals(self, signals: List[TradeSignal]) -> Table:
        """Render the 5 most recent signals as a Rich table."""
        table = Table(
            title="Recent Signals",
            box=box.SIMPLE_HEAD,
            border_style="magenta",
            show_lines=False,
            expand=True,
        )
        table.add_column("Time", width=8)
        table.add_column("Market", max_width=18)
        table.add_column("Dir", justify="center", width=5)
        table.add_column("Conf", justify="right", width=6)
        table.add_column("Edge", justify="right", width=7)

        for sig in signals[:5]:
            dir_color = "green" if sig.direction == Direction.YES else "red"
            conf_color = "green" if sig.confidence >= 70 else (
                "yellow" if sig.confidence >= 60 else "red"
            )
            time_str = sig.timestamp.strftime("%H:%M:%S")

            table.add_row(
                time_str,
                sig.market_id[:16],
                f"[{dir_color}]{sig.direction.value}[/{dir_color}]",
                f"[{conf_color}]{sig.confidence:.0f}[/{conf_color}]",
                f"{sig.edge * 100:.1f}%",
            )

        if not signals:
            table.add_row("—", "—", "—", "—", "—")

        return Panel(table, border_style="magenta", box=box.ROUNDED)

    def render_trades(self, trades: List[Trade]) -> Table:
        """Render the 10 most recent closed trades as a Rich table."""
        table = Table(
            title="Recent Closed Trades",
            box=box.SIMPLE_HEAD,
            border_style="blue",
            show_lines=False,
            expand=True,
        )
        table.add_column("Trade ID", width=12, no_wrap=True)
        table.add_column("Market", max_width=20)
        table.add_column("Dir", justify="center", width=5)
        table.add_column("Entry", justify="right", width=8)
        table.add_column("Exit", justify="right", width=8)
        table.add_column("P&L", justify="right", width=10)
        table.add_column("Outcome", justify="center", width=12)
        table.add_column("Conf", justify="right", width=6)

        closed = [t for t in trades if t.outcome != TradeOutcome.OPEN]
        for trade in closed[:10]:
            pnl = trade.realized_pnl or 0.0
            pnl_color = "green" if pnl >= 0 else "red"
            outcome_color = {
                TradeOutcome.WIN: "green",
                TradeOutcome.LOSS: "red",
                TradeOutcome.BREAK_EVEN: "yellow",
                TradeOutcome.OPEN: "dim",
            }.get(trade.outcome, "white")
            dir_color = "green" if trade.direction == Direction.YES else "red"

            table.add_row(
                trade.trade_id[:10],
                trade.market_id[:18],
                f"[{dir_color}]{trade.direction.value}[/{dir_color}]",
                f"{trade.entry_price:.4f}",
                f"{trade.exit_price:.4f}" if trade.exit_price else "—",
                f"[{pnl_color}]${pnl:+.2f}[/{pnl_color}]",
                f"[{outcome_color}]{trade.outcome.value}[/{outcome_color}]",
                f"{trade.confidence:.0f}",
            )

        if not closed:
            table.add_row("—", "—", "—", "—", "—", "—", "—", "—")

        return Panel(table, border_style="blue", box=box.ROUNDED)
