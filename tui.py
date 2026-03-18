#!/usr/bin/env python3
"""
tui.py — interactive streaming annotation TUI for konekaare.

Connects to a running hub (default http://localhost:8000), lets the user pick
an annotator and supply some text (inline or from a file), splits the text into
sentences or paragraphs, streams all units to the hub via WebSocket, and shows
results live as they arrive.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from urllib.parse import quote

import httpx
import websockets
from rich import box
from rich.console import Console
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

BASE_URL = "http://localhost:8000"
console = Console()


# ── helpers ───────────────────────────────────────────────────────────────────


def fetch_annotators(base_url: str) -> list[dict]:
    with httpx.Client() as client:
        resp = client.get(f"{base_url}/annotators", timeout=5)
        resp.raise_for_status()
        return resp.json()


def annotator_table(annotators: list[dict]) -> Table:
    t = Table(box=box.SIMPLE_HEAD, show_footer=False, pad_edge=False)
    t.add_column("#", style="dim", width=3)
    t.add_column("Name", style="bold cyan")
    t.add_column("Type", style="magenta", width=12)
    t.add_column("Description")
    for i, a in enumerate(annotators, 1):
        t.add_row(str(i), a["name"], a["annotation_type"], a.get("description", ""))
    return t


def split_sentences(text: str) -> list[str]:
    """Split on '.', stripping whitespace and empty results."""
    return [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]


def split_paragraphs(text: str) -> list[str]:
    """Split on newlines, stripping whitespace and empty results."""
    return [p.strip() for p in text.split("\n") if p.strip()]


def result_panel(units: list[str], statuses: list[str], span_cells: list[str], done: int) -> Panel:
    t = Table(box=box.SIMPLE, expand=True, show_header=True, pad_edge=False)
    t.add_column("#", width=4, style="dim")
    t.add_column("Text", ratio=4)
    t.add_column("Spans", ratio=4)
    t.add_column("", width=9)  # status

    for i, (unit, status, spans) in enumerate(zip(units, statuses, span_cells)):
        snippet = escape(unit[:52] + "…" if len(unit) > 52 else unit)
        if status == "done":
            status_cell = "[green]✓ done[/green]"
        elif status == "error":
            status_cell = "[red]✗ error[/red]"
        else:
            status_cell = "[dim]○[/dim]"
        t.add_row(str(i + 1), snippet, spans, status_cell)

    return Panel(
        t,
        title=f"[bold cyan]Streaming[/bold cyan] · [dim]{done}/{len(units)} done[/dim]",
        border_style="blue",
    )


# ── async streaming ───────────────────────────────────────────────────────────


async def stream(ws_url: str, units: list[str]) -> None:
    n = len(units)
    statuses = ["pending"] * n
    span_cells = ["[dim]…[/dim]"] * n
    done = 0

    console.print(f"  [dim]Connecting to {ws_url}[/dim]")

    async with websockets.connect(ws_url) as ws:
        for i, text in enumerate(units):
            await ws.send(json.dumps({"id": str(i), "text": text}))

        with Live(
            result_panel(units, statuses, span_cells, done),
            console=console,
            refresh_per_second=10,
            vertical_overflow="crop",
        ) as live:
            while done < n:
                raw = await ws.recv()
                msg = json.loads(raw)
                idx = int(msg["id"])

                if "error" in msg:
                    statuses[idx] = "error"
                    span_cells[idx] = f"[red]{escape(msg['error'])}[/red]"
                else:
                    spans = msg.get("spans", [])
                    if spans:
                        span_cells[idx] = "  ".join(
                            f"[bold]{escape(s['text'])}[/bold][dim]:{s['label']}[/dim]"
                            for s in spans
                        )
                    else:
                        span_cells[idx] = "[dim](none)[/dim]"
                    statuses[idx] = "done"

                done += 1
                live.update(result_panel(units, statuses, span_cells, done))


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    console.rule("[bold blue]konekaare[/bold blue]")
    console.print()

    # ── fetch annotators ──────────────────────────────────────────────────────
    try:
        all_ann = fetch_annotators(BASE_URL)
    except Exception as e:
        console.print(f"[red]Cannot reach {BASE_URL}: {e}[/red]")
        sys.exit(1)

    available = [a for a in all_ann if a.get("available", True)]
    if not available:
        console.print("[yellow]No annotators are available.[/yellow]")
        sys.exit(1)

    console.print(annotator_table(available))
    console.print()

    choices = [a["name"] for a in available]
    annotator = Prompt.ask("Annotator", choices=choices, default=choices[0])

    # ── text input ────────────────────────────────────────────────────────────
    console.print()
    source = Prompt.ask("Text or path to file").strip()
    path = Path(source)
    if path.is_file():
        text = path.read_text(encoding="utf-8")
        console.print(f"  [dim]Loaded {path.name} ({len(text):,} chars)[/dim]")
    else:
        text = source

    # ── split mode ────────────────────────────────────────────────────────────
    console.print()
    mode = Prompt.ask(
        "Split by [dim](s[/dim]=sentences [dim]· p[/dim]=paragraphs[dim])[/dim]",
        choices=["s", "p"],
        default="s",
        show_choices=False,
        show_default=False,
    )

    if mode == "s":
        units = split_sentences(text)
        mode_label = "sentences"
    else:
        units = split_paragraphs(text)
        mode_label = "paragraphs"

    if not units:
        console.print(f"[yellow]No {mode_label} found in the input.[/yellow]")
        sys.exit(1)

    console.print(f"  [dim]{len(units)} {mode_label}[/dim]")
    console.print()

    # ── stream ────────────────────────────────────────────────────────────────
    ws_url = f"ws://localhost:8000/annotate?annotators={quote(annotator)}"
    try:
        asyncio.run(stream(ws_url, units))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        sys.exit(1)

    console.print()
    console.rule("[green]Done[/green]")


if __name__ == "__main__":
    main()
