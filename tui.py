#!/usr/bin/env python3
"""
tui.py — interactive annotation TUI for annophis_mlhub.

Connects to a running hub (default http://localhost:8000), lets the user
build a pipeline of annotators, validates contracts at each step, runs the
pipeline step-by-step showing intermediate results, and displays the final
document.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.tree import Tree

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
    t.add_column("Requires", style="yellow")
    t.add_column("Produces", style="green")
    t.add_column("Description")
    for i, a in enumerate(annotators, 1):
        contract = a.get("contract", {})
        requires = ", ".join(contract.get("requires", {}).keys()) or "-"
        produces = ", ".join(contract.get("produces", [])) or "-"
        t.add_row(
            str(i),
            a["name"],
            a["annotation_type"],
            requires,
            produces,
            a.get("description", ""),
        )
    return t


def _tui_key_available(path: str, available: set[str]) -> bool:
    """Check whether a (possibly dot-separated) key path is available.

    Matches the full path first, then falls back to the root segment so that
    paths like ``meta.lang`` are satisfied when ``meta`` is in the initial keys.
    """
    if path in available:
        return True
    root = path.split(".")[0]
    return root in available


def validate_pipeline(
    annotators: list[dict], initial_keys: set[str]
) -> list[tuple[str, list[str]]]:
    """Walk the pipeline, checking contracts. Returns list of (name, missing_keys).

    Empty missing_keys means the step is valid.
    """
    available_keys = set(initial_keys)
    issues: list[tuple[str, list[str]]] = []
    for a in annotators:
        contract = a.get("contract", {})
        requires = list(contract.get("requires", {}).keys())
        produces = contract.get("produces", [])
        missing = sorted(
            r for r in requires if not _tui_key_available(r, available_keys)
        )
        issues.append((a["name"], missing))
        available_keys.update(produces)
    return issues


def pipeline_panel(pipeline: list[dict], initial_keys: set[str]) -> Panel:
    """Show the pipeline with contract validation status."""
    issues = validate_pipeline(pipeline, initial_keys)
    t = Table(box=box.SIMPLE, expand=True, pad_edge=False)
    t.add_column("Step", width=4, style="dim")
    t.add_column("Annotator", style="bold cyan")
    t.add_column("Requires", style="yellow")
    t.add_column("Produces", style="green")
    t.add_column("Status", width=14)

    for i, (ann, (_, missing)) in enumerate(zip(pipeline, issues), 1):
        contract = ann.get("contract", {})
        requires = ", ".join(contract.get("requires", {}).keys()) or "-"
        produces = ", ".join(contract.get("produces", [])) or "-"
        if missing:
            status = f"[red]missing: {', '.join(missing)}[/red]"
        else:
            status = "[green]ok[/green]"
        t.add_row(str(i), ann["name"], requires, produces, status)

    all_ok = all(not missing for _, missing in issues)
    title_status = "[green]valid[/green]" if all_ok else "[red]invalid[/red]"
    return Panel(
        t, title=f"[bold]Pipeline[/bold] · {title_status}", border_style="blue"
    )


def span_table(spans: list[dict]) -> Table:
    t = Table(box=box.SIMPLE, pad_edge=False, show_edge=False)
    t.add_column("Text", style="bold")
    t.add_column("Label", style="magenta")
    t.add_column("Start", style="dim", justify="right")
    t.add_column("End", style="dim", justify="right")
    for s in spans:
        t.add_row(escape(s["text"]), s["label"], str(s["start"]), str(s["end"]))
    return t


def document_tree(doc: dict) -> Tree:
    """Build a rich Tree showing all document keys and annotation layers."""
    tree = Tree("[bold]Document[/bold]")
    tree.add(
        f"[cyan]text[/cyan] = [dim]{escape(doc['text'][:80])}{'...' if len(doc['text']) > 80 else ''}[/dim]"
    )

    meta = doc.get("meta", {})
    if meta:
        meta_branch = tree.add("[cyan]meta[/cyan]")
        for k, v in meta.items():
            meta_branch.add(f"{escape(str(k))} = {escape(str(v))}")

    skip = {"text", "meta"}
    for key, value in doc.items():
        if key in skip:
            continue
        if isinstance(value, list) and value and isinstance(value[0], dict):
            branch = tree.add(f"[green]{escape(key)}[/green] ({len(value)} spans)")
            for s in value[:10]:
                branch.add(
                    f"[bold]{escape(s.get('text', ''))}[/bold] "
                    f"[magenta]{s.get('label', '')}[/magenta] "
                    f"[dim]{s.get('start', '')}:{s.get('end', '')}[/dim]"
                )
            if len(value) > 10:
                branch.add(f"[dim]... and {len(value) - 10} more[/dim]")
        else:
            tree.add(f"[green]{escape(key)}[/green] = {escape(str(value))}")

    return tree


def run_pipeline(base_url: str, text: str, meta: dict, pipeline: list[dict]) -> None:
    """Run the pipeline step by step, showing intermediate results."""
    doc = {"text": text, "meta": meta}

    console.print()
    console.rule("[bold blue]Pipeline execution[/bold blue]")

    with httpx.Client(timeout=60) as client:
        for i, ann in enumerate(pipeline, 1):
            name = ann["name"]
            console.print()
            console.print(
                f"  [bold]Step {i}/{len(pipeline)}[/bold]: [cyan]{name}[/cyan]"
            )

            resp = client.post(
                f"{base_url}/annotate",
                json={"document": doc, "annotators": [name]},
            )

            if resp.status_code != 200:
                detail = resp.json().get("detail", resp.text)
                console.print(
                    f"  [red]Error ({resp.status_code}): {escape(str(detail))}[/red]"
                )
                return

            doc = resp.json()

            # show what this step added
            ann_type = ann["annotation_type"]
            new_spans = doc.get(ann_type, [])
            if new_spans:
                console.print(f"  [green]+{ann_type}[/green]: {len(new_spans)} span(s)")
                console.print(span_table(new_spans))
            else:
                console.print(f"  [green]+{ann_type}[/green]: [dim](no spans)[/dim]")

    # final result
    console.print()
    console.rule("[bold green]Final document[/bold green]")
    console.print()
    console.print(document_tree(doc))


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    console.rule("[bold blue]annophis_mlhub[/bold blue]")
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

    by_name = {a["name"]: a for a in available}

    # ── build pipeline ────────────────────────────────────────────────────────
    console.print(
        "[dim]Build a pipeline by adding annotators in order.\n"
        "Type a name (or number) to add, 'done' to finish, 'rm' to remove last.[/dim]"
    )
    console.print()

    pipeline: list[dict] = []
    while True:
        prompt_text = f"Add annotator ({len(pipeline)} in pipeline)"
        raw = Prompt.ask(prompt_text, default="done").strip()

        if raw.lower() == "done":
            if not pipeline:
                console.print(
                    "[yellow]Pipeline is empty — add at least one annotator.[/yellow]"
                )
                continue
            break

        if raw.lower() == "rm":
            if pipeline:
                removed = pipeline.pop()
                console.print(f"  [dim]Removed {removed['name']}[/dim]")
            else:
                console.print("  [dim]Pipeline is already empty.[/dim]")
            continue

        # accept number or name
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(available):
                raw = available[idx]["name"]
            else:
                console.print(
                    f"  [red]Invalid number. Choose 1-{len(available)}.[/red]"
                )
                continue

        if raw not in by_name:
            console.print(f"  [red]Unknown annotator: {raw}[/red]")
            continue

        pipeline.append(by_name[raw])
        console.print(f"  [green]+[/green] {raw}")

    # ── validate pipeline contracts ───────────────────────────────────────────
    console.print()
    initial_keys = {"text", "meta"}
    console.print(pipeline_panel(pipeline, initial_keys))

    issues = validate_pipeline(pipeline, initial_keys)
    has_issues = any(missing for _, missing in issues)

    if has_issues:
        proceed = Prompt.ask(
            "[yellow]Pipeline has contract issues. Proceed anyway?[/yellow]",
            choices=["y", "n"],
            default="n",
        )
        if proceed == "n":
            console.print("[dim]Aborted.[/dim]")
            sys.exit(0)

    # ── text input ────────────────────────────────────────────────────────────
    console.print()
    source = Prompt.ask("Text or path to file").strip()
    path = Path(source)
    if path.is_file():
        text = path.read_text(encoding="utf-8")
        console.print(f"  [dim]Loaded {path.name} ({len(text):,} chars)[/dim]")
    else:
        text = source

    if not text.strip():
        console.print("[yellow]No text provided.[/yellow]")
        sys.exit(1)

    # ── optional meta ─────────────────────────────────────────────────────────
    meta_raw = Prompt.ask("Meta (JSON, or empty)", default="").strip()
    meta: dict = {}
    if meta_raw:
        try:
            import json

            meta = json.loads(meta_raw)
        except Exception:
            console.print("[yellow]Invalid JSON, ignoring meta.[/yellow]")

    # ── run ───────────────────────────────────────────────────────────────────
    try:
        run_pipeline(BASE_URL, text, meta, pipeline)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        sys.exit(1)

    console.print()
    console.rule("[green]Done[/green]")


if __name__ == "__main__":
    main()
