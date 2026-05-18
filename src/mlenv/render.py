"""Render snapshots and diffs for the terminal and as JSON."""

from __future__ import annotations

from rich.console import Group
from rich.table import Table
from rich.text import Text

from mlenv.diff import Change, Risk
from mlenv.models import SECTION_ORDER, Snapshot

_RISK_STYLE = {Risk.HIGH: "bold red", Risk.MEDIUM: "yellow", Risk.LOW: "dim"}
_RISK_LABEL = {Risk.HIGH: "high", Risk.MEDIUM: "med", Risk.LOW: "low"}


def render_snapshot(snapshot: Snapshot) -> Group:
    blocks: list[Table] = []
    names = list(SECTION_ORDER) + [n for n in snapshot.sections if n not in SECTION_ORDER]
    for name in names:
        values = snapshot.section(name)
        if not values:
            continue
        table = Table(title=name, title_justify="left", box=None, pad_edge=False)
        table.add_column("key", style="cyan")
        table.add_column("value")
        for key, value in values.items():
            table.add_row(key, value)
        blocks.append(table)
    return Group(*blocks)


def changes_to_json(changes: list[Change]) -> list[dict]:
    return [
        {
            "section": c.section,
            "key": c.key,
            "kind": c.kind.value,
            "old": c.old,
            "new": c.new,
            "risk": c.risk.value,
        }
        for c in changes
    ]


def render_changes(changes: list[Change]) -> Group:
    if not changes:
        return Group(Text("environments match", style="green"))
    table = Table(box=None, pad_edge=False)
    table.add_column("risk")
    table.add_column("section", style="cyan")
    table.add_column("key")
    table.add_column("change")
    for c in changes:
        style = _RISK_STYLE[c.risk]
        table.add_row(
            Text(_RISK_LABEL[c.risk], style=style),
            c.section,
            c.key,
            _format_change(c),
        )
    return Group(table)


def _format_change(change: Change) -> str:
    if change.kind.value == "added":
        return f"+ {change.new}"
    if change.kind.value == "removed":
        return f"- {change.old}"
    return f"{change.old} -> {change.new}"
