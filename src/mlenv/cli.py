"""Command-line interface for mlenv."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console

from mlenv import __version__
from mlenv.collect import collect_snapshot
from mlenv.diff import diff_snapshots, has_high_risk
from mlenv.render import changes_to_json, render_changes, render_snapshot
from mlenv.storage import dumps, load, save

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Snapshot the machine learning environment and diff two snapshots.",
)
_out = Console()
_err = Console(stderr=True)

EXIT_OK = 0
EXIT_HIGH_RISK = 1
EXIT_BAD_INPUT = 2


def _version_callback(value: bool) -> None:
    if value:
        _out.print(f"mlenv {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    _version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    """mlenv command-line interface."""


@app.command("snapshot")
def snapshot(
    output: Path | None = typer.Option(
        None, "-o", "--output", help="Write the snapshot here (default: stdout)."
    ),
) -> None:
    """Capture the current environment to a JSON snapshot."""

    snap = collect_snapshot()
    if output is None:
        _out.print_json(dumps(snap))
    else:
        save(snap, output)
        _err.print(f"mlenv: wrote {output}")


@app.command("show")
def show() -> None:
    """Pretty-print the current environment."""

    _out.print(render_snapshot(collect_snapshot()))


@app.command("diff")
def diff(
    old: Path = typer.Argument(..., help="Baseline snapshot."),
    new: Path = typer.Argument(..., help="Snapshot to compare against the baseline."),
    as_json: bool = typer.Option(False, "--json", help="Emit changes as JSON."),
    check: bool = typer.Option(
        False, "--check", help="Exit non-zero if any high-risk change is found."
    ),
) -> None:
    """Show what changed between two snapshots, highest risk first."""

    try:
        old_snap = load(old)
        new_snap = load(new)
    except (OSError, json.JSONDecodeError) as exc:
        _err.print(f"mlenv: could not read snapshot: {exc}")
        raise typer.Exit(EXIT_BAD_INPUT) from exc

    changes = diff_snapshots(old_snap, new_snap)
    if as_json:
        _out.print_json(json.dumps(changes_to_json(changes)))
    else:
        _out.print(render_changes(changes))

    if check and has_high_risk(changes):
        raise typer.Exit(EXIT_HIGH_RISK)


def entrypoint() -> None:
    try:
        app()
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print("mlenv: interrupted", file=sys.stderr)
        raise SystemExit(130) from None
