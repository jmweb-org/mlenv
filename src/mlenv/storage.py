"""Read and write snapshots as JSON."""

from __future__ import annotations

import json
from pathlib import Path

from mlenv.models import Snapshot


def dumps(snapshot: Snapshot) -> str:
    return json.dumps(snapshot.to_dict(), indent=2, sort_keys=False)


def loads(text: str) -> Snapshot:
    return Snapshot.from_dict(json.loads(text))


def save(snapshot: Snapshot, path: str | Path) -> None:
    Path(path).write_text(dumps(snapshot) + "\n", encoding="utf-8")


def load(path: str | Path) -> Snapshot:
    return loads(Path(path).read_text(encoding="utf-8"))
