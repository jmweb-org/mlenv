"""Pure diff between two snapshots.

The diff is generic over sections and keys, with a thin layer on top that
flags changes known to break training reproducibility (a different CUDA
build behind PyTorch, a Python minor-version bump, a change in GPU count or
model, and so on). No I/O, so it is exhaustively unit-tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from mlenv.models import (
    SECTION_CUDA,
    SECTION_GPUS,
    SECTION_PACKAGES,
    SECTION_PYTHON,
    Snapshot,
)


class ChangeKind(str, Enum):
    ADDED = "added"
    REMOVED = "removed"
    CHANGED = "changed"


class Risk(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


_RISK_ORDER = {Risk.HIGH: 0, Risk.MEDIUM: 1, Risk.LOW: 2}


@dataclass(frozen=True, slots=True)
class Change:
    section: str
    key: str
    kind: ChangeKind
    old: str | None
    new: str | None
    risk: Risk

    @property
    def sort_key(self) -> tuple[int, str, str]:
        return (_RISK_ORDER[self.risk], self.section, self.key)


# Packages whose major-version change is very likely to move results.
_SENSITIVE_PACKAGES = {
    "torch",
    "torchvision",
    "tensorflow",
    "jax",
    "jaxlib",
    "numpy",
    "transformers",
    "cuda",
}


def _major(version: str) -> str:
    return version.split(".", 1)[0].strip()


def _minor(version: str) -> str:
    parts = version.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else version


def _classify(section: str, key: str, old: str | None, new: str | None) -> Risk:
    if section == SECTION_CUDA:
        return Risk.HIGH
    if section == SECTION_GPUS:
        # A different device, count, or driver is high risk; ordering is not.
        return Risk.HIGH
    if section == SECTION_PYTHON and key in {"version", "implementation"}:
        if old and new and key == "version" and _minor(old) == _minor(new):
            return Risk.LOW
        return Risk.HIGH
    if section == SECTION_PACKAGES:
        if key in _SENSITIVE_PACKAGES:
            if old and new and _major(old) != _major(new):
                return Risk.HIGH
            return Risk.MEDIUM
        return Risk.LOW
    return Risk.LOW


def diff_section(name: str, old: dict[str, str], new: dict[str, str]) -> list[Change]:
    changes: list[Change] = []
    for key in sorted(set(old) | set(new)):
        before = old.get(key)
        after = new.get(key)
        if before == after:
            continue
        if before is None:
            kind = ChangeKind.ADDED
        elif after is None:
            kind = ChangeKind.REMOVED
        else:
            kind = ChangeKind.CHANGED
        changes.append(
            Change(
                section=name,
                key=key,
                kind=kind,
                old=before,
                new=after,
                risk=_classify(name, key, before, after),
            )
        )
    return changes


def diff_snapshots(old: Snapshot, new: Snapshot) -> list[Change]:
    """Return every change between ``old`` and ``new``, highest risk first."""

    names = list(dict.fromkeys([*old.sections, *new.sections]))
    changes: list[Change] = []
    for name in names:
        changes.extend(diff_section(name, old.section(name), new.section(name)))
    changes.sort(key=lambda c: c.sort_key)
    return changes


def has_high_risk(changes: list[Change]) -> bool:
    return any(c.risk is Risk.HIGH for c in changes)
