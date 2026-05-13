"""Data structures for an environment snapshot.

A snapshot is a small, ordered collection of named sections, each a mapping of
string keys to string values. Keeping everything as plain strings makes
snapshots trivial to serialize, diff, and compare across machines without
worrying about type coercion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SCHEMA_VERSION = 1

# Section names, declared once so collection, diffing and rendering agree.
SECTION_PYTHON = "python"
SECTION_PLATFORM = "platform"
SECTION_PACKAGES = "packages"
SECTION_CUDA = "cuda"
SECTION_GPUS = "gpus"
SECTION_ENV = "env"

SECTION_ORDER = (
    SECTION_PYTHON,
    SECTION_PLATFORM,
    SECTION_PACKAGES,
    SECTION_CUDA,
    SECTION_GPUS,
    SECTION_ENV,
)


@dataclass(frozen=True, slots=True)
class Snapshot:
    """A captured view of the machine learning environment."""

    sections: dict[str, dict[str, str]] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def section(self, name: str) -> dict[str, str]:
        return self.sections.get(name, {})

    def to_dict(self) -> dict:
        ordered: dict[str, dict[str, str]] = {}
        for name in SECTION_ORDER:
            if name in self.sections:
                ordered[name] = dict(self.sections[name])
        # Preserve any unknown sections after the known ones.
        for name, values in self.sections.items():
            if name not in ordered:
                ordered[name] = dict(values)
        return {"schema_version": self.schema_version, "sections": ordered}

    @classmethod
    def from_dict(cls, data: dict) -> Snapshot:
        sections = {
            str(name): {str(k): str(v) for k, v in values.items()}
            for name, values in data.get("sections", {}).items()
        }
        return cls(
            sections=sections,
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
        )
