"""The ICD-10 classifier dictionary: ``(code, name)`` entries plus a reference URL.

The dictionary is a flat list of ICD-10 codes with their canonical titles. A
small curated sample of common codes ships in the package
(:data:`_BUNDLED_PATH`) so coding works offline and tests are deterministic;
``scripts/fetch_icd.py`` can materialise a fuller dictionary from a public
source for real use.

Entries may carry ``"active": false`` for codes retired from the current
release: they stay resolvable for catalog lookups (historical data references
them) but the resolver never offers them as candidates.

A dictionary file may be accompanied by a ``<name>.meta.json`` sidecar whose
``version`` names the release it was materialised from; that string becomes the
``classifier_version`` stamped on every resolution.

``classifier_url`` points at the WHO ICD-10 (2019) online browser, the stable
public reference for a code: ``https://icd.who.int/browse10/2019/en#/<code>``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_BUNDLED_PATH = Path(__file__).parent / "icd10_sample.json"
_BUNDLED_VERSION = "icd10-2019-sample"
_CLASSIFIER_URL = "https://icd.who.int/browse10/2019/en#/{code}"


@dataclass(frozen=True, slots=True)
class IcdEntry:
    """One classifier record: a code, its canonical title and its lifecycle flag."""

    code: str
    name: str
    active: bool = True


def classifier_url(code: str) -> str:
    """Return the WHO ICD-10 browser reference for ``code``."""
    return _CLASSIFIER_URL.format(code=code)


def load_dictionary(path: Path) -> list[IcdEntry]:
    """Load ``(code, name[, active])`` entries from a JSON array file."""
    records: list[Any] = json.loads(path.read_text(encoding="utf-8"))
    return [
        IcdEntry(
            code=record["code"],
            name=record["name"],
            active=bool(record.get("active", True)),
        )
        for record in records
    ]


def load_bundled_dictionary() -> list[IcdEntry]:
    """Load the curated ICD-10 sample bundled with the package."""
    return load_dictionary(_BUNDLED_PATH)


def bundled_dictionary_version() -> str:
    """Version stamp of the bundled sample dictionary."""
    return _BUNDLED_VERSION


def dictionary_version(path: Path) -> str:
    """Version of the dictionary at ``path``: its meta sidecar, else the file stem."""
    meta_path = path.with_suffix(".meta.json")
    if meta_path.exists():
        meta: Any = json.loads(meta_path.read_text(encoding="utf-8"))
        version = meta.get("version") if isinstance(meta, dict) else None
        if isinstance(version, str) and version:
            return version
    return path.stem
