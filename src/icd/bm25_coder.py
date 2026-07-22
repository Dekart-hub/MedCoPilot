"""Baseline ICD-10 coder (T10): the top-1 view over the BM25 resolver.

The coder predates the T29 resolver and keeps its narrow contract — one best
:class:`~soap.soap.IcdCoding` or ``None`` — for callers that need nothing
else. Since T29 it is a thin wrapper over :class:`Bm25IcdResolver`, so both
ports rank with the same index, tokenizer and dictionary.
"""

from __future__ import annotations

from pathlib import Path

from soap.soap import IcdCoding

from .bm25_resolver import Bm25IcdResolver
from .coder import IcdCoder
from .dictionary import (
    IcdEntry,
    bundled_dictionary_version,
    dictionary_version,
    load_bundled_dictionary,
    load_dictionary,
)

_UNVERSIONED = "unversioned"


class Bm25IcdCoder(IcdCoder):
    """Top-1 ICD-10 coding via BM25 over the dictionary titles."""

    def __init__(self, entries: list[IcdEntry], *, version: str = _UNVERSIONED) -> None:
        self._resolver = Bm25IcdResolver(entries, version=version)

    def code(self, diagnosis_text: str) -> IcdCoding | None:
        return self._resolver.resolve(diagnosis_text).selected

    @classmethod
    def from_bundled(cls) -> Bm25IcdCoder:
        """Build a coder over the curated ICD-10 sample bundled with the package."""
        return cls(load_bundled_dictionary(), version=bundled_dictionary_version())

    @classmethod
    def from_json(cls, path: Path) -> Bm25IcdCoder:
        """Build a coder over a ``(code, name)`` JSON dictionary at ``path``."""
        return cls(load_dictionary(path), version=dictionary_version(path))
