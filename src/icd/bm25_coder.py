"""Baseline ICD-10 coder: Okapi BM25 over the classifier dictionary.

Zero external dependencies: a few dozen to a few thousand code titles fit in
memory, so a hand-rolled inverted index with Okapi BM25 ranks matches in
microseconds — no ANN or search server needed. The coder tokenises each code's
title once at construction, then scores a diagnosis query against them and
returns the top-1 entry as an :class:`~soap.soap.IcdCoding`.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

from soap.soap import IcdCoding

from .coder import IcdCoder
from .dictionary import (
    IcdEntry,
    classifier_url,
    load_bundled_dictionary,
    load_dictionary,
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Function words that carry no clinical signal; dropped so the query and titles
# match on content terms. IDF already down-weights them, but pruning keeps the
# ranking crisp on short titles.
_STOPWORDS: frozenset[str] = frozenset(
    {"and", "or", "of", "the", "with", "without", "in", "to", "due", "unspecified", "not"}
)


def _tokenize(text: str) -> list[str]:
    return [token for token in _TOKEN_RE.findall(text.lower()) if token not in _STOPWORDS]


class _Bm25Index:
    """Okapi BM25 over pre-tokenised documents."""

    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._doc_len = [len(doc) for doc in documents]
        self._count = len(documents)
        self._avgdl = sum(self._doc_len) / self._count if self._count else 0.0

        self._postings: dict[str, list[tuple[int, int]]] = {}
        for doc_id, doc in enumerate(documents):
            for term, freq in Counter(doc).items():
                self._postings.setdefault(term, []).append((doc_id, freq))

        self._idf: dict[str, float] = {}
        for term, postings in self._postings.items():
            df = len(postings)
            self._idf[term] = math.log(1 + (self._count - df + 0.5) / (df + 0.5))

    def best(self, query: list[str]) -> int | None:
        """Return the id of the highest-scoring document, or ``None`` if none match."""
        if not self._count or not query:
            return None
        scores: dict[int, float] = {}
        for term in set(query):
            postings = self._postings.get(term)
            if postings is None:
                continue
            idf = self._idf[term]
            for doc_id, freq in postings:
                norm = 1 - self._b + self._b * self._doc_len[doc_id] / self._avgdl
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * freq * (self._k1 + 1) / (
                    freq + self._k1 * norm
                )
        if not scores:
            return None
        return max(scores, key=lambda doc_id: scores[doc_id])


class Bm25IcdCoder(IcdCoder):
    """Top-1 ICD-10 coding via BM25 over the dictionary titles."""

    def __init__(self, entries: list[IcdEntry]) -> None:
        self._entries = entries
        self._index = _Bm25Index([_tokenize(entry.name) for entry in entries])

    def code(self, diagnosis_text: str) -> IcdCoding | None:
        doc_id = self._index.best(_tokenize(diagnosis_text))
        if doc_id is None:
            return None
        entry = self._entries[doc_id]
        return IcdCoding(
            code=entry.code,
            name=entry.name,
            classifier_url=classifier_url(entry.code),
        )

    @classmethod
    def from_bundled(cls) -> Bm25IcdCoder:
        """Build a coder over the curated ICD-10 sample bundled with the package."""
        return cls(load_bundled_dictionary())

    @classmethod
    def from_json(cls, path: Path) -> Bm25IcdCoder:
        """Build a coder over a ``(code, name)`` JSON dictionary at ``path``."""
        return cls(load_dictionary(path))
