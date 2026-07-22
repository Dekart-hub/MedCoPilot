"""BM25 ICD resolver: the ranked, auditable successor of the T10 top-1 coder.

Same zero-dependency Okapi BM25 as the coder, now returning the top-k pool:
candidates are de-duplicated by code (a dictionary may carry synonym rows for
one code), ordered deterministically by ``(score desc, code asc)`` so equal
scores never flap between runs, and stamped with the dictionary version.

Phase 1 semantics: any lexical match resolves to the top candidate; no score
or margin thresholds yet. Inactive dictionary entries are never indexed — they
exist only for catalog lookups against historical data.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

from soap.soap import IcdCandidate, IcdCoding, IcdResolution, IcdResolutionStatus

from .dictionary import (
    IcdEntry,
    bundled_dictionary_version,
    classifier_url,
    dictionary_version,
    load_bundled_dictionary,
    load_dictionary,
)
from .resolver import IcdResolver
from .tokenizer import tokenize

_DEFAULT_TOP_K = 10
# Raw BM25 hits fetched before code-level dedup: synonym rows for one code may
# crowd the head of the ranking, so the pre-dedup pool must be deeper than k.
_POOL_FACTOR = 5


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

    def top(self, query: list[str], limit: int) -> list[tuple[int, float]]:
        """Return up to ``limit`` ``(doc_id, score)`` pairs, best first.

        Ordering is deterministic: score descending, then doc id ascending, so
        ties never reorder between runs.
        """
        if not self._count or not query:
            return []
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
        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        return ranked[:limit]


class Bm25IcdResolver(IcdResolver):
    """Ranked ICD resolution via BM25 over the dictionary titles."""

    def __init__(
        self, entries: list[IcdEntry], *, version: str, top_k: int = _DEFAULT_TOP_K
    ) -> None:
        # First occurrence of a code is its canonical record; later rows with
        # the same code are synonym spellings that only feed retrieval.
        self._canonical: dict[str, IcdEntry] = {}
        for entry in entries:
            self._canonical.setdefault(entry.code, entry)
        self._indexed = [entry for entry in entries if entry.active]
        self._index = _Bm25Index([tokenize(entry.name) for entry in self._indexed])
        self._version = version
        self._top_k = top_k

    def resolve(self, diagnosis_text: str) -> IcdResolution:
        hits = self._index.top(tokenize(diagnosis_text), self._top_k * _POOL_FACTOR)
        best_by_code: dict[str, float] = {}
        for doc_id, score in hits:
            code = self._indexed[doc_id].code
            if code not in best_by_code:
                best_by_code[code] = score
        ranked = sorted(best_by_code.items(), key=lambda pair: (-pair[1], pair[0]))
        candidates = tuple(
            IcdCandidate(
                code=code,
                name=self._canonical[code].name,
                rank=rank,
                bm25_score=score,
            )
            for rank, (code, score) in enumerate(ranked[: self._top_k], start=1)
        )
        if not candidates:
            return IcdResolution(
                status=IcdResolutionStatus.NOT_FOUND,
                selected=None,
                candidates=(),
                classifier_version=self._version,
            )
        top = candidates[0]
        return IcdResolution(
            status=IcdResolutionStatus.RESOLVED,
            selected=IcdCoding(
                code=top.code, name=top.name, classifier_url=classifier_url(top.code)
            ),
            candidates=candidates,
            classifier_version=self._version,
        )

    def entry(self, code: str) -> IcdEntry | None:
        return self._canonical.get(code)

    @classmethod
    def from_bundled(cls, *, top_k: int = _DEFAULT_TOP_K) -> Bm25IcdResolver:
        """Build a resolver over the curated ICD-10 sample bundled with the package."""
        return cls(load_bundled_dictionary(), version=bundled_dictionary_version(), top_k=top_k)

    @classmethod
    def from_json(cls, path: Path, *, top_k: int = _DEFAULT_TOP_K) -> Bm25IcdResolver:
        """Build a resolver over the dictionary at ``path``, versioned by its sidecar."""
        return cls(load_dictionary(path), version=dictionary_version(path), top_k=top_k)
