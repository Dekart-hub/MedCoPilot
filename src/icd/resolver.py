"""ICD resolution port (T29): ranked candidates with a status, plus manual-entry
validation against the same catalog.

The resolver grows the T10 top-1 coder contract into an auditable one: every
resolution records its status, the ordered candidate pool the selection came
from, and the classifier version it was made against. Phase 1 always selects
the top candidate when anything matches; the ``AMBIGUOUS`` score/margin gates
arrive with phase 2 without reshaping this contract.

The resolver doubles as the catalog authority (:class:`IcdCatalog`): the same
dictionary that codes extractions validates manually entered codings, so no
code can reach persistence unless the catalog knows it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from soap.soap import IcdCoding, IcdResolution, IcdResolutionStatus

from .dictionary import IcdEntry, classifier_url

_NULL_VERSION = "none"


class IcdCatalog(Protocol):
    """Lookup capability of the classifier catalog, for validating manual input."""

    def entry(self, code: str) -> IcdEntry | None:
        """Return the catalog entry for exactly ``code``, or ``None`` if unknown."""
        ...


class IcdResolver(ABC):
    """Resolves free-text diagnosis wording to ranked ICD candidates."""

    @abstractmethod
    def resolve(self, diagnosis_text: str) -> IcdResolution:
        """Resolve ``diagnosis_text`` against the catalog.

        Never raises on unmatchable input: an empty or out-of-vocabulary text
        yields a ``NOT_FOUND`` resolution with an empty candidate pool.
        """
        ...

    @abstractmethod
    def entry(self, code: str) -> IcdEntry | None:
        """Return the catalog entry for exactly ``code``, or ``None`` if unknown."""
        ...


class NullIcdResolver(IcdResolver):
    """No-op resolver: everything is ``NOT_FOUND``. Mirrors :class:`NullIcdCoder`."""

    def resolve(self, diagnosis_text: str) -> IcdResolution:
        return IcdResolution(
            status=IcdResolutionStatus.NOT_FOUND,
            selected=None,
            candidates=(),
            classifier_version=_NULL_VERSION,
        )

    def entry(self, code: str) -> IcdEntry | None:
        return None


class UnknownIcdCode(Exception):
    """A manually entered ICD code is not in the classifier catalog."""

    def __init__(self, code: str) -> None:
        super().__init__(f"ICD code {code!r} is not in the classifier catalog")
        self.code = code


class InactiveIcdCode(Exception):
    """A manually entered ICD code is retired in the current catalog release."""

    def __init__(self, code: str) -> None:
        super().__init__(f"ICD code {code!r} is inactive in the current catalog release")
        self.code = code


class IcdTitleMismatch(Exception):
    """A manually entered title does not match the catalog title for its code."""

    def __init__(self, code: str, title: str, canonical: str) -> None:
        super().__init__(
            f"title {title!r} does not match the catalog title {canonical!r} for ICD code {code}"
        )
        self.code = code
        self.title = title
        self.canonical = canonical


def _normalized(text: str) -> str:
    return " ".join(text.split()).casefold()


def validate_manual_icd(coding: IcdCoding, catalog: IcdCatalog) -> IcdCoding:
    """Validate a manually entered coding and return its canonical form.

    The code must exist in the catalog and be active; the title must match the
    canonical name up to case and whitespace. The returned coding always
    carries the canonical name and the server-derived classifier URL — a
    client-provided variant of either never reaches persistence.
    """
    entry = catalog.entry(coding.code)
    if entry is None:
        raise UnknownIcdCode(coding.code)
    if not entry.active:
        raise InactiveIcdCode(coding.code)
    if _normalized(coding.name) != _normalized(entry.name):
        raise IcdTitleMismatch(coding.code, coding.name, entry.name)
    return IcdCoding(code=entry.code, name=entry.name, classifier_url=classifier_url(entry.code))
