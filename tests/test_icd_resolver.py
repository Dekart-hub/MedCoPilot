"""Unit tests for the BM25 ICD resolver (T29) over a hand-written fixture catalog.

The fixture — not the bundled production sample — pins the behaviours the
resolver contract promises: ranked deduplicated candidates, deterministic
ties, synonym rows, inactive codes, abbreviation expansion and manual-entry
validation. No network, no database.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from icd.bm25_resolver import Bm25IcdResolver
from icd.dictionary import classifier_url
from icd.resolver import (
    IcdTitleMismatch,
    InactiveIcdCode,
    NullIcdResolver,
    UnknownIcdCode,
    validate_manual_icd,
)
from soap.soap import IcdCoding, IcdResolutionStatus

_FIXTURE = Path(__file__).parent / "fixtures" / "icd_catalog.json"


@pytest.fixture(scope="module")
def resolver() -> Bm25IcdResolver:
    return Bm25IcdResolver.from_json(_FIXTURE)


def test_obvious_diagnosis_resolves_to_its_code(resolver: Bm25IcdResolver) -> None:
    resolution = resolver.resolve("type 2 diabetes mellitus without complications")
    assert resolution.status is IcdResolutionStatus.RESOLVED
    assert resolution.selected is not None
    assert resolution.selected.code == "E11.9"
    assert resolution.selected.classifier_url == classifier_url("E11.9")


def test_selected_is_the_top_candidate(resolver: Bm25IcdResolver) -> None:
    resolution = resolver.resolve("acute bronchitis")
    assert resolution.selected is not None
    assert resolution.candidates[0].code == resolution.selected.code


def test_candidates_are_ranked_from_one_and_ordered(resolver: Bm25IcdResolver) -> None:
    resolution = resolver.resolve("type 2 diabetes mellitus")
    ranks = [candidate.rank for candidate in resolution.candidates]
    assert ranks == list(range(1, len(ranks) + 1))
    scores = [candidate.bm25_score for candidate in resolution.candidates]
    assert scores == sorted(scores, reverse=True)


def test_empty_text_is_not_found(resolver: Bm25IcdResolver) -> None:
    resolution = resolver.resolve("")
    assert resolution.status is IcdResolutionStatus.NOT_FOUND
    assert resolution.selected is None
    assert resolution.candidates == ()


def test_out_of_vocabulary_text_is_not_found(resolver: Bm25IcdResolver) -> None:
    resolution = resolver.resolve("qwerty zxcvbn floooby")
    assert resolution.status is IcdResolutionStatus.NOT_FOUND
    assert resolution.selected is None


def test_synonym_rows_are_deduplicated_to_one_candidate(resolver: Bm25IcdResolver) -> None:
    # I10 appears twice in the fixture (canonical title + synonym); the pool
    # must offer the code once, under its canonical title.
    resolution = resolver.resolve("high blood pressure hypertension")
    i10 = [candidate for candidate in resolution.candidates if candidate.code == "I10"]
    assert len(i10) == 1
    assert i10[0].name == "Essential (primary) hypertension"


def test_synonym_wording_still_finds_the_code(resolver: Bm25IcdResolver) -> None:
    resolution = resolver.resolve("high blood pressure")
    assert resolution.selected is not None
    assert resolution.selected.code == "I10"
    assert resolution.selected.name == "Essential (primary) hypertension"


def test_tied_scores_break_deterministically_by_code(resolver: Bm25IcdResolver) -> None:
    # "Low back pain" and "Low back pain, unspecified" tokenize identically
    # ("unspecified" is a stopword) ⇒ identical BM25 scores; the lower code
    # must win, and repeated runs must agree.
    first = resolver.resolve("low back pain")
    second = resolver.resolve("low back pain")
    assert first.selected is not None
    assert first.selected.code == "M54.5"
    assert [c.code for c in first.candidates[:2]] == ["M54.5", "M54.50"]
    assert first == second


def test_inactive_codes_are_never_candidates(resolver: Bm25IcdResolver) -> None:
    resolution = resolver.resolve("bronchopneumonia")
    assert all(candidate.code != "J18.0" for candidate in resolution.candidates)


def test_clinical_abbreviation_is_expanded(resolver: Bm25IcdResolver) -> None:
    resolution = resolver.resolve("t2dm")
    assert resolution.selected is not None
    assert resolution.selected.code.startswith("E11")


def test_top_k_caps_the_pool() -> None:
    small = Bm25IcdResolver.from_json(_FIXTURE, top_k=2)
    resolution = small.resolve("type 2 diabetes mellitus")
    assert len(resolution.candidates) == 2


def test_classifier_version_comes_from_the_meta_sidecar(resolver: Bm25IcdResolver) -> None:
    assert resolver.resolve("pneumonia").classifier_version == "test-catalog-1"
    assert resolver.resolve("").classifier_version == "test-catalog-1"


def test_null_resolver_finds_nothing() -> None:
    resolution = NullIcdResolver().resolve("pneumonia")
    assert resolution.status is IcdResolutionStatus.NOT_FOUND
    assert resolution.selected is None


# --------------------------------------------------------------------------- #
# Manual-entry validation against the same catalog.
# --------------------------------------------------------------------------- #


def _manual(code: str, name: str) -> IcdCoding:
    return IcdCoding(code=code, name=name, classifier_url="https://client-supplied.example/x")


def test_valid_manual_coding_is_canonicalised(resolver: Bm25IcdResolver) -> None:
    coding = validate_manual_icd(
        _manual("j18.9".upper(), "pneumonia, unspecified ORGANISM"), resolver
    )
    assert coding.code == "J18.9"
    assert coding.name == "Pneumonia, unspecified organism"
    assert coding.classifier_url == classifier_url("J18.9")


def test_unknown_code_is_rejected(resolver: Bm25IcdResolver) -> None:
    with pytest.raises(UnknownIcdCode):
        validate_manual_icd(_manual("Z99.99", "Made-up condition"), resolver)


def test_inactive_code_is_rejected(resolver: Bm25IcdResolver) -> None:
    with pytest.raises(InactiveIcdCode):
        validate_manual_icd(_manual("J18.0", "Bronchopneumonia, unspecified organism"), resolver)


def test_title_mismatch_is_rejected(resolver: Bm25IcdResolver) -> None:
    with pytest.raises(IcdTitleMismatch):
        validate_manual_icd(_manual("J18.9", "Lobar pneumonia"), resolver)
