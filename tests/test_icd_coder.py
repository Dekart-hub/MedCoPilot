"""Unit tests for the BM25 ICD-10 coder over the bundled dictionary (no network)."""

from __future__ import annotations

import pytest

from icd.bm25_coder import Bm25IcdCoder
from icd.dictionary import classifier_url


@pytest.fixture(scope="module")
def coder() -> Bm25IcdCoder:
    return Bm25IcdCoder.from_bundled()


@pytest.mark.parametrize(
    ("diagnosis", "code_prefix"),
    [
        ("community-acquired pneumonia", "J18"),
        ("essential hypertension", "I10"),
        ("type 2 diabetes mellitus", "E11"),
        ("tension headache", "G44"),
        ("acute bronchitis", "J20"),
        ("low back pain", "M54"),
    ],
)
def test_diagnosis_maps_to_expected_code_family(
    coder: Bm25IcdCoder, diagnosis: str, code_prefix: str
) -> None:
    coding = coder.code(diagnosis)
    assert coding is not None
    assert coding.code.startswith(code_prefix)


def test_out_of_vocabulary_diagnosis_is_left_uncoded(coder: Bm25IcdCoder) -> None:
    assert coder.code("qwerty zxcvbn floooby") is None


def test_empty_diagnosis_is_left_uncoded(coder: Bm25IcdCoder) -> None:
    assert coder.code("") is None


def test_coding_carries_a_name_and_classifier_reference_url(coder: Bm25IcdCoder) -> None:
    coding = coder.code("pneumonia")
    assert coding is not None
    assert coding.name
    assert coding.classifier_url == classifier_url(coding.code)
    assert coding.classifier_url.startswith("https://icd.who.int/")
