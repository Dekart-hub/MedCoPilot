from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from dialogue import DialogueTurnId
from shared.value_objects import Id
from soap.coding import LexicalDiagnosisNormalizer, MkbIndex
from soap.coding.preprocess import normalize
from soap.soap import SoapClaim, SoapEvidence, SoapNote

# --- мини-каталог для юнит-тестов (без файлов НСИ) ----------------------- #

_VOL1 = [
    {"ID": "1", "MKB_CODE": "L03", "MKB_NAME": "Флегмона", "ID_PARENT": "0", "ACTUAL": "1"},
    {"ID": "2", "MKB_CODE": "L03.8", "MKB_NAME": "Флегмона других локализаций", "ID_PARENT": "1", "ACTUAL": "1"},
    {"ID": "3", "MKB_CODE": "L03.9", "MKB_NAME": "Флегмона неуточненная", "ID_PARENT": "1", "ACTUAL": "1"},
    {"ID": "4", "MKB_CODE": "E11", "MKB_NAME": "Сахарный диабет 2 типа", "ID_PARENT": "0", "ACTUAL": "1"},
]

_VOL3 = [
    {"S_NAME": "Флегмона шеи", "ICD-10": "L03.8"},
    {"S_NAME": "Флегмона головы", "ICD-10": "L03.8"},
    {"S_NAME": "Флегмона", "ICD-10": "L03.9"},
    {"S_NAME": "Сахарный диабет 2 типа", "ICD-10": "E11"},
]


def _index() -> MkbIndex:
    return MkbIndex.from_records(_VOL1, _VOL3)


def _note(assessment_text: str) -> SoapNote:
    def claim(text: str) -> SoapClaim:
        turn_id: DialogueTurnId = Id.new()
        return SoapClaim(
            id=Id.new(),
            claim=text,
            evidence=SoapEvidence(text=text, turn_id=turn_id),
        )

    return SoapNote(
        id=Id.new(),
        subjective=claim("болит шея"),
        objective=claim("отек, гиперемия"),
        assessment=claim(assessment_text),
        plan=claim("антибиотик"),
    )


def _normalize(text: str):
    normalizer = LexicalDiagnosisNormalizer(_index())
    return asyncio.run(normalizer.normalize(_note(text)))


# --- препроцессинг ------------------------------------------------------- #


def test_word_order_is_irrelevant():
    # Том 3 пишет «Флегмона головы», запрос — «головы флегмона».
    assert set(normalize("головы флегмона")) == set(normalize("Флегмона головы"))


def test_morphology_collapses_to_same_stem():
    assert normalize("флегмоны") == normalize("флегмона")


def test_abbreviation_is_expanded():
    assert normalize("сд") == normalize("сахарный диабет")


# --- матчинг ------------------------------------------------------------- #


def test_exact_formulation_resolves_to_code():
    coding = _normalize("Флегмона шеи")
    assert coding.best is not None
    assert coding.best.code == "L03.8"


def test_morphological_query_still_matches():
    coding = _normalize("флегмоны шеи")
    assert coding.best is not None
    assert coding.best.code == "L03.8"


def test_abbreviated_query_matches_via_expansion():
    coding = _normalize("СД 2 типа")
    assert coding.best is not None
    assert coding.best.code == "E11"


def test_coding_is_anchored_to_assessment_claim():
    note = _note("Флегмона шеи")
    normalizer = LexicalDiagnosisNormalizer(_index())
    coding = asyncio.run(normalizer.normalize(note))
    assert coding.soap_claim_id == note.assessment.id


def test_best_score_is_normalised_to_one():
    coding = _normalize("Флегмона шеи")
    assert coding.best is not None
    assert coding.best.score.score == 1.0


def test_no_match_gives_empty_candidates():
    coding = _normalize("перелом лучевой кости")
    assert coding.candidates == []


def test_title_comes_from_vol1_canonical_name():
    coding = _normalize("Флегмона шеи")
    assert coding.best is not None
    # Каноническое имя L03.8 из Тома 1, а не формулировка Тома 3.
    assert coding.best.title == "Флегмона других локализаций"
    assert coding.best.matched_formulation == "Флегмона шеи"


def test_coding_carries_classifier_reference():
    coding = _normalize("Флегмона шеи")
    assert coding.best is not None
    # Provenance: чем закодировали (OID МКБ-10 НСИ, Том 1).
    assert coding.best.classifier.system == "1.2.643.5.1.13.13.11.1005"
    assert coding.best.classifier.index_oid == "1.2.643.5.1.13.13.11.1489"


# --- иерархия (для будущего back-off) ------------------------------------ #


def test_parent_chain_walks_to_root():
    assert _index().parent_chain("L03.8") == ["L03"]


def test_children_of_lists_refinements():
    assert [c.code for c in _index().children_of("L03")] == ["L03.8", "L03.9"]
    assert _index().children_of("L03.8") == []
