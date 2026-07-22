"""The extractor attaches an ICD coding to assessment claims when a coder is wired,
and the full resolution (T29) when a resolver is wired."""

from __future__ import annotations

import asyncio
from typing import Any

from dialogue.dialogue import Dialogue
from icd.coder import IcdCoder
from icd.dictionary import IcdEntry
from icd.resolver import IcdResolver
from soap.llm_client import LlmClient
from soap.llm_extractor import LlmSoapExtractor
from soap.scorer import NullConfidenceScorer
from soap.soap import (
    IcdCandidate,
    IcdCoding,
    IcdResolution,
    IcdResolutionStatus,
    SoapReport,
)

_STUB_CODING = IcdCoding(
    code="I10",
    name="Essential (primary) hypertension",
    classifier_url="https://icd.who.int/browse10/2019/en#/I10",
)

_STUB_RESOLUTION = IcdResolution(
    status=IcdResolutionStatus.RESOLVED,
    selected=_STUB_CODING,
    candidates=(
        IcdCandidate(code="I10", name=_STUB_CODING.name, rank=1, bm25_score=7.5),
        IcdCandidate(
            code="I15.9", name="Secondary hypertension, unspecified", rank=2, bm25_score=3.1
        ),
    ),
    classifier_version="test-catalog-1",
)

_PAYLOAD: dict[str, Any] = {
    "subjective": [{"text": "Headache for three days.", "turn_index": 1, "quote": None}],
    "objective": [],
    "assessment": [{"text": "Essential hypertension.", "turn_index": 2, "quote": None}],
    "plan": [],
}


class _FakeLlmClient(LlmClient):
    """Canned planner and note payloads — no network."""

    async def complete_json(
        self, *, instructions: str, prompt: str, schema: dict[str, Any]
    ) -> dict[str, Any]:
        if "problems" in schema.get("properties", {}):
            return {"problems": [{"title": "Hypertension"}]}
        return _PAYLOAD


class _StubIcdCoder(IcdCoder):
    def code(self, diagnosis_text: str) -> IcdCoding | None:
        return _STUB_CODING


class _StubIcdResolver(IcdResolver):
    def resolve(self, diagnosis_text: str) -> IcdResolution:
        return _STUB_RESOLUTION

    def entry(self, code: str) -> IcdEntry | None:
        return None


def _dialogue() -> Dialogue:
    dialogue = Dialogue.start()
    dialogue.add_turn("patient", "I've had a headache for three days.")
    dialogue.add_turn("doctor", "Blood pressure is 140 over 90.")
    return dialogue


def _extract(coder: IcdCoder | None, resolver: IcdResolver | None = None) -> SoapReport:
    extractor = LlmSoapExtractor(
        _FakeLlmClient(), NullConfidenceScorer(), coder=coder, resolver=resolver
    )
    return asyncio.run(extractor.extract(_dialogue(), ""))


def test_assessment_claims_are_coded_when_a_coder_is_wired() -> None:
    assessments = _extract(_StubIcdCoder()).notes[0].assessment

    assert assessments  # the note carries an assessment claim to code
    assert all(claim.icd == _STUB_CODING for claim in assessments)


def test_assessment_claims_stay_uncoded_without_a_coder() -> None:
    assessments = _extract(None).notes[0].assessment

    assert assessments
    assert all(claim.icd is None for claim in assessments)
    assert all(claim.icd_resolution is None for claim in assessments)


def test_resolver_fills_resolution_and_mirrors_selected_into_icd() -> None:
    assessments = _extract(None, resolver=_StubIcdResolver()).notes[0].assessment

    assert assessments
    for claim in assessments:
        assert claim.icd_resolution == _STUB_RESOLUTION
        assert claim.icd == _STUB_RESOLUTION.selected


def test_resolver_wins_over_coder_when_both_are_wired() -> None:
    assessments = _extract(_StubIcdCoder(), resolver=_StubIcdResolver()).notes[0].assessment

    assert assessments
    assert all(claim.icd_resolution == _STUB_RESOLUTION for claim in assessments)
