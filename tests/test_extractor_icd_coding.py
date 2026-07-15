"""The extractor attaches an ICD coding to assessment claims when a coder is wired."""

from __future__ import annotations

import asyncio
from typing import Any

from dialogue.dialogue import Dialogue
from icd.coder import IcdCoder
from soap.llm_client import LlmClient
from soap.llm_extractor import LlmSoapExtractor
from soap.scorer import NullConfidenceScorer
from soap.soap import IcdCoding, SoapReport

_STUB_CODING = IcdCoding(
    code="I10",
    name="Essential (primary) hypertension",
    classifier_url="https://icd.who.int/browse10/2019/en#/I10",
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


def _dialogue() -> Dialogue:
    dialogue = Dialogue.start()
    dialogue.add_turn("patient", "I've had a headache for three days.")
    dialogue.add_turn("doctor", "Blood pressure is 140 over 90.")
    return dialogue


def _extract(coder: IcdCoder | None) -> SoapReport:
    extractor = LlmSoapExtractor(_FakeLlmClient(), NullConfidenceScorer(), coder=coder)
    return asyncio.run(extractor.extract(_dialogue(), ""))


def test_assessment_claims_are_coded_when_a_coder_is_wired() -> None:
    assessments = _extract(_StubIcdCoder()).notes[0].assessment

    assert assessments  # the note carries an assessment claim to code
    assert all(claim.icd == _STUB_CODING for claim in assessments)


def test_assessment_claims_stay_uncoded_without_a_coder() -> None:
    assessments = _extract(None).notes[0].assessment

    assert assessments
    assert all(claim.icd is None for claim in assessments)
