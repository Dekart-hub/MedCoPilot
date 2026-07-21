from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta

import pytest

from dialogue.dialogue import Dialogue
from ehr.publication import (
    EhrPublication,
    InvalidFhirReference,
    PublicationInvalidTransition,
    PublicationOutbox,
    PublicationRequiresVerifiedCorrection,
    PublicationStatus,
    SnapshotIntegrityError,
)
from shared.value_objects import Id
from soap.correction import (
    CorrectionInvalidTransition,
    CorrectionNotEditable,
    CorrectionStatus,
    SoapReportCorrection,
)
from soap.soap import (
    AssessmentClaim,
    IcdCoding,
    SoapClaim,
    SoapNote,
    SoapReport,
    TurnCitation,
)

_NOW = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)
_LATER = _NOW + timedelta(minutes=5)


def _dialogue_and_correction(*, verified: bool = True) -> tuple[Dialogue, SoapReportCorrection]:
    dialogue = Dialogue.start()
    patient = dialogue.add_turn("patient", "My head has hurt for three days.")
    doctor = dialogue.add_turn("doctor", "This is a tension headache.")
    correction = SoapReportCorrection.start(
        SoapReport(
            id=Id.new(),
            notes=[
                SoapNote(
                    id=Id.new(),
                    subjective=[
                        SoapClaim(
                            id=Id.new(),
                            text="Headache for three days.",
                            citations=[TurnCitation(patient.id, "head has hurt")],
                        )
                    ],
                    assessment=[
                        AssessmentClaim(
                            id=Id.new(),
                            text="Tension headache.",
                            citations=[TurnCitation(doctor.id)],
                            icd=IcdCoding(
                                code="G44.2",
                                name="Tension-type headache",
                                classifier_url="https://icd.who.int/browse10/2019/en#/G44.2",
                            ),
                        )
                    ],
                )
            ],
        ),
        created_at=_NOW,
    )
    if verified:
        correction.verify("doctor-1", at=_NOW)
    return dialogue, correction


def _publication() -> tuple[EhrPublication, SoapReportCorrection]:
    dialogue, correction = _dialogue_and_correction()
    publication = EhrPublication.request(
        correction,
        dialogue,
        patient_ref="Patient/p1",
        encounter_ref="Encounter/e1",
        author_ref="Practitioner/d1",
        at=_NOW,
    )
    return publication, correction


def test_publication_captures_full_verified_snapshot_and_stable_hash() -> None:
    publication, correction = _publication()
    snapshot = publication.snapshot

    assert snapshot.source_report_id == str(correction.source_report_id)
    assert snapshot.correction_id == str(correction.id)
    assert snapshot.verified_by == "doctor-1"
    assert [turn.speaker for turn in snapshot.dialogue_turns] == ["patient", "doctor"]
    assert snapshot.notes[0].subjective[0].citations[0].quote == "head has hurt"
    assert snapshot.notes[0].assessment[0].icd is not None
    assert snapshot.notes[0].assessment[0].icd.code == "G44.2"
    publication.verify_snapshot()


def test_snapshot_and_outbox_payload_are_immutable() -> None:
    publication, _ = _publication()
    event = PublicationOutbox.for_publication(publication, at=_NOW)

    with pytest.raises(FrozenInstanceError):
        publication.snapshot.dialogue_turns[0].text = "changed"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        publication.snapshot = publication.snapshot  # type: ignore[misc]
    with pytest.raises(AttributeError):
        event.payload = "{}"  # type: ignore[misc]

    event.verify_payload()
    payload = json.loads(event.payload)
    assert payload["snapshot_hash"] == publication.snapshot_hash


def test_tampered_snapshot_hash_is_rejected_on_construction() -> None:
    publication, _ = _publication()

    with pytest.raises(SnapshotIntegrityError):
        EhrPublication(
            id=publication.id,
            source_report_id=publication.source_report_id,
            correction_id=publication.correction_id,
            patient_ref=publication.patient_ref,
            encounter_ref=publication.encounter_ref,
            author_ref=publication.author_ref,
            snapshot=publication.snapshot,
            snapshot_hash="0" * 64,
            snapshot_schema_version=publication.snapshot_schema_version,
            status=PublicationStatus.PENDING,
            created_at=_NOW,
            updated_at=_NOW,
        )


def test_only_verified_correction_can_be_snapshotted() -> None:
    dialogue, correction = _dialogue_and_correction(verified=False)

    with pytest.raises(PublicationRequiresVerifiedCorrection):
        EhrPublication.request(
            correction,
            dialogue,
            patient_ref="Patient/p1",
            encounter_ref="Encounter/e1",
            author_ref="Practitioner/d1",
            at=_NOW,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("patient_ref", "p1"),
        ("encounter_ref", "Patient/e1"),
        ("author_ref", "Practitioner/bad/id"),
    ],
)
def test_publication_validates_typed_fhir_references(field: str, value: str) -> None:
    dialogue, correction = _dialogue_and_correction()
    references = {
        "patient_ref": "Patient/p1",
        "encounter_ref": "Encounter/e1",
        "author_ref": "Practitioner/d1",
    }
    references[field] = value

    with pytest.raises(InvalidFhirReference):
        EhrPublication.request(correction, dialogue, at=_NOW, **references)


def test_publication_lifecycle_is_terminal_after_delivery() -> None:
    publication, correction = _publication()
    event = PublicationOutbox.for_publication(publication, at=_NOW)
    correction.begin_publication(at=_NOW)

    with pytest.raises(CorrectionNotEditable):
        correction.add_note(at=_LATER)
    with pytest.raises(CorrectionInvalidTransition):
        correction.reopen(at=_LATER)
    with pytest.raises(CorrectionInvalidTransition):
        correction.verify("doctor-2", at=_LATER)

    publication.mark_delivered(remote_reference="Bundle/b1", remote_version="2", at=_LATER)
    event.mark_delivered(at=_LATER)
    correction.mark_published(at=_LATER)

    assert publication.status is PublicationStatus.DELIVERED
    assert correction.status is CorrectionStatus.PUBLISHED
    assert event.attempt_count == 1
    with pytest.raises(PublicationInvalidTransition):
        publication.mark_delivered(remote_reference="Bundle/b2", remote_version=None, at=_LATER)
    with pytest.raises(CorrectionNotEditable):
        correction.add_note(at=_LATER)
    with pytest.raises(CorrectionInvalidTransition):
        correction.reopen(at=_LATER)
    with pytest.raises(CorrectionInvalidTransition):
        correction.verify("doctor-2", at=_LATER)
    with pytest.raises(CorrectionInvalidTransition):
        correction.begin_publication(at=_LATER)
    with pytest.raises(CorrectionInvalidTransition):
        correction.mark_published(at=_LATER)


def test_outbox_failure_keeps_payload_and_schedules_retry() -> None:
    publication, _ = _publication()
    event = PublicationOutbox.for_publication(publication, at=_NOW)
    payload = event.payload

    event.record_failure("network unavailable", next_attempt_at=_LATER)

    assert event.payload == payload
    assert event.attempt_count == 1
    assert event.last_error == "network unavailable"
    assert event.next_attempt_at == _LATER
    assert event.delivered_at is None
