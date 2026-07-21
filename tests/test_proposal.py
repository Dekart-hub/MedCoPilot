"""Unit tests for the LLM editor-proposal domain: the decision state machine.

The invariants under test: forming a proposal never touches the correction, each
operation is decided on its own (mixed proposals supported), re-deciding is
idempotent while the opposite verdict conflicts, a target that has since changed
cannot be overwritten while edits to other notes stay decidable, ICD has no
channel through a proposal, ``reject_pending`` sweeps the active proposal, and a
session admits at most one active proposal, only for its own correction.
"""

from __future__ import annotations

from dataclasses import fields
from datetime import UTC, datetime

import pytest

from shared.value_objects import Id
from soap.correction import SoapReportCorrection
from soap.proposal import (
    ActiveProposalExists,
    AddNoteOperation,
    ConflictingDecision,
    CorrectionEditorSession,
    CorrectionNotProposable,
    DeleteNoteOperation,
    DuplicateOperationTarget,
    EmptyProposal,
    OperationDecision,
    ProposalStatus,
    ProposedClaim,
    ProposedNote,
    StaleOperationTarget,
    UnknownProposalTarget,
    UpdateNoteOperation,
    WrongCorrection,
)
from soap.soap import (
    AssessmentClaim,
    IcdCoding,
    SoapClaim,
    SoapNote,
    SoapReport,
    TurnCitation,
)

_CREATED = datetime(2026, 7, 21, 9, 0, tzinfo=UTC)
_LATER = datetime(2026, 7, 21, 10, 0, tzinfo=UTC)
_LATEST = datetime(2026, 7, 21, 11, 0, tzinfo=UTC)
_ICD = IcdCoding(code="I10", name="Essential hypertension", classifier_url="icd://I10")


def _claim(text: str = "Headache for three days.") -> SoapClaim:
    return SoapClaim(id=Id.new(), text=text, citations=[TurnCitation(turn_id=Id.new())])


def _assessment() -> AssessmentClaim:
    return AssessmentClaim(
        id=Id.new(),
        text="Essential hypertension.",
        citations=[TurnCitation(turn_id=Id.new())],
        icd=_ICD,
    )


def _correction() -> SoapReportCorrection:
    coded = SoapNote(id=Id.new(), subjective=[_claim()], assessment=[_assessment()])
    other = SoapNote(id=Id.new(), subjective=[_claim("Also reports fatigue.")])
    report = SoapReport(id=Id.new(), notes=[coded, other])
    return SoapReportCorrection.start(report, created_at=_CREATED)


def _session(correction: SoapReportCorrection) -> CorrectionEditorSession:
    return CorrectionEditorSession.start(correction.id, created_at=_CREATED)


def _proposed(text: str = "Rewritten subjective.") -> ProposedNote:
    return ProposedNote(
        subjective=[ProposedClaim(text=text, citations=[TurnCitation(turn_id=Id.new())])]
    )


def _propose(
    session: CorrectionEditorSession,
    correction: SoapReportCorrection,
    operations: list[AddNoteOperation | UpdateNoteOperation | DeleteNoteOperation],
) -> object:
    return session.propose(
        correction,
        user_request="Tidy the notes.",
        model_id="medgemma-27b",
        prompt_version="v1",
        operations=operations,
        at=_LATER,
    )


def test_forming_a_proposal_does_not_touch_the_correction() -> None:
    correction = _correction()
    session = _session(correction)
    revision_before = correction.revision
    text_before = correction.notes[0].subjective[0].text

    _propose(session, correction, [UpdateNoteOperation(correction.notes[0].id, _proposed())])

    assert correction.revision == revision_before
    assert correction.notes[0].subjective[0].text == text_before
    assert len(correction.notes) == 2


def test_a_proposal_records_the_base_correction_revision() -> None:
    correction = _correction()
    correction.add_note(at=_LATER, subjective=[_claim("Extra note.")])
    session = _session(correction)

    proposal = _propose(session, correction, [DeleteNoteOperation(correction.notes[1].id)])

    assert proposal.base_correction_revision == correction.revision == 2


def test_operations_are_decided_independently_into_a_mixed_proposal() -> None:
    correction = _correction()
    session = _session(correction)
    proposal = _propose(
        session,
        correction,
        [
            UpdateNoteOperation(correction.notes[0].id, _proposed()),
            DeleteNoteOperation(correction.notes[1].id),
        ],
    )
    accepted, rejected = proposal.operations

    proposal.accept_operation(accepted.id, correction, at=_LATER)
    proposal.reject_operation(rejected.id, at=_LATER, reason="keep the fatigue note")

    assert accepted.decision is OperationDecision.ACCEPTED
    assert rejected.decision is OperationDecision.REJECTED
    assert proposal.status() is ProposalStatus.MIXED


@pytest.mark.parametrize(
    "decisions,expected",
    [
        (ProposalStatus.ACCEPTED, ProposalStatus.ACCEPTED),
        (ProposalStatus.REJECTED, ProposalStatus.REJECTED),
    ],
)
def test_uniform_decisions_derive_a_uniform_status(
    decisions: ProposalStatus, expected: ProposalStatus
) -> None:
    correction = _correction()
    session = _session(correction)
    proposal = _propose(
        session,
        correction,
        [AddNoteOperation(_proposed("A")), AddNoteOperation(_proposed("B"))],
    )
    for operation in proposal.operations:
        if decisions is ProposalStatus.ACCEPTED:
            proposal.accept_operation(operation.id, correction, at=_LATER)
        else:
            proposal.reject_operation(operation.id, at=_LATER, reason="no")

    assert proposal.status() is expected


def test_repeating_a_decision_is_idempotent() -> None:
    correction = _correction()
    session = _session(correction)
    proposal = _propose(session, correction, [AddNoteOperation(_proposed())])
    operation = proposal.operations[0]
    proposal.accept_operation(operation.id, correction, at=_LATER)

    proposal.accept_operation(operation.id, correction, at=_LATEST)

    assert operation.decision is OperationDecision.ACCEPTED
    assert operation.decided_at == _LATER


def test_rejecting_an_accepted_operation_conflicts() -> None:
    correction = _correction()
    session = _session(correction)
    proposal = _propose(session, correction, [AddNoteOperation(_proposed())])
    operation = proposal.operations[0]
    proposal.accept_operation(operation.id, correction, at=_LATER)

    with pytest.raises(ConflictingDecision):
        proposal.reject_operation(operation.id, at=_LATEST, reason="changed my mind")


def test_accepting_a_rejected_operation_conflicts() -> None:
    correction = _correction()
    session = _session(correction)
    proposal = _propose(session, correction, [AddNoteOperation(_proposed())])
    operation = proposal.operations[0]
    proposal.reject_operation(operation.id, at=_LATER, reason="no")

    with pytest.raises(ConflictingDecision):
        proposal.accept_operation(operation.id, correction, at=_LATEST)


def test_accepting_a_stale_target_is_refused() -> None:
    correction = _correction()
    session = _session(correction)
    proposal = _propose(
        session, correction, [UpdateNoteOperation(correction.notes[0].id, _proposed())]
    )
    operation = proposal.operations[0]
    correction.update_note(correction.notes[0].id, at=_LATER, subjective=[_claim("Doctor edited.")])

    with pytest.raises(StaleOperationTarget):
        proposal.accept_operation(operation.id, correction, at=_LATEST)


def test_accepting_a_vanished_target_is_refused() -> None:
    correction = _correction()
    session = _session(correction)
    proposal = _propose(session, correction, [DeleteNoteOperation(correction.notes[0].id)])
    operation = proposal.operations[0]
    correction.delete_note(correction.notes[0].id, at=_LATER)

    with pytest.raises(StaleOperationTarget):
        proposal.accept_operation(operation.id, correction, at=_LATEST)


def test_editing_another_note_leaves_the_operation_decidable() -> None:
    correction = _correction()
    session = _session(correction)
    proposal = _propose(
        session, correction, [UpdateNoteOperation(correction.notes[0].id, _proposed())]
    )
    operation = proposal.operations[0]
    correction.update_note(
        correction.notes[1].id, at=_LATER, subjective=[_claim("Unrelated edit.")]
    )

    proposal.accept_operation(operation.id, correction, at=_LATEST)

    assert operation.decision is OperationDecision.ACCEPTED


def test_proposed_content_has_no_channel_for_icd() -> None:
    assert "icd" not in {field.name for field in fields(ProposedClaim)}
    assert "icd" not in {field.name for field in fields(ProposedNote)}


def test_update_snapshots_the_existing_icd_so_it_is_preserved() -> None:
    correction = _correction()
    session = _session(correction)
    proposal = _propose(
        session, correction, [UpdateNoteOperation(correction.notes[0].id, _proposed())]
    )
    operation = proposal.operations[0]

    assert operation.before is not None
    assert operation.before.assessment[0].icd == _ICD


def test_reject_pending_sweeps_the_active_proposal_and_records_the_reason() -> None:
    correction = _correction()
    session = _session(correction)
    proposal = _propose(
        session,
        correction,
        [
            AddNoteOperation(_proposed("added")),
            UpdateNoteOperation(correction.notes[0].id, _proposed("updated")),
            DeleteNoteOperation(correction.notes[1].id),
        ],
    )
    already, pending_update, pending_delete = proposal.operations
    proposal.accept_operation(already.id, correction, at=_LATER)

    session.reject_pending(at=_LATEST, reason="doctor edited the correction by hand")

    assert already.decision is OperationDecision.ACCEPTED
    assert pending_update.decision is OperationDecision.REJECTED
    assert pending_delete.decision is OperationDecision.REJECTED
    assert pending_update.decision_reason == "doctor edited the correction by hand"


def test_at_most_one_proposal_stays_active() -> None:
    correction = _correction()
    session = _session(correction)
    _propose(session, correction, [AddNoteOperation(_proposed())])

    with pytest.raises(ActiveProposalExists):
        _propose(session, correction, [AddNoteOperation(_proposed())])


def test_a_new_proposal_opens_once_the_previous_is_fully_decided() -> None:
    correction = _correction()
    session = _session(correction)
    first = _propose(session, correction, [AddNoteOperation(_proposed())])
    first.reject_operation(first.operations[0].id, at=_LATER, reason="no")

    second = _propose(session, correction, [DeleteNoteOperation(correction.notes[1].id)])

    assert session.active_proposal() is second


def test_a_non_draft_correction_refuses_proposals() -> None:
    correction = _correction()
    session = _session(correction)
    correction.verify("dr-house", at=_LATER)

    with pytest.raises(CorrectionNotProposable):
        _propose(session, correction, [AddNoteOperation(_proposed())])

    correction.begin_publication(at=_LATER)
    with pytest.raises(CorrectionNotProposable):
        _propose(session, correction, [AddNoteOperation(_proposed())])

    correction.mark_published(at=_LATER)
    with pytest.raises(CorrectionNotProposable):
        _propose(session, correction, [AddNoteOperation(_proposed())])


def test_a_session_refuses_a_proposal_for_another_correction() -> None:
    session = _session(_correction())
    foreign = _correction()

    with pytest.raises(WrongCorrection):
        _propose(session, foreign, [AddNoteOperation(_proposed())])


def test_two_operations_on_the_same_note_are_rejected() -> None:
    correction = _correction()
    session = _session(correction)

    with pytest.raises(DuplicateOperationTarget):
        _propose(
            session,
            correction,
            [
                UpdateNoteOperation(correction.notes[0].id, _proposed()),
                DeleteNoteOperation(correction.notes[0].id),
            ],
        )


def test_an_operation_targeting_an_unknown_note_is_rejected() -> None:
    correction = _correction()
    session = _session(correction)

    with pytest.raises(UnknownProposalTarget):
        _propose(session, correction, [UpdateNoteOperation(Id.new(), _proposed())])


def test_an_empty_proposal_is_rejected() -> None:
    correction = _correction()
    session = _session(correction)

    with pytest.raises(EmptyProposal):
        _propose(session, correction, [])
