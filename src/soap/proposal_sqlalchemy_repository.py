"""SQLAlchemy adapter for :class:`CorrectionEditorSessionRepository`.

Maps the pure :mod:`soap.proposal` aggregate onto the ORM rows in
:mod:`soap.proposal_orm` and back. Proposed content and before-snapshots are
stored as JSON blobs; a snapshot round-trips a full corrected note (ids,
citations and ICD included) so the durable log stays faithful. Transaction
control is the caller's, exactly like :mod:`soap.correction_sqlalchemy_repository`
— the adapter never commits.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from dialogue.dialogue import DialogueTurnId
from shared.value_objects import Id

from .correction import CorrectedNote, CorrectionId
from .proposal import (
    CorrectionEditorSession,
    CorrectionProposal,
    OperationDecision,
    OperationType,
    ProposalOperation,
    ProposedClaim,
    ProposedNote,
    SessionId,
)
from .proposal_orm import (
    SoapCorrectionEditorSessionRow,
    SoapCorrectionProposalOperationRow,
    SoapCorrectionProposalRow,
)
from .proposal_repository import CorrectionEditorSessionRepository
from .soap import AssessmentClaim, IcdCoding, SoapClaim, SoapSection, TurnCitation


class SqlAlchemyCorrectionEditorSessionRepository(CorrectionEditorSessionRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(self, session: CorrectionEditorSession) -> None:
        await self._session.merge(_session_to_row(session))

    async def get(self, session_id: SessionId) -> CorrectionEditorSession | None:
        return await self._fetch(SoapCorrectionEditorSessionRow.id == session_id.value)

    async def get_for_correction(
        self, correction_id: CorrectionId
    ) -> CorrectionEditorSession | None:
        return await self._fetch(
            SoapCorrectionEditorSessionRow.correction_id == correction_id.value
        )

    async def _fetch(self, condition: object) -> CorrectionEditorSession | None:
        statement = (
            select(SoapCorrectionEditorSessionRow)
            .where(condition)  # type: ignore[arg-type]
            .options(
                selectinload(SoapCorrectionEditorSessionRow.proposals).selectinload(
                    SoapCorrectionProposalRow.operations
                )
            )
        )
        row = (await self._session.execute(statement)).scalar_one_or_none()
        return _session_to_domain(row) if row is not None else None


def _session_to_row(session: CorrectionEditorSession) -> SoapCorrectionEditorSessionRow:
    return SoapCorrectionEditorSessionRow(
        id=session.id.value,
        correction_id=session.correction_id.value,
        created_at=session.created_at,
        updated_at=session.updated_at,
        proposals=[
            _proposal_to_row(proposal, position)
            for position, proposal in enumerate(session.proposals)
        ],
    )


def _proposal_to_row(proposal: CorrectionProposal, position: int) -> SoapCorrectionProposalRow:
    return SoapCorrectionProposalRow(
        id=proposal.id.value,
        position=position,
        user_request=proposal.user_request,
        base_correction_revision=proposal.base_correction_revision,
        model_id=proposal.model_id,
        prompt_version=proposal.prompt_version,
        created_at=proposal.created_at,
        updated_at=proposal.updated_at,
        operations=[
            _operation_to_row(operation, index)
            for index, operation in enumerate(proposal.operations)
        ],
    )


def _operation_to_row(
    operation: ProposalOperation, position: int
) -> SoapCorrectionProposalOperationRow:
    return SoapCorrectionProposalOperationRow(
        id=operation.id.value,
        position=position,
        type=operation.type.value,
        decision=operation.decision.value,
        target_note_id=operation.target_note_id.value
        if operation.target_note_id is not None
        else None,
        proposed_content=_proposed_to_json(operation.proposed)
        if operation.proposed is not None
        else None,
        before_snapshot=_snapshot_to_json(operation.before)
        if operation.before is not None
        else None,
        target_fingerprint=operation.target_fingerprint,
        decided_at=operation.decided_at,
        decision_reason=operation.decision_reason,
    )


def _session_to_domain(row: SoapCorrectionEditorSessionRow) -> CorrectionEditorSession:
    return CorrectionEditorSession(
        id=Id(row.id),
        correction_id=Id(row.correction_id),
        created_at=row.created_at,
        updated_at=row.updated_at,
        proposals=[_proposal_to_domain(proposal) for proposal in row.proposals],
    )


def _proposal_to_domain(row: SoapCorrectionProposalRow) -> CorrectionProposal:
    return CorrectionProposal(
        id=Id(row.id),
        user_request=row.user_request,
        base_correction_revision=row.base_correction_revision,
        model_id=row.model_id,
        prompt_version=row.prompt_version,
        created_at=row.created_at,
        updated_at=row.updated_at,
        operations=[_operation_to_domain(operation) for operation in row.operations],
    )


def _operation_to_domain(row: SoapCorrectionProposalOperationRow) -> ProposalOperation:
    return ProposalOperation(
        id=Id(row.id),
        type=OperationType(row.type),
        target_note_id=Id(row.target_note_id) if row.target_note_id is not None else None,
        proposed=_proposed_from_json(row.proposed_content)
        if row.proposed_content is not None
        else None,
        before=_snapshot_from_json(row.before_snapshot)
        if row.before_snapshot is not None
        else None,
        target_fingerprint=row.target_fingerprint,
        decision=OperationDecision(row.decision),
        decided_at=row.decided_at,
        decision_reason=row.decision_reason,
    )


def _proposed_to_json(note: ProposedNote) -> dict[str, object]:
    return {
        "sections": {
            SoapSection.SUBJECTIVE.value: [_proposed_claim_to_json(c) for c in note.subjective],
            SoapSection.OBJECTIVE.value: [_proposed_claim_to_json(c) for c in note.objective],
            SoapSection.ASSESSMENT.value: [_proposed_claim_to_json(c) for c in note.assessment],
            SoapSection.PLAN.value: [_proposed_claim_to_json(c) for c in note.plan],
        }
    }


def _proposed_claim_to_json(claim: ProposedClaim) -> dict[str, object]:
    return {"text": claim.text, "citations": [_citation_to_json(c) for c in claim.citations]}


def _proposed_from_json(data: Mapping[str, Any]) -> ProposedNote:
    sections = data["sections"]
    return ProposedNote(
        subjective=_proposed_claims(sections, SoapSection.SUBJECTIVE),
        objective=_proposed_claims(sections, SoapSection.OBJECTIVE),
        assessment=_proposed_claims(sections, SoapSection.ASSESSMENT),
        plan=_proposed_claims(sections, SoapSection.PLAN),
    )


def _proposed_claims(sections: Mapping[str, Any], section: SoapSection) -> list[ProposedClaim]:
    return [_proposed_claim_from_json(claim) for claim in sections[section.value]]


def _proposed_claim_from_json(data: Mapping[str, Any]) -> ProposedClaim:
    return ProposedClaim(text=data["text"], citations=_citations_from_json(data["citations"]))


def _snapshot_to_json(note: CorrectedNote) -> dict[str, object]:
    return {
        "note_id": str(note.id),
        "source_note_id": str(note.source_note_id) if note.source_note_id is not None else None,
        "sections": {
            section.value: [_snapshot_claim_to_json(claim) for claim in claims]
            for section, claims in note.sections()
        },
    }


def _snapshot_claim_to_json(claim: SoapClaim) -> dict[str, object]:
    icd = claim.icd if isinstance(claim, AssessmentClaim) else None
    return {
        "id": str(claim.id),
        "text": claim.text,
        "citations": [_citation_to_json(c) for c in claim.citations],
        "icd": None if icd is None else _icd_to_json(icd),
    }


def _snapshot_from_json(data: Mapping[str, Any]) -> CorrectedNote:
    sections = data["sections"]
    raw_source = data["source_note_id"]
    return CorrectedNote(
        id=Id(UUID(data["note_id"])),
        source_note_id=Id(UUID(raw_source)) if raw_source is not None else None,
        subjective=[_snapshot_claim(c) for c in sections[SoapSection.SUBJECTIVE.value]],
        objective=[_snapshot_claim(c) for c in sections[SoapSection.OBJECTIVE.value]],
        assessment=[_snapshot_assessment(c) for c in sections[SoapSection.ASSESSMENT.value]],
        plan=[_snapshot_claim(c) for c in sections[SoapSection.PLAN.value]],
    )


def _snapshot_claim(data: Mapping[str, Any]) -> SoapClaim:
    return SoapClaim(
        id=Id(UUID(data["id"])),
        text=data["text"],
        citations=_citations_from_json(data["citations"]),
    )


def _snapshot_assessment(data: Mapping[str, Any]) -> AssessmentClaim:
    return AssessmentClaim(
        id=Id(UUID(data["id"])),
        text=data["text"],
        citations=_citations_from_json(data["citations"]),
        icd=_icd_from_json(data["icd"]),
    )


def _icd_to_json(icd: IcdCoding) -> dict[str, str]:
    return {"code": icd.code, "name": icd.name, "classifier_url": icd.classifier_url}


def _icd_from_json(data: Mapping[str, Any] | None) -> IcdCoding | None:
    if data is None:
        return None
    return IcdCoding(code=data["code"], name=data["name"], classifier_url=data["classifier_url"])


def _citation_to_json(citation: TurnCitation) -> dict[str, str | None]:
    return {"turn_id": str(citation.turn_id), "quote": citation.quote}


def _citations_from_json(data: list[Mapping[str, Any]]) -> list[TurnCitation]:
    return [_citation_from_json(entry) for entry in data]


def _citation_from_json(entry: Mapping[str, Any]) -> TurnCitation:
    turn_id: DialogueTurnId = Id(UUID(entry["turn_id"]))
    return TurnCitation(turn_id=turn_id, quote=entry["quote"])
