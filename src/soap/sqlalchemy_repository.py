"""SQLAlchemy adapter for :class:`SoapReportRepository`.

Maps the pure domain aggregate onto the ORM rows in :mod:`soap.orm` and back.
Transaction control (commit/rollback) is left to the caller so a report and its
dialogue can share one unit of work.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from dialogue.dialogue import DialogueId, DialogueTurnId
from shared.value_objects import Id

from .orm import SoapClaimRow, SoapNoteRow, SoapReportRow
from .repository import SoapReportRepository
from .soap import (
    AssessmentClaim,
    IcdCoding,
    SoapClaim,
    SoapNote,
    SoapReport,
    SoapReportId,
    SoapSection,
    TurnCitation,
)


class SqlAlchemySoapReportRepository(SoapReportRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(self, report: SoapReport, *, dialogue_id: DialogueId) -> None:
        await self._session.merge(_to_row(report, dialogue_id))

    async def get(self, report_id: SoapReportId) -> SoapReport | None:
        return await self._fetch(SoapReportRow.id == report_id.value)

    async def get_by_dialogue_id(self, dialogue_id: DialogueId) -> SoapReport | None:
        return await self._fetch(SoapReportRow.dialogue_id == dialogue_id.value)

    async def _fetch(self, condition: object) -> SoapReport | None:
        statement = (
            select(SoapReportRow)
            .where(condition)  # type: ignore[arg-type]
            .options(selectinload(SoapReportRow.notes).selectinload(SoapNoteRow.claims))
        )
        row = (await self._session.execute(statement)).scalar_one_or_none()
        return _to_domain(row) if row is not None else None


def _to_row(report: SoapReport, dialogue_id: DialogueId) -> SoapReportRow:
    return SoapReportRow(
        id=report.id.value,
        dialogue_id=dialogue_id.value,
        notes=[_note_to_row(note, position) for position, note in enumerate(report.notes)],
    )


def _note_to_row(note: SoapNote, position: int) -> SoapNoteRow:
    claims: list[SoapClaimRow] = []
    for section, section_claims in note.sections():
        for claim in section_claims:
            claims.append(_claim_to_row(claim, section, len(claims)))
    return SoapNoteRow(
        id=note.id.value,
        position=position,
        confidence=note.confidence,
        claims=claims,
    )


def _claim_to_row(claim: SoapClaim, section: SoapSection, position: int) -> SoapClaimRow:
    icd = claim.icd if isinstance(claim, AssessmentClaim) else None
    return SoapClaimRow(
        id=claim.id.value,
        position=position,
        section=section.value,
        text=claim.text,
        citations=[_citation_to_json(citation) for citation in claim.citations],
        icd_code=icd.code if icd is not None else None,
        icd_name=icd.name if icd is not None else None,
        icd_classifier_url=icd.classifier_url if icd is not None else None,
    )


def _citation_to_json(citation: TurnCitation) -> dict[str, str | None]:
    return {"turn_id": str(citation.turn_id), "quote": citation.quote}


def _to_domain(row: SoapReportRow) -> SoapReport:
    return SoapReport(id=Id(row.id), notes=[_note_to_domain(note) for note in row.notes])


def _note_to_domain(row: SoapNoteRow) -> SoapNote:
    buckets: dict[str, list[SoapClaimRow]] = {section.value: [] for section in SoapSection}
    for claim in row.claims:
        buckets[claim.section].append(claim)

    def claims(section: SoapSection) -> list[SoapClaimRow]:
        return buckets[section.value]

    return SoapNote(
        id=Id(row.id),
        subjective=[_claim_to_domain(c) for c in claims(SoapSection.SUBJECTIVE)],
        objective=[_claim_to_domain(c) for c in claims(SoapSection.OBJECTIVE)],
        assessment=[_assessment_to_domain(c) for c in claims(SoapSection.ASSESSMENT)],
        plan=[_claim_to_domain(c) for c in claims(SoapSection.PLAN)],
        confidence=row.confidence,
    )


def _claim_to_domain(row: SoapClaimRow) -> SoapClaim:
    return SoapClaim(id=Id(row.id), text=row.text, citations=_citations_to_domain(row.citations))


def _assessment_to_domain(row: SoapClaimRow) -> AssessmentClaim:
    return AssessmentClaim(
        id=Id(row.id),
        text=row.text,
        citations=_citations_to_domain(row.citations),
        icd=_icd_to_domain(row),
    )


def _icd_to_domain(row: SoapClaimRow) -> IcdCoding | None:
    if row.icd_code is None or row.icd_name is None or row.icd_classifier_url is None:
        return None
    return IcdCoding(code=row.icd_code, name=row.icd_name, classifier_url=row.icd_classifier_url)


def _citations_to_domain(data: list[dict[str, str | None]]) -> list[TurnCitation]:
    return [_citation_to_domain(entry) for entry in data]


def _citation_to_domain(entry: dict[str, str | None]) -> TurnCitation:
    raw_turn_id = entry["turn_id"]
    assert raw_turn_id is not None  # a persisted citation always carries a turn id
    turn_id: DialogueTurnId = Id(UUID(raw_turn_id))
    return TurnCitation(turn_id=turn_id, quote=entry["quote"])
