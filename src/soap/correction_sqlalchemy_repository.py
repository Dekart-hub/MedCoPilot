"""SQLAlchemy adapter for :class:`SoapReportCorrectionRepository`.

Maps the pure domain aggregate onto the ORM rows in :mod:`soap.correction_orm`
and back. Transaction control (commit/rollback) is left to the caller so the
correction and whatever else the request touches share one unit of work — the
adapter never commits, exactly like :mod:`soap.sqlalchemy_repository`.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from dialogue.dialogue import DialogueTurnId
from shared.value_objects import Id

from .correction import (
    CorrectedNote,
    CorrectionId,
    CorrectionStatus,
    SoapReportCorrection,
)
from .correction_orm import SoapCorrectedClaimRow, SoapCorrectedNoteRow, SoapCorrectionRow
from .correction_repository import SoapReportCorrectionRepository
from .soap import (
    AssessmentClaim,
    IcdCandidate,
    IcdCoding,
    IcdResolution,
    IcdResolutionStatus,
    SoapClaim,
    SoapReportId,
    SoapSection,
    TurnCitation,
)


class SqlAlchemySoapReportCorrectionRepository(SoapReportCorrectionRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(self, correction: SoapReportCorrection) -> None:
        await self._session.merge(_to_row(correction))

    async def get(self, correction_id: CorrectionId) -> SoapReportCorrection | None:
        return await self._fetch(SoapCorrectionRow.id == correction_id.value)

    async def get_by_source_report_id(self, report_id: SoapReportId) -> SoapReportCorrection | None:
        return await self._fetch(SoapCorrectionRow.source_report_id == report_id.value)

    async def get_by_source_report_id_for_update(
        self, report_id: SoapReportId
    ) -> SoapReportCorrection | None:
        return await self._fetch(
            SoapCorrectionRow.source_report_id == report_id.value,
            for_update=True,
        )

    async def _fetch(
        self, condition: object, *, for_update: bool = False
    ) -> SoapReportCorrection | None:
        statement = (
            select(SoapCorrectionRow)
            .where(condition)  # type: ignore[arg-type]
            .options(
                selectinload(SoapCorrectionRow.notes).selectinload(SoapCorrectedNoteRow.claims)
            )
        )
        if for_update:
            statement = statement.with_for_update()
        row = (await self._session.execute(statement)).scalar_one_or_none()
        return _to_domain(row) if row is not None else None


def _to_row(correction: SoapReportCorrection) -> SoapCorrectionRow:
    return SoapCorrectionRow(
        id=correction.id.value,
        source_report_id=correction.source_report_id.value,
        status=correction.status.value,
        verified_by=correction.verified_by,
        verified_at=correction.verified_at,
        created_at=correction.created_at,
        updated_at=correction.updated_at,
        revision=correction.revision,
        notes=[_note_to_row(note, position) for position, note in enumerate(correction.notes)],
    )


def _note_to_row(note: CorrectedNote, position: int) -> SoapCorrectedNoteRow:
    claims: list[SoapCorrectedClaimRow] = []
    for section, section_claims in note.sections():
        for claim in section_claims:
            claims.append(_claim_to_row(claim, section, len(claims)))
    return SoapCorrectedNoteRow(
        id=note.id.value,
        position=position,
        source_note_id=note.source_note_id.value if note.source_note_id is not None else None,
        claims=claims,
    )


def _claim_to_row(claim: SoapClaim, section: SoapSection, position: int) -> SoapCorrectedClaimRow:
    icd = claim.icd if isinstance(claim, AssessmentClaim) else None
    resolution = claim.icd_resolution if isinstance(claim, AssessmentClaim) else None
    return SoapCorrectedClaimRow(
        id=claim.id.value,
        position=position,
        section=section.value,
        text=claim.text,
        citations=[_citation_to_json(citation) for citation in claim.citations],
        icd_code=icd.code if icd is not None else None,
        icd_name=icd.name if icd is not None else None,
        icd_classifier_url=icd.classifier_url if icd is not None else None,
        icd_status=resolution.status.value if resolution is not None else None,
        icd_classifier_version=resolution.classifier_version if resolution is not None else None,
        icd_candidates=(
            [_candidate_to_json(candidate) for candidate in resolution.candidates]
            if resolution is not None
            else None
        ),
    )


def _citation_to_json(citation: TurnCitation) -> dict[str, str | None]:
    return {"turn_id": str(citation.turn_id), "quote": citation.quote}


def _candidate_to_json(candidate: IcdCandidate) -> dict[str, object]:
    return {
        "code": candidate.code,
        "name": candidate.name,
        "rank": candidate.rank,
        "bm25_score": candidate.bm25_score,
    }


def _to_domain(row: SoapCorrectionRow) -> SoapReportCorrection:
    return SoapReportCorrection(
        id=Id(row.id),
        source_report_id=Id(row.source_report_id),
        status=CorrectionStatus(row.status),
        created_at=row.created_at,
        updated_at=row.updated_at,
        notes=[_note_to_domain(note) for note in row.notes],
        verified_by=row.verified_by,
        verified_at=row.verified_at,
        revision=row.revision,
    )


def _note_to_domain(row: SoapCorrectedNoteRow) -> CorrectedNote:
    buckets: dict[str, list[SoapCorrectedClaimRow]] = {section.value: [] for section in SoapSection}
    for claim in row.claims:
        buckets[claim.section].append(claim)

    def claims(section: SoapSection) -> list[SoapCorrectedClaimRow]:
        return buckets[section.value]

    return CorrectedNote(
        id=Id(row.id),
        source_note_id=Id(row.source_note_id) if row.source_note_id is not None else None,
        subjective=[_claim_to_domain(c) for c in claims(SoapSection.SUBJECTIVE)],
        objective=[_claim_to_domain(c) for c in claims(SoapSection.OBJECTIVE)],
        assessment=[_assessment_to_domain(c) for c in claims(SoapSection.ASSESSMENT)],
        plan=[_claim_to_domain(c) for c in claims(SoapSection.PLAN)],
    )


def _claim_to_domain(row: SoapCorrectedClaimRow) -> SoapClaim:
    return SoapClaim(id=Id(row.id), text=row.text, citations=_citations_to_domain(row.citations))


def _assessment_to_domain(row: SoapCorrectedClaimRow) -> AssessmentClaim:
    return AssessmentClaim(
        id=Id(row.id),
        text=row.text,
        citations=_citations_to_domain(row.citations),
        icd=_icd_to_domain(row),
        icd_resolution=_resolution_to_domain(row),
    )


def _icd_to_domain(row: SoapCorrectedClaimRow) -> IcdCoding | None:
    if row.icd_code is None or row.icd_name is None or row.icd_classifier_url is None:
        return None
    return IcdCoding(code=row.icd_code, name=row.icd_name, classifier_url=row.icd_classifier_url)


def _resolution_to_domain(row: SoapCorrectedClaimRow) -> IcdResolution | None:
    # Pre-T29 rows (and manual codings) carry no status: no resolution to rebuild.
    if row.icd_status is None:
        return None
    return IcdResolution(
        status=IcdResolutionStatus(row.icd_status),
        selected=_icd_to_domain(row),
        candidates=tuple(_candidate_to_domain(entry) for entry in (row.icd_candidates or [])),
        classifier_version=row.icd_classifier_version or "",
    )


def _candidate_to_domain(entry: dict[str, object]) -> IcdCandidate:
    return IcdCandidate(
        code=str(entry["code"]),
        name=str(entry["name"]),
        rank=int(entry["rank"]),  # type: ignore[call-overload]
        bm25_score=float(entry["bm25_score"]),  # type: ignore[arg-type]
    )


def _citations_to_domain(data: list[dict[str, str | None]]) -> list[TurnCitation]:
    return [_citation_to_domain(entry) for entry in data]


def _citation_to_domain(entry: dict[str, str | None]) -> TurnCitation:
    raw_turn_id = entry["turn_id"]
    assert raw_turn_id is not None  # a persisted citation always carries a turn id
    turn_id: DialogueTurnId = Id(UUID(raw_turn_id))
    return TurnCitation(turn_id=turn_id, quote=entry["quote"])
