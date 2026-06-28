from __future__ import annotations

from dataclasses import dataclass, field

from shared.entity import Entity
from shared.value_objects import FloatRangedScore, Id

from ..soap import SoapClaimId, SoapReportId

type SoapNoteCodingId = Id[SoapNoteCoding]
type SoapCodingReportId = Id[SoapCodingReport]

# OID'ы справочников НСИ Минздрава.
MKB10_SYSTEM_OID = "1.2.643.5.1.13.13.11.1005"  # Том 1 (коды и названия)
MKB10_INDEX_OID = "1.2.643.5.1.13.13.11.1489"  # Том 3 (указатель формулировок)


@dataclass(frozen=True, slots=True)
class ClassifierRef:
    """Референс на источник кодирования (provenance).

    Чем именно закодировали: система кодов (``system`` — OID Тома 1) и её
    ``version``, плюс источник формулировки, по которой нашли код (``index_oid``
    — Том 3 — и его ``index_version``). Нужен для аудита: МКБ-10 ревизуется,
    НСИ выпускает версии, и код без версии в ЭМК неаудируем. Аналог
    ``system``/``version`` в FHIR Coding.
    """

    system: str
    name: str = "МКБ-10 (НСИ Минздрава России)"
    version: str | None = None
    index_oid: str | None = None
    index_version: str | None = None


# Референс по умолчанию: МКБ-10 НСИ без известных версий (проставляются при
# сборке индекса из meta-файлов выгрузки).
DEFAULT_MKB10_REF = ClassifierRef(system=MKB10_SYSTEM_OID, index_oid=MKB10_INDEX_OID)


@dataclass(frozen=True, slots=True)
class DiagnosisCoding:
    """Один кандидат-код классификатора для диагноза.

    ``matched_formulation`` — формулировка из алфавитного указателя (Том 3),
    по которой код был найден; ``title`` — каноническое название рубрики
    (Том 1). ``score`` — относительная уверенность ретрива в пределах запроса
    (не калиброванная вероятность; калибровка — забота более поздних tier'ов).
    ``classifier`` — чем закодировали (provenance).
    """

    code: str
    title: str
    matched_formulation: str
    score: FloatRangedScore
    classifier: ClassifierRef = DEFAULT_MKB10_REF


@dataclass(eq=False, slots=True)
class SoapNoteCoding(Entity[SoapNoteCodingId]):
    """Результат нормализации одного assessment-клейма SOAP-ноты.

    Side-car: привязывается к ``SoapClaim`` ассессмента по ``soap_claim_id`` и
    не мутирует доменную модель SOAP. ``candidates`` — ранжированный список
    кодов-кандидатов (может быть пустым, если совпадений нет).
    """

    id: SoapNoteCodingId
    soap_claim_id: SoapClaimId
    candidates: list[DiagnosisCoding] = field(default_factory=list)

    @property
    def best(self) -> DiagnosisCoding | None:
        return self.candidates[0] if self.candidates else None


@dataclass(eq=False, slots=True)
class SoapCodingReport(Entity[SoapCodingReportId]):
    """Кодирование по всему отчёту — параллель ``SoapConfidenceReport``."""

    id: SoapCodingReportId
    soap_report_id: SoapReportId
    codings: list[SoapNoteCoding] = field(default_factory=list)
