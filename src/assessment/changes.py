from __future__ import annotations

from dataclasses import dataclass
from shared.value_objects import ClaimId

@dataclass(frozen=True, slots=True)
class SoapChange:
    """Базовый класс для любого изменения в SOAP."""
    claim_id: ClaimId

@dataclass(frozen=True, slots=True)
class TextModified(SoapChange):
    old_text: str
    new_text: str

@dataclass(frozen=True, slots=True)
class EvidenceChanged(SoapChange):
    old_evidence: str
    new_evidence: str

@dataclass(frozen=True, slots=True)
class NoteAdded(SoapChange):
    """Если ИИ пропустил факт, и врач добавил заметку с нуля."""
    note_text: str
    section: str

@dataclass(frozen=True, slots=True)
class NothingChanged(SoapChange):
    """Специальный объект, если врач проверил и подтвердил, что изменений нет."""
    pass