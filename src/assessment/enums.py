from enum import Enum


class AnnotationStatus(str, Enum):
    """Статус разметки SOAP отчёта."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class ChangeType(str, Enum):
    """Тип изменения, которое внёс врач."""
    TEXT_MODIFIED = "text_modified"
    EVIDENCE_CHANGED = "evidence_changed"
    CLAIM_ADDED = "claim_added"
    CLAIM_DELETED = "claim_deleted"