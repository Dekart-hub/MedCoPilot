from __future__ import annotations


class EhrError(Exception):
    """Base class for mock-EHR workflow failures."""


class DialogueNotFoundError(EhrError):
    pass


class ReportNotFoundError(EhrError):
    pass


class UnlinkedDialogueError(EhrError):
    pass


class ReportNotApprovedError(EhrError):
    pass


class ApprovalConflictError(EhrError):
    pass


class InvalidEhrReferenceError(EhrError):
    pass


class EhrGatewayError(EhrError):
    """The external mock EHR rejected a request or could not be reached."""


class EhrIntegrationDisabledError(EhrGatewayError):
    pass
