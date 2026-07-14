"""HTTP middleware: one structured log event per request, keyed by a request id.

The request id is bound to structlog's context so any log emitted while the
request is handled inherits it, and it is echoed back in the ``X-Request-ID``
response header for cross-service tracing.
"""

from __future__ import annotations

import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

_REQUEST_ID_HEADER = "X-Request-ID"


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Bind a request id and emit a structured summary for every request."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get(_REQUEST_ID_HEADER) or uuid.uuid4().hex
        log = structlog.get_logger("app.request").bind(request_id=request_id)
        structlog.contextvars.bind_contextvars(request_id=request_id)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception(
                "http_request_failed",
                method=request.method,
                path=request.url.path,
                duration_ms=_elapsed_ms(start),
            )
            raise
        else:
            response.headers[_REQUEST_ID_HEADER] = request_id
            log.info(
                "http_request",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=_elapsed_ms(start),
            )
            return response
        finally:
            structlog.contextvars.clear_contextvars()


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)
