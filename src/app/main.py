"""FastAPI application entrypoint.

Exposes ``app`` for ``uvicorn app.main:app``. This is the T2 skeleton extended
with structured logging: a liveness probe plus per-request logging. Readiness
and domain routes arrive in later tasks.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.middleware import RequestLoggingMiddleware
from config.logging import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Configure logging once uvicorn has installed its own loggers."""
    configure_logging()
    yield


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(title="MedCoPilot", lifespan=lifespan)
    app.add_middleware(RequestLoggingMiddleware)

    @app.get("/health", tags=["system"])
    async def health() -> dict[str, str]:
        """Liveness probe: the process is up and serving requests."""
        return {"status": "ok"}

    return app


app = create_app()
