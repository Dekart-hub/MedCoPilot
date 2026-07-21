"""FastAPI application entrypoint.

Exposes ``app`` for ``uvicorn app.main:app``. On startup it configures logging
and auto-applies Alembic migrations, then serves a liveness probe (``/health``)
and a DB-backed readiness probe (``/health/ready``).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.dependencies import SessionDep
from app.errors import register_exception_handlers
from app.middleware import RequestLoggingMiddleware
from app.routes import router
from config.logging import configure_logging
from config.settings import get_settings
from ehr.dispatcher import PublicationDispatcher
from infra.db import dispose_engine, get_sessionmaker
from infra.fhir import build_fhir_publication_gateway
from infra.migrations import run_migrations


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Configure logging, migrate the database, then release the pool on exit."""
    configure_logging()
    await _apply_migrations()
    settings = get_settings()
    stop = asyncio.Event()
    gateway = None
    task = None
    if settings.fhir_dispatcher_enabled and settings.database_url is not None:
        gateway = build_fhir_publication_gateway(settings)
        dispatcher = PublicationDispatcher(
            get_sessionmaker(),
            gateway,
            batch_size=settings.fhir_dispatcher_batch_size,
            poll_seconds=settings.fhir_dispatcher_poll_seconds,
            retry_initial_seconds=settings.fhir_retry_initial_seconds,
            retry_max_seconds=settings.fhir_retry_max_seconds,
        )
        task = asyncio.create_task(dispatcher.run_forever(stop))
        app.state.publication_dispatcher = dispatcher
    try:
        yield
    finally:
        stop.set()
        if task is not None:
            await task
        if gateway is not None:
            await gateway.aclose()
        await dispose_engine()


async def _apply_migrations() -> None:
    """Run ``alembic upgrade head`` off the event loop before serving traffic."""
    log = structlog.get_logger("app.startup")
    if get_settings().database_url is None:
        log.warning("database_url_not_configured; skipping migrations")
        return
    await asyncio.to_thread(run_migrations)
    log.info("database_migrations_applied")


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(title="MedCoPilot", lifespan=lifespan)
    app.add_middleware(RequestLoggingMiddleware)
    app.include_router(router)
    register_exception_handlers(app)

    @app.get("/health", tags=["system"])
    async def health() -> dict[str, str]:
        """Liveness probe: the process is up and serving requests."""
        return {"status": "ok"}

    @app.get("/health/ready", tags=["system"])
    async def readiness(session: SessionDep) -> JSONResponse:
        """Readiness probe: the database is reachable (``SELECT 1``)."""
        try:
            await session.execute(text("SELECT 1"))
        except SQLAlchemyError:
            return JSONResponse({"status": "unavailable"}, status_code=503)
        return JSONResponse({"status": "ready"})

    return app


app = create_app()
