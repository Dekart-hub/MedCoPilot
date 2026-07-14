"""FastAPI application entrypoint.

Exposes ``app`` for ``uvicorn app.main:app``. This is the T2 skeleton: a
liveness probe only. Readiness and domain routes arrive in later tasks.
"""

from __future__ import annotations

from fastapi import FastAPI


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(title="MedCoPilot")

    @app.get("/health", tags=["system"])
    async def health() -> dict[str, str]:
        """Liveness probe: the process is up and serving requests."""
        return {"status": "ok"}

    return app


app = create_app()
