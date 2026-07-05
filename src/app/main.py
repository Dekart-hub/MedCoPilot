from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request, status

from di import build_container, teardown_container

from .api.v1 import api_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Зависимости не готовы, пока не собран контейнер.
    app.state.ready = False
    container = await build_container()
    app.state.container = container
    app.state.ready = True
    try:
        yield
    finally:
        app.state.ready = False
        await teardown_container(container)
        app.state.container = None


def create_app() -> FastAPI:
    app = FastAPI(title="MedCoPilot", lifespan=lifespan)

    @app.get("/health", tags=["system"])
    async def health() -> dict[str, str]:
        """Liveness: процесс жив и отвечает."""
        return {"status": "ok"}

    @app.get("/ready", tags=["system"])
    async def ready(request: Request) -> dict[str, str]:
        """Readiness: зависимости подняты и можно принимать трафик."""
        if not getattr(request.app.state, "ready", False):
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service is not ready"
            )
        return {"status": "ready"}

    app.include_router(api_router)
    return app


app = create_app()
