"""Integration test: migrations apply to a clean DB and a session round-trips.

Requires a reachable PostgreSQL via ``DATABASE_URL``; skipped otherwise (CI and
the compose stack provide one). Migrations run synchronously first because
Alembic spins its own event loop, then the session check runs on a fresh loop.
"""

from __future__ import annotations

import asyncio
import os

import pytest
from sqlalchemy import text

from infra.db import dispose_engine, get_sessionmaker
from infra.migrations import run_migrations

pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL"),
    reason="DATABASE_URL not configured; skipping DB integration test",
)


async def _select_one_and_read_version() -> None:
    async with get_sessionmaker()() as session:
        assert (await session.execute(text("SELECT 1"))).scalar_one() == 1
        heads = (await session.execute(text("SELECT version_num FROM alembic_version"))).scalars()
        assert heads.all(), "alembic_version should carry a head after upgrade"
    await dispose_engine()


def test_migrations_apply_and_session_executes() -> None:
    run_migrations()
    asyncio.run(_select_one_and_read_version())
