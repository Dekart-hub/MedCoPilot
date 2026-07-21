"""Alembic environment: async online migrations driven by application settings.

The database URL is taken from ``settings.database_url`` (never hardcoded), and
``target_metadata`` points at the application's declarative ``Base`` so future
autogenerate runs (T5+) can diff the models.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

import dialogue.orm  # noqa: F401  (side effect: register tables on Base.metadata)
import soap.correction_orm  # noqa: F401  (side effect: register tables on Base.metadata)
import soap.orm  # noqa: F401  (side effect: register tables on Base.metadata)
import soap.proposal_orm  # noqa: F401  (side effect: register tables on Base.metadata)
from config.settings import get_settings
from infra.db import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _database_url() -> str:
    url = get_settings().database_url
    if url is None:
        raise RuntimeError("DATABASE_URL is not configured")
    return url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode against a URL, without a DBAPI."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def _do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def _run_async_migrations() -> None:
    connectable = async_engine_from_config(
        {"sqlalchemy.url": _database_url()},
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(_do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode with an async engine."""
    asyncio.run(_run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
