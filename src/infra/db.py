"""Async database infrastructure: engine, session factory and the DI dependency.

The engine and session factory are process-wide singletons built lazily from
``settings.database_url``. ``get_session`` is the FastAPI dependency later tasks
consume to obtain an :class:`~sqlalchemy.ext.asyncio.AsyncSession` per request;
``Base`` is the declarative base domain models register their tables against.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from config.settings import get_settings


class Base(DeclarativeBase):
    """Declarative base; domain models (T5+) register their tables here."""


def _database_url() -> str:
    url = get_settings().database_url
    if url is None:
        raise RuntimeError("DATABASE_URL is not configured")
    return url


@lru_cache
def get_engine() -> AsyncEngine:
    """Return the process-wide async engine built from settings."""
    return create_async_engine(_database_url(), pool_pre_ping=True)


@lru_cache
def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    """Return the process-wide async session factory bound to the engine."""
    return async_sessionmaker(get_engine(), expire_on_commit=False)


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding an ``AsyncSession`` scoped to the caller."""
    async with get_sessionmaker()() as session:
        yield session


async def dispose_engine() -> None:
    """Dispose the engine's connection pool if it was ever created."""
    if get_engine.cache_info().currsize:
        await get_engine().dispose()
